"""Run one RoadGuard LOFO fold: teacher, RGB baseline, distilled RGB student."""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from roadguard.data.dataset import RoadGuardDataset
from roadguard.data.transforms import build_transforms
from roadguard.evaluation.metrics import SegmentationMetrics
from roadguard.models.factory import build_model
from roadguard.training.checkpoints import load_checkpoint, save_epoch_checkpoints
from roadguard.training.config import load_config
from roadguard.training.engine import (
    _prepare_batch, train_distillation_one_epoch, train_one_epoch, validate_one_epoch,
)
from roadguard.training.losses import LogitDistillationLoss, build_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lofo_fold(path: str | Path, family: str) -> dict[str, pd.DataFrame]:
    """Load and validate a Notebook 1 LOFO fold CSV (test = unseen family, val = held-out objects)."""
    fold = pd.read_csv(path, dtype={"scenario_id": str, "frame_id": str})
    required = set(RoadGuardDataset.REQUIRED_COLUMNS) | {"anomaly_family", "anomaly_object", "split"}
    if not required.issubset(fold.columns):
        raise ValueError(f"LOFO fold missing columns: {sorted(required - set(fold.columns))}")
    if fold.empty or fold[list(required)].isna().any().any():
        raise ValueError("LOFO fold contains no rows or missing required values")
    if not fold["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unknown split label")
    if fold.duplicated(["scenario_id", "frame_id"]).any():
        raise ValueError("Duplicate frame in fold")
    if (fold.groupby("scenario_id")["split"].nunique() > 1).any():
        raise ValueError("Scenario leakage across partitions")
    if not (fold.loc[fold["anomaly_family"] == family, "split"] == "test").all():
        raise ValueError("Held-out family appears outside test")
    if not (fold.loc[fold["anomaly_family"] != family, "split"] != "test").all():
        raise ValueError("Other anomaly families appear in test")
    if not (fold["anomaly_family"] == family).any():
        raise ValueError(f"Held-out family {family!r} absent")
    objects = fold.groupby("split")["anomaly_object"].agg(set)
    if "train" in objects and "val" in objects and objects["train"] & objects["val"]:
        raise ValueError("Anomaly object leakage between train and val")
    parts = {s: fold.loc[fold["split"] == s].copy() for s in ("train", "val", "test")}
    if any(part.empty for part in parts.values()):
        raise ValueError("Every partition must contain samples")
    return parts


def make_loaders(parts: dict[str, pd.DataFrame], data_root: str | Path,
                 config: dict, *, teacher_train: bool = False) -> dict[str, DataLoader]:
    """Teacher training gets depth perturbations; other roles use clean depth."""
    size = tuple(config["data"]["image_size"])
    depth_args = config["teacher"] if teacher_train else {}
    train_tf = build_transforms(image_size=size, train=True, **depth_args)
    eval_tf = build_transforms(image_size=size, train=False)
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = RoadGuardDataset(parts[split], data_root, transform=train_tf if split == "train" else eval_tf)
        loaders[split] = DataLoader(
            dataset, batch_size=config["data"]["batch_size"], shuffle=split == "train",
            num_workers=config["data"]["num_workers"], pin_memory=torch.cuda.is_available(),
        )
    return loaders


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             in_channels: int, threshold: float) -> dict:
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    with torch.no_grad():
        for batch in loader:
            inputs, masks = _prepare_batch(batch, device, in_channels)
            metrics.update(model(inputs), masks)
    return metrics.compute()


def _build_model(config: dict, channels: int) -> torch.nn.Module:
    m = config["model"]
    return build_model(m["architecture"], in_channels=channels, num_classes=1,
                       base_channels=m["base_channels"], pretrained=m["pretrained"])


def _train_role(role: str, model: torch.nn.Module, loaders: dict, config: dict,
                output: Path, device: torch.device, supervised: torch.nn.Module,
                teacher: torch.nn.Module | None = None) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    kd_loss = None
    if teacher is not None:
        kd_loss = LogitDistillationLoss(supervised,
            kd_weight=config["student"]["kd_weight"],
            temperature=config["student"]["temperature"])
    best = float("inf")
    history = []
    channels = 4 if role == "teacher" else 3
    for epoch in range(config["training"]["epochs"]):
        if teacher is None:
            train_loss = train_one_epoch(model, loaders["train"], supervised, optimizer, device, channels)
        else:
            train_loss = train_distillation_one_epoch(model, teacher, loaders["train"], kd_loss, optimizer, device)
        # Identical supervised selection objective for all roles, including KD student.
        val_loss = validate_one_epoch(model, loaders["val"], supervised, device, channels)
        best = save_epoch_checkpoints(output / role, model, optimizer, epoch=epoch,
                                      val_loss=val_loss, best_val_loss=best, config=config)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_supervised_loss": val_loss})
        print(f"{role} epoch {epoch + 1}/{config['training']['epochs']}: "
              f"train={train_loss:.5f} val_supervised={val_loss:.5f}", flush=True)
    (output / role / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    load_checkpoint(output / role / "best.pt", model, map_location=device)
    model.eval()
    return history


def run_experiment(config_path: str | Path, fold_path: str | Path,
                   data_root: str | Path, output_root: str | Path,
                   device: str | None = None) -> dict:
    """Train all roles and evaluate best checkpoints once on the held-out test set.

    The fold_path is processed/lofo_splits/<family>.csv from Notebook 1.
    """
    config = load_config(config_path)
    if not 0.0 < config["evaluation"]["threshold"] < 1.0:
        raise ValueError("evaluation.threshold must be strictly between 0 and 1")
    seed = config["experiment"]["seed"]
    family = config["experiment"]["lofo_family"]
    parts = load_lofo_fold(fold_path, family)
    run_dir = Path(output_root) / config["experiment"]["name"] / family / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Experiment output already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "partition_counts.json").write_text(
        json.dumps({k: {"frames": len(v), "scenarios": int(v["scenario_id"].nunique())}
                    for k, v in parts.items()}, indent=2), encoding="utf-8")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(seed)
    teacher = _build_model(config, 4).to(device)
    seed_everything(seed)
    rgb_initial = deepcopy(_build_model(config, 3).state_dict())
    supervised = build_loss(config["training"]["supervised_loss"])
    teacher_loaders = make_loaders(parts, data_root, config, teacher_train=True)
    rgb_loaders = make_loaders(parts, data_root, config, teacher_train=False)
    _train_role("teacher", teacher, teacher_loaders, config, run_dir, device, supervised)
    teacher.requires_grad_(False)
    baseline = _build_model(config, 3).to(device)
    baseline.load_state_dict(rgb_initial)
    seed_everything(seed)
    _train_role("baseline", baseline, rgb_loaders, config, run_dir, device, supervised)
    student = _build_model(config, 3).to(device)
    student.load_state_dict(rgb_initial)
    seed_everything(seed)
    _train_role("student", student, rgb_loaders, config, run_dir, device, supervised, teacher=teacher)
    results = {}
    for role, model, channels in (("teacher", teacher, 4), ("baseline", baseline, 3), ("student", student, 3)):
        results[role] = evaluate(model, rgb_loaders["test"], device, channels,
                                 config["evaluation"]["threshold"])
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
