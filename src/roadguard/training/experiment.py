"""Run one RoadGuard LOFO fold: teacher, RGB baseline, distilled RGB student."""
from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from roadguard.data.dataset import RoadGuardDataset
from roadguard.data.transforms import build_transforms
from roadguard.evaluation.metrics import summarize
from roadguard.models.factory import build_model
from roadguard.training.checkpoints import load_checkpoint, save_checkpoint, select_epoch
from roadguard.training.config import load_config, validate_config
from roadguard.training.engine import EMA, build_scheduler, collect, train_one_epoch
from roadguard.training.losses import FocalDiceLoss, LogitDistillationLoss
from roadguard.training.sampling import make_scenario_balanced_sampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lofo_fold(path: str | Path, family: str) -> dict[str, pd.DataFrame]:
    """Load and validate a Notebook 1 LOFO fold CSV (test = unseen family, val = held-out objects)."""
    fold = pd.read_csv(path, dtype={"scenario_id": str, "frame_id": str})
    required = set(RoadGuardDataset.REQUIRED_COLUMNS) | {"anomaly_family", "anomaly_object", "anomaly_pixels", "split"}
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


def make_loaders(parts: dict[str, pd.DataFrame], data_root: str | Path, config: dict,
                 *, teacher_train: bool = False) -> dict[str, DataLoader]:
    """Train loader draws scenario-balanced; val/test are deterministic.

    Teacher training gets depth perturbations; other roles use clean depth. Every call returns
    loaders in the same initial random state, so call it once per role: baseline and student
    then see the identical sequence of training frames and augmentations.
    """
    size = tuple(config["data"]["image_size"])
    depth_args = config["teacher"] if teacher_train else {}
    train_tf = build_transforms(image_size=size, train=True, **depth_args)
    eval_tf = build_transforms(image_size=size, train=False)
    workers = config["data"]["num_workers"]
    common = dict(batch_size=config["data"]["batch_size"], num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = RoadGuardDataset(parts[split], data_root, transform=train_tf if split == "train" else eval_tf)
        if split == "train":
            sampler = make_scenario_balanced_sampler(
                parts["train"], config["training"]["pos_share_within"], seed=config["experiment"]["seed"])
            # A private generator seeds the worker processes, so the augmentation stream does not
            # depend on how much global randomness earlier roles consumed.
            generator = torch.Generator().manual_seed(config["experiment"]["seed"])
            loaders[split] = DataLoader(dataset, sampler=sampler, drop_last=True,
                                        generator=generator, **common)
        else:
            loaders[split] = DataLoader(dataset, shuffle=False, **common)
    if len(loaders["train"]) == 0:
        raise ValueError(f"{len(parts['train'])} training frames < batch_size (drop_last)")
    return loaders


def _build_model(config: dict, channels: int) -> torch.nn.Module:
    m = config["model"]
    return build_model(m["architecture"], in_channels=channels, num_classes=1,
                       base_channels=m["base_channels"], pretrained=m["pretrained"])


def _supervised_loss(config: dict) -> FocalDiceLoss:
    t = config["training"]
    return FocalDiceLoss(alpha=t["focal_alpha"], gamma=t["focal_gamma"])


def _train_role(role: str, model: torch.nn.Module, loaders: dict, config: dict, run_dir: Path,
                device: torch.device, teacher: torch.nn.Module | None = None) -> list[dict]:
    """Train one role; write ``best.pt`` (smoothed-validation pick), ``last.pt`` and ``history.json``.

    Validation macro AP (per-scenario AP, averaged) is the selection score for every role, so a
    student is chosen by the same rule as the baseline it is compared with.
    """
    t = config["training"]
    channels = 4 if role == "teacher" else 3
    role_dir = run_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    supervised = _supervised_loss(config)
    criterion = (supervised if teacher is None else LogitDistillationLoss(
        supervised, kd_weight=config["student"]["kd_weight"], temperature=config["student"]["temperature"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = build_scheduler(optimizer, t["epochs"] * len(loaders["train"]), t["warmup_frac"])
    ema = EMA(model, t["ema_decay"]) if t["ema_decay"] > 0 else None
    amp = t["amp"]
    scores, history, picked = [], [], False
    for epoch in range(t["epochs"]):
        started = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, channels,
                                     teacher=teacher, scheduler=scheduler, ema=ema,
                                     grad_clip=t["grad_clip"], amp=amp)
        evaluated = ema.module if ema is not None else model
        val = summarize(collect(evaluated, loaders["val"], device, channels, amp))
        scores.append(val["macro_scenario_ap"])
        chosen, smoothed = select_epoch(scores, t["smooth_window"], t["select_min_frac"], t["epochs"])
        if chosen == epoch:
            save_checkpoint(role_dir / "best.pt", evaluated, epoch=epoch, score=smoothed, config=config)
            picked = True
        save_checkpoint(role_dir / "last.pt", evaluated, epoch=epoch, score=scores[-1], config=config)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_macro_ap": val["macro_scenario_ap"],
                        "val_pixel_ap": val["pixel_ap"], "val_frame_auroc": val["frame_auroc"],
                        "minutes": (time.time() - started) / 60})
        (role_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"{role} {epoch + 1:>2}/{t['epochs']} loss={train_loss:.4f} "
              f"val_macroAP={val['macro_scenario_ap']:.4f} val_pixAP={val['pixel_ap']:.4f} "
              f"{'*' if chosen == epoch else ''}", flush=True)
    if not picked:  # no eligible finite score: fall back to the final epoch
        (role_dir / "best.pt").write_bytes((role_dir / "last.pt").read_bytes())
    return history


def evaluate_checkpoint(model: torch.nn.Module, checkpoint: str | Path, loaders: dict,
                        device: torch.device, channels: int, amp: bool = True) -> dict:
    """Load ``checkpoint``; pick the threshold on VALIDATION (best Dice) and apply it once to TEST.

    Returns {"epoch", "val", "test", "test_scenarios"} where ``test_scenarios`` is a DataFrame.
    """
    meta = load_checkpoint(checkpoint, model, map_location=device)
    val = collect(model, loaders["val"], device, channels, amp)
    threshold, _ = val.pooled.best_threshold("dice")
    if threshold != threshold:  # NaN: validation has no positives
        threshold = 0.0
    test = collect(model, loaders["test"], device, channels, amp)
    return {"epoch": meta["epoch"] + 1, "val": summarize(val, threshold), "test": summarize(test, threshold),
            "test_scenarios": test.scenario_table(threshold), "val_scenarios": val.scenario_table(threshold)}


def run_experiment(config: str | Path | dict, fold_path: str | Path, data_root: str | Path,
                   output_root: str | Path, device: str | None = None) -> dict:
    """Train the configured roles on one LOFO fold and report each role on the unseen family.

    Layout: <output_root>/<name>/<family>/seed_<seed>/<role>/{best.pt,last.pt,history.json,
    test_scenarios*.csv,val_scenarios*.csv} plus config.json and report.json. ``config`` is a YAML
    path or an already-built dictionary (it is validated either way). Roles run in the order
    teacher -> baseline -> student; the baseline and student start from identical RGB weights and are
    trained on identical sample sequences, so their difference reflects distillation alone.
    """
    config = validate_config(config) if isinstance(config, dict) else load_config(config)
    seed = config["experiment"]["seed"]
    family = config["experiment"]["lofo_family"]
    roles = config["experiment"]["roles"]
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
    amp = config["training"]["amp"]
    seed_everything(seed)
    rgb_initial = deepcopy(_build_model(config, 3).state_dict())

    models: dict[str, torch.nn.Module] = {}
    report: dict = {"config": config}
    for role in roles:
        channels = 4 if role == "teacher" else 3
        seed_everything(seed)
        model = _build_model(config, channels).to(device)
        if channels == 3:
            model.load_state_dict(rgb_initial)
        teacher = None
        if role == "student":
            teacher = models["teacher"]
            load_checkpoint(run_dir / "teacher" / "best.pt", teacher, map_location=device)
            teacher.requires_grad_(False)
        role_loaders = make_loaders(parts, data_root, config, teacher_train=role == "teacher")
        _train_role(role, model, role_loaders, config, run_dir, device, teacher)
        models[role] = model
        report[role] = {}
        for which, name in (("selected", "best.pt"), ("last", "last.pt")):
            result = evaluate_checkpoint(model, run_dir / role / name, role_loaders, device, channels, amp)
            suffix = "" if which == "selected" else "_last"
            result.pop("test_scenarios").to_csv(run_dir / role / f"test_scenarios{suffix}.csv", index=False)
            result.pop("val_scenarios").to_csv(run_dir / role / f"val_scenarios{suffix}.csv", index=False)
            report[role][which] = result
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report
