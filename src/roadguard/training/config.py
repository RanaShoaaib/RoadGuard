"""Load and validate fixed RoadGuard teacher/baseline/student experiments."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import math

import yaml


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _keys(value: dict, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")


def _number(value: Any, label: str, *, minimum: float = 0, strictly: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value <= minimum if strictly else value < minimum:
        raise ValueError(f"{label} must be {'>' if strictly else '>='} {minimum}")
    return float(value)


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a validated, independent configuration dictionary."""
    config = deepcopy(_mapping(raw, "configuration"))
    sections = {"experiment", "model", "data", "teacher", "student", "training", "evaluation"}
    _keys(config, sections, "configuration")
    missing = sections - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    for name in sections:
        _mapping(config[name], name)

    e = config["experiment"]
    _keys(e, {"name", "seed", "lofo_family"}, "experiment")
    for field in ("name", "lofo_family"):
        if not isinstance(e.get(field), str) or not e[field].strip():
            raise ValueError(f"experiment.{field} must be a nonempty string")
    _integer(e.get("seed"), "experiment.seed", 0)

    m = config["model"]
    _keys(m, {"architecture", "base_channels", "pretrained"}, "model")
    if m.get("architecture") not in {"unet", "segformer"}:
        raise ValueError("model.architecture must be unet or segformer")
    _integer(m.get("base_channels", 32), "model.base_channels")
    m.setdefault("base_channels", 32)
    if not isinstance(m.get("pretrained", False), bool):
        raise ValueError("model.pretrained must be boolean")
    m.setdefault("pretrained", False)

    d = config["data"]
    _keys(d, {"image_size", "batch_size", "num_workers"}, "data")
    size = d.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("data.image_size must contain [height, width]")
    for v in size:
        _integer(v, "data.image_size dimension", 16)
    _integer(d.get("batch_size"), "data.batch_size")
    _integer(d.get("num_workers", 0), "data.num_workers", 0)
    d.setdefault("num_workers", 0)

    t = config["teacher"]
    _keys(t, {"depth_noise_std_m", "depth_dropout_probability", "depth_dropout_fraction"}, "teacher")
    _number(t.get("depth_noise_std_m", 0), "teacher.depth_noise_std_m")
    for field in ("depth_dropout_probability", "depth_dropout_fraction"):
        value = _number(t.get(field, 0), f"teacher.{field}")
        if value > 1:
            raise ValueError(f"teacher.{field} must be <= 1")
    for field in ("depth_noise_std_m", "depth_dropout_probability", "depth_dropout_fraction"):
        t.setdefault(field, 0.0)

    s = config["student"]
    _keys(s, {"kd_weight", "temperature"}, "student")
    kd = _number(s.get("kd_weight"), "student.kd_weight")
    if kd > 1:
        raise ValueError("student.kd_weight must be <= 1")
    _number(s.get("temperature"), "student.temperature", strictly=True)

    tr = config["training"]
    _keys(tr, {"epochs", "optimizer", "learning_rate", "weight_decay", "supervised_loss"}, "training")
    _integer(tr.get("epochs"), "training.epochs")
    if tr.get("optimizer") != "adamw":
        raise ValueError("training.optimizer must be adamw")
    _number(tr.get("learning_rate"), "training.learning_rate", strictly=True)
    _number(tr.get("weight_decay"), "training.weight_decay")
    if tr.get("supervised_loss") not in {"bce_dice", "focal_dice"}:
        raise ValueError("training.supervised_loss must be bce_dice or focal_dice")

    ev = config["evaluation"]
    _keys(ev, {"threshold"}, "evaluation")
    threshold = _number(ev.get("threshold"), "evaluation.threshold")
    if threshold > 1:
        raise ValueError("evaluation.threshold must be <= 1")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate its contents."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return validate_config(raw)
