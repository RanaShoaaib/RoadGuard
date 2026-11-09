"""Load and validate fixed RoadGuard teacher/baseline/student experiments."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import math

import yaml


ROLES = ("teacher", "baseline", "student")

# Training-only RGB augmentation; the defaults leave it off. Horizontal flip is always on.
AUGMENT_DEFAULTS = {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "crop_min_scale": 1.0}

# The recipe settled by the notebook-04 ladder; every key can be overridden in the YAML.
TRAINING_DEFAULTS = {
    "warmup_frac": 0.05,      # linear LR warmup, then cosine decay
    "grad_clip": 1.0,
    "ema_decay": 0.998,       # 0 disables EMA weights
    "smooth_window": 3,       # epochs averaged when picking the checkpoint
    "select_min_frac": 0.5,   # ignore the high-LR first half of training when picking
    "pos_share_within": 0.7,  # share of a scenario's draws taken from frames showing the anomaly
    "focal_alpha": 0.25,      # weight of positive pixels in the focal term (negatives get 1 - alpha)
    "focal_gamma": 2.0,
    "amp": True,              # bf16 autocast, CUDA only
}


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
    sections = {"experiment", "model", "data", "teacher", "student", "training"}
    _keys(config, sections, "configuration")
    missing = sections - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    for name in sections:
        _mapping(config[name], name)

    e = config["experiment"]
    _keys(e, {"name", "seed", "lofo_family", "roles"}, "experiment")
    for field in ("name", "lofo_family"):
        if not isinstance(e.get(field), str) or not e[field].strip():
            raise ValueError(f"experiment.{field} must be a nonempty string")
    _integer(e.get("seed"), "experiment.seed", 0)
    roles = e.setdefault("roles", list(ROLES))
    if not isinstance(roles, list) or not roles or not set(roles) <= set(ROLES) or len(set(roles)) != len(roles):
        raise ValueError(f"experiment.roles must be a nonempty subset of {list(ROLES)}")
    if "student" in roles and "teacher" not in roles:
        raise ValueError("experiment.roles: a student needs a teacher in the same run")
    e["roles"] = [r for r in ROLES if r in roles]  # always train in teacher -> baseline -> student order

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
    _keys(d, {"image_size", "batch_size", "num_workers", "eval_native", "augment"}, "data")
    size = d.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("data.image_size must contain [height, width]")
    for v in size:
        _integer(v, "data.image_size dimension", 16)
    _integer(d.get("batch_size"), "data.batch_size")
    _integer(d.get("num_workers", 0), "data.num_workers", 0)
    d.setdefault("num_workers", 0)
    if not isinstance(d.setdefault("eval_native", True), bool):
        raise ValueError("data.eval_native must be boolean")
    augment = d.setdefault("augment", {})
    if not isinstance(augment, dict):
        raise ValueError("data.augment must be a mapping")
    _keys(augment, set(AUGMENT_DEFAULTS), "data.augment")
    for key, default in AUGMENT_DEFAULTS.items():
        augment.setdefault(key, default)
    for key in ("brightness", "contrast", "saturation"):
        if not 0 <= _number(augment[key], f"data.augment.{key}") < 1:
            raise ValueError(f"data.augment.{key} must be in [0, 1)")
    if not 0 < _number(augment["crop_min_scale"], "data.augment.crop_min_scale") <= 1:
        raise ValueError("data.augment.crop_min_scale must be in (0, 1]")

    t = config["teacher"]
    _keys(t, {"depth_noise_std_m", "depth_dropout_probability", "depth_dropout_fraction"}, "teacher")
    _number(t.get("depth_noise_std_m", 0), "teacher.depth_noise_std_m")
    for field in ("depth_dropout_probability", "depth_dropout_fraction"):
        value = _number(t.get(field, 0), f"teacher.{field}")
        if value > 1:
            raise ValueError(f"teacher.{field} must be <= 1")
    if t.get("depth_dropout_fraction", 0.2) == 0:
        raise ValueError("teacher.depth_dropout_fraction must be > 0")
    for field in ("depth_noise_std_m", "depth_dropout_probability", "depth_dropout_fraction"):
        t.setdefault(field, 0.2 if field == "depth_dropout_fraction" else 0.0)

    s = config["student"]
    _keys(s, {"kd_weight", "temperature"}, "student")
    kd = _number(s.get("kd_weight"), "student.kd_weight")
    if kd > 1:
        raise ValueError("student.kd_weight must be <= 1")
    _number(s.get("temperature"), "student.temperature", strictly=True)

    tr = config["training"]
    _keys(tr, set(TRAINING_DEFAULTS) | {"epochs", "optimizer", "learning_rate", "weight_decay"}, "training")
    _integer(tr.get("epochs"), "training.epochs")
    if tr.get("optimizer") != "adamw":
        raise ValueError("training.optimizer must be adamw")
    _number(tr.get("learning_rate"), "training.learning_rate", strictly=True)
    _number(tr.get("weight_decay"), "training.weight_decay")
    for key, default in TRAINING_DEFAULTS.items():
        tr.setdefault(key, default)
    for key in ("warmup_frac", "ema_decay", "select_min_frac"):
        if not 0 <= _number(tr[key], f"training.{key}") < 1:
            raise ValueError(f"training.{key} must be in [0, 1)")
    if not 0 < tr["pos_share_within"] < 1:
        raise ValueError("training.pos_share_within must be in (0, 1)")
    if not 0 <= tr["focal_alpha"] <= 1:
        raise ValueError("training.focal_alpha must be in [0, 1]")
    _number(tr["focal_gamma"], "training.focal_gamma")
    _number(tr["grad_clip"], "training.grad_clip", strictly=True)
    _integer(tr["smooth_window"], "training.smooth_window")
    if not isinstance(tr["amp"], bool):
        raise ValueError("training.amp must be boolean")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate its contents."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return validate_config(raw)
