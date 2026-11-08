"""Checkpoint saving/loading and epoch selection for RoadGuard experiments."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, *, epoch: int, score: float,
                    config: dict[str, Any]) -> None:
    """Atomically save model weights with their epoch (zero-based) and validation score."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": int(epoch), "score": float(score), "config": dict(config),
               "model_state_dict": model.state_dict()}
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, model: nn.Module, *,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Restore model weights and return the checkpoint metadata (epoch, score, config).

    Only load checkpoints created by trusted code.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    missing = {"epoch", "score", "config", "model_state_dict"} - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return {key: checkpoint[key] for key in ("epoch", "score", "config")}


def select_epoch(scores: list[float], window: int = 3, min_frac: float = 0.5,
                 total_epochs: int | None = None) -> tuple[int | None, float]:
    """Best epoch by trailing-window mean of ``scores``, among epochs >= ``min_frac`` of training.

    Validation is small and noisy, so a raw "best epoch" is a lottery; averaging the last
    ``window`` epochs and ignoring the high-learning-rate phase makes the pick stable. Returns
    (zero-based epoch or None, its smoothed score). NaN scores are skipped; ties keep the earlier epoch.
    """
    n = total_epochs or len(scores)
    first = max(0, math.ceil(min_frac * n) - 1)
    best_epoch, best_score = None, -float("inf")
    for i in range(first, len(scores)):
        recent = [x for x in scores[max(0, i - window + 1): i + 1] if x == x]
        if not recent:
            continue
        mean = sum(recent) / len(recent)
        if mean > best_score:
            best_epoch, best_score = i, mean
    return best_epoch, best_score
