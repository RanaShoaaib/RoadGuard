"""Checkpoint utilities for RoadGuard training experiments."""
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
    best_val_loss: float | None = None,
) -> None:
    """Atomically save a resumable training checkpoint.

    Epoch is zero-based. Store best_val_loss separately from current val_loss
    so a latest checkpoint can resume best-model selection correctly.
    """
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "best_val_loss": float(val_loss if best_val_loss is None else best_val_loss),
        "config": dict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model (and optional optimizer) and return checkpoint metadata.

    Only load checkpoints created by trusted code; optimizer state and
    configuration are not suitable for loading from untrusted sources.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    required = {"epoch", "val_loss", "best_val_loss", "config", "model_state_dict", "optimizer_state_dict"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {key: checkpoint[key] for key in ("epoch", "val_loss", "best_val_loss", "config")}


def save_epoch_checkpoints(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    config: dict[str, Any],
) -> float:
    """Always save latest; update best only when validation loss improves."""
    directory = Path(directory)
    improved = val_loss < best_val_loss
    new_best = float(val_loss if improved else best_val_loss)
    if improved:
        save_checkpoint(directory / "best.pt", model, optimizer, epoch=epoch,
                        val_loss=val_loss, best_val_loss=new_best, config=config)
    save_checkpoint(directory / "latest.pt", model, optimizer, epoch=epoch,
                    val_loss=val_loss, best_val_loss=new_best, config=config)
    return new_best
