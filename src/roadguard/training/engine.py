"""Training and evaluation passes for RoadGuard binary segmentation.

One training function serves all three roles: the RGB+depth teacher (``in_channels=4``), the RGB
baseline, and the RGB student (pass a frozen ``teacher`` and a distillation criterion).
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from roadguard.evaluation.metrics import EvalAccumulator


def _prepare_batch(batch: dict, device: torch.device | str, in_channels: int):
    """Move inputs to device and optionally concatenate the depth channel."""
    if in_channels not in (3, 4):
        raise ValueError("in_channels must be 3 (RGB) or 4 (RGB + depth)")
    image = batch["image"].to(device=device, dtype=torch.float32)
    mask = batch["mask"].to(device=device, dtype=torch.float32)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    if mask.shape != (image.shape[0], 1, *image.shape[-2:]):
        raise ValueError("mask must have shape [B, 1, H, W]")
    if in_channels == 4:
        depth = batch["depth"].to(device=device, dtype=torch.float32)
        if depth.shape != mask.shape:
            raise ValueError("decoded depth must have shape [B, 1, H, W]")
        image = torch.cat((image, depth), dim=1)
    return image, mask


def _autocast(device: torch.device | str, amp: bool):
    device_type = torch.device(device).type
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp and device_type == "cuda")


class EMA:
    """Exponential moving average of a model's weights; ``module`` is what gets evaluated and saved.

    Parameters are averaged; buffers (BatchNorm statistics) are copied from the live model. The
    decay ramps up from zero so the first steps are not dominated by the random initialisation.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.steps = 0
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - decay)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float):
    """Linear warmup then cosine decay to zero; call ``step()`` once per optimizer step."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup = max(1, int(warmup_frac * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device | str,
                    in_channels: int = 3, *, teacher: nn.Module | None = None,
                    scheduler=None, ema: EMA | None = None,
                    grad_clip: float | None = None, amp: bool = False) -> float:
    """Train for one epoch; return the sample-weighted mean loss.

    With a ``teacher`` the model is a student: the frozen RGB+depth teacher supplies logits and
    ``criterion`` must accept (student_logits, targets, teacher_logits).
    """
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss, total_samples = 0.0, 0
    for batch in loader:
        inputs, targets = _prepare_batch(batch, device, in_channels)
        teacher_logits = None
        if teacher is not None:
            teacher_inputs, _ = _prepare_batch(batch, device, 4)
            with torch.no_grad(), _autocast(device, amp):
                teacher_logits = teacher(teacher_inputs).float()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(inputs)
        logits = logits.float()
        loss = criterion(logits, targets) if teacher is None else criterion(logits, targets, teacher_logits)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError("criterion must return a finite scalar loss")
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.detach().item() * inputs.shape[0]
        total_samples += inputs.shape[0]
    if total_samples == 0:
        raise ValueError("DataLoader yielded no samples")
    return total_loss / total_samples


@torch.no_grad()
def collect(model: nn.Module, loader: DataLoader, device: torch.device | str,
            in_channels: int = 3, amp: bool = False) -> EvalAccumulator:
    """Run ``model`` over ``loader`` and accumulate logit-space evaluation curves."""
    model.eval()
    accumulator = EvalAccumulator()
    seen = 0
    for batch in loader:
        inputs, targets = _prepare_batch(batch, device, in_channels)
        with _autocast(device, amp):
            logits = model(inputs)
        accumulator.update(logits.float(), targets, batch["scenario_id"])
        seen += inputs.shape[0]
    if seen == 0:
        raise ValueError("DataLoader yielded no samples")
    return accumulator
