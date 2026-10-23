"""Single-epoch training and validation for RoadGuard binary segmentation."""

import torch
from torch import nn
from torch.utils.data import DataLoader


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


def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               device: torch.device | str, in_channels: int,
               optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs, targets = _prepare_batch(batch, device, in_channels)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise ValueError("criterion must return a finite scalar loss")
            if training:
                loss.backward()
                optimizer.step()
            batch_size = inputs.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader yielded no samples")
    return total_loss / total_samples


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device | str,
                    in_channels: int = 3) -> float:
    """Train for one epoch; return sample-weighted mean batch loss."""
    return _run_epoch(model, loader, criterion, device, in_channels, optimizer)


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device | str, in_channels: int = 3) -> float:
    """Validate without gradient computation or optimizer updates."""
    return _run_epoch(model, loader, criterion, device, in_channels, optimizer=None)


def _run_distillation_epoch(student: nn.Module, teacher: nn.Module,
                            loader: DataLoader, criterion: nn.Module,
                            device: torch.device | str,
                            optimizer: torch.optim.Optimizer | None) -> float:
    """Run one RGB-student / RGB+depth-teacher epoch with teacher frozen.

    The teacher stays in eval mode and is evaluated without autograd. The
    criterion must accept (student_logits, targets, teacher_logits).
    """
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        student_inputs, targets = _prepare_batch(batch, device, in_channels=3)
        teacher_inputs, _ = _prepare_batch(batch, device, in_channels=4)
        with torch.no_grad():
            teacher_logits = teacher(teacher_inputs)
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            student_logits = student(student_inputs)
            loss = criterion(student_logits, targets, teacher_logits)
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise ValueError("criterion must return a finite scalar loss")
            if training:
                loss.backward()
                optimizer.step()
        batch_size = student_inputs.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader yielded no samples")
    return total_loss / total_samples


def train_distillation_one_epoch(student: nn.Module, teacher: nn.Module,
                                 loader: DataLoader, criterion: nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 device: torch.device | str) -> float:
    """Update RGB student using a frozen RGB+depth teacher; return mean loss."""
    return _run_distillation_epoch(student, teacher, loader, criterion, device, optimizer)


def validate_distillation_one_epoch(student: nn.Module, teacher: nn.Module,
                                    loader: DataLoader, criterion: nn.Module,
                                    device: torch.device | str) -> float:
    """Compute student distillation validation loss without parameter updates."""
    return _run_distillation_epoch(student, teacher, loader, criterion, device, None)
