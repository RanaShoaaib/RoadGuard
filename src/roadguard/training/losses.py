"""Binary segmentation losses for RoadGuard (expects raw logits)."""

import torch
from torch import nn
from torch.nn import functional as F


def _validate(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B, 1, H, W]")
    if logits.shape != targets.shape:
        raise ValueError("targets must have the same shape as logits")
    if not torch.is_floating_point(logits) or not torch.is_floating_point(targets):
        raise ValueError("logits and targets must be floating-point tensors")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("logits and targets must be finite")
    if not ((targets >= 0) & (targets <= 1)).all():
        raise ValueError("targets must be in [0, 1]")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Mean per-image soft Dice loss; empty/empty masks score as perfect."""
    _validate(logits, targets)
    if smooth <= 0:
        raise ValueError("smooth must be positive")
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    score = (2 * intersection + smooth) / (denominator + smooth)
    return (1 - score).mean()


class FocalDiceLoss(nn.Module):
    """Weighted binary focal loss plus per-image soft Dice."""

    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 1.0,
                 alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0) -> None:
        super().__init__()
        _validate_weights(focal_weight, dice_weight, smooth)
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()
        dice = dice_loss(logits, targets, self.smooth)
        return self.focal_weight * focal + self.dice_weight * dice


def _validate_weights(first: float, second: float, smooth: float) -> None:
    if first < 0 or second < 0 or first + second == 0:
        raise ValueError("component weights must be nonnegative and not both zero")
    if smooth <= 0:
        raise ValueError("smooth must be positive")


class LogitDistillationLoss(nn.Module):
    """Binary segmentation supervised loss plus teacher-logit distillation.

    The teacher provides soft sigmoid targets at temperature T. The constant
    target entropy is subtracted so the KD term is zero when distributions match.
    """

    def __init__(self, supervised_loss: nn.Module, kd_weight: float = 0.5,
                 temperature: float = 2.0) -> None:
        super().__init__()
        if not isinstance(supervised_loss, nn.Module):
            raise TypeError("supervised_loss must be an nn.Module")
        if not 0.0 <= kd_weight <= 1.0:
            raise ValueError("kd_weight must be in [0, 1]")
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.supervised_loss = supervised_loss
        self.kd_weight = kd_weight
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, targets: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        _validate(student_logits, targets)
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("teacher_logits must have the same shape as student_logits")
        if not torch.is_floating_point(teacher_logits) or not torch.isfinite(teacher_logits).all():
            raise ValueError("teacher_logits must be finite floating-point tensors")

        supervised = self.supervised_loss(student_logits, targets)
        if self.kd_weight == 0:
            return supervised

        t = self.temperature
        teacher_prob = torch.sigmoid(teacher_logits.detach() / t)
        student_scaled = student_logits / t
        cross_entropy = F.binary_cross_entropy_with_logits(student_scaled, teacher_prob)
        entropy = F.binary_cross_entropy(teacher_prob, teacher_prob)
        kd = (cross_entropy - entropy) * (t * t)
        return (1 - self.kd_weight) * supervised + self.kd_weight * kd
