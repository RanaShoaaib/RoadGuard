"""Tests for RoadGuard binary segmentation losses."""

import pytest
import torch
from torch.nn import functional as F

from roadguard.training.losses import FocalDiceLoss, dice_loss


def test_forward_backward():
    logits = torch.randn(2, 1, 16, 20, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    loss = FocalDiceLoss()(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_focal_gamma_zero_reduces_to_alpha_weighted_bce():
    logits = torch.tensor([[[[0.0, 1.0]]]])
    targets = torch.tensor([[[[1.0, 0.0]]]])
    expected = (torch.tensor([0.25, 0.75]) * F.binary_cross_entropy_with_logits(
        logits.flatten(), targets.flatten(), reduction="none")).mean()
    result = FocalDiceLoss(gamma=0, dice_weight=0)(logits, targets)
    assert torch.allclose(result, expected)


def test_dice_per_image_average():
    logits = torch.zeros(2, 1, 1, 2)
    targets = torch.tensor([[[[1., 1.]]], [[[0., 0.]]]])
    # At p=0.5, Dice scores are 3/4 and 1/2 with smooth=1.
    assert torch.allclose(dice_loss(logits, targets), torch.tensor(0.375))


def test_empty_mask_finite_and_differentiable():
    logits = torch.full((2, 1, 8, 8), -5.0, requires_grad=True)
    targets = torch.zeros_like(logits)
    value = FocalDiceLoss()(logits, targets)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_reject_invalid_targets():
    logits = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="same shape"):
        FocalDiceLoss()(logits, torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FocalDiceLoss()(logits, torch.full_like(logits, 255.0))


def test_invalid_parameters():
    with pytest.raises(ValueError, match="weights"):
        FocalDiceLoss(focal_weight=0, dice_weight=0)
    with pytest.raises(ValueError, match="alpha"):
        FocalDiceLoss(alpha=1.5)
    with pytest.raises(ValueError, match="gamma"):
        FocalDiceLoss(gamma=-1)


from roadguard.training.losses import LogitDistillationLoss


def test_distillation_forward_backward_and_frozen_teacher():
    supervised = FocalDiceLoss()
    student = torch.randn(2, 1, 5, 7, requires_grad=True)
    teacher = torch.randn(2, 1, 5, 7, requires_grad=True)
    targets = torch.randint(0, 2, student.shape).float()
    loss = LogitDistillationLoss(supervised, kd_weight=0.5, temperature=2.0)(student, targets, teacher)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_distillation_zero_weight_matches_supervised():
    student = torch.randn(1, 1, 3, 4)
    teacher = torch.randn_like(student)
    targets = torch.zeros_like(student)
    supervised = FocalDiceLoss()
    actual = LogitDistillationLoss(supervised, kd_weight=0)(student, targets, teacher)
    assert torch.allclose(actual, supervised(student, targets))


def test_distillation_identical_logits_zero_kd_term():
    logits = torch.tensor([[[[-2.0, 0.0, 2.0]]]])
    targets = torch.ones_like(logits)
    criterion = LogitDistillationLoss(FocalDiceLoss(), kd_weight=1.0, temperature=3.0)
    assert criterion(logits, targets, logits).abs() < 1e-6


def test_distillation_kd_matches_bernoulli_kl():
    student = torch.tensor([[[[0.0, 1.0]]]])
    teacher = torch.tensor([[[[2.0, -1.0]]]])
    targets = torch.zeros_like(student)
    t = 2.0
    p = torch.sigmoid(teacher / t)
    q = torch.sigmoid(student / t)
    expected = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean() * t * t
    actual = LogitDistillationLoss(FocalDiceLoss(), kd_weight=1.0, temperature=t)(student, targets, teacher)
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("kwargs,match", [
    ({"kd_weight": -0.1}, "kd_weight"),
    ({"kd_weight": 1.1}, "kd_weight"),
    ({"temperature": 0}, "temperature"),
    ({"temperature": float("inf")}, "temperature"),
])
def test_distillation_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LogitDistillationLoss(FocalDiceLoss(), **kwargs)


def test_distillation_rejects_bad_teacher_logits():
    student = torch.zeros(1, 1, 2, 3)
    targets = torch.zeros_like(student)
    criterion = LogitDistillationLoss(FocalDiceLoss())
    with pytest.raises(ValueError, match="same shape"):
        criterion(student, targets, torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match="finite"):
        criterion(student, targets, torch.full_like(student, float("nan")))
