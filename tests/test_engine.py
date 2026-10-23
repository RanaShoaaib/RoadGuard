"""Tests for RoadGuard epoch engine."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from roadguard.training.engine import train_one_epoch, validate_one_epoch


class Samples(Dataset):
    def __init__(self, count=5, depth_channels=1):
        self.count = count
        self.depth_channels = depth_channels

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "image": torch.full((3, 8, 8), float(index) / 10),
            "depth": torch.ones(self.depth_channels, 8, 8),
            "mask": torch.full((1, 8, 8), float(index % 2)),
        }


@pytest.mark.parametrize("channels", [3, 4])
def test_train_updates_weights(channels):
    model = nn.Conv2d(channels, 1, 1)
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = train_one_epoch(model, DataLoader(Samples(), batch_size=2),
                           nn.BCEWithLogitsLoss(), optimizer, "cpu", channels)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(model.weight, original)
    assert model.training


@pytest.mark.parametrize("channels", [3, 4])
def test_validation_preserves_parameters_and_mode(channels):
    model = nn.Conv2d(channels, 1, 1)
    original = model.weight.detach().clone()
    loss = validate_one_epoch(model, DataLoader(Samples(), batch_size=2),
                              nn.BCEWithLogitsLoss(), "cpu", channels)
    assert loss >= 0
    assert torch.equal(model.weight, original)
    assert model.training is False
    assert model.weight.grad is None


def test_weighted_average_by_number_of_samples():
    class ConstantModel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, *x.shape[-2:], device=x.device)

    class MeanTargetLoss(nn.Module):
        def forward(self, logits, targets):
            return targets.mean()

    # Batch means: (0+1)/2, (0+1)/2, 0; sample-weighted mean = 2/5.
    result = validate_one_epoch(ConstantModel(), DataLoader(Samples(), batch_size=2),
                                MeanTargetLoss(), "cpu")
    assert result == pytest.approx(0.4)


def test_rgb_only_does_not_require_depth():
    class RGBOnly(Dataset):
        def __len__(self):
            return 1
        def __getitem__(self, i):
            return {"image": torch.zeros(3, 8, 8), "mask": torch.zeros(1, 8, 8)}
    model = nn.Conv2d(3, 1, 1)
    assert validate_one_epoch(model, DataLoader(RGBOnly()), nn.BCEWithLogitsLoss(), "cpu") >= 0


def test_four_channels_requires_decoded_depth():
    model = nn.Conv2d(4, 1, 1)
    with pytest.raises(ValueError, match="decoded depth"):
        validate_one_epoch(model, DataLoader(Samples(depth_channels=3)),
                           nn.BCEWithLogitsLoss(), "cpu", 4)


def test_empty_loader_rejected():
    model = nn.Conv2d(3, 1, 1)
    with pytest.raises(ValueError, match="no samples"):
        validate_one_epoch(model, DataLoader(Samples(count=0)), nn.BCEWithLogitsLoss(), "cpu")


def test_invalid_channel_count():
    model = nn.Conv2d(3, 1, 1)
    with pytest.raises(ValueError, match="in_channels"):
        validate_one_epoch(model, DataLoader(Samples(count=1)), nn.BCEWithLogitsLoss(), "cpu", 5)


def test_nonfinite_loss_rejected():
    class BadLoss(nn.Module):
        def forward(self, logits, targets):
            return logits.mean() * float("nan")
    with pytest.raises(ValueError, match="finite scalar"):
        validate_one_epoch(nn.Conv2d(3, 1, 1), DataLoader(Samples(count=1)), BadLoss(), "cpu")


from roadguard.training.engine import (train_distillation_one_epoch,
                                       validate_distillation_one_epoch)
from roadguard.training.losses import LogitDistillationLoss


def _kd_setup():
    student = nn.Conv2d(3, 1, 1)
    teacher = nn.Sequential(nn.BatchNorm2d(4), nn.Conv2d(4, 1, 1))
    criterion = LogitDistillationLoss(nn.BCEWithLogitsLoss(), kd_weight=0.5)
    return student, teacher, criterion


def test_distillation_train_updates_only_student():
    student, teacher, criterion = _kd_setup()
    student_before = [p.detach().clone() for p in student.parameters()]
    teacher_before = [p.detach().clone() for p in teacher.parameters()]
    stats_before = teacher[0].running_mean.clone()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    teacher.train()
    loss = train_distillation_one_epoch(
        student, teacher, DataLoader(Samples(), batch_size=2), criterion, optimizer, "cpu"
    )
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(p, old) for p, old in zip(student.parameters(), student_before))
    assert all(torch.equal(p, old) for p, old in zip(teacher.parameters(), teacher_before))
    assert torch.equal(teacher[0].running_mean, stats_before)
    assert teacher.training is False
    assert student.training is True
    assert all(p.grad is None for p in teacher.parameters())


def test_distillation_validation_preserves_both_models():
    student, teacher, criterion = _kd_setup()
    student_before = [p.detach().clone() for p in student.parameters()]
    teacher_before = [p.detach().clone() for p in teacher.parameters()]
    loss = validate_distillation_one_epoch(
        student, teacher, DataLoader(Samples(), batch_size=2), criterion, "cpu"
    )
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.equal(p, old) for p, old in zip(student.parameters(), student_before))
    assert all(torch.equal(p, old) for p, old in zip(teacher.parameters(), teacher_before))
    assert not student.training and not teacher.training
    assert all(p.grad is None for p in student.parameters())


def test_distillation_requires_decoded_depth():
    student, teacher, criterion = _kd_setup()
    with pytest.raises(ValueError, match="decoded depth"):
        validate_distillation_one_epoch(
            student, teacher, DataLoader(Samples(depth_channels=3)), criterion, "cpu"
        )


def test_distillation_empty_loader_rejected():
    student, teacher, criterion = _kd_setup()
    with pytest.raises(ValueError, match="no samples"):
        validate_distillation_one_epoch(
            student, teacher, DataLoader(Samples(count=0)), criterion, "cpu"
        )


def test_distillation_sample_weighted_mean():
    class Constant(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.channels = channels
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, *x.shape[-2:], device=x.device)

    class TargetMean(nn.Module):
        def forward(self, student_logits, targets, teacher_logits):
            return targets.mean()

    result = validate_distillation_one_epoch(
        Constant(3), Constant(4), DataLoader(Samples(), batch_size=2), TargetMean(), "cpu"
    )
    assert result == pytest.approx(0.4)


def test_distillation_rejects_nonfinite_loss():
    class BadKD(nn.Module):
        def forward(self, student_logits, targets, teacher_logits):
            return student_logits.mean() * float("nan")
    student, teacher, _ = _kd_setup()
    with pytest.raises(ValueError, match="finite scalar"):
        validate_distillation_one_epoch(
            student, teacher, DataLoader(Samples(count=1)), BadKD(), "cpu"
        )
