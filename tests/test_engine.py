"""Tests for the RoadGuard training and evaluation passes."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from roadguard.training.engine import EMA, build_scheduler, collect, train_one_epoch
from roadguard.training.losses import FocalDiceLoss, LogitDistillationLoss


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
            "scenario_id": "s%d" % (index % 2),
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


def test_loss_is_sample_weighted_mean():
    class MeanTargetLoss(nn.Module):
        def forward(self, logits, targets):
            return targets.mean() + 0 * logits.mean()
    model = nn.Conv2d(3, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    # Batch means: (0+1)/2, (0+1)/2, 0 -> sample-weighted 2/5.
    result = train_one_epoch(model, DataLoader(Samples(), batch_size=2), MeanTargetLoss(), optimizer, "cpu")
    assert result == pytest.approx(0.4)


def test_scheduler_ema_and_clipping_are_applied():
    model = nn.Conv2d(3, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(Samples(), batch_size=2)
    scheduler = build_scheduler(optimizer, total_steps=3, warmup_frac=0.34)
    ema = EMA(model, decay=0.9)
    start = ema.module.weight.detach().clone()
    train_one_epoch(model, loader, nn.BCEWithLogitsLoss(), optimizer, "cpu",
                    scheduler=scheduler, ema=ema, grad_clip=1.0)
    assert ema.steps == 3
    assert not torch.equal(ema.module.weight, start)
    assert not torch.equal(ema.module.weight, model.weight)       # averaged, not copied
    assert optimizer.param_groups[0]["lr"] < 0.1 * 0.01           # cosine decayed to the end


def test_rgb_only_does_not_require_depth():
    class RGBOnly(Dataset):
        def __len__(self):
            return 1
        def __getitem__(self, i):
            return {"image": torch.zeros(3, 8, 8), "mask": torch.zeros(1, 8, 8), "scenario_id": "a"}
    model = nn.Conv2d(3, 1, 1)
    assert len(collect(model, DataLoader(RGBOnly()), "cpu").scen) == 1


def test_four_channels_requires_decoded_depth():
    with pytest.raises(ValueError, match="decoded depth"):
        collect(nn.Conv2d(4, 1, 1), DataLoader(Samples(depth_channels=3)), "cpu", 4)


def test_empty_loader_rejected():
    with pytest.raises(ValueError, match="no samples"):
        collect(nn.Conv2d(3, 1, 1), DataLoader(Samples(count=0)), "cpu")


def test_invalid_channel_count():
    with pytest.raises(ValueError, match="in_channels"):
        collect(nn.Conv2d(3, 1, 1), DataLoader(Samples(count=1)), "cpu", 5)


def test_nonfinite_loss_rejected():
    class BadLoss(nn.Module):
        def forward(self, logits, targets):
            return logits.mean() * float("nan")
    model = nn.Conv2d(3, 1, 1)
    with pytest.raises(ValueError, match="finite scalar"):
        train_one_epoch(model, DataLoader(Samples(count=1)), BadLoss(),
                        torch.optim.SGD(model.parameters(), lr=0.1), "cpu")


def test_collect_preserves_parameters_and_sets_eval():
    model = nn.Conv2d(3, 1, 1)
    original = model.weight.detach().clone()
    acc = collect(model, DataLoader(Samples(), batch_size=2), "cpu")
    assert torch.equal(model.weight, original) and model.weight.grad is None
    assert not model.training
    assert set(acc.scen) == {"s0", "s1"}


def test_distillation_updates_only_student_and_keeps_teacher_frozen():
    student = nn.Conv2d(3, 1, 1)
    teacher = nn.Sequential(nn.BatchNorm2d(4), nn.Conv2d(4, 1, 1))
    criterion = LogitDistillationLoss(FocalDiceLoss(), kd_weight=0.5)
    student_before = [p.detach().clone() for p in student.parameters()]
    teacher_before = [p.detach().clone() for p in teacher.parameters()]
    stats_before = teacher[0].running_mean.clone()
    teacher.train()
    loss = train_one_epoch(student, DataLoader(Samples(), batch_size=2), criterion,
                           torch.optim.SGD(student.parameters(), lr=0.1), "cpu", 3, teacher=teacher)
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(p, o) for p, o in zip(student.parameters(), student_before))
    assert all(torch.equal(p, o) for p, o in zip(teacher.parameters(), teacher_before))
    assert torch.equal(teacher[0].running_mean, stats_before)
    assert teacher.training is False and student.training is True
    assert all(p.grad is None for p in teacher.parameters())


def test_distillation_requires_decoded_depth():
    student, teacher = nn.Conv2d(3, 1, 1), nn.Conv2d(4, 1, 1)
    criterion = LogitDistillationLoss(FocalDiceLoss())
    with pytest.raises(ValueError, match="decoded depth"):
        train_one_epoch(student, DataLoader(Samples(depth_channels=3)), criterion,
                        torch.optim.SGD(student.parameters(), lr=0.1), "cpu", teacher=teacher)


def test_ema_update_formula_and_buffers_copied():
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2))
    ema = EMA(model, decay=0.9)
    ema.steps = 1000                                    # past warm-up so the decay is exactly 0.9
    before = [p.clone() for p in ema.module.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        model[1].running_mean.fill_(5.0)
    ema.update(model)
    for e, b, p in zip(ema.module.parameters(), before, model.parameters()):
        assert torch.allclose(e, 0.9 * b + 0.1 * p, atol=1e-6)
    assert torch.allclose(ema.module[1].running_mean, model[1].running_mean)
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_ema_rejects_bad_decay():
    with pytest.raises(ValueError, match="decay"):
        EMA(nn.Linear(1, 1), 1.0)


def test_collect_scores_at_native_resolution_when_available():
    """The model sees 8x8 but metrics are computed on the 16x16 original masks."""
    class Native(Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, i):
            native = torch.zeros(1, 16, 16)
            native[:, 4:12, 4:12] = float(i % 2)
            return {"image": torch.zeros(3, 8, 8), "mask": torch.zeros(1, 8, 8), "mask_native": native,
                    "scenario_id": "s%d" % (i % 2)}
    acc = collect(nn.Conv2d(3, 1, 1), DataLoader(Native(), batch_size=2), "cpu")
    pooled = acc.pooled
    assert int(pooled.pos.sum() + pooled.neg.sum()) == 4 * 16 * 16        # every native pixel scored once
    assert int(pooled.pos.sum()) == 2 * 8 * 8                             # positives from the native masks
