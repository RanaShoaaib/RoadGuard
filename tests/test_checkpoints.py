"""Unit tests for RoadGuard checkpoint management."""
import copy

import pytest
import torch
from torch import nn

from roadguard.training.checkpoints import (
    load_checkpoint, save_checkpoint, save_epoch_checkpoints,
)


def make_model_optimizer():
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def train_step(model, optimizer):
    optimizer.zero_grad()
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()


def test_round_trip_model_and_metadata(tmp_path):
    model, optimizer = make_model_optimizer()
    train_step(model, optimizer)
    original = copy.deepcopy(model.state_dict())
    path = tmp_path / "nested" / "latest.pt"
    save_checkpoint(path, model, optimizer, epoch=2, val_loss=0.3,
                    best_val_loss=0.2, config={"fold": "vehicle"})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    info = load_checkpoint(path, model, optimizer)
    assert info == {"epoch": 2, "val_loss": 0.3, "best_val_loss": 0.2,
                    "config": {"fold": "vehicle"}}
    for key, value in original.items():
        assert torch.equal(model.state_dict()[key], value)


def test_optimizer_restored(tmp_path):
    model, optimizer = make_model_optimizer()
    train_step(model, optimizer)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, epoch=0, val_loss=0.5, config={})
    restored_model, restored_optimizer = make_model_optimizer()
    load_checkpoint(path, restored_model, restored_optimizer)
    old_states = list(optimizer.state.values())
    new_states = list(restored_optimizer.state.values())
    assert len(old_states) == len(new_states) > 0
    for old, new in zip(old_states, new_states):
        assert torch.equal(old["exp_avg"], new["exp_avg"])
        assert torch.equal(old["exp_avg_sq"], new["exp_avg_sq"])


def test_model_only_load(tmp_path):
    model, optimizer = make_model_optimizer()
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, optimizer, epoch=0, val_loss=0.1, config={})
    other, _ = make_model_optimizer()
    assert load_checkpoint(path, other)["epoch"] == 0
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters()))


def test_best_and_latest_selection(tmp_path):
    model, optimizer = make_model_optimizer()
    best = float("inf")
    for epoch, val_loss in enumerate([0.5, 0.7, 0.4]):
        best = save_epoch_checkpoints(tmp_path, model, optimizer, epoch=epoch,
                                      val_loss=val_loss, best_val_loss=best,
                                      config={"model": "unet"})
        if epoch == 1:
            assert load_checkpoint(tmp_path / "best.pt", model)["epoch"] == 0
            assert load_checkpoint(tmp_path / "latest.pt", model)["epoch"] == 1
    assert best == pytest.approx(0.4)
    assert load_checkpoint(tmp_path / "best.pt", model)["epoch"] == 2
    assert load_checkpoint(tmp_path / "latest.pt", model)["best_val_loss"] == pytest.approx(0.4)


def test_equal_loss_does_not_replace_best(tmp_path):
    model, optimizer = make_model_optimizer()
    best = save_epoch_checkpoints(tmp_path, model, optimizer, epoch=0,
                                  val_loss=0.5, best_val_loss=float("inf"), config={})
    save_epoch_checkpoints(tmp_path, model, optimizer, epoch=1,
                           val_loss=0.5, best_val_loss=best, config={})
    assert load_checkpoint(tmp_path / "best.pt", model)["epoch"] == 0


def test_separate_experiment_directories(tmp_path):
    model, optimizer = make_model_optimizer()
    for fold in ("vehicle", "pedestrian"):
        save_epoch_checkpoints(tmp_path / fold, model, optimizer, epoch=0,
                               val_loss=0.4, best_val_loss=float("inf"), config={"fold": fold})
    for fold in ("vehicle", "pedestrian"):
        assert load_checkpoint(tmp_path / fold / "best.pt", model)["config"]["fold"] == fold


def test_negative_epoch_rejected(tmp_path):
    model, optimizer = make_model_optimizer()
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "x.pt", model, optimizer, epoch=-1,
                        val_loss=0.2, config={})


def test_missing_checkpoint(tmp_path):
    model, _ = make_model_optimizer()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)
