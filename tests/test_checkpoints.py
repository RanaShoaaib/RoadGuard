"""Unit tests for checkpoint saving/loading and epoch selection."""
import pytest
import torch
from torch import nn

from roadguard.training.checkpoints import load_checkpoint, save_checkpoint, select_epoch


def test_round_trip_model_and_metadata(tmp_path):
    model = nn.Linear(3, 1)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "nested" / "best.pt"
    save_checkpoint(path, model, epoch=2, score=0.3, config={"fold": "special"})
    assert not path.with_name(path.name + ".tmp").exists()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    info = load_checkpoint(path, model)
    assert info == {"epoch": 2, "score": 0.3, "config": {"fold": "special"}}
    for key, value in original.items():
        assert torch.equal(model.state_dict()[key], value)


def test_negative_epoch_rejected(tmp_path):
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path / "x.pt", nn.Linear(1, 1), epoch=-1, score=0.2, config={})


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", nn.Linear(1, 1))


E2_VAL_MACRO_AP = [0.4481, 0.2213, 0.4319, 0.6567, 0.4765, 0.3896, 0.2756, 0.3645, 0.5264, 0.4512,
                   0.5348, 0.4016, 0.4406, 0.4716, 0.4616, 0.4690, 0.4447, 0.4961, 0.4993, 0.4992]


def test_select_epoch_ignores_early_noisy_peak_from_real_e2_log():
    assert max(range(20), key=lambda i: E2_VAL_MACRO_AP[i]) == 3      # the raw pick that hurt E2
    i, s = select_epoch(E2_VAL_MACRO_AP, window=3, min_frac=0.5, total_epochs=20)
    assert i is not None and i >= 9 and i != 3
    assert abs(s - sum(E2_VAL_MACRO_AP[i - 2:i + 1]) / 3) < 1e-9


def test_select_epoch_plain_best_and_nan_handling():
    assert select_epoch([0.1, 0.5, 0.2], window=1, min_frac=0.0)[0] == 1
    assert select_epoch([float("nan")] * 4, window=2, min_frac=0.5)[0] is None
    assert select_epoch([0.2, float("nan"), 0.4, 0.3], window=2, min_frac=0.0)[0] == 2
    assert select_epoch([0.5, 0.5], window=1, min_frac=0.0)[0] == 0      # ties keep the earlier epoch
