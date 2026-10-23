"""Unit tests for LOFO fold validation and depth augmentation routing."""
import pandas as pd
import pytest

from roadguard.training.experiment import load_lofo_fold, make_loaders


def sample_fold():
    rows = []
    for source, family, scenario, split in [
        ("static", "special", "s1", "test"),
        ("static", "home", "s2", "train"),
        ("static", "home", "s3", "val"),
        ("normality", "normality", "n1", "train"),
        ("normality", "normality", "n2", "val"),
        ("normality", "normality", "n3", "test"),
    ]:
        rows.append(dict(source=source, anomaly_family=family, scenario_id=scenario,
                         frame_id="0", split=split, rgb_path="r.png",
                         depth_path="d.png", mask_path="m.png"))
    return pd.DataFrame(rows)


def test_fold_keeps_normality_and_special(tmp_path):
    path = tmp_path / "special.csv"
    sample_fold().to_csv(path, index=False)
    parts = load_lofo_fold(path, "special")
    assert {k: len(v) for k, v in parts.items()} == {"train": 2, "val": 2, "test": 2}
    assert set(parts["test"]["source"]) == {"static", "normality"}


def test_fold_rejects_scenario_leakage(tmp_path):
    fold = sample_fold()
    fold.loc[1, "scenario_id"] = "s3"
    fold.loc[1, "frame_id"] = "1"
    path = tmp_path / "special.csv"
    fold.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Scenario leakage"):
        load_lofo_fold(path, "special")


def test_fold_rejects_heldout_in_training(tmp_path):
    fold = sample_fold()
    fold.loc[0, "split"] = "train"
    path = tmp_path / "special.csv"
    fold.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Held-out family"):
        load_lofo_fold(path, "special")


def test_teacher_only_depth_augmentation():
    config = {"data": {"image_size": [32, 32], "batch_size": 2, "num_workers": 0},
              "teacher": {"depth_noise_std_m": 1.0,
                          "depth_dropout_probability": 0.3,
                          "depth_dropout_fraction": 0.2}}
    parts = {k: sample_fold().query("split == @k") for k in ("train", "val", "test")}
    teacher = make_loaders(parts, ".", config, teacher_train=True)
    rgb = make_loaders(parts, ".", config, teacher_train=False)
    assert teacher["train"].dataset.transform.depth_noise_std_m == 1.0
    assert teacher["train"].dataset.transform.depth_dropout_probability == 0.3
    assert rgb["train"].dataset.transform.depth_noise_std_m == 0.0
    assert rgb["train"].dataset.transform.depth_dropout_probability == 0.0
    assert teacher["val"].dataset.transform.depth_noise_std_m == 0.0
    assert teacher["test"].dataset.transform.depth_dropout_probability == 0.0
