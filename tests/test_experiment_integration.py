"""Small real end-to-end smoke test for the three-model LOFO experiment."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from roadguard.training.experiment import run_experiment


@pytest.mark.parametrize("held_out_family", ["special"])
def test_complete_three_role_experiment(tmp_path, held_out_family):
    """Exercise actual images, CARLA depth decoding, training, KD, checkpoints, evaluation."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    # Two training scenarios, two validation scenarios, two test scenarios.
    scenarios = [
        ("static", "home", "home_train", "train", True),
        ("normality", "normality", "normal_train", "train", False),
        ("static", "home", "home_val", "val", True),
        ("normality", "normality", "normal_val", "val", False),
        ("static", held_out_family, "special_test", "test", True),
        ("normality", "normality", "normal_test", "test", False),
    ]
    for idx, (source, family, scenario, split, anomalous) in enumerate(scenarios):
        rgb = np.full((32, 32, 3), 65 + idx * 20, dtype=np.uint8)
        rgb[:, :, 1] = 90
        # CARLA depth encoding: R is least-significant byte, B most-significant.
        depth = np.zeros((32, 32, 3), dtype=np.uint8)
        depth[:, :, 2] = 2 + idx
        mask = np.zeros((32, 32), dtype=np.uint8)
        if anomalous:
            mask[8:24, 8:24] = 255
        names = [f"{scenario}_{kind}.png" for kind in ("rgb", "depth", "mask")]
        for name, array in zip(names, (rgb, depth, mask)):
            Image.fromarray(array).save(data_root / name)
        rows.append({
            "source": source, "anomaly_family": family, "scenario_id": scenario,
            "frame_id": "0", "split": split, "rgb_path": names[0],
            "depth_path": names[1], "mask_path": names[2],
        })
    fold_path = tmp_path / "special.csv"
    pd.DataFrame(rows).to_csv(fold_path, index=False)
    config = {
        "experiment": {"name": "smoke", "seed": 42, "lofo_family": held_out_family},
        "model": {"architecture": "unet", "base_channels": 2, "pretrained": False},
        "data": {"image_size": [32, 32], "batch_size": 2, "num_workers": 0},
        "teacher": {"depth_noise_std_m": 1.0,
                    "depth_dropout_probability": 1.0,
                    "depth_dropout_fraction": 0.2},
        "student": {"kd_weight": 0.5, "temperature": 2.0},
        "training": {"epochs": 1, "optimizer": "adamw", "learning_rate": 1e-3,
                     "weight_decay": 0.0, "supervised_loss": "bce_dice"},
        "evaluation": {"threshold": 0.5},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = run_experiment(config_path, fold_path, data_root,
                                 tmp_path / "outputs", device="cpu")
    finally:
        torch.set_num_threads(previous_threads)

    run_dir = tmp_path / "outputs" / "smoke" / held_out_family / "seed_42"
    assert set(results) == {"teacher", "baseline", "student"}
    for role in results:
        role_dir = run_dir / role
        assert (role_dir / "best.pt").is_file()
        assert (role_dir / "latest.pt").is_file()
        history = json.loads((role_dir / "history.json").read_text())
        assert len(history) == 1
        assert "val_supervised_loss" in history[0]
        assert results[role]["positive_pixels"] > 0
        assert results[role]["negative_pixels"] > 0
        assert 0 <= results[role]["pixel_auprc"] <= 1
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "config.json").is_file()
    counts = json.loads((run_dir / "partition_counts.json").read_text())
    assert {key: counts[key]["scenarios"] for key in counts} == {
        "train": 2, "val": 2, "test": 2,
    }
    # Output protection prevents overwriting a completed run.
    with pytest.raises(FileExistsError):
        run_experiment(config_path, fold_path, data_root,
                       tmp_path / "outputs", device="cpu")
