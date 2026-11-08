"""Small real end-to-end smoke test for the three-model LOFO experiment."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from roadguard.training.experiment import run_experiment


@pytest.mark.parametrize("architecture", ["unet", "segformer"])
def test_complete_three_role_experiment(tmp_path, architecture, monkeypatch):
    if architecture == "segformer":
        pytest.importorskip("transformers")
    held_out_family = "special"
    """Exercise actual images, CARLA depth decoding, training, KD, checkpoints, evaluation."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    # (family, object, scenario, split, anomaly visible): train and val use different objects.
    scenarios = [
        ("home", "box", "home_train", "train", True),
        ("home", "box", "home_train_empty", "train", False),
        ("home", "chair", "home_val", "val", True),
        ("home", "chair", "home_val_empty", "val", False),
        (held_out_family, "naga", "special_test", "test", True),
        (held_out_family, "naga", "special_test_empty", "test", False),
    ]
    for idx, (family, obj, scenario, split, anomalous) in enumerate(scenarios):
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
            "anomaly_family": family, "anomaly_object": obj, "scenario_id": scenario,
            "frame_id": "0", "split": split, "rgb_path": names[0],
            "anomaly_pixels": int((mask > 0).sum()),
            "depth_path": names[1], "mask_path": names[2],
        })
    # More training frames (same images, new ids) so that two independent sample streams cannot
    # coincide by chance.
    rows += [{**row, "frame_id": f"copy{i}"} for row in list(rows) if row["split"] == "train" for i in range(12)]
    fold_path = tmp_path / "special.csv"
    pd.DataFrame(rows).to_csv(fold_path, index=False)
    config = {
        "experiment": {"name": "smoke", "seed": 42, "lofo_family": held_out_family},
        "model": {"architecture": architecture, "base_channels": 2, "pretrained": False},
        "data": {"image_size": [32, 32], "batch_size": 2, "num_workers": 0},
        "teacher": {"depth_noise_std_m": 1.0,
                    "depth_dropout_probability": 1.0,
                    "depth_dropout_fraction": 0.2},
        "student": {"kd_weight": 0.5, "temperature": 2.0},
        "training": {"epochs": 2, "optimizer": "adamw", "learning_rate": 1e-3,
                     "weight_decay": 0.0, "select_min_frac": 0.0, "amp": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    from roadguard.training import experiment
    real_train_role, drawn = experiment._train_role, {}

    def recording_train_role(role, model, loaders, *args, **kwargs):
        drawn[role] = [list(loaders["train"].sampler) for _ in range(2)]   # first two epochs' draws
        return real_train_role(role, model, loaders, *args, **kwargs)

    monkeypatch.setattr(experiment, "_train_role", recording_train_role)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = run_experiment(config_path, fold_path, data_root,
                                 tmp_path / "outputs", device="cpu")
    finally:
        torch.set_num_threads(previous_threads)

    run_dir = tmp_path / "outputs" / "smoke" / held_out_family / "seed_42"
    assert set(results) == {"config", "teacher", "baseline", "student"}
    for role in ("teacher", "baseline", "student"):
        role_dir = run_dir / role
        for name in ("best.pt", "last.pt", "test_scenarios.csv", "test_scenarios_last.csv",
                     "val_scenarios.csv"):
            assert (role_dir / name).is_file()
        history = json.loads((role_dir / "history.json").read_text())
        assert len(history) == 2 and "val_macro_ap" in history[0]
        for which in ("selected", "last"):
            test = results[role][which]["test"]
            assert 0 <= test["pixel_ap"] <= 1
            assert "at_thr_dice" in test and "empty_fpr_pixels" in test
    assert drawn["baseline"] == drawn["student"]       # matched training samples for the comparison
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "config.json").is_file()
    # The reports module reads the finished run back.
    from roadguard.evaluation.reports import aggregate, fold_average, load_reports, per_scenario_ap
    table = load_reports(tmp_path / "outputs", ["smoke"], [held_out_family], [42],
                         roles=("teacher", "baseline", "student"))
    assert len(table) == 6 and set(table["role"]) == {"teacher", "baseline", "student"}
    assert len(aggregate(table)) == 6 and len(fold_average(table)) == 6
    assert per_scenario_ap(tmp_path / "outputs", ["smoke"], held_out_family, [42], role="student").shape[1] == 1
    counts = json.loads((run_dir / "partition_counts.json").read_text())
    assert {key: counts[key]["scenarios"] for key in counts} == {
        "train": 2, "val": 2, "test": 2,
    }
    # Output protection prevents overwriting a completed run.
    with pytest.raises(FileExistsError):
        run_experiment(config_path, fold_path, data_root,
                       tmp_path / "outputs", device="cpu")
