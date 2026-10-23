"""Tests for fixed RoadGuard experiment configuration."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from roadguard.training.config import load_config, validate_config


@pytest.fixture
def valid():
    return {
        "experiment": {"name": "test", "seed": 42, "lofo_family": "vehicle"},
        "model": {"architecture": "unet", "base_channels": 32, "pretrained": False},
        "data": {"image_size": [256, 256], "batch_size": 8, "num_workers": 0},
        "teacher": {"depth_noise_std_m": 1.0, "depth_dropout_probability": .3,
                    "depth_dropout_fraction": .2},
        "student": {"kd_weight": .5, "temperature": 2.0},
        "training": {"epochs": 20, "optimizer": "adamw", "learning_rate": 1e-4,
                     "weight_decay": .01, "supervised_loss": "bce_dice"},
        "evaluation": {"threshold": .5},
    }


def test_valid_config_and_no_mutation(valid):
    original = deepcopy(valid)
    assert validate_config(valid) == valid
    assert valid == original


def test_load_yaml(tmp_path, valid):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(valid), encoding="utf-8")
    assert load_config(path) == valid


@pytest.mark.parametrize("section,key,value,match", [
    ("experiment", "lofo_family", "", "lofo_family"),
    ("experiment", "seed", True, "seed"),
    ("model", "architecture", "resnet", "architecture"),
    ("data", "image_size", [0, 256], "image_size"),
    ("data", "batch_size", 0, "batch_size"),
    ("teacher", "depth_noise_std_m", -1, "depth_noise"),
    ("teacher", "depth_dropout_probability", 1.1, "dropout_probability"),
    ("teacher", "depth_dropout_fraction", -0.1, "dropout_fraction"),
    ("student", "kd_weight", 1.5, "kd_weight"),
    ("student", "temperature", 0, "temperature"),
    ("training", "epochs", 0, "epochs"),
    ("training", "optimizer", "sgd", "optimizer"),
    ("training", "learning_rate", 0, "learning_rate"),
    ("training", "supervised_loss", "mse", "supervised_loss"),
    ("evaluation", "threshold", 1.2, "threshold"),
])
def test_invalid_values(valid, section, key, value, match):
    valid[section][key] = value
    with pytest.raises(ValueError, match=match):
        validate_config(valid)


def test_missing_section(valid):
    del valid["student"]
    with pytest.raises(ValueError, match="Missing"):
        validate_config(valid)


def test_unknown_key(valid):
    valid["training"]["dropout"] = 0.1
    with pytest.raises(ValueError, match="Unknown"):
        validate_config(valid)


def test_invalid_yaml_root(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
