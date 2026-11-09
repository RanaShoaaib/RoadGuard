"""Tests for fixed RoadGuard experiment configuration."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from roadguard.training.config import AUGMENT_DEFAULTS, ROLES, TRAINING_DEFAULTS, load_config, validate_config


@pytest.fixture
def valid():
    return {
        "experiment": {"name": "test", "seed": 42, "lofo_family": "vehicle", "roles": list(ROLES)},
        "model": {"architecture": "unet", "base_channels": 32, "pretrained": False},
        "data": {"image_size": [256, 256], "batch_size": 8, "num_workers": 0, "eval_native": True,
                 "augment": dict(AUGMENT_DEFAULTS)},
        "teacher": {"depth_noise_std_m": 1.0, "depth_dropout_probability": .3,
                    "depth_dropout_fraction": .2},
        "student": {"kd_weight": .5, "temperature": 2.0},
        "training": {"epochs": 20, "optimizer": "adamw", "learning_rate": 1e-4,
                     "weight_decay": .01, **TRAINING_DEFAULTS},
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
    ("training", "focal_alpha", 1.5, "focal_alpha"),
    ("training", "ema_decay", 1.0, "ema_decay"),
    ("training", "pos_share_within", 0, "pos_share_within"),
    ("training", "amp", "yes", "amp"),
    ("data", "eval_native", "yes", "eval_native"),
    ("data", "augment", {"brightness": 1.0}, "brightness"),
    ("data", "augment", {"crop_min_scale": 0.0}, "crop_min_scale"),
    ("data", "augment", {"rotation": 10}, "rotation"),
    ("experiment", "roles", ["student"], "teacher"),
    ("experiment", "roles", ["oracle"], "roles"),
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


def test_recipe_defaults_filled(valid):
    for key in TRAINING_DEFAULTS:
        del valid["training"][key]
    del valid["experiment"]["roles"]
    config = validate_config(valid)
    assert config["training"]["focal_alpha"] == 0.25
    assert config["training"]["ema_decay"] == 0.998
    assert config["experiment"]["roles"] == ["teacher", "baseline", "student"]


def test_roles_keep_canonical_order(valid):
    valid["experiment"]["roles"] = ["baseline", "teacher"]
    assert validate_config(valid)["experiment"]["roles"] == ["teacher", "baseline"]


def test_shipped_configs_validate():
    for path in sorted((Path(__file__).parents[1] / "configs").glob("*.yaml")):
        if path.name != "data_sources.yaml":
            load_config(path)


def test_data_defaults_filled(valid):
    del valid["data"]["eval_native"], valid["data"]["augment"]
    config = validate_config(valid)
    assert config["data"]["eval_native"] is True
    assert config["data"]["augment"] == AUGMENT_DEFAULTS            # augmentation off by default


def test_partial_augment_section_keeps_other_defaults(valid):
    valid["data"]["augment"] = {"crop_min_scale": 0.75}
    augment = validate_config(valid)["data"]["augment"]
    assert augment["crop_min_scale"] == 0.75 and augment["brightness"] == 0.0


def test_shipped_configs_use_native_input_and_mild_augmentation():
    for name in ("unet", "segformer"):
        config = load_config(Path(__file__).parents[1] / "configs" / f"{name}.yaml")
        assert config["data"]["image_size"] == [512, 768]            # native frames, no resizing
        assert all(v % 32 == 0 for v in config["data"]["image_size"])  # valid for U-Net and SegFormer
        augment = config["data"]["augment"]
        assert augment["crop_min_scale"] < 1 and augment["brightness"] > 0
