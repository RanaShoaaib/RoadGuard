import io
import tarfile
import pytest
import json
import pandas as pd
import numpy as np
from PIL import Image

from roadguard.data.preparation import (
    extract_anovox_archive,
    load_scenario_metadata,
    generate_anomaly_mask,
    inspect_semantic_labels,
    generate_anomaly_masks
)


def test_extract_anovox_archive(tmp_path):
    source = tmp_path / "source"

    files = {
        "Scenario_abc/RGB-CAM_test/RGB-CAM_test_1.png": b"rgb",
        "Scenario_abc/DEPTH_CAM_test/DEPTH_CAM_test_1.png": b"depth",
        "Scenario_abc/SEMANTIC-CAM_test/SEMANTIC-CAM_test_1.png": b"semantic",
        "Scenario_abc/LIDAR_test/LIDAR_test_1.npy": b"lidar",
        "Scenario_Configuration_Files/config.json": b"{}",
    }

    archive = tmp_path / "sample.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        for name, content in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

            tar.add(path, arcname=f"AnoVox_Static_Mono_Town04/{name}")

    destination = tmp_path / "processed"

    result = extract_anovox_archive(archive, destination)

    assert result["rgb"] == 1
    assert result["depth"] == 1
    assert result["semantic"] == 1
    assert result["metadata"] == 1

    assert len(list(destination.rglob("*.png"))) == 3
    assert not list(destination.rglob("*.npy"))

    # Verify repeated execution
    second_result = extract_anovox_archive(archive, destination)

    assert second_result["rgb"] == 0
    assert second_result["depth"] == 0
    assert second_result["semantic"] == 0
    assert second_result["metadata"] == 0


def test_extract_anovox_archive_rejects_unsafe_path(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    destination = tmp_path / "processed"

    with tarfile.open(archive, "w:gz") as tar:
        content = b"malicious data"

        member = tarfile.TarInfo(
            "AnoVox_Static_Mono_Town04/../Scenario_abc/"
            "RGB-CAM_test/RGB-CAM_test_1.png"
        )
        member.size = len(content)

        tar.addfile(member, io.BytesIO(content))

    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract_anovox_archive(archive, destination)

    assert not list(tmp_path.rglob("RGB-CAM_test_1.png"))


def test_extract_anovox_archive_repairs_incomplete_file(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    destination = tmp_path / "processed"

    member_name = (
        "AnoVox_Static_Mono_Town04/"
        "Scenario_abc/RGB-CAM_test/RGB-CAM_test_1.png"
    )

    original_content = b"complete image content"

    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(member_name)
        member.size = len(original_content)
        tar.addfile(member, io.BytesIO(original_content))

    target = destination / member_name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"incomplete")

    result = extract_anovox_archive(archive, destination)

    assert target.read_bytes() == original_content
    assert result["rgb"] == 1


def test_extract_anovox_archive_rejects_symlink_escape(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    destination = tmp_path / "processed"
    outside = tmp_path / "outside"

    destination.mkdir()
    outside.mkdir()

    try:
        (destination / "AnoVox_Static_Mono_Town04").symlink_to(
            outside,
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"Symlink creation not permitted: {exc}")

    member_name = (
        "AnoVox_Static_Mono_Town04/"
        "Scenario_abc/RGB-CAM_test/RGB-CAM_test_1.png"
    )

    content = b"rgb"

    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(member_name)
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))

    with pytest.raises(ValueError, match="Archive target escapes destination"):
        extract_anovox_archive(archive, destination)

    assert not list(outside.rglob("*.png"))


def test_load_scenario_metadata(tmp_path):
    config = {
        "scenario_definition": {
            "map": "Town04",
            "scenarios": [
                {
                    "id": "scenario_1",
                    "anomaly_config": {
                        "anomalytype": "STATIC",
                        "anomaly_bp_name": "static.prop.o_box_home",
                    },
                },
                {
                    "id": "scenario_2",
                    "anomaly_config": {
                        "anomalytype": "STATIC",
                        "anomaly_bp_name": "static.prop.o_wolf_animal",
                    },
                },
            ],
        }
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = load_scenario_metadata(config_path, source="static")

    assert len(result) == 2
    assert result["anomaly_family"].tolist() == ["home", "animal"]
    assert result["town"].tolist() == ["Town04", "Town04"]
    assert result["source"].tolist() == ["static", "static"]


def test_load_normality_metadata(tmp_path):
    config = {
        "scenario_definition": {
            "map": "Town04",
            "scenarios": [
                {"id": "normal_1"},
                {"id": "normal_2"},
            ],
        }
    }

    config_path = tmp_path / "normality.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = load_scenario_metadata(config_path, source="normality")

    assert len(result) == 2
    assert set(result["anomaly_family"]) == {"normality"}
    assert result["anomaly_blueprint"].isna().all()


def test_load_scenario_metadata_invalid_source(tmp_path):
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match="Invalid source"):
        load_scenario_metadata(
            config_path,
            source="dynamic",
        )


def test_generate_anomaly_mask(tmp_path):
    semantic = np.array(
        [
            [0, 1, 29, 30],
            [10, 100, 214, 0],
            [31, 32, 33, 34],
        ],
        dtype=np.uint8,
    )

    semantic_path = tmp_path / "semantic.png"
    mask_path = tmp_path / "masks" / "mask.png"

    Image.fromarray(semantic).save(semantic_path)

    anomaly_pixels = generate_anomaly_mask(
        semantic_path,
        mask_path,
    )

    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    expected = np.array(
        [
            [0, 0, 255, 255],
            [0, 255, 0, 0],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(mask, expected)
    assert anomaly_pixels == 7


def test_inspect_semantic_labels(tmp_path):
    semantic = np.array(
        [[0, 29, 29], [30, 100, 0]],
        dtype=np.uint8,
    )

    path = tmp_path / "semantic.png"
    Image.fromarray(semantic).save(path)

    result = inspect_semantic_labels(path)

    assert result == {0: 2, 29: 2, 30: 1, 100: 1}


def test_generate_anomaly_masks(tmp_path):
    data_root = tmp_path / "processed"

    semantic_dir = (
        data_root
        / "AnoVox_Static_Mono_Town04"
        / "Scenario_abc"
        / "SEMANTIC-CAM_test"
    )
    semantic_dir.mkdir(parents=True)

    semantic = np.array(
        [
            [0, 29, 114],
            [32, 0, 30],
        ],
        dtype=np.uint8,
    )

    Image.fromarray(semantic).save(
        semantic_dir / "SEMANTIC-CAM_test_100.png"
    )

    result = generate_anomaly_masks(data_root)

    assert len(result) == 1
    assert result.iloc[0]["scenario_id"] == "abc"
    assert result.iloc[0]["frame_id"] == "100"
    assert result.iloc[0]["anomaly_pixels"] == 3

    mask_path = data_root / result.iloc[0]["mask_path"]

    assert mask_path.is_file()

    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    assert np.array_equal(
        mask,
        np.array(
            [
                [0, 255, 0],
                [255, 0, 255],
            ],
            dtype=np.uint8,
        ),
    )


def test_generate_anomaly_masks_rejects_invalid_existing_mask(tmp_path):
    data_root = tmp_path / "processed"
    scenario_dir = data_root / "Scenario_abc"

    semantic_dir = scenario_dir / "SEMANTIC-CAM_test"
    mask_dir = scenario_dir / "MASK"

    semantic_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    semantic = np.array([[0, 29], [30, 0]], dtype=np.uint8)
    invalid_mask = np.array([[0, 128], [255, 0]], dtype=np.uint8)

    Image.fromarray(semantic).save(
        semantic_dir / "SEMANTIC-CAM_test_100.png"
    )
    Image.fromarray(invalid_mask).save(
        mask_dir / "MASK_100.png"
    )

    with pytest.raises(ValueError, match="Invalid mask values"):
        generate_anomaly_masks(data_root)


def test_generate_anomaly_masks_rejects_duplicate_frame_ids(tmp_path):
    data_root = tmp_path / "processed"
    scenario_dir = data_root / "Scenario_abc"

    semantic_dir_1 = scenario_dir / "SEMANTIC-CAM_one"
    semantic_dir_2 = scenario_dir / "SEMANTIC-CAM_two"

    semantic_dir_1.mkdir(parents=True)
    semantic_dir_2.mkdir(parents=True)

    semantic = np.array(
        [[0, 29], [30, 0]],
        dtype=np.uint8,
    )

    Image.fromarray(semantic).save(
        semantic_dir_1 / "SEMANTIC-CAM_one_100.png"
    )

    Image.fromarray(semantic).save(
        semantic_dir_2 / "SEMANTIC-CAM_two_100.png"
    )

    with pytest.raises(ValueError, match="Duplicate frame ID"):
        generate_anomaly_masks(data_root)