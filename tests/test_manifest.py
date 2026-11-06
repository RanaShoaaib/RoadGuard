import numpy as np
import pandas as pd
from PIL import Image
import pytest

from roadguard.data.manifest import build_master_manifest, validate_manifest



def test_build_master_manifest_discovers_rgb_depth(tmp_path):
    data_root = tmp_path / "processed"

    scenario_dir = (
        data_root
        / "AnoVox_Static_Mono_Town04"
        / "Scenario_abc"
    )

    rgb_dir = scenario_dir / "RGB-CAM_test"
    depth_dir = scenario_dir / "DEPTH_CAM_test"
    semantic_dir = scenario_dir / "SEMANTIC-CAM_test"
    mask_dir = scenario_dir / "MASK"

    for directory in (rgb_dir, depth_dir, semantic_dir, mask_dir):
        directory.mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint8)

    semantic = np.array(
        [[0, 29, 0], [30, 0, 0]],
        dtype=np.uint8,
    )

    mask = np.isin(
        semantic, [29, 30]
    ).astype(np.uint8) * 255

    rgb_path = rgb_dir / "RGB-CAM_test_100.png"
    depth_path = depth_dir / "DEPTH_CAM_test_100.png"
    semantic_path = semantic_dir / "SEMANTIC-CAM_test_100.png"
    mask_path = mask_dir / "MASK_100.png"

    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(depth).save(depth_path)
    Image.fromarray(semantic).save(semantic_path)
    Image.fromarray(mask).save(mask_path)

    mask_metadata = pd.DataFrame([{
        "scenario_id": "abc",
        "frame_id": "100",
        "semantic_path": str(semantic_path.relative_to(data_root)),
        "mask_path": str(mask_path.relative_to(data_root)),
        "anomaly_pixels": 2,
    }])

    scenario_metadata = pd.DataFrame([{
        "scenario_id": "abc",
        "town": "Town04",
        "anomaly_family": "home",
        "anomaly_blueprint": "static.prop.o_box_home",
        "anomaly_object": "box",
    }])

    result = build_master_manifest(
        data_root,
        scenario_metadata=scenario_metadata,
        mask_metadata=mask_metadata,
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert "source" not in result.columns
    assert row["scenario_id"] == "abc"
    assert row["frame_id"] == "100"

    assert row["rgb_path"] == str(rgb_path.relative_to(data_root))
    assert row["depth_path"] == str(depth_path.relative_to(data_root))
    assert row["semantic_path"] == str(semantic_path.relative_to(data_root))
    assert row["mask_path"] == str(mask_path.relative_to(data_root))

    assert row["anomaly_pixels"] == 2
    assert bool(row["has_anomaly"]) is True

    assert row["town"] == "Town04"
    assert row["anomaly_family"] == "home"
    assert row["anomaly_blueprint"] == "static.prop.o_box_home"
    assert row["anomaly_object"] == "box"



def test_build_master_manifest_rejects_missing_scenario_metadata(tmp_path):
    data_root = tmp_path / "processed"
    scenario_dir = (
        data_root / "AnoVox_Static_Mono_Town04" / "Scenario_abc"
    )

    rgb_dir = scenario_dir / "RGB-CAM_test"
    depth_dir = scenario_dir / "DEPTH_CAM_test"

    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint8)

    Image.fromarray(rgb).save(rgb_dir / "RGB-CAM_test_100.png")
    Image.fromarray(depth).save(depth_dir / "DEPTH_CAM_test_100.png")

    mask_metadata = pd.DataFrame([{
        "scenario_id": "abc",
        "frame_id": "100",
        "semantic_path": (
            "AnoVox_Static_Mono_Town04/Scenario_abc/"
            "SEMANTIC-CAM_test/SEMANTIC-CAM_test_100.png"
        ),
        "mask_path": (
            "AnoVox_Static_Mono_Town04/Scenario_abc/MASK/MASK_100.png"
        ),
        "anomaly_pixels": 0,
    }])

    scenario_metadata = pd.DataFrame([{
        "scenario_id": "different_scenario",
        "town": "Town04",
        "anomaly_family": "home",
        "anomaly_blueprint": "static.prop.o_box_home",
        "anomaly_object": "box",
    }])

    with pytest.raises(ValueError, match="Missing scenario metadata"):
        build_master_manifest(
            data_root,
            scenario_metadata=scenario_metadata,
            mask_metadata=mask_metadata,
        )



def test_build_master_manifest_rejects_duplicate_rgb_frames(tmp_path):
    data_root = tmp_path / "processed"

    scenario_dir = (
        data_root
        / "AnoVox_Static_Mono_Town04"
        / "Scenario_abc"
    )

    rgb_dir_1 = scenario_dir / "RGB-CAM_one"
    rgb_dir_2 = scenario_dir / "RGB-CAM_two"
    depth_dir = scenario_dir / "DEPTH_CAM_test"

    for directory in (rgb_dir_1, rgb_dir_2, depth_dir):
        directory.mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint8)

    Image.fromarray(rgb).save(
        rgb_dir_1 / "RGB-CAM_one_100.png"
    )

    Image.fromarray(rgb).save(
        rgb_dir_2 / "RGB-CAM_two_100.png"
    )

    Image.fromarray(depth).save(
        depth_dir / "DEPTH_CAM_test_100.png"
    )

    with pytest.raises(ValueError, match="Duplicate RGB frame"):
        build_master_manifest(
            data_root,
            scenario_metadata=pd.DataFrame(),
            mask_metadata=pd.DataFrame(),
        )


@pytest.fixture
def valid_manifest_data(tmp_path):
    data_root = tmp_path / "processed"

    scenario_dir = (
        data_root
        / "AnoVox_Static_Mono_Town04"
        / "Scenario_abc"
    )

    rgb_dir = scenario_dir / "RGB-CAM_test"
    depth_dir = scenario_dir / "DEPTH_CAM_test"
    semantic_dir = scenario_dir / "SEMANTIC-CAM_test"
    mask_dir = scenario_dir / "MASK"

    for directory in (rgb_dir, depth_dir, semantic_dir, mask_dir):
        directory.mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint8)
    semantic = np.array(
        [[0, 29, 0], [30, 0, 0]],
        dtype=np.uint8,
    )
    mask = np.isin(semantic, [29, 30]).astype(np.uint8) * 255

    paths = {
        "rgb": rgb_dir / "RGB-CAM_test_100.png",
        "depth": depth_dir / "DEPTH_CAM_test_100.png",
        "semantic": semantic_dir / "SEMANTIC-CAM_test_100.png",
        "mask": mask_dir / "MASK_100.png",
    }

    for modality, array in zip(
        ("rgb", "depth", "semantic", "mask"),
        (rgb, depth, semantic, mask),
    ):
        Image.fromarray(array).save(paths[modality])

    manifest = pd.DataFrame([{
        "scenario_id": "abc",
        "frame_id": "100",
        "rgb_path": str(paths["rgb"].relative_to(data_root)),
        "depth_path": str(paths["depth"].relative_to(data_root)),
        "semantic_path": str(paths["semantic"].relative_to(data_root)),
        "mask_path": str(paths["mask"].relative_to(data_root)),
        "anomaly_pixels": 2,
    }])

    return data_root, manifest, paths


def test_validate_manifest_success(valid_manifest_data):
    data_root, manifest, _ = valid_manifest_data

    assert validate_manifest(manifest, data_root) is None


def test_validate_manifest_missing_file(valid_manifest_data):
    data_root, manifest, paths = valid_manifest_data

    paths["depth"].unlink()

    with pytest.raises(ValueError, match="Missing depth file"):
        validate_manifest(manifest, data_root)


def test_validate_manifest_dimension_mismatch(valid_manifest_data):
    data_root, manifest, paths = valid_manifest_data

    incorrect_depth = np.zeros((4, 3), dtype=np.uint8)
    Image.fromarray(incorrect_depth).save(paths["depth"])

    with pytest.raises(ValueError, match="Image dimension mismatch"):
        validate_manifest(manifest, data_root)


def test_validate_manifest_invalid_mask(valid_manifest_data):
    data_root, manifest, paths = valid_manifest_data

    invalid_mask = np.array(
        [[0, 128, 0], [255, 0, 0]],
        dtype=np.uint8,
    )
    Image.fromarray(invalid_mask).save(paths["mask"])

    with pytest.raises(ValueError, match="Invalid mask values"):
        validate_manifest(manifest, data_root)


def test_validate_manifest_pixel_count_mismatch(valid_manifest_data):
    data_root, manifest, _ = valid_manifest_data

    manifest.loc[0, "anomaly_pixels"] = 5

    with pytest.raises(ValueError, match="Anomaly pixel count mismatch"):
        validate_manifest(manifest, data_root)


def test_validate_manifest_orphan_depth(valid_manifest_data):
    data_root, manifest, paths = valid_manifest_data

    orphan_path = paths["depth"].parent / "DEPTH_CAM_test_200.png"

    Image.fromarray(
        np.zeros((2, 3), dtype=np.uint8)
    ).save(orphan_path)

    with pytest.raises(ValueError, match="Orphan depth files"):
        validate_manifest(manifest, data_root)


def test_validate_manifest_duplicate_frames(valid_manifest_data):
    data_root, manifest, _ = valid_manifest_data

    manifest = pd.concat(
        [manifest, manifest],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate frame keys"):
        validate_manifest(manifest, data_root)

