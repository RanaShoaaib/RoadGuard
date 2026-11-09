import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from roadguard.data.dataset import RoadGuardDataset


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "extracted"
    root.mkdir()
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    depth = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    Image.fromarray(rgb).save(root / "rgb.png")
    Image.fromarray(depth).save(root / "depth.png")
    Image.fromarray(mask).save(root / "mask.png")
    manifest = pd.DataFrame([{
        "rgb_path": "rgb.png", "depth_path": "depth.png", "mask_path": "mask.png",
        "scenario_id": "scenario_01", "frame_id": "0001", "split": "train",
    }])
    return root, manifest


def test_default_loading(sample):
    root, manifest = sample
    ds = RoadGuardDataset(manifest, root)
    item = ds[0]
    assert len(ds) == 1
    assert item["image"].shape == (3, 4, 6)
    assert item["depth"].shape == (3, 4, 6)
    assert item["mask"].shape == (1, 4, 6)
    assert all(item[key].dtype == torch.float32 for key in ("image", "depth", "mask"))
    assert torch.all(item["image"][0] == 1)
    assert item["depth"][0, 2, 3].item() == 45.0  # raw encoded byte retained
    assert set(torch.unique(item["mask"]).tolist()) == {0.0, 1.0}
    assert item["mask"].sum().item() == 4
    assert item["frame_id"] == "0001"
    assert item["scenario_id"] == "scenario_01"


def test_paired_transform(sample):
    root, manifest = sample

    def flip_all(image, depth, mask):
        return tuple(x.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for x in (image, depth, mask))

    original = RoadGuardDataset(manifest, root)[0]
    flipped = RoadGuardDataset(manifest, root, transform=flip_all)[0]
    for key in ("image", "depth", "mask"):
        assert torch.equal(flipped[key], original[key].flip(-1))


def test_dataloader_collation(sample):
    root, manifest = sample
    batch = next(iter(DataLoader(RoadGuardDataset(manifest, root), batch_size=1)))
    assert batch["image"].shape == (1, 3, 4, 6)
    assert batch["depth"].shape == (1, 3, 4, 6)
    assert batch["mask"].shape == (1, 1, 4, 6)
    assert batch["scenario_id"] == ["scenario_01"]


def test_missing_columns(sample):
    root, manifest = sample
    with pytest.raises(ValueError, match="Missing manifest columns"):
        RoadGuardDataset(manifest.drop(columns="depth_path"), root)


def test_invalid_mask(sample):
    root, manifest = sample
    Image.fromarray(np.full((4, 6), 128, dtype=np.uint8)).save(root / "mask.png")
    with pytest.raises(ValueError, match="Invalid binary mask"):
        RoadGuardDataset(manifest, root)[0]


def test_mismatched_dimensions(sample):
    root, manifest = sample
    Image.fromarray(np.zeros((2, 6), dtype=np.uint8)).save(root / "depth.png")
    with pytest.raises(ValueError, match="size mismatch"):
        RoadGuardDataset(manifest, root)[0]


def test_invalid_depth_channels(sample):
    root, manifest = sample
    Image.fromarray(np.zeros((4, 6), dtype=np.uint8)).save(root / "depth.png")
    with pytest.raises(ValueError, match="uint8 RGB with 3 channels"):
        RoadGuardDataset(manifest, root)[0]


def test_depth_rgb_bytes_preserved(sample):
    root, manifest = sample
    original = np.asarray(Image.open(root / "depth.png"))
    actual = RoadGuardDataset(manifest, root)[0]["depth"].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, original.astype(np.float32))


def test_native_mask_is_original_resolution_and_optional(sample):
    from roadguard.data.transforms import build_transforms
    root, manifest = sample
    tf = build_transforms(image_size=(2, 3), train=False)
    default = RoadGuardDataset(manifest, root, transform=tf)[0]
    native = RoadGuardDataset(manifest, root, transform=tf, native_mask=True)[0]
    assert "mask_native" not in default
    assert native["mask"].shape == (1, 2, 3)              # resized, matches the model input
    assert native["mask_native"].shape == (1, 4, 6)       # untouched original
    assert native["mask_native"].sum() == 4 and native["mask_native"].dtype == torch.float32
