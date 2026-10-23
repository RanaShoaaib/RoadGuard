"""Regression tests for encoded CARLA depth and decoded depth integration."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from roadguard.data.dataset import RoadGuardDataset
from roadguard.data.transforms import build_transforms


@pytest.fixture
def sample(tmp_path):
    h, w = 20, 24
    Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8)).save(tmp_path / 'rgb.png')
    encoded = np.zeros((h, w, 3), dtype=np.uint8)
    encoded[..., 0] = 10
    encoded[..., 1] = 20
    encoded[..., 2] = 30
    Image.fromarray(encoded, mode='RGB').save(tmp_path / 'depth.png')
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:8, 4:9] = 255
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    manifest = pd.DataFrame([dict(rgb_path='rgb.png', depth_path='depth.png',
                                  mask_path='mask.png', scenario_id='s1', frame_id='f1')])
    return tmp_path, manifest


def test_untransformed_encoded_depth_is_three_channels(sample):
    root, manifest = sample
    item = RoadGuardDataset(manifest, root)[0]
    assert item['depth'].shape == (3, 20, 24)
    assert torch.equal(item['depth'][:, 0, 0], torch.tensor([10., 20., 30.]))


def test_decoded_depth_is_one_channel_and_normalized(sample):
    root, manifest = sample
    transform = build_transforms(image_size=(32, 32), train=False)
    item = RoadGuardDataset(manifest, root, transform=transform)[0]
    assert item['image'].shape == (3, 32, 32)
    assert item['depth'].shape == (1, 32, 32)
    assert item['mask'].shape == (1, 32, 32)
    assert item['depth'].dtype == torch.float32
    assert torch.isfinite(item['depth']).all()
    assert (item['depth'] >= 0).all() and (item['depth'] <= 1).all()


def test_teacher_depth_augmentation_keeps_single_channel(sample):
    root, manifest = sample
    transform = build_transforms(image_size=(32, 32), train=True,
                                 depth_noise_std_m=1.0,
                                 depth_dropout_probability=1.0,
                                 depth_dropout_fraction=0.25)
    item = RoadGuardDataset(manifest, root, transform=transform)[0]
    assert item['depth'].shape == (1, 32, 32)
    assert torch.isfinite(item['depth']).all()


def test_invalid_raw_depth_rejected_before_transform(sample):
    root, manifest = sample
    Image.fromarray(np.zeros((20, 24), dtype=np.uint8)).save(root / 'depth.png')
    with pytest.raises(ValueError, match='Encoded depth image must be uint8 RGB'):
        RoadGuardDataset(manifest, root)[0]
