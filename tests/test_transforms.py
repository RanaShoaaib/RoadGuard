"""Tests for synchronized ANOVOX depth/RGB/mask transforms."""
import numpy as np
import pytest
import torch
from PIL import Image

from roadguard.data.transforms import build_transforms, decode_depth, IMAGENET_MEAN, IMAGENET_STD


def encode_depth(meters):
    encoded = np.rint(np.asarray(meters) / 1000 * 16777215).astype(np.uint32)
    return np.stack([encoded & 255, (encoded >> 8) & 255, (encoded >> 16) & 255], axis=-1).astype(np.uint8)


def sample():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :3, 0] = 255
    rgb[:, 3:, 1] = 255
    meters = np.full((4, 6), 20.0)
    meters[:, 3:] = 80.0
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, 3:] = 255
    return Image.fromarray(rgb), Image.fromarray(encode_depth(meters)), Image.fromarray(mask)


def test_depth_decoder_known_values():
    meters = np.array([[0, 10, 100, 1000]], dtype=np.float64)
    decoded = decode_depth(encode_depth(meters))
    np.testing.assert_allclose(decoded, meters, atol=0.001)
    assert decoded.dtype == np.float32


def test_depth_decoder_rejects_wrong_shape_or_dtype():
    with pytest.raises(ValueError, match="uint8 RGB"):
        decode_depth(np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8 RGB"):
        decode_depth(np.zeros((4, 6, 3), dtype=np.uint16))


def test_output_shapes_ranges_and_mask():
    image, depth, mask = build_transforms(image_size=(8, 12))(*sample())
    assert image.shape == (3, 8, 12)
    assert depth.shape == (1, 8, 12)
    assert mask.shape == (1, 8, 12)
    assert all(x.dtype == torch.float32 for x in (image, depth, mask))
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    assert 0 <= depth.min() and depth.max() <= 1
    assert torch.allclose(depth[:, :, :5], torch.full((1, 8, 5), 0.2), atol=0.001)
    assert torch.allclose(depth[:, :, 7:], torch.full((1, 8, 5), 0.8), atol=0.001)


def test_imagenet_normalization():
    image, _, _ = build_transforms(image_size=(4, 6))(*sample())
    expected = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert image[0, 0, 0].item() == pytest.approx(expected, abs=1e-5)


def test_forced_flip_is_synchronized():
    args = dict(image_size=(4, 6))
    normal = build_transforms(**args, train=False)(*sample())
    flipped = build_transforms(**args, train=True, flip_probability=1)(*sample())
    for a, b in zip(normal, flipped):
        assert torch.allclose(b, a.flip(-1))


def test_eval_is_deterministic():
    transform = build_transforms(train=False)
    first = transform(*sample())
    second = transform(*sample())
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_clip_depth_to_maximum():
    _, depth, _ = build_transforms(image_size=(4, 6), max_depth_m=50)(*sample())
    assert torch.allclose(depth[0, :, :3], torch.full((4, 3), 0.4), atol=0.001)
    assert torch.all(depth[0, :, 3:] == 1.0)


def test_invalid_configuration():
    with pytest.raises(ValueError, match="image_size"):
        build_transforms(image_size=(0, 256))
    with pytest.raises(ValueError, match="flip_probability"):
        build_transforms(flip_probability=1.5)
    with pytest.raises(ValueError, match="max_depth_m"):
        build_transforms(max_depth_m=0)


def test_noise_only_changes_depth_during_training():
    torch.manual_seed(3)
    clean = build_transforms(image_size=(4, 6), train=True, flip_probability=0)(*sample())
    torch.manual_seed(3)
    noisy = build_transforms(image_size=(4, 6), train=True, flip_probability=0,
                             depth_noise_std_m=2.0)(*sample())
    assert torch.equal(clean[0], noisy[0])
    assert torch.equal(clean[2], noisy[2])
    assert not torch.equal(clean[1], noisy[1])
    assert torch.all((noisy[1] >= 0) & (noisy[1] <= 1))

def test_dropout_masks_only_depth():
    original = build_transforms(image_size=(4, 6))(*sample())
    augmented = build_transforms(image_size=(4, 6), train=True, flip_probability=0,
                                 depth_dropout_probability=1, depth_dropout_fraction=0.5)(*sample())
    assert torch.equal(original[0], augmented[0])
    assert torch.equal(original[2], augmented[2])
    assert (augmented[1] == 0).sum() == 6
    assert (augmented[1] > 0).sum() == 18

def test_eval_ignores_depth_augmentation():
    plain = build_transforms(image_size=(4, 6), train=False)(*sample())
    augmented = build_transforms(image_size=(4, 6), train=False, depth_noise_std_m=10,
                                 depth_dropout_probability=1)(*sample())
    assert all(torch.equal(a, b) for a, b in zip(plain, augmented))

def test_depth_augmentation_is_reproducible_with_seed():
    transform = build_transforms(image_size=(4, 6), train=True, flip_probability=0,
                                 depth_noise_std_m=1, depth_dropout_probability=1)
    torch.manual_seed(17)
    first = transform(*sample())
    torch.manual_seed(17)
    second = transform(*sample())
    assert all(torch.equal(a, b) for a, b in zip(first, second))

@pytest.mark.parametrize("kwargs,match", [
    ({"depth_noise_std_m": -1}, "depth_noise_std_m"),
    ({"depth_dropout_probability": 1.1}, "depth_dropout_probability"),
    ({"depth_dropout_fraction": 0}, "depth_dropout_fraction"),
])
def test_invalid_depth_augmentation_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_transforms(**kwargs)
