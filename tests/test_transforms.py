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


def _frame(h=64, w=96, box=(20, 24, 4, 4)):
    rgb = Image.fromarray(np.random.default_rng(0).integers(60, 200, (h, w, 3), dtype=np.uint8))
    depth = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    mask = np.zeros((h, w), dtype=np.uint8)
    y, x, bh, bw = box
    mask[y:y + bh, x:x + bw] = 255
    return rgb, depth, Image.fromarray(mask)


def test_crop_always_keeps_the_whole_anomaly_and_is_synchronized():
    tf = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0, crop_min_scale=0.4)
    grew = 0
    for seed in range(60):
        torch.manual_seed(seed)
        _, depth, mask = tf(*_frame())
        assert mask.sum() >= 16 and depth.shape == mask.shape       # anomaly never lost or cut
        grew += int(mask.sum() > 16)                                # crop zoomed in on it
    assert grew > 30


def test_crop_skipped_when_anomaly_larger_than_window():
    tf = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0, crop_min_scale=0.25)
    for seed in range(10):
        torch.manual_seed(seed)
        _, _, mask = tf(*_frame(box=(0, 0, 64, 96)))
        assert mask.sum() == 64 * 96


def test_color_jitter_changes_rgb_only_and_stays_valid():
    plain = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0)
    jitter = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0,
                              brightness=0.3, contrast=0.3, saturation=0.3)
    torch.manual_seed(0)
    rgb_a, depth_a, mask_a = plain(*_frame())
    torch.manual_seed(0)
    rgb_b, depth_b, mask_b = jitter(*_frame())
    assert not torch.allclose(rgb_a, rgb_b)
    assert torch.equal(depth_a, depth_b) and torch.equal(mask_a, mask_b)
    assert torch.isfinite(rgb_b).all() and rgb_b.min() >= -2.2 and rgb_b.max() <= 2.7   # clamped to [0, 1] pre-normalisation

def test_eval_ignores_appearance_and_crop_augmentation():
    plain = build_transforms(image_size=(64, 96), train=False)
    strong = build_transforms(image_size=(64, 96), train=False, brightness=0.5, contrast=0.5,
                              saturation=0.5, crop_min_scale=0.3)
    for a, b in zip(plain(*_frame()), strong(*_frame())):
        assert torch.equal(a, b)


def test_augmentation_is_reproducible_with_seed():
    tf = build_transforms(image_size=(64, 96), train=True, brightness=0.3, crop_min_scale=0.5)
    torch.manual_seed(5); first = tf(*_frame())
    torch.manual_seed(5); second = tf(*_frame())
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("kwargs", [{"brightness": 1.0}, {"contrast": -0.1}, {"saturation": float("nan")},
                                    {"crop_min_scale": 0.0}, {"crop_min_scale": 1.2}])
def test_invalid_appearance_configuration(kwargs):
    with pytest.raises(ValueError):
        build_transforms(train=True, **kwargs)


def test_augment_probability_controls_share_of_images_augmented():
    plain = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0)
    tf = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0, brightness=0.3,
                          crop_min_scale=0.5, augment_probability=0.5)
    torch.manual_seed(0)
    reference = plain(*_frame())[0]
    changed = 0
    for seed in range(200):
        torch.manual_seed(seed)
        changed += int(not torch.equal(tf(*_frame())[0], reference))
    assert 70 < changed < 130                                  # about half of the images are augmented


@pytest.mark.parametrize("probability,expect_changed", [(0.0, False), (1.0, True)])
def test_augment_probability_extremes(probability, expect_changed):
    plain = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0)
    tf = build_transforms(image_size=(64, 96), train=True, flip_probability=0.0, brightness=0.3,
                          crop_min_scale=0.5, augment_probability=probability)
    reference = plain(*_frame())[0]
    outputs = []
    for seed in range(20):
        torch.manual_seed(seed)
        outputs.append(not torch.equal(tf(*_frame())[0], reference))
    assert all(outputs) if expect_changed else not any(outputs)


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan")])
def test_invalid_augment_probability(value):
    with pytest.raises(ValueError, match="augment_probability"):
        build_transforms(train=True, augment_probability=value)
