"""Synchronized RGB, CARLA-encoded depth, and binary-mask transforms."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_depth(depth_rgb: np.ndarray) -> np.ndarray:
    """Decode ANOVOX/CARLA RGB depth bytes to meters (H, W), float32.

    Channel order was checked against sample ANOVOX depth geometry.
    Values near 1000 m typically represent sky / maximum sensor range.
    """
    array = np.asarray(depth_rgb)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Encoded depth must be a uint8 RGB array [H, W, 3]")
    d = array.astype(np.float32)
    return (d[..., 0] + 256.0 * d[..., 1] + 65536.0 * d[..., 2]) * (1000.0 / 16777215.0)


def _resize(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, **kwargs).squeeze(0)


@dataclass(frozen=True)
class PairedTransforms:
    image_size: tuple[int, int] = (256, 256)  # (height, width)
    train: bool = False
    flip_probability: float = 0.5
    max_depth_m: float = 100.0
    depth_noise_std_m: float = 0.0
    depth_dropout_probability: float = 0.0
    depth_dropout_fraction: float = 0.2
    # Training-only appearance / scale augmentation (all off by default).
    brightness: float = 0.0       # RGB multiplied by U(1 - b, 1 + b)
    contrast: float = 0.0         # RGB contrast scaled by U(1 - c, 1 + c)
    saturation: float = 0.0       # RGB saturation scaled by U(1 - s, 1 + s)
    crop_min_scale: float = 1.0   # crop covers U(crop_min_scale, 1) of the frame area; 1 disables
    augment_probability: float = 1.0   # share of training images that get the jitter and crop (together)

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.image_size):
            raise ValueError("image_size must be (positive height, positive width)")
        if not (0.0 <= self.flip_probability <= 1.0):
            raise ValueError("flip_probability must be in [0, 1]")
        if not np.isfinite(self.max_depth_m) or self.max_depth_m <= 0:
            raise ValueError("max_depth_m must be finite and positive")
        if not np.isfinite(self.depth_noise_std_m) or self.depth_noise_std_m < 0:
            raise ValueError("depth_noise_std_m must be finite and nonnegative")
        if not np.isfinite(self.depth_dropout_probability) or not 0 <= self.depth_dropout_probability <= 1:
            raise ValueError("depth_dropout_probability must be in [0, 1]")
        if not np.isfinite(self.depth_dropout_fraction) or not 0 < self.depth_dropout_fraction <= 1:
            raise ValueError("depth_dropout_fraction must be in (0, 1]")
        for name in ("brightness", "contrast", "saturation"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not np.isfinite(self.crop_min_scale) or not 0 < self.crop_min_scale <= 1:
            raise ValueError("crop_min_scale must be in (0, 1]")
        if not np.isfinite(self.augment_probability) or not 0 <= self.augment_probability <= 1:
            raise ValueError("augment_probability must be in [0, 1]")

    @staticmethod
    def _factor(amount: float) -> float:
        return 1.0 + (torch.rand(()).item() * 2 - 1) * amount

    def _color_jitter(self, rgb: torch.Tensor) -> torch.Tensor:
        """Mild brightness/contrast/saturation change on an RGB tensor in [0, 1]."""
        if self.brightness > 0:
            rgb = rgb * self._factor(self.brightness)
        if self.contrast > 0:
            mean = rgb.mean()
            rgb = (rgb - mean) * self._factor(self.contrast) + mean
        if self.saturation > 0:
            gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0)
            rgb = gray + (rgb - gray) * self._factor(self.saturation)
        return rgb.clamp(0.0, 1.0)

    def _random_crop(self, rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor):
        """Crop the same random window from all three tensors; the window always contains the
        whole anomaly (the frame is left uncropped if the anomaly is larger than the window)."""
        if self.crop_min_scale >= 1.0:
            return rgb, depth, mask
        height, width = mask.shape[-2:]
        scale = self.crop_min_scale + (1.0 - self.crop_min_scale) * torch.rand(()).item()
        crop_h, crop_w = max(1, round(height * scale ** 0.5)), max(1, round(width * scale ** 0.5))
        top_lo, top_hi, left_lo, left_hi = 0, height - crop_h, 0, width - crop_w
        rows, cols = torch.nonzero(mask[0] > 0.5, as_tuple=True)
        if rows.numel() > 0:
            y0, y1, x0, x1 = rows.min().item(), rows.max().item(), cols.min().item(), cols.max().item()
            if y1 - y0 + 1 > crop_h or x1 - x0 + 1 > crop_w:
                return rgb, depth, mask
            top_lo, top_hi = max(0, y1 + 1 - crop_h), min(y0, height - crop_h)
            left_lo, left_hi = max(0, x1 + 1 - crop_w), min(x0, width - crop_w)
        top = top_lo + int(torch.randint(top_hi - top_lo + 1, (1,)).item())
        left = left_lo + int(torch.randint(left_hi - left_lo + 1, (1,)).item())
        window = (slice(None), slice(top, top + crop_h), slice(left, left + crop_w))
        return rgb[window], depth[window], mask[window]

    def __call__(self, image: Image.Image, depth: Image.Image, mask: Image.Image):
        rgb = np.asarray(image)
        raw_depth = np.asarray(depth)
        raw_mask = np.asarray(mask)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB image must be uint8 [H, W, 3]")
        if raw_mask.ndim != 2 or not np.isin(raw_mask, (0, 255)).all():
            raise ValueError("Mask must be binary uint8 with values 0 and 255")
        if rgb.shape[:2] != raw_depth.shape[:2] or rgb.shape[:2] != raw_mask.shape:
            raise ValueError("RGB, depth, and mask must have matching spatial dimensions")

        rgb_t = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255.0
        depth_m = decode_depth(raw_depth)
        depth_t = torch.from_numpy(np.array(depth_m, copy=True)).unsqueeze(0)
        # Perturb finite, in-range depth only; do not turn sky/range-saturated
        # values into spurious near-camera measurements.
        if self.train and self.depth_noise_std_m > 0:
            valid = (depth_t > 0) & (depth_t < self.max_depth_m)
            noisy = depth_t + torch.randn_like(depth_t) * self.depth_noise_std_m
            depth_t = torch.where(valid, noisy, depth_t)
        depth_t = depth_t.clamp(0.0, self.max_depth_m) / self.max_depth_m
        mask_t = torch.from_numpy(np.array(raw_mask, copy=True)).unsqueeze(0).float() / 255.0

        if self.train and (self.augment_probability >= 1.0 or torch.rand(()).item() < self.augment_probability):
            rgb_t = self._color_jitter(rgb_t)
            rgb_t, depth_t, mask_t = self._random_crop(rgb_t, depth_t, mask_t)

        rgb_t = _resize(rgb_t, self.image_size, "bilinear")
        depth_t = _resize(depth_t, self.image_size, "bilinear")
        mask_t = _resize(mask_t, self.image_size, "nearest")

        if self.train and torch.rand(()) < self.flip_probability:
            rgb_t = rgb_t.flip(-1)
            depth_t = depth_t.flip(-1)
            mask_t = mask_t.flip(-1)

        # Spatial depth dropout: one randomly positioned rectangular region.
        if self.train and self.depth_dropout_probability > 0:
            if torch.rand(()).item() < self.depth_dropout_probability:
                h, w = depth_t.shape[-2:]
                dh = max(1, round(h * self.depth_dropout_fraction))
                dw = max(1, round(w * self.depth_dropout_fraction))
                top = int(torch.randint(h - dh + 1, (1,)).item())
                left = int(torch.randint(w - dw + 1, (1,)).item())
                depth_t[:, top:top + dh, left:left + dw] = 0.0

        mean = rgb_t.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = rgb_t.new_tensor(IMAGENET_STD).view(3, 1, 1)
        rgb_t = (rgb_t - mean) / std
        return rgb_t.contiguous(), depth_t.contiguous(), mask_t.contiguous()


def build_transforms(
    image_size: tuple[int, int] = (256, 256),
    train: bool = False,
    flip_probability: float = 0.5,
    max_depth_m: float = 100.0,
    depth_noise_std_m: float = 0.0,
    depth_dropout_probability: float = 0.0,
    depth_dropout_fraction: float = 0.2,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    crop_min_scale: float = 1.0,
    augment_probability: float = 1.0,
) -> PairedTransforms:
    """Build train or deterministic evaluation transforms; size is (H, W)."""
    return PairedTransforms(
        image_size=image_size, train=train, flip_probability=flip_probability,
        max_depth_m=max_depth_m, depth_noise_std_m=depth_noise_std_m,
        depth_dropout_probability=depth_dropout_probability,
        depth_dropout_fraction=depth_dropout_fraction, brightness=brightness,
        contrast=contrast, saturation=saturation, crop_min_scale=crop_min_scale,
        augment_probability=augment_probability,
    )
