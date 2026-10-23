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
) -> PairedTransforms:
    """Build train or deterministic evaluation transforms; size is (H, W)."""
    return PairedTransforms(
        image_size=image_size, train=train, flip_probability=flip_probability,
        max_depth_m=max_depth_m, depth_noise_std_m=depth_noise_std_m,
        depth_dropout_probability=depth_dropout_probability,
        depth_dropout_fraction=depth_dropout_fraction,
    )
