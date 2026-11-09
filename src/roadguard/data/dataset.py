"""Manifest-backed RGB, depth, and anomaly-mask dataset for RoadGuard."""
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RoadGuardDataset(Dataset):
    """Load aligned RGB, depth, and binary mask samples.

    An optional paired transform takes (image, depth, mask) PIL images and
    returns three PIL images, NumPy arrays, or tensors in the same order.
    RGB uint8 inputs become float32 in [0, 1]. Untransformed CARLA depth
    retains its three encoded uint8 channels (as float32); the paired transform
    decodes depth to a normalized single-channel float tensor. With ``native_mask=True`` each
    sample also carries ``mask_native`` (the untouched original-resolution mask, [1, H0, W0]).
    """

    REQUIRED_COLUMNS = ("rgb_path", "depth_path", "mask_path", "scenario_id", "frame_id")

    def __init__(self, manifest: pd.DataFrame, data_root: str | Path,
                 transform: Callable | None = None, native_mask: bool = False) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(manifest.columns)
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        if manifest.empty:
            raise ValueError("Manifest is empty")
        if manifest[list(self.REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError("Manifest contains missing required values")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform
        self.native_mask = native_mask

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        rgb_path = self.data_root / str(row["rgb_path"])
        depth_path = self.data_root / str(row["depth_path"])
        mask_path = self.data_root / str(row["mask_path"])

        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        with Image.open(depth_path) as source:
            depth = source.copy()  # Do not convert to L: preserve bit depth.
        with Image.open(mask_path) as source:
            mask = source.copy()

        if image.size != depth.size or image.size != mask.size:
            raise ValueError(f"Image/depth/mask size mismatch: {rgb_path}, {depth_path}, {mask_path}")
        original_depth = np.asarray(depth)
        if original_depth.ndim != 3 or original_depth.shape[2] != 3 or original_depth.dtype != np.uint8:
            raise ValueError(f"Encoded depth image must be uint8 RGB with 3 channels: {depth_path}")
        original_mask = np.asarray(mask)
        if original_mask.ndim != 2 or not np.isin(original_mask, (0, 255)).all():
            raise ValueError(f"Invalid binary mask (expected 2D 0/255): {mask_path}")

        if self.transform is not None:
            image, depth, mask = self.transform(image, depth, mask)

        image_tensor = self._image_tensor(image)
        depth_tensor = self._depth_tensor(depth)
        mask_tensor = self._mask_tensor(mask)
        if len({image_tensor.shape[-2:], depth_tensor.shape[-2:], mask_tensor.shape[-2:]}) != 1:
            raise ValueError("Transformed image, depth, and mask have different spatial sizes")
        sample = {
            "image": image_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "scenario_id": str(row["scenario_id"]),
            "frame_id": str(row["frame_id"]),
        }
        if self.native_mask:
            # Original-resolution mask, used to score predictions independently of the input size.
            sample["mask_native"] = torch.from_numpy(original_mask > 0).float().unsqueeze(0)
        return sample

    @staticmethod
    def _image_tensor(image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            result = image.float()
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("RGB image must have shape HWC with 3 channels")
            result = torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1).float()
            if array.dtype == np.uint8:
                result = result / 255.0
        if result.ndim != 3 or result.shape[0] != 3:
            raise ValueError("Image tensor must have shape [3, H, W]")
        return result.contiguous()

    @staticmethod
    def _depth_tensor(depth) -> torch.Tensor:
        """Accept encoded RGB depth or decoded single-channel depth.

        The paired transform owns decoding; this method never decodes RGB bytes.
        """
        if isinstance(depth, torch.Tensor):
            result = depth.float()
        else:
            array = np.asarray(depth)
            if array.ndim == 2:
                result = torch.from_numpy(np.array(array, copy=True, order="C")).unsqueeze(0).float()
            elif array.ndim == 3 and array.shape[2] in (1, 3):
                result = torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1).float()
            else:
                raise ValueError("Depth must have shape HW, HWC with 1 or 3 channels, or CHW tensor")
        if result.ndim != 3 or result.shape[0] not in (1, 3):
            raise ValueError("Depth tensor must have shape [1, H, W] or [3, H, W]")
        return result.contiguous()

    @staticmethod
    def _single_channel_tensor(value, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            result = value.float()
        else:
            result = torch.from_numpy(np.array(value, copy=True, order="C")).float()
        if result.ndim == 2:
            result = result.unsqueeze(0)
        if result.ndim != 3 or result.shape[0] != 1:
            raise ValueError(f"{name} tensor must have shape [1, H, W]")
        return result.contiguous()

    @classmethod
    def _mask_tensor(cls, mask) -> torch.Tensor:
        result = cls._single_channel_tensor(mask, "Mask")
        values = torch.unique(result)
        if torch.all((values == 0) | (values == 255)):
            result = result / 255.0
        elif not torch.all((values == 0) | (values == 1)):
            raise ValueError("Transformed mask must be binary: 0/1 or 0/255")
        return result.contiguous()
