from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


def _archive_dirs(data_root: Path) -> list[Path]:
    """Directories under ``data_root`` that hold ANOVOX scenarios (one per prepared archive)."""
    return [
        path for path in sorted(data_root.iterdir())
        if path.is_dir() and any(path.glob("Scenario_*"))
    ]


def build_master_manifest(
    data_root: str | Path,
    scenario_metadata: pd.DataFrame,
    mask_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build a frame-level manifest for the ANOVOX archive directories under ``data_root``."""

    data_root = Path(data_root)
    records = []
    seen_frames = set()

    for dataset_dir in _archive_dirs(data_root):
        for scenario_dir in sorted(dataset_dir.glob("Scenario_*")):
            if not scenario_dir.is_dir():
                continue

            scenario_id = scenario_dir.name.removeprefix("Scenario_")

            for rgb_path in sorted(scenario_dir.glob("RGB-CAM*/*.png")):
                frame_id = rgb_path.stem.rsplit("_", 1)[-1]

                frame_key = (scenario_id, frame_id)

                if frame_key in seen_frames:
                    raise ValueError(
                        f"Duplicate RGB frame: {scenario_id}/{frame_id}"
                    )

                seen_frames.add(frame_key)

                depth_files = list(
                    scenario_dir.glob(f"DEPTH_CAM*/*_{frame_id}.png")
                )

                if len(depth_files) != 1:
                    raise ValueError(
                        f"Expected one depth image for "
                        f"{scenario_id}/{frame_id}, "
                        f"found {len(depth_files)}"
                    )

                records.append({
                    "scenario_id": scenario_id,
                    "frame_id": frame_id,
                    "rgb_path": str(rgb_path.relative_to(data_root)),
                    "depth_path": str(depth_files[0].relative_to(data_root)),
                })

    manifest = pd.DataFrame(
        records,
        columns=[
            "scenario_id",
            "frame_id",
            "rgb_path",
            "depth_path",
        ],
    )

    manifest = manifest.merge(
        mask_metadata[
            [
                "scenario_id",
                "frame_id",
                "semantic_path",
                "mask_path",
                "anomaly_pixels",
            ]
        ],
        on=["scenario_id", "frame_id"],
        how="left",
        validate="one_to_one",
    )

    if manifest["semantic_path"].isna().any():
        raise ValueError("Missing semantic images or anomaly masks")

    # Integrate scenario-level metadata.
    manifest = manifest.merge(
        scenario_metadata[
            [
                "scenario_id",
                "town",
                "anomaly_family",
                "anomaly_blueprint",
                "anomaly_object",
            ]
        ],
        on=["scenario_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if (manifest["_merge"] != "both").any():
        raise ValueError("Missing scenario metadata")

    manifest = manifest.drop(columns="_merge")

    manifest["has_anomaly"] = manifest["anomaly_pixels"] > 0

    return manifest



def validate_manifest(
    manifest: pd.DataFrame,
    data_root: str | Path,
) -> None:
    """Validate ANOVOX frame completeness, alignment, and mask integrity."""

    data_root = Path(data_root)

    required_columns = [
        "scenario_id", "frame_id",
        "rgb_path", "depth_path", "semantic_path",
        "mask_path", "anomaly_pixels",
    ]

    missing_columns = set(required_columns) - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")

    if manifest.empty:
        raise ValueError("Manifest is empty")

    frame_keys = ["scenario_id", "frame_id"]

    if manifest.duplicated(frame_keys).any():
        raise ValueError("Duplicate frame keys in manifest")

    # Validate each frame.
    for row in manifest.itertuples(index=False):
        frame_key = f"{row.scenario_id}/{row.frame_id}"
        images = {}

        for modality in ("rgb", "depth", "semantic", "mask"):
            relative_path = getattr(row, f"{modality}_path")
            path = data_root / relative_path

            if not path.is_file():
                raise ValueError(
                    f"Missing {modality} file for {frame_key}: {path}"
                )

            with Image.open(path) as image:
                images[modality] = np.asarray(image)

        # Compare spatial dimensions (height, width).
        dimensions = {
            modality: image.shape[:2]
            for modality, image in images.items()
        }

        if len(set(dimensions.values())) != 1:
            raise ValueError(
                f"Image dimension mismatch for {frame_key}: {dimensions}"
            )

        mask = images["mask"]

        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D for {frame_key}")

        if not np.isin(mask, [0, 255]).all():
            raise ValueError(f"Invalid mask values for {frame_key}")

        anomaly_pixels = int(np.count_nonzero(mask))

        if anomaly_pixels != row.anomaly_pixels:
            raise ValueError(
                f"Anomaly pixel count mismatch for {frame_key}: "
                f"manifest={row.anomaly_pixels}, actual={anomaly_pixels}"
            )

    # Check for files not represented in the manifest.
    modality_patterns = {
        "rgb": "RGB-CAM*/*.png",
        "depth": "DEPTH_CAM*/*.png",
        "semantic": "SEMANTIC-CAM*/*.png",
        "mask": "MASK/*.png",
    }

    for modality, pattern in modality_patterns.items():
        expected_paths = set(manifest[f"{modality}_path"])

        actual_paths = {
            str(path.relative_to(data_root))
            for dataset_dir in _archive_dirs(data_root)
            for scenario_dir in dataset_dir.glob("Scenario_*")
            if scenario_dir.is_dir()
            for path in scenario_dir.glob(pattern)
        }

        orphan_paths = actual_paths - expected_paths

        if orphan_paths:
            raise ValueError(
                f"Orphan {modality} files found: "
                f"{sorted(orphan_paths)[:5]}"
            )
