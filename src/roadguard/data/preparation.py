import json
import pandas as pd
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import numpy as np
from PIL import Image



def extract_anovox_archive(
    archive_path: str | Path,
    destination: str | Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Selectively extract required files from an ANOVOX TAR.GZ archive."""

    archive_path = Path(archive_path)
    destination = Path(destination)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    counts = {
        "rgb": 0,
        "depth": 0,
        "semantic": 0,
        "metadata": 0,
        "skipped": 0,
    }

    sensor_categories = {
        "RGB-CAM": "rgb",
        "DEPTH_CAM": "depth",
        "SEMANTIC-CAM": "semantic",
    }

    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            path = PurePosixPath(member.name)

            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
            ):
                raise ValueError(f"Unsafe archive path: {member.name}")

            parts = path.parts
            category = None

            if (
                len(parts) == 3
                and parts[1] == "Scenario_Configuration_Files"
                and path.suffix == ".json"
            ):
                category = "metadata"

            elif len(parts) == 4 and parts[1].startswith("Scenario_"):
                sensor_folder = parts[2]

                for prefix, name in sensor_categories.items():
                    if sensor_folder.startswith(prefix) and path.suffix.lower() == ".png":
                        category = name
                        break

            if category is None:
                counts["skipped"] += 1
                continue

            target = destination.joinpath(*parts)
            destination_root = destination.resolve()
            resolved_target = target.resolve()

            if not resolved_target.is_relative_to(destination_root):
                raise ValueError(
                    f"Archive target escapes destination: {member.name}"
                )

            if target.is_file() and not overwrite:
                if target.stat().st_size == member.size:
                    counts["skipped"] += 1
                    continue

            target.parent.mkdir(parents=True, exist_ok=True)

            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")

            temporary = target.with_name(target.name + ".partial")

            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)

                if temporary.stat().st_size != member.size:
                    raise IOError(f"Incomplete extraction: {member.name}")

                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

            counts[category] += 1

    return counts



def load_scenario_metadata(
    config_path: str | Path,
    source: str,
) -> pd.DataFrame:
    """Load scenario metadata from an ANOVOX configuration JSON."""
    source = source.lower()

    if source not in {"static", "normality"}:
        raise ValueError(f"Invalid source '{source}'. Expected 'static' or 'normality'.")

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Scenario configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    definition = config["scenario_definition"]
    town = definition["map"]

    records = []

    for scenario in definition["scenarios"]:
        anomaly_config = scenario.get("anomaly_config") or {}
        blueprint = anomaly_config.get("anomaly_bp_name")

        if source == "normality":
            family = "normality"
        elif blueprint:
            family = blueprint.rsplit("_", 1)[-1].lower()
        else:
            raise ValueError(
                f"Missing anomaly blueprint for scenario {scenario['id']}"
            )

        records.append({
            "scenario_id": scenario["id"],
            "town": town,
            "source": source,
            "anomaly_family": family,
            "anomaly_blueprint": blueprint,
        })

    metadata = pd.DataFrame(
        records,
        columns=[
            "scenario_id",
            "town",
            "source",
            "anomaly_family",
            "anomaly_blueprint",
        ],
    )

    if metadata["scenario_id"].duplicated().any():
        raise ValueError("Duplicate scenario IDs in configuration")

    return metadata


def generate_anomaly_mask(
    semantic_path: str | Path,
    mask_path: str | Path,
    anomaly_ids: tuple[int, ...] = (29, 30, 31, 32, 33, 34, 100),
) -> int:
    """
    Generate a binary anomaly mask from an ANOVOX semantic image.

    Returns the number of anomaly pixels.
    """
    semantic_path = Path(semantic_path)
    mask_path = Path(mask_path)

    with Image.open(semantic_path) as image:
        semantic = np.asarray(image)

    if semantic.ndim != 2:
        raise ValueError(
            f"Expected a 2D semantic label image, got {semantic.shape}"
        )

    mask = np.isin(semantic, anomaly_ids).astype(np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(mask_path)

    return int(mask.sum())



def inspect_semantic_labels(
    semantic_path: str | Path,
) -> dict[int, int]:
    """Return the pixel count for each semantic ID in an image."""

    semantic_path = Path(semantic_path)

    with Image.open(semantic_path) as image:
        semantic = np.asarray(image)

    if semantic.ndim != 2:
        raise ValueError(
            f"Expected a 2D semantic image, got {semantic.shape}"
        )

    labels, counts = np.unique(semantic, return_counts=True)

    return {
        int(label): int(count)
        for label, count in zip(labels, counts)
    }


def generate_anomaly_masks(
    data_root: str | Path,
    anomaly_ids: tuple[int, ...] = (29, 30, 31, 32, 33, 34, 100),
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generate binary masks for all extracted ANOVOX semantic images.

    Returns a DataFrame containing frame-level mask metadata.
    """
    data_root = Path(data_root)
    records = []
    seen_frames = set()

    semantic_files = sorted(
        path
        for path in data_root.rglob("*.png")
        if path.parent.name.startswith("SEMANTIC-CAM")
    )

    if not semantic_files:
        raise ValueError(f"No semantic images found under {data_root}")

    for semantic_path in semantic_files:
        scenario_dir = semantic_path.parent.parent

        frame_id = semantic_path.stem.rsplit("_", 1)[-1]

        frame_key = (scenario_dir.resolve(), frame_id)

        if frame_key in seen_frames:
            raise ValueError(
                f"Duplicate frame ID '{frame_id}' in scenario {scenario_dir}"
            )

        seen_frames.add(frame_key)

        mask_path = (
            scenario_dir
            / "MASK"
            / f"MASK_{frame_id}.png"
        )

        if mask_path.is_file() and not overwrite:
            with Image.open(mask_path) as image:
                mask = np.asarray(image)

            with Image.open(semantic_path) as image:
                semantic = np.asarray(image)

            if mask.ndim != 2 or mask.shape != semantic.shape:
                raise ValueError(f"Invalid mask dimensions: {mask_path}")

            if not np.isin(mask, [0, 255]).all():
                raise ValueError(f"Invalid mask values: {mask_path}")

            anomaly_pixels = int(np.count_nonzero(mask))
        else:
            anomaly_pixels = generate_anomaly_mask(
                semantic_path,
                mask_path,
                anomaly_ids=anomaly_ids,
            )

        records.append({
            "scenario_id": scenario_dir.name.removeprefix("Scenario_"),
            "frame_id": frame_id,
            "semantic_path": str(semantic_path.relative_to(data_root)),
            "mask_path": str(mask_path.relative_to(data_root)),
            "anomaly_pixels": anomaly_pixels,
        })

    return pd.DataFrame(records)