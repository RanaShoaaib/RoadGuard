"""ANOVOX data acquisition: stream archives from Zenodo, derive anomaly masks, build per-archive packages.

Nothing raw is ever stored. Each configured archive is read as one sequential gzip stream, only the RGB,
depth and semantic images plus the scenario configuration are kept, anomaly masks are written as frames
arrive, and the result is packaged as one uncompressed TAR on Drive (``towns/<name>.tar``) next to a
manifest CSV and a JSON sidecar. Rerunning skips archives that are already prepared with the same source
and settings.

Archive layout expected (ANOVOX Static Mono):
    <root>/Scenario_<id>/<RGB-CAM*|DEPTH_CAM*|SEMANTIC-CAM*>/<name>_<frame>.png
    <root>/Scenario_Configuration_Files/<name>.json

Note: the configuration JSON sits at the END of each archive and gzip cannot be read backwards, so
anomaly families are only known once the whole archive has been streamed.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from roadguard.data.manifest import build_master_manifest, validate_manifest

ANOMALY_IDS = (29, 30, 31, 32, 33, 34, 100)
SENSOR_PREFIXES = {"RGB-CAM": "rgb", "DEPTH_CAM": "depth", "SEMANTIC-CAM": "semantic"}


class ArchiveIntegrityError(RuntimeError):
    """The streamed archive does not match the checksum published by Zenodo."""


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchiveSpec:
    """One archive to prepare. Give either a Zenodo record id or a direct URL."""

    name: str
    zenodo: str | None = None
    url: str | None = None
    md5: str | None = None  # optional integrity check when ``url`` is used

    def signature(self) -> dict:
        return {"zenodo": self.zenodo, "url": self.url}


def load_data_config(path: str | Path) -> dict:
    """Read ``configs/data_sources.yaml`` and validate it.

    Returns {"archives": [ArchiveSpec, ...], "anomaly_ids": tuple[int, ...], "splits": dict}.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    archives = []
    for entry in raw.get("archives") or []:
        spec = ArchiveSpec(
            name=str(entry["name"]),
            zenodo=str(entry["zenodo"]) if entry.get("zenodo") is not None else None,
            url=entry.get("url"),
            md5=entry.get("md5"),
        )
        if (spec.zenodo is None) == (spec.url is None):
            raise ValueError(f"Archive {spec.name!r}: give exactly one of 'zenodo' or 'url'")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec.name):
            raise ValueError(f"Invalid archive name: {spec.name!r}")
        archives.append(spec)

    if not archives:
        raise ValueError("No archives configured")
    names = [a.name for a in archives]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive names in configuration")

    splits = {"seed": 42, "val_frac": 0.2, "min_family_scenarios": 8}
    splits.update(raw.get("splits") or {})
    return {
        "archives": archives,
        "anomaly_ids": tuple(int(i) for i in raw.get("anomaly_ids", ANOMALY_IDS)),
        "splits": splits,
    }


# ----------------------------------------------------------------------------------------------
# Zenodo / HTTP
# ----------------------------------------------------------------------------------------------
def resolve_zenodo(record_id: str | int, timeout: float = 60.0) -> dict:
    """Look up the single .tar.gz of a Zenodo record: name, size (bytes), md5 and a download URL."""
    import requests

    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=timeout)
    response.raise_for_status()
    files = response.json().get("files", [])
    tars = [f for f in files if str(f.get("key", "")).endswith((".tar.gz", ".tgz"))]
    if len(tars) != 1:
        raise ValueError(f"Record {record_id}: expected exactly one .tar.gz, found {[f.get('key') for f in files]}")
    f = tars[0]
    url = (f.get("links") or {}).get("self") or f"https://zenodo.org/records/{record_id}/files/{f['key']}?download=1"
    return {
        "name": f["key"],
        "size": int(f.get("size", 0)),
        "md5": str(f.get("checksum", "")).replace("md5:", ""),
        "url": url,
    }


def resolve_archive(spec: ArchiveSpec) -> dict:
    """Resolve an ArchiveSpec to {name, size, md5, url}."""
    if spec.zenodo is not None:
        return resolve_zenodo(spec.zenodo)
    return {"name": spec.url.split("?")[0].rsplit("/", 1)[-1], "size": 0, "md5": spec.md5 or "", "url": spec.url}


class _CountingReader(io.RawIOBase):
    """Counts compressed bytes, hashes them (md5) and prints throttled progress."""

    def __init__(self, raw, total: int | None = None, label: str = "", log_every_s: float = 60.0):
        self._raw, self.total, self.label, self.bytes = raw, total, label, 0
        self.md5 = hashlib.md5()
        self._t0 = self._last = time.time()
        self._log_every = log_every_s

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self._raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.md5.update(data)
        self.bytes += n
        now = time.time()
        if self._log_every and now - self._last >= self._log_every:
            self._last = now
            mb = self.bytes / 1e6
            rate = mb / max(now - self._t0, 1e-9)
            total = f" / {self.total / 1e6:,.0f}" if self.total else ""
            print(f"   [{self.label}] {mb:,.0f}{total} MB read ({rate:.1f} MB/s)", flush=True)
        return n

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def open_url_stream(url: str, total: int | None = None, label: str = "", log_every_s: float = 60.0):
    """Open ``url`` as a sequential byte stream of the raw (still gzip-compressed) bytes."""
    import requests

    response = requests.get(url, stream=True, timeout=(30, 300))
    response.raise_for_status()
    response.raw.decode_content = False
    reader = _CountingReader(
        response.raw,
        total or int(response.headers.get("Content-Length") or 0) or None,
        label,
        log_every_s,
    )
    return io.BufferedReader(reader, buffer_size=1 << 20), reader, response


def _safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path.parts


def _transient_errors() -> tuple[type[BaseException], ...]:
    errors: tuple[type[BaseException], ...] = (OSError, EOFError, zlib.error, tarfile.ReadError)
    try:
        import requests
        import urllib3

        errors += (requests.exceptions.RequestException, urllib3.exceptions.HTTPError)
    except ImportError:  # pragma: no cover
        pass
    return errors


# ----------------------------------------------------------------------------------------------
# Scenario metadata and masks
# ----------------------------------------------------------------------------------------------
def anomaly_object(blueprint: str, family: str) -> str:
    """'static.prop.o_naga_special' -> 'naga': the object identity, comparable across towns."""
    name = blueprint.split(".")[-1]
    if name.startswith("o_"):
        name = name[2:]
    if name.lower().endswith("_" + family):
        name = name[: -(len(family) + 1)]
    return name.lower()


def load_scenario_metadata(config_path: str | Path) -> pd.DataFrame:
    """Load scenario metadata from an ANOVOX Static configuration JSON."""
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Scenario configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    definition = config["scenario_definition"]
    town = definition["map"]

    records = []

    for scenario in definition["scenarios"]:
        anomaly_config = scenario.get("anomaly_config") or {}
        blueprint = anomaly_config.get("anomaly_bp_name")

        if not blueprint:
            raise ValueError(f"Missing anomaly blueprint for scenario {scenario['id']}")

        family = blueprint.rsplit("_", 1)[-1].lower()

        records.append({
            "scenario_id": scenario["id"],
            "town": town,
            "anomaly_family": family,
            "anomaly_blueprint": blueprint,
            "anomaly_object": anomaly_object(blueprint, family),
        })

    metadata = pd.DataFrame(
        records,
        columns=["scenario_id", "town", "anomaly_family", "anomaly_blueprint", "anomaly_object"],
    )

    if metadata["scenario_id"].duplicated().any():
        raise ValueError("Duplicate scenario IDs in configuration")

    return metadata


def generate_anomaly_mask(
    semantic_path: str | Path,
    mask_path: str | Path,
    anomaly_ids: tuple[int, ...] = ANOMALY_IDS,
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
        raise ValueError(f"Expected a 2D semantic label image, got {semantic.shape}")

    mask = np.isin(semantic, anomaly_ids).astype(np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(mask_path)

    return int(mask.sum())


def inspect_semantic_labels(semantic_path: str | Path) -> dict[int, int]:
    """Return the pixel count for each semantic ID in an image."""
    semantic_path = Path(semantic_path)

    with Image.open(semantic_path) as image:
        semantic = np.asarray(image)

    if semantic.ndim != 2:
        raise ValueError(f"Expected a 2D semantic image, got {semantic.shape}")

    labels, counts = np.unique(semantic, return_counts=True)

    return {int(label): int(count) for label, count in zip(labels, counts)}


# ----------------------------------------------------------------------------------------------
# Streaming extraction
# ----------------------------------------------------------------------------------------------
def stream_archive(
    url: str,
    destination: str | Path,
    anomaly_ids: tuple[int, ...] = ANOMALY_IDS,
    total: int | None = None,
    expected_md5: str | None = None,
    label: str = "stream",
    max_retries: int = 3,
    log_every_s: float = 60.0,
    min_free_gb: float = 3.0,
) -> dict:
    """Stream one ``.tar.gz`` and keep RGB, depth, semantic, configuration and generated masks.

    Anomaly masks are written next to the semantic images (``<scenario>/MASK/MASK_<frame>.png``) as
    each semantic image arrives. A dropped connection restarts the stream; files already written with
    the right size are not rewritten. Returns a dict with ``mask_metadata`` (the frame table that
    ``build_master_manifest`` expects), ``archive_root``, ``config_names``, ``counts``,
    ``compressed_mb``, ``seconds``, ``retries`` and ``md5_checked``.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < min_free_gb * 1024 ** 3:
        raise OSError(f"Less than {min_free_gb} GB free in {destination}")
    destination_root = destination.resolve()
    anomaly_ids = tuple(anomaly_ids)

    frames: dict[tuple[str, str], dict] = {}
    state = {"root": None, "config_names": [], "compressed_mb": 0.0, "md5_checked": False}
    counts = {"rgb": 0, "depth": 0, "semantic": 0, "metadata": 0, "skipped": 0}
    transient = _transient_errors()
    t0, retries = time.time(), 0

    for attempt in range(max_retries + 1):
        buffered = reader = response = None
        counts = dict.fromkeys(counts, 0)
        try:
            buffered, reader, response = open_url_stream(
                url, total, label if not attempt else f"{label}#{attempt}", log_every_s)
            with tarfile.open(fileobj=buffered, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = _safe_parts(member.name)
                    state["root"] = state["root"] or parts[0]

                    category = None
                    if (len(parts) == 3 and parts[1] == "Scenario_Configuration_Files"
                            and parts[2].endswith(".json")):
                        category = "metadata"
                    elif len(parts) == 4 and parts[1].startswith("Scenario_") and parts[3].lower().endswith(".png"):
                        category = next((c for p, c in SENSOR_PREFIXES.items() if parts[2].startswith(p)), None)
                    if category is None:
                        counts["skipped"] += 1
                        continue

                    target = destination.joinpath(*parts)
                    if not target.resolve().is_relative_to(destination_root):
                        raise ValueError(f"Archive target escapes destination: {member.name}")

                    if not (target.is_file() and target.stat().st_size == member.size):
                        source = tar.extractfile(member)
                        if source is None:
                            raise ValueError(f"Cannot read archive member: {member.name}")
                        target.parent.mkdir(parents=True, exist_ok=True)
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
                    if category == "metadata" and parts[2] not in state["config_names"]:
                        state["config_names"].append(parts[2])
                    if category == "semantic":
                        frame_id = Path(parts[3]).stem.rsplit("_", 1)[-1]
                        scenario_dir = target.parent.parent
                        mask_path = scenario_dir / "MASK" / f"MASK_{frame_id}.png"
                        frames[(parts[1], frame_id)] = {
                            "scenario_id": parts[1].removeprefix("Scenario_"),
                            "frame_id": frame_id,
                            "semantic_path": str(target.relative_to(destination)),
                            "mask_path": str(mask_path.relative_to(destination)),
                            "anomaly_pixels": generate_anomaly_mask(target, mask_path, anomaly_ids),
                        }
            while buffered.read(1 << 20):   # tar stops at its end marker: hash the remaining gzip bytes too
                pass
            state["compressed_mb"] = round(reader.bytes / 1e6, 1)
            if expected_md5:
                if reader.md5.hexdigest() != expected_md5.lower():
                    raise ArchiveIntegrityError(
                        f"{label}: md5 {reader.md5.hexdigest()} != published {expected_md5}")
                state["md5_checked"] = True
            break
        except ArchiveIntegrityError:
            raise
        except transient as exc:
            if attempt >= max_retries:
                raise
            retries += 1
            print(f"   stream interrupted ({type(exc).__name__}: {exc}); retry {retries}/{max_retries}", flush=True)
            time.sleep(5 * (attempt + 1))
        finally:
            for obj in (buffered, response):
                try:
                    if obj is not None:
                        obj.close()
                except Exception:  # noqa: BLE001
                    pass

    return {
        "mask_metadata": pd.DataFrame(list(frames.values()),
                                      columns=["scenario_id", "frame_id", "semantic_path", "mask_path",
                                               "anomaly_pixels"]),
        "archive_root": state["root"],
        "config_names": state["config_names"],
        "counts": counts,
        "compressed_mb": state["compressed_mb"],
        "seconds": round(time.time() - t0, 1),
        "retries": retries,
        "md5_checked": state["md5_checked"],
    }


# ----------------------------------------------------------------------------------------------
# One archive -> one package on Drive
# ----------------------------------------------------------------------------------------------
def _package_paths(output_dir: Path, name: str) -> dict[str, Path]:
    towns = output_dir / "towns"
    return {"tar": towns / f"{name}.tar", "manifest": towns / f"{name}_manifest.csv",
            "sidecar": towns / f"{name}.json"}


def prepare_archive(
    spec: ArchiveSpec,
    output_dir: str | Path,
    stage_dir: str | Path,
    anomaly_ids: tuple[int, ...] = ANOMALY_IDS,
    resolver=resolve_archive,
    keep_stage: bool = False,
    min_free_gb: float = 3.0,
) -> dict:
    """Stream one archive, validate it locally, and write ``towns/<name>.tar`` + manifest + sidecar.

    Skipped when the three outputs exist and were built from the same source and anomaly ids.
    The sidecar is written last, so it marks a completely prepared archive. If a run dies after the
    stream finished (stage kept on the local disk), the next run continues from the stage instead of
    streaming again. Scenarios without a single anomaly pixel are dropped and listed in the sidecar.
    """
    output_dir, stage = Path(output_dir), Path(stage_dir) / spec.name
    paths = _package_paths(output_dir, spec.name)
    paths["tar"].parent.mkdir(parents=True, exist_ok=True)
    signature = {**spec.signature(), "anomaly_ids": list(anomaly_ids)}

    if all(p.is_file() for p in paths.values()):
        sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
        if sidecar.get("signature") == signature:
            return {"status": "skipped", **sidecar}

    extracted = stage / "extracted"
    marker, frames_csv = stage / "stream_complete.json", stage / "mask_metadata.csv"
    if marker.is_file() and frames_csv.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        mask_metadata = pd.read_csv(frames_csv, dtype={"scenario_id": str, "frame_id": str})
        print(f"{spec.name}: continuing from the completed local stage")
    else:
        if stage.exists():
            shutil.rmtree(stage)
        resolved = resolver(spec)
        if "static" not in resolved["name"].lower():
            raise ValueError(f"{spec.name}: expected a Static anomaly archive, got {resolved['name']!r}")
        print(f"{spec.name}: {resolved['name']} ({resolved.get('size', 0) / 1e9:.1f} GB)", flush=True)
        result = stream_archive(
            resolved["url"], extracted, anomaly_ids=anomaly_ids, total=resolved.get("size") or None,
            expected_md5=resolved.get("md5") or None, label=spec.name, min_free_gb=min_free_gb)
        if result["archive_root"] is None or result["mask_metadata"].empty:
            raise RuntimeError(f"{spec.name}: no frames found (archive layout differs from expectation)")
        mask_metadata = result["mask_metadata"]
        info = {k: result[k] for k in ("archive_root", "config_names", "counts", "compressed_mb", "seconds",
                                       "retries", "md5_checked")}
        info["archive"] = resolved["name"]
        mask_metadata.to_csv(frames_csv, index=False)
        marker.write_text(json.dumps(info, indent=2), encoding="utf-8")

    root = info["archive_root"]
    configs = []
    for name in info["config_names"]:
        text = json.loads((extracted / root / "Scenario_Configuration_Files" / name).read_text(encoding="utf-8"))
        if "scenario_definition" in text:
            configs.append(name)
    if len(configs) != 1:
        raise RuntimeError(f"{spec.name}: expected one scenario configuration, found {configs or 'none'}")
    metadata = load_scenario_metadata(extracted / root / "Scenario_Configuration_Files" / configs[0])

    per_scenario = mask_metadata.groupby("scenario_id")["anomaly_pixels"].sum()
    empty = sorted(per_scenario[per_scenario == 0].index)
    for scenario_id in empty:
        shutil.rmtree(extracted / root / f"Scenario_{scenario_id}", ignore_errors=True)
    mask_metadata = mask_metadata[~mask_metadata["scenario_id"].isin(empty)]
    metadata = metadata[~metadata["scenario_id"].isin(empty)]

    manifest = build_master_manifest(extracted, metadata, mask_metadata)
    validate_manifest(manifest, extracted)

    temporary = paths["tar"].with_suffix(".tar.partial")
    with tarfile.open(temporary, "w") as tar:
        tar.add(extracted / root, arcname=root)
    temporary.replace(paths["tar"])
    manifest.to_csv(paths["manifest"], index=False)

    sidecar = {
        "name": spec.name,
        "signature": signature,
        "archive": info["archive"],
        "archive_root": root,
        "town": sorted(manifest["town"].unique()),
        "scenarios": int(manifest["scenario_id"].nunique()),
        "frames": len(manifest),
        "dropped_scenarios_without_anomaly": empty,
        "families": manifest.drop_duplicates("scenario_id")["anomaly_family"].value_counts().to_dict(),
        "md5_checked": info["md5_checked"],
        "compressed_mb": info["compressed_mb"],
        "stream_seconds": info["seconds"],
        "retries": info["retries"],
        "tar_gb": round(paths["tar"].stat().st_size / 1024 ** 3, 2),
        "prepared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    if not keep_stage:
        shutil.rmtree(stage, ignore_errors=True)
    return {"status": "prepared", **sidecar}


def load_archive_manifests(output_dir: str | Path, names: list[str]) -> pd.DataFrame:
    """Concatenate the per-archive manifests of ``names`` into one master manifest."""
    parts = []
    for name in names:
        path = _package_paths(Path(output_dir), name)["manifest"]
        if not path.is_file():
            raise FileNotFoundError(f"Archive {name!r} is not prepared: {path}")
        parts.append(pd.read_csv(path, dtype={"scenario_id": str, "frame_id": str}))
    manifest = pd.concat(parts, ignore_index=True)
    if manifest.duplicated(["scenario_id", "frame_id"]).any():
        raise ValueError("Duplicate frames across archives")
    return manifest


def stage_prepared_dataset(output_dir: str | Path, stage_dir: str | Path, names: list[str] | None = None) -> dict:
    """Unpack the prepared archives onto local disk for training.

    Result: ``<stage>/extracted`` (all archive roots) and ``<stage>/processed`` (master manifest and
    ``lofo_splits``). Returns {"data_root", "processed_dir"}. Existing staged data is reused.
    """
    output_dir, stage_dir = Path(output_dir), Path(stage_dir)
    tars = sorted((output_dir / "towns").glob("*.tar"))
    if names is not None:
        tars = [output_dir / "towns" / f"{n}.tar" for n in names]
    missing = [t for t in tars if not t.is_file()]
    if missing or not tars:
        raise FileNotFoundError(f"Prepared archives missing: {[str(t) for t in missing] or 'none found'}")

    data_root, processed = stage_dir / "extracted", stage_dir / "processed"
    done = stage_dir / "staged_archives.json"
    wanted = sorted(t.name for t in tars)
    if done.is_file() and json.loads(done.read_text(encoding="utf-8")) == wanted and processed.is_dir():
        return {"data_root": data_root, "processed_dir": processed}

    stage_dir.mkdir(parents=True, exist_ok=True)
    needed = sum(t.stat().st_size for t in tars) * 1.1
    if shutil.disk_usage(stage_dir).free < needed:
        raise RuntimeError("Insufficient local disk space to unpack the prepared archives")
    for sub in (data_root, processed):
        if sub.exists():
            shutil.rmtree(sub)
    data_root.mkdir(parents=True)
    for tar_path in tars:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(data_root, filter="data")
    shutil.copytree(output_dir / "processed", processed)
    done.write_text(json.dumps(wanted), encoding="utf-8")
    return {"data_root": data_root, "processed_dir": processed}
