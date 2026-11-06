import hashlib
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import roadguard.data.preparation as preparation
from roadguard.data.preparation import (
    ArchiveIntegrityError,
    ArchiveSpec,
    anomaly_object,
    generate_anomaly_mask,
    inspect_semantic_labels,
    load_archive_manifests,
    load_data_config,
    load_scenario_metadata,
    prepare_archive,
    stage_prepared_dataset,
    stream_archive,
)

ROOT = "AnoVox_Static_Mono_Town99"
H, W, N_FRAMES = 24, 32, 12
SPECIAL, EMPTY, WOLF = (
    "aaaa0001-0000-0000-0000-000000000001",
    "bbbb0002-0000-0000-0000-000000000002",
    "cccc0003-0000-0000-0000-000000000003",
)
SCENARIOS = {  # id -> (blueprint, frames that show the anomaly)
    SPECIAL: ("static.prop.o_naga_special", range(3, 8)),
    EMPTY: ("static.prop.o_cardboardbox03_home", range(0, 0)),   # anomaly never visible
    WOLF: ("static.prop.o_wolf_animal", range(6, 12)),
}
ANOMALY_PIXELS = 7 * 14


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def make_archive(path, root=ROOT, town="Town99", scenarios=SCENARIOS):
    """Synthetic Static archive. As in the real data, the configuration JSON is the LAST member."""
    config = {"scenario_definition": {"map": town, "scenarios": [
        {"id": sid, "anomaly_config": {"anomaly_bp_name": bp}} for sid, (bp, _) in scenarios.items()]}}
    rng = np.random.default_rng(0)
    members = []
    for sid, (_, visible) in scenarios.items():
        for frame in range(N_FRAMES):
            fid = f"{frame:04d}"
            semantic = np.ones((H, W), np.uint8)
            if frame in visible:
                semantic[5:12, 6:20] = 32
            members += [
                (f"{root}/Scenario_{sid}/RGB-CAM_front/RGB-CAM_front_{fid}.png",
                 _png(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))),
                (f"{root}/Scenario_{sid}/DEPTH_CAM_front/DEPTH_CAM_front_{fid}.png",
                 _png(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))),
                (f"{root}/Scenario_{sid}/SEMANTIC-CAM_front/SEMANTIC-CAM_front_{fid}.png", _png(semantic)),
                (f"{root}/Scenario_{sid}/LIDAR_top/LIDAR_top_{fid}.bin", b"lidar" * 50),   # must be ignored
            ]
    members.append((f"{root}/Scenario_Configuration_Files/scenario_config_map_{town}.json",
                    json.dumps(config).encode()))
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            _add(tar, name, data)


class _Server:
    def __init__(self, path, break_first=False):
        data = path.read_bytes()
        outer = self
        outer.requests = 0
        outer.md5 = hashlib.md5(data).hexdigest()
        outer.size = len(data)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                if break_first and outer.requests == 1:
                    self.wfile.write(data[: len(data) // 2])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/a.tar.gz"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def resolver(self, name=f"{ROOT}.tar.gz"):
        return lambda spec: {"name": name, "size": self.size, "md5": self.md5, "url": self.url}

    def close(self):
        self.httpd.shutdown()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "127.0.0.1,localhost")


@pytest.fixture
def served(tmp_path):
    servers = []

    def _make(root=ROOT, town="Town99", scenarios=SCENARIOS, **kwargs):
        path = tmp_path / f"archive{len(servers)}.tar.gz"
        make_archive(path, root, town, scenarios)
        server = _Server(path, **kwargs)
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.close()


# ----------------------------------------------------------------------------- configuration
def test_load_data_config(tmp_path):
    path = tmp_path / "data_sources.yaml"
    path.write_text("archives:\n  - {name: Town03, zenodo: '123'}\n  - {name: Town04, url: 'http://x/a.tar.gz'}\n",
                    encoding="utf-8")
    config = load_data_config(path)
    assert [a.name for a in config["archives"]] == ["Town03", "Town04"]
    assert config["archives"][0].zenodo == "123" and config["archives"][1].url == "http://x/a.tar.gz"
    assert config["anomaly_ids"] == (29, 30, 31, 32, 33, 34, 100)
    assert config["splits"] == {"seed": 42, "val_frac": 0.2, "min_family_scenarios": 8}


@pytest.mark.parametrize("text, message", [
    ("archives:\n  - {name: A, zenodo: '1', url: 'http://x'}\n", "exactly one"),
    ("archives:\n  - {name: A}\n", "exactly one"),
    ("archives:\n  - {name: A, zenodo: '1'}\n  - {name: A, zenodo: '2'}\n", "Duplicate"),
    ("archives:\n  - {name: 'a/b', zenodo: '1'}\n", "Invalid archive name"),
    ("archives: []\n", "No archives"),
])
def test_load_data_config_rejects_invalid(tmp_path, text, message):
    path = tmp_path / "data_sources.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_data_config(path)


# ----------------------------------------------------------------------------- metadata and masks
def test_anomaly_object():
    assert anomaly_object("static.prop.o_naga_special", "special") == "naga"
    assert anomaly_object("static.prop.o_wolf_animal", "animal") == "wolf"
    assert anomaly_object("static.prop.o_box_home", "home") == "box"


def test_load_scenario_metadata(tmp_path):
    config = {"scenario_definition": {"map": "Town04", "scenarios": [
        {"id": "scenario_1", "anomaly_config": {"anomaly_bp_name": "static.prop.o_box_home"}},
        {"id": "scenario_2", "anomaly_config": {"anomaly_bp_name": "static.prop.o_wolf_animal"}},
    ]}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = load_scenario_metadata(config_path)

    assert result["anomaly_family"].tolist() == ["home", "animal"]
    assert result["anomaly_object"].tolist() == ["box", "wolf"]
    assert result["town"].tolist() == ["Town04", "Town04"]
    assert "source" not in result.columns


def test_load_scenario_metadata_requires_blueprint(tmp_path):
    config = {"scenario_definition": {"map": "Town04", "scenarios": [{"id": "normal_1"}]}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing anomaly blueprint"):
        load_scenario_metadata(config_path)


def test_generate_anomaly_mask(tmp_path):
    semantic = np.array([[0, 1, 29, 30], [10, 100, 214, 0], [31, 32, 33, 34]], dtype=np.uint8)
    semantic_path = tmp_path / "semantic.png"
    mask_path = tmp_path / "masks" / "mask.png"
    Image.fromarray(semantic).save(semantic_path)

    anomaly_pixels = generate_anomaly_mask(semantic_path, mask_path)

    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    expected = np.array([[0, 0, 255, 255], [0, 255, 0, 0], [255, 255, 255, 255]], dtype=np.uint8)
    assert np.array_equal(mask, expected)
    assert anomaly_pixels == 7


def test_inspect_semantic_labels(tmp_path):
    path = tmp_path / "semantic.png"
    Image.fromarray(np.array([[0, 29, 29], [30, 100, 0]], dtype=np.uint8)).save(path)

    assert inspect_semantic_labels(path) == {0: 2, 29: 2, 30: 1, 100: 1}


# ----------------------------------------------------------------------------- streaming
def test_stream_archive_keeps_required_files_and_writes_masks(tmp_path, served):
    server = served()
    out = tmp_path / "out"

    result = stream_archive(server.url, out, total=server.size, expected_md5=server.md5, min_free_gb=0)

    assert result["archive_root"] == ROOT
    assert result["md5_checked"] is True
    assert result["config_names"] == ["scenario_config_map_Town99.json"]
    assert result["counts"]["rgb"] == result["counts"]["depth"] == result["counts"]["semantic"] == 3 * N_FRAMES
    assert not list(out.rglob("*.bin"))                              # lidar ignored
    frames = result["mask_metadata"]
    assert len(frames) == 3 * N_FRAMES
    assert set(frames.loc[frames.scenario_id == SPECIAL, "anomaly_pixels"]) == {0, ANOMALY_PIXELS}
    assert frames.loc[frames.scenario_id == EMPTY, "anomaly_pixels"].sum() == 0
    row = frames[(frames.scenario_id == SPECIAL) & (frames.frame_id == "0004")].iloc[0]
    with Image.open(out / row.mask_path) as image:
        mask = np.asarray(image)
    assert set(np.unique(mask)) == {0, 255} and int(np.count_nonzero(mask)) == ANOMALY_PIXELS


def test_stream_archive_rejects_wrong_md5(tmp_path, served):
    server = served()
    with pytest.raises(ArchiveIntegrityError, match="md5"):
        stream_archive(server.url, tmp_path / "out", expected_md5="0" * 32, min_free_gb=0)


def test_stream_archive_retries_dropped_connection(tmp_path, served):
    clean, flaky = served(), served(break_first=True)

    expected = stream_archive(clean.url, tmp_path / "clean", min_free_gb=0)
    result = stream_archive(flaky.url, tmp_path / "flaky", expected_md5=flaky.md5, min_free_gb=0)

    assert result["retries"] == 1 and flaky.requests == 2
    pd.testing.assert_frame_equal(
        result["mask_metadata"].sort_values(["scenario_id", "frame_id"]).reset_index(drop=True),
        expected["mask_metadata"].sort_values(["scenario_id", "frame_id"]).reset_index(drop=True),
    )
    assert sorted(p.name for p in (tmp_path / "flaky").rglob("*.png")) == \
        sorted(p.name for p in (tmp_path / "clean").rglob("*.png"))


def test_stream_archive_repairs_incomplete_file(tmp_path, served):
    server = served()
    out = tmp_path / "out"
    target = out / ROOT / f"Scenario_{SPECIAL}" / "RGB-CAM_front" / "RGB-CAM_front_0000.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"incomplete")

    stream_archive(server.url, out, min_free_gb=0)

    with Image.open(target) as image:
        assert image.size == (W, H)


def _serve_members(tmp_path, members):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            _add(tar, name, data)
    return _Server(path)


def test_stream_archive_rejects_unsafe_path(tmp_path):
    server = _serve_members(tmp_path, [(f"{ROOT}/../Scenario_abc/RGB-CAM_t/RGB-CAM_t_1.png", b"x")])
    try:
        with pytest.raises(ValueError, match="Unsafe archive path"):
            stream_archive(server.url, tmp_path / "out", min_free_gb=0)
    finally:
        server.close()
    assert not list(tmp_path.rglob("RGB-CAM_t_1.png"))


def test_stream_archive_rejects_symlink_escape(tmp_path):
    out, outside = tmp_path / "out", tmp_path / "outside"
    out.mkdir()
    outside.mkdir()
    try:
        (out / ROOT).symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation not permitted: {exc}")
    server = _serve_members(tmp_path, [(f"{ROOT}/Scenario_abc/RGB-CAM_t/RGB-CAM_t_1.png", b"rgb")])
    try:
        with pytest.raises(ValueError, match="escapes destination"):
            stream_archive(server.url, out, min_free_gb=0)
    finally:
        server.close()
    assert not list(outside.rglob("*.png"))


# ----------------------------------------------------------------------------- one archive -> package
def _spec():
    return ArchiveSpec(name="Town99", zenodo="1")


def test_prepare_archive_end_to_end_and_skip(tmp_path, served):
    server = served()
    out, stage = tmp_path / "drive", tmp_path / "stage"

    result = prepare_archive(_spec(), out, stage, resolver=server.resolver(), min_free_gb=0)

    assert result["status"] == "prepared"
    assert result["scenarios"] == 2 and result["frames"] == 2 * N_FRAMES
    assert result["dropped_scenarios_without_anomaly"] == [EMPTY]
    assert result["families"] == {"special": 1, "animal": 1}
    assert result["md5_checked"] is True
    assert (out / "towns" / "Town99.tar").is_file() and (out / "towns" / "Town99.json").is_file()
    assert not (stage / "Town99").exists()

    manifest = pd.read_csv(out / "towns" / "Town99_manifest.csv", dtype={"scenario_id": str, "frame_id": str})
    assert set(manifest.anomaly_object) == {"naga", "wolf"}
    assert "source" not in manifest.columns
    with tarfile.open(out / "towns" / "Town99.tar") as tar:
        names = tar.getnames()
    assert any("MASK_0004.png" in n for n in names) and not any(EMPTY in n for n in names)

    requests = server.requests
    again = prepare_archive(_spec(), out, stage, resolver=server.resolver(), min_free_gb=0)
    assert again["status"] == "skipped" and server.requests == requests

    changed = prepare_archive(_spec(), out, stage, anomaly_ids=(32,), resolver=server.resolver(), min_free_gb=0)
    assert changed["status"] == "prepared" and server.requests == requests + 1


def test_prepare_archive_resumes_from_completed_stage(tmp_path, served, monkeypatch):
    server = served()
    out, stage = tmp_path / "drive", tmp_path / "stage"
    real = preparation.build_master_manifest

    def broken(*args, **kwargs):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(preparation, "build_master_manifest", broken)
    with pytest.raises(RuntimeError, match="manifest failed"):
        prepare_archive(_spec(), out, stage, resolver=server.resolver(), min_free_gb=0)
    assert server.requests == 1 and (stage / "Town99" / "stream_complete.json").is_file()

    monkeypatch.setattr(preparation, "build_master_manifest", real)
    result = prepare_archive(_spec(), out, stage, resolver=server.resolver(), min_free_gb=0)

    assert result["status"] == "prepared" and server.requests == 1     # no second download


def test_prepare_archive_rejects_non_static_archive_before_streaming(tmp_path, served):
    server = served()
    with pytest.raises(ValueError, match="Static"):
        prepare_archive(_spec(), tmp_path / "drive", tmp_path / "stage",
                        resolver=server.resolver(name="AnoVox_Normality_Mono_Town04.tar.gz"), min_free_gb=0)
    assert server.requests == 0


# ----------------------------------------------------------------------------- master manifest and staging
def test_load_archive_manifests_and_stage_prepared_dataset(tmp_path, served):
    server = served()
    out, stage = tmp_path / "drive", tmp_path / "stage"
    prepare_archive(_spec(), out, tmp_path / "work", resolver=server.resolver(), min_free_gb=0)

    manifest = load_archive_manifests(out, ["Town99"])
    assert len(manifest) == 2 * N_FRAMES
    with pytest.raises(FileNotFoundError, match="not prepared"):
        load_archive_manifests(out, ["Town99", "Town00"])

    (out / "processed" / "lofo_splits").mkdir(parents=True)
    manifest.to_csv(out / "processed" / "master_manifest.csv", index=False)

    staged = stage_prepared_dataset(out, stage)

    for column in ("rgb_path", "depth_path", "semantic_path", "mask_path"):
        assert all((staged["data_root"] / p).is_file() for p in manifest[column])
    assert (staged["processed_dir"] / "master_manifest.csv").is_file()
    assert stage_prepared_dataset(out, stage) == staged                   # reuse without unpacking again

    with pytest.raises(FileNotFoundError):
        stage_prepared_dataset(out, tmp_path / "stage2", names=["Town00"])


# ----------------------------------------------------------------------------- the whole of Notebook 1
def test_notebook_flow_two_archives_to_training_fold(tmp_path, served):
    """Two configured archives -> packages -> master manifest -> LOFO folds -> staged data -> training loader."""
    from roadguard.data.splits import create_lofo_splits
    from roadguard.training.experiment import load_lofo_fold

    def scen(prefix, spec):   # id -> (blueprint, visible frames)
        return {f"{prefix}{i:03d}-0000-0000-0000-000000000000": (f"static.prop.o_{obj}_{family}", range(2, 9))
                for i, (obj, family) in enumerate(spec)}

    towns = {
        "Town11": scen("a", [("naga", "special"), ("ufo", "special"), ("wolf", "animal"), ("deer", "animal")]),
        "Town12": scen("b", [("cat", "special"), ("fox", "animal"), ("box", "home"), ("bin", "home")]),
    }
    out = tmp_path / "drive"
    specs = []
    for town, scenarios in towns.items():
        server = served(root=f"AnoVox_Static_Mono_{town}", town=town, scenarios=scenarios)
        spec = ArchiveSpec(name=town, zenodo=town)
        prepare_archive(spec, out, tmp_path / "work", resolver=server.resolver(f"AnoVox_Static_Mono_{town}.tar.gz"),
                        min_free_gb=0)
        specs.append(spec)

    manifest = load_archive_manifests(out, [s.name for s in specs])
    assert set(manifest.town) == {"Town11", "Town12"} and manifest.scenario_id.nunique() == 8
    assert (~manifest.has_anomaly).any() and manifest.has_anomaly.any()      # frames with and without anomaly

    folds = create_lofo_splits(manifest, val_frac=0.4, min_family_scenarios=2, seed=7)
    assert set(folds) == {"animal", "home", "special"}
    (out / "processed" / "lofo_splits").mkdir(parents=True)
    manifest.to_csv(out / "processed" / "master_manifest.csv", index=False)
    for family, fold in folds.items():
        fold.to_csv(out / "processed" / "lofo_splits" / f"{family}.csv", index=False)

    staged = stage_prepared_dataset(out, tmp_path / "stage")
    for family in folds:
        parts = load_lofo_fold(staged["processed_dir"] / "lofo_splits" / f"{family}.csv", family)
        assert set(parts["test"].anomaly_family) == {family}
        for column in ("rgb_path", "depth_path", "mask_path"):
            assert all((staged["data_root"] / p).is_file() for part in parts.values() for p in part[column])
