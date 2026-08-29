from __future__ import annotations

import tarfile
from pathlib import Path

from latency_bench.envs import humanoidbench_sonic_assets as assets


def test_prepare_g1_sonic_encoder_uses_pinned_path(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)

    encoder = assets.prepare_g1_sonic_encoder(tmp_path / "cache")

    assert encoder == tmp_path / "sonic_v1_1/model_encoder.onnx"
    assert [call["filename"] for call in calls] == [assets.SONIC_ENCODER_FILENAME]
    assert {call["revision"] for call in calls} == {assets.SONIC_REVISION}


def test_prepare_g1_sonic_planner_uses_pinned_path(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)

    planner = assets.prepare_g1_sonic_planner(tmp_path / "cache")

    assert planner == tmp_path / assets.SONIC_PLANNER_FILENAME
    assert [call["filename"] for call in calls] == [assets.SONIC_PLANNER_FILENAME]
    assert {call["revision"] for call in calls} == {assets.SONIC_REVISION}


def test_prepare_g1_sonic_skill_references_extracts_pinned_csvs(tmp_path, monkeypatch):
    calls = []
    archive_root = tmp_path / "archive_root"
    csv_paths = {
        name: archive_root / member
        for name, member in assets.BONES_SEED_SKILL_CSVS.items()
    }
    for name, csv_path in csv_paths.items():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(f"Frame,skill\n0,{name}\n", encoding="utf-8")

    def fake_download(**kwargs):
        calls.append(kwargs)
        archive = Path(kwargs["local_dir"]) / kwargs["filename"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="w:gz") as bundle:
            for name, csv_path in csv_paths.items():
                bundle.add(csv_path, arcname=assets.BONES_SEED_SKILL_CSVS[name])
        return archive

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)
    extracted = assets.prepare_g1_sonic_skill_references(tmp_path / "cache")

    assert set(extracted) == {"fast_run", "obstacle_jump"}
    for name, path in extracted.items():
        assert path == tmp_path / "cache" / "bones_seed" / assets.BONES_SEED_SKILL_CSVS[name]
        assert path.read_text(encoding="utf-8") == f"Frame,skill\n0,{name}\n"
    assert calls[0]["repo_id"] == assets.BONES_SEED_REPO_ID
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["filename"] == assets.BONES_SEED_ARCHIVE_FILENAME
    assert calls[0]["revision"] == assets.BONES_SEED_REVISION
    assert "token" not in calls[0]
