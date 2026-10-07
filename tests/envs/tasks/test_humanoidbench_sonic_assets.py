from __future__ import annotations

from latency_bench.envs.tasks.humanoidbench import assets as assets


def test_prepare_g1_sonic_assets_returns_pinned_scene_root(tmp_path, monkeypatch):
    snapshot = tmp_path / "assets"
    (snapshot / "sonic_g1").mkdir(parents=True)
    monkeypatch.setattr(assets, "snapshot_download", lambda **_kwargs: snapshot)

    asset_root = assets.prepare_g1_sonic_assets(tmp_path / "cache")

    assert asset_root == snapshot / "sonic_g1"
    assert (asset_root / "tasks" / "truck.xml").is_file()
