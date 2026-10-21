from __future__ import annotations

from latency_bench.envs.tasks.humanoidbench import assets as assets


def test_prepare_g1_sonic_assets_returns_pinned_scene_root(tmp_path, monkeypatch):
    snapshot = tmp_path / "assets"
    (snapshot / "assets/humanoidbench/g1-sonic/sonic_g1").mkdir(parents=True)
    monkeypatch.setattr(assets, "snapshot_download", lambda **_kwargs: snapshot)

    package_root = tmp_path / "humanoid_bench"
    truck = package_root / "assets/tasks/truck.xml"
    truck.parent.mkdir(parents=True)
    truck.write_text("<mujoco/>")
    monkeypatch.setattr(assets, "files", lambda package: package_root)

    asset_root = assets.prepare_g1_sonic_assets(tmp_path / "cache")

    assert asset_root == snapshot / "assets/humanoidbench/g1-sonic/sonic_g1"
    assert (asset_root / "tasks" / "truck.xml").read_text() == truck.read_text()
