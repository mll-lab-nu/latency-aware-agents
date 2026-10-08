from __future__ import annotations

from latency_bench.envs import humanoidbench_sonic_assets as assets


def test_prepare_g1_sonic_assets_returns_pinned_scene_root(tmp_path, monkeypatch):
    snapshot = tmp_path / "assets"
    (snapshot / "assets/humanoidbench/g1-sonic/sonic_g1").mkdir(parents=True)
    monkeypatch.setattr(assets, "snapshot_download", lambda **_kwargs: snapshot)

    repo_root = tmp_path / "repo"
    truck = repo_root / "third_party/humanoid-bench/humanoid_bench/assets/tasks/truck.xml"
    truck.parent.mkdir(parents=True)
    truck.write_text("<mujoco/>")
    monkeypatch.setattr(assets, "REPO_ROOT", repo_root)

    asset_root = assets.prepare_g1_sonic_assets(tmp_path / "cache")

    assert asset_root == snapshot / "assets/humanoidbench/g1-sonic/sonic_g1"
    assert (asset_root / "tasks" / "truck.xml").read_text() == truck.read_text()
