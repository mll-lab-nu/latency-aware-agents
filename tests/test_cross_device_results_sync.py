from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _make_task(out: Path, *, with_profile: bool) -> SimpleNamespace:
    output_dir = out / "runs" / "measure" / "openvla-flappy" / "flappy" / "rtx3090" / "best_seed_0"
    output_dir.mkdir(parents=True)
    (output_dir / "episode_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    profile_output_dir = None
    if with_profile:
        profile_output_dir = out / "profiles" / "openvla-flappy" / "flappy" / "L_3090"
        profile_output_dir.mkdir(parents=True)
        (profile_output_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    (out / "manifests").mkdir(parents=True)
    (out / "manifests" / "runs.jsonl").write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(output_dir=output_dir, profile_output_dir=profile_output_dir)


def _runtime(alias: str = "openvla") -> SimpleNamespace:
    return SimpleNamespace(matched_alias=alias, selected_alias=None)


def test_upload_run_results_mirrors_slices_to_raw_prefix(tmp_path: Path, monkeypatch):
    from scripts.cross_device import runner

    out = tmp_path / "cross_device"
    task = _make_task(out, with_profile=True)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "upload_folder_to_hf", lambda **kw: calls.append(kw))
    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setenv("HF_PROFILE_REPO_TYPE", "dataset")
    monkeypatch.delenv("CROSS_DEVICE_NO_RESULT_UPLOAD", raising=False)

    manifest: dict[str, object] = {}
    runner._upload_run_results(task, out, manifest, {"device": {"id": "rtx3090"}}, _runtime())

    paths = {c["path_in_repo"] for c in calls}
    assert "raw/rtx3090/openvla/cross_device/manifests" in paths
    assert (
        "raw/rtx3090/openvla/cross_device/runs/measure/openvla-flappy/flappy/rtx3090/best_seed_0"
        in paths
    )
    assert "raw/rtx3090/openvla/cross_device/profiles/openvla-flappy/flappy/L_3090" in paths
    assert manifest["results_uploaded"] is True
    assert all(c["repo_id"] == "latency-sensitive-bench/profiles" for c in calls)
    assert all(c["repo_type"] == "dataset" for c in calls)


def test_upload_run_results_skipped_without_repo(tmp_path: Path, monkeypatch):
    from scripts.cross_device import runner

    out = tmp_path / "cross_device"
    task = _make_task(out, with_profile=False)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "upload_folder_to_hf", lambda **kw: calls.append(kw))
    monkeypatch.delenv("HF_PROFILE_REPO", raising=False)

    runner._upload_run_results(task, out, {}, {"device": {"id": "rtx3090"}}, _runtime())
    assert calls == []


def test_upload_run_results_opt_out_env(tmp_path: Path, monkeypatch):
    from scripts.cross_device import runner

    out = tmp_path / "cross_device"
    task = _make_task(out, with_profile=False)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "upload_folder_to_hf", lambda **kw: calls.append(kw))
    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setenv("CROSS_DEVICE_NO_RESULT_UPLOAD", "1")

    runner._upload_run_results(task, out, {}, {"device": {"id": "rtx3090"}}, _runtime())
    assert calls == []


def test_resolve_latency_trace_finds_nested_raw_layout(tmp_path: Path):
    from scripts.cross_device.build_figure_data import _resolve_latency_trace, slugify

    nested = (
        tmp_path
        / "raw"
        / "rtx3090"
        / "openvla"
        / "cross_device"
        / "profiles"
        / slugify("openvla-flappy")
        / slugify("Flappy")
        / "L_3090"
    )
    nested.mkdir(parents=True)
    (nested / "latencies.jsonl").write_text("{}\n", encoding="utf-8")

    found = _resolve_latency_trace(tmp_path, policy="openvla-flappy", env="Flappy", profile_id="L_3090")
    assert found == nested / "latencies.jsonl"


def test_resolve_latency_trace_prefers_flat_layout(tmp_path: Path):
    from scripts.cross_device.build_figure_data import _resolve_latency_trace, slugify

    flat = tmp_path / slugify("openvla-flappy") / slugify("Flappy") / "L_3090"
    flat.mkdir(parents=True)
    (flat / "latencies.jsonl").write_text("{}\n", encoding="utf-8")

    found = _resolve_latency_trace(tmp_path, policy="openvla-flappy", env="Flappy", profile_id="L_3090")
    assert found == flat / "latencies.jsonl"
