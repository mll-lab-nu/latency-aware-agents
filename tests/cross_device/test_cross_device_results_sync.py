from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from latency_bench.analysis.cross_device_figure_data import _resolve_latency_trace
from latency_bench.cross_device import runner


def _make_task(out: Path, *, with_profile: bool) -> SimpleNamespace:
    output_dir = out / "runs" / "measure" / "openvla-flappy" / "best_seed_0"
    output_dir.mkdir(parents=True)
    (output_dir / "episode_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    profile_output_dir = None
    if with_profile:
        profile_output_dir = out / "profiles" / "openvla" / "flappy" / "rtx3090" / "instance_abcd1234"
        profile_output_dir.mkdir(parents=True)
        (profile_output_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    (out / "manifests").mkdir(parents=True)
    (out / "manifests" / "runs.jsonl").write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(output_dir=output_dir, profile_output_dir=profile_output_dir, instance_id="instance_abcd1234")


def test_upload_run_results_mirrors_slices_to_raw_prefix(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None)))

    out = tmp_path / "cross_device"
    task = _make_task(out, with_profile=True)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "upload_folder_to_hf", lambda **kw: calls.append(kw))
    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setenv("HF_PROFILE_REPO_TYPE", "dataset")
    monkeypatch.delenv("CROSS_DEVICE_NO_RESULT_UPLOAD", raising=False)

    manifest: dict[str, object] = {}
    runner._upload_run_results(task, out, manifest, {"device": {"id": "rtx3090"}})

    paths = {c["path_in_repo"] for c in calls}
    assert "raw/rtx3090/instances/instance_abcd1234/cross_device/manifests" in paths
    assert (
        "raw/rtx3090/instances/instance_abcd1234/cross_device/runs/measure/openvla-flappy/best_seed_0"
        in paths
    )
    assert all("/profiles/" not in str(path) for path in paths)
    assert manifest["results_uploaded"] is True
    assert all(c["repo_id"] == "latency-sensitive-bench/profiles" for c in calls)
    assert all(c["repo_type"] == "dataset" for c in calls)


def test_upload_run_results_skipped_without_repo(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None)))

    out = tmp_path / "cross_device"
    task = _make_task(out, with_profile=False)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "upload_folder_to_hf", lambda **kw: calls.append(kw))
    monkeypatch.delenv("HF_PROFILE_REPO", raising=False)

    runner._upload_run_results(task, out, {}, {"device": {"id": "rtx3090"}})
    assert calls == []


def test_upload_run_results_opt_out_env(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None)))

    out = tmp_path / "cross_device"
    task = _make_task(out, with_profile=False)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "upload_folder_to_hf", lambda **kw: calls.append(kw))
    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setenv("CROSS_DEVICE_NO_RESULT_UPLOAD", "1")

    runner._upload_run_results(task, out, {}, {"device": {"id": "rtx3090"}})
    assert calls == []


def test_resolve_latency_trace_reads_trace_next_to_profile(tmp_path: Path):
    instance_dir = tmp_path / "profiles" / "openvla" / "flappy" / "rtx3090" / "instance_rtx3090"
    instance_dir.mkdir(parents=True)
    (instance_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    (instance_dir / "e2e_latencies.jsonl").write_text("{}\n", encoding="utf-8")

    found = _resolve_latency_trace(
        {"profile_path": str(instance_dir / "profile.json")},
        trace_name="e2e_latencies.jsonl",
    )
    assert found == instance_dir / "e2e_latencies.jsonl"
