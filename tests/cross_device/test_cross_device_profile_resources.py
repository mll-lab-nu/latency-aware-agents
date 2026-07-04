from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from latency_bench.cross_device.models import ProtocolTask


def _task(
    tmp_path: Path,
    *,
    stage: str = "sim_same",
    target_profile_path: Path | None = None,
    target_profile_instance_id: str | None = "instance_local",
) -> ProtocolTask:
    row = {
        "policy": "dry-policy",
        "env": "Flappy",
        "model_type": "test_model",
        "env_name": "flappy",
    }
    return ProtocolTask(
        stage=stage,
        experiment_id="exp2_same_device_calibration",
        experiment_ids=["exp2_same_device_calibration"],
        panel="b",
        row=row,
        target_device={"id": "a100"},
        replay_device=None,
        seed=0,
        seed_set=[0],
        n_episodes=2,
        checkpoint_kind="best",
        output_dir=tmp_path / "runs",
        generated_config_path=tmp_path / "generated.yaml",
        target_profile_path=target_profile_path,
        profile_output_dir=None,
        instance_id="instance_local",
        instance_origin="manual_override",
        target_profile_instance_id=target_profile_instance_id,
    )


def test_profile_resources_hydrate_real_sim_before_skip(tmp_path: Path, monkeypatch):
    from latency_bench.cross_device import profile_resources

    profiles_dir = tmp_path / "profiles"
    profile_path = profiles_dir / "test_model" / "flappy" / "a100" / "instance_local" / "profile.json"
    task = _task(tmp_path, target_profile_path=profile_path)
    args = SimpleNamespace(dry_run=False)
    calls: list[tuple[Path, Path, str, str, str | None]] = []

    def ensure_profile_cached(
        *,
        profile_path: Path,
        profiles_dir: Path,
        hf_repo: str,
        repo_type: str,
        revision: str | None,
    ) -> Path:
        calls.append((profile_path, profiles_dir, hf_repo, repo_type, revision))
        profile_path.parent.mkdir(parents=True)
        profile_path.write_text("{}\n", encoding="utf-8")
        return profile_path

    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setenv("HF_PROFILE_REPO_TYPE", "dataset")
    monkeypatch.setattr(profile_resources, "ensure_profile_cached", ensure_profile_cached)

    reasons = profile_resources.profile_resource_skip_reasons(
        task=task,
        args=args,
        stage="sim_same",
        profiles_dir=profiles_dir,
        for_doctor=False,
    )

    assert reasons == []
    assert calls == [(profile_path, profiles_dir, "latency-sensitive-bench/profiles", "dataset", None)]


def test_profile_resources_keep_dry_run_read_only(tmp_path: Path, monkeypatch):
    from latency_bench.cross_device import profile_resources

    profiles_dir = tmp_path / "profiles"
    profile_path = profiles_dir / "test_model" / "flappy" / "a100" / "instance_local" / "profile.json"
    task = _task(tmp_path, target_profile_path=profile_path)
    args = SimpleNamespace(dry_run=True)
    calls: list[Path] = []

    def ensure_profile_cached(**kwargs):
        calls.append(kwargs["profile_path"])

    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setattr(profile_resources, "ensure_profile_cached", ensure_profile_cached)

    reasons = profile_resources.profile_resource_skip_reasons(
        task=task,
        args=args,
        stage="sim_same",
        profiles_dir=profiles_dir,
        for_doctor=False,
    )

    assert reasons == []
    assert calls == []


def test_profile_resources_doctor_reports_missing_without_hydrating(tmp_path: Path, monkeypatch):
    from latency_bench.cross_device import profile_resources

    profiles_dir = tmp_path / "profiles"
    profile_path = profiles_dir / "test_model" / "flappy" / "a100" / "instance_local" / "profile.json"
    task = _task(tmp_path, target_profile_path=profile_path)
    args = SimpleNamespace(dry_run=False)
    calls: list[Path] = []

    def ensure_profile_cached(**kwargs):
        calls.append(kwargs["profile_path"])

    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setattr(profile_resources, "ensure_profile_cached", ensure_profile_cached)

    reasons = profile_resources.profile_resource_skip_reasons(
        task=task,
        args=args,
        stage="sim_same",
        profiles_dir=profiles_dir,
        for_doctor=True,
    )

    assert reasons == [f"missing profile: {profile_path}"]
    assert calls == []


def test_profile_resources_report_missing_instance_by_stage(tmp_path: Path):
    from latency_bench.cross_device import profile_resources

    args = SimpleNamespace(dry_run=False)
    measure_task = _task(tmp_path, stage="measure", target_profile_instance_id=None)
    sim_cross_task = _task(tmp_path, stage="sim_cross", target_profile_instance_id=None)

    assert profile_resources.profile_resource_skip_reasons(
        task=measure_task,
        args=args,
        stage="measure",
        profiles_dir=tmp_path / "profiles",
        for_doctor=False,
    ) == ["missing local profile instance: set CROSS_DEVICE_INSTANCE_ID"]
    assert profile_resources.profile_resource_skip_reasons(
        task=sim_cross_task,
        args=args,
        stage="sim_cross",
        profiles_dir=tmp_path / "profiles",
        for_doctor=False,
    ) == ["missing profile: no instance for target a100; pass --target <device>/<instance>"]
