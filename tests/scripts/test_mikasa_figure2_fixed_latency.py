from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tasks.mikasa_intercept_grab_fast import figure2_fixed_latency as sweep


def test_cli_paths_preserve_virtualenv_symlinks(tmp_path: Path):
    executable = tmp_path / "base-python"
    executable.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(executable)

    assert sweep._path(str(venv_python)) == venv_python.absolute()


def test_figure2_matrix_has_six_curves_and_thirty_conditions():
    jobs = sweep.jobs(sweep.FORMAL_LATENCIES)

    assert len(sweep.ARTIFACTS) == 6
    assert len(jobs) == 30
    assert {
        (job["model"], job["horizon"])
        for job in jobs
    } == {
        ("qwenoft", 1),
        ("qwenoft", 8),
        ("qwenpi_v3", 1),
        ("qwenpi_v3", 8),
        ("qwengr00t", 1),
        ("qwengr00t", 8),
    }
    assert {
        job["latency_frames"] * 1000.0 / sweep.FPS for job in jobs
    } == {0.0, 50.0, 100.0, 150.0, 200.0}
    assert {artifact["source"]["path"] for artifact in sweep.ARTIFACTS} == {
        "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z",
        "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/h8-naive-dagger-20260923",
        "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z",
        "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h8/h8-naive-dagger-20260923",
        "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h1/mikasa_intercept_grab_fast_qwengr00t_formal_latest_r8gpu7_20260907T075003Z",
        "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h8/h8-naive-dagger-20260923",
    }
    assert {artifact["source"]["revision"] for artifact in sweep.ARTIFACTS} == {
        "56b2c095d76ea5b10dd304e6e03c2ca81801630b",
        "5b533937a72e9fa923ecf28b1c1be5a16bad488d",
    }
    assert {
        artifact["model"]: artifact["source"]["selected_step"]
        for artifact in sweep.ARTIFACTS if artifact["horizon"] == 8
    } == {"qwenoft": 5000, "qwenpi_v3": 4000, "qwengr00t": 3000}


def test_eval_config_uses_the_fixed_mikasa_protocol(tmp_path: Path):
    job = {**sweep.ARTIFACTS[1], "latency_frames": 4}
    backbone = tmp_path / "Qwen3-VL-4B-Instruct"
    config = sweep.build_eval_config(
        h1_root=tmp_path / "h1",
        h8_root=tmp_path / "h8",
        backbone_path=backbone,
        output_dir=tmp_path / "output",
        job=job,
        vla_python=tmp_path / "vla-python",
        episodes=200,
        seed=sweep.START_SEED,
    )

    assert config["latency"] == {
        "method": "fixed",
        "fixed_latency_ms": 200.0,
    }
    assert config["env"]["env_fps"] == 20
    assert config["env"]["obs_fps"] == 20
    assert config["env"]["simulator_device"] == "cuda:0"
    assert config["policy"]["backbone_path"] == str(backbone)
    assert config["executor"] == {
        "mode": "simulated",
        "sync_cuda": False,
        "parallel_envs": 10,
        "inference_pool": True,
        "simulated_worker_capacity": 1,
        "inference_devices": ["cuda:0"],
        "inference_batch_size": 8,
    }
    assert config["evaluation"] == {
        "episodes": 200,
        "max_env_steps": 60,
    }


def test_all_models_use_the_synchronized_runtime_contract():
    assert sweep.inference_code() == {
        "benchmark_revision": "0ce3400b86f01252c563e76a9e6f12f1ffdedb8d",
        "starvla_revision": "3d2cd7fce72def48d9020d794392489031ba3f46",
        "mikasa_revision": "34ece90941291dceaf942938b9f6c92f9dbae01f",
        "working_tree": "local source synchronized before launch",
    }
    assert sweep.ARTIFACTS[2]["source"]["inference_config_override"] == {
        "field": "framework.action_model.diffusion_model_cfg.interleave_self_attention",
        "hub_value": True,
        "eval_value": False,
    }


def _write_h8_fixture(run_dir: Path, *, latency_frames: int) -> None:
    run_dir.mkdir(parents=True)
    episodes = []
    actions = []
    steps = []
    for episode_id in range(2):
        episodes.append(
            {
                "episode_id": episode_id,
                "episode_return_env": float(episode_id + 1),
                "survival_steps": 60,
                "metadata": {
                    "episode_seed": sweep.START_SEED + episode_id,
                    "task_metrics": {"success": float(episode_id == 0)},
                },
            }
        )
        actions.append(
            {
                "obs_env_step": 0,
                "ready_raw_frame": latency_frames,
                "apply_raw_frame": latency_frames,
                "latency_raw_frames": latency_frames,
                "applied": True,
                "dropped": False,
                "metadata": {"episode_id": episode_id},
            }
        )
        for env_step in range(60):
            before_action = env_step < latency_frames
            source_step = None if before_action else 0
            info = {
                "active_action_obs_env_step": source_step,
                "in_flight_count": 1 if before_action else 0,
                "observation_submitted": env_step == 0,
                "observation_dropped": env_step != 0,
            }
            if not before_action:
                info["chunk_age_raw_frames"] = env_step
            steps.append(
                {
                    "episode_id": episode_id,
                    "env_step": env_step,
                    "active_action": {"is_noop": before_action},
                    "info": info,
                }
            )
    for filename, rows in (
        ("episode_metrics.jsonl", episodes),
        ("actions.jsonl", actions),
        ("steps.jsonl", steps),
    ):
        (run_dir / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_condition_summary_checks_seeds_metrics_and_h8_execution(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_h8_fixture(run_dir, latency_frames=2)
    job = {**sweep.ARTIFACTS[1], "latency_frames": 2}

    result = sweep.summarize_condition(
        run_dir,
        job=job,
        bundle=tmp_path / "bundle",
        episodes=2,
        seed=sweep.START_SEED,
    )

    assert result["success_count"] == 1
    assert result["success_rate"] == 0.5
    assert result["mean_return"] == 1.5
    assert result["std_return"] == 0.5
    assert result["mean_length"] == 60.0
    assert result["execution_audit"]["initial_noop_transitions"] == 4
    assert result["execution_audit"]["executed_later_heads"] == 116
    assert result["execution_audit"]["passed"] is True


def test_zero_latency_h8_replans_with_chunk_head_zero(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_h8_fixture(run_dir, latency_frames=0)
    steps = [
        {
            **step,
            "info": {
                **step["info"],
                "active_action_obs_env_step": step["env_step"],
                "chunk_age_raw_frames": 0,
                "in_flight_count": 0,
                "observation_submitted": True,
                "observation_dropped": False,
            },
        }
        for step in (
            json.loads(line)
            for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        )
    ]
    (run_dir / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8"
    )

    result = sweep.summarize_condition(
        run_dir,
        job={**sweep.ARTIFACTS[1], "latency_frames": 0},
        bundle=tmp_path / "bundle",
        episodes=2,
        seed=sweep.START_SEED,
    )

    assert result["execution_audit"]["executed_later_heads"] == 0
    assert result["execution_audit"]["passed"] is True


def test_condition_summary_rejects_wrong_seed_sequence(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_h8_fixture(run_dir, latency_frames=2)
    records = [
        json.loads(line)
        for line in (run_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records[1]["metadata"]["episode_seed"] += 1
    (run_dir / "episode_metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="episode seeds mismatch"):
        sweep.summarize_condition(
            run_dir,
            job={**sweep.ARTIFACTS[1], "latency_frames": 2},
            bundle=tmp_path / "bundle",
            episodes=2,
            seed=sweep.START_SEED,
        )


def test_aggregate_requires_complete_matching_raw_conditions(tmp_path: Path):
    for artifact in sweep.ARTIFACTS:
        job = {**artifact, "latency_frames": 2}
        run_dir = sweep.condition_dir(tmp_path, job)
        _write_h8_fixture(run_dir, latency_frames=2)
        sweep.summarize_condition(
            run_dir,
            job=job,
            bundle=tmp_path / "bundle",
            episodes=2,
            seed=sweep.START_SEED,
        )
        (run_dir / "complete.json").write_text(
            json.dumps({
                "model": job["model"],
                "horizon": job["horizon"],
                "latency_frames": 2,
                "episodes": 2,
                "seed": sweep.START_SEED,
            }),
            encoding="utf-8",
        )

    result = sweep.aggregate(
        output_root=tmp_path,
        revision=sweep.MODEL_REVISION,
        episodes=2,
        seed=sweep.START_SEED,
        latencies=(2,),
    )
    assert len(result["curves"]) == 6
    assert all(curve["latencies"]["2"]["success_count"] == 1 for curve in result["curves"].values())

    run_dir = sweep.condition_dir(tmp_path, {**sweep.ARTIFACTS[0], "latency_frames": 2})
    records = [
        json.loads(line)
        for line in (run_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records[1]["metadata"]["episode_seed"] = sweep.START_SEED
    (run_dir / "episode_metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="episode seeds mismatch"):
        sweep.aggregate(
            output_root=tmp_path,
            revision=sweep.MODEL_REVISION,
            episodes=2,
            seed=sweep.START_SEED,
            latencies=(2,),
        )
