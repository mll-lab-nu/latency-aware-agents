from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.types import Action, StepResult
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.latency.profile import save_profile_manifest, write_latency_samples
from latency_bench.run import run_from_config


@register_env_adapter("raw_clipped_return_test")
def _build_raw_clipped_return_env(config: dict, noop_action: Action) -> EnvAdapter:
    class RawClippedReturnEnv(DummyEnvAdapter):
        def step(self, action: Action) -> StepResult:
            result = super().step(action)
            info = dict(result.info)
            if result.done or result.truncated:
                info.update(
                    {
                        "episodic_raw_return": 123.0,
                        "episodic_clipped_return": 2.0,
                    }
                )
            return StepResult(
                observation=result.observation,
                reward=result.reward,
                done=result.done,
                truncated=result.truncated,
                info=info,
            )

    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    return RawClippedReturnEnv(
        fps_env=float(env_cfg["fps_env"]),
        max_steps=int(evaluation_cfg["eval_max_steps"]),
        noop_action=noop_action,
    )


def _config(tmp_path: Path, *, latency: dict | None = None, env_name: str = "dummy") -> dict:
    return {
        "experiment": {
            "name": "eval_logging_test",
            "seed": 7,
            "num_episodes": 1,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": env_name,
            "fps_env": 25,
            "noop_action": "noop",
            "action_map": {
                "noop": "noop",
                "jump": "jump",
            },
        },
        "control": {
            "frequency_hz": 25,
            "observe_every_n_frames": 1,
        },
        "latency": latency or {
            "type": "fixed",
            "fixed_latency_ms": 80.0,
        },
        "scheduler": {
            "persistence_mode": "noop_after_oneshot",
            "stale_action_mode": "fifo",
        },
        "policy": {
            "type": "constant",
            "action": "jump",
        },
        "evaluation": {
            "eval_max_steps": 3,
            "replay_host": "test_replay_host",
        },
        "logging": {
            "output_dir": str(tmp_path),
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_profile(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    write_latency_samples([10.0, 20.0, 30.0], profile_dir / "latencies.jsonl")
    save_profile_manifest(
        {
            "schema_version": 1,
            "profile_id": "L_eval_logging",
            "profile_type": "rollout_profile",
            "latency_scope": "end_to_end_action_ready",
            "policy_id": "pi_test",
            "env_id": "env_test",
            "target_device": "target_gpu",
            "fps": 25.0,
            "frame_ms": 40.0,
            "n_warmup": 0,
            "n_samples": 3,
            "batch_size": 1,
            "trace_path": "latencies.jsonl",
            "summary": {},
        },
        profile_dir / "profile.json",
    )
    return profile_dir


def test_episode_metrics_log_evaluation_raw_fact_metadata(tmp_path: Path):
    run_from_config(_config(tmp_path), write_summary=False)

    metrics = _read_jsonl(tmp_path / "episode_metrics.jsonl")[0]

    assert metrics["episode_return_env"] == pytest.approx(3.0)
    assert metrics["episode_return"] == pytest.approx(3.0)
    assert metrics["return_raw"] is None
    assert metrics["return_clipped"] is None
    assert metrics["metadata"]["mode"] == "simulated"
    assert metrics["metadata"]["episode_seed"] == 7
    assert metrics["metadata"]["policy_id"] == "constant"
    assert metrics["metadata"]["env_id"] == "dummy"
    assert metrics["metadata"]["latency_type"] == "fixed"
    assert metrics["metadata"]["fps"] == pytest.approx(25.0)
    assert metrics["metadata"]["frame_ms"] == pytest.approx(40.0)
    assert metrics["metadata"]["replay_host"] == "test_replay_host"


def test_return_raw_and_clipped_are_only_from_explicit_info(tmp_path: Path):
    config = _config(tmp_path, env_name="raw_clipped_return_test")
    run_from_config(config, write_summary=False)

    metrics = _read_jsonl(tmp_path / "episode_metrics.jsonl")[0]

    assert metrics["episode_return_env"] == pytest.approx(3.0)
    assert metrics["episode_return"] == pytest.approx(3.0)
    assert metrics["return_raw"] == pytest.approx(123.0)
    assert metrics["return_clipped"] == pytest.approx(2.0)


def test_empirical_profile_metadata_is_logged(tmp_path: Path):
    profile_dir = _write_profile(tmp_path)
    config = _config(
        tmp_path / "run",
        latency={
            "type": "empirical_sample",
            "profile_path": str(profile_dir),
            "seed": 0,
        },
    )
    run_from_config(config, write_summary=False)

    metrics = _read_jsonl(tmp_path / "run" / "episode_metrics.jsonl")[0]
    action = _read_jsonl(tmp_path / "run" / "actions.jsonl")[0]
    latency = _read_jsonl(tmp_path / "run" / "latencies.jsonl")[0]

    for record in (metrics["metadata"], action, latency):
        assert record["profile_id"] == "L_eval_logging"
        assert record["profile_type"] == "rollout_profile"
        assert record["latency_scope"] == "end_to_end_action_ready"
        assert record["target_device"] == "target_gpu"
    assert action["sample_index"] is not None
    assert latency["sample_index"] is not None


def test_simulated_latency_records_distinguish_sampled_latency_and_action_age(tmp_path: Path):
    run_from_config(_config(tmp_path), write_summary=False)

    actions = _read_jsonl(tmp_path / "actions.jsonl")
    latencies = _read_jsonl(tmp_path / "latencies.jsonl")
    steps = _read_jsonl(tmp_path / "steps.jsonl")
    applied = [action for action in actions if action["action_applied_step"] is not None]

    assert latencies[0]["latency_ms_sampled"] == pytest.approx(80.0)
    assert latencies[0]["latency_ms_measured"] is None
    assert actions[0]["latency_ms_sampled"] == pytest.approx(80.0)
    assert actions[0]["latency_ms_measured"] is None
    assert actions[0]["effective_k"] == pytest.approx(2.0)
    assert applied
    assert applied[0]["action_age_frames"] == applied[0]["action_applied_step"] - applied[0]["obs_step"]
    assert steps[0]["info"]["scheduled_time_ms"] == pytest.approx(0.0)
    assert steps[0]["info"]["frame_time_ms"] == pytest.approx(40.0)
    assert "active_action_id" in steps[0]["info"]
