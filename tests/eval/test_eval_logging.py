from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.types import Action, StepResult
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.latency.burst_profile import build_hidden_regime_burst_model
from latency_bench.latency.profile import build_inverse_cdf_distribution, save_latency_distribution, save_profile_manifest, write_latency_samples
from latency_bench.run import run_from_config
from latency_bench.utils.io import load_jsonl_records


@register_env_adapter("raw_return_test")
def _build_raw_return_env(config: dict, noop_action: Action) -> EnvAdapter:
    class RawReturnEnv(DummyEnvAdapter):
        def step(self, action: Action) -> StepResult:
            result = super().step(action)
            info = dict(result.info)
            if result.done or result.truncated:
                info.update(
                    {
                        "episodic_raw_return": 123.0,
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
    return RawReturnEnv(
        env_fps=float(env_cfg["env_fps"]),
        max_steps=int(evaluation_cfg["eval_max_steps"]),
        noop_action=noop_action,
    )


def _config(tmp_path: Path, *, latency: dict | None = None, env_name: str = "dummy") -> dict:
    return {
        "experiment": {
            "name": "eval_logging_test",
            "seed": 7,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": env_name,
            "env_fps": 25,
            "obs_fps": 25,
            "noop_action": "noop",
            "action_map": {
                "noop": "noop",
                "jump": "jump",
            },
        },
        "latency": latency or {
            "type": "fixed",
            "fixed_latency_ms": 80.0,
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "issue_order_fifo",
        },
        "policy": {
            "type": "constant",
            "action": "jump",
        },
        "evaluation": {
            "eval_episodes": 1,
            "eval_max_steps": 3,
            "replay_host": "test_replay_host",
        },
        "logging": {
            "output_dir": str(tmp_path),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def _write_profile(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    write_latency_samples([10.0, 20.0, 30.0], profile_dir / "latencies.jsonl")
    save_latency_distribution(
        build_inverse_cdf_distribution([10.0, 20.0, 30.0]),
        profile_dir / "latency_distribution.json",
    )
    burst_samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "action_ready_ms": float(frames) * 40.0,
            "worker_service_ms": float(frames) * 40.0,
            "worker_service_ms_bin": frames * 40,
        }
        for index, frames in enumerate([1, 2, 3])
    ]
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(build_hidden_regime_burst_model(burst_samples), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_profile_manifest(
        {
            "profile_id": "L_eval_logging",
            "policy_id": "pi_test",
            "env_id": "env_test",
            "target_device": "target_gpu",
            "env_fps": 25.0,
            "frame_ms": 40.0,
            "n_samples": 3,
            "trace_path": "latencies.jsonl",
            "distribution_path": "latency_distribution.json",
            "latency_method": "temporal",
            "sample_model_type": "hidden_regime",
            "burst_model_path": "latency_burst_model.json",
            "summary": {},
        },
        profile_dir / "profile.json",
    )
    return profile_dir


def test_episode_metrics_log_evaluation_raw_fact_metadata(tmp_path: Path):
    run_from_config(_config(tmp_path), write_summary=False)

    metrics = load_jsonl_records(tmp_path / "episode_metrics.jsonl")[0]

    assert metrics["episode_return_env"] == pytest.approx(3.0)
    assert metrics["episode_return"] == pytest.approx(3.0)
    assert metrics["return_raw"] is None
    assert "return_clipped" not in metrics
    assert metrics["metadata"]["mode"] == "simulated"
    assert metrics["metadata"]["episode_seed"] == 7
    assert metrics["metadata"]["policy_id"] == "constant"
    assert metrics["metadata"]["env_id"] == "dummy"
    assert metrics["metadata"]["latency_type"] == "fixed"
    assert metrics["metadata"]["env_fps"] == pytest.approx(25.0)
    assert metrics["metadata"]["obs_fps"] == pytest.approx(25.0)
    assert metrics["metadata"]["frame_ms"] == pytest.approx(40.0)
    assert metrics["metadata"]["replay_host"] == "test_replay_host"


def test_return_raw_is_only_from_explicit_info(tmp_path: Path):
    config = _config(tmp_path, env_name="raw_return_test")
    run_from_config(config, write_summary=False)

    metrics = load_jsonl_records(tmp_path / "episode_metrics.jsonl")[0]

    assert metrics["episode_return_env"] == pytest.approx(3.0)
    assert metrics["episode_return"] == pytest.approx(3.0)
    assert metrics["return_raw"] == pytest.approx(123.0)
    assert "return_clipped" not in metrics


def test_empirical_profile_metadata_is_logged(tmp_path: Path):
    profile_dir = _write_profile(tmp_path)
    config = _config(
        tmp_path / "run",
        latency={
            "type": "profile_sample",
            "profile_path": str(profile_dir),
            "seed": 0,
        },
    )
    run_from_config(config, write_summary=False)

    metrics = load_jsonl_records(tmp_path / "run" / "episode_metrics.jsonl")[0]
    action = load_jsonl_records(tmp_path / "run" / "actions.jsonl")[0]
    latency = load_jsonl_records(tmp_path / "run" / "latencies.jsonl")[0]

    for record in (metrics["metadata"], action["latency"], latency):
        assert record["profile_id"] == "L_eval_logging"
        assert record["target_device"] == "target_gpu"


def test_simulated_latency_records_log_latency_raw_fields(tmp_path: Path):
    run_from_config(_config(tmp_path), write_summary=False)

    actions = load_jsonl_records(tmp_path / "actions.jsonl")
    latencies = load_jsonl_records(tmp_path / "latencies.jsonl")
    steps = load_jsonl_records(tmp_path / "steps.jsonl")
    applied = [action for action in actions if action["apply_raw_frame"] is not None]

    assert latencies[0]["latency_ms"] == pytest.approx(80.0)
    assert actions[0]["latency"]["latency_ms"] == pytest.approx(80.0)
    assert actions[0]["latency"]["obs_id"] == 0
    assert actions[0]["obs_env_step"] == 0
    assert actions[0]["latency_raw_frames"] == 2
    assert actions[0]["ready_raw_frame"] == 2
    assert applied
    assert applied[0]["apply_raw_frame"] == 2
    assert steps[0]["info"]["scheduled_time_ms"] == pytest.approx(0.0)
    assert steps[0]["info"]["frame_time_ms"] == pytest.approx(40.0)
    assert "active_action_id" in steps[0]["info"]
