from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.types import Action, EpisodeMetrics, StepRecord, StepResult
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.eval import driver
from latency_bench.latency.burst_profile import (
    build_hidden_regime_burst_model,
    build_worker_slot_distributions_from_sessions,
)
from latency_bench.latency.profile import save_latency_distribution, save_profile_manifest
from latency_bench.eval.driver import run_from_config
from latency_bench.eval.reporting import _summary_latency_label
from latency_bench.logging.metrics import compute_episode_metrics
from latency_bench.logging.summary import build_queue_eval_summary
from latency_bench.logging.wandb_eval import eval_scalar_payload
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
            "method": "fixed",
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
        },
        "logging": {
            "output_dir": str(tmp_path),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def test_calibration_stops_after_complete_episodes_with_enough_admitted_samples(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["evaluation"].update(eval_episodes=500, calibration_samples=5)

    class ProbeExecutor:
        logger = None

        def run_warmup(self):
            return {}

        def run_episode(self, episode_id, seed, eval_max_steps):
            return EpisodeMetrics(
                episode_id=episode_id, episode_return=1, survival_steps=3,
                episode_return_env=1, metadata={"submitted_observation_frames": 3},
            )

        def close(self):
            pass

    monkeypatch.setattr(driver, "build_executor", lambda *a, **kw: ProbeExecutor())
    result = run_from_config(config, write_summary=False)
    assert [episode.episode_id for episode in result] == [0, 1]


def _write_profile(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    burst_samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "worker_id": index % 2,
            "observation_to_action_latency_ms": float(frames) * 40.0,
            "worker_service_time_ms": float(frames) * 40.0,
            "worker_service_time_ms_bin": frames * 40,
        }
        for index, frames in enumerate([1, 1, 1, 1, 1, 1])
    ]
    burst_model = build_hidden_regime_burst_model(burst_samples)
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(burst_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_latency_distribution(
        build_worker_slot_distributions_from_sessions(
            [burst_samples],
            spike_threshold_ms_by_worker_slot=burst_model[
                "spike_threshold_ms_by_worker_slot"
            ],
        ),
        profile_dir / "latency_distribution.json",
    )
    save_profile_manifest(
        {
            "model_id": "openvla",
            "gpu_class": "2x-test-gpu",
            "workload_id": "flappy",
            "instance_id": "instance_test",
            "source_run_id": "run_test",
            "env_fps": 25.0,
            "frame_ms": 40.0,
            "n_samples": 6,
            "distribution_path": "latency_distribution.json",
            "latency_method": "temporal",
            "sample_model_type": "hidden_regime",
            "burst_model_path": "latency_burst_model.json",
            "summary": {},
        },
        profile_dir / "profile.json",
    )
    return profile_dir


def test_iid_profile_with_null_fixed_latency_uses_profile_label():
    config = {
        "executor": {"mode": "simulated"},
        "latency": {"method": "iid", "fixed_latency_ms": None},
    }

    assert _summary_latency_label(config) == "profile_sample"


@pytest.mark.parametrize("injected_name", ["env", "env_backend"])
def test_injected_env_skips_starvla_env_preparation(
    tmp_path: Path,
    monkeypatch,
    injected_name: str,
):
    import latency_bench.eval.driver as driver
    from latency_bench.policy import starvla

    class Executor:
        logger = None

        def run_warmup(self):
            return {}

        def close(self):
            return None

    injected_env = object()
    captured = {}
    monkeypatch.setattr(
        starvla,
        "prepare_starvla_checkpoint_input_config",
        lambda _config: pytest.fail("an injected environment is already constructed"),
    )
    monkeypatch.setattr(
        driver,
        "build_executor",
        lambda config, **kwargs: captured.update(kwargs) or Executor(),
    )
    config = {
        "experiment": {"name": "native-profile", "seed": 0},
        "executor": {"mode": "simulated"},
        "policy": {"type": "starvla"},
        "evaluation": {"eval_episodes": 0, "eval_max_steps": 1},
        "logging": {"output_dir": str(tmp_path)},
    }

    run_from_config(
        config,
        write_summary=False,
        inference_devices=["cuda:1"],
        **{injected_name: injected_env},
    )

    assert captured[injected_name] is injected_env
    assert captured["inference_devices"] == ["cuda:1"]


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


def test_summary_keeps_new_model_identity_separate_from_source_profile(tmp_path):
    from latency_bench.policy.base import ConstantPolicyRunner

    profile = _write_profile(tmp_path)
    config = _config(tmp_path / "evaluation", latency={"method": "iid", "profile_path": str(profile)})
    config["policy"] = {"type": "starvla", "checkpoint_path": "/new-vla/checkpoint-2"}
    run_from_config(config, policy=ConstantPolicyRunner(Action("jump")))
    summary = json.loads((tmp_path / "evaluation/queue_eval_latency_profile_sample.json").read_text())
    assert summary["checkpoint_path"] == "/new-vla/checkpoint-2"
    assert summary["source_profile_path"] == str(profile)


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
            "method": "temporal",
            "profile_path": str(profile_dir),
            "seed": 0,
        },
    )
    run_from_config(config, write_summary=False)

    metrics = load_jsonl_records(tmp_path / "run" / "episode_metrics.jsonl")[0]
    action = load_jsonl_records(tmp_path / "run" / "actions.jsonl")[0]
    latency = load_jsonl_records(tmp_path / "run" / "latencies.jsonl")[0]

    for record in (metrics["metadata"], action["latency"], latency):
        assert record["model_id"] == "openvla"
        assert record["gpu_class"] == "2x-test-gpu"
        assert record["instance_id"] == "instance_test"


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


@pytest.mark.parametrize("latency", [0, "profile_sample"])
def test_task_metrics_flow_from_final_step_into_summary(latency):
    metrics = [
        compute_episode_metrics(
            episode_id=episode_id,
            step_records=[
                StepRecord(
                    episode_id=episode_id,
                    env_step=0,
                    sim_time_ms=0.0,
                    active_action=Action("noop"),
                    reward=1.0,
                    done=True,
                    info={"task_metrics": {"success": success}},
                )
            ],
            action_events=[],
            latency_values_ms=[],
        )
        for episode_id, success in enumerate((1.0, 0.0))
    ]

    summary = build_queue_eval_summary(
        metrics,
        experiment_name="task-metrics",
        seed=0,
        latency=latency,
    )

    assert metrics[0].metadata["task_metrics"] == {"success": 1.0}
    assert summary["task_metrics"] == {
        "success": {"mean": 0.5, "std": 0.5, "values": [1.0, 0.0]}
    }
    scalars = eval_scalar_payload([summary], env_steps=0, prefix="eval")
    metric_prefix = (
        "eval/fixed/latency_0"
        if latency == 0
        else "eval/profile_sample/profile_sample"
    )
    assert scalars[f"{metric_prefix}/completed_episodes"] == 2.0
    assert scalars[f"{metric_prefix}/task_metrics/success/mean"] == 0.5
    assert scalars[f"{metric_prefix}/task_metrics/success/std"] == 0.5


def test_eval_scalar_payload_requires_returns_for_completed_episode_count():
    with pytest.raises(KeyError, match="returns"):
        eval_scalar_payload(
            [
                {
                    "suite_name": "fixed_0",
                    "latency": 0,
                    "mean_return": 0.0,
                    "std_return": 0.0,
                    "mean_length": 0.0,
                }
            ],
            env_steps=0,
            prefix="eval",
        )


def test_task_metrics_are_required_for_every_episode():
    metrics = [
        EpisodeMetrics(
            episode_id=0,
            episode_return=0.0,
            survival_steps=1,
            metadata={"task_metrics": {"success": 1.0}},
        ),
        EpisodeMetrics(episode_id=1, episode_return=0.0, survival_steps=1),
    ]

    with pytest.raises(KeyError, match="task_metrics"):
        build_queue_eval_summary(
            metrics,
            experiment_name="task-metrics",
            seed=0,
            latency="measured",
        )


def test_task_metric_moments_preserve_step_weighted_statistics():
    metrics = [
        EpisodeMetrics(
            episode_id=0,
            episode_return=0.0,
            survival_steps=2,
            metadata={
                "task_metric_moments": {
                    "edge_violation": {"count": 2, "sum": 1.0, "sum_sq": 1.0}
                }
            },
        ),
        EpisodeMetrics(
            episode_id=1,
            episode_return=0.0,
            survival_steps=1,
            metadata={
                "task_metric_moments": {
                    "edge_violation": {"count": 1, "sum": 1.0, "sum_sq": 1.0}
                }
            },
        ),
    ]

    summary = build_queue_eval_summary(
        metrics,
        experiment_name="evaluation",
        seed=2,
        latency="zero",
    )

    assert summary["task_metrics"]["edge_violation"] == {
        "mean": pytest.approx(2 / 3),
        "std": pytest.approx(2**0.5 / 3),
        "count": 3,
    }
