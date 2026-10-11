"""Named evaluation conditions and checkpoint reuse are run-level contracts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from latency_bench.core.config import resolve_runtime_config
from latency_bench.core.types import EpisodeMetrics
from latency_bench.eval.suites import has_eval_suites, parse_eval_suites


def _config(tmp_path):
    return {
        "experiment": {"name": "conditions", "seed": 7},
        "executor": {"mode": "simulated"},
        "env": {
            "name": "dummy",
            "env_fps": 30,
            "obs_fps": 30,
            "noop_action": "noop",
            "action_map": {"noop": 0, "jump": 1},
        },
        "latency": {"method": "zero"},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "policy": {"type": "constant", "action": "jump"},
        "evaluation": {"episodes": 1, "max_env_steps": 2, "suites": []},
        "logging": {"output_dir": str(tmp_path)},
    }


@pytest.mark.parametrize("raw_reward", [None, True])
def test_eval_raw_reward_remains_opt_in_when_resolving_config(tmp_path, raw_reward):
    config = _config(tmp_path)
    if raw_reward is not None:
        config["evaluation"]["eval_raw_reward"] = raw_reward
    resolve_runtime_config(config)
    assert config["evaluation"]["eval_raw_reward"] is (raw_reward is True)


def test_named_conditions_use_single_run_latency_units(tmp_path):
    config = _config(tmp_path)
    conditions = [
        {"name": "zero", "latency": {"method": "zero"}},
        {"name": "slow", "latency": {"method": "fixed", "fixed_latency_ms": 100}},
        {
            "name": "normal",
            "latency": {
                "method": "normal",
                "mean_latency_ms": 50,
                "std_latency_ms": 10,
                "min_latency_ms": 0,
                "max_latency_ms": 100,
                "seed": 42,
            },
        },
    ]
    config["evaluation"]["suites"] = conditions
    assert parse_eval_suites(config) == conditions
    assert has_eval_suites(config)


@pytest.mark.parametrize("names", [["same", "same"], ["../outside"], ["."]])
def test_suite_output_names_are_unique_path_segments(tmp_path, names):
    config = _config(tmp_path)
    config["evaluation"]["suites"] = [
        {"name": name, "latency": {"method": "zero"}} for name in names
    ]
    with pytest.raises(ValueError):
        parse_eval_suites(config)


@pytest.mark.parametrize("mode", ["realtime", "paused"])
def test_executor_timing_does_not_read_simulated_conditions(tmp_path, mode):
    config = _config(tmp_path)
    config["executor"]["mode"] = mode
    config["evaluation"]["suites"] = object()
    assert not has_eval_suites(config)


@pytest.mark.parametrize("method", ["iid", "temporal"])
def test_profile_condition_preserves_seed_and_output_summary(
    monkeypatch, tmp_path, method
):
    import latency_bench.eval.sweeps as sweeps

    config = _config(tmp_path)
    latency = {
        "method": method,
        "profile_path": str(tmp_path / "profile.json"),
        "seed": 42,
    }
    config["evaluation"]["suites"] = [{"name": "rtx_3090", "latency": latency}]
    seen = []

    def run(config, *, extra_metadata, **kwargs):
        seen.append(deepcopy(config))
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10,
                survival_steps=5,
                metadata=extra_metadata,
            )
        ]

    monkeypatch.setattr(sweeps, "run_from_config", run)
    summaries = sweeps.run_eval_suites_from_config(config)
    assert seen[0]["latency"] == latency
    assert seen[0]["evaluation"]["suites"] == []
    assert summaries[0]["latency_seed"] == 42
    path = tmp_path / "rtx_3090/queue_eval_latency_rtx_3090.json"
    assert json.loads(path.read_text())["latency_seed"] == 42


@pytest.mark.parametrize("policy_type", ["sample_factory", "starvla"])
def test_checkpoint_eval_suites_build_policy_once_and_reuse_it(
    monkeypatch, tmp_path, policy_type
):
    import latency_bench.eval.sweeps as sweeps

    config = _config(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config["policy"] = {
        "type": policy_type,
        "source": "local",
        "root": str(tmp_path),
        "checkpoint": "model.pt",
    }
    config["evaluation"]["suites"] = [
        {"name": name, "latency": {"method": "zero"}} for name in ("a", "b")
    ]
    built, used = [], []
    sentinel = object()
    monkeypatch.setattr(
        sweeps, "build_policy", lambda *args: built.append(sentinel) or sentinel
    )
    if policy_type == "starvla":
        import latency_bench.policy.starvla as starvla

        monkeypatch.setattr(
            starvla, "prepare_starvla_checkpoint_input_config", lambda config: None
        )

    def run(config, *, policy, **kwargs):
        used.append(policy)
        return [EpisodeMetrics(episode_id=0, episode_return=1, survival_steps=1)]

    monkeypatch.setattr(sweeps, "run_from_config", run)
    sweeps.run_eval_suites_from_config(config)
    assert built == [sentinel]
    assert used == [sentinel, sentinel]
    built.clear()
    injected = object()
    sweeps.run_eval_suites_from_config(config, policy=injected)
    assert built == []
    assert used[-2:] == [injected, injected]
