"""Long-lived Asterix reproduction contracts, owned by the harness experiment."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.adapters.dummy import DummyEnvAdapter
from latency_bench.envs.factory import build_env_adapter
from latency_bench.latency.samplers import build_sampler
from scripts.experiments.asterix_agent.prompt import format_prompt
from scripts.experiments.asterix_agent.run import episode_config, evaluate, freeze_config
from latency_bench.executors.factory import build_executor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.policy.base import ConstantPolicyRunner


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/asterix_agent/reproduce.yaml"


@pytest.mark.parametrize("seed", [2, 5, 6])
def test_raw_ale_trajectory_matches_benchmark_adapter(seed):
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    pytest.importorskip("stable_baselines3")
    config = load_config(CONFIG)
    actions = ActionResolver(config["env"])
    adapter = build_env_adapter(config, actions.noop_action)
    raw = gym.make("ALE/Asterix-v5", render_mode="rgb_array", **config["env"]["make_kwargs"]["base_make_kwargs"])
    try:
        observation, info = raw.reset(seed=seed)
        noops = int(raw.unwrapped.np_random.integers(1, 31))
        for _ in range(noops):
            observation, _, done, truncated, info = raw.step(0)
            if done or truncated:
                observation, info = raw.reset()
        adapter.reset(seed=seed)
        for frame in range(1000):
            np.testing.assert_array_equal(adapter.render_game_frame(), observation)
            if frame % 4 == 0:
                expected = Image.fromarray(observation).resize((224, 224), Image.Resampling.BILINEAR)
                actual = Image.fromarray(adapter.render_game_frame()).resize((224, 224), Image.Resampling.BILINEAR)
                np.testing.assert_array_equal(actual, expected)
            action = (frame // 7) % 9
            observation, reward, done, truncated, info = raw.step(action)
            result = adapter.step(actions.resolve(action))
            assert (result.reward, result.done, result.truncated) == (reward, done, truncated)
            assert result.info["lives"] == info["lives"]
            if done or truncated:
                break
    finally:
        adapter.close()
        raw.close()


def test_pooled_profile_preserves_original_seed_stream(tmp_path):
    distribution = {"spearman_rho": 0.5,
                    "observation_to_action_latency_ms": {"quantile_levels": [0, 1], "latency_ms": [700, 2000]},
                    "worker_service_time_ms": {"quantile_levels": [0, 1], "latency_ms": [700, 2000]}}
    (tmp_path / "distribution.json").write_text(json.dumps({"schema_version": 3, "worker_slots": {"0": {"all": distribution}}}))
    profile = {"distribution_path": "distribution.json", "model_id": "gpt-6.1-sol", "gpu_class": "api",
               "workload_id": "asterix", "instance_id": "pooled", "source_run_id": "original"}
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    config = load_config(CONFIG)
    config["latency"]["profile_path"] = str(tmp_path / "profile.json")
    clock = EnvClock(env_fps=7.5, obs_fps=1.875)
    for seed in config["evaluation"]["seeds"]:
        run = episode_config(config, tmp_path, seed)
        reference = deepcopy(run["latency"])
        reference.pop("profile_worker_slot")
        baseline = build_sampler(reference, clock=clock)
        pooled = build_sampler(run["latency"], clock=clock)
        assert run["latency"]["seed"] == 42 + seed
        for worker in range(8):
            assert pooled.sample({"worker_slot": worker}) == baseline.sample({"worker_slot": 0})


def test_frozen_prompt_uses_applied_commands_and_timestamped_images():
    snapshot = {"frame": 16, "images": [{"frame": frame} for frame in [0, 8, 16]],
                "current_plan": {"action": 2}, "executed_actions": [{"frame": 7, "action": 2}],
                "in_flight": [{"answer": "UNAVAILABLE"}]}
    text = format_prompt("STRATEGY\n", snapshot)
    assert text == ("STRATEGY\nCurrent observation frame: 16. Image frames: [0, 8, 16]. Current held action: 2.\n"
                    "Applied commands, oldest first, age_in_raw_frames:action: 9:2.\n")


@pytest.mark.parametrize("field", ["prompt_file", "model_instructions_file"])
def test_frozen_input_contents_are_checked_and_worker_paths_use_snapshots(tmp_path, field):
    """Retained preflight contract: path equality must not conceal changed worker text."""
    config = load_config(CONFIG)
    config["logging"]["output_dir"] = str(tmp_path / "output")
    for key in ("prompt_file", "model_instructions_file"):
        source = tmp_path / f"{key}.txt"
        source.write_text(f"ORIGINAL {key}")
        config["policy"][key] = str(source)
    freeze_config(config)
    worker = episode_config(config, tmp_path / "episode", 2)
    for key in ("prompt_file", "model_instructions_file"):
        assert Path(worker["policy"][key]).read_text() == f"ORIGINAL {key}"
        assert worker["policy"][key] != config["policy"][key]
    Path(config["policy"][field]).write_text("CHANGED AFTER PREFLIGHT")
    assert yaml.safe_load((tmp_path / "output/frozen_config.yaml").read_text()) == config
    with pytest.raises(ValueError, match=f"Frozen input changed after preflight: {field}"):
        evaluate(config)
    assert Path(episode_config(config, tmp_path / "later", 2)["policy"][field]).read_text() == f"ORIGINAL {field}"
    with pytest.raises(FileExistsError):
        freeze_config(config)
    assert Path(worker["policy"][field]).read_text() == f"ORIGINAL {field}"


def test_harness_factory_keeps_public_history_with_environment_action_history(tmp_path):
    """Retained factory contract: environment history cannot bypass the harness."""
    config = load_config(CONFIG)
    config["logging"]["output_dir"] = str(tmp_path)
    config["executor"]["inference_pool"] = False
    config["latency"] = {"method": "zero"}
    config["env"]["action_history_decisions"] = 8
    policy = ConstantPolicyRunner(Action(2))
    executor = build_executor(config, policy=policy,
                              env=DummyEnvAdapter(env_fps=7.5, noop_action=Action(0, is_noop=True)))
    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
        executor.run_episode(0, seed=2, max_env_steps=2)
    finally:
        executor.close()
    assert (tmp_path / "public/episode_000/request_000000.json").exists()
    with pytest.raises(ValueError, match="without env_backend"):
        build_executor(config, policy=policy, env_backend=object())
