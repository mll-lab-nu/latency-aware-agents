from __future__ import annotations

import json

import numpy as np
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Observation
from latency_bench.policy.demon_attack_ppo import DemonAttackPpoPolicyRunner
from latency_bench.policy.factory import build_policy
from modeling.demon_attack_ppo import (
    DemonAttackPpoAgent,
    DemonAttackPpoObservationConfig,
    save_demon_attack_ppo_checkpoint,
)


def _env_config() -> dict:
    return {
        "name": "demon_attack",
        "env_fps": 60,
        "noop_action": 0,
    }


def _config(checkpoint_path: str) -> dict:
    return {
        "experiment": {
            "name": "demon_attack_ppo_policy_test",
            "seed": 0,
        },
        "env": _env_config(),
        "policy": {
            "type": "demon_attack_ppo",
            "checkpoint_path": checkpoint_path,
            "device": "cpu",
            "deterministic": True,
        },
    }


def _write_checkpoint(tmp_path, observation_config: DemonAttackPpoObservationConfig | None = None):
    obs_config = observation_config or DemonAttackPpoObservationConfig(
        image_size=84,
        base_channels=1,
        stack_frames=4,
        obs_mode="image_only",
        frame_skip_values=(0,),
    )
    agent = DemonAttackPpoAgent(
        in_channels=obs_config.in_channels,
        image_size=obs_config.image_size,
        n_actions=6,
    )
    path = tmp_path / "demon_attack_ppo.pt"
    save_demon_attack_ppo_checkpoint(
        path=path,
        agent=agent,
        optimizer=None,
        observation_config=obs_config,
        global_step=123,
        train_config={"seed": 7},
    )
    return path, agent, obs_config


def test_demon_attack_ppo_policy_factory_builds_runner_with_numeric_actions(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = _config(str(path))

    policy = build_policy(config, ActionResolver(config["env"]))

    assert isinstance(policy, DemonAttackPpoPolicyRunner)
    assert sorted(policy.action_by_raw_id) == [0, 1, 2, 3, 4, 5]
    assert policy.action_by_raw_id[0].is_noop
    assert policy.action_by_raw_id[5].value == 5


def test_named_action_map_entries_override_numeric_fallback(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = _config(str(path))
    config["env"]["action_map"] = {
        "noop": 0,
        "fire": 1,
    }
    config["env"]["noop_action"] = "noop"
    config["policy"]["actions"] = ["noop", "fire"]

    policy = build_policy(config, ActionResolver(config["env"]))

    assert policy.action_by_raw_id[0].name == "noop"
    assert policy.action_by_raw_id[1].name == "fire"
    assert policy.action_by_raw_id[2].value == 2


def test_demon_attack_ppo_policy_runner_predicts_from_synthetic_observation(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = _config(str(path))
    policy = build_policy(config, ActionResolver(config["env"]))
    observation = Observation(
        data=np.zeros((84, 84), dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    output = policy.predict(observation)

    assert output.action.value in range(6)
    assert output.raw_output in range(6)
    assert output.metadata["policy_type"] == "demon_attack_ppo"
    json.dumps(output.metadata)


def test_demon_attack_policy_runner_matches_latency_channel_preprocessing(tmp_path):
    obs_config = DemonAttackPpoObservationConfig(
        image_size=84,
        base_channels=1,
        stack_frames=2,
        obs_mode="latency_channel",
        frame_skip_values=(0, 4),
    )
    path, _, _ = _write_checkpoint(tmp_path, observation_config=obs_config)
    policy = build_policy(_config(str(path)), ActionResolver(_env_config()))
    frame_a = np.full((84, 84), 10, dtype=np.uint8)
    frame_b = np.full((84, 84), 30, dtype=np.uint8)

    policy.predict(
        Observation(
            data=frame_a,
            env_step=0,
            sim_time_ms=0.0,
            metadata={"frame_skip": 0},
        )
    )
    policy.predict(
        Observation(
            data=frame_b,
            env_step=1,
            sim_time_ms=16.7,
            metadata={"frame_skip": 4},
        )
    )

    stacked = policy.current_stacked_observation

    assert stacked is not None
    assert stacked.shape == (84, 84, obs_config.in_channels)
    assert np.all(stacked[:, :, 0] == 10)
    assert np.all(stacked[:, :, 1] == 30)
    assert np.all(stacked[:, :, -1] == 255)


def test_demon_attack_zero_latency_config_builds_policy_layer(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = load_config("configs/experiments/demon_attack_ppo_zero_latency.yaml")
    config["policy"]["checkpoint_path"] = str(path)
    config["policy"]["device"] = "cpu"

    policy = build_policy(config, ActionResolver(config["env"]))

    assert isinstance(policy, DemonAttackPpoPolicyRunner)
