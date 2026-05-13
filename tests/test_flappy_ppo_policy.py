from __future__ import annotations

import json

import numpy as np
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Observation
from latency_bench.executors.factory import build_executor
from latency_bench.policy.flappy_ppo import FlappyPpoPolicyRunner
from latency_bench.policy.factory import build_policy
from modeling.flappy_ppo import (
    FlappyPpoAgent,
    FlappyPpoObservationConfig,
    FlappyPpoTorchObservationPipeline,
    build_agent_from_checkpoint,
    load_flappy_ppo_checkpoint,
    observation_config_from_checkpoint,
    save_flappy_ppo_checkpoint,
)


def _env_config() -> dict:
    return {
        "name": "flappy",
        "env_fps": 30,
        "noop_action": "noop",
        "action_map": {
            "noop": 0,
            "flap": 1,
        },
        "oneshot_actions": ["flap"],
    }


def _config(checkpoint_path: str) -> dict:
    return {
        "experiment": {
            "name": "flappy_ppo_policy_test",
            "seed": 0,
        },
        "env": _env_config(),
        "policy": {
            "type": "flappy_ppo",
            "checkpoint_path": checkpoint_path,
            "device": "cpu",
            "deterministic": True,
            "actions": ["noop", "flap"],
        },
    }


def _write_checkpoint(tmp_path, observation_config: FlappyPpoObservationConfig | None = None):
    obs_config = observation_config or FlappyPpoObservationConfig(
        image_size=84,
        grayscale=False,
        stack_frames=4,
    )
    agent = FlappyPpoAgent(
        in_channels=obs_config.in_channels,
        image_size=obs_config.image_size,
        n_actions=2,
    )
    path = tmp_path / "flappy_ppo.pt"
    save_flappy_ppo_checkpoint(
        path=path,
        agent=agent,
        optimizer=None,
        observation_config=obs_config,
        global_step=123,
        train_config={"seed": 7},
    )
    return path, agent, obs_config


def test_checkpoint_save_load_restores_model_and_observation_config(tmp_path):
    path, agent, obs_config = _write_checkpoint(tmp_path)

    checkpoint = load_flappy_ppo_checkpoint(path, map_location="cpu")
    loaded_config = observation_config_from_checkpoint(checkpoint)
    loaded_agent = build_agent_from_checkpoint(checkpoint, device="cpu")

    assert checkpoint["format"] == "flappy_ppo"
    assert checkpoint["global_step"] == 123
    assert loaded_config.to_dict() == obs_config.to_dict()
    for key, value in agent.state_dict().items():
        assert torch.equal(value, loaded_agent.state_dict()[key])


def test_flappy_ppo_policy_factory_builds_runner(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = _config(str(path))

    policy = build_policy(config, ActionResolver(config["env"]))

    assert isinstance(policy, FlappyPpoPolicyRunner)
    assert policy.action_by_raw_id[1].name == "flap"
    assert policy.action_by_raw_id[1].is_oneshot


def test_flappy_ppo_policy_runner_predict_returns_noop_or_flap(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = _config(str(path))
    policy = build_policy(config, ActionResolver(config["env"]))
    observation = Observation(
        data=np.zeros((84, 84, 3), dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    output = policy.predict(observation)

    assert output.action.name in {"noop", "flap"}
    assert output.action.value in {0, 1}
    assert output.raw_output in {0, 1}
    json.dumps(output.metadata)


def test_flappy_ppo_policy_runner_resets_frame_stack_between_episodes(tmp_path):
    obs_config = FlappyPpoObservationConfig(
        image_size=84,
        grayscale=False,
        stack_frames=4,
    )
    path, _, _ = _write_checkpoint(tmp_path, observation_config=obs_config)
    policy = build_policy(_config(str(path)), ActionResolver(_env_config()))

    first_episode_frame = np.full((84, 84, 3), 10, dtype=np.uint8)
    second_episode_frame = np.full((84, 84, 3), 200, dtype=np.uint8)
    policy.predict(Observation(data=first_episode_frame, env_step=0, sim_time_ms=0.0))
    policy.predict(Observation(data=first_episode_frame, env_step=1, sim_time_ms=33.3))
    policy.predict(Observation(data=second_episode_frame, env_step=0, sim_time_ms=0.0))

    stacked = policy.current_stacked_observation

    assert stacked is not None
    assert stacked.shape == (84, 84, obs_config.in_channels)
    assert np.all(stacked == 200)


def test_flappy_ppo_policy_runner_matches_shared_raw_rgb_preprocessing(tmp_path):
    obs_config = FlappyPpoObservationConfig(
        image_size=84,
        grayscale=False,
        stack_frames=2,
    )
    path, _, _ = _write_checkpoint(tmp_path, observation_config=obs_config)
    policy = build_policy(_config(str(path)), ActionResolver(_env_config()))
    shared = FlappyPpoTorchObservationPipeline(obs_config, n_envs=1, device="cpu")
    frame_a = np.full((84, 84, 3), 10, dtype=np.uint8)
    frame_b = np.full((84, 84, 3), 30, dtype=np.uint8)

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
            sim_time_ms=33.3,
            metadata={"frame_skip": 4},
        )
    )
    shared.reset(torch.as_tensor(frame_a).unsqueeze(0), frame_skips=[0])
    shared.step(
        torch.as_tensor(frame_b).unsqueeze(0),
        dones=np.asarray([False], dtype=np.bool_),
        frame_skips=[4],
    )

    assert np.array_equal(policy.current_stacked_observation, shared.current_observation[0])


def test_flappy_ppo_zero_latency_config_builds_executor_to_policy_layer(tmp_path):
    path, _, _ = _write_checkpoint(tmp_path)
    config = load_config("configs/experiments/flappy_ppo_zero_latency.yaml")
    config["policy"]["checkpoint_path"] = str(path)
    config["policy"]["device"] = "cpu"
    config["env"]["gpu_render_device"] = "cpu"
    config["logging"]["output_dir"] = str(tmp_path / "logs")

    executor = build_executor(config)

    try:
        assert isinstance(executor.policy, FlappyPpoPolicyRunner)
    finally:
        executor.env.close()
