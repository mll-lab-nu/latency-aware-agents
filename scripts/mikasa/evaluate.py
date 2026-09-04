#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "starVLA"))
sys.path.insert(0, str(ROOT / "third_party" / "MIKASA-Robo"))

import mikasa_robo_suite.vla.memory_envs  # noqa: E402,F401
from baselines.ppo.ppo_memtasks import (  # noqa: E402
    AgentStateOnly,
    FlattenRGBDObservationWrapper,
)
from deployment.model_server.tools.websocket_policy_client import (  # noqa: E402
    WebsocketClientPolicy,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # noqa: E402
from mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets import (  # noqa: E402
    env_info,
)
from mikasa_robo_suite.vla.utils.apply_wrappers import (  # noqa: E402
    apply_mikasa_vla_wrappers,
)
from latency_bench.core.types import Action, Observation, StepResult  # noqa: E402
from latency_bench.envs.base import EnvAdapter  # noqa: E402


ENV_ID = "InterceptGrabFast-VLA-v0"
INSTRUCTION = "Intercept the rolling ball and grasp it to stop it."
START_SEED = 4242424242
MIKASA_IMAGE_VIEWS_INFO_KEY = "mikasa_image_views"


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].cpu().item()
    return np.asarray(value).reshape(-1)[0].item()


def _make_raw_env(obs_mode: str):
    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pose",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )


def _make_ppo_env():
    env = _make_raw_env("state")
    wrappers, _ = env_info(ENV_ID)
    for wrapper, kwargs in wrappers:
        env = wrapper(env, **kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=False, depth=False, state=True)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)


def _make_vla_env():
    return apply_mikasa_vla_wrappers(_make_raw_env("rgb"), include_overlays=False)


class _PpoPolicy:
    def __init__(self, env, checkpoint: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = AgentStateOnly(env).to(self.device)
        self.agent.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.agent.eval()

    def forward(self, observation):
        with torch.no_grad():
            return self.agent.get_action(
                {key: value.to(self.device) for key, value in observation.items()},
                deterministic=True,
            )


class _StarVlaPolicy:
    def __init__(self, host: str, port: int, unnorm_key: str | None):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.unnorm_key = unnorm_key

    def close(self) -> None:
        self.client.close()

    def forward(self, observation):
        rgb = observation["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        top = np.asarray(rgb[0, :, :, :3], dtype=np.uint8)
        wrist = np.asarray(rgb[0, :, :, 3:6], dtype=np.uint8)
        response = self.client.predict_action(
            {
                "examples": [{"image": [top, wrist], "lang": INSTRUCTION}],
                "unnorm_key": self.unnorm_key,
            }
        )
        return torch.as_tensor(
            response["data"]["actions"][0, 0], dtype=torch.float32
        ).unsqueeze(0)


class _MikasaProfileEnv(EnvAdapter):
    """Expose the one-environment MIKASA loop to the common realtime executor."""

    env_fps = 20.0
    noop_action = Action(value=[0.0] * 7, name="noop", is_noop=True)

    def __init__(self, env=None):
        self.env = _make_vla_env() if env is None else env
        self.env_step = 0
        self._observation = None
        self._success = False

    def reset(self, seed: int | None = None) -> Observation:
        self._observation, _ = self.env.reset(seed=seed)
        self.env_step = 0
        self._success = False
        return self.observe()

    def step(self, action: Action) -> StepResult:
        tensor_action = torch.as_tensor(action.value, dtype=torch.float32).reshape(1, 7)
        observation, reward, terminated, truncated, info = self.env.step(tensor_action)
        self._observation = observation
        self.env_step += 1
        success = bool(_scalar(info["success"]))
        self._success = self._success or success
        return StepResult(
            observation=None,
            reward=float(_scalar(reward)),
            done=bool(_scalar(terminated)),
            truncated=bool(_scalar(truncated)),
            info={
                "success": success,
                "task_metrics": {"success": float(self._success)},
            },
        )

    def observe(self) -> Observation:
        rgb = self._observation["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        views = np.stack(
            [
                np.asarray(rgb[0, :, :, :3], dtype=np.uint8),
                np.asarray(rgb[0, :, :, 3:6], dtype=np.uint8),
            ]
        )
        return Observation(
            data=None,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata={MIKASA_IMAGE_VIEWS_INFO_KEY: views, "slot_id": 0},
        )

    def render_frame(self):
        return self.observe().metadata[MIKASA_IMAGE_VIEWS_INFO_KEY][0]

    def render_game_frame(self):
        return self.render_frame()

    def close(self) -> None:
        self.env.close()


def _evaluate(policy_kind: str, args: argparse.Namespace) -> dict[str, Any]:
    env = _make_ppo_env() if policy_kind == "ppo" else _make_vla_env()
    policy = _PpoPolicy(env, args.checkpoint) if policy_kind == "ppo" else args.vla_policy
    seeds = []
    successes = []
    returns = []
    lengths = []
    try:
        for episode_index in range(args.episodes):
            seed = START_SEED + episode_index
            observation, _ = env.reset(seed=seed)
            success_once = False
            episode_return = 0.0
            for step in range(60):
                action = policy.forward(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                success_once = success_once or bool(_scalar(info["success"]))
                episode_return += float(_scalar(reward))
                if bool(_scalar(terminated)) or bool(_scalar(truncated)):
                    break
            seeds.append(seed)
            successes.append(success_once)
            returns.append(episode_return)
            lengths.append(step + 1)
    finally:
        env.close()
    summary = {
        "seeds": seeds,
        "successes": successes,
        "success_rate": float(np.mean(successes)),
        "returns": returns,
        "lengths": lengths,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _profile(argv: list[str]) -> None:
    from latency_bench.core.config import load_config
    from latency_bench.eval.driver import run_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.profile_config)
    run_from_config(
        config,
        env=_MikasaProfileEnv(),
        inference_devices=config["executor"]["inference_devices"],
    )


def main() -> None:
    if sys.argv[1:2] == ["profile"]:
        _profile(sys.argv[2:])
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("ppo", "vla"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.vla_policy = (
        _StarVlaPolicy(args.host, args.port, args.unnorm_key)
        if args.policy == "vla"
        else None
    )
    try:
        _evaluate(args.policy, args)
    finally:
        if args.vla_policy is not None:
            args.vla_policy.close()


if __name__ == "__main__":
    main()
