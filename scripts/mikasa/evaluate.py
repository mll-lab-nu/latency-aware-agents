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
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "MIKASA-Robo"))

from latency_bench.core.types import Action, Observation  # noqa: E402
from latency_bench.executors.gpu_batched_env_step_backend import (  # noqa: E402
    GpuBatchedEnvStepBackendBase,
    SlotStepOutcome,
)


ENV_ID = "InterceptGrabFast-VLA-v0"
INSTRUCTION = "Intercept the rolling ball and grasp it to stop it."
START_SEED = 4242424242
MIKASA_IMAGE_VIEWS_INFO_KEY = "mikasa_image_views"
MIKASA_STATE_INFO_KEY = "mikasa_proprio"


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].cpu().item()
    return np.asarray(value).reshape(-1)[0].item()


def _make_raw_env(
    obs_mode: str,
    num_envs: int = 1,
    simulator_device: str = "gpu",
):
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401

    return gym.make(
        ENV_ID,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pose",
        render_mode="all",
        sim_backend=simulator_device,
        render_backend=simulator_device,
        reward_mode="normalized_dense",
    )


def _make_ppo_env():
    from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets import (
        env_info,
    )

    env = _make_raw_env("state")
    wrappers, _ = env_info(ENV_ID)
    for wrapper, kwargs in wrappers:
        env = wrapper(env, **kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=False, depth=False, state=True)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)


def _make_vla_env(num_envs: int = 1, simulator_device: str = "gpu"):
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    return apply_mikasa_vla_wrappers(
        _make_raw_env(
            "rgb",
            num_envs=num_envs,
            simulator_device=simulator_device,
        ),
        include_overlays=False,
    )


class _PpoPolicy:
    def __init__(self, env, checkpoint: Path):
        from baselines.ppo.ppo_memtasks import AgentStateOnly

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


class MikasaEnvStepBackend(GpuBatchedEnvStepBackendBase):
    """Own the native MIKASA simulator and its 7D action contract."""

    backend_name = "mikasa_gpu_batched"

    def __init__(self, *, config: dict[str, Any], num_slots: int, env=None):
        noop_action = Action(
            value=np.asarray(config["env"]["noop_action"], dtype=np.float32),
            name="noop",
            is_noop=True,
        )
        super().__init__(
            config=config,
            noop_action=noop_action,
            num_slots=num_slots,
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
        )
        self.env = (
            _make_vla_env(
                num_envs=num_slots,
                simulator_device=config["env"]["simulator_device"],
            )
            if env is None
            else env
        )
        self._episode_seeds = [0] * num_slots
        self._success = np.zeros(num_slots, dtype=np.bool_)
        self._observation, _ = self.env.reset(seed=self._episode_seeds)

    def _reset_slot_observation(self, slot_id: int, *, seed: int | None) -> Observation:
        if seed is not None:
            self._episode_seeds[slot_id] = int(seed)
        env_idx = torch.tensor([slot_id], device=self.env.unwrapped.device)
        self._observation, _ = self.env.reset(
            seed=self._episode_seeds,
            options={"env_idx": env_idx},
        )
        self._env_steps[slot_id] = 0
        self._success[slot_id] = False
        return self._observation_for_slot(slot_id)

    def _observe_slot_observations(
        self,
        slot_ids: list[int],
    ) -> dict[int, Observation]:
        return {slot_id: self._observation_for_slot(slot_id) for slot_id in slot_ids}

    def _step_cores(
        self,
        slot_ids: list[int],
        *,
        actions: np.ndarray,
        active_mask: np.ndarray,
    ) -> Any:
        del slot_ids, active_mask
        tensor_actions = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.env.unwrapped.device,
        )
        self._observation, reward, terminated, truncated, info = self.env.step(
            tensor_actions
        )
        return reward, terminated, truncated, info

    def _slot_step_outcome(self, state: Any, slot_id: int) -> SlotStepOutcome:
        reward, terminated, truncated, info = state
        success = bool(_slot_value(info["success"], slot_id))
        self._success[slot_id] |= success
        return SlotStepOutcome(
            reward=float(_slot_value(reward, slot_id)),
            done=bool(_slot_value(terminated, slot_id)),
            truncated=bool(_slot_value(truncated, slot_id)),
            info={
                "success": success,
                "task_metrics": {"success": float(self._success[slot_id])},
            },
        )

    def _observation_for_slot(self, slot_id: int) -> Observation:
        rgb = self._observation["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        views = np.stack(
            [
                np.asarray(rgb[slot_id, :, :, :3], dtype=np.uint8),
                np.asarray(rgb[slot_id, :, :, 3:6], dtype=np.uint8),
            ]
        )
        return Observation(
            data=None,
            env_step=int(self._env_steps[slot_id]),
            sim_time_ms=float(self._env_steps[slot_id]) * self._frame_ms,
            metadata={
                MIKASA_IMAGE_VIEWS_INFO_KEY: views,
                MIKASA_STATE_INFO_KEY: self._observation["proprio"][slot_id].detach().cpu().numpy(),
                "slot_id": slot_id,
            },
        )

    def close(self) -> None:
        if not self.closed:
            self.env.close()
        super().close()


def _slot_value(value: Any, slot_id: int) -> Any:
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[slot_id].cpu().item()
    return np.asarray(value).reshape(-1)[slot_id].item()


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    env = _make_ppo_env()
    policy = _PpoPolicy(env, args.checkpoint)
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


def _latency_eval(argv: list[str]) -> None:
    from latency_bench.core.config import load_config
    from latency_bench.eval.config import apply_runtime_overrides
    from latency_bench.eval.driver import run_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--model-config-path", type=Path)
    parser.add_argument("--task-contract-path", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--latency-method", choices=("zero", "temporal"))
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument("--latency-seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.eval_config)
    apply_runtime_overrides(
        config,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        task_contract_path=args.task_contract_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
        latency_method=args.latency_method,
        latency_profile_path=args.profile_path,
        latency_seed=args.latency_seed,
    )
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    backend = MikasaEnvStepBackend(
        config=config,
        num_slots=int(config["evaluation"]["eval_parallel_envs"]),
    )
    run_from_config(
        config,
        env_backend=backend,
        inference_devices=config["executor"]["inference_devices"],
    )


def main() -> None:
    if sys.argv[1:2] == ["latency-eval"]:
        _latency_eval(sys.argv[2:])
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("ppo",), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _evaluate(args)


if __name__ == "__main__":
    main()
