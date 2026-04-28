#!/usr/bin/env python3
"""
CleanRL-style PPO for FlappyBird-v0 with queue/scheduled-action latency.

This is intentionally structured like cleanrl_ppo_demon_attack.py, but uses the
Flappy Bird image-rendering pipeline from the SB3 script:
  FlappyBird-v0 numeric env -> QueueLatencyWrapper -> RGB/gray render -> NCHW frame stack.

Latency convention:
  For queue latency L, action chosen at decision step t is scheduled to become
  the active environment action at env step t + L. Until then, the most recently
  matured action is executed, initialized to NOOP/action 0.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from bisect import insort_right
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from torch.distributions.categorical import Categorical

import flappy_bird_gymnasium  # noqa: F401 registers FlappyBird-v0


# =========================
# Helpers
# =========================


def parse_int_list(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def linear_schedule(initial: float, final: float, progress_remaining: float) -> float:
    return float(final) + float(progress_remaining) * (float(initial) - float(final))


def _unwrap_attr(env: gym.Env, attr_name: str, default: Any) -> Any:
    cur: Optional[gym.Env] = env
    while cur is not None:
        if hasattr(cur, attr_name):
            return getattr(cur, attr_name)
        cur = getattr(cur, "env", None)
    return default


def serialize_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): serialize_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# =========================
# Queue latency + image wrappers
# =========================


class QueueLatencyWrapper(gym.Wrapper):
    """
    Queue/scheduled-action latency wrapper for true input delay.

    Every step the policy outputs an action, but that action is scheduled to
    become active `current_frame_skip` env decisions later. The environment
    executes the most recently matured action, starting from NOOP/action 0.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip_values: List[int],
        latency_mode: str = "episode_fixed",
        latency_sampling_mode: str = "uniform_all",
        episodes_per_stage: int = 10,
        randomize_within_stage: bool = False,
        within_episode_change_interval: int = 1,
        within_episode_change_prob: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(env)
        assert len(frame_skip_values) > 0, "frame_skip_values cannot be empty"

        if latency_mode == "within_episode_change":
            latency_mode = "within_episode_random"
        assert latency_mode in {"episode_fixed", "within_episode_random", "within_episode_curriculum"}
        assert latency_sampling_mode in {"uniform_all", "curriculum"}

        self.frame_skip_values = [max(0, int(x)) for x in frame_skip_values]
        self.latency_mode = str(latency_mode)
        self.latency_sampling_mode = str(latency_sampling_mode)
        self.episodes_per_stage = max(1, int(episodes_per_stage))
        self.randomize_within_stage = bool(randomize_within_stage)
        self.within_episode_change_interval = max(1, int(within_episode_change_interval))
        self.within_episode_change_prob = float(within_episode_change_prob)

        self.rng = np.random.RandomState(seed)
        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[Tuple[int, int, int, int]] = []
        self._schedule_seq = 0

    def _get_stage_index(self) -> int:
        stage = self.episode_idx // self.episodes_per_stage
        return min(stage, len(self.frame_skip_values) - 1)

    def _sample_frame_skip(self) -> int:
        if self.latency_sampling_mode == "uniform_all":
            return int(self.rng.choice(self.frame_skip_values))
        stage = self._get_stage_index()
        if self.randomize_within_stage:
            candidates = self.frame_skip_values[: stage + 1]
            return int(self.rng.choice(candidates))
        return int(self.frame_skip_values[stage])

    def _current_curriculum_frame_skip(self) -> int:
        stage = self.decision_step_idx // self.within_episode_change_interval
        stage = min(stage, len(self.frame_skip_values) - 1)
        return int(self.frame_skip_values[stage])

    def _maybe_update_frame_skip_for_this_decision(self) -> None:
        if self.latency_mode == "episode_fixed":
            return
        if self.latency_mode == "within_episode_random":
            if self.decision_step_idx == 0:
                return
            if self.decision_step_idx % self.within_episode_change_interval != 0:
                return
            if self.rng.rand() <= self.within_episode_change_prob:
                self.current_frame_skip = self._sample_frame_skip()
            return
        if self.latency_mode == "within_episode_curriculum":
            self.current_frame_skip = self._current_curriculum_frame_skip()
            return

    def _schedule_action(self, action: int) -> int:
        latency_for_action = int(self.current_frame_skip)
        apply_step = self.env_step_idx + latency_for_action
        self._schedule_seq += 1
        insort_right(
            self.pending_actions,
            (apply_step, self._schedule_seq, int(action), latency_for_action),
        )
        return apply_step

    def _advance_pending_actions(self) -> None:
        while self.pending_actions and self.pending_actions[0][0] <= self.env_step_idx:
            _, _, matured_action, _ = self.pending_actions.pop(0)
            self.current_applied_action = int(matured_action)

    def reset(self, **kwargs):
        self.current_applied_action = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions.clear()
        self._schedule_seq = 0

        if self.latency_mode == "within_episode_curriculum":
            self.current_frame_skip = int(self.frame_skip_values[0])
        else:
            self.current_frame_skip = self._sample_frame_skip()

        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}

        info["queue_latency"] = int(self.current_frame_skip)
        info["frame_skip"] = int(self.current_frame_skip)
        info["episode_idx"] = int(self.episode_idx)
        info["latency_mode"] = self.latency_mode
        info["applied_action"] = int(self.current_applied_action)
        info["pending_queue_len"] = 0
        self.episode_idx += 1
        return obs, info

    def step(self, action):
        self._maybe_update_frame_skip_for_this_decision()
        scheduled_delay = int(self.current_frame_skip)
        scheduled_apply_step = self._schedule_action(int(action))
        self._advance_pending_actions()

        obs, reward, terminated, truncated, info = self.env.step(int(self.current_applied_action))
        if not isinstance(info, dict):
            info = {}

        info["queue_latency"] = scheduled_delay
        info["frame_skip"] = scheduled_delay
        info["decision_step_idx"] = int(self.decision_step_idx)
        info["env_step_idx"] = int(self.env_step_idx)
        info["latency_mode"] = self.latency_mode
        info["applied_action"] = int(self.current_applied_action)
        info["queued_action"] = int(action)
        info["scheduled_apply_step"] = int(scheduled_apply_step)
        info["pending_queue_len"] = int(len(self.pending_actions))

        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


class RenderToImageWrapper(gym.Wrapper):
    """Converts FlappyBird numeric observations into rendered image observations."""

    def __init__(self, env: gym.Env, image_size: int = 84, grayscale: bool = False):
        super().__init__(env)
        self.image_size = int(image_size)
        self.grayscale = bool(grayscale)
        channels = 1 if self.grayscale else 3
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, channels),
            dtype=np.uint8,
        )

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
        image = image.convert("L" if self.grayscale else "RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        out = np.asarray(image, dtype=np.uint8)
        if self.grayscale:
            out = out[..., None]
        return out

    def _get_image_obs(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("render() returned None. Use FlappyBird-v0 with render_mode='rgb_array'.")
        return self._convert_frame(frame)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._get_image_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_image_obs(), float(reward), terminated, truncated, info


class LatencyImageObsWrapper(gym.ObservationWrapper):
    """Append a latency plane to HWC image obs before stacking."""

    def __init__(self, env: gym.Env, max_frame_skip: int = 15):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        h, w, c = self.observation_space.shape
        self.max_frame_skip = max(1, int(max_frame_skip))
        self.observation_space = Box(low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8)

    def observation(self, obs):
        obs_arr = np.asarray(obs, dtype=np.uint8)
        frame_skip = int(_unwrap_attr(self.env, "current_frame_skip", 0))
        skip_norm = min(frame_skip / float(self.max_frame_skip), 1.0)
        skip_value = int(round(skip_norm * 255.0))
        latency_plane = np.full(obs_arr.shape[:2] + (1,), skip_value, dtype=np.uint8)
        return np.concatenate([obs_arr, latency_plane], axis=2)


class NCHWFrameStack(gym.Wrapper):
    """
    CleanRL-style frame stack.

    Input is HxWxC. Output is (C * n_stack) x H x W, e.g. RGB stack => 12x84x84,
    grayscale stack => 4x84x84.
    """

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = int(n_stack)
        assert self.n_stack >= 1
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 3, f"Expected HWC image, got {self.observation_space.shape}"
        h, w, c = self.observation_space.shape
        self.frames: deque[np.ndarray] = deque(maxlen=self.n_stack)
        self.observation_space = Box(low=0, high=255, shape=(c * self.n_stack, h, w), dtype=np.uint8)

    def _format_frame(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.uint8)
        if obs.ndim != 3:
            raise ValueError(f"Expected HWC obs, got shape={obs.shape}")
        return obs.transpose(2, 0, 1).copy()

    def _get_obs(self) -> np.ndarray:
        assert len(self.frames) == self.n_stack
        return np.concatenate(list(self.frames), axis=0).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._format_frame(obs)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs.copy())
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._format_frame(obs)
        self.frames.append(obs.copy())
        return self._get_obs(), float(reward), terminated, truncated, info


class ClipRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


# =========================
# Env builder
# =========================


def make_env(
    seed: int,
    idx: int,
    queue_latency_values: List[int],
    fps: int,
    image_size: int,
    grayscale: bool,
    stack_frames: int,
    clip_reward: bool,
    obs_mode: str,
    use_lidar: bool,
    normalize_obs: bool,
    initial_safe_steps: int,
    initial_safe_reward: float,
    disable_flap_during_initial_safe_steps: bool,
    latency_mode: str,
    latency_sampling_mode: str,
    episodes_per_stage: int,
    randomize_within_stage: bool,
    within_episode_change_interval: int,
    within_episode_change_prob: float,
):
    def thunk():
        env = gym.make(
            "FlappyBird-v0",
            use_lidar=bool(use_lidar),
            normalize_obs=bool(normalize_obs),
            obs_mode="numeric",
            render_mode="rgb_array",
            fps=int(fps),
            initial_safe_steps=int(initial_safe_steps),
            initial_safe_reward=float(initial_safe_reward),
            disable_flap_during_initial_safe_steps=bool(disable_flap_during_initial_safe_steps),
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if clip_reward:
            env = ClipRewardWrapper(env)

        env = QueueLatencyWrapper(
            env,
            frame_skip_values=queue_latency_values,
            latency_mode=latency_mode,
            latency_sampling_mode=latency_sampling_mode,
            episodes_per_stage=episodes_per_stage,
            randomize_within_stage=randomize_within_stage,
            within_episode_change_interval=within_episode_change_interval,
            within_episode_change_prob=within_episode_change_prob,
            seed=seed + idx,
        )

        env = RenderToImageWrapper(env, image_size=image_size, grayscale=grayscale)

        if obs_mode == "latency_channel":
            env = LatencyImageObsWrapper(env, max_frame_skip=max(queue_latency_values))
        elif obs_mode != "image_only":
            raise ValueError(f"Unsupported obs_mode for this CleanRL script: {obs_mode}")

        if stack_frames > 1:
            env = NCHWFrameStack(env, n_stack=stack_frames)
        else:
            # Still convert single HWC image to CHW.
            env = NCHWFrameStack(env, n_stack=1)

        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_vector_env(args, queue_latency_values: List[int], eval_mode: bool = False):
    env_fns = [
        make_env(
            seed=args.seed + (100_000 if eval_mode else 0),
            idx=i,
            queue_latency_values=queue_latency_values,
            fps=args.fps,
            image_size=args.image_size,
            grayscale=args.grayscale,
            stack_frames=args.stack_frames,
            clip_reward=bool(args.clip_reward),
            obs_mode=args.obs_mode,
            use_lidar=args.use_lidar,
            normalize_obs=args.normalize_obs,
            initial_safe_steps=args.initial_safe_steps,
            initial_safe_reward=args.initial_safe_reward,
            disable_flap_during_initial_safe_steps=args.disable_flap_during_initial_safe_steps,
            latency_mode=args.latency_mode if not eval_mode else "episode_fixed",
            latency_sampling_mode=args.latency_sampling_mode if not eval_mode else "uniform_all",
            episodes_per_stage=args.episodes_per_stage if not eval_mode else 10**9,
            randomize_within_stage=args.randomize_within_stage if not eval_mode else False,
            within_episode_change_interval=args.within_episode_change_interval if not eval_mode else 1,
            within_episode_change_prob=args.within_episode_change_prob if not eval_mode else 1.0,
        )
        for i in range(args.n_envs if not eval_mode else 1)
    ]
    if args.vector_env == "async" and not eval_mode:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)


# =========================
# Agent
# =========================


class Agent(nn.Module):
    def __init__(self, obs_shape, num_actions: int):
        super().__init__()
        assert len(obs_shape) == 3, f"Expected C,H,W obs shape; got {obs_shape}"
        c, h, w = [int(x) for x in obs_shape]
        assert h == 84 and w == 84, f"Expected 84x84 image; got {obs_shape}"

        conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = int(conv(torch.zeros(1, c, h, w)).shape[1])

        self.network = nn.Sequential(
            conv,
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x.float() / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.float() / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    @torch.no_grad()
    def get_deterministic_action(self, x):
        hidden = self.network(x.float() / 255.0)
        logits = self.actor(hidden)
        return torch.argmax(logits, dim=-1)


# =========================
# Logging, eval, checkpointing
# =========================


def extract_episode_stats(infos):
    stats = []
    if not isinstance(infos, dict):
        return stats

    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                ep = info["episode"]
                stats.append((float(np.asarray(ep["r"]).item()), float(np.asarray(ep["l"]).item())))

    if "episode" in infos:
        ep = infos["episode"]
        if isinstance(ep, dict) and "r" in ep and "l" in ep:
            r = np.asarray(ep["r"]).reshape(-1)
            l = np.asarray(ep["l"]).reshape(-1)
            for rr, ll in zip(r, l):
                stats.append((float(rr), float(ll)))

    return stats


def save_checkpoint(path: str, agent: Agent, optimizer: optim.Optimizer, args, global_step: int):
    safe_mkdir(os.path.dirname(path) or ".")
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "global_step": int(global_step),
        },
        path,
    )


def load_checkpoint(path: str, agent: Agent, optimizer: Optional[optim.Optimizer], device) -> int:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        agent.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[load] could not load optimizer state: {e}")
        return int(ckpt.get("global_step", 0))
    agent.load_state_dict(ckpt)
    return 0


@torch.no_grad()
def evaluate(agent: Agent, args, device, latency_value: int, eval_episodes: Optional[int] = None):
    old_n_envs = args.n_envs
    old_clip_reward = args.clip_reward
    old_latency_mode = args.latency_mode
    old_latency_sampling_mode = args.latency_sampling_mode

    args.n_envs = 1
    if args.eval_raw_reward:
        args.clip_reward = False
    args.latency_mode = "episode_fixed"
    args.latency_sampling_mode = "uniform_all"

    envs = make_vector_env(args, queue_latency_values=[int(latency_value)], eval_mode=True)
    episodes = int(eval_episodes if eval_episodes is not None else args.eval_episodes)
    returns: List[float] = []
    lengths: List[int] = []

    for ep_idx in range(episodes):
        obs_np, _ = envs.reset(seed=args.seed + 100_000 + ep_idx + int(latency_value) * 1000)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done and (args.eval_max_steps <= 0 or ep_len < args.eval_max_steps):
            obs_t = torch.as_tensor(obs_np, dtype=torch.uint8, device=device)
            action = agent.get_deterministic_action(obs_t).cpu().numpy()
            obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(action)
            done = bool(np.logical_or(terminated_np, truncated_np)[0])
            ep_ret += float(reward_np[0])
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    envs.close()
    args.n_envs = old_n_envs
    args.clip_reward = old_clip_reward
    args.latency_mode = old_latency_mode
    args.latency_sampling_mode = old_latency_sampling_mode

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
    }


# =========================
# Main PPO loop
# =========================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-steps", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)

    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--outdir", type=str, default="./flappy_cleanrl_data")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)

    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--train-frame-stride", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--learning-rate-final", type=float, default=2.5e-6)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--clip-range-final", type=float, default=0.025)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale frames; otherwise RGB frames are stacked.")
    parser.add_argument("--stack-frames", type=int, default=4)
    parser.add_argument("--clip-reward", action="store_true")
    parser.add_argument("--eval-raw-reward", action="store_true")
    parser.add_argument("--vector-env", type=str, default="sync", choices=["sync", "async"])

    parser.add_argument("--use-lidar", action="store_true")
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--initial-safe-steps", type=int, default=0)
    parser.add_argument("--initial-safe-reward", type=float, default=0.1)
    parser.add_argument("--disable-flap-during-initial-safe-steps", action="store_true")

    parser.add_argument("--obs-mode", type=str, default="image_only", choices=["image_only", "latency_channel"])
    parser.add_argument("--frame-skip-values", type=str, default="0")
    parser.add_argument(
        "--latency-mode",
        type=str,
        default="episode_fixed",
        choices=["episode_fixed", "within_episode_random", "within_episode_curriculum", "within_episode_change"],
    )
    parser.add_argument("--latency-sampling-mode", type=str, default="uniform_all", choices=["uniform_all", "curriculum"])
    parser.add_argument("--episodes-per-stage", type=int, default=10)
    parser.add_argument("--randomize-within-stage", action="store_true")
    parser.add_argument("--within-episode-change-interval", type=int, default=1)
    parser.add_argument("--within-episode-change-prob", type=float, default=1.0)

    parser.add_argument("--mid-eval", action="store_true")
    parser.add_argument("--mid-eval-freq", type=int, default=250_000)
    parser.add_argument("--mid-eval-episodes", type=int, default=20)
    parser.add_argument("--mid-eval-frame-skip-values", type=str, default="0")

    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-frame-skip-values", type=str, default="0,1,2,3,4,5,10,15")
    parser.add_argument("--eval-max-steps", type=int, default=5000)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="flappy_cleanrl_queue_latency")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")

    args = parser.parse_args()

    safe_mkdir(os.path.dirname(args.save) or ".")
    safe_mkdir(args.outdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = args.wandb_run_name or f"cleanrl_flappy_queue_{int(time.time())}"
    wandb_run = None
    if args.wandb:
        import wandb

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config=vars(args),
            save_code=True,
        )

    train_latency_values = parse_int_list(args.frame_skip_values)
    if not train_latency_values:
        raise ValueError("--frame-skip-values must contain at least one latency value")

    print("=== CLEANRL PPO FLAPPY QUEUE LATENCY ===")
    print(f"device={device}")
    print(f"train_steps={args.train_steps}")
    print(f"n_envs={args.n_envs}")
    print(f"n_steps={args.n_steps}")
    print(f"rollout_batch={args.n_envs * args.n_steps}")
    print(f"minibatch_size={args.batch_size}")
    print(f"n_epochs={args.n_epochs}")
    print(f"fps={args.fps}")
    print(f"obs_mode={args.obs_mode}")
    print(f"grayscale={args.grayscale}")
    print(f"stack_frames={args.stack_frames}")
    print(f"train_queue_latency_values={args.frame_skip_values}")
    print(f"train_clip_reward={args.clip_reward}")
    print(f"eval_raw_reward={args.eval_raw_reward}")
    print(f"save={args.save}")
    print("=========================================")

    envs = make_vector_env(args, queue_latency_values=train_latency_values, eval_mode=False)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    obs_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n
    print(f"single_observation_space={envs.single_observation_space}")
    print(f"single_action_space={envs.single_action_space}")

    agent = Agent(obs_shape=obs_shape, num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_global_step = 0
    if args.load:
        print(f"[load] loading checkpoint => {args.load}")
        start_global_step = load_checkpoint(args.load, agent, optimizer, device)
        print(f"[load] checkpoint_global_step={start_global_step}")

    rollout_batch_size = args.n_envs * args.n_steps
    minibatch_size = int(args.batch_size)
    train_frame_stride = max(1, int(args.train_frame_stride))
    collection_steps_per_update = args.n_steps * train_frame_stride

    assert rollout_batch_size % minibatch_size == 0, (
        f"rollout batch {rollout_batch_size} must be divisible by minibatch size {minibatch_size}"
    )

    obs = torch.zeros((args.n_steps, args.n_envs) + obs_shape, dtype=torch.uint8, device=device)
    actions = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)

    b_inds = np.arange(rollout_batch_size)
    num_updates = args.train_steps // (args.n_envs * collection_steps_per_update)

    global_step = int(start_global_step)
    start_time = time.time()
    next_checkpoint_step = global_step + int(args.checkpoint_freq)
    next_mid_eval_step = global_step + int(args.mid_eval_freq)
    best_mid_eval_return = -float("inf")

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)
    next_done = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    print(f"train_frame_stride={train_frame_stride}")
    print(f"collection_steps_per_update={collection_steps_per_update}")
    print(f"num_updates={num_updates}")

    for update in range(1, num_updates + 1):
        progress_remaining = 1.0 - (update - 1.0) / max(1, num_updates)
        lr_now = linear_schedule(args.learning_rate, args.learning_rate_final, progress_remaining)
        clip_now = linear_schedule(args.clip_range, args.clip_range_final, progress_remaining)
        optimizer.param_groups[0]["lr"] = lr_now

        stored_step = 0
        for collect_step in range(collection_steps_per_update):
            global_step += args.n_envs
            keep_this_step = (collect_step % train_frame_stride) == (train_frame_stride - 1)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            if keep_this_step:
                obs[stored_step] = next_obs
                dones[stored_step] = next_done
                actions[stored_step] = action
                logprobs[stored_step] = logprob
                values[stored_step] = value.flatten()

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated_np, truncated_np)

            if keep_this_step:
                rewards[stored_step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
                stored_step += 1

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            for ep_ret, ep_len in extract_episode_stats(infos):
                print(
                    f"[train-episode] global_step={global_step} "
                    f"episodic_return={ep_ret:.3f} episodic_length={ep_len:.1f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "charts/episodic_return": ep_ret,
                            "charts/episodic_length": ep_len,
                            "charts/SPS": int((global_step - start_global_step) / max(1e-6, time.time() - start_time)),
                        },
                        step=global_step,
                    )

        assert stored_step == args.n_steps, f"Expected {args.n_steps} PPO steps, stored {stored_step}"

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        clipfracs = []
        old_approx_kl = torch.tensor(0.0, device=device)
        approx_kl = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_now).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_now, 1.0 + clip_now)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_now, clip_now)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
        sps = int((global_step - start_global_step) / max(1e-6, time.time() - start_time))

        print(
            f"[update] {update}/{num_updates} "
            f"global_step={global_step} SPS={sps} lr={lr_now:.6g} clip={clip_now:.4f} "
            f"pg_loss={pg_loss.item():.4f} value_loss={v_loss.item():.4f} "
            f"entropy={entropy_loss.item():.4f} approx_kl={approx_kl.item():.6f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "charts/learning_rate": lr_now,
                    "charts/clip_coef": clip_now,
                    "charts/SPS": sps,
                    "losses/value_loss": float(v_loss.item()),
                    "losses/policy_loss": float(pg_loss.item()),
                    "losses/entropy": float(entropy_loss.item()),
                    "losses/old_approx_kl": float(old_approx_kl.item()),
                    "losses/approx_kl": float(approx_kl.item()),
                    "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "losses/explained_variance": float(explained_var),
                    "global_step": int(global_step),
                },
                step=global_step,
            )

        if args.checkpoint_freq > 0 and global_step >= next_checkpoint_step:
            base = os.path.basename(args.save).replace(".pt", "")
            ckpt_path = os.path.join(os.path.dirname(args.save) or ".", f"{base}_{global_step}_steps.pt")
            save_checkpoint(ckpt_path, agent, optimizer, args, global_step)
            print(f"[checkpoint] saved => {ckpt_path}")
            next_checkpoint_step += int(args.checkpoint_freq)

        if args.mid_eval and args.mid_eval_freq > 0 and global_step >= next_mid_eval_step:
            mid_latencies = parse_int_list(args.mid_eval_frame_skip_values)
            if len(mid_latencies) != 1:
                raise ValueError("--mid-eval-frame-skip-values must have exactly one value.")
            latency_value = int(mid_latencies[0])
            result = evaluate(agent, args, device, latency_value, eval_episodes=args.mid_eval_episodes)
            tag = "raw" if args.eval_raw_reward else "clipped"
            mean_ret = result["mean_return"]
            print(
                f"[mid-eval] step={global_step} latency={latency_value} "
                f"mean_return={mean_ret:.3f} std_return={result['std_return']:.3f} "
                f"mean_len={result['mean_length']:.1f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/{tag}/latency_{latency_value}/mean_return": mean_ret,
                        f"eval/{tag}/latency_{latency_value}/std_return": result["std_return"],
                        f"eval/{tag}/latency_{latency_value}/mean_length": result["mean_length"],
                    },
                    step=global_step,
                )
            if mean_ret > best_mid_eval_return:
                best_mid_eval_return = mean_ret
                best_path = os.path.join(os.path.dirname(args.save) or ".", "best_model.pt")
                save_checkpoint(best_path, agent, optimizer, args, global_step)
                print(f"[mid-eval] new best saved => {best_path}")
            next_mid_eval_step += int(args.mid_eval_freq)

    if args.train_steps > 0:
        save_checkpoint(args.save, agent, optimizer, args, global_step)
        print(f"[train] saved final model => {args.save}")

    envs.close()

    if args.eval_episodes > 0:
        tag = "raw" if args.eval_raw_reward else "clipped"
        final_results: Dict[str, Any] = {"reward_tag": tag, "global_step": int(global_step), "latencies": {}}
        print(f"[final-eval] episodes={args.eval_episodes} reward_tag={tag}")
        for latency_value in parse_int_list(args.eval_frame_skip_values):
            result = evaluate(agent, args, device, int(latency_value))
            final_results["latencies"][str(latency_value)] = serialize_jsonable(result)
            print(
                f"[eval] reward={tag} latency={latency_value} "
                f"mean_return={result['mean_return']:.3f} "
                f"std_return={result['std_return']:.3f} "
                f"mean_len={result['mean_length']:.1f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"final_eval/{tag}/latency_{latency_value}/mean_return": result["mean_return"],
                        f"final_eval/{tag}/latency_{latency_value}/std_return": result["std_return"],
                        f"final_eval/{tag}/latency_{latency_value}/mean_length": result["mean_length"],
                    },
                    step=global_step,
                )

        eval_path = os.path.join(args.outdir, "final_eval_results.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
        print(f"[final-eval] wrote => {eval_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
