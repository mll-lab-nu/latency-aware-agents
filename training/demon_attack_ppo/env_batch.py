from __future__ import annotations

from bisect import insort_right
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Sequence

import gymnasium as gym
import numpy as np
import torch

from training.atari_episode_stats import is_true_atari_episode_end


class QueueLatencyWrapper(gym.Wrapper):
    """True input-delay wrapper for queue latency."""

    def __init__(self, env: gym.Env, frame_skip_values: Sequence[int], seed: int):
        super().__init__(env)
        self.frame_skip_values = [int(x) for x in frame_skip_values]
        self.rng = np.random.RandomState(seed)
        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[tuple[int, int, int, int]] = []
        self._schedule_seq = 0

    def _sample_frame_skip(self) -> int:
        return int(self.rng.choice(self.frame_skip_values))

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
        self.current_frame_skip = int(self.frame_skip_values[0])
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["frame_skip"] = int(self.current_frame_skip)
        info["episode_idx"] = int(self.episode_idx)
        info["applied_action"] = int(self.current_applied_action)
        info["pending_queue_len"] = 0
        self.episode_idx += 1
        return obs, info

    def step(self, action):
        self.current_frame_skip = self._sample_frame_skip()
        scheduled_delay = int(self.current_frame_skip)
        scheduled_apply_step = self._schedule_action(int(action))
        self._advance_pending_actions()

        obs, reward, terminated, truncated, info = self.env.step(
            int(self.current_applied_action)
        )

        info = dict(info or {})
        info["frame_skip"] = scheduled_delay
        info["decision_step_idx"] = int(self.decision_step_idx)
        info["env_step_idx"] = int(self.env_step_idx)
        info["applied_action"] = int(self.current_applied_action)
        info["queued_action"] = int(action)
        info["scheduled_apply_step"] = int(scheduled_apply_step)
        info["pending_queue_len"] = int(len(self.pending_actions))

        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


def make_demon_attack_env(
    seed: int,
    env_id: str = "ALE/DemonAttack-v5",
    frame_skip_values: Sequence[int] = (0,),
    screen_size: int = 84,
    noop_max: int = 30,
    action_repeat: int = 1,
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
) -> gym.Env:
    _register_ale()

    from stable_baselines3.common.atari_wrappers import AtariWrapper

    action_repeat = int(action_repeat)
    if action_repeat < 1:
        raise ValueError("action_repeat must be >= 1")

    env = gym.make(
        env_id,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=float(repeat_action_probability),
        full_action_space=bool(full_action_space),
        render_mode=None,
    )
    env = AtariWrapper(
        env,
        noop_max=int(noop_max),
        frame_skip=action_repeat,
        screen_size=int(screen_size),
        terminal_on_life_loss=bool(terminal_on_life_loss),
        clip_reward=bool(clip_reward),
        action_repeat_probability=0.0,
    )
    env = QueueLatencyWrapper(env, frame_skip_values=frame_skip_values, seed=seed)
    return env


def _register_ale() -> None:
    try:
        import ale_py
    except Exception:
        return
    register_envs = getattr(gym, "register_envs", None)
    if callable(register_envs):
        register_envs(ale_py)


class SyncDemonAttackBatch:
    def __init__(
        self,
        n_envs: int,
        seed: int,
        frame_skip_values: Sequence[int],
        device: str | torch.device,
        env_id: str = "ALE/DemonAttack-v5",
        screen_size: int = 84,
        noop_max: int = 30,
        action_repeat: int = 1,
        terminal_on_life_loss: bool = False,
        clip_reward: bool = True,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        env_fns: Sequence[Callable[[], gym.Env]] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.episode_returns = np.zeros(int(n_envs), dtype=np.float32)
        self.episode_lengths = np.zeros(int(n_envs), dtype=np.int32)
        action_repeat = int(action_repeat)
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if env_fns is not None:
            self.envs = [fn() for fn in env_fns]
        else:
            self.envs = [
                make_demon_attack_env(
                    seed=seed + idx,
                    env_id=env_id,
                    frame_skip_values=frame_skip_values,
                    screen_size=screen_size,
                    noop_max=noop_max,
                    action_repeat=action_repeat,
                    terminal_on_life_loss=terminal_on_life_loss,
                    clip_reward=clip_reward,
                    repeat_action_probability=repeat_action_probability,
                    full_action_space=full_action_space,
                )
                for idx in range(int(n_envs))
            ]
        if not self.envs:
            raise ValueError("n_envs must be positive")
        self._last_frames: list[np.ndarray] = []

    @property
    def n_envs(self) -> int:
        return len(self.envs)

    @property
    def n_actions(self) -> int:
        action_space = self.envs[0].action_space
        n = getattr(action_space, "n", None)
        if n is None:
            raise ValueError("DemonAttack PPO requires a discrete action space")
        return int(n)

    def reset(self) -> torch.Tensor:
        frames: list[np.ndarray] = []
        for idx, env in enumerate(self.envs):
            frame, _ = env.reset()
            self.episode_returns[idx] = 0.0
            self.episode_lengths[idx] = 0
            frames.append(np.asarray(frame, dtype=np.uint8))
        self._last_frames = frames
        return self._frames_to_tensor(frames)

    def step(self, actions: np.ndarray, profiler: Any | None = None):
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)
        infos: List[Dict] = []
        frames: list[np.ndarray] = []
        timer = (
            profiler.time("atari_frame_batch_s") if profiler is not None else nullcontext()
        )
        with timer:
            for idx, (env, action) in enumerate(zip(self.envs, actions)):
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)
                info = dict(info or {})
                self.episode_returns[idx] += float(reward)
                self.episode_lengths[idx] += 1
                if done:
                    true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
                    if true_episode_end:
                        info["episodic_return"] = float(self.episode_returns[idx])
                        info["episodic_length"] = int(self.episode_lengths[idx])
                        self.episode_returns[idx] = 0.0
                        self.episode_lengths[idx] = 0
                    info["terminal_observation"] = np.asarray(obs, dtype=np.uint8)
                    reset_obs, reset_info = env.reset()
                    info["reset_info"] = dict(reset_info or {})
                    obs = reset_obs
                rewards[idx] = float(reward)
                dones[idx] = done
                infos.append(info)
                frames.append(np.asarray(obs, dtype=np.uint8))
        self._last_frames = frames
        return rewards, dones, infos, self._frames_to_tensor(frames)

    def frame_skips(self) -> np.ndarray:
        return np.asarray(
            [int(getattr(env, "current_frame_skip", 0)) for env in self.envs],
            dtype=np.int64,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _frames_to_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        batch = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)
        return torch.as_tensor(batch, dtype=torch.uint8, device=self.device)
