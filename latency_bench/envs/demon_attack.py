from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.raw_rgb import RawRgbFrameStackBuffer
from latency_bench.utils.array import looks_chw


class DemonAttackEnvAdapter(EnvAdapter):
    """Benchmark adapter for Atari DemonAttack with training-matched preprocessing."""
    OBSERVATION_TYPE = "atari_frame_v1"

    def __init__(
        self,
        env_id: str = "ALE/DemonAttack-v5",
        env_fps: float = 60.0,
        noop_action: Action | None = None,
        screen_size: int = 84,
        noop_max: int = 30,
        reset_mode: str = "fire",
        terminal_on_life_loss: bool = False,
        clip_reward: bool = True,
        continue_life_loss_terminal: bool = False,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        expected_action_meanings: list[str] | tuple[str, ...] | None = None,
        frame_stack: int = 1,
        export_env_raw_rgb_frames: bool = False,
    ):
        self._env_kwargs = {
            "env_id": env_id,
            "screen_size": screen_size,
            "noop_max": noop_max,
            "reset_mode": reset_mode,
            "terminal_on_life_loss": terminal_on_life_loss,
            "clip_reward": clip_reward,
            "repeat_action_probability": repeat_action_probability,
            "full_action_space": full_action_space,
        }
        self._expected_action_meanings = expected_action_meanings
        self.gym_env = self._build_gym_env()
        self.env_id = env_id
        self.env_fps = env_fps
        self.frame_stack = frame_stack
        self.export_env_raw_rgb_frames = export_env_raw_rgb_frames
        self.noop_action = noop_action or Action(value=0, name="0", is_noop=True)
        self.continue_life_loss_terminal = continue_life_loss_terminal
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_frame: Any = None
        self._observed_frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=self.frame_stack)

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self._observed_frames.clear()
        if seed is not None and self._last_frame is not None:
            self.gym_env.close()
            self.gym_env = self._build_gym_env()
        frame, info = self.gym_env.reset(seed=seed)
        self._last_frame = frame
        self._last_info = dict(info or {})
        self._reset_frame_stack(frame)
        if self.export_env_raw_rgb_frames:
            self._reset_raw_rgb_frame_stack()
        return self._make_observation(info=self._last_info)

    def step(self, action: Action) -> StepResult:
        gym_action = int(action.value)
        frame, reward, terminated, truncated, info = self.gym_env.step(gym_action)
        self.env_step += 1
        self._last_frame = frame
        self._last_info = dict(info or {})
        true_episode_end = _is_true_episode_end(
            self.gym_env,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self._last_info,
        )
        if self.continue_life_loss_terminal and bool(terminated or truncated) and not true_episode_end:
            terminal_info = dict(self._last_info)
            terminal_env_step = int(self.env_step)
            reset_frame, reset_info = self.gym_env.reset()
            self.env_step = 0
            self._last_frame = reset_frame
            self._last_info = dict(reset_info or {})
            self._reset_frame_stack(reset_frame)
            if self.export_env_raw_rgb_frames:
                self._reset_raw_rgb_frame_stack()
            observation = self._make_observation(info=self._last_info)
            step_info = dict(terminal_info)
            step_info.update(
                {
                    "env_step": terminal_env_step,
                    "sim_time_ms": terminal_env_step * self.frame_ms,
                    "applied_action": gym_action,
                    "applied_action_name": action.name,
                    "observation": "atari_frame_v1",
                    "soft_reset": True,
                    "ignored_life_loss_terminal": True,
                    "original_terminated": bool(terminated),
                    "original_truncated": bool(truncated),
                    TRUE_EPISODE_END_INFO_KEY: False,
                    "reset_info": dict(reset_info or {}),
                }
            )
            return StepResult(
                observation=observation,
                reward=float(reward),
                done=False,
                truncated=False,
                info=step_info,
            )

        self._append_frame(frame)
        if self.export_env_raw_rgb_frames and not bool(terminated or truncated):
            self._append_raw_rgb_frame()
        observation = self._make_observation(info=self._last_info)
        step_info = dict(self._last_info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": gym_action,
                "applied_action_name": action.name,
                "observation": "atari_frame_v1",
                TRUE_EPISODE_END_INFO_KEY: bool(true_episode_end) if bool(terminated or truncated) else False,
            }
        )
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(terminated),
            truncated=bool(truncated),
            info=step_info,
        )

    def observe(self) -> Observation:
        if self._last_frame is None:
            raise RuntimeError("DemonAttackEnvAdapter has no current observation; call reset() first")
        metadata = self._metadata(self._last_info)
        return Observation(
            data=self._policy_frame_stack(),
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def render_game_frame(self) -> np.ndarray:
        return self.gym_env.unwrapped.ale.getScreenRGB()

    def close(self) -> None:
        self.gym_env.close()

    def _build_gym_env(self) -> gym.Env:
        env = make_demon_attack_benchmark_env(**self._env_kwargs)
        _validate_expected_action_meanings(env, self._expected_action_meanings)
        return env

    def _reset_frame_stack(self, frame: Any) -> None:
        self._observed_frames.clear()
        self._append_frame(frame)

    def _append_frame(self, frame: Any) -> None:
        self._observed_frames.append(np.asarray(frame, dtype=np.uint8))

    def _policy_frame_stack(self) -> np.ndarray:
        frames = list(self._observed_frames)
        if not frames:
            raise RuntimeError("DemonAttack observe() has no current frame; call reset() first")
        if len(frames) < self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames
        frames = [_single_frame_data(frame) for frame in frames[-self.frame_stack :]]
        if self.frame_stack == 1:
            return frames[-1]
        axis = 0 if looks_chw(frames[0]) else -1
        return np.concatenate(frames, axis=axis)


def _single_frame_data(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected Demon Attack image frame with 2 or 3 dims, got {arr.shape!r}")
    if arr.shape[-1] in {1, 3} and arr.shape[0] not in {1, 3}:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _is_true_episode_end(
    env: gym.Env,
    *,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> bool:
    if bool(truncated):
        return True
    if not bool(terminated):
        return False
    if any(key in info for key in (TRUE_EPISODE_END_INFO_KEY, "game_over", "lives")):
        return is_true_atari_episode_end(terminated=True, truncated=False, info=info)

    was_real_done = _episodic_life_was_real_done(env)
    if was_real_done is not None:
        return bool(was_real_done)

    return is_true_atari_episode_end(terminated=True, truncated=False, info=info)


def _episodic_life_was_real_done(env: gym.Env) -> bool | None:
    current: Any = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "was_real_done"):
            return bool(getattr(current, "was_real_done"))
        current = getattr(current, "env", None)
    return None


def _validate_expected_action_meanings(
    env: gym.Env,
    expected_action_meanings: list[str] | tuple[str, ...] | None,
) -> None:
    if expected_action_meanings is None:
        return
    get_action_meanings = getattr(env, "get_action_meanings", None)
    if not callable(get_action_meanings):
        raise ValueError(
            "Demon Attack expected_action_meanings was provided, but the created env "
            "does not expose get_action_meanings()."
        )
    actual = list(get_action_meanings())
    expected = [str(item) for item in expected_action_meanings]
    if actual != expected:
        raise ValueError(
            "Demon Attack action meanings mismatch: "
            f"expected_action_meanings={expected}, actual={actual}"
        )


def make_demon_attack_benchmark_env(
    env_id: str = "ALE/DemonAttack-v5",
    screen_size: int = 84,
    noop_max: int = 30,
    reset_mode: str = "fire",
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
) -> gym.Env:
    _register_ale()
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        NoopResetEnv,
        WarpFrame,
    )

    env = gym.make(
        env_id,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        render_mode=None,
    )
    mode = reset_mode
    if mode in {"noop", "fire"} and noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if terminal_on_life_loss:
        env = EpisodicLifeEnv(env)
    if mode == "fire" and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    elif mode not in {"raw", "noop", "fire"}:
        raise ValueError(f"Unsupported Demon Attack reset_mode: {reset_mode!r}")
    env = WarpFrame(env, width=screen_size, height=screen_size)
    if clip_reward:
        env = ClipRewardEnv(env)
    return env


def _register_ale() -> None:
    try:
        import ale_py
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ale_py. Install Atari support with the repo requirements and "
            "ensure the current environment can import ale_py before running Demon Attack."
        ) from exc
    register_envs = getattr(gym, "register_envs", None)
    if callable(register_envs):
        register_envs(ale_py)
    has_ale_namespace = any(str(env_id).startswith("ALE/") for env_id in gym.envs.registry.keys())
    if not has_ale_namespace:
        raise RuntimeError(
            "ale_py imported but Gymnasium did not register the ALE namespace. "
            "Expected ALE/* environments such as ALE/DemonAttack-v5 to be available."
        )
