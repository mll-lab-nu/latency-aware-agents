from __future__ import annotations

import copy
from collections import deque
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Tuple

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter


def _noop_action_from_space(space) -> Any:
    n = getattr(space, "n", None)
    if n is not None:
        return 0
    if isinstance(space, Tuple):
        return tuple(_noop_action_from_space(subspace) for subspace in space.spaces)
    if isinstance(space, Box):
        import numpy as np

        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"Unsupported action space for Deadly Corridor no-op action: {space}")


def _coerce_noop_action_for_space(value: Any, space) -> Any:
    if isinstance(space, Tuple):
        if isinstance(value, (list, tuple)):
            if len(value) != len(space.spaces):
                raise ValueError(
                    f"Deadly Corridor no-op action length {len(value)} does not match action space {space}"
                )
            return tuple(
                _coerce_noop_action_for_space(item, subspace)
                for item, subspace in zip(value, space.spaces)
            )
        if value == 0:
            return _noop_action_from_space(space)
    return value


def _spec_with_reward_scaling(spec: Any, disable_reward_scaling: bool) -> Any:
    if not disable_reward_scaling:
        return spec
    spec_to_use = copy.copy(spec)
    spec_to_use.reward_scaling = 1.0
    return spec_to_use


def _synchronous_eval_fps_from_config(config: dict[str, Any], default: int = 35) -> int:
    env_cfg = config.get("env", {})
    try:
        fps = int(float(env_cfg.get("env_fps", default)))
    except (TypeError, ValueError) as exc:
        raise ValueError("env_fps must be positive") from exc
    if fps <= 0:
        raise ValueError("env_fps must be positive")
    return fps


def _build_sample_factory_eval_cfg(config: dict[str, Any]) -> Any:
    from training.deadly_corridor_sf import integration
    from training.sample_factory_common.common import maybe_set_cli_override

    integration.register_deadly_corridor_components()
    base_cfg = integration.parse_cfg(integration.build_cli_args_from_config(config))
    eval_fps = _synchronous_eval_fps_from_config(config)
    cfg = copy.deepcopy(base_cfg)
    if _requires_sample_factory_checkpoint_config(config):
        from sample_factory.cfg.arguments import load_from_checkpoint

        cfg = load_from_checkpoint(cfg)

    for key in (
        "seed",
        "res_w",
        "res_h",
        "wide_aspect_ratio",
    ):
        if hasattr(base_cfg, key):
            maybe_set_cli_override(cfg, key, getattr(base_cfg, key))
    maybe_set_cli_override(cfg, "frame_stack", 1)
    explicit_max_episode_steps = int(getattr(base_cfg, "max_episode_steps", 0) or 0)
    if explicit_max_episode_steps > 0:
        maybe_set_cli_override(cfg, "max_episode_steps", explicit_max_episode_steps)
    else:
        eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 0) or 0)
        if eval_max_steps > 0:
            maybe_set_cli_override(cfg, "max_episode_steps", eval_max_steps)

    maybe_set_cli_override(cfg, "mode", "eval")
    maybe_set_cli_override(cfg, "latency_type", "zero")
    maybe_set_cli_override(cfg, "fixed_latency_ms", 0.0)
    maybe_set_cli_override(cfg, "env_frameskip", 1)
    maybe_set_cli_override(cfg, "eval_env_frameskip", 1)
    maybe_set_cli_override(cfg, "num_envs", 1)
    maybe_set_cli_override(cfg, "no_render", True)
    maybe_set_cli_override(cfg, "save_video", False)
    maybe_set_cli_override(cfg, "fps", eval_fps)
    maybe_set_cli_override(cfg, "eval_deterministic", bool(getattr(base_cfg, "eval_deterministic", True)))
    maybe_set_cli_override(cfg, "disable_reward_scaling", bool(getattr(base_cfg, "eval_raw_reward", False)))
    return cfg


def _requires_sample_factory_checkpoint_config(config: dict[str, Any]) -> bool:
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    return policy_type == "deadly_corridor_sf"


def _seed_initialized_vizdoom_game(env: Any, seed: int) -> bool:
    unwrapped = getattr(env, "unwrapped", env)
    game = getattr(unwrapped, "game", None)
    if game is None:
        return False
    unwrapped.seed(int(seed))
    game.set_seed(int(unwrapped.curr_seed))
    return True


class DeadlyCorridorEnvAdapter(EnvAdapter):
    """Latency-bench adapter for ViZDoom Deadly Corridor using the SF Doom env stack."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        noop_action: Action | None = None,
    ):
        env_cfg = config["env"]
        env_id = str(env_cfg.get("env_id", "doom_deadly_corridor"))
        env_fps = float(env_cfg.get("env_fps", 35))
        if env_fps <= 0:
            raise ValueError("env_fps must be positive")
        self.frame_stack = int(env_cfg.get("frame_stack", 1) or 1)
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")

        from sample_factory.utils.attr_dict import AttrDict
        from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec

        cfg = _build_sample_factory_eval_cfg(config)
        spec = next((item for item in DOOM_ENVS if item.name == str(env_id)), None)
        if spec is None:
            raise ValueError(f"Unknown ViZDoom env spec: {env_id}")
        spec_to_use = _spec_with_reward_scaling(
            spec,
            disable_reward_scaling=bool(getattr(cfg, "disable_reward_scaling", False)),
        )
        self.gym_env = make_doom_env_from_spec(
            spec_to_use,
            str(env_id),
            cfg,
            AttrDict(worker_index=0, vector_index=0, env_id=0),
            render_mode=None,
        )
        self.cfg = cfg
        self.env_id = env_id
        self.env_fps = float(env_fps)
        action_space = self.gym_env.action_space
        noop_value = _noop_action_from_space(action_space)
        if noop_action is None:
            self.noop_action = Action(value=noop_value, name=str(noop_value), is_noop=True)
        else:
            coerced_noop_value = _coerce_noop_action_for_space(noop_action.value, action_space)
            self.noop_action = Action(
                value=coerced_noop_value,
                name=str(coerced_noop_value),
                is_noop=True,
                is_oneshot=noop_action.is_oneshot,
            )
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_frame: Any = None
        self._observed_frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self._observed_frames.clear()
        if seed is not None:
            if _seed_initialized_vizdoom_game(self.gym_env, int(seed)):
                obs, info = self.gym_env.reset()
            else:
                try:
                    obs, info = self.gym_env.reset(seed=seed)
                except TypeError:
                    obs, info = self.gym_env.reset()
        else:
            obs, info = self.gym_env.reset()
        self._last_frame = obs
        self._last_info = dict(info or {})
        return self._make_observation(info=self._last_info)

    def step(self, action: Action) -> StepResult:
        gym_action = action.value
        obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
        self.env_step += 1
        self._last_frame = obs
        self._last_info = dict(info or {})
        observation = self._make_observation(info=self._last_info)
        step_info = dict(self._last_info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": gym_action,
                "applied_action_name": action.name,
                "observation": "vizdoom_frame_v1",
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
            raise RuntimeError("DeadlyCorridorEnvAdapter has no current observation; call reset() first")
        metadata = self._metadata(self._last_info)
        self._observed_frames.append(_single_frame_data(self._last_frame))
        return Observation(
            data=self._policy_frame_stack(),
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def render_frame(self) -> Any:
        return self._last_frame

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation": "vizdoom_frame_v1",
            }
        )
        return info

    def close(self) -> None:
        self.gym_env.close()

    def _make_observation(self, info: dict[str, Any]) -> Observation:
        metadata = self._metadata(info)
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def _metadata(self, info: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation": "vizdoom_frame_v1",
            }
        )
        return metadata

    def _policy_frame_stack(self) -> np.ndarray:
        frames = list(self._observed_frames)
        if not frames:
            raise RuntimeError("Deadly Corridor observe() has no current frame; call reset() first")
        if len(frames) < self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames
        frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[-self.frame_stack :]]
        if self.frame_stack == 1:
            return frames[-1]
        axis = 0 if _looks_chw(frames[0]) else -1
        return np.concatenate(frames, axis=axis)


def _single_frame_data(frame: Any) -> np.ndarray:
    value = frame.get("obs") if isinstance(frame, dict) else frame
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected Deadly Corridor image frame with 2 or 3 dims, got {arr.shape!r}")
    return arr


def _looks_chw(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}
