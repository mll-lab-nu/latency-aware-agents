from __future__ import annotations

import copy
from typing import Any

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
        fps = int(float(env_cfg.get("fps_env", default)))
    except (TypeError, ValueError) as exc:
        raise ValueError("fps_env must be positive") from exc
    if fps <= 0:
        raise ValueError("fps_env must be positive")
    return fps


def _build_sample_factory_eval_cfg(config: dict[str, Any]) -> Any:
    from training.deadly_corridor_sf import integration
    from training.sample_factory_common.common import maybe_set_cli_override

    integration.register_deadly_corridor_components()
    base_cfg = integration.parse_cfg(integration.build_cli_args_from_config(config))
    eval_fps = _synchronous_eval_fps_from_config(config)
    cfg = integration._prepare_eval_cfg(base_cfg, latency_value=0)
    if _requires_sample_factory_checkpoint_config(config):
        from sample_factory.cfg.arguments import load_from_checkpoint

        cfg = load_from_checkpoint(cfg)

    for key in (
        "seed",
        "env_frameskip",
        "res_w",
        "res_h",
        "wide_aspect_ratio",
        "frame_stack",
    ):
        if hasattr(base_cfg, key):
            maybe_set_cli_override(cfg, key, getattr(base_cfg, key))
    explicit_max_episode_steps = int(getattr(base_cfg, "max_episode_steps", 0) or 0)
    if explicit_max_episode_steps > 0:
        maybe_set_cli_override(cfg, "max_episode_steps", explicit_max_episode_steps)
    else:
        eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 0) or 0)
        if eval_max_steps > 0:
            maybe_set_cli_override(cfg, "max_episode_steps", eval_max_steps)

    env_frameskip = int(getattr(cfg, "env_frameskip", getattr(base_cfg, "env_frameskip", 4)) or 1)
    maybe_set_cli_override(cfg, "mode", "eval")
    maybe_set_cli_override(cfg, "queue_latency_values", "0")
    maybe_set_cli_override(cfg, "latency_mode", "episode_fixed")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    maybe_set_cli_override(cfg, "eval_env_frameskip", env_frameskip)
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
        fps_env = float(env_cfg.get("fps_env", 35))
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")

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
        self.fps_env = float(fps_env)
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

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        if seed is not None:
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
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation": "vizdoom_frame_v1",
            }
        )
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )
