from __future__ import annotations

from typing import Any

from gymnasium.spaces import Box

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter


def _noop_action_from_space(space) -> Any:
    n = getattr(space, "n", None)
    if n is not None:
        return 0
    if isinstance(space, Box):
        import numpy as np

        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"Unsupported action space for Deadly Corridor no-op action: {space}")


class DeadlyCorridorEnvAdapter(EnvAdapter):
    """Latency-bench adapter for ViZDoom Deadly Corridor using the SF Doom env stack."""

    def __init__(
        self,
        env_id: str = "doom_deadly_corridor",
        fps_env: float = 35.0,
        noop_action: Action | None = None,
        env_frameskip: int = 4,
        res_w: int = 128,
        res_h: int = 72,
        wide_aspect_ratio: bool = False,
    ):
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")

        from training.deadly_corridor_sf.integration import (
            DoomDiscrete7ActionWrapper,
            SF_ENV_NAME_DISCRETE7,
            get_deadly_corridor_spec,
            parse_cfg,
            register_deadly_corridor_components,
        )

        register_deadly_corridor_components()
        cfg = parse_cfg(
            [
                "--mode",
                "train",
                "--algo",
                "APPO",
                "--env",
                str(env_id),
                "--experiment",
                "latency_bench_deadly_corridor",
                "--env_frameskip",
                str(int(env_frameskip)),
                "--res_w",
                str(int(res_w)),
                "--res_h",
                str(int(res_h)),
                "--wide_aspect_ratio",
                "True" if wide_aspect_ratio else "False",
                "--queue-latency-values",
                "0",
            ]
        )
        from sample_factory.utils.attr_dict import AttrDict
        from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec

        spec = get_deadly_corridor_spec(str(env_id))
        self.gym_env = make_doom_env_from_spec(
            spec,
            str(env_id),
            cfg,
            AttrDict(worker_index=0, vector_index=0, env_id=0),
            render_mode=None,
        )
        if str(env_id) == SF_ENV_NAME_DISCRETE7:
            self.gym_env = DoomDiscrete7ActionWrapper(self.gym_env)
        self.cfg = cfg
        self.env_id = env_id
        self.fps_env = float(fps_env)
        action_space = self.gym_env.action_space
        noop_value = getattr(self.gym_env, "noop_action", _noop_action_from_space(action_space))
        self.noop_action = noop_action or Action(value=noop_value, name=str(noop_value), is_noop=True)
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
