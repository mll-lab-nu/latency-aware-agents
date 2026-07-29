from __future__ import annotations

import copy
from collections import deque
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Tuple

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from latency_bench.utils.array import looks_chw
from latency_bench.envs.raw_rgb import RawRgbFrameStackBuffer


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
    from training.common.utils import maybe_set_cli_override

    integration.register_deadly_corridor_components()
    base_cfg = integration.SAMPLE_FACTORY_CONFIG_PARSER.parse_eval(
        integration.build_cli_args_from_config(config)
    )
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
    OBSERVATION_TYPE = "vizdoom_frame_v1"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        noop_action: Action | None = None,
        export_env_raw_rgb_frames: bool = False,
    ):
        env_cfg = config["env"]
        env_id = str(env_cfg.get("env_id", "doom_deadly_corridor"))
        env_fps = float(env_cfg.get("env_fps", 35))
        self.frame_stack = int(env_cfg.get("frame_stack", 1) or 1)

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
        self.export_env_raw_rgb_frames = bool(export_env_raw_rgb_frames)
        self._last_info: dict[str, Any] = {}
        self._last_frame: Any = None
        self._observed_frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=self.frame_stack)

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
        self._reset_frame_stack(obs)
        if self.export_env_raw_rgb_frames:
            self._reset_raw_rgb_frame_stack()
        return self._make_observation(info=self._last_info)

    def step(self, action: Action) -> StepResult:
        gym_action = action.value
        obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
        self.env_step += 1
        self._last_frame = obs
        self._last_info = dict(info or {})
        self._append_frame(obs)
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
        return Observation(
            data=self._policy_frame_stack(),
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def render_game_frame(self) -> np.ndarray:
        return np.transpose(self.gym_env.unwrapped.game.get_state().screen_buffer, (1, 2, 0))

    def close(self) -> None:
        self.gym_env.close()

    def _reset_frame_stack(self, frame: Any) -> None:
        self._observed_frames.clear()
        self._append_frame(frame)

    def _append_frame(self, frame: Any) -> None:
        self._observed_frames.append(_single_frame_data(frame))

    def _policy_frame_stack(self) -> np.ndarray:
        frames = list(self._observed_frames)
        if not frames:
            raise RuntimeError("Deadly Corridor observe() has no current frame; call reset() first")
        if len(frames) < self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames
        frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[-self.frame_stack :]]
        if self.frame_stack == 1:
            return frames[-1]
        axis = 0 if looks_chw(frames[0]) else -1
        return np.concatenate(frames, axis=axis)


def _single_frame_data(frame: Any) -> np.ndarray:
    value = frame.get("obs") if isinstance(frame, dict) else frame
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected Deadly Corridor image frame with 2 or 3 dims, got {arr.shape!r}")
    return arr


# Fixed semantic button order the StarVLA multibinary head is trained against.
# Mirrors starVLA.training.rl_games.eval_core._semantic_to_runtime_multibinary.
DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
)


def _deadly_runtime_button_names(gym_env: Any) -> list[str]:
    """Return the live ViZDoom action-button order (ports eval_core helper).

    The MultiBinary action vector is indexed by the game's available-button
    order, which is not guaranteed to equal the semantic order the head emits.
    """

    def _button_name(button: Any) -> str:
        name = getattr(button, "name", None)
        if name is not None:
            return str(name)
        text = str(button)
        return text.split(".")[-1] if "." in text else text

    for candidate in (gym_env, getattr(gym_env, "unwrapped", None)):
        if candidate is None:
            continue
        for attr_name in ("game", "_game"):
            game = getattr(candidate, attr_name, None)
            if game is None:
                continue
            getter = getattr(game, "get_available_buttons", None)
            if getter is None:
                continue
            names = [_button_name(button) for button in getter()]
            if names:
                return names
    return list(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER)


def _semantic_to_runtime_multibinary(semantic_values: list[int], runtime_order: list[str]) -> list[int]:
    semantic_map = {
        name: int(semantic_values[idx])
        for idx, name in enumerate(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER)
        if idx < len(semantic_values)
    }
    return [semantic_map.get(name, 0) for name in runtime_order]


class DeadlyCorridorVlaEnvAdapter(EnvAdapter):
    """Deadly Corridor adapter for StarVLA eval, matching eval_core's env.

    Unlike :class:`DeadlyCorridorEnvAdapter` (sample_factory, factorised action
    tuple), this uses the gymnasium ``VizdoomDeadlyCorridor-MultiBinary`` env so
    the model's multibinary head can fire arbitrary button subsets, exactly like
    ``starVLA.training.rl_games.eval_core``. Native ``frame_skip=1`` is used so
    latency_bench's observation-cadence scheduler owns the obs_stride stepping
    (see ObservationCadenceDecisionScheduler); setting a native skip would
    double-count it.
    """

    OBSERVATION_TYPE = "vizdoom_frame_v1"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        noop_action: Action | None = None,
        export_env_raw_rgb_frames: bool = True,
    ):
        import gymnasium as gym
        import vizdoom.gymnasium_wrapper  # noqa: F401  (registers the env ids)

        env_cfg = config["env"]
        self.env_fps = float(env_cfg.get("env_fps", 35))
        self.frame_stack = int(env_cfg.get("frame_stack", 1) or 1)
        attempts = [
            ("VizdoomDeadlyCorridor-MultiBinary-v1", {}),
            ("VizdoomDeadlyCorridor-MultiBinary-v0", {}),
            ("VizdoomDeadlyCorridor-v1", {"max_buttons_pressed": 0}),
            ("VizdoomDeadlyCorridor-v0", {"max_buttons_pressed": 0}),
        ]
        last_exc: Exception | None = None
        self.gym_env = None
        for env_id, kwargs in attempts:
            try:
                # frame_skip=1: the latency_bench scheduler advances obs_stride raw
                # frames per decision and holds the action between observations.
                self.gym_env = gym.make(env_id, render_mode="rgb_array", frame_skip=1, **kwargs)
                self.env_id = env_id
                break
            except Exception as exc:  # noqa: BLE001 - try the next registered id
                last_exc = exc
        if self.gym_env is None:
            raise RuntimeError(f"Failed to create Deadly Corridor MultiBinary env: {last_exc}")

        self._runtime_button_order = _deadly_runtime_button_names(self.gym_env)
        self._num_buttons = len(self._runtime_button_order)
        self.noop_action = noop_action or Action(
            value=[0] * self._num_buttons, name="NOOP", is_noop=True
        )
        self.export_env_raw_rgb_frames = bool(export_env_raw_rgb_frames)
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_frame: Any = None
        self._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=self.frame_stack)

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        try:
            obs, info = self.gym_env.reset(seed=seed)
        except TypeError:
            obs, info = self.gym_env.reset()
        self._last_frame = obs
        self._last_info = dict(info or {})
        if self.export_env_raw_rgb_frames:
            self._reset_raw_rgb_frame_stack()
        return self._make_observation(info=self._last_info)

    def step(self, action: Action) -> StepResult:
        # action.value is a 7-dim multibinary vector in semantic order; re-order
        # to the live game's button layout before stepping the MultiBinary env.
        semantic = [int(v) for v in np.asarray(action.value).reshape(-1).tolist()]
        expected = len(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER)
        if len(semantic) != expected:
            raise ValueError(
                "DeadlyCorridorVlaEnvAdapter expects a "
                f"{expected}-dim multibinary action in semantic order, got "
                f"{len(semantic)} values ({action.value!r}). This usually means the "
                "policy decoded a non-multibinary layout; ensure the deadly head is "
                "action_layout=multibinary_7 and reached the multibinary decode path."
            )
        runtime_buttons = _semantic_to_runtime_multibinary(semantic, self._runtime_button_order)
        gym_action = np.asarray(runtime_buttons, dtype=np.int8)
        obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
        self.env_step += 1
        self._last_frame = obs
        self._last_info = dict(info or {})
        if self.export_env_raw_rgb_frames and not bool(terminated or truncated):
            self._append_raw_rgb_frame()
        observation = self._make_observation(info=self._last_info)
        step_info = dict(self._last_info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": runtime_buttons,
                "applied_action_name": action.name,
                "observation": self.OBSERVATION_TYPE,
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
        return self._make_observation(info=self._last_info)

    def render_game_frame(self) -> np.ndarray:
        frame = self.gym_env.render()
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self.gym_env.close()
