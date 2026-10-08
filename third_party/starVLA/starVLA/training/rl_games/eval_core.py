from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from starVLA.training.rl_games.action_decode import (
    DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER,
    decode_defend_the_line_multibinary,
    decode_defend_the_line_tuple,
    deadly_tuple_to_semantic_buttons,
    decode_deadly_factorized_11,
    decode_deadly_multibinary_7,
    decode_discrete_argmax,
    decode_rl_games_actions,
    resolve_asterix_action_decode_spec,
    resolve_deadly_action_decode_spec,
)

TASK_SEED_INDEX = {
    "flappy": 0,
    "demon_attack": 1,
    "deadly_corridor": 2,
    "defend_the_line": 3,
    "asterix": 4,
    "atlantis": 5,
    "air_raid": 6,
}

DEFAULT_VIDEO_FPS = 30


class ActionLatencyQueue:
    def __init__(self, latency: int, default_action):
        self.latency = max(0, int(latency))
        self.default_action = default_action
        self.current_action = default_action
        self.pending = []
        self.step_count = 0

    def reset(self):
        self.current_action = self.default_action
        self.pending = []
        self.step_count = 0

    def schedule_and_get(self, action):
        if self.latency == 0:
            self.step_count += 1
            return action
        apply_at = self.step_count + self.latency
        self.pending.append((apply_at, action))
        while self.pending and self.pending[0][0] <= self.step_count:
            _, matured = self.pending.pop(0)
            self.current_action = matured
        self.step_count += 1
        return self.current_action


class AtariResetWrapper:
    def __init__(self, env: Any, noop_max: int, *, fire_reset: bool = False):
        self.env = env
        self.noop_max = int(noop_max)
        self.fire_reset = bool(fire_reset)

    def reset(self, **kwargs: Any) -> tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        if self.noop_max <= 0:
            pass
        else:
            np_random = getattr(self.env, "np_random", None)
            if np_random is None or not hasattr(np_random, "integers"):
                raise RuntimeError("Atari no-op reset requires env.np_random with an integers method")

            noops = int(np_random.integers(1, self.noop_max + 1))
            for _ in range(noops):
                obs, _, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        if self.fire_reset:
            for fire_action in (1, 2):
                obs, _, terminated, truncated, info = self.env.step(fire_action)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def render(self) -> Any:
        return self.env.render()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


# Backward-compatible name used by downstream alignment tests.
DemonAttackNoopResetWrapper = AtariResetWrapper


@dataclass
class EvalResult:
    per_latency: Dict[str, Dict]
    aggregate: Dict
    path: str | None = None


def _extract_image_observation(obs: Any) -> Any:
    if isinstance(obs, dict):
        for key in (
            "screen",
            "rgb",
            "image",
            "obs",
            "observation",
            "pixels",
        ):
            value = obs.get(key)
            if value is not None and np.asarray(value).ndim >= 2:
                return value
        for value in obs.values():
            if hasattr(value, "shape") and np.asarray(value).ndim >= 2:
                return value
        raise TypeError(f"dict observation has no image-like value; keys={sorted(obs.keys())}")
    return obs


def _resize_rgb(raw_obs: Any, image_size: Optional[int], *, prefer_area: bool = False) -> np.ndarray:
    raw_obs = _extract_image_observation(raw_obs)
    if raw_obs is None:
        if image_size is None:
            raise ValueError("Cannot create an empty RGB observation when image_size is None")
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    raw_obs = np.asarray(raw_obs)
    if raw_obs.dtype != np.uint8:
        raw_obs = raw_obs.astype(np.uint8)
    if raw_obs.ndim == 2:
        raw_obs = np.stack([raw_obs] * 3, axis=-1)
    elif raw_obs.ndim == 3 and raw_obs.shape[2] == 1:
        raw_obs = np.repeat(raw_obs, 3, axis=2)
    if image_size is None:
        return raw_obs
    if raw_obs.shape[:2] == (image_size, image_size):
        return raw_obs
    if prefer_area:
        try:
            import cv2  # noqa: WPS433

            return cv2.resize(raw_obs, (image_size, image_size), interpolation=cv2.INTER_AREA)
        except ImportError:
            pass
    return np.array(Image.fromarray(raw_obs).resize((image_size, image_size), Image.BICUBIC))


def _load_latency_prompt_map(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    out: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        latency = int(value.get("latency", key))
        out[latency] = {
            "latency": latency,
            "prompt": str(value["prompt"]),
            "latency_ms": value.get("latency_ms", None),
        }
    return out


def _cfg_get(container: Any, key: str, default_value: Any) -> Any:
    if container is None:
        return default_value
    if isinstance(container, Mapping):
        return container.get(key, default_value)
    try:
        return container[key]
    except (AttributeError, KeyError, TypeError):
        return getattr(container, key, default_value)


def _plain_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            converted = OmegaConf.to_container(value, resolve=True)
            return dict(converted or {}) if isinstance(converted, Mapping) else {}
    except Exception:
        pass
    return dict(value) if isinstance(value, Mapping) else {}


def _get_available_button_names(env) -> List[str]:
    def _button_name(button):
        name = getattr(button, "name", None)
        if name:
            return str(name)
        text = str(button)
        if "." in text:
            text = text.split(".")[-1]
        return text

    for candidate in (env, getattr(env, "unwrapped", None)):
        if candidate is None:
            continue
        for attr_name in ("game", "_game"):
            game = getattr(candidate, attr_name, None)
            if game is None:
                continue
            getter = getattr(game, "get_available_buttons", None)
            if getter is None:
                continue
            buttons = getter()
            names = [_button_name(button) for button in buttons]
            if names:
                return names
    return list(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER)


def _semantic_to_runtime_multibinary(semantic_values: List[int], runtime_order: List[str]) -> List[int]:
    semantic_map = {
        name: int(semantic_values[idx])
        for idx, name in enumerate(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER)
    }
    return [semantic_map.get(name, 0) for name in runtime_order]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_prompt_mode(prompt: str, prompt_mode: Any) -> str:
    mode = str(prompt_mode or "").strip().lower()
    if mode in {"", "default", "none", "raw"}:
        return prompt
    if mode == "latency_neutral":
        marker = " Current action latency is "
        head, found, _tail = str(prompt).partition(marker)
        return head.rstrip() if found else str(prompt)
    raise ValueError(f"Unsupported rl_games.env_eval.prompt_mode={prompt_mode!r}")


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _stable_task_seed_index(task: str) -> int:
    if task in TASK_SEED_INDEX:
        return TASK_SEED_INDEX[task]
    return 100 + sum((idx + 1) * ord(char) for idx, char in enumerate(task)) % 10000


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _rgb_frame(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _render_frame(env, fallback_obs: Any = None) -> np.ndarray | None:
    frame = None
    try:
        frame = env.render()
    except Exception:
        frame = None
    if frame is None:
        frame = fallback_obs
    try:
        return _rgb_frame(_extract_image_observation(frame))
    except Exception:
        return _rgb_frame(frame)


class _LiveImageTransform:
    def __init__(self, *, task: str, config: Dict[str, Any]):
        self.task = task
        self.config = dict(config)
        self.name = str(self.config.get("image_transform", "raw_rgb") or "raw_rgb").strip().lower()
        self.enabled = self.name not in {"", "none", "raw", "raw_rgb"}
        if self.enabled and self.name not in {"flappy_ghost_trail", "demon_attack_ghost_trail"}:
            raise ValueError(f"Unsupported rl_games.env_eval.image_transform={self.name!r}")
        if self.enabled and self.name == "flappy_ghost_trail" and self.task != "flappy":
            raise ValueError("image_transform=flappy_ghost_trail is only supported for task=flappy")
        if self.enabled and self.name == "demon_attack_ghost_trail" and self.task != "demon_attack":
            raise ValueError("image_transform=demon_attack_ghost_trail is only supported for task=demon_attack")
        self.history_frames = max(0, int(self.config.get("history_frames", 5)))
        self._frames: List[np.ndarray] = []
        self._steps: List[int] = []
        self._reset_frame: np.ndarray | None = None  # demon_attack_ghost_trail only

    def reset(self, frame: np.ndarray | None, *, step: int = 0) -> None:
        self._frames = []
        self._steps = []
        if self.name == "demon_attack_ghost_trail":
            # Don't seed _frames with the reset frame — the ghost trail is built only
            # from real action-result frames (same as training deque which starts empty).
            # Store it separately as a fallback for current() at step 0.
            self._reset_frame = np.asarray(frame, dtype=np.uint8) if frame is not None else None
        else:
            self.append(frame, step=step)

    def append(self, frame: np.ndarray | None, *, step: int) -> None:
        if not self.enabled or frame is None:
            return
        self._frames.append(np.asarray(frame, dtype=np.uint8))
        self._steps.append(int(step))
        # demon_attack_ghost_trail accumulates one raw ALE frame per call (frameskip=1
        # env stepped 4× per decision), so cap at exactly history_frames raw frames.
        max_frames = self.history_frames if self.name == "demon_attack_ghost_trail" else self.history_frames + 1
        if len(self._frames) > max_frames:
            self._frames = self._frames[-max_frames:]
            self._steps = self._steps[-max_frames:]

    def current(self, fallback_obs: Any) -> Any:
        if not self.enabled:
            return fallback_obs
        if not self._frames:
            # demon_attack_ghost_trail at step 0: no raw frames yet, return raw reset frame
            return self._reset_frame if self._reset_frame is not None else fallback_obs
        # latency_bench is optional for standalone StarVLA evaluation without this feature.
        from latency_bench.data.ghost_trail import (
            GhostTrailConfig, build_flappy_ghost_trail_window, build_demon_attack_ghost_trail_window,
        )

        config = GhostTrailConfig(
            image_transform=self.name,
            history_frames=self.history_frames,
            gamma=float(self.config.get("gamma", 1.3)),
            min_alpha=int(self.config.get("min_alpha", 35)),
            ground_fraction=float(self.config.get("ground_fraction", 0.22)),
            scroll_px_per_step=float(self.config.get("scroll_px_per_step", 4.0)),
        )
        if self.name == "demon_attack_ghost_trail":
            return build_demon_attack_ghost_trail_window(self._frames)
        return build_flappy_ghost_trail_window(self._frames, self._steps, config=config)


def _write_video(path: Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to record RL-games eval videos") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=int(fps))


class _TaskEvaluator:
    def __init__(self, task: str, cfg, task_eval_cfg=None, stage_eval_cfg=None):
        self.task = task
        self.cfg = cfg
        self.env_eval_cfg = cfg.rl_games.env_eval
        self.task_eval_cfg = task_eval_cfg
        self.stage_eval_cfg = stage_eval_cfg
        self.default_prompt = str(getattr(self.env_eval_cfg, "task_description", "") or "")
        if task_eval_cfg is not None:
            self.default_prompt = str(getattr(task_eval_cfg, "task_description", self.default_prompt) or self.default_prompt)
        image_size_value = (
            getattr(task_eval_cfg, "image_size", getattr(self.env_eval_cfg, "image_size", 224))
            if task_eval_cfg is not None
            else getattr(self.env_eval_cfg, "image_size", 224)
        )
        self.image_size = _as_optional_int(image_size_value)
        self.frameskip = max(1, int(
            getattr(task_eval_cfg, "frameskip", getattr(self.env_eval_cfg, "frameskip", 1))
            if task_eval_cfg is not None
            else getattr(self.env_eval_cfg, "frameskip", 1)
        ))
        vla_data_cfg = getattr(getattr(cfg, "datasets", None), "vla_data", None)
        self.include_state = bool(getattr(vla_data_cfg, "include_state", False))
        self.pack_image_sequence = _as_bool(getattr(vla_data_cfg, "pack_image_sequence", False), default=False)
        self.image_sequence_length = _as_int(getattr(vla_data_cfg, "image_sequence_length", None), 1)
        if self.image_sequence_length <= 0:
            raise ValueError(f"image_sequence_length must be positive, got {self.image_sequence_length}")
        self.state_dim = int(getattr(cfg.framework.action_model, "state_dim", 1) or 1)
        chunk_execution_cfg = getattr(self.env_eval_cfg, "action_chunk_execution", None)
        self.action_chunk_execution_enabled = _as_bool(
            getattr(chunk_execution_cfg, "enabled", False),
            default=False,
        )
        action_horizon = int(getattr(cfg.framework.action_model, "action_horizon", 1) or 1)
        configured_chunk_size = getattr(chunk_execution_cfg, "chunk_size", None)
        self.action_chunk_execution_size = action_horizon if configured_chunk_size is None else int(configured_chunk_size)
        if self.action_chunk_execution_enabled and self.action_chunk_execution_size <= 0:
            raise ValueError(
                f"action_chunk_execution.chunk_size must be positive when enabled, "
                f"got {self.action_chunk_execution_size}"
            )
        self.fixed_episode_seeds = _as_bool(getattr(self.env_eval_cfg, "fixed_episode_seeds", True), default=True)
        self.eval_seed = _as_int(getattr(self.env_eval_cfg, "seed", getattr(cfg, "seed", 42)), 42)
        self.latency_seed_stride = _as_int(getattr(self.env_eval_cfg, "latency_seed_stride", None), 0)
        self.task_seed_stride = _as_int(getattr(self.env_eval_cfg, "task_seed_stride", None), 0)
        self.image_transform_config = self._resolve_image_transform_config()
        self.prompt_mode = self._resolve_prompt_mode()

    def _resolve_image_transform_config(self) -> Dict[str, Any]:
        image_transform = str(
            _cfg_get(
                self.stage_eval_cfg,
                "image_transform",
                _cfg_get(self.task_eval_cfg, "image_transform", _cfg_get(self.env_eval_cfg, "image_transform", "raw_rgb")),
            )
            or "raw_rgb"
        ).strip().lower()
        ghost_cfg = {
            **_plain_mapping(_cfg_get(self.env_eval_cfg, "ghost_trail", None)),
            **_plain_mapping(_cfg_get(self.task_eval_cfg, "ghost_trail", None)),
            **_plain_mapping(_cfg_get(self.stage_eval_cfg, "ghost_trail", None)),
        }
        default_history_frames = 30 if image_transform == "demon_attack_ghost_trail" else 5
        return {
            "image_transform": image_transform,
            "history_frames": int(ghost_cfg.get("history_frames", default_history_frames)),
            "gamma": float(ghost_cfg.get("gamma", 1.3)),
            "min_alpha": int(ghost_cfg.get("min_alpha", 35)),
            "ground_fraction": float(ghost_cfg.get("ground_fraction", 0.22)),
            "scroll_px_per_step": float(ghost_cfg.get("scroll_px_per_step", 4.0)),
        }

    def _resolve_prompt_mode(self) -> str:
        stage_mode = _cfg_get(self.stage_eval_cfg, "prompt_mode", None)
        task_mode = _cfg_get(self.task_eval_cfg, "prompt_mode", None)
        env_mode = _cfg_get(self.env_eval_cfg, "prompt_mode", "default")
        mode = stage_mode if stage_mode is not None else task_mode if task_mode is not None else env_mode
        return str(mode or "default").strip().lower()

    def _make_image_transform(self) -> _LiveImageTransform:
        return _LiveImageTransform(task=self.task, config=self.image_transform_config)

    def _make_atari_env(self, default_env_id: str):
        import ale_py  # noqa: F401
        import gymnasium as gym

        atari_cfg = getattr(self.env_eval_cfg, "atari", None)
        env_id = str(getattr(atari_cfg, "env_id", None) or default_env_id)
        noop_max = _as_int(getattr(atari_cfg, "noop_max", None), 30)
        gym_frameskip = _as_int(getattr(atari_cfg, "gym_frameskip", None), self.frameskip)
        mode = _as_int(getattr(atari_cfg, "mode", None), 0)
        difficulty = _as_int(getattr(atari_cfg, "difficulty", None), 0)
        repeat_action_probability = float(getattr(atari_cfg, "repeat_action_probability", 0.0))
        full_action_space = _as_bool(getattr(atari_cfg, "full_action_space", False), default=False)
        fire_reset = _as_bool(getattr(atari_cfg, "fire_reset", False), default=False)
        gym_kwargs = {
            "obs_type": "rgb",
            "frameskip": gym_frameskip,
            "repeat_action_probability": repeat_action_probability,
            "full_action_space": full_action_space,
            "mode": mode,
            "difficulty": difficulty,
            "render_mode": "rgb_array",
        }
        max_episode_frames = getattr(atari_cfg, "max_num_frames_per_episode", None)
        if max_episode_frames not in (None, ""):
            gym_kwargs["max_num_frames_per_episode"] = int(max_episode_frames)
        env = gym.make(
            env_id,
            **gym_kwargs,
        )
        return AtariResetWrapper(env=env, noop_max=noop_max, fire_reset=fire_reset)

    def _make_env(self):
        if self.task == "flappy":
            import flappy_bird_gymnasium  # noqa: F401
            import gymnasium as gym

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            return gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

        if self.task == "demon_attack":
            import ale_py  # noqa: F401
            import gymnasium as gym

            demon_attack_cfg = getattr(self.env_eval_cfg, "demon_attack", None)
            noop_max = _as_int(getattr(demon_attack_cfg, "noop_max", None), 30)
            attempts = [
                "ALE/DemonAttack-v5",
                "ALE/DemonAttack-v4",
            ]
            last_exc = None
            for env_id in attempts:
                try:
                    env = gym.make(
                        env_id,
                        frameskip=1 if self._is_demon_ghost_trail() else self.frameskip,
                        repeat_action_probability=0.0,
                        render_mode="rgb_array",
                    )
                    return AtariResetWrapper(env=env, noop_max=noop_max)
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"Failed to create DemonAttack env: {last_exc}")

        if self.task == "deadly_corridor":
            import gymnasium as gym
            import vizdoom.gymnasium_wrapper  # noqa: F401

            attempts = [
                ("VizdoomDeadlyCorridor-MultiBinary-v1", {}),
                ("VizdoomDeadlyCorridor-MultiBinary-v0", {}),
                ("VizdoomDeadlyCorridor-v1", {"max_buttons_pressed": 0}),
                ("VizdoomDeadlyCorridor-v0", {"max_buttons_pressed": 0}),
            ]
            last_exc = None
            for env_id, kwargs in attempts:
                try:
                    return gym.make(
                        env_id,
                        render_mode="rgb_array",
                        frame_skip=self.frameskip,
                        **kwargs,
                    )
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"Failed to create Deadly Corridor env: {last_exc}")

        if self.task == "defend_the_line":
            import gymnasium as gym
            import vizdoom.gymnasium_wrapper  # noqa: F401

            attempts = [
                ("VizdoomDefendLine-MultiBinary-v1", {}),
                ("VizdoomDefendLine-MultiBinary-v0", {}),
                ("VizdoomDefendLine-v1", {"max_buttons_pressed": 0}),
                ("VizdoomDefendLine-v0", {"max_buttons_pressed": 0}),
            ]
            last_exc = None
            for env_id, kwargs in attempts:
                try:
                    return gym.make(
                        env_id,
                        render_mode="rgb_array",
                        frame_skip=self.frameskip,
                        **kwargs,
                    )
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"Failed to create Defend the Line env: {last_exc}")

        if self.task == "asterix":
            return self._make_atari_env("ALE/Asterix-v5")

        if self.task == "atlantis":
            return self._make_atari_env("ALE/Atlantis-v5")

        if self.task == "air_raid":
            return self._make_atari_env("ALE/AirRaid-v5")

        raise ValueError(f"Unsupported task: {self.task}")

    def _is_demon_ghost_trail(self) -> bool:
        return (
            self.task == "demon_attack"
            and str(self.image_transform_config.get("image_transform", "")).strip().lower()
            == "demon_attack_ghost_trail"
        )

    def _uses_native_frameskip(self) -> bool:
        # demon_attack ghost trail runs the gym env at frameskip=1 and handles
        # action repetition manually so all 4 raw frames are captured per decision.
        if self._is_demon_ghost_trail():
            return False
        return self.task in {"demon_attack", "deadly_corridor", "defend_the_line", "asterix", "atlantis"}

    def _model_observation(self, env, obs):
        if self.task == "flappy":
            frame = env.render()
            if frame is None:
                return obs
            return frame
        if self.task in {"asterix", "atlantis", "air_raid"}:
            frame = env.render()
            if frame is None:
                frame = obs
            return _resize_rgb(frame, image_size=84, prefer_area=True)
        return obs

    def _decode_action(self, raw_action: np.ndarray, runtime_button_order: Optional[List[str]] = None):
        if self.task == "flappy":
            return decode_discrete_argmax(raw_action, 2)
        if self.task == "demon_attack":
            return decode_discrete_argmax(raw_action, 6)
        if self.task == "air_raid":
            return decode_discrete_argmax(raw_action, 6)
        if self.task == "asterix":
            asterix_cfg = getattr(self.env_eval_cfg, "asterix", None)
            action_layout = resolve_asterix_action_decode_spec(
                self.cfg,
                action_layout=str(getattr(asterix_cfg, "action_layout", "discrete_9")),
            )
            decoded = decode_rl_games_actions(
                normalized_actions=raw_action,
                env_name=self.task,
                asterix_action_layout=action_layout,
            )
            return int(np.asarray(decoded["actions"]).reshape(-1)[0])
        if self.task == "atlantis":
            return decode_discrete_argmax(raw_action, 4)
        if self.task == "defend_the_line":
            return decode_defend_the_line_multibinary(raw_action)
        if self.task == "deadly_corridor":
            deadly_cfg = self.env_eval_cfg.deadly
            action_layout, multibinary_threshold = resolve_deadly_action_decode_spec(
                self.cfg,
                action_layout=str(getattr(deadly_cfg, "action_layout", "multibinary_7")),
                multibinary_threshold=getattr(deadly_cfg, "multibinary_threshold", None),
            )
            decoded = decode_rl_games_actions(
                normalized_actions=raw_action,
                env_name=self.task,
                deadly_action_layout=action_layout,
                deadly_multibinary_threshold=multibinary_threshold,
            )
            semantic = np.asarray(decoded["actions"])
            if decoded["action_output_type"] == "rl_games_deadly_corridor_tuple":
                semantic = deadly_tuple_to_semantic_buttons(semantic)
            semantic_values = semantic.tolist()
            if runtime_button_order is None:
                return semantic_values
            return _semantic_to_runtime_multibinary(semantic_values, runtime_button_order)
        return decode_discrete_argmax(raw_action, raw_action.shape[-1])

    def _episode_seed(self, latency: int, episode: int) -> int | None:
        if not self.fixed_episode_seeds:
            return None
        task_offset = _stable_task_seed_index(self.task) * self.task_seed_stride
        latency_offset = int(latency) * self.latency_seed_stride
        return int(self.eval_seed + task_offset + latency_offset + int(episode))

    def _episode_seed_for_run(
        self,
        latency: int,
        episode: int,
        seed_overrides: Dict[int, int | None] | None = None,
    ) -> int | None:
        if seed_overrides is not None and int(episode) in seed_overrides:
            seed = seed_overrides[int(episode)]
            return None if seed is None else int(seed)
        return self._episode_seed(latency=latency, episode=episode)

    def _reset_env(self, env, seed: int | None):
        if seed is None:
            return env.reset()
        try:
            return env.reset(seed=int(seed))
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(int(seed))
            return env.reset()

    def _vectorized_batch_size(self) -> int:
        vectorized_cfg = getattr(self.env_eval_cfg, "vectorized", None)
        if not _as_bool(getattr(vectorized_cfg, "enabled", False), default=False):
            return 1
        return max(1, _as_int(getattr(vectorized_cfg, "batch_size", 1), 1))

    def _queue_default_action(self):
        if self.task == "defend_the_line":
            return [0, 0, 0]
        return [0] * 7 if self.task == "deadly_corridor" else 0

    def _reset_model_memory(self, model, slot_id: int) -> None:
        reset = getattr(model, "reset_memory", None)
        if callable(reset):
            reset(slot_id)

    def _initial_model_history(self, model_obs) -> List[Any]:
        return [model_obs for _ in range(self.image_sequence_length)]

    def _advance_model_history(self, model_history: List[Any], model_obs) -> List[Any]:
        if len(model_history) != self.image_sequence_length:
            raise ValueError(
                f"Expected model history length={self.image_sequence_length}, got {len(model_history)}"
            )
        return [*model_history[1:], model_obs]

    def _make_model_example(self, model_obs, prompt: str, slot_id: int) -> Dict[str, Any]:
        if self.pack_image_sequence:
            model_frames = list(model_obs)
            if len(model_frames) != self.image_sequence_length:
                raise ValueError(
                    f"Expected {self.image_sequence_length} model frames, got {len(model_frames)}"
                )
            images = [
                Image.fromarray(
                    _resize_rgb(
                        frame,
                        image_size=self.image_size,
                        prefer_area=self.task == "flappy",
                    )
                )
                for frame in model_frames
            ]
        else:
            obs_rgb = _resize_rgb(
                model_obs,
                image_size=self.image_size,
                prefer_area=self.task == "flappy",
            )
            images = [Image.fromarray(obs_rgb)]
        example = {
            "image": images,
            "lang": prompt,
            "slot_id": int(slot_id),
        }
        if self.include_state:
            example["state"] = np.zeros((1, self.state_dim), dtype=np.float32)
        return example

    def _step_env_once(self, env, action):
        step_reward = 0.0
        terminated = False
        truncated = False
        repeat_count = 1 if self._uses_native_frameskip() else self.frameskip
        for _ in range(repeat_count):
            obs, reward, terminated, truncated, _ = env.step(action)
            step_reward += float(reward)
            if terminated or truncated:
                break
        return obs, step_reward, terminated, truncated

    def _step_demon_ghost_trail(self, env, action, image_transform: "_LiveImageTransform", decision_step: int):
        """Step demon_attack env (frameskip=1) self.frameskip times, appending each raw frame."""
        step_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self.frameskip):
            obs, reward, terminated, truncated, _ = env.step(action)
            step_reward += float(reward)
            raw_frame = _render_frame(env, fallback_obs=obs)
            image_transform.append(raw_frame, step=decision_step)
            if terminated or truncated:
                break
        return obs, step_reward, terminated, truncated

    def _validate_action_chunk(self, actions: np.ndarray, batch_size: int) -> None:
        if actions.ndim != 3:
            raise RuntimeError(f"Expected normalized_actions shape [B,T,D], got {actions.shape}")
        if actions.shape[0] != batch_size:
            raise RuntimeError(
                f"Expected {batch_size} batched actions, got normalized_actions shape {actions.shape}"
            )
        if self.action_chunk_execution_enabled and actions.shape[1] < self.action_chunk_execution_size:
            raise RuntimeError(
                f"action_chunk_execution.chunk_size={self.action_chunk_execution_size} requires at least "
                f"{self.action_chunk_execution_size} predicted action slots, got normalized_actions shape {actions.shape}"
            )

    def _empty_latency_metrics(self, latency: int) -> Dict[str, Any]:
        return {
            "latency": int(latency),
            "num_episodes": 0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "episode_rewards": [],
            "episode_lengths": [],
            "decoded_action_hist": {},
            "fixed_episode_seeds": bool(self.fixed_episode_seeds),
            "eval_seed": int(self.eval_seed),
            "episode_seeds": [],
            "episode_indices": [],
        }

    def _build_latency_metrics(
        self,
        latency: int,
        num_episodes: int,
        rewards: List[float],
        lengths: List[int],
        action_hist: Counter,
        episode_seeds: List[int | None],
        episode_indices: Optional[List[int]] = None,
        vectorized_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        std_length = float(np.std(lengths)) if lengths else 0.0
        metrics = {
            "latency": int(latency),
            "num_episodes": int(num_episodes),
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "std_reward": std_reward,
            "std_length": std_length,
            "episode_rewards": [float(reward) for reward in rewards],
            "episode_lengths": [int(length) for length in lengths],
            "decoded_action_hist": dict(action_hist),
            "fixed_episode_seeds": bool(self.fixed_episode_seeds),
            "eval_seed": int(self.eval_seed),
            "episode_seeds": episode_seeds,
        }
        if episode_indices is not None:
            metrics["episode_indices"] = [int(episode) for episode in episode_indices]
        if vectorized_batch_size is not None:
            metrics["vectorized_eval"] = True
            metrics["vectorized_batch_size"] = int(vectorized_batch_size)
        return metrics

    def _run_latency_vectorized(
        self,
        model,
        latency: int,
        prompt: str,
        max_steps: int,
        num_episodes: int,
        episode_indices: List[int],
        progress_desc: str | None,
        progress_position: int,
        batch_size: int,
        on_episode_complete: Callable[[Dict[str, Any]], None] | None = None,
        seed_overrides: Dict[int, int | None] | None = None,
    ) -> Dict[str, Any]:
        pending = list(episode_indices)
        if not pending:
            return self._empty_latency_metrics(latency)

        rewards_by_episode: Dict[int, float] = {}
        lengths_by_episode: Dict[int, int] = {}
        seeds_by_episode: Dict[int, int | None] = {}
        action_hist = Counter()
        slots: List[Dict[str, Any]] = []

        def _start_episode(slot: Dict[str, Any], episode: int) -> None:
            episode_seed = self._episode_seed_for_run(
                latency=latency,
                episode=episode,
                seed_overrides=seed_overrides,
            )
            self._reset_model_memory(model, slot["slot_id"])
            obs, _ = self._reset_env(slot["env"], episode_seed)
            model_obs = self._model_observation(slot["env"], obs)
            image_transform = self._make_image_transform()
            initial_frame = _render_frame(slot["env"], fallback_obs=model_obs) if image_transform.enabled else None
            image_transform.reset(initial_frame, step=0)
            slot.update({
                "active": True,
                "episode": int(episode),
                "seed": episode_seed,
                "model_obs": (
                    self._initial_model_history(model_obs)
                    if self.pack_image_sequence
                    else model_obs
                ),
                "image_transform": image_transform,
                "queue": ActionLatencyQueue(latency=latency, default_action=self._queue_default_action()),
                "reward": 0.0,
                "steps": 0,
            })
            seeds_by_episode[int(episode)] = episode_seed

        try:
            for slot_id in range(min(max(1, int(batch_size)), len(pending))):
                env = self._make_env()
                slot = {
                    "env": env,
                    "slot_id": int(slot_id),
                    "runtime_button_order": _get_available_button_names(env) if self.task == "deadly_corridor" else None,
                    "active": False,
                }
                _start_episode(slot, pending.pop(0))
                slots.append(slot)

            progress = tqdm(
                total=len(episode_indices),
                desc=progress_desc or f"{self.task} latency={latency}",
                leave=False,
                position=progress_position,
                dynamic_ncols=True,
            )
            try:
                while any(slot.get("active", False) for slot in slots):
                    active_slots = [slot for slot in slots if slot.get("active", False)]
                    examples = [
                        self._make_model_example(
                            (
                                slot["model_obs"]
                                if self.pack_image_sequence
                                else slot["image_transform"].current(slot["model_obs"])
                            ),
                            prompt,
                            slot["slot_id"],
                        )
                        for slot in active_slots
                    ]
                    output = model.predict_action(examples=examples)
                    actions = output["normalized_actions"]
                    if actions.ndim != 3:
                        raise RuntimeError(f"Expected normalized_actions shape [B,T,D], got {actions.shape}")
                    if actions.shape[0] != len(active_slots):
                        raise RuntimeError(
                            f"Expected {len(active_slots)} batched actions, got normalized_actions shape {actions.shape}"
                        )

                    for slot, raw_action in zip(active_slots, actions[:, 0, :]):
                        decoded = self._decode_action(
                            raw_action,
                            runtime_button_order=slot["runtime_button_order"],
                        )
                        effective_action = slot["queue"].schedule_and_get(decoded)
                        action_hist[str(effective_action)] += 1

                        next_step = int(slot["steps"]) + 1
                        if self._is_demon_ghost_trail():
                            obs, step_reward, terminated, truncated = self._step_demon_ghost_trail(
                                slot["env"], effective_action, slot["image_transform"], next_step
                            )
                        else:
                            obs, step_reward, terminated, truncated = self._step_env_once(slot["env"], effective_action)
                            fallback_obs = slot["model_obs"][-1] if self.pack_image_sequence else slot["model_obs"]
                            frame = (
                                _render_frame(slot["env"], fallback_obs=fallback_obs)
                                if slot["image_transform"].enabled
                                else None
                            )
                            slot["image_transform"].append(frame, step=next_step)
                        next_model_obs = self._model_observation(slot["env"], obs)
                        slot["model_obs"] = (
                            self._advance_model_history(slot["model_obs"], next_model_obs)
                            if self.pack_image_sequence
                            else next_model_obs
                        )
                        slot["reward"] += step_reward
                        slot["steps"] = next_step
                        done = terminated or truncated or slot["steps"] >= max_steps

                        if done:
                            episode = int(slot["episode"])
                            rewards_by_episode[episode] = float(slot["reward"])
                            lengths_by_episode[episode] = int(slot["steps"])
                            progress.update(1)
                            progress.set_postfix({
                                "reward": f"{slot['reward']:.2f}",
                                "steps": int(slot["steps"]),
                                "batch": len(active_slots),
                            })
                            if on_episode_complete is not None:
                                completed_episodes = [
                                    int(item)
                                    for item in episode_indices
                                    if int(item) in rewards_by_episode
                                ]
                                on_episode_complete(
                                    self._build_latency_metrics(
                                        latency=latency,
                                        num_episodes=len(completed_episodes),
                                        rewards=[rewards_by_episode[item] for item in completed_episodes],
                                        lengths=[lengths_by_episode[item] for item in completed_episodes],
                                        action_hist=action_hist,
                                        episode_seeds=[seeds_by_episode[item] for item in completed_episodes],
                                        episode_indices=completed_episodes,
                                        vectorized_batch_size=batch_size,
                                    )
                                )
                            if pending:
                                _start_episode(slot, pending.pop(0))
                            else:
                                slot["active"] = False
            finally:
                progress.close()
        finally:
            for slot in slots:
                slot["env"].close()

        ordered_episodes = [int(episode) for episode in episode_indices]
        rewards = [float(rewards_by_episode[episode]) for episode in ordered_episodes]
        lengths = [int(lengths_by_episode[episode]) for episode in ordered_episodes]
        episode_seeds = [seeds_by_episode[episode] for episode in ordered_episodes]
        return self._build_latency_metrics(
            latency=latency,
            num_episodes=num_episodes,
            rewards=rewards,
            lengths=lengths,
            action_hist=action_hist,
            episode_seeds=episode_seeds,
            episode_indices=ordered_episodes,
            vectorized_batch_size=batch_size,
        )

    def run_latency(
        self,
        model,
        latency: int,
        prompt: str,
        max_steps: int,
        num_episodes: int,
        episode_indices: Optional[List[int]] = None,
        progress_desc: str | None = None,
        progress_position: int = 0,
        on_episode_complete: Callable[[Dict[str, Any]], None] | None = None,
        video_dir: str | Path | None = None,
        video_fps: int = DEFAULT_VIDEO_FPS,
        seed_overrides: Dict[int, int | None] | None = None,
    ) -> Dict[str, Any]:
        episode_indices = list(range(num_episodes)) if episode_indices is None else list(episode_indices)
        if not episode_indices:
            return self._empty_latency_metrics(latency)

        vectorized_batch_size = self._vectorized_batch_size()
        if self.action_chunk_execution_enabled and vectorized_batch_size > 1:
            raise ValueError(
                "rl_games.env_eval.action_chunk_execution is not supported with vectorized eval. "
                "Set rl_games.env_eval.vectorized.enabled=false."
            )
        if vectorized_batch_size > 1 and video_dir is None:
            return self._run_latency_vectorized(
                model=model,
                latency=latency,
                prompt=prompt,
                max_steps=max_steps,
                num_episodes=num_episodes,
                episode_indices=episode_indices,
                progress_desc=progress_desc,
                progress_position=progress_position,
                batch_size=vectorized_batch_size,
                on_episode_complete=on_episode_complete,
                seed_overrides=seed_overrides,
            )

        env = self._make_env()
        runtime_button_order = _get_available_button_names(env) if self.task == "deadly_corridor" else None
        queue = ActionLatencyQueue(latency=latency, default_action=self._queue_default_action())

        rewards: List[float] = []
        lengths: List[int] = []
        action_hist = Counter()
        episode_seeds: List[int | None] = []

        episode_iter = tqdm(
            episode_indices,
            desc=progress_desc or f"{self.task} latency={latency}",
            total=len(episode_indices),
            leave=False,
            position=progress_position,
            dynamic_ncols=True,
        )
        slot_id = 0
        for episode in episode_iter:
            episode_seed = self._episode_seed_for_run(
                latency=latency,
                episode=episode,
                seed_overrides=seed_overrides,
            )
            episode_seeds.append(episode_seed)
            self._reset_model_memory(model, slot_id)
            obs, _ = self._reset_env(env, episode_seed)
            model_obs = self._model_observation(env, obs)
            model_input = self._initial_model_history(model_obs) if self.pack_image_sequence else model_obs
            image_transform = self._make_image_transform()
            initial_frame = _render_frame(env, fallback_obs=model_obs) if image_transform.enabled else None
            image_transform.reset(initial_frame, step=0)
            queue.reset()
            done = False
            total_reward = 0.0
            steps = 0
            frames: List[np.ndarray] = []
            if video_dir is not None:
                frame = _render_frame(env, fallback_obs=model_obs)
                if frame is not None:
                    frames.append(frame)

            while not done and steps < max_steps:
                example = self._make_model_example(
                    model_input if self.pack_image_sequence else image_transform.current(model_obs),
                    prompt,
                    slot_id,
                )
                output = model.predict_action(examples=[example])
                actions = output["normalized_actions"]
                self._validate_action_chunk(actions, batch_size=1)
                chunk_size = self.action_chunk_execution_size if self.action_chunk_execution_enabled else 1
                for raw_action in actions[0, :chunk_size, :]:
                    decoded = self._decode_action(raw_action, runtime_button_order=runtime_button_order)
                    effective_action = queue.schedule_and_get(decoded)
                    action_hist[str(effective_action)] += 1

                    obs, step_reward, terminated, truncated = self._step_env_once(env, effective_action)
                    model_obs = self._model_observation(env, obs)
                    model_input = (
                        self._advance_model_history(model_input, model_obs)
                        if self.pack_image_sequence
                        else model_obs
                    )
                    next_step = steps + 1
                    frame = _render_frame(env, fallback_obs=model_obs) if image_transform.enabled else None
                    image_transform.append(frame, step=next_step)
                    if video_dir is not None:
                        frame = _render_frame(env, fallback_obs=model_obs)
                        if frame is not None:
                            frames.append(frame)

                    total_reward += step_reward
                    steps = next_step
                    done = terminated or truncated or steps >= max_steps
                    if done:
                        break

            if video_dir is not None:
                seed_part = "none" if episode_seed is None else str(int(episode_seed))
                output_path = (
                    Path(video_dir)
                    / f"episode_{int(episode):03d}_seed_{_slug(seed_part)}.mp4"
                )
                _write_video(output_path, frames, fps=video_fps)

            rewards.append(total_reward)
            lengths.append(steps)
            episode_iter.set_postfix({
                "reward": f"{total_reward:.2f}",
                "steps": steps,
            })
            if on_episode_complete is not None:
                on_episode_complete(
                    self._build_latency_metrics(
                        latency=latency,
                        num_episodes=len(rewards),
                        rewards=rewards,
                        lengths=lengths,
                        action_hist=action_hist,
                        episode_seeds=episode_seeds,
                        episode_indices=[int(item) for item in episode_indices[:len(rewards)]],
                    )
                )

        env.close()
        return self._build_latency_metrics(
            latency=latency,
            num_episodes=num_episodes,
            rewards=rewards,
            lengths=lengths,
            action_hist=action_hist,
            episode_seeds=episode_seeds,
            episode_indices=[int(episode) for episode in episode_indices],
        )


class RlGamesEvalRunner:
    def __init__(
        self,
        cfg,
        output_dir: str,
        video_output_dir: str | None = None,
        video_fps: int = DEFAULT_VIDEO_FPS,
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.video_output_dir = video_output_dir
        self.video_fps = int(video_fps)
        self.prompt_mode = str(
            getattr(cfg.rl_games.env_eval, "prompt_mode", "raw") or "raw"
        ).strip().lower()

    def _stage_cfg(self, stage: str):
        env_eval = self.cfg.rl_games.env_eval
        if stage in {"mid_train", "post_train"}:
            return getattr(env_eval, stage, None)
        return None

    def _cross_task_cfg(self, task: str):
        cross_cfg = getattr(self.cfg.rl_games, "cross_task", None)
        eval_tasks = getattr(cross_cfg, "eval_tasks", None) if cross_cfg is not None else None
        if eval_tasks is None:
            return None
        if isinstance(eval_tasks, dict):
            return eval_tasks.get(task)
        return getattr(eval_tasks, task, None)

    def _task_stage_cfg(self, task: str | None, stage: str):
        if task is None:
            return self._stage_cfg(stage)
        task_cfg = self._cross_task_cfg(task)
        if task_cfg is None:
            return self._stage_cfg(stage)
        return getattr(task_cfg, stage, None)

    def is_enabled(self, stage: str) -> bool:
        env_eval = self.cfg.rl_games.env_eval
        if not bool(getattr(env_eval, "enabled", False)):
            return False
        stage_cfg = self._stage_cfg(stage)
        if stage_cfg is None:
            return True
        return bool(getattr(stage_cfg, "enabled", True))

    def interval_steps(self) -> int:
        stage_cfg = self._stage_cfg("mid_train")
        if stage_cfg is None or getattr(stage_cfg, "interval_steps", None) is None:
            raise ValueError("Missing required RL-games config field: rl_games.env_eval.mid_train.interval_steps")
        return int(stage_cfg.interval_steps)

    def _get_latency_values(self, stage: str, task: str | None = None) -> List[int]:
        env_eval = self.cfg.rl_games.env_eval
        stage_cfg = self._task_stage_cfg(task, stage)
        values = None
        if stage_cfg is not None:
            values = _cfg_get(container=stage_cfg, key="latencies", default_value=None)
        if not values:
            latency_cfg = getattr(env_eval, "latency", {})
            values = _cfg_get(container=latency_cfg, key="values", default_value=[]) or []
            if not values:
                prompt_map_path = _cfg_get(container=latency_cfg, key="prompt_map_path", default_value=None)
                prompt_map = _load_latency_prompt_map(prompt_map_path)
                if prompt_map:
                    return sorted(prompt_map)
                values = [0]
        return [int(v) for v in values]

    def _num_episodes(self, stage: str, task: str | None = None) -> int:
        stage_cfg = self._task_stage_cfg(task, stage)
        if stage_cfg is not None and getattr(stage_cfg, "num_episodes", None) is not None:
            return int(stage_cfg.num_episodes)
        global_stage_cfg = self._stage_cfg(stage)
        if global_stage_cfg is None or getattr(global_stage_cfg, "num_episodes", None) is None:
            raise ValueError(f"Missing required RL-games config field: rl_games.env_eval.{stage}.num_episodes")
        return int(global_stage_cfg.num_episodes)

    def _max_steps_per_episode(self, stage: str, task: str | None = None) -> int:
        stage_cfg = self._task_stage_cfg(task, stage)
        if stage_cfg is not None and getattr(stage_cfg, "max_steps_per_episode", None) is not None:
            return int(stage_cfg.max_steps_per_episode)
        global_stage_cfg = self._stage_cfg(stage)
        if global_stage_cfg is None or getattr(global_stage_cfg, "max_steps_per_episode", None) is None:
            raise ValueError(
                f"Missing required RL-games config field: rl_games.env_eval.{stage}.max_steps_per_episode"
            )
        return int(global_stage_cfg.max_steps_per_episode)

    def _prompt_map_path(self, task: str | None = None):
        if task is not None:
            task_cfg = self._cross_task_cfg(task)
            if task_cfg is not None and getattr(task_cfg, "prompt_map_path", None):
                return getattr(task_cfg, "prompt_map_path")
        return getattr(getattr(self.cfg.rl_games.env_eval, "latency", {}), "prompt_map_path", None)

    def _resolve_prompt(self, latency: int, mapping: Dict[int, Dict[str, Any]], task: str | None = None) -> str:
        if latency in mapping:
            prompt = str(mapping[latency]["prompt"])
            return _apply_prompt_mode(prompt, self.prompt_mode)
        task_cfg = self._cross_task_cfg(task) if task is not None else None
        prompt = str(getattr(task_cfg, "task_description", "") or "") if task_cfg is not None else ""
        if not prompt:
            prompt = str(getattr(self.cfg.rl_games.env_eval, "task_description", "") or "")
        if prompt:
            return _apply_prompt_mode(prompt, self.prompt_mode)
        if task_cfg is not None and self._prompt_map_path(task):
            raise ValueError(f"No eval prompt found for task={task!r}, latency={latency} in {self._prompt_map_path(task)}")
        task = task or str(getattr(self.cfg.rl_games, "task", "flappy"))
        if task == "flappy":
            return _apply_prompt_mode(
                "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. Choose the action: NOOP, FLAP.",
                self.prompt_mode,
            )
        if task == "demon_attack":
            return _apply_prompt_mode(
                "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.",
                self.prompt_mode,
            )
        if task == "air_raid":
            return _apply_prompt_mode(
                "Protect both buildings from flying saucers. Choose exactly one action from: noop, fire, right, left, rightfire, leftfire.",
                self.prompt_mode,
            )
        if task == "deadly_corridor":
            return _apply_prompt_mode(
                (
                    "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, "
                    "fight enemies, and reach the green armor vest at the far end. The available actions are: "
                    "MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK."
                ),
                self.prompt_mode,
            )
        if task == "asterix":
            return _apply_prompt_mode(
                "Collect useful objects and avoid lyres. Choose exactly one action from: noop, up, right, left, down, upright, upleft, downright, downleft.",
                self.prompt_mode,
            )
        if task == "atlantis":
            return _apply_prompt_mode(
                "Defend Atlantis by firing at descending enemies. Choose exactly one action from: noop, fire, rightfire, leftfire.",
                self.prompt_mode,
            )
        return _apply_prompt_mode("Act optimally in the current environment.", self.prompt_mode)

    def _get_tasks(self) -> List[str]:
        task = str(getattr(self.cfg.rl_games, "task", "flappy"))
        if task != "cross_task":
            return [task]
        cross_cfg = getattr(self.cfg.rl_games, "cross_task", None)
        if cross_cfg is not None and getattr(cross_cfg, "eval_tasks", None):
            eval_tasks = getattr(cross_cfg, "eval_tasks")
            if isinstance(eval_tasks, dict):
                return [str(x) for x in eval_tasks.keys()]
            return [str(x) for x in eval_tasks]
        return ["flappy", "demon_attack"]

    @staticmethod
    def _merge_latency_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_list:
            return {}

        episode_rows = []
        action_hist = Counter()
        for metrics in metrics_list:
            rewards = metrics.get("episode_rewards", [])
            lengths = metrics.get("episode_lengths", [])
            seeds = metrics.get("episode_seeds", [])
            indices = metrics.get("episode_indices", list(range(len(rewards))))
            episode_rows.extend(
                (
                    int(index),
                    float(reward),
                    int(length),
                    seed,
                )
                for index, reward, length, seed in zip(indices, rewards, lengths, seeds)
            )
            action_hist.update(metrics.get("decoded_action_hist", {}))

        episode_rows.sort(key=lambda row: row[0])
        episode_indices = [row[0] for row in episode_rows]
        rewards = [row[1] for row in episode_rows]
        lengths = [row[2] for row in episode_rows]
        episode_seeds = [row[3] for row in episode_rows]
        first = metrics_list[0]
        merged = {
            "latency": int(first.get("latency", 0)),
            "num_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "std_length": float(np.std(lengths)) if lengths else 0.0,
            "episode_rewards": rewards,
            "episode_lengths": lengths,
            "decoded_action_hist": dict(action_hist),
            "fixed_episode_seeds": bool(first.get("fixed_episode_seeds", True)),
            "eval_seed": int(first.get("eval_seed", 0)),
            "episode_seeds": episode_seeds,
            "episode_indices": episode_indices,
        }
        vectorized_batch_sizes = [
            int(metrics.get("vectorized_batch_size", 1))
            for metrics in metrics_list
            if bool(metrics.get("vectorized_eval", False))
        ]
        if vectorized_batch_sizes:
            merged["vectorized_eval"] = True
            merged["vectorized_batch_size"] = max(vectorized_batch_sizes)
        return merged

    @staticmethod
    def _aggregate_result(per_latency: Dict[str, Dict], *, stage: str, step: int, cfg, task_count: int) -> Dict:
        aggregate = {
            "stage": stage,
            "step": int(step),
            "task": str(getattr(cfg.rl_games, "task", "flappy")),
            "model_alias": str(getattr(cfg.rl_games, "model_alias", "openvla")),
            "fixed_episode_seeds": _as_bool(getattr(cfg.rl_games.env_eval, "fixed_episode_seeds", True), default=True),
            "eval_seed": int(getattr(cfg.rl_games.env_eval, "seed", getattr(cfg, "seed", 42)) or 42),
            "total_episodes": 0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "task_count": task_count,
        }
        all_rewards = []
        all_lengths = []
        for metrics in per_latency.values():
            aggregate["total_episodes"] += int(metrics.get("num_episodes", 0))
            all_rewards.extend(float(reward) for reward in metrics.get("episode_rewards", []))
            all_lengths.extend(float(length) for length in metrics.get("episode_lengths", []))

        if all_rewards:
            aggregate["mean_reward"] = float(np.mean(all_rewards))
            aggregate["std_reward"] = float(np.std(all_rewards))
        if all_lengths:
            aggregate["mean_length"] = float(np.mean(all_lengths))
            aggregate["std_length"] = float(np.std(all_lengths))
        if per_latency:
            aggregate["macro_mean_reward"] = float(
                np.mean([float(metrics.get("mean_reward", 0.0)) for metrics in per_latency.values()])
            )
            aggregate["macro_mean_length"] = float(
                np.mean([float(metrics.get("mean_length", 0.0)) for metrics in per_latency.values()])
            )
        vectorized_batch_sizes = [
            int(metrics.get("vectorized_batch_size", 1))
            for metrics in per_latency.values()
            if bool(metrics.get("vectorized_eval", False))
        ]
        if vectorized_batch_sizes:
            aggregate["vectorized_eval"] = True
            aggregate["vectorized_batch_size"] = max(vectorized_batch_sizes)
        return aggregate

    def merge_results(self, results: List[EvalResult], *, step: int, stage: str) -> EvalResult:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            if result is None:
                continue
            for key, metrics in result.per_latency.items():
                grouped.setdefault(key, []).append(metrics)

        per_latency = {
            key: self._merge_latency_metrics(metrics_list)
            for key, metrics_list in grouped.items()
        }
        aggregate = self._aggregate_result(
            per_latency=per_latency,
            stage=stage,
            step=step,
            cfg=self.cfg,
            task_count=len(self._get_tasks()),
        )
        aggregate["distributed_eval"] = True
        result = EvalResult(per_latency=per_latency, aggregate=aggregate)
        return result

    def save(self, result: EvalResult, step: int, stage: str) -> str:
        result.path = self._save(result=result, step=step, stage=stage)
        return result.path

    def run(
        self,
        model,
        step: int,
        stage: str = "mid_train",
        shard_rank: int = 0,
        shard_count: int = 1,
        save: bool = True,
        episode_seed_overrides: Dict[str, Dict[int, int | None]] | None = None,
    ) -> EvalResult:
        tasks = [
            task_name
            for task_name in self._get_tasks()
            if getattr(self._task_stage_cfg(task_name, stage), "enabled", True)
        ]
        per_latency: Dict[str, Dict] = {}

        rollout_plan = [
            (task_name, latency)
            for task_name in tasks
            for latency in self._get_latency_values(stage=stage, task=task_name)
        ]
        total_rollouts = len(rollout_plan)
        rollout_iter = tqdm(
            rollout_plan,
            desc=f"rl-games {stage} eval step={step}",
            total=total_rollouts,
            leave=True,
            position=0,
            dynamic_ncols=True,
        )

        prompt_maps = {
            task_name: _load_latency_prompt_map(self._prompt_map_path(task_name))
            for task_name in tasks
        }
        for task_name, latency in rollout_iter:
            rollout_iter.set_postfix({"task": task_name, "latency": latency})
            task_eval_cfg = self._cross_task_cfg(task_name)
            stage_eval_cfg = self._task_stage_cfg(task_name, stage)
            task_eval = _TaskEvaluator(
                task=task_name,
                cfg=self.cfg,
                task_eval_cfg=task_eval_cfg,
                stage_eval_cfg=stage_eval_cfg,
            )
            prompt = self._resolve_prompt(latency=latency, mapping=prompt_maps.get(task_name, {}), task=task_name)
            num_episodes = self._num_episodes(stage=stage, task=task_name)
            key = f"{task_name}/latency_{latency}"
            seed_overrides = (
                episode_seed_overrides.get(key)
                if episode_seed_overrides is not None
                else None
            )
            all_episode_indices = (
                sorted(int(episode) for episode in seed_overrides)
                if seed_overrides is not None
                else list(range(num_episodes))
            )
            episode_indices = [
                episode
                for episode in all_episode_indices
                if int(episode) % max(1, int(shard_count)) == int(shard_rank)
            ]

            def save_episode_progress(metrics: Dict[str, Any]) -> None:
                per_latency[key] = metrics
                aggregate = self._aggregate_result(
                    per_latency=per_latency,
                    stage=stage,
                    step=step,
                    cfg=self.cfg,
                    task_count=len(tasks),
                )
                aggregate["distributed_eval"] = int(shard_count) > 1
                self._save(result=EvalResult(per_latency=per_latency, aggregate=aggregate), step=step, stage=stage)

            metrics = task_eval.run_latency(
                model=model,
                latency=latency,
                prompt=prompt,
                max_steps=self._max_steps_per_episode(stage=stage, task=task_name),
                num_episodes=len(all_episode_indices) if seed_overrides is not None else num_episodes,
                episode_indices=episode_indices,
                progress_desc=f"{stage} {task_name} latency={latency}",
                progress_position=1,
                on_episode_complete=save_episode_progress if save else None,
                video_dir=(
                    Path(self.video_output_dir)
                    / stage
                    / f"step_{int(step)}"
                    / _slug(task_name)
                    / f"latency_{int(latency)}"
                    if self.video_output_dir is not None
                    else None
                ),
                video_fps=self.video_fps,
                seed_overrides=seed_overrides,
            )
            per_latency[key] = metrics

        aggregate = self._aggregate_result(
            per_latency=per_latency,
            stage=stage,
            step=step,
            cfg=self.cfg,
            task_count=len(tasks),
        )
        aggregate["distributed_eval"] = int(shard_count) > 1

        result = EvalResult(per_latency=per_latency, aggregate=aggregate)
        if save:
            self.save(result=result, step=step, stage=stage)
        return result

    def _save(self, result: EvalResult, step: int, stage: str) -> str:
        eval_dir = os.path.join(self.output_dir, "eval", stage)
        os.makedirs(eval_dir, exist_ok=True)
        payload = {
            "per_latency": result.per_latency,
            "aggregate": result.aggregate,
        }
        output_path = os.path.join(eval_dir, f"step_{step}.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path
