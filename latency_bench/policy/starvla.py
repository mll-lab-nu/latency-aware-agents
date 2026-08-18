from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.timing import current_profiler
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.data.ghost_trail import GhostTrailConfig, build_flappy_ghost_trail_window
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.starvla_prompts import load_latency_prompt_map, resolve_starvla_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
STARVLA_ROOT = REPO_ROOT / "third_party" / "starVLA"
STATEFUL_STARVLA_MODEL_IDS: tuple[str, ...] = (
    "pi0",
    "pi-0",
    "pi05",
    "pi-0.5",
    "gr00t",
    "qwenpi",
    "qwenpi_v3",
    "qwengr00t",
)
STATELESS_STARVLA_MODEL_IDS: tuple[str, ...] = (
    "openvla",
    "qwenoft",
)

DEMON_ATTACK_ACTION_LABELS: tuple[str, ...] = (
    "NOOP",
    "FIRE",
    "RIGHT",
    "LEFT",
    "RIGHTFIRE",
    "LEFTFIRE",
)
DEADLY_CORRIDOR_TURN_LABELS: tuple[str, ...] = (
    "TURN_NOOP",
    "TURN_LEFT",
    "TURN_RIGHT",
)
DEADLY_CORRIDOR_MOVE_LABELS: tuple[str, ...] = (
    "MOVE_NOOP",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
)
DEADLY_CORRIDOR_STRAFE_LABELS: tuple[str, ...] = (
    "STRAFE_NOOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
)
DEADLY_CORRIDOR_ATTACK_LABELS: tuple[str, ...] = (
    "ATTACK_NOOP",
    "ATTACK",
)


class StarVlaPolicyRunner(PolicyRunner):
    def __init__(
        self,
        *,
        wrapper: Any,
        checkpoint_path: str,
        device: str,
        unnorm_key: str | None,
        env_name: str,
        action_resolver: ActionResolver,
        action_refs: Sequence[Any],
        latency_prompt_map: dict[str, Any] | None = None,
        base_prompt: str | None = None,
        latency_prompt_key: int | str | None = None,
        prompt_mode: str | None = None,
        obs_resize: tuple[int, int] | None = None,
        image_transform_config: Mapping[str, Any] | None = None,
        observation_stride_raw_frames: int,
        model_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._obs_resize = tuple(obs_resize) if obs_resize else None
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._unnorm_key = unnorm_key
        self._env_name = env_name
        self._action_by_raw_id = {
            raw_action_id: action_resolver.resolve(action_ref)
            for raw_action_id, action_ref in enumerate(action_refs)
        }
        self._latency_prompt_map = latency_prompt_map
        self._base_prompt = base_prompt
        self._latency_prompt_key = latency_prompt_key
        self._prompt_mode = str(prompt_mode or "default").strip().lower()
        self._image_transform_config = dict(image_transform_config or {"image_transform": "raw_rgb"})
        self._image_transform = str(
            self._image_transform_config.get("image_transform", "raw_rgb") or "raw_rgb"
        ).strip().lower()
        model_cfg = (
            _normalized_model_cfg_from_wrapper(wrapper)
            if model_cfg is None
            else _normalized_model_cfg(model_cfg)
        )
        self._include_state = _include_state_from_model_cfg(model_cfg)
        self._state_dim = _state_dim_from_model_cfg(model_cfg) if self._include_state else None
        vla_data = (model_cfg.get("datasets", {}) or {}).get("vla_data", {}) or {}
        self._pack_image_sequence = (
            bool(vla_data["pack_image_sequence"])
            if "pack_image_sequence" in vla_data
            else False
        )
        self._image_sequence_length = (
            int(vla_data["image_sequence_length"])
            if self._pack_image_sequence
            else 1
        )
        self._observation_stride_raw_frames = int(observation_stride_raw_frames)
        self._image_sequence_raw_span = (
            1
            + (self._image_sequence_length - 1)
            * self._observation_stride_raw_frames
        )
        self._num_obs_frames = int(vla_data.get("num_obs_frames", 1) or 1)
        self._image_mode = str(vla_data.get("image_mode", "single"))
        self._stitch_grid = tuple(vla_data.get("stitch_grid", [2, 2]))
        framework_cfg = model_cfg["framework"]
        kv_cfg = framework_cfg["kv_memory"] if "kv_memory" in framework_cfg else {}
        self._kv_memory_enabled = bool(kv_cfg["enabled"]) if "enabled" in kv_cfg else False

    def reset_state(self, slot_id: int | None = None) -> None:
        # Clears the model's per-slot KV memory at episode boundaries (req3).
        # No-op unless the framework maintains KV memory.
        reset = getattr(self._wrapper, "reset_memory", None)
        if callable(reset):
            reset(slot_id)

    def predict(self, observation: Observation) -> PolicyOutput:
        return self.predict_batch([observation])[0]

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        profiler = current_profiler()
        with profiler.time("policy_build_example_ms"):
            examples = [self._build_example(observation) for observation in observations]
        with profiler.time("policy_wrapper_predict_action_ms"):
            prediction = self._wrapper.predict_action(
                examples=examples, unnorm_key=self._unnorm_key, profiler=profiler
            )
        with profiler.time("policy_decode_ms"):
            outputs = [
                self._decode_prediction(
                    prediction=prediction,
                    index=index,
                    observation=observation,
                    example=example,
                )
                for index, (observation, example) in enumerate(zip(observations, examples))
            ]
        return outputs

    def _decode_prediction(
        self,
        *,
        prediction: dict[str, Any],
        index: int,
        observation: Observation,
        example: dict[str, Any],
    ) -> PolicyOutput:
        actions = np.asarray(prediction["actions"])
        raw_action_scores = (
            np.asarray(prediction["raw_action_scores"])
            if "raw_action_scores" in prediction
            else None
        )
        return self._policy_output(
            observation=observation,
            example=example,
            action_payload=actions[index, 0],
            action_output_type=prediction["action_output_type"],
            raw_action_scores=None if raw_action_scores is None else raw_action_scores[index, 0],
        )

    def _build_example(self, observation: Observation) -> dict[str, Any]:
        frame_source = observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        frames = observation_data_to_hwc_uint8_frames(frame_source)  # oldest .. newest
        transformed = self._transformed_frame(frames=frames, observation=observation)

        if self._pack_image_sequence:
            if transformed is not None:
                raise ValueError(
                    "WanOFT packed image sequences require image_transform=raw_rgb"
                )
            if len(frames) < self._image_sequence_raw_span:
                raise ValueError(
                    "WanOFT packed image sequence requires "
                    f"{self._image_sequence_raw_span} raw frames for "
                    f"{self._image_sequence_length} decision observations at stride "
                    f"{self._observation_stride_raw_frames}, got {len(frames)}"
                )
            frames = frames[
                -self._image_sequence_raw_span
                :: self._observation_stride_raw_frames
            ]
        elif transformed is not None:
            frames = [transformed]
        elif self._image_mode == "single" or self._kv_memory_enabled:
            frames = frames[-1:]
        else:
            # Select the temporal observation window to match training (_pack_sample).
            raw_span = 1 + (self._num_obs_frames - 1) * self._observation_stride_raw_frames
            frames = frames[-raw_span :: self._observation_stride_raw_frames]

        prompt = resolve_starvla_prompt(
            env_name=self._env_name,
            observation_metadata=observation.metadata,
            latency_prompt_map=self._latency_prompt_map,
            base_prompt=self._base_prompt,
            latency_prompt_key=self._latency_prompt_key,
            prompt_mode=self._prompt_mode,
        )

        if self._image_mode == "stitch":
            if transformed is not None:
                raise ValueError("image_transform is not compatible with image_mode=stitch")
            # Tile the window into one image; matches _pack_sample's stitch branch
            # (raw frames passed to stitch_frames, which resizes each cell to 224).
            images = [_get_stitch_frames()(frames, grid=self._stitch_grid, size=(224, 224))]
        else:
            if self._obs_resize is not None:
                height, width = self._obs_resize
                # match training preprocessing exactly: gr00t LeRobotSingleDataset._pack_sample
                # does `Image.fromarray(image).resize((224, 224))` (PIL default resample = BICUBIC).
                frames = [
                    np.asarray(Image.fromarray(frame).resize((width, height)), dtype=np.uint8)
                    for frame in frames
                ]
            images = [Image.fromarray(frame) for frame in frames]

        example = {
            "image": images,
            "lang": prompt,
        }
        if self._kv_memory_enabled:
            example["slot_id"] = observation.metadata["slot_id"]
        elif "slot_id" in observation.metadata:
            example["slot_id"] = observation.metadata["slot_id"]
        if self._include_state:
            example["state"] = np.zeros((1, self._state_dim), dtype=np.float32)
        return example

    def _transformed_frame(
        self,
        *,
        frames: Sequence[npt.NDArray[np.uint8]],
        observation: Observation,
    ) -> npt.NDArray[np.uint8] | None:
        if self._image_transform in {"", "none", "raw", "raw_rgb"}:
            return None
        if self._image_transform not in {"flappy_ghost_trail", "demon_attack_ghost_trail"}:
            raise ValueError(f"Unsupported StarVLA image_transform={self._image_transform!r}")
        if self._image_transform == "flappy_ghost_trail" and self._env_name != "flappy":
            raise ValueError("image_transform=flappy_ghost_trail is only supported for env_name=flappy")
        if self._image_transform == "demon_attack_ghost_trail" and self._env_name != "demon_attack":
            raise ValueError("image_transform=demon_attack_ghost_trail is only supported for env_name=demon_attack")

        config = GhostTrailConfig(
            image_transform=self._image_transform,
            history_frames=int(self._image_transform_config.get("history_frames", 5)),
            gamma=float(self._image_transform_config.get("gamma", 1.3)),
            min_alpha=int(self._image_transform_config.get("min_alpha", 35)),
            ground_fraction=float(self._image_transform_config.get("ground_fraction", 0.22)),
            scroll_px_per_step=float(self._image_transform_config.get("scroll_px_per_step", 4.0)),
        )
        if self._image_transform == "demon_attack_ghost_trail":
            # env_step counts raw ALE frames (buffer updated 4× per decision step).
            # frames[-0:] == frames, so env_step=0 falls back to the full reset-fill buffer.
            valid_count = min(len(frames), int(observation.env_step))
        else:
            max_frames = max(1, int(config.history_frames) + 1)
            valid_count = min(len(frames), max(1, int(observation.env_step) + 1), max_frames)
        window = [np.asarray(frame, dtype=np.uint8) for frame in frames[-valid_count:]]

        if self._image_transform == "demon_attack_ghost_trail":
            from latency_bench.data.ghost_trail_demon import build_demon_attack_ghost_trail_window
            steps_arg = list(range(len(window)))
            return build_demon_attack_ghost_trail_window(window, steps_arg, config=config)

        current_step = int(observation.env_step)
        start_step = current_step - valid_count + 1
        steps = list(range(start_step, current_step + 1))
        return build_flappy_ghost_trail_window(window, steps, config=config)

    def _policy_output(
        self,
        *,
        observation: Observation,
        example: dict[str, Any],
        action_payload: npt.NDArray[Any],
        action_output_type: str,
        raw_action_scores: npt.NDArray[Any] | None,
    ) -> PolicyOutput:
        payload = np.asarray(action_payload)
        action, action_metadata = action_from_starvla_payload(
            payload=payload,
            env_name=self._env_name,
            action_by_raw_id=self._action_by_raw_id,
            action_output_type=action_output_type,
        )
        metadata = {
            "policy_type": "starvla",
            "prompt_source": "latency_prompt_map" if self._latency_prompt_map is not None else "base",
            "checkpoint_path": self._checkpoint_path,
            "unnorm_key": self._unnorm_key,
            "device": self._device,
            "prompt_sha256": sha256_text(example["lang"]),
            "input_frame_count": len(example["image"]),
            "image_transform": self._image_transform,
            "action_output_type": action_output_type,
            "action_payload": to_jsonable_action_payload(payload),
            "kv_memory_enabled": self._kv_memory_enabled,
            **action_metadata,
        }
        if self._pack_image_sequence:
            metadata["image_sequence_length"] = self._image_sequence_length
            metadata["input_frame_raw_stride"] = self._observation_stride_raw_frames
            metadata["input_frame_raw_span"] = self._image_sequence_raw_span
        if "slot_id" in example:
            metadata["slot_id"] = example["slot_id"]
        if raw_action_scores is not None:
            metadata["raw_action_scores"] = [
                float(item) for item in np.asarray(raw_action_scores, dtype=np.float32).tolist()
            ]
        if "latency_raw_frames" in observation.metadata:
            metadata["latency_raw_frames"] = observation.metadata["latency_raw_frames"]
        if "latency_ms" in observation.metadata:
            metadata["latency_ms"] = observation.metadata["latency_ms"]
        if self._latency_prompt_key is not None:
            metadata["latency_prompt_key"] = self._latency_prompt_key
        return PolicyOutput(
            action=action,
            raw_output=metadata["action_payload"],
            metadata=metadata,
        )


def observation_data_to_hwc_uint8_frames(data: Any) -> list[npt.NDArray[np.uint8]]:
    frame = _extract_observation_array(data)
    if frame.ndim == 4 and frame.shape[-1] == 3:
        return [_as_uint8_image(item) for item in frame]
    if frame.ndim == 4 and frame.shape[1] == 3:
        return [_as_uint8_image(np.transpose(item, (1, 2, 0))) for item in frame]
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return [_as_uint8_image(frame)]
    if frame.ndim == 3 and frame.shape[0] == 3:
        return [_as_uint8_image(np.transpose(frame, (1, 2, 0)))]
    if (
        frame.ndim == 3
        and frame.shape[0] % 3 == 0
        and frame.shape[0] < frame.shape[1]
        and frame.shape[0] < frame.shape[2]
    ):
        return [
            _as_uint8_image(np.transpose(frame[start : start + 3, :, :], (1, 2, 0)))
            for start in range(0, frame.shape[0], 3)
        ]
    if frame.ndim == 3 and frame.shape[-1] % 3 == 0:
        return [
            _as_uint8_image(frame[:, :, start : start + 3])
            for start in range(0, frame.shape[-1], 3)
        ]
    if frame.ndim == 3 and frame.shape[0] % 3 == 0:
        return [
            _as_uint8_image(np.transpose(frame[start : start + 3, :, :], (1, 2, 0)))
            for start in range(0, frame.shape[0], 3)
        ]
    return [_as_uint8_image(frame)]


def decode_starvla_action(
    *,
    vector: npt.NDArray[Any],
    env_name: str,
    action_by_raw_id: Mapping[int, Action],
) -> tuple[Action, dict[str, Any]]:
    deadly_layout = None
    if str(env_name) == "deadly_corridor":
        action_dim = int(np.asarray(vector).shape[-1])
        deadly_layouts = {
            7: "deadly_corridor_semantic_7",
            11: "deadly_corridor_factorized_11",
            54: "deadly_corridor_joint_54",
        }
        if action_dim not in deadly_layouts:
            raise ValueError(
                "Deadly Corridor StarVLA action vector expected 7, 11, or 54 "
                f"values, got {action_dim}"
            )
        deadly_layout = deadly_layouts[action_dim]

    decode_rl_games_actions, _ = _load_rl_games_action_decode()
    prediction = decode_rl_games_actions(
        normalized_actions=np.asarray(vector),
        env_name=str(env_name),
    )
    action, metadata = action_from_starvla_payload(
        payload=np.asarray(prediction["actions"]),
        env_name=env_name,
        action_by_raw_id=action_by_raw_id,
        action_output_type=prediction["action_output_type"],
    )
    if deadly_layout is not None:
        metadata["action_layout"] = deadly_layout
        if deadly_layout == "deadly_corridor_joint_54":
            turn, move, strafe, attack = action.value
            metadata["raw_action_id"] = turn * 18 + move * 6 + strafe * 2 + attack
        elif deadly_layout == "deadly_corridor_semantic_7":
            semantic_actions = (
                [0, 1, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 2, 0],
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 1],
            )
            metadata["raw_action_id"] = semantic_actions.index(action.value)
    return action, metadata


# Fixed semantic button order the StarVLA multibinary head is trained against.
# Mirrors starVLA.training.rl_games.eval_core._semantic_to_runtime_multibinary;
# the env adapter re-orders this to the live ViZDoom button layout.
DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
)


def action_from_starvla_payload(
    *,
    payload: npt.NDArray[Any],
    env_name: str,
    action_by_raw_id: Mapping[int, Action],
    action_output_type: str = "",
) -> tuple[Action, dict[str, Any]]:
    if str(env_name) == "demon_attack":
        return demon_attack_action_from_id(int(np.asarray(payload).reshape(-1)[0]))
    if str(env_name) == "deadly_corridor":
        # Multibinary heads emit an already-thresholded 7-dim button vector in
        # fixed semantic order; the env adapter re-orders it to the live ViZDoom
        # button layout. Keep it as-is rather than reinterpreting it as a
        # [turn, move, strafe, attack] categorical tuple.
        if str(action_output_type) == "rl_games_deadly_corridor_multibinary":
            buttons = [int(item) for item in np.asarray(payload).reshape(-1).tolist()]
            active = [
                DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER[idx]
                for idx, pressed in enumerate(buttons)
                if idx < len(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER) and pressed
            ]
            action_name = "+".join(active) if active else "NOOP"
            return Action(
                value=buttons,
                name=action_name,
                is_noop=not any(buttons),
                is_oneshot=False,
            ), {
                "decoded_multibinary_buttons": buttons,
                "action_label": action_name,
                "action_layout": "deadly_corridor_multibinary_7",
            }
        return deadly_corridor_action_from_tuple(
            action_value=[int(item) for item in np.asarray(payload).reshape(-1).tolist()],
            metadata={"action_layout": "deadly_corridor_tuple"},
        )
    raw_action_id = int(np.asarray(payload).reshape(-1)[0])
    return action_by_raw_id[raw_action_id], {"raw_action_id": raw_action_id}


def to_jsonable_action_payload(payload: npt.NDArray[Any]) -> Any:
    value = np.asarray(payload).tolist()
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def demon_attack_action_from_id(action_id: int) -> tuple[Action, dict[str, Any]]:
    action = Action(
        value=action_id,
        name=DEMON_ATTACK_ACTION_LABELS[action_id],
        is_noop=action_id == 0,
        is_oneshot=False,
    )
    return action, {"raw_action_id": action_id, "action_label": action.name}


def deadly_corridor_action_from_tuple(
    *,
    action_value: list[int],
    metadata: dict[str, Any],
) -> tuple[Action, dict[str, Any]]:
    turn, move, strafe, attack = action_value
    action_value = [turn, move, strafe, attack]
    turn_label = DEADLY_CORRIDOR_TURN_LABELS[turn]
    move_label = DEADLY_CORRIDOR_MOVE_LABELS[move]
    strafe_label = DEADLY_CORRIDOR_STRAFE_LABELS[strafe]
    attack_label = DEADLY_CORRIDOR_ATTACK_LABELS[attack]
    active_labels = [
        label
        for label in (turn_label, move_label, strafe_label, attack_label)
        if not label.endswith("_NOOP")
    ]
    action_name = "+".join(active_labels) if active_labels else "NOOP"
    return Action(
        value=action_value,
        name=action_name,
        is_noop=action_value == [0, 0, 0, 0],
        is_oneshot=False,
    ), {
        "decoded_action_tuple": action_value,
        "turn_label": turn_label,
        "move_label": move_label,
        "strafe_label": strafe_label,
        "attack_label": attack_label,
        "action_label": action_name,
        **metadata,
    }


def _extract_observation_array(data: Any) -> npt.NDArray[Any]:
    if isinstance(data, Mapping):
        return np.asarray(data["observation"])
    return np.asarray(data)


def _as_uint8_image(frame: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _normalized_model_cfg(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    _ensure_starvla_path()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.create(model_cfg)
    apply_config_compat(cfg)
    _apply_model_family_include_state_compat(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def _normalized_model_cfg_from_wrapper(wrapper: Any) -> dict[str, Any]:
    return _normalized_model_cfg(wrapper._model_cfg)


def _load_starvla_model_config(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return _normalized_model_cfg(OmegaConf.load(path))


def _apply_model_family_include_state_compat(cfg: Any) -> None:
    from omegaconf import OmegaConf

    if OmegaConf.select(cfg, "datasets.vla_data.include_state") is not None:
        return

    model_ids = (
        _normalized_optional_config_string(cfg, ("model",)),
        _normalized_optional_config_string(cfg, ("rl_games", "model_alias")),
        _normalized_optional_config_string(cfg, ("framework", "name")),
    )
    if any(model_id in STATEFUL_STARVLA_MODEL_IDS for model_id in model_ids if model_id is not None):
        OmegaConf.update(cfg, "datasets.vla_data.include_state", True, force_add=True)
        return
    if any(model_id in STATELESS_STARVLA_MODEL_IDS for model_id in model_ids if model_id is not None):
        OmegaConf.update(cfg, "datasets.vla_data.include_state", False, force_add=True)


def _normalized_optional_config_string(cfg: Any, path: tuple[str, ...]) -> str | None:
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, ".".join(path))
    if value is None:
        return None
    return str(value).strip().lower()


def _state_dim_from_model_cfg(model_cfg: dict[str, Any]) -> int:
    return model_cfg["framework"]["action_model"]["state_dim"]


def _include_state_from_model_cfg(model_cfg: dict[str, Any]) -> bool:
    return model_cfg["datasets"]["vla_data"]["include_state"]


_STITCH_FRAMES = None


def _get_stitch_frames():
    """Lazily import starVLA's stitch_frames (starVLA path is added at runtime)."""
    global _STITCH_FRAMES
    if _STITCH_FRAMES is None:
        _ensure_starvla_path()
        from starVLA.training.trainer_utils.trainer_tools import stitch_frames

        _STITCH_FRAMES = stitch_frames
    return _STITCH_FRAMES


def _ensure_starvla_path() -> None:
    starvla_root = str(STARVLA_ROOT)
    if starvla_root not in sys.path:
        sys.path.insert(0, starvla_root)


def _observation_stride_raw_frames(config: Mapping[str, Any]) -> int:
    env_cfg = config["env"]
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    ).obs_stride_raw_frames


def apply_starvla_model_input_config(
    config: dict[str, Any],
    *,
    model_cfg: Mapping[str, Any],
    image_transform: str = "raw_rgb",
) -> None:
    """Match latency_bench's raw frame stack to a saved StarVLA input contract."""
    vla_data = model_cfg["datasets"]["vla_data"]
    pack_image_sequence = (
        bool(vla_data["pack_image_sequence"])
        if "pack_image_sequence" in vla_data
        else False
    )
    normalized_transform = str(image_transform).strip().lower()
    raw_image_transform = normalized_transform in {"", "none", "raw", "raw_rgb"}
    if pack_image_sequence:
        if not raw_image_transform:
            raise ValueError(
                "WanOFT packed image sequences require image_transform=raw_rgb"
            )
        input_frame_count = int(vla_data["image_sequence_length"])
    else:
        if not raw_image_transform:
            return
        if model_cfg["framework"]["kv_memory"]["enabled"]:
            return
        if vla_data["image_mode"] == "single":
            return
        input_frame_count = int(vla_data["num_obs_frames"])

    observation_stride = _observation_stride_raw_frames(config)
    required_raw_frames = 1 + (input_frame_count - 1) * observation_stride
    config["env"]["frame_stack"] = max(
        int(config["env"]["frame_stack"]),
        required_raw_frames,
    )


def prepare_starvla_checkpoint_input_config(config: dict[str, Any]) -> None:
    """Read checkpoint metadata before env construction and size its frame stack."""
    if config["policy"]["type"] != "starvla":
        return

    policy_cfg = config["policy"]
    if "model_config_path" in policy_cfg:
        model_cfg = _load_starvla_model_config(policy_cfg["model_config_path"])
    else:
        _ensure_starvla_path()
        from starVLA.model.framework.share_tools import read_mode_config

        saved_model_cfg, _norm_stats = read_mode_config(policy_cfg["checkpoint_path"])
        model_cfg = _normalized_model_cfg(saved_model_cfg)
    image_transform_cfg = (
        policy_cfg["image_transform_config"]
        if "image_transform_config" in policy_cfg
        else {}
    )
    image_transform = (
        image_transform_cfg["image_transform"]
        if "image_transform" in image_transform_cfg
        else "raw_rgb"
    )
    apply_starvla_model_input_config(
        config,
        model_cfg=model_cfg,
        image_transform=image_transform,
    )


def _load_policy_wrapper_class() -> Any:
    _ensure_starvla_path()
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    return PolicyServerWrapper


def _profiler_stage(profiler: Any, name: str) -> Any:
    from contextlib import nullcontext

    return profiler.time(name) if profiler is not None else nullcontext()


def _load_rl_games_action_decode() -> tuple[Any, Any]:
    _ensure_starvla_path()
    from deployment.model_server.rl_games_action_decode import (
        decode_rl_games_actions,
        resolve_deadly_action_decode_spec,
    )

    return decode_rl_games_actions, resolve_deadly_action_decode_spec


class LiveStarVlaWrapper:
    """In-process stand-in for ``PolicyServerWrapper`` over a *live* framework.

    During training the trainer already holds the model in memory
    (``accelerator.unwrap_model(self.model)`` — the same object eval_core calls).
    This wrapper exposes only the rl_games-mode surface ``StarVlaPolicyRunner``
    uses — ``predict_action`` (framework forward + rl_games decode),
    ``reset_memory`` passthrough, and the ``_model_cfg`` attribute — so no
    checkpoint reload is needed. The disk-backed ``PolicyNormProcessor`` is never
    built because rl_games decoding ignores un-normalization stats.
    """

    def __init__(
        self,
        *,
        framework: Any,
        model_cfg: dict[str, Any],
        env_name: str,
        rl_games_action_env_dim: int | None = None,
        action_layout: str | None = None,
        multibinary_threshold: float | None = None,
    ) -> None:
        self._framework = framework
        self._model_cfg = model_cfg
        self._rl_games_env_name = str(env_name)
        self._rl_games_action_env_dim = rl_games_action_env_dim
        (
            self._decode_rl_games_actions,
            resolve_deadly_action_decode_spec,
        ) = _load_rl_games_action_decode()
        if self._rl_games_env_name == "deadly_corridor":
            self._action_layout, self._multibinary_threshold = resolve_deadly_action_decode_spec(
                model_cfg,
                action_layout=action_layout,
                multibinary_threshold=multibinary_threshold,
            )

    def reset_memory(self, slot_id: int | None = None) -> None:
        reset = getattr(self._framework, "reset_memory", None)
        if callable(reset):
            reset(slot_id)

    def predict_action(
        self,
        examples: list[dict[str, Any]],
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # unnorm_key is unused in rl_games mode; kept for interface parity.
        del unnorm_key
        profiler = kwargs["profiler"] if "profiler" in kwargs else None
        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)
        decode_kwargs: dict[str, Any] = {}
        if self._rl_games_env_name == "gymnasium":
            decode_kwargs["action_env_dim"] = self._rl_games_action_env_dim
        with _profiler_stage(profiler, "starvla_wrapper_rl_games_decode_ms"):
            return self._decode_rl_games_actions(
                normalized_actions=normalized,
                env_name=self._rl_games_env_name,
                deadly_action_layout=(
                    self._action_layout
                    if self._rl_games_env_name == "deadly_corridor"
                    else None
                ),
                deadly_multibinary_threshold=(
                    self._multibinary_threshold
                    if self._rl_games_env_name == "deadly_corridor"
                    else None
                ),
                **decode_kwargs,
            )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_array(value: npt.NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_gymnasium_starvla_contract(
    *,
    env_cfg: Mapping[str, Any],
    policy_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> None:
    expected = gymnasium_task_contract(env_cfg)
    manifest = json.loads(
        Path(policy_cfg["task_manifest_path"]).read_text(encoding="utf-8")
    )
    if manifest["integration_name"] != "gymnasium":
        raise ValueError("StarVLA task manifest is not a Gymnasium handoff")
    if manifest["gymnasium_task"] != expected:
        raise ValueError(
            "Evaluation Gymnasium task contract does not match the StarVLA dataset manifest"
        )
    if model_cfg["rl_games"]["gymnasium"]["task_contract"] != expected:
        raise ValueError(
            "Evaluation Gymnasium task contract does not match the StarVLA model config"
        )
    if manifest["active_action_dim"] != len(expected["action_labels"]):
        raise ValueError(
            "StarVLA dataset active_action_dim does not match its Gymnasium action catalog"
        )
    if (
        model_cfg["framework"]["action_model"]["action_env_dim"]
        != manifest["active_action_dim"]
    ):
        raise ValueError(
            "StarVLA model action_env_dim does not match the dataset manifest"
        )


def build_starvla_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
) -> PolicyRunner:
    policy_cfg = config["policy"]
    env_cfg = config["env"]
    integration_env_name = env_cfg["name"]
    if integration_env_name == "gymnasium":
        model_cfg = _load_starvla_model_config(policy_cfg["model_config_path"])
        _validate_gymnasium_starvla_contract(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            model_cfg=model_cfg,
        )
        semantic_env_name = env_cfg["task_name"]
        action_refs = env_cfg["action_order"]
        base_prompt = env_cfg["base_prompt"]
    else:
        semantic_env_name = integration_env_name
        action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
        base_prompt = None
    unnorm_key = policy_cfg["unnorm_key"] if "unnorm_key" in policy_cfg else None
    wrapper_cls = _load_policy_wrapper_class()
    wrapper_kwargs: dict[str, Any] = dict(
        ckpt_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        use_bf16=True,
        unnorm_key=unnorm_key,
        action_output_mode="rl_games",
        rl_games_env_name=integration_env_name,
        rl_games_action_layout=(
            policy_cfg["action_layout"] if "action_layout" in policy_cfg else None
        ),
        rl_games_multibinary_threshold=(
            policy_cfg["multibinary_threshold"]
            if "multibinary_threshold" in policy_cfg
            else None
        ),
    )
    if integration_env_name == "gymnasium":
        wrapper_kwargs["rl_games_action_env_dim"] = len(action_refs)
    wrapper = wrapper_cls(**wrapper_kwargs)
    latency_prompt_map = (
        load_latency_prompt_map(policy_cfg["latency_prompt_map_path"])
        if "latency_prompt_map_path" in policy_cfg
        else None
    )
    latency_prompt_key = policy_cfg.get("latency_prompt_key")
    prompt_mode = policy_cfg.get("prompt_mode")
    image_transform_config = policy_cfg.get("image_transform_config")
    if integration_env_name != "gymnasium":
        model_cfg = (
            _load_starvla_model_config(policy_cfg["model_config_path"])
            if "model_config_path" in policy_cfg
            else None
        )
    return StarVlaPolicyRunner(
        wrapper=wrapper,
        checkpoint_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        unnorm_key=unnorm_key,
        env_name=semantic_env_name,
        action_resolver=action_resolver,
        action_refs=action_refs,
        latency_prompt_map=latency_prompt_map,
        base_prompt=base_prompt,
        latency_prompt_key=latency_prompt_key,
        prompt_mode=prompt_mode,
        obs_resize=tuple(env_cfg["obs_resize"]) if env_cfg.get("obs_resize") else None,
        image_transform_config=image_transform_config,
        observation_stride_raw_frames=_observation_stride_raw_frames(config),
        model_cfg=model_cfg,
    )


def build_live_starvla_policy(
    *,
    framework: Any,
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    action_resolver: ActionResolver,
) -> StarVlaPolicyRunner:
    """Build a StarVLA policy around a *live* in-memory framework (no reload).

    Mirrors ``build_starvla_policy`` but swaps the ckpt-loading
    ``PolicyServerWrapper`` for :class:`LiveStarVlaWrapper`, so the trainer's
    resident model is evaluated directly. ``model_cfg`` is the in-memory model
    config (e.g. ``read_mode_config`` output) the wrapper would otherwise read
    from disk.
    """
    policy_cfg = config["policy"]
    unnorm_key = policy_cfg["unnorm_key"] if "unnorm_key" in policy_cfg else None
    env_cfg = config["env"]
    integration_env_name = env_cfg["name"]
    if integration_env_name == "gymnasium":
        normalized_model_cfg = _normalized_model_cfg(model_cfg)
        _validate_gymnasium_starvla_contract(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            model_cfg=normalized_model_cfg,
        )
        semantic_env_name = env_cfg["task_name"]
        action_refs = env_cfg["action_order"]
        base_prompt = env_cfg["base_prompt"]
    else:
        semantic_env_name = integration_env_name
        action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
        base_prompt = None
    wrapper_kwargs: dict[str, Any] = dict(
        framework=framework,
        model_cfg=model_cfg,
        env_name=integration_env_name,
        action_layout=policy_cfg["action_layout"] if "action_layout" in policy_cfg else None,
        multibinary_threshold=(
            policy_cfg["multibinary_threshold"]
            if "multibinary_threshold" in policy_cfg
            else None
        ),
    )
    if integration_env_name == "gymnasium":
        wrapper_kwargs["rl_games_action_env_dim"] = len(action_refs)
    wrapper = LiveStarVlaWrapper(**wrapper_kwargs)
    latency_prompt_map = (
        load_latency_prompt_map(policy_cfg["latency_prompt_map_path"])
        if "latency_prompt_map_path" in policy_cfg
        else None
    )
    latency_prompt_key = policy_cfg.get("latency_prompt_key")
    prompt_mode = policy_cfg.get("prompt_mode")
    image_transform_config = policy_cfg.get("image_transform_config")
    return StarVlaPolicyRunner(
        wrapper=wrapper,
        checkpoint_path=policy_cfg.get("checkpoint_path", ""),
        device=policy_cfg.get("device", "cuda"),
        unnorm_key=unnorm_key,
        env_name=semantic_env_name,
        action_resolver=action_resolver,
        action_refs=action_refs,
        latency_prompt_map=latency_prompt_map,
        base_prompt=base_prompt,
        latency_prompt_key=latency_prompt_key,
        prompt_mode=prompt_mode,
        obs_resize=tuple(env_cfg["obs_resize"]) if env_cfg.get("obs_resize") else None,
        image_transform_config=image_transform_config,
        observation_stride_raw_frames=_observation_stride_raw_frames(config),
        model_cfg=normalized_model_cfg if integration_env_name == "gymnasium" else None,
    )


__all__ = [
    "LiveStarVlaWrapper",
    "StarVlaPolicyRunner",
    "apply_starvla_model_input_config",
    "build_live_starvla_policy",
    "build_starvla_policy",
    "decode_starvla_action",
    "observation_data_to_hwc_uint8_frames",
    "prepare_starvla_checkpoint_input_config",
]
