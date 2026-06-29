from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from latency_bench.core.actions import ActionResolver
from latency_bench.core.timing import current_profiler
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.data.ghost_trail import GhostTrailConfig, build_flappy_ghost_trail_window
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import register_policy
from latency_bench.policy.starvla_prompts import load_latency_prompt_map, resolve_starvla_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
STARVLA_ROOT = REPO_ROOT / "starVLA"
ENV_RAW_RGB_FRAME_STACK_INFO_KEY = "env_raw_rgb_frame_stack"
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
        latency_prompt_key: int | str | None = None,
        prompt_mode: str | None = None,
        obs_resize: tuple[int, int] | None = None,
        image_transform_config: Mapping[str, Any] | None = None,
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
        self._latency_prompt_key = latency_prompt_key
        self._prompt_mode = str(prompt_mode or "default").strip().lower()
        self._image_transform_config = dict(image_transform_config or {"image_transform": "raw_rgb"})
        self._image_transform = str(
            self._image_transform_config.get("image_transform", "raw_rgb") or "raw_rgb"
        ).strip().lower()
        model_cfg = _normalized_model_cfg_from_wrapper(wrapper)
        self._include_state = _include_state_from_model_cfg(model_cfg)
        self._state_dim = _state_dim_from_model_cfg(model_cfg) if self._include_state else None
        vla_data = (model_cfg.get("datasets", {}) or {}).get("vla_data", {}) or {}
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

        if transformed is not None:
            frames = [transformed]
        else:
            # Select the temporal observation window to match training (_pack_sample).
            if self._image_mode == "single":
                frames = frames[-1:]
            else:
                n = min(self._num_obs_frames, len(frames))
                frames = frames[-n:]

        prompt = resolve_starvla_prompt(
            env_name=self._env_name,
            observation_metadata=observation.metadata,
            latency_prompt_map=self._latency_prompt_map,
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
        if self._image_transform != "flappy_ghost_trail":
            raise ValueError(f"Unsupported StarVLA image_transform={self._image_transform!r}")
        if self._env_name != "flappy":
            raise ValueError("image_transform=flappy_ghost_trail is only supported for env_name=flappy")

        config = GhostTrailConfig(
            image_transform=self._image_transform,
            history_frames=int(self._image_transform_config.get("history_frames", 5)),
            gamma=float(self._image_transform_config.get("gamma", 1.3)),
            min_alpha=int(self._image_transform_config.get("min_alpha", 35)),
            ground_fraction=float(self._image_transform_config.get("ground_fraction", 0.22)),
            scroll_px_per_step=float(self._image_transform_config.get("scroll_px_per_step", 4.0)),
        )
        max_frames = max(1, int(config.history_frames) + 1)
        valid_count = min(len(frames), max(1, int(observation.env_step) + 1), max_frames)
        window = [np.asarray(frame, dtype=np.uint8) for frame in frames[-valid_count:]]
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
    if str(env_name) == "demon_attack":
        return demon_attack_action_from_vector(vector)
    if str(env_name) == "deadly_corridor":
        return deadly_corridor_action_from_vector(vector)
    return indexed_action_from_vector(vector, action_by_raw_id)


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


def indexed_action_from_vector(
    vector: npt.NDArray[Any],
    action_by_raw_id: Mapping[int, Action],
) -> tuple[Action, dict[str, Any]]:
    # argmax over the first n_actions dims only — the head may be a wider
    # (bridge-padded) vector; matches starVLA eval_core decode_discrete_argmax(raw, n).
    n_actions = len(action_by_raw_id)
    raw_action_id = int(np.argmax(np.asarray(vector)[:n_actions]))
    return action_by_raw_id[raw_action_id], {"raw_action_id": raw_action_id}


def demon_attack_action_from_vector(vector: npt.NDArray[Any]) -> tuple[Action, dict[str, Any]]:
    n_actions = len(DEMON_ATTACK_ACTION_LABELS)
    action_id = int(np.argmax(np.asarray(vector)[:n_actions]))
    return demon_attack_action_from_id(action_id)


def demon_attack_action_from_id(action_id: int) -> tuple[Action, dict[str, Any]]:
    action = Action(
        value=action_id,
        name=DEMON_ATTACK_ACTION_LABELS[action_id],
        is_noop=action_id == 0,
        is_oneshot=False,
    )
    return action, {"raw_action_id": action_id, "action_label": action.name}


def deadly_corridor_action_from_vector(vector: npt.NDArray[Any]) -> tuple[Action, dict[str, Any]]:
    if vector.shape[0] == 54:
        return deadly_corridor_action_from_joint_vector(vector)
    if vector.shape[0] == 11:
        return deadly_corridor_action_from_factorized_vector(vector)
    if vector.shape[0] == 7:
        return deadly_corridor_action_from_semantic_vector(vector)
    raise ValueError(f"Deadly Corridor StarVLA action vector expected 7, 11, or 54 values, got {vector.shape[0]}")


def deadly_corridor_action_from_factorized_vector(vector: npt.NDArray[Any]) -> tuple[Action, dict[str, Any]]:
    turn = int(np.argmax(vector[0:3]))
    move = int(np.argmax(vector[3:6]))
    strafe = int(np.argmax(vector[6:9]))
    attack = int(np.argmax(vector[9:11]))
    return deadly_corridor_action_from_tuple(
        action_value=[turn, move, strafe, attack],
        metadata={"action_layout": "deadly_corridor_factorized_11"},
    )


def deadly_corridor_action_from_joint_vector(vector: npt.NDArray[Any]) -> tuple[Action, dict[str, Any]]:
    joint_id = int(np.argmax(vector))
    attack = joint_id % 2
    strafe = (joint_id // 2) % 3
    move = (joint_id // 6) % 3
    turn = (joint_id // 18) % 3
    return deadly_corridor_action_from_tuple(
        action_value=[turn, move, strafe, attack],
        metadata={
            "raw_action_id": joint_id,
            "action_layout": "deadly_corridor_joint_54",
        },
    )


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


def deadly_corridor_action_from_semantic_vector(vector: npt.NDArray[Any]) -> tuple[Action, dict[str, Any]]:
    action_id = int(np.argmax(vector))
    semantic_actions = (
        ([0, 1, 0, 0], "MOVE_FORWARD"),
        ([0, 2, 0, 0], "MOVE_BACKWARD"),
        ([0, 0, 1, 0], "MOVE_LEFT"),
        ([0, 0, 2, 0], "MOVE_RIGHT"),
        ([1, 0, 0, 0], "TURN_LEFT"),
        ([2, 0, 0, 0], "TURN_RIGHT"),
        ([0, 0, 0, 1], "ATTACK"),
    )
    action_value, action_name = semantic_actions[action_id]
    return Action(
        value=action_value,
        name=action_name,
        is_noop=False,
        is_oneshot=False,
    ), {
        "raw_action_id": action_id,
        "decoded_action_tuple": action_value,
        "action_label": action_name,
        "action_layout": "deadly_corridor_semantic_7",
    }


def _extract_observation_array(data: Any) -> npt.NDArray[Any]:
    if isinstance(data, Mapping):
        return np.asarray(data["observation"])
    return np.asarray(data)


def _as_uint8_image(frame: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _normalized_model_cfg_from_wrapper(wrapper: Any) -> dict[str, Any]:
    _ensure_starvla_path()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.create(wrapper._model_cfg)
    apply_config_compat(cfg)
    _apply_model_family_include_state_compat(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


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


def _load_policy_wrapper_class() -> Any:
    _ensure_starvla_path()
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    return PolicyServerWrapper


def _profiler_stage(profiler: Any, name: str) -> Any:
    from contextlib import nullcontext

    return profiler.time(name) if profiler is not None else nullcontext()


def _load_decode_rl_games_actions() -> Any:
    _ensure_starvla_path()
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    return decode_rl_games_actions


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
        action_layout: str = "",
        multibinary_threshold: float = 0.0,
    ) -> None:
        self._framework = framework
        self._model_cfg = model_cfg
        self._rl_games_env_name = str(env_name)
        self._decode_rl_games_actions = _load_decode_rl_games_actions()
        self._action_layout = str(action_layout or "").strip().lower()
        self._multibinary_threshold = float(multibinary_threshold)

    def _uses_deadly_multibinary(self) -> bool:
        return (
            self._rl_games_env_name == "deadly_corridor"
            and self._action_layout == "multibinary_7"
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
        with _profiler_stage(profiler, "starvla_wrapper_rl_games_decode_ms"):
            if self._uses_deadly_multibinary():
                # Independent per-button thresholding (matches eval_core), instead
                # of the shared argmax semantic_7 decode which can only ever fire a
                # single button. The result is a 7-dim multibinary vector in fixed
                # semantic order; DeadlyCorridorVlaEnvAdapter re-orders it to the
                # live ViZDoom button layout before stepping.
                buttons = (
                    np.asarray(normalized, dtype=np.float32)[..., :7]
                    >= self._multibinary_threshold
                ).astype(np.int64)
                return {
                    "actions": buttons,
                    "raw_action_scores": normalized,
                    "action_output_type": "rl_games_deadly_corridor_multibinary",
                }
            return self._decode_rl_games_actions(
                normalized_actions=normalized,
                env_name=self._rl_games_env_name,
            )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_array(value: npt.NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@register_policy("starvla")
def _build_starvla_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
) -> PolicyRunner:
    policy_cfg = config["policy"]
    unnorm_key = policy_cfg["unnorm_key"] if "unnorm_key" in policy_cfg else None
    wrapper_cls = _load_policy_wrapper_class()
    wrapper = wrapper_cls(
        ckpt_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        use_bf16=True,
        unnorm_key=unnorm_key,
        action_output_mode="rl_games",
        rl_games_env_name=config["env"]["name"],
    )
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
        checkpoint_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        unnorm_key=unnorm_key,
        env_name=config["env"]["name"],
        action_resolver=action_resolver,
        action_refs=policy_cfg.get("actions", action_resolver.default_action_refs()),
        latency_prompt_map=latency_prompt_map,
        latency_prompt_key=latency_prompt_key,
        prompt_mode=prompt_mode,
        obs_resize=tuple(config["env"]["obs_resize"]) if config["env"].get("obs_resize") else None,
        image_transform_config=image_transform_config,
    )


def build_live_starvla_policy(
    *,
    framework: Any,
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    action_resolver: ActionResolver,
) -> StarVlaPolicyRunner:
    """Build a StarVLA policy around a *live* in-memory framework (no reload).

    Mirrors ``_build_starvla_policy`` but swaps the ckpt-loading
    ``PolicyServerWrapper`` for :class:`LiveStarVlaWrapper`, so the trainer's
    resident model is evaluated directly. ``model_cfg`` is the in-memory model
    config (e.g. ``read_mode_config`` output) the wrapper would otherwise read
    from disk.
    """
    policy_cfg = config["policy"]
    unnorm_key = policy_cfg["unnorm_key"] if "unnorm_key" in policy_cfg else None
    env_name = config["env"]["name"]
    wrapper = LiveStarVlaWrapper(
        framework=framework,
        model_cfg=model_cfg,
        env_name=env_name,
        action_layout=str(policy_cfg.get("action_layout", "") or ""),
        multibinary_threshold=float(policy_cfg.get("multibinary_threshold", 0.0) or 0.0),
    )
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
        env_name=env_name,
        action_resolver=action_resolver,
        action_refs=policy_cfg.get("actions", action_resolver.default_action_refs()),
        latency_prompt_map=latency_prompt_map,
        latency_prompt_key=latency_prompt_key,
        prompt_mode=prompt_mode,
        obs_resize=tuple(config["env"]["obs_resize"]) if config["env"].get("obs_resize") else None,
        image_transform_config=image_transform_config,
    )


__all__ = [
    "LiveStarVlaWrapper",
    "StarVlaPolicyRunner",
    "build_live_starvla_policy",
    "decode_starvla_action",
    "observation_data_to_hwc_uint8_frames",
]
