from __future__ import annotations

import json
from contextlib import nullcontext
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from latency_bench.core.actions import ActionResolver
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.timing import current_profiler
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.data.ghost_trail import (
    GhostTrailConfig, build_flappy_ghost_trail_window, build_demon_attack_ghost_trail_window,
)
from latency_bench.data.state_normalization import min_max_normalize_state
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.envs.gymnasium_task import (
    gymnasium_action_space_contract,
    gymnasium_task_contract,
)
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy import _starvla_input as model_input, _starvla_actions as action_decode
from latency_bench.policy.starvla_prompts import load_latency_prompt_map, resolve_starvla_prompt


class StarVlaPolicyRunner(PolicyRunner):
    """Translate observations and model outputs using the task action contract."""

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
        env_fps: float,
        obs_fps: float,
        model_cfg: Mapping[str, Any] | None = None,
        state_normalization: Mapping[str, Any] | None = None,
        state_source: str | None = None,
        image_views_info_key: str | None = None,
        action_output_type: str | None = None,
        action_noise_seeded: bool = False,
    ) -> None:
        self._env_fps = env_fps
        self._obs_fps = obs_fps
        self._wrapper = wrapper
        self._action_noise_seeded = action_noise_seeded
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
        model_cfg = model_input._normalized_model_cfg(wrapper._model_cfg if model_cfg is None else model_cfg)
        self._include_state = model_cfg["datasets"]["vla_data"]["include_state"]
        self._state_dim = model_cfg["framework"]["action_model"]["state_dim"] if self._include_state else None
        self._state_normalization = dict(state_normalization or {})
        self._state_source = state_source
        self._image_views_info_key = image_views_info_key
        self._action_output_type = action_output_type
        self._input_plan = model_input.resolve_model_input_plan(
            model_cfg, observation_stride_raw_frames,
        )

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
            action_output_type=(
                prediction["action_output_type"]
                if self._action_output_type is None
                else self._action_output_type
            ),
            raw_action_scores=None if raw_action_scores is None else raw_action_scores[index, 0],
        )

    def _build_example(self, observation: Observation) -> dict[str, Any]:
        frame_source = observation.metadata[
            ENV_RAW_RGB_FRAME_STACK_INFO_KEY
            if self._image_views_info_key is None
            else self._image_views_info_key
        ]
        frames = observation_data_to_hwc_uint8_frames(frame_source)  # oldest .. newest
        transformed = self._transformed_frame(frames=frames, observation=observation)

        if self._image_views_info_key is not None:
            pass
        elif self._input_plan.packed:
            if transformed is not None:
                raise ValueError(
                    "WanOFT packed image sequences require image_transform=raw_rgb"
                )
            if len(frames) < self._input_plan.raw_span:
                raise ValueError(
                    "WanOFT packed image sequence requires "
                    f"{self._input_plan.raw_span} raw frames for "
                    f"{self._input_plan.frame_count} decision observations at stride "
                    f"{self._input_plan.stride}, got {len(frames)}"
                )
            frames = frames[
                -self._input_plan.raw_span
                :: self._input_plan.stride
            ]
        elif transformed is not None:
            frames = [transformed]
        elif self._input_plan.image_mode == "single" or self._input_plan.kv_memory:
            frames = frames[-1:]
        else:
            # Select the temporal observation window to match training (_pack_sample).
            raw_span = self._input_plan.raw_span
            frames = frames[-raw_span :: self._input_plan.stride]

        prompt = resolve_starvla_prompt(
            env_name=self._env_name,
            observation_metadata={**observation.metadata, "env_fps": self._env_fps, "obs_fps": self._obs_fps},
            latency_prompt_map=self._latency_prompt_map,
            base_prompt=self._base_prompt,
            latency_prompt_key=self._latency_prompt_key,
            prompt_mode=self._prompt_mode,
        )

        if self._input_plan.image_mode == "stitch":
            if transformed is not None:
                raise ValueError("image_transform is not compatible with image_mode=stitch")
            # Tile the window into one image; matches _pack_sample's stitch branch
            # (raw frames passed to stitch_frames, which resizes each cell to 224).
            images = [_get_stitch_frames()(frames, grid=self._input_plan.stitch_grid, size=(224, 224))]
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
        if self._action_noise_seeded:
            example["action_noise_seed"] = derive_seed(
                observation.metadata["action_noise_seed"], stream=observation.env_step,
            )
        if self._input_plan.kv_memory:
            example["slot_id"] = observation.metadata["slot_id"]
        elif "slot_id" in observation.metadata:
            example["slot_id"] = observation.metadata["slot_id"]
        if self._include_state:
            if self._state_source == "transport":
                state = np.asarray(observation.data["transport"], dtype=np.float32)
                example["state"] = state.reshape(1, self._state_dim)
            elif self._state_normalization:
                state = np.asarray(
                    observation.metadata["gymnasium_state"], dtype=np.float32
                )
                state_min = np.asarray(self._state_normalization["min"], dtype=np.float32)
                state_max = np.asarray(self._state_normalization["max"], dtype=np.float32)
                state = min_max_normalize_state(state, state_min, state_max)
                example["state"] = state.reshape(1, self._state_dim)
            else:
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
            return build_demon_attack_ghost_trail_window(window)

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
        action, action_metadata = action_decode.action_from_starvla_payload(
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
            "input_frame_count": len(example["image"]),
            "image_transform": self._image_transform,
            "action_output_type": action_output_type,
            "action_payload": action_decode.to_jsonable_action_payload(payload),
            "kv_memory_enabled": self._input_plan.kv_memory,
            **action_metadata,
        }
        if self._input_plan.packed:
            metadata["image_sequence_length"] = self._input_plan.frame_count
            metadata["input_frame_raw_stride"] = self._input_plan.stride
            metadata["input_frame_raw_span"] = self._input_plan.raw_span
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


def _extract_observation_array(data: Any) -> npt.NDArray[Any]:
    if isinstance(data, Mapping):
        return np.asarray(data["observation"])
    return np.asarray(data)


def _as_uint8_image(frame: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    return np.ascontiguousarray(frame, dtype=np.uint8)


_STITCH_FRAMES = None


def _get_stitch_frames():
    """Lazily import starVLA's stitch_frames (starVLA path is added at runtime)."""
    global _STITCH_FRAMES
    if _STITCH_FRAMES is None:
        model_input._ensure_starvla_path()
        from starVLA.training.trainer_utils.trainer_tools import stitch_frames

        _STITCH_FRAMES = stitch_frames
    return _STITCH_FRAMES


def prepare_starvla_checkpoint_input_config(config: dict[str, Any]) -> None:
    """Apply the saved checkpoint input contract before env construction."""
    if config["policy"]["type"] != "starvla":
        return

    policy_cfg = config["policy"]
    if "task_contract_path" in policy_cfg:
        if config["env"]["name"] == "gymnasium":
            contract = json.loads(
                Path(policy_cfg["task_contract_path"]).read_text(encoding="utf-8")
            )
            config["env"]["state_space"] = {"labels": contract["state_labels"]}
            if contract["robot_type"] in ("latency_balance_profile_h8", "latency_balance_profile_h16"):
                config["env"]["name"] = "balance_profile"
                config["env"]["action_context_horizon"] = contract["action_horizon"]
        config["env"]["frame_stack"] = 1
        return
    if "model_config_path" in policy_cfg:
        model_cfg = model_input._load_starvla_model_config(policy_cfg["model_config_path"])
    else:
        model_input._ensure_starvla_path()
        from starVLA.model.framework.share_tools import read_mode_config

        saved_model_cfg, _norm_stats = read_mode_config(policy_cfg["checkpoint_path"])
        model_cfg = model_input._normalized_model_cfg(saved_model_cfg)
    if config["env"]["name"] == "gymnasium":
        image_size = model_cfg["rl_games"]["env_eval"]["image_size"]
        config["env"]["obs_resize"] = [image_size, image_size]
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
    model_input.apply_starvla_model_input_config(
        config,
        model_cfg=model_cfg,
        image_transform=image_transform,
    )


def _load_policy_wrapper_class() -> Any:
    model_input._ensure_starvla_path()
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    return PolicyServerWrapper


def _profiler_stage(profiler: Any, name: str) -> Any:
    return profiler.time(name) if profiler is not None else nullcontext()


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
        gymnasium_action_space_type: str = "discrete",
        action_layout: str | None = None,
        multibinary_threshold: float | None = None,
    ) -> None:
        self._framework = framework
        self._model_cfg = model_cfg
        self._rl_games_env_name = str(env_name)
        self._rl_games_action_env_dim = rl_games_action_env_dim
        self._gymnasium_action_space_type = gymnasium_action_space_type
        (
            self._decode_rl_games_actions,
            resolve_deadly_action_decode_spec,
            resolve_asterix_action_decode_spec,
        ) = action_decode._load_rl_games_action_decode()
        self._action_layout = action_layout
        self._multibinary_threshold = multibinary_threshold
        if self._rl_games_env_name == "deadly_corridor":
            self._action_layout, self._multibinary_threshold = resolve_deadly_action_decode_spec(
                model_cfg,
                action_layout=action_layout,
                multibinary_threshold=multibinary_threshold,
            )
        elif self._rl_games_env_name == "asterix":
            self._action_layout = resolve_asterix_action_decode_spec(
                model_cfg,
                action_layout=action_layout,
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
            if self._gymnasium_action_space_type == "box":
                decode_kwargs["gymnasium_action_space_type"] = "box"
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
                asterix_action_layout=(
                    self._action_layout
                    if self._rl_games_env_name == "asterix"
                    else None
                ),
                **decode_kwargs,
            )


_LEGACY_GYMNASIUM_TASK_NAMES = {
    "ant_rgb_state": "ant",
    "half_cheetah_rgb_state": "half_cheetah",
    "hopper_rgb_state": "hopper",
    "humanoid_rgb_state": "humanoid",
    "inverted_pendulum_rgb_state": "inverted_pendulum",
    "swimmer_rgb_state": "swimmer",
    "walker2d_rgb_state": "walker2d",
}


def _canonical_gymnasium_contract_namespace(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = dict(contract)
    task_name = canonical["task_name"]
    if task_name in _LEGACY_GYMNASIUM_TASK_NAMES:
        canonical["task_name"] = _LEGACY_GYMNASIUM_TASK_NAMES[task_name]
    if canonical["env_id"] == "LatencyBench/HopperRgbState-v0":
        canonical["env_id"] = "LatencyBench/Hopper-v0"
    canonical["registration_imports"] = [
        "latency_bench.envs.gymnasium_hopper"
        if module == "latency_bench.envs.gymnasium_hopper_rgb_state"
        else module
        for module in canonical["registration_imports"]
    ]
    return canonical


def _validate_gymnasium_starvla_contract(
    *,
    env_cfg: Mapping[str, Any],
    policy_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    eval_contract = gymnasium_task_contract(env_cfg)
    manifest_task = manifest.get("gymnasium_task")
    expected = policy_cfg.get(
        "gymnasium_training_task_contract", manifest_task or eval_contract
    )
    comparable_eval_contract = {**eval_contract, "make_kwargs": expected["make_kwargs"]}
    # Calibration changes wall-clock speed, while preserving physical steps per observation.
    if math.isclose(
        eval_contract["env_fps"] / eval_contract["obs_fps"],
        expected["env_fps"] / expected["obs_fps"],
        rel_tol=1e-12,
    ):
        comparable_eval_contract.update(
            env_fps=expected["env_fps"], obs_fps=expected["obs_fps"]
        )
    if _canonical_gymnasium_contract_namespace(
        comparable_eval_contract
    ) != _canonical_gymnasium_contract_namespace(expected):
        raise ValueError(
            "Evaluation Gymnasium task contract does not match the StarVLA training contract or dataset manifest"
        )
    if manifest.get("integration_name", "gymnasium") != "gymnasium":
        raise ValueError("StarVLA task manifest is not a Gymnasium handoff")
    if manifest_task is not None:
        if _canonical_gymnasium_contract_namespace(
            manifest_task
        ) != _canonical_gymnasium_contract_namespace(expected):
            raise ValueError(
                "Evaluation Gymnasium task contract does not match the StarVLA dataset manifest"
            )
    model_contract = model_cfg["datasets"]["vla_data"].get("gymnasium_task_contract")
    if model_contract is not None:
        if _canonical_gymnasium_contract_namespace(
            model_contract
        ) != _canonical_gymnasium_contract_namespace(expected):
            raise ValueError(
                "Evaluation Gymnasium task contract does not match the StarVLA model config"
            )
    action_space = gymnasium_action_space_contract(env_cfg)
    action_layout = str(policy_cfg.get("action_layout", "") or "").strip().lower()
    is_asterix_factorized = (
        str(env_cfg.get("task_name", "")) == "asterix"
        and action_layout in {"factorized_6", "factorized6", "asterix_factorized_6", "asterix_factorized6"}
    )
    if not is_asterix_factorized and manifest["active_action_dim"] != len(action_space["labels"]):
        raise ValueError(
            "StarVLA dataset active_action_dim does not match its Gymnasium action catalog"
        )
    model_action_dim = int(model_cfg["framework"]["action_model"]["action_env_dim"])
    manifest_action_dim = int(manifest["active_action_dim"])
    if policy_cfg.get("gymnasium_cross_task_eval"):
        if model_action_dim < manifest_action_dim:
            raise ValueError(
                "StarVLA model action_env_dim is smaller than the evaluation action space"
            )
    elif model_action_dim != manifest_action_dim:
        raise ValueError(
            "StarVLA model action_env_dim does not match the dataset manifest"
        )
    model_uses_state = bool(model_cfg["datasets"]["vla_data"]["include_state"])
    manifest_has_state_metadata = (
        "uses_state" in manifest or "state_labels" in manifest
    )
    manifest_uses_state = bool(manifest.get("uses_state", model_uses_state))
    if manifest_has_state_metadata:
        if policy_cfg.get("state_source") != "transport" and manifest_uses_state != ("state_space" in expected):
            raise ValueError(
                "StarVLA dataset uses_state does not match the Gymnasium state space"
            )
        if manifest_uses_state != model_uses_state:
            raise ValueError(
                "StarVLA dataset uses_state does not match the model include_state"
            )
    if manifest_has_state_metadata and manifest_uses_state:
        state_labels = manifest["state_labels"]
        expected_state_labels = expected["state_space"]["labels"] if policy_cfg.get("state_source") != "transport" else state_labels
        if state_labels != expected_state_labels:
            raise ValueError(
                "StarVLA dataset state_labels do not match the Gymnasium state space"
            )
        if manifest["state_dim"] != len(state_labels):
            raise ValueError(
                "StarVLA dataset state_dim does not match its state_labels"
            )
        if (
            model_cfg["framework"]["action_model"]["state_dim"]
            != manifest["state_dim"]
        ):
            raise ValueError(
                "StarVLA model state_dim does not match the dataset manifest"
            )
        if not manifest["state_normalization"]:
            raise ValueError(
                "StarVLA state-enabled dataset manifest is missing state_normalization"
            )


def _starvla_runner_kwargs(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    model_cfg: Mapping[str, Any] | None,
    *,
    base_prompt: str | None,
) -> dict[str, Any]:
    """Resolve task and input settings shared by checkpoint and resident models."""
    env_cfg = config["env"]
    policy_cfg = config["policy"]
    if env_cfg["name"] == "gymnasium":
        if "task_manifest_path" in policy_cfg:
            task_manifest = json.loads(
                Path(policy_cfg["task_manifest_path"]).read_text(encoding="utf-8")
            )
        else:
            task_manifest = {
                "integration_name": "gymnasium",
                "gymnasium_task": policy_cfg.get(
                    "gymnasium_training_task_contract",
                    gymnasium_task_contract(env_cfg),
                ),
                "active_action_dim": int(
                    policy_cfg.get(
                        "gymnasium_active_action_dim",
                        len(env_cfg.get("action_order", [])),
                    )
                ),
            }
        _validate_gymnasium_starvla_contract(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            model_cfg=model_cfg,
            manifest=task_manifest,
        )
        semantic_env_name = env_cfg["task_name"]
        action_refs = env_cfg.get("action_order", [])
        base_prompt = env_cfg["base_prompt"]
        state_normalization = task_manifest.get("state_normalization")
    else:
        semantic_env_name = env_cfg["name"]
        action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
        state_normalization = policy_cfg["state_normalization"] if "state_normalization" in policy_cfg else None
    kwargs = dict(
        unnorm_key=policy_cfg.get("unnorm_key"),
        env_name=semantic_env_name,
        action_resolver=action_resolver,
        action_refs=action_refs,
        latency_prompt_map=(
            load_latency_prompt_map(policy_cfg["latency_prompt_map_path"])
            if "latency_prompt_map_path" in policy_cfg
            else None
        ),
        base_prompt=base_prompt,
        latency_prompt_key=policy_cfg.get("latency_prompt_key"),
        prompt_mode=policy_cfg.get("prompt_mode"),
        obs_resize=tuple(env_cfg["obs_resize"]) if env_cfg.get("obs_resize") else None,
        image_transform_config=policy_cfg.get("image_transform_config"),
        observation_stride_raw_frames=model_input._observation_stride_raw_frames(config),
        env_fps=env_cfg["env_fps"],
        obs_fps=env_cfg["obs_fps"],
        model_cfg=model_cfg,
        state_normalization=state_normalization,
        state_source=policy_cfg["state_source"] if "state_source" in policy_cfg else None,
    )
    # Opt-in noise control; existing real/checkpoint callers retain global RNG.
    if "action_noise_seeded" in policy_cfg:
        kwargs["action_noise_seeded"] = policy_cfg["action_noise_seeded"]
    return kwargs


def build_starvla_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
) -> PolicyRunner:
    policy_cfg = config["policy"]
    if "task_contract_path" in policy_cfg:
        model_input._ensure_starvla_path()
        from latency_bench.policy.starvla_tasks import build_task_starvla_policy

        return build_task_starvla_policy(config)
    env_cfg = config["env"]
    integration_env_name = env_cfg["name"]
    model_cfg = (
        model_input._load_starvla_model_config(policy_cfg["model_config_path"])
        if integration_env_name == "gymnasium" or "model_config_path" in policy_cfg
        else None
    )
    runner_kwargs = _starvla_runner_kwargs(
        config, action_resolver, model_cfg, base_prompt=env_cfg.get("base_prompt")
    )
    wrapper_cls = _load_policy_wrapper_class()
    wrapper_kwargs: dict[str, Any] = dict(
        ckpt_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        use_bf16=True,
        unnorm_key=runner_kwargs["unnorm_key"],
        action_output_mode=(
            policy_cfg["action_output_mode"]
            if "action_output_mode" in policy_cfg
            else "rl_games"
        ),
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
    if "backbone_path" in policy_cfg:
        wrapper_kwargs["backbone_path"] = policy_cfg["backbone_path"]
    if integration_env_name == "gymnasium":
        action_space = gymnasium_action_space_contract(env_cfg)
        wrapper_kwargs["rl_games_action_env_dim"] = len(action_space["labels"])
        if action_space["type"] == "box":
            wrapper_kwargs["rl_games_gymnasium_action_space_type"] = "box"
        wrapper_kwargs["rl_games_env_name"] = integration_env_name
    wrapper = wrapper_cls(**wrapper_kwargs)
    return StarVlaPolicyRunner(
        wrapper=wrapper,
        checkpoint_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        **runner_kwargs,
        image_views_info_key=(
            policy_cfg["image_views_info_key"]
            if "image_views_info_key" in policy_cfg
            else None
        ),
        action_output_type=(
            policy_cfg["action_output_type"]
            if "action_output_type" in policy_cfg
            else None
        ),
    )


def build_live_starvla_policy(
    *,
    framework: Any,
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    action_resolver: ActionResolver | None = None,
) -> PolicyRunner:
    """Build a StarVLA policy around a *live* in-memory framework (no reload).

    Mirrors ``build_starvla_policy`` but swaps the ckpt-loading
    ``PolicyServerWrapper`` for :class:`LiveStarVlaWrapper`, so the trainer's
    resident model is evaluated directly. ``model_cfg`` is the in-memory model
    config (e.g. ``read_mode_config`` output) the wrapper would otherwise read
    from disk.
    """
    policy_cfg = config["policy"]
    if "task_contract_path" in policy_cfg:
        from latency_bench.policy.starvla_tasks import TaskStarVlaPolicyRunner

        contract = json.loads(
            Path(policy_cfg["task_contract_path"]).read_text(encoding="utf-8")
        )
        return TaskStarVlaPolicyRunner(
            framework,
            policy_config=policy_cfg,
            model_config=model_cfg,
            contract=contract,
        )

    env_cfg = config["env"]
    integration_env_name = env_cfg["name"]
    normalized_model_cfg = (
        model_input._normalized_model_cfg(model_cfg)
        if integration_env_name == "gymnasium"
        else None
    )
    # Resident evaluation historically takes non-Gymnasium prompts from the map.
    runner_kwargs = _starvla_runner_kwargs(
        config, action_resolver, normalized_model_cfg, base_prompt=None
    )
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
        action_space = gymnasium_action_space_contract(env_cfg)
        wrapper_kwargs["rl_games_action_env_dim"] = len(action_space["labels"])
        if action_space["type"] == "box":
            wrapper_kwargs["gymnasium_action_space_type"] = "box"
        wrapper_kwargs["env_name"] = integration_env_name
    wrapper = LiveStarVlaWrapper(**wrapper_kwargs)
    return StarVlaPolicyRunner(
        wrapper=wrapper,
        checkpoint_path=policy_cfg.get("checkpoint_path", ""),
        device=policy_cfg.get("device", "cuda"),
        **runner_kwargs,
    )


__all__ = [
    "LiveStarVlaWrapper",
    "StarVlaPolicyRunner",
    "build_live_starvla_policy",
    "build_starvla_policy",
    "observation_data_to_hwc_uint8_frames",
    "prepare_starvla_checkpoint_input_config",
]
