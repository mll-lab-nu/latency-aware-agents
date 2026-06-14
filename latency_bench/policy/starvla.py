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
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import register_policy
from latency_bench.policy.starvla_prompts import load_latency_prompt_map, resolve_starvla_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
STARVLA_ROOT = REPO_ROOT / "starVLA"
ENV_RAW_RGB_FRAME_STACK_INFO_KEY = "env_raw_rgb_frame_stack"

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
        unnorm_key: str,
        env_name: str,
        action_resolver: ActionResolver,
        action_refs: Sequence[Any],
        latency_prompt_map: dict[str, Any] | None = None,
        obs_resize: tuple[int, int] | None = None,
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
        self._include_state = _include_state_from_wrapper(wrapper)
        self._state_dim = _state_dim_from_wrapper(wrapper) if self._include_state else None

    def predict(self, observation: Observation) -> PolicyOutput:
        return self.predict_batch([observation])[0]

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        examples = [self._build_example(observation) for observation in observations]
        prediction = self._wrapper.predict_action(examples=examples, unnorm_key=self._unnorm_key)
        actions = np.asarray(prediction["actions"])
        raw_action_scores = (
            np.asarray(prediction["raw_action_scores"])
            if "raw_action_scores" in prediction
            else None
        )
        return [
            self._policy_output(
                observation=observation,
                example=example,
                action_payload=actions[index, 0],
                action_output_type=prediction["action_output_type"],
                raw_action_scores=None if raw_action_scores is None else raw_action_scores[index, 0],
            )
            for index, (observation, example) in enumerate(zip(observations, examples))
        ]

    def _build_example(self, observation: Observation) -> dict[str, Any]:
        frame_source = observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        frames = observation_data_to_hwc_uint8_frames(frame_source)
        if self._obs_resize is not None:
            height, width = self._obs_resize
            # match training preprocessing exactly: gr00t LeRobotSingleDataset._pack_sample
            # does `Image.fromarray(image).resize((224, 224))` (PIL default resample = BICUBIC).
            frames = [
                np.asarray(Image.fromarray(frame).resize((width, height)), dtype=np.uint8)
                for frame in frames
            ]
        prompt = resolve_starvla_prompt(
            env_name=self._env_name,
            observation_metadata=observation.metadata,
            latency_prompt_map=self._latency_prompt_map,
        )
        example = {
            "image": [Image.fromarray(frame) for frame in frames],
            "lang": prompt,
        }
        if self._include_state:
            example["state"] = np.zeros((1, self._state_dim), dtype=np.float32)
        return example

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
        )
        metadata = {
            "policy_type": "starvla",
            "prompt_source": "latency_prompt_map" if self._latency_prompt_map is not None else "base",
            "checkpoint_path": self._checkpoint_path,
            "unnorm_key": self._unnorm_key,
            "device": self._device,
            "prompt_sha256": sha256_text(example["lang"]),
            "input_frame_count": len(example["image"]),
            "action_output_type": action_output_type,
            "action_payload": to_jsonable_action_payload(payload),
            **action_metadata,
        }
        if raw_action_scores is not None:
            metadata["raw_action_scores"] = [
                float(item) for item in np.asarray(raw_action_scores, dtype=np.float32).tolist()
            ]
        if "latency_raw_frames" in observation.metadata:
            metadata["latency_raw_frames"] = observation.metadata["latency_raw_frames"]
        if "latency_ms" in observation.metadata:
            metadata["latency_ms"] = observation.metadata["latency_ms"]
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


def action_from_starvla_payload(
    *,
    payload: npt.NDArray[Any],
    env_name: str,
    action_by_raw_id: Mapping[int, Action],
) -> tuple[Action, dict[str, Any]]:
    if str(env_name) == "demon_attack":
        return demon_attack_action_from_id(int(np.asarray(payload).reshape(-1)[0]))
    if str(env_name) == "deadly_corridor":
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


def _state_dim_from_wrapper(wrapper: Any) -> int:
    return wrapper._model_cfg["framework"]["action_model"]["state_dim"]


def _include_state_from_wrapper(wrapper: Any) -> bool:
    return wrapper._model_cfg["datasets"]["vla_data"]["include_state"]


def _load_policy_wrapper_class() -> Any:
    starvla_root = str(STARVLA_ROOT)
    if starvla_root not in sys.path:
        sys.path.insert(0, starvla_root)
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    return PolicyServerWrapper


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
    wrapper_cls = _load_policy_wrapper_class()
    wrapper = wrapper_cls(
        ckpt_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        use_bf16=True,
        unnorm_key=policy_cfg["unnorm_key"],
        action_output_mode="rl_games",
        rl_games_env_name=config["env"]["name"],
    )
    latency_prompt_map = (
        load_latency_prompt_map(policy_cfg["latency_prompt_map_path"])
        if "latency_prompt_map_path" in policy_cfg
        else None
    )
    return StarVlaPolicyRunner(
        wrapper=wrapper,
        checkpoint_path=policy_cfg["checkpoint_path"],
        device=policy_cfg["device"],
        unnorm_key=policy_cfg["unnorm_key"],
        env_name=config["env"]["name"],
        action_resolver=action_resolver,
        action_refs=policy_cfg.get("actions", action_resolver.default_action_refs()),
        latency_prompt_map=latency_prompt_map,
        obs_resize=tuple(config["env"]["obs_resize"]) if config["env"].get("obs_resize") else None,
    )


__all__ = [
    "StarVlaPolicyRunner",
    "decode_starvla_action",
    "observation_data_to_hwc_uint8_frames",
]
