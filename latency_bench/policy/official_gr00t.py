"""NVIDIA Isaac-GR00T policy boundary.

The official package is intentionally imported only when this policy is built.
The benchmark's Python 3.10 environment therefore keeps the existing StarVLA
GR00T path independent from the official Python 3.12 environment.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_GR00T_ACTION_DIM,
)
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.policy.base import PolicyRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_GR00T_ROOT = REPO_ROOT / "third_party" / "Isaac-GR00T"
OFFICIAL_GR00T_PYTHON = OFFICIAL_GR00T_ROOT / ".venv" / "bin" / "python"

OFFICIAL_GR00T_PROPRIO_INFO_KEY = "official_gr00t_proprio"
OFFICIAL_GR00T_VIDEO_KEY = "ego_view"
OFFICIAL_GR00T_STATE_KEY = "state"
OFFICIAL_GR00T_LANGUAGE_KEY = "annotation.human.task_description"
# Direct Gr00tPolicy returns the configured action modality key.
OFFICIAL_GR00T_ACTION_KEY = "action"


def _load_gr00t_policy() -> type[Any]:
    from gr00t.policy import Gr00tPolicy

    return Gr00tPolicy


def official_gr00t_python_executable() -> str:
    return str(OFFICIAL_GR00T_PYTHON)


def official_gr00t_worker_sys_path() -> list[str]:
    paths = json.loads(
        subprocess.check_output(
            [
                official_gr00t_python_executable(),
                "-I",
                "-c",
                "import json, sys; print(json.dumps(sys.path))",
            ],
            text=True,
        )
    )
    return [str(REPO_ROOT), *paths]


class OfficialGr00tPolicyRunner(PolicyRunner):
    """Adapt LatencyBench observations to the official nested GR00T API."""

    def __init__(
        self,
        *,
        policy: Any,
        prompt: str,
        model_path: str,
        embodiment_tag: str,
        device: str,
    ) -> None:
        self._policy = policy
        self._prompt = prompt
        self._model_path = model_path
        self._embodiment_tag = embodiment_tag
        self._device = device

    def predict(self, observation: Observation) -> PolicyOutput:
        return self.predict_batch([observation])[0]

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        batch = _batch_official_observations(
            [
                _official_observation(observation, prompt=self._prompt)
                for observation in observations
            ]
        )
        action_dict, info = self._policy.get_action(batch)
        del info
        chunks = np.asarray(action_dict[OFFICIAL_GR00T_ACTION_KEY])
        if (
            chunks.ndim != 3
            or chunks.shape[0] != len(observations)
            or chunks.shape[2] != G1_GR00T_ACTION_DIM
        ):
            raise ValueError(
                "official GR00T action contract requires "
                f"shape ({len(observations)}, *, {G1_GR00T_ACTION_DIM}), got {chunks.shape}"
            )
        return [
            PolicyOutput(
                action=Action(value=chunks[index, 0], name="official_gr00t"),
                action_chunk=chunks[index],
                metadata={
                    "policy_type": "official_gr00t",
                    "model_path": self._model_path,
                    "embodiment_tag": self._embodiment_tag,
                    "device": self._device,
                },
            )
            for index in range(len(observations))
        ]


def _official_observation(observation: Observation, *, prompt: str) -> dict[str, Any]:
    frames = observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
    state = observation.metadata[OFFICIAL_GR00T_PROPRIO_INFO_KEY]
    return {
        "video": {
            OFFICIAL_GR00T_VIDEO_KEY: frames[-1:].reshape(1, 1, *frames.shape[1:])
        },
        "state": {
            OFFICIAL_GR00T_STATE_KEY: state.reshape(1, 1, -1)
        },
        "language": {OFFICIAL_GR00T_LANGUAGE_KEY: [[prompt]]},
    }


def _batch_official_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "video": {
            OFFICIAL_GR00T_VIDEO_KEY: np.concatenate(
                [observation["video"][OFFICIAL_GR00T_VIDEO_KEY] for observation in observations],
                axis=0,
            )
        },
        "state": {
            OFFICIAL_GR00T_STATE_KEY: np.concatenate(
                [observation["state"][OFFICIAL_GR00T_STATE_KEY] for observation in observations],
                axis=0,
            )
        },
        "language": {
            OFFICIAL_GR00T_LANGUAGE_KEY: [
                observation["language"][OFFICIAL_GR00T_LANGUAGE_KEY][0]
                for observation in observations
            ]
        },
    }


def build_official_gr00t_policy(
    config: dict[str, Any], action_resolver: ActionResolver
) -> PolicyRunner:
    del action_resolver
    policy_cfg = config["policy"]
    env_cfg = config["env"]
    model_path = policy_cfg["model_path"]
    embodiment_tag = policy_cfg["embodiment_tag"]
    device = policy_cfg["device"]
    official_policy = _load_gr00t_policy()(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        device=device,
        strict=policy_cfg["strict"],
    )
    return OfficialGr00tPolicyRunner(
        policy=official_policy,
        prompt=env_cfg["base_prompt"],
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        device=device,
    )


__all__ = [
    "OfficialGr00tPolicyRunner",
    "build_official_gr00t_policy",
    "official_gr00t_python_executable",
    "official_gr00t_worker_sys_path",
]
