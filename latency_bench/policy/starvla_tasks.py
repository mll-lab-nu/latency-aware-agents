"""StarVLA inference using the task's training observation/action contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.data.starvla_tasks import denormalize, normalize
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.starvla import STARVLA_ROOT


class TaskStarVlaPolicyRunner(PolicyRunner):
    """Map task RGB/state into a StarVLA model and decode its action chunk."""

    def __init__(self, framework, *, policy_config: dict, model_config: dict, contract: dict):
        self.framework = framework
        self.policy_config = policy_config
        self.model_config = model_config
        self.contract = contract

    def _example(self, observation: Observation) -> dict:
        cfg = self.policy_config
        state = normalize(
            observation.metadata[cfg["state_info_key"]],
            self.contract["normalization"]["state"],
        ).reshape(1, self.contract["state_dim"])
        data_cfg = self.model_config["datasets"]["vla_data"]
        height, width = data_cfg["obs_image_size"]
        images = [
            Image.fromarray(frame).resize((width, height))
            for frame in observation.metadata[cfg["image_views_info_key"]]
        ]
        if data_cfg["image_mode"] == "stitch_views":
            from starVLA.training.trainer_utils.trainer_tools import stitch_frames

            # MIKASA's two simultaneous views form one Wan observation, not a video.
            images = [stitch_frames(images, grid=data_cfg["stitch_grid"], size=(width, height))]
        example = {"image": images, "state": state, "lang": self.contract["prompt"]}
        if "action_prefix" in observation.metadata:
            example["action_prefix"] = normalize(
                observation.metadata["action_prefix"],
                self.contract["normalization"]["action"],
            )
            example["action_prefix_mask"] = observation.metadata["action_prefix_mask"]
        return example

    def predict(self, observation: Observation) -> PolicyOutput:
        return self.predict_batch([observation])[0]

    def predict_batch(self, observations: list[Observation]) -> list[PolicyOutput]:
        prediction = self.framework.predict_action(
            examples=[self._example(observation) for observation in observations]
        )
        actions = denormalize(
            prediction["normalized_actions"], self.contract["normalization"]["action"]
        )
        return [
            PolicyOutput(
                action=Action(value=chunk[0].tolist(), name="task_command"),
                action_chunk=chunk,
                raw_output=chunk.tolist(),
                metadata={"policy_type": "starvla", "task": self.contract["task"]},
            )
            for chunk in actions
        ]


def build_task_starvla_policy(config: dict) -> TaskStarVlaPolicyRunner:
    # StarVLA and torch are optional in the simulator process; workers own them.
    import torch
    from huggingface_hub import snapshot_download
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.framework.share_tools import read_mode_config

    cfg = config["policy"]
    model_config, _ = read_mode_config(cfg["checkpoint_path"])
    if "backbone_path" in cfg:
        backbone_path = cfg["backbone_path"]
    else:
        backbone = model_config["backbone"]
        backbone_path = snapshot_download(
            backbone["repo_id"],
            allow_patterns=backbone["allow_patterns"],
            local_dir=STARVLA_ROOT / "playground/Pretrained_models" / backbone["repo_id"].split("/")[-1],
        )
    framework = baseframework.from_pretrained(cfg["checkpoint_path"], backbone_path=backbone_path)
    framework = framework.to(device=cfg["device"], dtype=torch.bfloat16).eval()
    contract = json.loads(Path(cfg["task_contract_path"]).read_text(encoding="utf-8"))
    return TaskStarVlaPolicyRunner(
        framework, policy_config=cfg, model_config=model_config, contract=contract
    )
