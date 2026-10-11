"""StarVLA inference using the task's training observation/action contract."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.integrations.starvla.task_contracts import denormalize, normalize
from latency_bench.integrations.starvla import runtime as starvla_runtime
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.model_source import model_file
from latency_bench.policy.starvla_prompts import load_latency_prompt_map, resolve_starvla_prompt


class TaskStarVlaPolicyRunner(PolicyRunner):
    """Map task RGB/state into a StarVLA model and decode its action chunk."""

    def __init__(self, framework, *, policy_config: dict, model_config: dict, contract: dict):
        self.framework = framework
        self.policy_config = policy_config
        self.model_config = model_config
        self.contract = contract
        self.latency_prompt_map = (
            load_latency_prompt_map(policy_config["latency_prompt_map_path"])
            if "latency_prompt_map_path" in policy_config
            else None
        )

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
        language = self.contract["prompt"]
        if self.latency_prompt_map is not None:
            language = resolve_starvla_prompt(
                env_name=self.contract["task"],
                observation_metadata=observation.metadata,
                latency_prompt_map=self.latency_prompt_map,
                prompt_mode=cfg["prompt_mode"] if "prompt_mode" in cfg else None,
                latency_prompt_key=cfg["latency_prompt_key"] if "latency_prompt_key" in cfg else None,
            )
        example = {"image": images, "state": state, "lang": language}
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
        # Prefix heads were excluded from the loss; retain the frozen controller plan.
        for chunk, observation in zip(actions, observations):
            if "action_prefix" in observation.metadata:
                mask = observation.metadata["action_prefix_mask"]
                chunk[mask] = observation.metadata["action_prefix"][mask]
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
    starvla_runtime.ensure_starvla_path()
    import torch
    from starVLA.model.framework.base_framework import baseframework

    cfg = config["policy"]
    if "model_config" in cfg:
        model_config = starvla_runtime.load_model_config(model_file(cfg, "model_config"))
    else:
        model_config = starvla_runtime.load_checkpoint_config(model_file(cfg, "checkpoint"))
    framework = baseframework.from_pretrained(
        model_file(cfg, "checkpoint"), backbone_path=cfg["backbone_path"]
    )
    framework = framework.to(device=config["executor"]["inference_devices"][0], dtype=torch.bfloat16).eval()
    contract = json.loads(Path(cfg["task_contract_path"]).read_text(encoding="utf-8"))
    return TaskStarVlaPolicyRunner(
        framework, policy_config=cfg, model_config=model_config, contract=contract
    )
