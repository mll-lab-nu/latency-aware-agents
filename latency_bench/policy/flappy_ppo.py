from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import register_policy
from modeling.flappy_ppo import (
    FlappyPpoAgent,
    FlappyPpoEvalObservationPipeline,
    FlappyPpoObservationConfig,
    build_agent_from_checkpoint,
    load_flappy_ppo_checkpoint,
    observation_config_from_checkpoint,
)


class FlappyPpoPolicyRunner(PolicyRunner):
    def __init__(
        self,
        agent: FlappyPpoAgent,
        observation_config: FlappyPpoObservationConfig,
        action_by_raw_id: Mapping[int, Action],
        device: str | torch.device = "cpu",
        deterministic: bool = True,
        checkpoint_global_step: int = 0,
    ):
        self.device = torch.device(device)
        self.agent = agent.to(self.device)
        self.agent.eval()
        self.observation_config = observation_config
        self.preprocessor = FlappyPpoEvalObservationPipeline(observation_config)
        self.action_by_raw_id = dict(action_by_raw_id)
        self.deterministic = bool(deterministic)
        self.checkpoint_global_step = int(checkpoint_global_step)

    @property
    def current_stacked_observation(self):
        return self.preprocessor.current_observation

    def predict(self, observation: Observation) -> PolicyOutput:
        obs_tensor = self.preprocessor.build_tensor(observation, device=self.device)
        with torch.no_grad():
            if self.deterministic:
                logits = self.agent.get_logits(obs_tensor)
                raw_action = int(torch.argmax(logits, dim=1).item())
            else:
                action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                raw_action = int(action.item())

        action = self.action_by_raw_id.get(raw_action)
        if action is None:
            raise ValueError(f"Flappy PPO emitted unmapped action id: {raw_action}")
        return PolicyOutput(
            action=action,
            raw_output=raw_action,
            metadata={
                "policy_type": "flappy_ppo",
                "checkpoint_global_step": self.checkpoint_global_step,
            },
        )


@register_policy("flappy_ppo")
def _build_flappy_ppo_policy(config: dict[str, Any], action_resolver: ActionResolver) -> PolicyRunner:
    policy_cfg = config["policy"]
    checkpoint_path = Path(policy_cfg["checkpoint_path"])
    device = _resolve_device(policy_cfg.get("device", "auto"))
    checkpoint = load_flappy_ppo_checkpoint(checkpoint_path, map_location=device)
    observation_config = observation_config_from_checkpoint(checkpoint)
    agent = build_agent_from_checkpoint(checkpoint, device=device)
    action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
    actions = action_resolver.resolve_many(action_refs)
    action_by_raw_id = _build_action_map(actions, action_resolver)
    return FlappyPpoPolicyRunner(
        agent=agent,
        observation_config=observation_config,
        action_by_raw_id=action_by_raw_id,
        device=device,
        deterministic=bool(policy_cfg.get("deterministic", True)),
        checkpoint_global_step=int(checkpoint.get("global_step", 0)),
    )


def _resolve_device(device: Any) -> str:
    device_name = str(device)
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _build_action_map(actions: list[Action], action_resolver: ActionResolver) -> dict[int, Action]:
    action_by_raw_id: dict[int, Action] = {}
    for action in actions:
        try:
            raw_id = int(action.value)
        except (TypeError, ValueError):
            continue
        action_by_raw_id[raw_id] = action

    for raw_id in (0, 1):
        if raw_id not in action_by_raw_id:
            action_by_raw_id[raw_id] = action_resolver.resolve(raw_id)

    return action_by_raw_id


__all__ = ["FlappyPpoPolicyRunner"]

