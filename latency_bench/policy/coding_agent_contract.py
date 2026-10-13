"""Native agent inputs follow a selected VLA contract; outputs use env units."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from gymnasium import spaces

from latency_bench.core.types import Action, PolicyOutput
from latency_bench.data.prompts import BASE_PROMPTS, IMAGE_TRANSFORM_CONTEXTS
from latency_bench.integrations import starvla_runtime


def prepare_coding_agent_config(config):
    """Prepare the reference observation before constructing any environment."""
    from latency_bench.policy.starvla import prepare_starvla_checkpoint_input_config

    # One native conversation owns one episode and one authentication directory.
    if (config["executor"]["mode"] != "simulated"
            or config["executor"]["simulated_inference_pool"]
            or config["evaluation"]["eval_parallel_envs"] != 1):
        raise ValueError("coding_agent requires simulated mode, no inference pool, and one parallel environment")
    policy = config["policy"]
    reference = policy["reference_policy"]
    prepare_starvla_checkpoint_input_config({**config, "policy": reference})
    config["env"]["export_env_raw_rgb_frames"] = True
    for key in ("task_contract_path", "action_prefix", "action_layout"):
        if key in reference:
            policy[key] = reference[key]


def build_reference_observer(config, action_resolver):
    """Use the existing input builders without loading weights or inference code."""
    from latency_bench.policy.starvla import StarVlaPolicyRunner, _starvla_runner_kwargs
    from latency_bench.policy.starvla_tasks import TaskStarVlaPolicyRunner

    cfg = config["policy"]["reference_policy"]
    model_cfg = starvla_runtime.load_model_config(cfg["model_config_path"])
    if "task_contract_path" in cfg:
        contract = json.loads(Path(cfg["task_contract_path"]).read_text())
        runner = TaskStarVlaPolicyRunner(
            None, policy_config=cfg, model_config=model_cfg, contract=contract,
        )
        inputs = {"image_mode": model_cfg["datasets"]["vla_data"]["image_mode"],
                  "resize_hw": model_cfg["datasets"]["vla_data"]["obs_image_size"],
                  "video_keys": contract["video_keys"],
                  "state": {"format": "min_max_normalized_to_minus1_plus1", "dim": contract["state_dim"],
                            "normalization": contract["normalization"]["state"]}}
        if "state_labels" in contract:
            inputs["state"]["labels"] = contract["state_labels"]
        elif config["env"]["name"] == "mikasa_intercept_grab_fast" and contract["state_dim"] == 7:
            from latency_bench.envs.tasks.mikasa import MIKASA_STATE_LABELS

            inputs["state"]["labels"] = list(MIKASA_STATE_LABELS)
            inputs["state"]["raw_units"] = "TCP xyz in metres, roll/pitch/yaw in radians, gripper width in metres"
        return runner._example, contract["prompt"], contract, inputs
    env = config["env"]
    prompt = env["base_prompt"] if "base_prompt" in env else BASE_PROMPTS[env["name"]]
    if "image_transform_config" in cfg and "image_transform" in cfg["image_transform_config"]:
        transform = cfg["image_transform_config"]["image_transform"]
        if transform in IMAGE_TRANSFORM_CONTEXTS:
            prompt += IMAGE_TRANSFORM_CONTEXTS[transform]
    kwargs = _starvla_runner_kwargs(
        {**config, "policy": cfg}, action_resolver, model_cfg, base_prompt=prompt,
    )
    runner = StarVlaPolicyRunner(
        wrapper=None, checkpoint_path=cfg["checkpoint_path"], device="cpu", **kwargs,
        image_views_info_key=cfg["image_views_info_key"] if "image_views_info_key" in cfg else None,
    )
    inputs = {"image_plan": asdict(runner._input_plan), "resize_hw": runner._obs_resize, "state": None}
    if runner._include_state:
        state = {"dim": runner._state_dim}
        if runner._state_source == "transport":
            state["format"] = "transport"
            if "action_history_decisions" in env:
                state.update(decisions=env["action_history_decisions"], order="oldest_to_newest",
                             row_layout="admitted flag, encoded issued action, encoded last applied action; categorical vectors omit category 0, multibinary vectors use turn/move/strafe/attack order")
        elif runner._state_normalization:
            state.update(format="min_max_normalized_to_minus1_plus1", normalization=runner._state_normalization,
                         labels=env["state_space"]["labels"])
        else:
            state["format"] = "constant_zero_vector"
        inputs["state"] = state
    return runner.build_observation_input, prompt, None, inputs


def bind_coding_agent_action_space(config, env):
    """Record the actual adapter space, including physical joint limits/order."""
    if config["policy"]["type"] != "coding_agent":
        return
    space = env.gym_env.action_space
    if isinstance(space, spaces.Discrete):
        spec = {"type": "discrete", "start": int(space.start), "n": int(space.n)}
    elif isinstance(space, spaces.MultiBinary):
        spec = {"type": "multibinary", "shape": list(space.shape)}
    elif isinstance(space, spaces.MultiDiscrete):
        spec = {"type": "multidiscrete", "nvec": space.nvec.tolist(), "start": space.start.tolist()}
    elif isinstance(space, spaces.Tuple):
        spec = {"type": "multidiscrete", "nvec": [int(part.n) for part in space.spaces],
                "start": [int(part.start) for part in space.spaces]}
    elif isinstance(space, spaces.Box):
        spec = {"type": "box", "low": space.low.tolist(), "high": space.high.tolist(), "dtype": space.dtype.name}
    else:
        raise TypeError(f"Unsupported coding-agent action space: {space}")
    noop = env.noop_action.value
    spec["noop"] = noop.tolist() if isinstance(noop, np.ndarray) else noop
    env_cfg = config["env"]
    if env_cfg["name"] in ("deadly_corridor", "defend_the_line"):
        if spec["type"] == "multibinary":
            from latency_bench.policy._starvla_actions import DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER

            spec["labels"] = list(DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER)
        else:
            spec["labels"] = ["turn", "move", "strafe", "attack"]
            spec["choices"] = [["none", "left", "right"], ["none", "forward", "backward"],
                               ["none", "left", "right"], ["none", "fire"]]
    elif env_cfg["name"] == "balance_profile" or (
            env_cfg["name"] == "gymnasium" and env_cfg["task_name"] == "g1_sonic_balance_simple"):
        from latency_bench.envs.tasks.humanoidbench.contract import (
            G1_BODY_JOINT_NAMES, G1_HAND_JOINT_NAMES, MUJOCO_INDEX_FOR_POLICY,
        )

        spec["labels"] = [G1_BODY_JOINT_NAMES[i] for i in MUJOCO_INDEX_FOR_POLICY] + list(G1_HAND_JOINT_NAMES)
        spec["units"] = "radians; absolute joint positions in the listed policy order"
    elif env_cfg["name"] == "mikasa_intercept_grab_fast":
        from latency_bench.envs.tasks.mikasa import MIKASA_ACTION_LABELS

        spec["labels"] = list(MIKASA_ACTION_LABELS)
        spec["control_mode"] = "pd_ee_delta_pose"
        spec["units"] = ("Normalized root-frame translation increments (scale 0.1 metres), root-aligned "
                         "rotation-vector increments (maximum norm 0.1 radians); gripper -1 closes, +1 opens")
    elif "action_space" in env_cfg:
        spec["labels"] = env_cfg["action_space"]["labels"]
    config["policy"]["env_action_space"] = spec


class AgentActionCodec:
    """Validate external JSON at the action API, with no clipping or repair."""

    def __init__(self, spec, action_resolver, task_contract):
        self.spec = spec
        self.resolver = action_resolver
        self.horizon = 1 if task_contract is None else task_contract["action_horizon"]
        self.task_contract = task_contract
        kind = spec["type"]
        if kind == "discrete":
            self.space = spaces.Discrete(spec["n"], start=spec["start"])
        elif kind == "multibinary":
            self.space = spaces.MultiBinary(tuple(spec["shape"]))
        elif kind == "multidiscrete":
            self.space = spaces.MultiDiscrete(spec["nvec"], start=spec["start"])
        elif kind == "box":
            self.space = spaces.Box(np.asarray(spec["low"], dtype=spec["dtype"]),
                                    np.asarray(spec["high"], dtype=spec["dtype"]))
        else:
            raise ValueError(f"Unsupported coding-agent action type: {kind}")

    def description(self):
        description = {**self.spec, "action_horizon": self.horizon}
        if self.spec["type"] == "discrete":
            description["actions"] = [
                {"id": i, "name": self.resolver.resolve(i).name}
                for i in range(self.space.start, self.space.start + self.space.n)
            ]
        if self.task_contract is not None and "action_labels" in self.task_contract:
            description["labels"] = self.task_contract["action_labels"]
        return description

    def _action(self, value):
        kind = self.spec["type"]
        if kind == "discrete":
            valid = type(value) is int and self.space.contains(value)
        else:
            array = np.asarray(value)
            # Reject JSON strings, booleans, fractional buttons and nonfinite numbers.
            scalar_types = (int, float) if kind == "box" else (int,)
            numeric = isinstance(value, list) and all(type(item) in scalar_types for item in value)
            valid = (numeric and array.shape == self.space.shape
                     and np.isfinite(array).all())
            if valid and kind == "box":
                valid = bool(((array >= self.space.low) & (array <= self.space.high)).all())
            elif valid:
                valid = array.dtype.kind in ("i", "u") and self.space.contains(array)
        if not valid:
            raise ValueError(f"Action is outside the declared environment space: {value!r}")
        if kind == "discrete":
            return self.resolver.resolve(value)
        return Action(value=value, name="agent_command", is_noop=np.array_equal(value, self.spec["noop"]))

    def decode(self, text, observation):
        payload = json.loads(text)
        key = "action" if self.horizon == 1 else "actions"
        if not isinstance(payload, dict) or set(payload) != {key}:
            raise ValueError(f"Expected exactly one JSON field: {key}")
        values = [payload[key]] if self.horizon == 1 else payload[key]
        if not isinstance(values, list) or len(values) != self.horizon:
            raise ValueError(f"Expected {self.horizon} actions in the chunk")
        actions = [self._action(value) for value in values]
        chunk = None
        if self.task_contract is not None:
            chunk = np.asarray(values, dtype=np.float32)
            # Match TaskStarVlaPolicyRunner: prefix heads retain the committed plan.
            if "action_prefix" in observation.metadata:
                mask = observation.metadata["action_prefix_mask"]
                chunk[mask] = observation.metadata["action_prefix"][mask]
            actions[0] = self._action(chunk[0].tolist())
        return PolicyOutput(action=actions[0], action_chunk=chunk, raw_output=text)
