"""Resolve fixed action conditioning and freeze it with each policy observation."""

import json
import math
from pathlib import Path

import numpy as np

from latency_bench.core.types import Action, Observation


def resolve_action_prefix(settings, *, env_fps, action_horizon, action_dim):
    """Resolve the user configuration once; the saved result is the run contract."""
    mode = settings["mode"]
    if mode == "none":
        return {"mode": "none"}
    if mode not in ("profile_mean", "profile_p95", "fixed_frames"):
        raise ValueError(f"Unknown action prefix mode: {mode}")
    if mode == "fixed_frames":
        resolved = {
            **settings, "env_fps": env_fps,
            "action_horizon": action_horizon, "action_dim": action_dim,
        }
    elif "frames" in settings:
        resolved = settings
    else:
        profile_path = Path(settings["profile_path"])
        statistic = "mean_ms" if mode == "profile_mean" else "p95_ms"
        latency_ms = json.loads(profile_path.read_text())["summary"][statistic]
        resolved = {
            **settings, "profile_path": str(profile_path), statistic: latency_ms,
            "env_fps": env_fps, "frames": math.ceil(latency_ms * env_fps / 1000),
            "action_horizon": action_horizon, "action_dim": action_dim,
        }
    frames = resolved["frames"]
    if not 0 <= frames < action_horizon:
        raise ValueError(
            f"action prefix requires 0 <= K < H, got K={frames}, H={action_horizon}"
        )
    return resolved


def prepare_action_prefix(config):
    """Resolve task-policy configuration before creating executors or workers.

    Policies outside the task-contract path have no action conditioning unless
    explicitly configured. Task-policy recipes must select their mode.
    """
    policy = config["policy"]
    if "action_prefix" not in policy and "task_contract_path" not in policy:
        policy["action_prefix"] = {"mode": "none"}
    settings = policy["action_prefix"]
    if settings["mode"] == "none":
        return settings
    contract = (settings if "frames" in settings else
                json.loads(Path(policy["task_contract_path"]).read_text()))
    settings = resolve_action_prefix(
        settings, env_fps=config["env"]["env_fps"],
        action_horizon=contract["action_horizon"], action_dim=contract["action_dim"],
    )
    policy["action_prefix"] = settings
    return settings


def action_prefix_arrays(scheduler, frame, settings):
    """Project only controller-visible actions, without advancing the live queue."""
    if settings is None or settings["mode"] == "none":
        return None, None
    frames, horizon = settings["frames"], settings["action_horizon"]
    actions = np.zeros((horizon, settings["action_dim"]), dtype=np.float32)
    for index, action in enumerate(scheduler.project_actions(frame, frames)):
        actions[index] = action.value
    return actions, np.arange(horizon) < frames


def encode_prefix_state(proprio, action_chunk, prefix_mask):
    """Encode observation-relative committed actions, excluding every teacher suffix."""
    prefix = np.where(np.asarray(prefix_mask)[..., None], action_chunk, 0)
    return np.concatenate((proprio, prefix.reshape(-1))).astype(np.float32)


def with_action_prefix(observation, scheduler, settings):
    """Freeze prefix and any environment-provided action context before IPC."""
    prefix, mask = action_prefix_arrays(scheduler, observation.env_step, settings)
    metadata = observation.metadata.copy()
    if "action_prefix_state_key" in metadata:
        key = metadata.pop("action_prefix_state_key")
        metadata[key] = encode_prefix_state(metadata[key], prefix, mask)
    if "returned_action_context" in metadata:
        context = metadata.pop("returned_action_context")
        horizon = context["horizon"]
        chunk = np.zeros((horizon, len(context["low"])), dtype=np.float32)
        indices = np.zeros(horizon, dtype=np.float32)
        event, index = scheduler.scheduled_chunk_event(
            observation.env_step, observation.sim_time_ms)
        if event is not None:
            end = horizon if context["frames"] is None else min(horizon, index + context["frames"])
            for head in range(index, end):
                action = event.action_chunk[head]
                value = action.value if isinstance(action, Action) else action
                ordered = np.asarray(value)[context["order"]]
                chunk[head] = 2 * (ordered - context["low"]) / (context["high"] - context["low"]) - 1
            indices[index] = 1
        key = context["state_key"]
        metadata[key] = np.concatenate((metadata[key], chunk.reshape(-1), indices)).astype(np.float32)
    # This layer owns these fields, including their absence for a baseline.
    metadata.pop("action_prefix", None)
    metadata.pop("action_prefix_mask", None)
    if prefix is not None:
        metadata.update(action_prefix=prefix, action_prefix_mask=mask)
    return Observation(
        data=observation.data, env_step=observation.env_step,
        sim_time_ms=observation.sim_time_ms, metadata=metadata,
    )
