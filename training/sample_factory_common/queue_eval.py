from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from latency_bench.core.latency_distribution import LatencyDistributionSpec, latency_distribution_label
from latency_bench.core.types import LatencySourceType
from training.atari_episode_stats import (
    TRUE_EPISODE_END_INFO_KEY,
    _batched_info_for_agent,
    _flatten_numeric_episode_extra_stats,
    is_true_atari_episode_end,
)
from training.sample_factory_common.checkpoint_io import load_sample_factory_checkpoint_from_paths
from training.sample_factory_common.common import maybe_set_cli_override
from training.sample_factory_common.latency_config import (
    latency_distribution_spec_from_flat_config,
    raw_frames_to_latency_ms,
)


EvalCfgAdjuster = Callable[[Any, Any], None]
DemonAttackEvalEpisodeResult = dict[str, float | int | bool]

_FLAPPY_SF_ENV = "latency_flappy"
_DEMON_ATTACK_SF_ENV = "latency_demon_attack"
_DEMON_ATTACK_VECTOR_BACKENDS = {"serial", "subproc"}


class _QueueEvalEpisodeTracker:
    def __init__(self, *, eval_episodes: int, eval_parallel_envs: int):
        self.eval_episodes = int(eval_episodes)
        self.eval_parallel_envs = int(eval_parallel_envs)
        if self.eval_episodes < 0:
            raise ValueError("eval_episodes must be non-negative")
        if self.eval_parallel_envs <= 0:
            raise ValueError("eval_parallel_envs must be positive")

        initial_slots = min(self.eval_parallel_envs, self.eval_episodes)
        self.started_episodes = int(initial_slots)
        self.completed_episodes = 0
        self.slot_episode_id: list[int | None] = list(range(initial_slots))

    @property
    def active_slots(self) -> int:
        return sum(episode_id is not None for episode_id in self.slot_episode_id)

    @property
    def active_mask(self) -> list[bool]:
        return [episode_id is not None for episode_id in self.slot_episode_id]

    def complete_slot(self, slot: int) -> tuple[int, int | None]:
        slot_idx = int(slot)
        episode_id = self.slot_episode_id[slot_idx]
        if episode_id is None:
            raise RuntimeError(f"slot {slot_idx} is inactive")
        if self.completed_episodes >= self.eval_episodes:
            raise RuntimeError("cannot complete more episodes than requested")

        completed_episode_id = int(episode_id)
        self.completed_episodes += 1

        next_episode_id: int | None = None
        if self.started_episodes < self.eval_episodes:
            next_episode_id = int(self.started_episodes)
            self.started_episodes += 1
        self.slot_episode_id[slot_idx] = next_episode_id
        return completed_episode_id, next_episode_id


def _is_flappy_queue_eval_cfg(cfg: Any) -> bool:
    return str(_cfg_value(cfg, "env") or "") == _FLAPPY_SF_ENV


def _eval_parallel_envs(base_cfg: Any) -> int:
    value = getattr(base_cfg, "eval_parallel_envs", 1)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("evaluation.eval_parallel_envs must be positive")
    return parsed


def _eval_episode_seed(base_cfg: Any, episode_id: int) -> int:
    return int(getattr(base_cfg, "seed", 0) or 0) + 104729 + int(episode_id)


def _demon_attack_vector_backend(*, eval_parallel_envs: int) -> str:
    override = str(os.environ.get("DEMON_EVAL_VECTOR_BACKEND", "")).strip().lower()
    backend = override or ("subproc" if int(eval_parallel_envs) > 1 else "serial")
    if backend not in _DEMON_ATTACK_VECTOR_BACKENDS:
        raise ValueError(
            "DEMON_EVAL_VECTOR_BACKEND must be one of "
            f"{sorted(_DEMON_ATTACK_VECTOR_BACKENDS)}, got {backend!r}"
        )
    return backend


def _demon_attack_effective_parallel_envs(
    *,
    eval_parallel_envs: int,
    eval_episodes: int,
) -> int:
    return min(int(eval_parallel_envs), int(eval_episodes))


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_eval_only_max_episode_steps(cfg: Any, base_cfg: Any) -> None:
    if _is_demon_attack_queue_eval_cfg(cfg) or _is_demon_attack_queue_eval_cfg(base_cfg):
        return

    current_max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
    if current_max_episode_steps > 0:
        return

    eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 0) or 0)
    if eval_max_steps > 0:
        maybe_set_cli_override(cfg, "max_episode_steps", eval_max_steps)


def _apply_demon_attack_raw_eval_runtime_overrides(cfg: Any, base_cfg: Any) -> None:
    if not (_is_demon_attack_queue_eval_cfg(cfg) or _is_demon_attack_queue_eval_cfg(base_cfg)):
        return
    if not bool(getattr(base_cfg, "eval_raw_reward", getattr(cfg, "eval_raw_reward", False))):
        return

    maybe_set_cli_override(cfg, "terminal_on_life_loss", True)
    maybe_set_cli_override(cfg, "clip_reward", False)


def _apply_queue_eval_runtime_overrides(cfg: Any, base_cfg: Any, latency_value: int) -> None:
    maybe_set_cli_override(cfg, "latency_type", "zero" if int(latency_value) == 0 else "fixed")
    maybe_set_cli_override(
        cfg,
        "fixed_latency_ms",
        raw_frames_to_latency_ms(latency_value, env_fps=float(getattr(base_cfg, "env_fps"))),
    )
    eval_parallel_envs = _eval_parallel_envs(base_cfg)
    eval_episodes = int(getattr(base_cfg, "eval_episodes", 20))
    effective_parallel_envs = min(eval_parallel_envs, max(0, eval_episodes))
    maybe_set_cli_override(cfg, "eval_parallel_envs", eval_parallel_envs)
    maybe_set_cli_override(cfg, "num_envs", 1)
    maybe_set_cli_override(cfg, "num_envs_per_worker", 1)
    maybe_set_cli_override(cfg, "env_frameskip", 1)
    maybe_set_cli_override(cfg, "eval_env_frameskip", 1)
    if _is_flappy_queue_eval_cfg(cfg) or _is_flappy_queue_eval_cfg(base_cfg):
        use_gpu_render = _config_bool(getattr(cfg, "use_gpu_render", getattr(base_cfg, "use_gpu_render", False)))
        if eval_parallel_envs > 1 and not use_gpu_render:
            raise ValueError("Flappy queue eval eval_parallel_envs>1 requires env.use_gpu_render=true")
        if use_gpu_render:
            maybe_set_cli_override(cfg, "gpu_render_batch_size", max(1, effective_parallel_envs))
    if (
        (_is_demon_attack_queue_eval_cfg(cfg) or _is_demon_attack_queue_eval_cfg(base_cfg))
        and bool(getattr(base_cfg, "eval_raw_reward", getattr(cfg, "eval_raw_reward", False)))
    ):
        maybe_set_cli_override(cfg, "terminal_on_life_loss", True)
    maybe_set_cli_override(cfg, "no_render", True)
    maybe_set_cli_override(cfg, "save_video", False)
    maybe_set_cli_override(cfg, "fps", 0)
    maybe_set_cli_override(cfg, "eval_deterministic", bool(getattr(base_cfg, "eval_deterministic", True)))
    _apply_eval_only_max_episode_steps(cfg, base_cfg)
    _apply_demon_attack_raw_eval_runtime_overrides(cfg, base_cfg)


def _apply_queue_eval_distribution_runtime_overrides(cfg: Any, base_cfg: Any) -> None:
    maybe_set_cli_override(cfg, "latency_type", getattr(base_cfg, "latency_type"))
    maybe_set_cli_override(cfg, "mean_latency_ms", getattr(base_cfg, "mean_latency_ms", None))
    maybe_set_cli_override(cfg, "std_latency_ms", getattr(base_cfg, "std_latency_ms", None))
    maybe_set_cli_override(cfg, "min_latency_ms", getattr(base_cfg, "min_latency_ms", None))
    maybe_set_cli_override(cfg, "max_latency_ms", getattr(base_cfg, "max_latency_ms", None))
    maybe_set_cli_override(cfg, "latency_seed", getattr(base_cfg, "latency_seed", None))
    maybe_set_cli_override(cfg, "add_latency_info", True)
    max_pending_actions = getattr(base_cfg, "max_pending_actions", None)
    if max_pending_actions is not None:
        maybe_set_cli_override(cfg, "max_pending_actions", int(max_pending_actions))


def _cfg_value(cfg: Any, key: str) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key)
    return getattr(cfg, key, None)


def _is_demon_attack_queue_eval_cfg(cfg: Any) -> bool:
    return str(_cfg_value(cfg, "env") or "") == _DEMON_ATTACK_SF_ENV


def _numeric_scalar(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if hasattr(value, "numel") and callable(value.numel) and value.numel() == 1:
        try:
            item = value.item()
        except (TypeError, ValueError):
            return None
        return None if isinstance(item, bool) else _numeric_scalar(item)
    array = np.asarray(value) if not isinstance(value, (dict, list, tuple, str, bytes)) else None
    if array is not None and array.size == 1:
        item = array.item()
        return None if isinstance(item, bool) else _numeric_scalar(item)
    return None


def _metric_from_info(
    info: dict[str, Any],
    direct_keys: tuple[str, ...],
    extra_stat_keys: tuple[str, ...],
) -> float | None:
    for key in direct_keys:
        value = _numeric_scalar(info.get(key))
        if value is not None:
            return value

    extra_stats = _flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
    for key in extra_stat_keys:
        value = extra_stats.get(key)
        if value is not None:
            return float(value)
    return None


def _truthy_flag_from_info(
    info: dict[str, Any],
    direct_keys: tuple[str, ...],
    extra_stat_keys: tuple[str, ...],
) -> bool:
    for key in direct_keys:
        if key not in info:
            continue
        value = info.get(key)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        numeric_value = _numeric_scalar(value)
        if numeric_value is not None:
            return bool(numeric_value)

    extra_stats = _flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
    for key in extra_stat_keys:
        value = extra_stats.get(key)
        if value is not None:
            return bool(value)
    return False


def _bool_for_agent(value: Any, agent_idx: int) -> bool:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu().numpy()

    array = np.asarray(value)
    if array.shape == ():
        return bool(array.item())
    return bool(array.reshape(-1)[agent_idx])


def _demon_attack_true_eval_episode_end(info: dict[str, Any]) -> bool:
    if TRUE_EPISODE_END_INFO_KEY in info:
        return bool(info[TRUE_EPISODE_END_INFO_KEY])
    truncated = _truthy_flag_from_info(
        info,
        ("TimeLimit.truncated", "truncated"),
        ("episode/truncated", "truncated"),
    )
    return is_true_atari_episode_end(
        terminated=True,
        truncated=truncated,
        info=info,
    )


def _demon_attack_eval_episode_result(
    info: dict[str, Any],
    *,
    fallback_return: float,
    fallback_length: int,
    eval_raw_reward: bool,
    count_life_loss_terminal: bool = False,
) -> DemonAttackEvalEpisodeResult | None:
    if not count_life_loss_terminal and not _demon_attack_true_eval_episode_end(info):
        return None

    raw_return = _metric_from_info(
        info,
        ("episodic_raw_return",),
        ("episode/raw_return", "raw_return"),
    )
    episodic_return = _metric_from_info(
        info,
        ("episodic_return",),
        ("episode/return", "return"),
    )
    length = _metric_from_info(
        info,
        ("episodic_length",),
        ("episode/length", "length"),
    )

    if raw_return is None:
        raw_return = episodic_return
    if eval_raw_reward and raw_return is None:
        raise ValueError(
            "DemonAttack raw eval requires raw episode stats; refusing to use fallback_return as raw_return"
        )

    selected_return = raw_return
    if selected_return is None:
        selected_return = episodic_return
    if selected_return is None:
        selected_return = float(fallback_return)

    if raw_return is None:
        raw_return = float(selected_return)

    timeout = _truthy_flag_from_info(
        info,
        ("TimeLimit.truncated", "timeout"),
        ("episode/timeout", "timeout"),
    ) or _truthy_flag_from_info(
        info,
        ("truncated",),
        ("episode/truncated", "truncated"),
    )

    result: DemonAttackEvalEpisodeResult = {
        "return": float(selected_return),
        "raw_return": float(raw_return),
        "length": int(length if length is not None else fallback_length),
    }
    if timeout:
        result["timeout"] = True
    return result


def _demon_attack_eval_budget_timeout_info(
    *,
    accumulated_return: float,
    episode_length: int,
) -> dict[str, Any]:
    stats = {
        "episode/raw_return": float(accumulated_return),
        "episode/return": float(accumulated_return),
        "episode/length": int(episode_length),
        "episode/truncated": 1.0,
        "episode/timeout": 1.0,
    }
    info = {
        "TimeLimit.truncated": True,
        "truncated": True,
        "episodic_return": float(accumulated_return),
        "episodic_length": int(episode_length),
        "episode_extra_stats": stats,
    }
    info["episodic_raw_return"] = float(accumulated_return)
    return info


def _demon_attack_eval_step_budget_timeout(
    episode_result: DemonAttackEvalEpisodeResult,
    *,
    eval_max_steps: int,
    timed_out_by_eval_budget: bool,
) -> bool:
    if not timed_out_by_eval_budget or int(eval_max_steps) <= 0:
        return False
    return int(episode_result["length"]) >= int(eval_max_steps)


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std_or_nan(values: list[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def _build_queue_eval_result(
    *,
    latency_value: int,
    returns: list[float],
    lengths: list[int],
    eval_episodes: int,
    eval_max_steps: int,
    total_frames: int,
    hit_eval_step_budget: bool,
    raw_returns: list[float] | None = None,
    eval_raw_reward: bool = False,
    env_name: str | None = None,
    timeout_episodes: int = 0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "latency": int(latency_value),
        "mean_return": _mean_or_nan(returns),
        "std_return": _std_or_nan(returns),
        "mean_length": _mean_or_nan([float(length) for length in lengths]),
        "returns": [float(item) for item in returns],
        "lengths": [int(item) for item in lengths],
        "completed_episodes": int(len(returns)),
        "requested_episodes": int(eval_episodes),
        "eval_max_steps": int(eval_max_steps),
        "total_frames": int(total_frames),
        "hit_eval_step_budget": bool(hit_eval_step_budget),
        "timeout_episodes": int(timeout_episodes),
    }

    if raw_returns is not None:
        result["raw_returns"] = [float(item) for item in raw_returns]
        result["mean_raw_return"] = _mean_or_nan(raw_returns)
        result["std_raw_return"] = _std_or_nan(raw_returns)
        result["eval_return_source"] = "raw_return"

    if (
        str(env_name or "") == _DEMON_ATTACK_SF_ENV
        and hit_eval_step_budget
        and len(returns) < int(eval_episodes)
    ):
        if returns:
            result["warning"] = (
                f"Only {len(returns)}/{int(eval_episodes)} DemonAttack eval episodes completed "
                f"before eval stopped at eval_max_steps={int(eval_max_steps)}."
            )
        else:
            result["warning"] = (
                "No DemonAttack eval episode completed within "
                f"eval_max_steps={int(eval_max_steps)}; completed-episode stats are unavailable."
            )
    return result


def _latency_samples_from_infos(infos: Any, num_agents: int, active_mask: np.ndarray | None = None) -> list[int]:
    samples: list[int] = []
    for agent_i in range(int(num_agents)):
        if active_mask is not None and not bool(active_mask[agent_i]):
            continue
        info = _batched_info_for_agent(infos, int(agent_i))
        value = _numeric_scalar(info.get("latency_raw_frames"))
        if value is not None:
            samples.append(max(0, int(value)))
    return samples


def _attach_distribution_eval_metadata(
    result: dict[str, Any],
    spec: LatencyDistributionSpec,
    sampled_frames: list[int],
) -> dict[str, Any]:
    result["latency_distribution"] = spec.to_dict()
    result["latency_distribution_label"] = latency_distribution_label(spec)
    result["sample_count_frames"] = int(len(sampled_frames))
    if sampled_frames:
        values = np.asarray(sampled_frames, dtype=np.float64)
        result["sample_mean_frames"] = float(np.mean(values))
        result["sample_std_frames"] = float(np.std(values))
        result["sample_min_frames"] = int(np.min(values))
        result["sample_max_frames"] = int(np.max(values))
    else:
        result["sample_mean_frames"] = float("nan")
        result["sample_std_frames"] = float("nan")
        result["sample_min_frames"] = float("nan")
        result["sample_max_frames"] = float("nan")
    return result


def init_queue_eval_wandb(base_cfg):
    try:
        import wandb
    except ImportError:
        return None, False

    project = getattr(base_cfg, "wandb_project", None)
    if not project:
        return None, False

    if wandb.run is not None:
        return wandb.run, False

    run = wandb.init(
        project=project,
        group=getattr(base_cfg, "wandb_group", None),
        job_type=getattr(base_cfg, "wandb_job_type", None) or "sample_factory_eval",
        name=f"{getattr(base_cfg, 'experiment', 'sample_factory')}_eval",
        config={
            "experiment": getattr(base_cfg, "experiment", None),
            "env": getattr(base_cfg, "env", None),
            "seed": getattr(base_cfg, "seed", None),
            "eval_episodes": getattr(base_cfg, "eval_episodes", None),
            "eval_max_steps": getattr(base_cfg, "eval_max_steps", None),
        },
        reinit=True,
    )
    return run, True


def log_queue_eval_result(wandb_run, result: dict[str, Any]) -> None:
    if wandb_run is None:
        return

    if "latency_distribution_label" in result:
        label = str(result["latency_distribution_label"])
        prefix = f"queue_eval/distribution/{label}"
        payload = {
            "queue_eval/distribution_label": label,
            f"{prefix}/mean_return": float(result["mean_return"]),
            f"{prefix}/std_return": float(result["std_return"]),
            f"{prefix}/mean_length": float(result["mean_length"]),
        }
        for result_key, suffix in (
            ("mean_raw_return", "mean_raw_return"),
            ("std_raw_return", "std_raw_return"),
            ("completed_episodes", "completed_episodes"),
            ("timeout_episodes", "timeout_episodes"),
            ("sample_count_frames", "sample_count_frames"),
            ("sample_mean_frames", "sample_mean_frames"),
            ("sample_std_frames", "sample_std_frames"),
            ("sample_min_frames", "sample_min_frames"),
            ("sample_max_frames", "sample_max_frames"),
        ):
            if result_key in result:
                payload[f"{prefix}/{suffix}"] = float(result[result_key])
        wandb_run.log(payload)
        for key, value in payload.items():
            if key.startswith(prefix):
                wandb_run.summary[key] = value
        wandb_run.summary["queue_eval/distribution_label"] = label
        return

    latency = int(result["latency"])
    payload = {
        "queue_eval/latency": latency,
        "queue_eval/mean_return": float(result["mean_return"]),
        "queue_eval/std_return": float(result["std_return"]),
        "queue_eval/mean_length": float(result["mean_length"]),
    }
    for result_key, wandb_key in (
        ("mean_raw_return", "queue_eval/mean_raw_return"),
        ("std_raw_return", "queue_eval/std_raw_return"),
        ("completed_episodes", "queue_eval/completed_episodes"),
        ("timeout_episodes", "queue_eval/timeout_episodes"),
    ):
        if result_key in result:
            payload[wandb_key] = float(result[result_key])
    wandb_run.log(payload)
    wandb_run.summary[f"queue_eval/latency_{latency}/mean_return"] = float(result["mean_return"])
    wandb_run.summary[f"queue_eval/latency_{latency}/std_return"] = float(result["std_return"])
    wandb_run.summary[f"queue_eval/latency_{latency}/mean_length"] = float(result["mean_length"])
    for result_key, summary_suffix in (
        ("mean_raw_return", "mean_raw_return"),
        ("std_raw_return", "std_raw_return"),
        ("completed_episodes", "completed_episodes"),
        ("timeout_episodes", "timeout_episodes"),
    ):
        if result_key in result:
            wandb_run.summary[f"queue_eval/latency_{latency}/{summary_suffix}"] = float(result[result_key])


def finish_queue_eval_wandb(wandb_run, owned_run: bool) -> None:
    if wandb_run is None or not owned_run:
        return
    wandb_run.finish()


def queue_eval_output_dir(base_cfg) -> Path | None:
    output_dir = getattr(base_cfg, "output_dir", None)
    if not output_dir:
        return None
    path = Path(str(output_dir)).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def has_queue_latency_distribution(base_cfg) -> bool:
    latency_type = LatencySourceType(str(getattr(base_cfg, "latency_type")))
    return latency_type in {LatencySourceType.NORMAL_DISTRIBUTION, LatencySourceType.UNIFORM_DISTRIBUTION}


def write_queue_eval_result(base_cfg, result: dict[str, Any]) -> None:
    output_dir = queue_eval_output_dir(base_cfg)
    if output_dir is None:
        return

    experiment_name = str(getattr(base_cfg, "experiment", "sample_factory_eval"))
    seed = int(getattr(base_cfg, "seed", 0) or 0)
    distribution_label = result.get("latency_distribution_label")
    payload = {
        "experiment_name": experiment_name,
        "seed": seed,
        "checkpoint_kind": str(getattr(base_cfg, "load_checkpoint_kind", "latest")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mean_return": float(result["mean_return"]),
        "std_return": float(result["std_return"]),
        "mean_length": float(result["mean_length"]),
        "returns": [float(item) for item in result.get("returns", [])],
        "lengths": [int(item) for item in result.get("lengths", [])],
    }
    if distribution_label is not None:
        payload["latency_distribution_label"] = str(distribution_label)
        payload["latency_distribution"] = dict(result.get("latency_distribution") or {})
    else:
        payload["latency"] = int(result["latency"])
    for key in ("raw_returns",):
        if key in result:
            payload[key] = [float(item) for item in result.get(key, [])]
    for key in (
        "mean_raw_return",
        "std_raw_return",
        "completed_episodes",
        "timeout_episodes",
        "requested_episodes",
        "eval_max_steps",
        "total_frames",
        "sample_count_frames",
    ):
        if key in result:
            value = result[key]
            if key.endswith("episodes") or key in {"eval_max_steps", "total_frames", "sample_count_frames"}:
                payload[key] = int(value)
            else:
                payload[key] = float(value)
    for key in (
        "sample_mean_frames",
        "sample_std_frames",
        "sample_min_frames",
        "sample_max_frames",
    ):
        if key in result:
            payload[key] = float(result[key])
    for key in ("hit_eval_step_budget",):
        if key in result:
            payload[key] = bool(result[key])
    for key in ("eval_return_source", "warning"):
        if key in result:
            payload[key] = str(result[key])

    if distribution_label is not None:
        result_file = output_dir / f"queue_eval_distribution_{distribution_label}.json"
    else:
        latency = int(result["latency"])
        result_file = output_dir / f"queue_eval_latency_{latency}.json"
    with result_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    meta = result.get("_meta")
    if isinstance(meta, dict):
        if distribution_label is not None:
            meta_file = output_dir / f"queue_eval_distribution_{distribution_label}.meta.json"
        else:
            meta_file = output_dir / f"queue_eval_latency_{int(result['latency'])}.meta.json"
        with meta_file.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
            handle.write("\n")

    jsonl_path = output_dir / "queue_eval_results.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_sf_policy_state_dict(cfg, actor_critic, device: torch.device) -> None:
    from sample_factory.algo.learning.learner import Learner

    policy_id = getattr(cfg, "policy_index", 0)
    load_checkpoint_kind = getattr(cfg, "load_checkpoint_kind", "latest")
    name_prefix = dict(latest="checkpoint", best="best")[load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = load_sample_factory_checkpoint_from_paths(checkpoints, device)
    try:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    except RuntimeError as exc:
        raise ValueError(
            "Sample Factory eval checkpoint/model shape mismatch. Check env.obs_resize "
            "and use the same observation config for train and eval."
        ) from exc


def prepare_queue_eval_cfg(base_cfg, latency_value: int, adjust_cfg: EvalCfgAdjuster | None = None):
    cfg = copy.deepcopy(base_cfg)
    _apply_queue_eval_runtime_overrides(cfg, base_cfg, latency_value)
    if adjust_cfg is not None:
        adjust_cfg(cfg, base_cfg)
    _apply_demon_attack_raw_eval_runtime_overrides(cfg, base_cfg)
    return cfg


def prepare_queue_eval_distribution_cfg(base_cfg, adjust_cfg: EvalCfgAdjuster | None = None):
    cfg = copy.deepcopy(base_cfg)
    _apply_queue_eval_runtime_overrides(cfg, base_cfg, 0)
    _apply_queue_eval_distribution_runtime_overrides(cfg, base_cfg)
    if adjust_cfg is not None:
        adjust_cfg(cfg, base_cfg)
    _apply_demon_attack_raw_eval_runtime_overrides(cfg, base_cfg)
    return cfg


@torch.no_grad()
def evaluate_queue_latency(base_cfg, latency_value: int, adjust_cfg: EvalCfgAdjuster | None = None) -> dict[str, Any]:
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    cfg = prepare_queue_eval_cfg(base_cfg, latency_value, adjust_cfg=adjust_cfg)
    cfg = load_from_checkpoint(cfg)
    _apply_queue_eval_runtime_overrides(cfg, base_cfg, latency_value)
    if adjust_cfg is not None:
        adjust_cfg(cfg, base_cfg)
    _apply_demon_attack_raw_eval_runtime_overrides(cfg, base_cfg)
    distribution_spec = (
        latency_distribution_spec_from_flat_config(cfg, env_fps=float(getattr(cfg, "env_fps")))
        if has_queue_latency_distribution(cfg)
        else None
    )
    sampled_latency_frames: list[int] = []

    eval_episodes = int(getattr(base_cfg, "eval_episodes", 20))
    eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 30000))
    max_total_frames = int(1e9 if eval_max_steps <= 0 else eval_max_steps * max(1, eval_episodes))
    is_demon_attack_eval = _is_demon_attack_queue_eval_cfg(cfg) or _is_demon_attack_queue_eval_cfg(base_cfg)
    is_flappy_eval = _is_flappy_queue_eval_cfg(cfg) or _is_flappy_queue_eval_cfg(base_cfg)
    eval_parallel_envs = _eval_parallel_envs(base_cfg)
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
    demon_attack_vector_backend = "serial"
    if is_demon_attack_eval:
        from training.sample_factory_common.queue_eval_vector_env import (
            DemonAttackQueueEvalSubprocVectorEnv,
            DemonAttackQueueEvalVectorEnv,
            demon_attack_queue_eval_mp_start_method,
        )

        demon_attack_vector_backend = _demon_attack_vector_backend(eval_parallel_envs=eval_parallel_envs)
        effective_parallel_envs = _demon_attack_effective_parallel_envs(
            eval_parallel_envs=eval_parallel_envs,
            eval_episodes=eval_episodes,
        )
        mp_start_method = demon_attack_queue_eval_mp_start_method()
        print(
            "env=demon_attack "
            f"eval_parallel_envs={int(eval_parallel_envs)} "
            f"vector_backend={demon_attack_vector_backend} "
            f"effective_workers={int(effective_parallel_envs)} "
            f"mp_start_method={mp_start_method}",
            flush=True,
        )
        vector_env_cls = (
            DemonAttackQueueEvalSubprocVectorEnv
            if demon_attack_vector_backend == "subproc"
            else DemonAttackQueueEvalVectorEnv
        )
        env = vector_env_cls(
            cfg=cfg,
            num_slots=effective_parallel_envs,
            env_config=env_config,
        )
    else:
        env = make_env_func_batched(
            cfg,
            env_config=env_config,
            render_mode=None,
        )
    env_info = extract_env_info(env, cfg)
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)
    load_sf_policy_state_dict(cfg, actor_critic, device)

    returns: list[float] = []
    raw_returns: list[float] | None = None
    lengths: list[int] = []
    use_explicit_queue_accounting = (
        (is_flappy_eval or is_demon_attack_eval)
        and hasattr(env, "set_auto_reset_done")
        and hasattr(env, "reset_agents")
        and int(getattr(env, "num_agents", 1)) > 0
    )
    eval_raw_reward = bool(getattr(base_cfg, "eval_raw_reward", False))
    if is_demon_attack_eval:
        raw_returns = []
    count_demon_life_loss_terminal = (
        not eval_raw_reward
        and bool(getattr(cfg, "terminal_on_life_loss", getattr(base_cfg, "terminal_on_life_loss", False)))
    )

    tracker: _QueueEvalEpisodeTracker | None = None
    episode_ids: list[int] = []
    slot_ids: list[int] = []
    if use_explicit_queue_accounting:
        tracker = _QueueEvalEpisodeTracker(
            eval_episodes=eval_episodes,
            eval_parallel_envs=int(getattr(env, "num_agents", eval_parallel_envs)),
        )
        env.set_auto_reset_done(False)
        env.set_active_agents(tracker.active_mask)
        initial_seed = _eval_episode_seed(base_cfg, 0) if tracker.active_slots > 0 else None
        obs, _ = env.reset(seed=initial_seed)
    else:
        obs, _ = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = torch.zeros([env.num_agents], dtype=torch.float32, device=device)
    episode_length = np.zeros([env.num_agents], dtype=np.int64)
    demon_episode_raw_reward = np.zeros([env.num_agents], dtype=np.float64)
    total_frames = 0
    hit_eval_step_budget = False
    timeout_episodes = 0

    def record_demon_attack_episode(
        agent_i: int,
        episode_result: DemonAttackEvalEpisodeResult | None,
        *,
        timed_out_by_eval_budget: bool = False,
        reset_plan: dict[int, int | None] | None = None,
    ) -> bool:
        nonlocal hit_eval_step_budget, timeout_episodes
        if episode_result is None or len(returns) >= eval_episodes:
            return False

        returns.append(float(episode_result["return"]))
        assert raw_returns is not None
        raw_returns.append(float(episode_result["raw_return"]))
        lengths.append(int(episode_result["length"]))
        if _demon_attack_eval_step_budget_timeout(
            episode_result,
            eval_max_steps=eval_max_steps,
            timed_out_by_eval_budget=timed_out_by_eval_budget,
        ):
            timeout_episodes += 1
            hit_eval_step_budget = True
        if tracker is not None:
            completed_episode_id, next_episode_id = tracker.complete_slot(int(agent_i))
            episode_ids.append(int(completed_episode_id))
            slot_ids.append(int(agent_i))
            if next_episode_id is not None:
                if reset_plan is None:
                    raise RuntimeError("tracked DemonAttack eval requires a reset_plan")
                reset_plan[int(agent_i)] = (
                    int(next_episode_id),
                    _eval_episode_seed(base_cfg, int(next_episode_id)),
                )
            elif reset_plan is not None:
                reset_plan.pop(int(agent_i), None)
        episode_reward[agent_i] = 0.0
        episode_length[agent_i] = 0
        demon_episode_raw_reward[agent_i] = 0.0
        return True

    while (
        (tracker is not None and tracker.completed_episodes < eval_episodes)
        or (tracker is None and len(returns) < eval_episodes and total_frames < max_total_frames)
    ):
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        policy_outputs = actor_critic(normalized_obs, rnn_states)
        actions = policy_outputs["actions"]
        if bool(getattr(base_cfg, "eval_deterministic", True)):
            actions = argmax_actions(actor_critic.action_distribution())
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)
        active_mask = np.ones([env.num_agents], dtype=np.bool_)
        if tracker is not None:
            active_mask = np.asarray(tracker.active_mask, dtype=np.bool_)
            env.set_active_agents(active_mask.tolist())
        rnn_states = policy_outputs["new_rnn_states"]
        if tracker is not None and not bool(active_mask.all()):
            inactive_mask_tensor = torch.as_tensor(~active_mask, dtype=torch.bool, device=device)
            rnn_states[inactive_mask_tensor] = 0.0
        obs, rew, terminated, truncated, infos = env.step(actions)
        if distribution_spec is not None:
            sampled_latency_frames.extend(_latency_samples_from_infos(infos, env.num_agents, active_mask))
        dones = make_dones(terminated, truncated)
        reward_tensor = rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew, dtype=torch.float32, device=device)
        reward_tensor = reward_tensor.to(device=device, dtype=torch.float32)
        if tracker is None:
            episode_reward += reward_tensor
            episode_length += 1
            total_frames += 1
        else:
            active_mask_tensor = torch.as_tensor(active_mask, dtype=torch.bool, device=device)
            episode_reward[active_mask_tensor] += reward_tensor[active_mask_tensor]
            episode_length[active_mask] += 1
            total_frames += int(active_mask.sum())

        reset_plan: dict[int, int | None] = {}
        for agent_i, done_flag in enumerate(dones.cpu().numpy()):
            if tracker is not None and not bool(active_mask[agent_i]):
                continue
            done = bool(done_flag)
            if is_demon_attack_eval:
                info = _batched_info_for_agent(infos, int(agent_i))
                raw_step_reward = _numeric_scalar(info.get("raw_reward"))
                if raw_step_reward is None:
                    raw_step_reward = float(reward_tensor[agent_i].item())
                demon_episode_raw_reward[agent_i] += float(raw_step_reward)
                if done:
                    if _bool_for_agent(terminated, int(agent_i)):
                        info.setdefault("terminated", True)
                    if _bool_for_agent(truncated, int(agent_i)):
                        info.setdefault("truncated", True)
                        info.setdefault("TimeLimit.truncated", True)
                    episode_result = _demon_attack_eval_episode_result(
                        info,
                        fallback_return=float(demon_episode_raw_reward[agent_i]),
                        fallback_length=int(episode_length[agent_i]),
                        eval_raw_reward=eval_raw_reward,
                        count_life_loss_terminal=count_demon_life_loss_terminal,
                    )
                    rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])
                    if not record_demon_attack_episode(
                        int(agent_i),
                        episode_result,
                        reset_plan=reset_plan,
                    ) and tracker is not None:
                        reset_plan[int(agent_i)] = (tracker.slot_episode_id[int(agent_i)], None)
                if (
                    eval_max_steps > 0
                    and len(returns) < eval_episodes
                    and int(episode_length[agent_i]) >= eval_max_steps
                ):
                    episode_result = _demon_attack_eval_episode_result(
                        _demon_attack_eval_budget_timeout_info(
                            accumulated_return=float(demon_episode_raw_reward[agent_i]),
                            episode_length=int(episode_length[agent_i]),
                        ),
                        fallback_return=float(demon_episode_raw_reward[agent_i]),
                        fallback_length=int(episode_length[agent_i]),
                        eval_raw_reward=eval_raw_reward,
                        count_life_loss_terminal=count_demon_life_loss_terminal,
                    )
                    rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])
                    if record_demon_attack_episode(
                        int(agent_i),
                        episode_result,
                        timed_out_by_eval_budget=True,
                        reset_plan=reset_plan,
                    ):
                        if tracker is None:
                            obs, _ = env.reset()
                continue

            forced_eval_timeout = (
                tracker is not None
                and eval_max_steps > 0
                and int(episode_length[agent_i]) >= eval_max_steps
                and not done
            )
            if forced_eval_timeout:
                done = True

            if done and len(returns) < eval_episodes:
                returns.append(float(episode_reward[agent_i].item()))
                lengths.append(int(episode_length[agent_i]))
                if forced_eval_timeout:
                    hit_eval_step_budget = True
                    timeout_episodes += 1
                if tracker is not None:
                    completed_episode_id, next_episode_id = tracker.complete_slot(int(agent_i))
                    episode_ids.append(int(completed_episode_id))
                    slot_ids.append(int(agent_i))
                    if next_episode_id is not None:
                        reset_plan[int(agent_i)] = (
                            int(next_episode_id),
                            _eval_episode_seed(base_cfg, int(next_episode_id)),
                        )
                episode_reward[agent_i] = 0.0
                episode_length[agent_i] = 0
                rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])

        if tracker is not None:
            env.set_active_agents(tracker.active_mask)
            if reset_plan:
                obs, _ = env.reset_agents(reset_plan, active_mask=tracker.active_mask)

    env.close()
    if not is_demon_attack_eval and not returns:
        returns = [float("nan")]
        lengths = [0]
    if tracker is None:
        result = _build_queue_eval_result(
            latency_value=int(latency_value),
            returns=returns,
            lengths=lengths,
            eval_episodes=eval_episodes,
            eval_max_steps=eval_max_steps,
            total_frames=total_frames,
            hit_eval_step_budget=hit_eval_step_budget,
            raw_returns=raw_returns,
            eval_raw_reward=eval_raw_reward,
            env_name=str(getattr(cfg, "env", "")),
            timeout_episodes=timeout_episodes,
        )
        if distribution_spec is not None:
            _attach_distribution_eval_metadata(result, distribution_spec, sampled_latency_frames)
        return result

    result = _build_queue_eval_result(
        latency_value=int(latency_value),
        returns=returns,
        lengths=lengths,
        eval_episodes=eval_episodes,
        eval_max_steps=eval_max_steps,
        total_frames=total_frames,
        hit_eval_step_budget=hit_eval_step_budget,
        raw_returns=raw_returns,
        eval_raw_reward=eval_raw_reward,
        env_name=str(getattr(cfg, "env", "")),
        timeout_episodes=timeout_episodes,
    )
    result["_meta"] = {
        "eval_parallel_envs": int(eval_parallel_envs),
        "effective_parallel_envs": int(len(tracker.slot_episode_id)),
        "started_episodes": int(tracker.started_episodes),
        "completed_episodes": int(tracker.completed_episodes),
        "episode_ids": [int(item) for item in episode_ids],
        "slot_ids": [int(item) for item in slot_ids],
        "vector_backend": str(demon_attack_vector_backend if is_demon_attack_eval else "sample_factory"),
    }
    if hasattr(env, "worker_pids") and callable(env.worker_pids):
        result["_meta"]["worker_pids"] = [int(pid) for pid in env.worker_pids()]
    if distribution_spec is not None:
        _attach_distribution_eval_metadata(result, distribution_spec, sampled_latency_frames)
    return result


@torch.no_grad()
def evaluate_queue_latency_distribution(base_cfg, adjust_cfg: EvalCfgAdjuster | None = None) -> dict[str, Any]:
    def distribution_adjuster(cfg: Any, original_base_cfg: Any) -> None:
        _apply_queue_eval_distribution_runtime_overrides(cfg, original_base_cfg)
        if adjust_cfg is not None:
            adjust_cfg(cfg, original_base_cfg)

    result = evaluate_queue_latency(base_cfg, latency_value=0, adjust_cfg=distribution_adjuster)
    if "latency_distribution_label" not in result:
        _attach_distribution_eval_metadata(
            result,
            latency_distribution_spec_from_flat_config(base_cfg, env_fps=float(getattr(base_cfg, "env_fps"))),
            [],
        )
    return result
