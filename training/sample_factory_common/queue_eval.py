from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from training.atari_episode_stats import (
    TRUE_EPISODE_END_INFO_KEY,
    _batched_info_for_agent,
    _flatten_numeric_episode_extra_stats,
    is_true_atari_episode_end,
)
from training.sample_factory_common.checkpoint_io import load_sample_factory_checkpoint_from_paths
from training.sample_factory_common.common import maybe_set_cli_override


EvalCfgAdjuster = Callable[[Any, Any], None]
DemonAttackEvalEpisodeResult = dict[str, float | int | bool]

_DEMON_ATTACK_SF_ENV = "latency_demon_attack"


def _apply_eval_only_max_episode_steps(cfg: Any, base_cfg: Any) -> None:
    if _is_demon_attack_queue_eval_cfg(cfg) or _is_demon_attack_queue_eval_cfg(base_cfg):
        return

    current_max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
    if current_max_episode_steps > 0:
        return

    eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 0) or 0)
    if eval_max_steps > 0:
        maybe_set_cli_override(cfg, "max_episode_steps", eval_max_steps)


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
) -> DemonAttackEvalEpisodeResult | None:
    if not _demon_attack_true_eval_episode_end(info):
        return None

    raw_return = _metric_from_info(
        info,
        ("episodic_raw_return",),
        ("episode/raw_return", "raw_return"),
    )
    clipped_return = _metric_from_info(
        info,
        ("episodic_clipped_return",),
        ("episode/clipped_return", "clipped_return"),
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
    if clipped_return is None and not eval_raw_reward:
        clipped_return = float(fallback_return)

    selected_return = raw_return if eval_raw_reward else clipped_return
    if selected_return is None:
        selected_return = episodic_return
    if selected_return is None:
        selected_return = float(fallback_return)

    if raw_return is None:
        raw_return = float(selected_return)
    if clipped_return is None:
        clipped_return = float(selected_return)

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
        "clipped_return": float(clipped_return),
        "length": int(length if length is not None else fallback_length),
    }
    if timeout:
        result["timeout"] = True
    return result


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
    clipped_returns: list[float] | None = None,
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
    if clipped_returns is not None:
        result["clipped_returns"] = [float(item) for item in clipped_returns]
        result["mean_clipped_return"] = _mean_or_nan(clipped_returns)
        result["std_clipped_return"] = _std_or_nan(clipped_returns)
    if eval_raw_reward:
        result["eval_return_source"] = "raw_return"
    elif clipped_returns is not None:
        result["eval_return_source"] = "clipped_return"

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
        ("mean_clipped_return", "queue_eval/mean_clipped_return"),
        ("std_clipped_return", "queue_eval/std_clipped_return"),
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
        ("mean_clipped_return", "mean_clipped_return"),
        ("std_clipped_return", "std_clipped_return"),
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


def write_queue_eval_result(base_cfg, result: dict[str, Any]) -> None:
    output_dir = queue_eval_output_dir(base_cfg)
    if output_dir is None:
        return

    experiment_name = str(getattr(base_cfg, "experiment", "sample_factory_eval"))
    seed = int(getattr(base_cfg, "seed", 0) or 0)
    payload = {
        "experiment_name": experiment_name,
        "seed": seed,
        "checkpoint_kind": str(getattr(base_cfg, "load_checkpoint_kind", "latest")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latency": int(result["latency"]),
        "mean_return": float(result["mean_return"]),
        "std_return": float(result["std_return"]),
        "mean_length": float(result["mean_length"]),
        "returns": [float(item) for item in result.get("returns", [])],
        "lengths": [int(item) for item in result.get("lengths", [])],
    }
    for key in (
        "raw_returns",
        "clipped_returns",
    ):
        if key in result:
            payload[key] = [float(item) for item in result.get(key, [])]
    for key in (
        "mean_raw_return",
        "mean_clipped_return",
        "std_raw_return",
        "std_clipped_return",
        "completed_episodes",
        "timeout_episodes",
        "requested_episodes",
        "eval_max_steps",
        "total_frames",
    ):
        if key in result:
            value = result[key]
            if key.endswith("episodes") or key in {"eval_max_steps", "total_frames"}:
                payload[key] = int(value)
            else:
                payload[key] = float(value)
    for key in ("hit_eval_step_budget",):
        if key in result:
            payload[key] = bool(result[key])
    for key in ("eval_return_source", "warning"):
        if key in result:
            payload[key] = str(result[key])

    latency = int(result["latency"])
    latency_file = output_dir / f"queue_eval_latency_{latency}.json"
    with latency_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
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
            "Sample Factory eval checkpoint/model shape mismatch. Check env.obs_resize/env.obs_mode "
            "and use the same observation config for train and eval."
        ) from exc


def prepare_queue_eval_cfg(base_cfg, latency_value: int, adjust_cfg: EvalCfgAdjuster | None = None):
    cfg = copy.deepcopy(base_cfg)
    maybe_set_cli_override(cfg, "queue_latency_values", str(int(latency_value)))
    maybe_set_cli_override(cfg, "latency_mode", "episode_fixed")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    maybe_set_cli_override(cfg, "episodes_per_stage", int(10**9))
    maybe_set_cli_override(cfg, "randomize_within_stage", False)
    maybe_set_cli_override(cfg, "num_envs", 1)
    maybe_set_cli_override(cfg, "num_envs_per_worker", 1)
    maybe_set_cli_override(cfg, "no_render", True)
    maybe_set_cli_override(cfg, "save_video", False)
    maybe_set_cli_override(cfg, "fps", 0)
    maybe_set_cli_override(cfg, "eval_deterministic", bool(getattr(base_cfg, "eval_deterministic", True)))
    _apply_eval_only_max_episode_steps(cfg, base_cfg)
    if adjust_cfg is not None:
        adjust_cfg(cfg, base_cfg)
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
    maybe_set_cli_override(cfg, "queue_latency_values", str(int(latency_value)))
    maybe_set_cli_override(cfg, "latency_mode", "episode_fixed")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    _apply_eval_only_max_episode_steps(cfg, base_cfg)
    if adjust_cfg is not None:
        adjust_cfg(cfg, base_cfg)

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
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
    clipped_returns: list[float] | None = None
    lengths: list[int] = []
    eval_episodes = int(getattr(base_cfg, "eval_episodes", 20))
    eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 30000))
    max_total_frames = int(1e9 if eval_max_steps <= 0 else eval_max_steps * max(1, eval_episodes))
    is_demon_attack_eval = _is_demon_attack_queue_eval_cfg(cfg) or _is_demon_attack_queue_eval_cfg(base_cfg)
    eval_raw_reward = bool(getattr(base_cfg, "eval_raw_reward", False))
    if is_demon_attack_eval:
        raw_returns = []
        clipped_returns = []

    obs, _ = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = torch.zeros([env.num_agents], dtype=torch.float32, device=device)
    episode_length = np.zeros([env.num_agents], dtype=np.int64)
    total_frames = 0
    hit_eval_step_budget = False
    timeout_episodes = 0

    def record_demon_attack_episode(
        agent_i: int,
        episode_result: DemonAttackEvalEpisodeResult | None,
        *,
        timed_out_by_eval_budget: bool = False,
    ) -> bool:
        nonlocal hit_eval_step_budget, timeout_episodes
        if episode_result is None or len(returns) >= eval_episodes:
            return False

        returns.append(float(episode_result["return"]))
        assert raw_returns is not None
        assert clipped_returns is not None
        raw_returns.append(float(episode_result["raw_return"]))
        clipped_returns.append(float(episode_result["clipped_return"]))
        lengths.append(int(episode_result["length"]))
        if _demon_attack_eval_step_budget_timeout(
            episode_result,
            eval_max_steps=eval_max_steps,
            timed_out_by_eval_budget=timed_out_by_eval_budget,
        ):
            timeout_episodes += 1
            hit_eval_step_budget = True
        episode_reward[agent_i] = 0.0
        episode_length[agent_i] = 0
        return True

    while len(returns) < eval_episodes and total_frames < max_total_frames:
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        policy_outputs = actor_critic(normalized_obs, rnn_states)
        actions = policy_outputs["actions"]
        if bool(getattr(base_cfg, "eval_deterministic", True)):
            actions = argmax_actions(actor_critic.action_distribution())
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)
        rnn_states = policy_outputs["new_rnn_states"]
        obs, rew, terminated, truncated, infos = env.step(actions)
        dones = make_dones(terminated, truncated)
        reward_tensor = rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew, dtype=torch.float32, device=device)
        episode_reward += reward_tensor.to(device=device, dtype=torch.float32)
        episode_length += 1
        total_frames += 1

        for agent_i, done_flag in enumerate(dones.cpu().numpy()):
            done = bool(done_flag)
            if is_demon_attack_eval:
                if done:
                    info = _batched_info_for_agent(infos, int(agent_i))
                    if _bool_for_agent(terminated, int(agent_i)):
                        info.setdefault("terminated", True)
                    if _bool_for_agent(truncated, int(agent_i)):
                        info.setdefault("truncated", True)
                        info.setdefault("TimeLimit.truncated", True)
                    episode_result = _demon_attack_eval_episode_result(
                        info,
                        fallback_return=float(episode_reward[agent_i].item()),
                        fallback_length=int(episode_length[agent_i]),
                        eval_raw_reward=eval_raw_reward,
                    )
                    rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])
                    record_demon_attack_episode(int(agent_i), episode_result)
                if (
                    eval_max_steps > 0
                    and len(returns) < eval_episodes
                    and int(episode_length[agent_i]) >= eval_max_steps
                ):
                    episode_result = _demon_attack_eval_episode_result(
                        {"TimeLimit.truncated": True, "truncated": True},
                        fallback_return=float(episode_reward[agent_i].item()),
                        fallback_length=int(episode_length[agent_i]),
                        eval_raw_reward=eval_raw_reward,
                    )
                    rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])
                    if record_demon_attack_episode(
                        int(agent_i),
                        episode_result,
                        timed_out_by_eval_budget=True,
                    ):
                        obs, _ = env.reset()
                continue

            if done and len(returns) < eval_episodes:
                returns.append(float(episode_reward[agent_i].item()))
                lengths.append(int(episode_length[agent_i]))
                episode_reward[agent_i] = 0.0
                episode_length[agent_i] = 0
                rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])

    env.close()
    if not is_demon_attack_eval and not returns:
        returns = [float("nan")]
        lengths = [0]
    return _build_queue_eval_result(
        latency_value=int(latency_value),
        returns=returns,
        lengths=lengths,
        eval_episodes=eval_episodes,
        eval_max_steps=eval_max_steps,
        total_frames=total_frames,
        hit_eval_step_budget=hit_eval_step_budget,
        raw_returns=raw_returns,
        clipped_returns=clipped_returns,
        eval_raw_reward=eval_raw_reward,
        env_name=str(getattr(cfg, "env", "")),
        timeout_episodes=timeout_episodes,
    )
