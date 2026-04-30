from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch

from training.sample_factory_common.common import maybe_set_cli_override


EvalCfgAdjuster = Callable[[Any, Any], None]


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
    wandb_run.log(payload)
    wandb_run.summary[f"queue_eval/latency_{latency}/mean_return"] = float(result["mean_return"])
    wandb_run.summary[f"queue_eval/latency_{latency}/std_return"] = float(result["std_return"])
    wandb_run.summary[f"queue_eval/latency_{latency}/mean_length"] = float(result["mean_length"])


def finish_queue_eval_wandb(wandb_run, owned_run: bool) -> None:
    if wandb_run is None or not owned_run:
        return
    wandb_run.finish()


def load_sf_policy_state_dict(cfg, actor_critic, device: torch.device) -> None:
    from sample_factory.algo.learning.learner import Learner

    policy_id = getattr(cfg, "policy_index", 0)
    load_checkpoint_kind = getattr(cfg, "load_checkpoint_kind", "latest")
    name_prefix = dict(latest="checkpoint", best="best")[load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if not checkpoint_dict:
        raise RuntimeError("Could not load Sample Factory checkpoint")
    actor_critic.load_state_dict(checkpoint_dict["model"])


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
    lengths: list[int] = []
    eval_episodes = int(getattr(base_cfg, "eval_episodes", 20))
    eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 30000))
    max_total_frames = int(1e9 if eval_max_steps <= 0 else eval_max_steps * max(1, eval_episodes))

    obs, _ = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = torch.zeros([env.num_agents], dtype=torch.float32, device=device)
    episode_length = np.zeros([env.num_agents], dtype=np.int64)
    total_frames = 0

    while len(returns) < eval_episodes and total_frames < max_total_frames:
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
        actions = policy_outputs["actions"]
        if bool(getattr(base_cfg, "eval_deterministic", True)):
            actions = argmax_actions(actor_critic.action_distribution())
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)
        rnn_states = policy_outputs["new_rnn_states"]
        obs, rew, terminated, truncated, _ = env.step(actions)
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        dones = make_dones(terminated, truncated)
        reward_tensor = rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew, dtype=torch.float32, device=device)
        episode_reward += reward_tensor.float()
        episode_length += 1
        total_frames += 1

        for agent_i, done_flag in enumerate(dones.cpu().numpy()):
            if done_flag and len(returns) < eval_episodes:
                returns.append(float(episode_reward[agent_i].item()))
                lengths.append(int(episode_length[agent_i]))
                episode_reward[agent_i] = 0.0
                episode_length[agent_i] = 0
                rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])

    env.close()
    if not returns:
        returns = [float("nan")]
        lengths = [0]
    return {
        "latency": int(latency_value),
        "mean_return": float(np.nanmean(returns)),
        "std_return": float(np.nanstd(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
    }
