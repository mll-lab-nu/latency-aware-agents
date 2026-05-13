from __future__ import annotations

import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from modeling.demon_attack_ppo import (
    DemonAttackPpoAgent,
    load_demon_attack_ppo_checkpoint,
    observation_config_from_args,
    parse_int_list,
    save_demon_attack_ppo_checkpoint,
)
from training.demon_attack_ppo.env_batch import SyncDemonAttackBatch
from training.demon_attack_ppo.logging import (
    JsonlEventLogger,
    MetricLogger,
    emit_event,
    format_checkpoint_save,
    format_rollout_progress,
    format_train_end,
    format_train_start,
    format_update,
)
from training.demon_attack_ppo.observation_pipeline import ObservationPipeline
from training.demon_attack_ppo.profiler import (
    UpdateProfiler,
    append_profile_csv,
    build_profile_row,
    init_profile_csv,
    print_profile_row,
)


def save_checkpoint(path: str, agent: DemonAttackPpoAgent, optimizer, args, global_step: int) -> None:
    save_demon_attack_ppo_checkpoint(
        path=path,
        agent=agent,
        optimizer=optimizer,
        observation_config=observation_config_from_args(args),
        global_step=global_step,
        train_config=vars(args),
    )


def _checkpoint_due(*, global_step: int, next_checkpoint_step: int) -> bool:
    return int(global_step) >= int(next_checkpoint_step)


def _advance_checkpoint_step(
    *,
    next_checkpoint_step: int,
    global_step: int,
    checkpoint_freq: int,
) -> int:
    if checkpoint_freq <= 0:
        return int(next_checkpoint_step)

    checkpoint_step = int(next_checkpoint_step)
    while checkpoint_step <= int(global_step):
        checkpoint_step += int(checkpoint_freq)
    return checkpoint_step


def train(args, wandb_module) -> str:
    frame_skip_values = parse_int_list(args.frame_skip_values)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    envs = SyncDemonAttackBatch(
        n_envs=args.n_envs,
        seed=args.seed,
        frame_skip_values=frame_skip_values,
        device=device,
        env_id=args.env_id,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        action_repeat=getattr(args, "action_repeat", 1),
        terminal_on_life_loss=args.terminal_on_life_loss,
        clip_reward=args.clip_reward,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=args.full_action_space,
    )

    raw_frames = envs.reset()
    pipeline = ObservationPipeline(args=args, n_envs=args.n_envs, device=device)
    image_size = int(pipeline.observation_config.image_size)
    next_obs = pipeline.reset(raw_frames, envs.frame_skips())
    next_done = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    agent = DemonAttackPpoAgent(
        in_channels=pipeline.in_channels,
        image_size=image_size,
        n_actions=envs.n_actions,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0

    if args.load:
        checkpoint = load_demon_attack_ppo_checkpoint(args.load, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
    next_checkpoint_step = int(args.checkpoint_freq)
    if args.checkpoint_freq > 0:
        next_checkpoint_step = _advance_checkpoint_step(
            next_checkpoint_step=next_checkpoint_step,
            global_step=global_step,
            checkpoint_freq=int(args.checkpoint_freq),
        )

    rollout_size = int(args.n_envs * args.n_steps)
    minibatch_size = min(int(args.batch_size), rollout_size)
    num_updates = max(1, int(args.train_steps) // rollout_size)
    logger = MetricLogger(wandb_module)
    stats_window_size = max(1, int(getattr(args, "stats_window_size", 100)))
    wandb_run_name = getattr(args, "wandb_run_name", "") or "n/a"
    run_id = getattr(args, "run_id", "")
    if not run_id:
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        args.run_id = run_id
    metrics_jsonl = getattr(args, "metrics_jsonl", "")
    jsonl_logger = JsonlEventLogger(metrics_jsonl, run_id)
    profiler = UpdateProfiler(
        enabled=args.profile,
        sync_cuda=args.profile_sync_cuda,
        device=device,
    )
    if args.profile and args.profile_csv:
        init_profile_csv(args.profile_csv)

    emit_event(
        jsonl_logger,
        {
            "event": "train_start",
            "run_id": run_id,
            "global_step": int(global_step),
            "update": None,
            "device": str(device),
            "env_id": args.env_id,
            "obs_mode": args.obs_mode,
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "rollout_size": int(rollout_size),
            "batch_size": int(minibatch_size),
            "num_updates": int(num_updates),
            "frame_skip_values": args.frame_skip_values,
            "action_repeat": int(getattr(args, "action_repeat", 1)),
            "terminal_on_life_loss": bool(args.terminal_on_life_loss),
            "checkpoint_freq": int(args.checkpoint_freq),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "stats_window_size": int(stats_window_size),
            "save_path": args.save,
            "wandb_project": getattr(args, "wandb_project", ""),
            "wandb_run_name": wandb_run_name,
            "seed": getattr(args, "seed", None),
            "image_size": image_size,
            "stack_frames": getattr(args, "stack_frames", None),
            "profile": bool(getattr(args, "profile", False)),
            "load_path": getattr(args, "load", ""),
            "observation": "atari_frame_v1",
            "preprocess": "torch_stack_v1",
        },
        format_train_start,
    )

    obs = torch.zeros(
        (args.n_steps, args.n_envs, pipeline.in_channels, image_size, image_size),
        dtype=torch.uint8,
        device=device,
    )
    actions = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.n_steps, args.n_envs), device=device)
    rewards = torch.zeros((args.n_steps, args.n_envs), device=device)
    dones = torch.zeros((args.n_steps, args.n_envs), device=device)
    values = torch.zeros((args.n_steps, args.n_envs), device=device)

    recent_returns = deque(maxlen=stats_window_size)
    recent_lengths = deque(maxlen=stats_window_size)
    episode_count = 0
    last_ep_ret = float("nan")
    last_ep_len = float("nan")
    ep_rew_mean = float("nan")
    ep_len_mean = float("nan")
    start_time = time.time()
    last_progress_log = start_time
    last_update = None
    train_error = None
    try:
        for update in range(1, num_updates + 1):
            last_update = update
            profiler.reset()
            profiler.synchronize()
            update_start = time.perf_counter()
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / float(num_updates)
                lr_now = args.learning_rate_final + frac * (
                    args.learning_rate - args.learning_rate_final
                )
                optimizer.param_groups[0]["lr"] = lr_now
                clip_coef_now = args.clip_coef_final + frac * (
                    args.clip_coef - args.clip_coef_final
                )
            else:
                clip_coef_now = args.clip_coef

            with profiler.time("rollout_total_s"):
                for step in range(args.n_steps):
                    global_step += args.n_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    with profiler.time("action_infer_s"):
                        with torch.no_grad():
                            action, logprob, _, value = agent.get_action_and_value(next_obs)
                            values[step] = value.flatten()
                        actions[step] = action
                        logprobs[step] = logprob

                    with profiler.time("action_cpu_copy_s"):
                        action_np = action.detach().cpu().numpy()

                    with profiler.time("env_step_s"):
                        rewards_np, dones_np, infos, raw_frames = envs.step(
                            action_np,
                            profiler=profiler,
                        )

                    with profiler.time("reward_done_h2d_s"):
                        rewards[step] = torch.as_tensor(
                            rewards_np, dtype=torch.float32, device=device
                        )
                        next_done = torch.as_tensor(
                            dones_np.astype(np.float32),
                            dtype=torch.float32,
                            device=device,
                        )

                    with profiler.time("preprocess_obs_s"):
                        next_obs = pipeline.step(
                            raw_frames,
                            dones_np,
                            envs.frame_skips(),
                            profiler=profiler,
                        )

                    now = time.time()
                    if args.log_interval_seconds > 0 and (
                        now - last_progress_log >= args.log_interval_seconds
                    ):
                        elapsed = max(now - start_time, 1e-9)
                        emit_event(
                            jsonl_logger,
                            {
                                "event": "rollout_progress",
                                "run_id": run_id,
                                "global_step": int(global_step),
                                "update": int(update),
                                "num_updates": int(num_updates),
                                "rollout_step": int(step + 1),
                                "n_steps": int(args.n_steps),
                                "sps": int(global_step / elapsed),
                                "episode_count": int(episode_count),
                                "last_ep_rew": last_ep_ret,
                                "last_ep_len": last_ep_len,
                                "ep_rew_mean": ep_rew_mean,
                                "ep_len_mean": ep_len_mean,
                                "elapsed_seconds": elapsed,
                            },
                            format_rollout_progress,
                        )
                        last_progress_log = now

                    for info in infos:
                        if "episodic_return" not in info:
                            continue
                        ep_ret = float(info["episodic_return"])
                        ep_len = int(info["episodic_length"])
                        episode_count += 1
                        last_ep_ret = ep_ret
                        last_ep_len = float(ep_len)
                        recent_returns.append(ep_ret)
                        recent_lengths.append(ep_len)
                        ep_rew_mean = float(np.mean(recent_returns))
                        ep_len_mean = float(np.mean(recent_lengths))
                        logger.add_scalar("charts/episodic_return", ep_ret, global_step)
                        logger.add_scalar("charts/episodic_length", ep_len, global_step)
                        logger.add_scalar(
                            "charts/avg_episodic_return",
                            ep_rew_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "charts/avg_episodic_length",
                            ep_len_mean,
                            global_step,
                        )
                        logger.add_scalar("rollout/ep_rew_mean", ep_rew_mean, global_step)
                        logger.add_scalar("rollout/ep_len_mean", ep_len_mean, global_step)
                        logger.add_scalar("rollout/episodes", episode_count, global_step)

            with profiler.time("gae_s"):
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards, device=device)
                    lastgaelam = 0
                    for t in reversed(range(args.n_steps)):
                        if t == args.n_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + args.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + args.gamma
                            * args.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values

            with profiler.time("flatten_s"):
                b_obs = obs.reshape((-1, pipeline.in_channels, image_size, image_size))
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

            b_inds = np.arange(rollout_size)
            clipfracs = []
            with profiler.time("ppo_update_s"):
                for _ in range(args.n_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, rollout_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        profiler.add_count("num_minibatches")

                        with profiler.time("minibatch_forward_loss_s"):
                            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                                b_obs[mb_inds], b_actions.long()[mb_inds]
                            )
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs.append(
                                    ((ratio - 1.0).abs() > clip_coef_now)
                                    .float()
                                    .mean()
                                    .item()
                                )

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (
                                    mb_advantages - mb_advantages.mean()
                                ) / (mb_advantages.std() + 1e-8)

                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(
                                ratio, 1 - clip_coef_now, 1 + clip_coef_now
                            )
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            newvalue = newvalue.view(-1)
                            if args.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + torch.clamp(
                                    newvalue - b_values[mb_inds],
                                    -clip_coef_now,
                                    clip_coef_now,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss = 0.5 * torch.max(
                                    v_loss_unclipped, v_loss_clipped
                                ).mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = (
                                pg_loss
                                - args.ent_coef * entropy_loss
                                + v_loss * args.vf_coef
                            )

                        with profiler.time("backward_s"):
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                        with profiler.time("optimizer_s"):
                            optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            elapsed_seconds = max(time.time() - start_time, 1e-9)
            sps = int(global_step / elapsed_seconds)
            current_lr = optimizer.param_groups[0]["lr"]
            clipfrac_mean = float(np.mean(clipfracs)) if clipfracs else float("nan")

            logger.add_scalar("charts/learning_rate", current_lr, global_step)
            logger.add_scalar("charts/clip_coef", clip_coef_now, global_step)
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", clipfrac_mean, global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)

            profile_row = None
            if args.profile:
                profiler.synchronize()
                total_s = time.perf_counter() - update_start
                included = update > args.profile_warmup_updates
                profile_row = build_profile_row(
                    args=args,
                    device=device,
                    profiler=profiler,
                    update=update,
                    global_step=global_step,
                    total_s=total_s,
                    rollout_size=rollout_size,
                    minibatch_size=minibatch_size,
                    included=included,
                    end_to_end_sps=sps,
                )
            update_event = {
                "event": "update",
                "run_id": run_id,
                "global_step": int(global_step),
                "update": int(update),
                "num_updates": int(num_updates),
                "sps": int(sps),
                "episode_count": int(episode_count),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "loss": loss.item(),
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": clipfrac_mean,
                "explained_var": explained_var,
                "learning_rate": current_lr,
                "clip_coef": clip_coef_now,
                "elapsed_seconds": elapsed_seconds,
            }
            if profile_row is not None:
                update_event.update(profile_row)
            emit_event(jsonl_logger, update_event, format_update)

            if profile_row is not None:
                print_profile_row(profile_row)
                if included and args.profile_csv:
                    append_profile_csv(args.profile_csv, profile_row)

            if args.checkpoint_freq > 0 and _checkpoint_due(
                global_step=global_step,
                next_checkpoint_step=next_checkpoint_step,
            ):
                save_checkpoint(args.save, agent, optimizer, args, global_step)
                emit_event(
                    jsonl_logger,
                    {
                        "event": "checkpoint_save",
                        "run_id": run_id,
                        "global_step": int(global_step),
                        "update": int(update),
                        "path": args.save,
                        "reason": "interval",
                        "episode_count": int(episode_count),
                    },
                    format_checkpoint_save,
                )
                next_checkpoint_step = _advance_checkpoint_step(
                    next_checkpoint_step=next_checkpoint_step,
                    global_step=global_step,
                    checkpoint_freq=int(args.checkpoint_freq),
                )
    except Exception as exc:
        train_error = exc
        emit_event(
            jsonl_logger,
            {
                "event": "train_end",
                "run_id": run_id,
                "global_step": int(global_step),
                "update": last_update,
                "num_updates": int(num_updates),
                "episode_count": int(episode_count),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "save_path": args.save,
                "status": "error",
                "elapsed_seconds": max(time.time() - start_time, 0.0),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            format_train_end,
        )
        raise
    finally:
        envs.close()
        logger.close()
        if train_error is not None:
            jsonl_logger.close()

    try:
        save_checkpoint(args.save, agent, optimizer, args, global_step)
        emit_event(
            jsonl_logger,
            {
                "event": "checkpoint_save",
                "run_id": run_id,
                "global_step": int(global_step),
                "update": last_update,
                "path": args.save,
                "reason": "final",
                "episode_count": int(episode_count),
            },
            format_checkpoint_save,
        )
        emit_event(
            jsonl_logger,
            {
                "event": "train_end",
                "run_id": run_id,
                "global_step": int(global_step),
                "update": last_update,
                "num_updates": int(num_updates),
                "episode_count": int(episode_count),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "save_path": args.save,
                "status": "success",
                "elapsed_seconds": max(time.time() - start_time, 0.0),
            },
            format_train_end,
        )
        return args.save
    except Exception as exc:
        emit_event(
            jsonl_logger,
            {
                "event": "train_end",
                "run_id": run_id,
                "global_step": int(global_step),
                "update": last_update,
                "num_updates": int(num_updates),
                "episode_count": int(episode_count),
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "save_path": args.save,
                "status": "error",
                "elapsed_seconds": max(time.time() - start_time, 0.0),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            format_train_end,
        )
        raise
    finally:
        jsonl_logger.close()
