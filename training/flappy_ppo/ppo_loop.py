from __future__ import annotations

import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from modeling.flappy_ppo import (
    FlappyPpoAgent,
    load_flappy_ppo_checkpoint,
    observation_config_from_args,
    parse_int_list,
    save_flappy_ppo_checkpoint,
)
from training.flappy_ppo.env_batch import SyncFlappyBatch
from training.flappy_ppo.logging import (
    MetricLogger,
    build_writer,
    tensorboard_available,
)
from training.flappy_ppo.observation_pipeline import ObservationPipeline
from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path
from training.flappy_ppo.profiler import (
    UpdateProfiler,
    append_profile_csv,
    build_profile_row,
    init_profile_csv,
    print_profile_row,
)


ensure_flappy_gymnasium_on_path()

from flappy_bird_gymnasium.gpu_render import (  # noqa: E402
    FlappyRenderAssets,
    TorchFlappyRenderer,
)


def save_checkpoint(path: str, agent: FlappyPpoAgent, optimizer, args, global_step: int) -> None:
    save_flappy_ppo_checkpoint(
        path=path,
        agent=agent,
        optimizer=optimizer,
        observation_config=observation_config_from_args(args),
        global_step=global_step,
        train_config=vars(args),
    )


def train(args, wandb_module=None) -> str:
    frame_skip_values = parse_int_list(args.frame_skip_values)
    if not frame_skip_values:
        raise ValueError("--frame-skip-values must contain at least one value")
    if args.observation_render_mode == "direct_84" and args.renderer != "torch":
        raise ValueError("--observation-render-mode direct_84 requires --renderer torch")
    if args.observation_render_mode == "direct_84" and args.resize_backend != "torch":
        raise ValueError("--observation-render-mode direct_84 requires --resize-backend torch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    render_mode = "rgb_array" if args.renderer == "cpu_oracle" else None
    envs = SyncFlappyBatch(
        n_envs=args.n_envs,
        seed=args.seed,
        frame_skip_values=frame_skip_values,
        render_mode=render_mode,
    )
    states = envs.reset()

    renderer = None
    if args.renderer == "torch":
        assets = FlappyRenderAssets.for_state(states[0])
        renderer = TorchFlappyRenderer(assets, device=device)

    pipeline = ObservationPipeline(args=args, envs=envs, device=device, renderer=renderer)
    next_obs = pipeline.reset(states)
    next_done = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    agent = FlappyPpoAgent(
        in_channels=pipeline.in_channels,
        image_size=args.image_size,
        n_actions=2,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0

    if args.load:
        checkpoint = load_flappy_ppo_checkpoint(args.load, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))

    rollout_size = int(args.n_envs * args.n_steps)
    minibatch_size = min(int(args.batch_size), rollout_size)
    num_updates = max(1, int(args.train_steps) // rollout_size)
    logger = MetricLogger(build_writer(args), wandb_module=wandb_module)
    stats_window_size = max(1, int(getattr(args, "stats_window_size", 100)))
    tensorboard_enabled = bool(getattr(args, "tensorboard", False) and tensorboard_available())
    wandb_enabled = wandb_module is not None
    run_name = (
        getattr(args, "run_name", "")
        or getattr(args, "wandb_run_name", "")
        or "n/a"
    )
    profiler = UpdateProfiler(
        enabled=args.profile,
        sync_cuda=args.profile_sync_cuda,
        device=device,
    )
    if args.profile and args.profile_csv:
        init_profile_csv(args.profile_csv)

    print(
        "[train] "
        f"device={device} renderer={args.renderer} "
        f"observation_render_mode={args.observation_render_mode} "
        f"resize_backend={args.resize_backend} "
        f"obs_mode={args.obs_mode} n_envs={args.n_envs} n_steps={args.n_steps} "
        f"rollout_size={rollout_size} batch_size={minibatch_size} "
        f"updates={num_updates} frame_skip_values={args.frame_skip_values} "
        f"stats_window_size={stats_window_size} tensorboard={tensorboard_enabled} "
        f"wandb={wandb_enabled} run_name={run_name} profile={args.profile} save={args.save}",
        flush=True,
    )

    obs = torch.zeros(
        (args.n_steps, args.n_envs, pipeline.in_channels, args.image_size, args.image_size),
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
    try:
        for update in range(1, num_updates + 1):
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
                        rewards_np, dones_np, infos, states = envs.step(action_np)

                    with profiler.time("reward_done_h2d_s"):
                        rewards[step] = torch.as_tensor(
                            rewards_np, dtype=torch.float32, device=device
                        )
                        next_done = torch.as_tensor(
                            dones_np.astype(np.float32),
                            dtype=torch.float32,
                            device=device,
                        )

                    with profiler.time("render_obs_s"):
                        next_obs = pipeline.step(states, dones_np, profiler=profiler)

                    now = time.time()
                    if args.log_interval_seconds > 0 and (
                        now - last_progress_log >= args.log_interval_seconds
                    ):
                        elapsed = max(now - start_time, 1e-9)
                        print(
                            "[rollout] "
                            f"update={update}/{num_updates} "
                            f"step={step + 1}/{args.n_steps} "
                            f"global_step={global_step} "
                            f"SPS={int(global_step / elapsed)} "
                            f"episodes={episode_count} "
                            f"last_ep_rew={last_ep_ret:.3f} last_ep_len={last_ep_len:.1f} "
                            f"ep_rew_mean={ep_rew_mean:.3f} ep_len_mean={ep_len_mean:.1f} ",
                            flush=True,
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
                        logger.add_scalar(
                            "rollout/ep_rew_mean",
                            ep_rew_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "rollout/ep_len_mean",
                            ep_len_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "rollout/episodes",
                            episode_count,
                            global_step,
                        )

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
                b_obs = obs.reshape(
                    (-1, pipeline.in_channels, args.image_size, args.image_size)
                )
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
                                v_loss = 0.5 * (
                                    (newvalue - b_returns[mb_inds]) ** 2
                                ).mean()

                            entropy_loss = entropy.mean()
                            loss = (
                                pg_loss
                                - args.ent_coef * entropy_loss
                                + v_loss * args.vf_coef
                            )

                        with profiler.time("backward_s"):
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                agent.parameters(), args.max_grad_norm
                            )

                        with profiler.time("optimizer_s"):
                            optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = int(global_step / (time.time() - start_time))
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
            print(
                "[update] "
                f"global_step={global_step} update={update}/{num_updates} SPS={sps} "
                f"episodes={episode_count} ep_rew_mean={ep_rew_mean:.3f} "
                f"ep_len_mean={ep_len_mean:.1f} loss={loss.item():.6f} "
                f"value_loss={v_loss.item():.6f} policy_loss={pg_loss.item():.6f} "
                f"entropy={entropy_loss.item():.6f} approx_kl={approx_kl.item():.6f} "
                f"clipfrac={clipfrac_mean:.3f} explained_var={explained_var:.3f} "
                f"lr={current_lr:.6g} clip_coef={clip_coef_now:.6g} ",
                flush=True,
            )

            if args.profile:
                profiler.synchronize()
                total_s = time.perf_counter() - update_start
                included = update > args.profile_warmup_updates
                row = build_profile_row(
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
                print_profile_row(row)
                if included and args.profile_csv:
                    append_profile_csv(args.profile_csv, row)

            if args.checkpoint_freq > 0 and global_step % args.checkpoint_freq < args.n_envs:
                save_checkpoint(args.save, agent, optimizer, args, global_step)
    finally:
        envs.close()
        logger.close()

    save_checkpoint(args.save, agent, optimizer, args, global_step)
    return args.save
