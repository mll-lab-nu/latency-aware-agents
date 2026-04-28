#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from bisect import insort_right
from typing import Any, Dict, List, Tuple

import envpool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def parse_int_list(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(initial: float, final: float, progress_remaining: float) -> float:
    return final + progress_remaining * (initial - final)


class VectorQueueLatency:
    """
    Action queue on top of EnvPool.

    Queue latency L is counted in EnvPool agent-decision steps.
    With native frame skip K, raw-frame delay = K * L.
    """

    def __init__(self, num_envs: int, latency_values: List[int], seed: int = 0):
        assert len(latency_values) > 0
        self.num_envs = int(num_envs)
        self.latency_values = [max(0, int(x)) for x in latency_values]
        self.rng = np.random.RandomState(seed)
        self.current_latency = np.zeros(self.num_envs, dtype=np.int64)
        self.current_applied_action = np.zeros(self.num_envs, dtype=np.int64)
        self.env_step_idx = np.zeros(self.num_envs, dtype=np.int64)
        self.pending: List[List[Tuple[int, int, int]]] = [[] for _ in range(self.num_envs)]
        self.seq = 0

    def reset_all(self):
        self.current_applied_action[:] = 0
        self.env_step_idx[:] = 0
        self.pending = [[] for _ in range(self.num_envs)]
        self.seq = 0
        self.current_latency[:] = [self.rng.choice(self.latency_values) for _ in range(self.num_envs)]

    def reset_done(self, done: np.ndarray):
        done = np.asarray(done, dtype=bool)
        for i, d in enumerate(done):
            if d:
                self.current_applied_action[i] = 0
                self.env_step_idx[i] = 0
                self.pending[i].clear()
                self.current_latency[i] = int(self.rng.choice(self.latency_values))

    def apply(self, chosen_actions: np.ndarray) -> np.ndarray:
        chosen_actions = np.asarray(chosen_actions, dtype=np.int64)
        applied = np.zeros_like(chosen_actions)

        for i in range(self.num_envs):
            latency = int(self.current_latency[i])
            apply_step = int(self.env_step_idx[i] + latency)
            self.seq += 1
            insort_right(self.pending[i], (apply_step, self.seq, int(chosen_actions[i])))

            while self.pending[i] and self.pending[i][0][0] <= self.env_step_idx[i]:
                _, _, matured_action = self.pending[i].pop(0)
                self.current_applied_action[i] = int(matured_action)

            applied[i] = int(self.current_applied_action[i])
            self.env_step_idx[i] += 1

        return applied


class Agent(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        assert len(obs_shape) == 3, f"Expected CHW obs shape, got {obs_shape}"
        c, h, w = obs_shape
        assert h == 84 and w == 84, f"Expected 84x84 obs, got {obs_shape}"

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.network(x.float() / 255.0))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        hidden = self.network(x.float() / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    @torch.no_grad()
    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.network(x.float() / 255.0)
        logits = self.actor(hidden)
        return torch.argmax(logits, dim=-1)


def make_envpool_env(
    env_id: str,
    num_envs: int,
    seed: int,
    native_frame_skip: int,
    clip_reward: bool,
    terminal_on_life_loss: bool,
    noop_max: int,
    screen_size: int,
    stack_frames: int,
):
    """
    EnvPool Atari env.

    This uses EnvPool's built-in Atari preprocessing instead of Python Gym wrappers.
    It should be used for fast no-latency or latency-queue-emulated Atari runs.
    """
    kwargs = dict(
        env_type="gym",
        num_envs=int(num_envs),
        seed=int(seed),
        episodic_life=bool(terminal_on_life_loss),
        reward_clip=bool(clip_reward),
        frame_skip=int(native_frame_skip),
        noop_max=int(noop_max),
        img_height=int(screen_size),
        img_width=int(screen_size),
        stack_num=int(stack_frames),
        gray_scale=True,
    )

    try:
        envs = envpool.make(env_id, **kwargs)
    except TypeError as e:
        # Fallback for older EnvPool builds with fewer exposed kwargs.
        print(f"[envpool] full kwargs failed: {e}")
        kwargs_min = dict(
            env_type="gym",
            num_envs=int(num_envs),
            seed=int(seed),
            episodic_life=bool(terminal_on_life_loss),
            reward_clip=bool(clip_reward),
        )
        if int(native_frame_skip) != 4:
            raise RuntimeError(
                "Your EnvPool build did not accept frame_skip. "
                "For --native-frame-skip 1, install/upgrade envpool with Atari frame_skip support."
            )
        envs = envpool.make(env_id, **kwargs_min)

    envs.num_envs = int(num_envs)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    return envs


def reset_env(envs):
    out = envs.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(envs, actions):
    out = envs.step(actions)
    if len(out) == 5:
        obs, rewards, terminated, truncated, infos = out
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, infos
    obs, rewards, dones, infos = out
    return obs, rewards, dones, infos


def extract_episode_stats(infos: Dict[str, Any]) -> List[Tuple[float, float]]:
    stats: List[Tuple[float, float]] = []
    if not isinstance(infos, dict):
        return stats

    if "final_info" in infos:
        for info in infos["final_info"]:
            if info is None:
                continue
            ep = info.get("episode", None)
            if ep is not None:
                r = float(np.asarray(ep["r"]).item())
                l = float(np.asarray(ep["l"]).item())
                if l > 0:
                    stats.append((r, l))

    if "episode" in infos:
        ep = infos["episode"]
        if isinstance(ep, dict) and "r" in ep and "l" in ep:
            r_arr = np.asarray(ep["r"]).reshape(-1)
            l_arr = np.asarray(ep["l"]).reshape(-1)
            mask = infos.get("_episode", None)
            if mask is not None:
                mask_arr = np.asarray(mask).reshape(-1).astype(bool)
            else:
                mask_arr = l_arr > 0
            for rr, ll, valid in zip(r_arr, l_arr, mask_arr):
                if valid and float(ll) > 0:
                    stats.append((float(rr), float(ll)))

    # EnvPool often exposes episode stats as arrays named r/l or reward/elapsed_step.
    for r_key, l_key in [("r", "l"), ("reward", "elapsed_step")]:
        if r_key in infos and l_key in infos:
            r_arr = np.asarray(infos[r_key]).reshape(-1)
            l_arr = np.asarray(infos[l_key]).reshape(-1)
            for rr, ll in zip(r_arr, l_arr):
                if float(ll) > 0:
                    stats.append((float(rr), float(ll)))

    return stats


def save_checkpoint(path: str, agent: Agent, optimizer: optim.Optimizer, args, global_step: int):
    safe_mkdir(os.path.dirname(path) or ".")
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "global_step": int(global_step),
        },
        path,
    )


@torch.no_grad()
def evaluate(agent: Agent, args, device: torch.device, latency_value: int):
    eval_clip_reward = False if args.eval_raw_reward else bool(args.clip_reward)
    eval_terminal_on_life_loss = bool(args.eval_terminal_on_life_loss)

    envs = make_envpool_env(
        env_id=args.env_id,
        num_envs=1,
        seed=args.seed + 100_000 + int(latency_value),
        native_frame_skip=args.native_frame_skip,
        clip_reward=eval_clip_reward,
        terminal_on_life_loss=eval_terminal_on_life_loss,
        noop_max=args.noop_max,
        screen_size=args.screen_size,
        stack_frames=args.stack_frames,
    )

    queue = VectorQueueLatency(num_envs=1, latency_values=[int(latency_value)], seed=args.seed + 12345)
    returns, lengths = [], []

    for ep in range(int(args.eval_episodes)):
        obs = reset_env(envs)
        queue.reset_all()
        done = np.array([False])
        ep_ret = 0.0
        ep_len = 0

        while (not bool(done[0])) and (args.eval_max_steps <= 0 or ep_len < args.eval_max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=device)
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)
            action = agent.get_deterministic_action(obs_t).cpu().numpy()
            applied_action = queue.apply(action)
            obs, reward, done, infos = step_env(envs, applied_action)
            ep_ret += float(np.asarray(reward).reshape(-1)[0])
            ep_len += 1
            queue.reset_done(done)

        returns.append(ep_ret)
        lengths.append(ep_len)

    envs.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="DemonAttack-v5")
    parser.add_argument("--train-steps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)

    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./demon_attack_data")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)

    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--learning-rate-final", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--clip-range-final", type=float, default=0.1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--native-frame-skip", type=int, default=1)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--stack-frames", type=int, default=4)

    parser.add_argument("--clip-reward", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument("--eval-terminal-on-life-loss", action="store_true")
    parser.add_argument("--eval-raw-reward", action="store_true")

    # Kept for CLI compatibility with your current bash.
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--obs-mode", type=str, default="image_only", choices=["image_only"])

    # Queue latency options. Implemented externally on top of EnvPool actions.
    parser.add_argument("--frame-skip-values", type=str, default="0")
    parser.add_argument("--latency-mode", type=str, default="episode_fixed")
    parser.add_argument("--latency-sampling-mode", type=str, default="uniform_all")
    parser.add_argument("--episodes-per-stage", type=int, default=10)
    parser.add_argument("--randomize-within-stage", action="store_true")
    parser.add_argument("--within-episode-change-interval", type=int, default=1)
    parser.add_argument("--within-episode-change-prob", type=float, default=1.0)

    parser.add_argument("--mid-eval", action="store_true")
    parser.add_argument("--mid-eval-freq", type=int, default=250_000)
    parser.add_argument("--mid-eval-episodes", type=int, default=20)
    parser.add_argument("--mid-eval-frame-skip-values", type=str, default="0")

    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-frame-skip-values", type=str, default="0,1,2,3,4,5,10,15,40")
    parser.add_argument("--eval-max-steps", type=int, default=30_000)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="demon_attack_rl_queue_latency")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")

    args = parser.parse_args()
    assert args.obs_mode == "image_only"

    # EnvPool naming is usually "DemonAttack-v5", not "ALE/DemonAttack-v5".
    if args.env_id.startswith("ALE/"):
        print(f"[envpool] converting env id {args.env_id} -> {args.env_id.split('/', 1)[1]}")
        args.env_id = args.env_id.split("/", 1)[1]

    if args.latency_mode != "episode_fixed" or args.latency_sampling_mode != "uniform_all":
        print("[warning] EnvPool version supports fixed/uniform latency values only. Current training command uses L=0, so this is fine.")

    safe_mkdir(os.path.dirname(args.save) or ".")
    safe_mkdir(args.outdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = args.wandb_run_name or f"cleanrl_envpool_demon_attack_{int(time.time())}"
    wandb_run = None
    if args.wandb:
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config=vars(args),
            save_code=True,
        )

    train_latency_values = parse_int_list(args.frame_skip_values)
    envs = make_envpool_env(
        env_id=args.env_id,
        num_envs=args.n_envs,
        seed=args.seed,
        native_frame_skip=args.native_frame_skip,
        clip_reward=bool(args.clip_reward),
        terminal_on_life_loss=bool(args.terminal_on_life_loss),
        noop_max=args.noop_max,
        screen_size=args.screen_size,
        stack_frames=args.stack_frames,
    )
    queue = VectorQueueLatency(num_envs=args.n_envs, latency_values=train_latency_values, seed=args.seed)
    queue.reset_all()

    obs_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space
    num_actions = action_space.n

    print("=== CLEANRL PPO DEMONATTACK ENVPOOL ===")
    print(f"device={device}")
    print(f"env_id={args.env_id}")
    print(f"obs_shape={obs_shape}")
    print(f"num_actions={num_actions}")
    print(f"train_steps={args.train_steps}")
    print(f"n_envs={args.n_envs}")
    print(f"n_steps={args.n_steps}")
    print(f"rollout_batch={args.n_envs * args.n_steps}")
    print(f"minibatch_size={args.batch_size}")
    print(f"n_epochs={args.n_epochs}")
    print(f"native_frame_skip/action_repeat K={args.native_frame_skip}")
    print(f"train_queue_latency_values={args.frame_skip_values}")
    print(f"train_clip_reward={args.clip_reward}")
    print(f"train_terminal_on_life_loss={args.terminal_on_life_loss}")
    print(f"eval_raw_reward={args.eval_raw_reward}")
    print(f"save={args.save}")
    print("=======================================")

    agent = Agent(obs_shape=obs_shape, num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    rollout_batch_size = args.n_envs * args.n_steps
    minibatch_size = int(args.batch_size)
    assert rollout_batch_size % minibatch_size == 0

    num_updates = args.train_steps // rollout_batch_size

    obs = torch.zeros((args.n_steps, args.n_envs) + obs_shape, dtype=torch.uint8, device=device)
    actions = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.n_steps, args.n_envs), dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()
    next_obs_np = reset_env(envs)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)
    next_done = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    b_inds = np.arange(rollout_batch_size)
    next_checkpoint_step = int(args.checkpoint_freq)
    next_mid_eval_step = int(args.mid_eval_freq)
    best_mid_eval_return = -float("inf")

    for update in range(1, num_updates + 1):
        progress_remaining = 1.0 - (update - 1.0) / max(1, num_updates)
        lr_now = linear_schedule(args.learning_rate, args.learning_rate_final, progress_remaining)
        clip_now = linear_schedule(args.clip_range, args.clip_range_final, progress_remaining)
        optimizer.param_groups[0]["lr"] = lr_now

        for step in range(args.n_steps):
            global_step += args.n_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            applied_action = queue.apply(action.cpu().numpy())

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, done_np, infos = step_env(envs, applied_action)
            queue.reset_done(done_np)

            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            for ep_ret, ep_len in extract_episode_stats(infos):
                print(f"[train-episode] global_step={global_step} episodic_return={ep_ret:.3f} episodic_length={ep_len:.1f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "charts/episodic_return": ep_ret,
                            "charts/episodic_length": ep_len,
                            "charts/SPS": int(global_step / max(1e-6, time.time() - start_time)),
                        },
                        step=global_step,
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        clipfracs = []
        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_now).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_now, 1.0 + clip_now)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_now, clip_now)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / max(1e-6, time.time() - start_time))

        print(
            f"[update] {update}/{num_updates} global_step={global_step} SPS={sps} "
            f"lr={lr_now:.6g} pg_loss={pg_loss.item():.4f} value_loss={v_loss.item():.4f} "
            f"entropy={entropy_loss.item():.4f} approx_kl={approx_kl.item():.6f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "charts/learning_rate": lr_now,
                    "charts/clip_coef": clip_now,
                    "charts/SPS": sps,
                    "losses/value_loss": float(v_loss.item()),
                    "losses/policy_loss": float(pg_loss.item()),
                    "losses/entropy": float(entropy_loss.item()),
                    "losses/old_approx_kl": float(old_approx_kl.item()),
                    "losses/approx_kl": float(approx_kl.item()),
                    "losses/clipfrac": float(np.mean(clipfracs)),
                    "losses/explained_variance": float(explained_var),
                    "global_step": int(global_step),
                },
                step=global_step,
            )

        if global_step >= next_checkpoint_step:
            base = os.path.basename(args.save).replace(".pt", "")
            ckpt_path = os.path.join(os.path.dirname(args.save) or ".", f"{base}_{global_step}_steps.pt")
            save_checkpoint(ckpt_path, agent, optimizer, args, global_step)
            print(f"[checkpoint] saved => {ckpt_path}")
            next_checkpoint_step += int(args.checkpoint_freq)

        if args.mid_eval and global_step >= next_mid_eval_step:
            mid_latencies = parse_int_list(args.mid_eval_frame_skip_values)
            if len(mid_latencies) != 1:
                raise ValueError("--mid-eval-frame-skip-values must have exactly one value.")
            old_eval_episodes = args.eval_episodes
            args.eval_episodes = args.mid_eval_episodes
            latency_value = int(mid_latencies[0])
            result = evaluate(agent, args, device, latency_value)
            args.eval_episodes = old_eval_episodes

            tag = "raw" if args.eval_raw_reward else "clipped"
            mean_ret = result["mean_return"]
            print(
                f"[mid-eval] step={global_step} latency={latency_value} "
                f"mean_return={mean_ret:.3f} std={result['std_return']:.3f} mean_len={result['mean_length']:.1f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/{tag}/latency_{latency_value}/mean_return": mean_ret,
                        f"eval/{tag}/latency_{latency_value}/std_return": result["std_return"],
                        f"eval/{tag}/latency_{latency_value}/mean_length": result["mean_length"],
                    },
                    step=global_step,
                )
            if mean_ret > best_mid_eval_return:
                best_mid_eval_return = mean_ret
                best_path = os.path.join(os.path.dirname(args.save) or ".", "best_model.pt")
                save_checkpoint(best_path, agent, optimizer, args, global_step)
                print(f"[mid-eval] new best saved => {best_path}")
            next_mid_eval_step += int(args.mid_eval_freq)

    save_checkpoint(args.save, agent, optimizer, args, global_step)
    print(f"[train] saved final model => {args.save}")
    envs.close()

    if args.eval_episodes > 0:
        tag = "raw" if args.eval_raw_reward else "clipped"
        print(f"[final-eval] episodes={args.eval_episodes} reward_tag={tag}")
        for latency_value in parse_int_list(args.eval_frame_skip_values):
            result = evaluate(agent, args, device, int(latency_value))
            print(
                f"[eval] reward={tag} latency={latency_value} "
                f"mean_return={result['mean_return']:.3f} std_return={result['std_return']:.3f} mean_len={result['mean_length']:.1f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"final_eval/{tag}/latency_{latency_value}/mean_return": result["mean_return"],
                        f"final_eval/{tag}/latency_{latency_value}/std_return": result["std_return"],
                        f"final_eval/{tag}/latency_{latency_value}/mean_length": result["mean_length"],
                    },
                    step=global_step,
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
