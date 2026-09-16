"""Complete-episode Monte Carlo PPO for native fixed-delay action chunks.

The CPU vector workers own simulation and explicit episode quotas. The parent
owns batched inference, the frozen behavior policy, and both optimizers.
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import deque
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import wandb
from torch import nn
from torch.distributions import Independent, Normal

from sample_factory.algo.fast_td3.models import Actor, EmpiricalNormalization
from training.gymnasium_sf.integration import build_train_cfg_from_config, make_gymnasium_sf_env


OBS_KEYS = ("obs", "scheduled_chunk", "scheduled_index")


class PPOPolicy(nn.Module):
    """BC-compatible chunk actor with a frozen physical observation normalizer."""

    def __init__(self, obs_space, action_space, settings):
        super().__init__()
        self.physical_dim = obs_space["obs"].shape[0]
        self.input_dim = sum(np.prod(obs_space[key].shape).item() for key in OBS_KEYS)
        self.actor = Actor(self.input_dim, action_space.shape[0])
        self.empirical_obs_normalizer = EmpiricalNormalization(self.physical_dim, torch.device("cpu"))
        action_dim = action_space.shape[0] // settings["chunk_horizon"]
        self.active = slice(settings["active_heads"][0] * action_dim,
                            settings["active_heads"][1] * action_dim)
        self.std = settings["action_std"]

    def normalize(self, observations):
        device = next(self.parameters()).device
        parts = [torch.as_tensor(observations[key], device=device, dtype=torch.float32).flatten(1) for key in OBS_KEYS]
        parts[0] = self.empirical_obs_normalizer(parts[0], update_stats=False)
        return torch.cat(parts, dim=1)

    def logits(self, obs):
        return self.actor.net[:-1](obs)

    def distribution(self, means):
        # The fixed tanh Jacobian cancels in PPO ratios; retain the sampled latent.
        return Independent(Normal(means, self.std), 1)


class EpisodeWorker(gym.Wrapper):
    """Advance one admitted chunk at a time and idle after the assigned seeds."""

    def __init__(self, cfg, worker, workers, settings):
        torch.set_num_threads(1)
        cfg = copy.copy(cfg)
        cfg.episode_metrics_path = ""
        env = make_gymnasium_sf_env(cfg.env, cfg, SimpleNamespace(worker_index=worker, vector_index=0, env_id=worker))
        super().__init__(env)
        self.worker, self.workers = worker, workers
        self.stride = settings["decision_frames"]
        self.horizon = settings["episode_frames"]
        self.seeds = deque()
        self.frames, self.total, self.seed = 0, 0.0, 0
        self.active = False
        self.last_obs = {key: np.zeros(env.observation_space[key].shape, dtype=np.float32) for key in OBS_KEYS}
        self.observation_space = gym.spaces.Dict({
            **{key: env.observation_space[key] for key in OBS_KEYS},
            "active": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "remaining": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "seed": gym.spaces.Box(0, np.iinfo(np.int64).max, (1,), dtype=np.int64),
        })

    def assign_seeds(self, seeds):
        self.seeds = deque(seeds[self.worker::self.workers])

    def observation(self):
        return {**self.last_obs, "active": np.array([self.active], dtype=np.float32),
                "remaining": np.array([1 - self.frames / self.horizon], dtype=np.float32),
                "seed": np.array([self.seed], dtype=np.int64)}

    def reset(self, *, seed=None, options=None):
        self.frames, self.total = 0, 0.0
        self.active = bool(self.seeds)
        if self.active:
            self.seed = self.seeds.popleft()
            obs, _ = self.env.reset(seed=self.seed)
            self.last_obs = {key: obs[key] for key in OBS_KEYS}
        return self.observation(), {}

    def step(self, action):
        if not self.active:
            return self.observation(), 0.0, False, False, {"frames": 0, "simulation_seconds": 0.0}
        started = time.perf_counter()
        for frames in range(1, self.stride + 1):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            self.total += reward
            self.frames += 1
            if terminated or truncated:
                break
        self.last_obs = {key: obs[key] for key in OBS_KEYS}
        info = {"frames": frames, "simulation_seconds": time.perf_counter() - started}
        if terminated or truncated:
            info["episode"] = {"seed": self.seed, "length": self.frames, "return": self.total}
        return self.observation(), frames / self.horizon, terminated, truncated, info


def vector_env(cfg, workers, settings):
    return gym.vector.AsyncVectorEnv(
        [partial(EpisodeWorker, cfg, i, workers, settings) for i in range(workers)],
        context="forkserver", shared_memory=True,
    )


@torch.no_grad()
def collect(env, policy, value, seeds, settings, *, stochastic, training):
    """Collect every preassigned episode under a single immutable policy."""
    started = time.perf_counter()
    env.call("assign_seeds", list(seeds))
    observations, _ = env.reset()
    generators = {seed: np.random.default_rng(seed) for seed in seeds}
    episodes, rows = [], []
    inference_seconds, step_seconds, simulation_seconds = 0.0, 0.0, 0.0
    batch_sizes, total_frames = [], 0
    device = next(policy.parameters()).device
    while observations["active"].any():
        selected = np.flatnonzero(observations["active"][:, 0])
        tick = time.perf_counter()
        # Keep GEMM shapes fixed so a different episode ending cannot change
        # another episode's deterministic actions through FP32 batch rounding.
        obs = policy.normalize(observations)
        logits = policy.logits(obs)[selected]
        obs = obs[selected]
        means = logits[:, policy.active]
        seed_ids = observations["seed"][selected, 0]
        if stochastic:
            noise = np.stack([generators[seed].standard_normal(means.shape[1]) for seed in seed_ids])
            latent = means + policy.std * torch.as_tensor(noise, dtype=torch.float32, device=device)
        else:
            latent = means
        actions = logits.clone()
        actions[:, policy.active] = latent
        outgoing = np.zeros((env.num_envs, env.single_action_space.shape[0]), dtype=np.float32)
        outgoing[selected] = actions.tanh().cpu().numpy()
        if training:
            remaining = torch.as_tensor(observations["remaining"][selected], device=device)
            rows.append({"obs": obs, "remaining": remaining, "latent": latent,
                         "means": means, "log_prob": policy.distribution(means).log_prob(latent),
                         "value": value(torch.cat((obs, remaining), 1)).squeeze(1),
                         "seed": seed_ids.copy(),
                         "frame": np.rint((1 - observations["remaining"][selected, 0]) * settings["episode_frames"]).astype(np.int64)})
        inference_seconds += time.perf_counter() - tick
        batch_sizes.append(len(selected))
        tick = time.perf_counter()
        observations, _, terminated, truncated, info = env.step(outgoing)
        step_seconds += time.perf_counter() - tick
        for i in selected:
            terminal = terminated[i] or truncated[i]
            details = info["final_info"][i] if terminal else {key: info[key][i] for key in ("frames", "simulation_seconds")}
            total_frames += details["frames"]
            simulation_seconds += details["simulation_seconds"]
            if terminal:
                episodes.append(details["episode"])
    episodes.sort(key=lambda episode: episode["seed"])
    elapsed = time.perf_counter() - started
    metrics = {"episodes": len(episodes), "frames": int(total_frames), "seconds": elapsed,
               "frames_per_second": total_frames / elapsed, "inference_seconds": inference_seconds,
               "step_wait_seconds": step_seconds, "summed_worker_simulation_seconds": simulation_seconds,
               "inference_batch": env.num_envs, "mean_active_envs": np.mean(batch_sizes).item(),
               "mean_length": np.mean([episode["length"] for episode in episodes]).item(),
               "mean_return": np.mean([episode["return"] for episode in episodes]).item()}
    if not training:
        return episodes, metrics, None
    data = {key: torch.cat([row[key] for row in rows])
            for key in ("obs", "remaining", "latent", "means", "log_prob", "value")}
    lengths = {episode["seed"]: episode["length"] for episode in episodes}
    returns = np.concatenate([[lengths[seed] - frame for seed, frame in zip(row["seed"], row["frame"])] for row in rows])
    data["returns"] = torch.as_tensor(returns, dtype=torch.float32, device=device) / settings["episode_frames"]
    data["advantages"] = data["returns"] - data["value"]
    return episodes, metrics, data


def update(policy, value, actor_optimizer, value_optimizer, data, settings, *, epochs, actor):
    actor_steps, value_steps, actor_stopped = 0, 0, not actor
    device = data["obs"].device
    for _ in range(epochs):
        for indices in torch.randperm(len(data["obs"]), device=device).split(settings["minibatch_size"]):
            obs = data["obs"][indices]
            if not actor_stopped:
                means = policy.logits(obs)[:, policy.active]
                kl = ((means.detach() - data["means"][indices]) / policy.std).square().sum(1).mean() / 2
                actor_stopped = kl.item() > settings["target_kl"]
                if not actor_stopped:
                    ratio = (policy.distribution(means).log_prob(data["latent"][indices]) - data["log_prob"][indices]).exp()
                    advantage = data["advantages"][indices]
                    clipped = ratio.clamp(1 - settings["clip_ratio"], 1 + settings["clip_ratio"])
                    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_optimizer.step()
                    actor_steps += 1
            predictions = value(torch.cat((obs, data["remaining"][indices]), 1)).squeeze(1)
            value_loss = (predictions - data["returns"][indices]).square().mean()
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            value_optimizer.step()
            value_steps += 1
    with torch.no_grad():
        means = policy.logits(data["obs"])[:, policy.active]
        ratio = (policy.distribution(means).log_prob(data["latent"]) - data["log_prob"]).exp()
        predicted = value(torch.cat((data["obs"], data["remaining"]), 1)).squeeze(1)
        advantage = data["advantages"]
        variance = data["returns"].var().item()
        metrics = {
            "actor_loss": -torch.minimum(ratio * advantage, ratio.clamp(1 - settings["clip_ratio"], 1 + settings["clip_ratio"]) * advantage).mean().item(),
            "value_loss": (predicted - data["returns"]).square().mean().item(),
            "value_explained_variance": 1 - (predicted - data["returns"]).var().item() / variance if variance else None,
            "advantage_mean": advantage.mean().item(), "advantage_std": advantage.std().item(),
            "kl": ((means - data["means"]) / policy.std).square().sum(1).mean().item() / 2,
            "clip_fraction": ((ratio - 1).abs() > settings["clip_ratio"]).float().mean().item(),
            "actor_updates": actor_steps, "value_updates": value_steps,
        }
    return metrics


def episode_summary(episodes):
    lengths = np.array([episode["length"] for episode in episodes])
    return {"n": len(episodes), "mean_length": lengths.mean().item(),
            "mean_return": np.mean([episode["return"] for episode in episodes]).item(),
            "ge100": np.count_nonzero(lengths >= 100),
            "ge500": np.count_nonzero(lengths >= 500), "episodes": episodes}


def comparison(episodes, baseline):
    delta = np.array([row["length"] - base["length"] for row, base in zip(episodes, baseline)])
    generator = np.random.default_rng(42)
    samples = generator.choice(delta, (10000, len(delta)), replace=True).mean(1)
    interval = np.quantile(samples, [.025, .975]).tolist()
    return {"mean_length_gain": delta.mean().item(), "paired_95_interval": interval,
            "accepted": interval[0] > 0 and episode_summary(episodes)["ge100"] >= episode_summary(baseline)["ge100"]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def make_policy(cfg, settings, device):
    env = make_gymnasium_sf_env(cfg.env, cfg, SimpleNamespace(worker_index=0, vector_index=0, env_id=0))
    try:
        return PPOPolicy(env.observation_space, env.action_space, settings).to(device)
    finally:
        env.close()


def run_from_config(config):
    settings = config["training"]["ppo"]
    cfg = build_train_cfg_from_config(config)
    cfg.episode_metrics_path = ""
    seed = config["experiment"]["seed"]
    # Forkserver children inherit these before importing NumPy/MuJoCo.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mp.set_forkserver_preload(["numpy", "torch", "training.gymnasium_sf.ppo"])
    device = torch.device("cuda" if cfg.device == "gpu" else "cpu")
    directory = Path(config["backend"]["train_dir"]) / config["experiment"]["name"]
    if cfg.mode == "train" and directory.exists():
        if cfg.restart_behavior == "restart":
            directory.rename(directory.with_name(f"{directory.name}_old_{time.time_ns()}"))
        elif cfg.restart_behavior == "overwrite":
            shutil.rmtree(directory)
    checkpoint_dir = directory / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy = make_policy(cfg, settings, device)
    policy.load_state_dict(torch.load(config["backend"]["initial_model_path"], map_location=device, weights_only=False)["model"])
    reference = copy.deepcopy(policy).requires_grad_(False)
    value = nn.Sequential(nn.Linear(policy.input_dim + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=settings["actor_lr"])
    value_optimizer = torch.optim.Adam(value.parameters(), lr=settings["value_lr"])
    counters = {"round": 0, "actor_updates": 0, "value_updates": 0, "env_frames": 0, "episodes": 0}
    best_length, best_round = -1.0, 0
    latest = checkpoint_dir / "checkpoint_latest.pth"
    workers = settings["workers"]
    resume = config["backend"]["restart_behavior"] == "resume"
    if resume or cfg.mode == "eval":
        kind = cfg.load_checkpoint_kind if cfg.mode == "eval" else "latest"
        path = checkpoint_dir / f"checkpoint_{kind}.pth"
        state = torch.load(path, map_location=device, weights_only=False)
        policy.load_state_dict(state["model"])
        value.load_state_dict(state["value"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        value_optimizer.load_state_dict(state["value_optimizer"])
        counters, workers = state["counters"], state["workers"]
        best_length, best_round = state["best_length"], state["best_round"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)

    name = config["experiment"]["name"]
    if cfg.mode == "eval":
        directory = directory / f"eval_{kind}_{time.time_ns()}"
        directory.mkdir()
        name = directory.name + "_" + name
    run = wandb.init(project=config["logging"]["wandb_project"], entity=config["logging"]["wandb_entity"],
                     group=config["logging"]["wandb_group"], name=name,
                     id=json.loads((directory / "run.json").read_text())["id"] if resume and cfg.mode == "train" else None,
                     resume="allow", mode="online", dir=str(directory), config=config)
    run.define_metric("budget/round")
    for section in ("train/*", "sampling/*", "eval/*", "perf/*"):
        run.define_metric(section, step_metric="budget/round")
    write_json(directory / "config.json", config)
    write_json(directory / "run.json", {"id": run.id, "url": run.url, "status": "running", "pid": os.getpid()})
    print(json.dumps({"wandb_url": run.url, "directory": str(directory)}), flush=True)
    env = None
    exit_code = 1
    try:
        if not resume and cfg.mode == "train":
            measurements = []
            for count in settings["benchmark_workers"]:
                env = vector_env(cfg, count, settings)
                try:
                    collect(env, reference, value, range(610001, 610001 + count), settings, stochastic=False, training=False)
                    for repeat in range(settings["benchmark_repeats"]):
                        _, metrics, _ = collect(env, reference, value, range(600001, 600001 + settings["episodes_per_round"]), settings, stochastic=False, training=False)
                        measurements.append({"workers": count, "repeat": repeat, **metrics})
                        run.log({"benchmark/" + key: value for key, value in measurements[-1].items()})
                        write_json(directory / "throughput.json", measurements)
                        print(json.dumps({"benchmark": measurements[-1]}), flush=True)
                finally:
                    env.close(terminate=True)
                    env = None
            if measurements:
                medians = {count: np.median([row["frames_per_second"] for row in measurements if row["workers"] == count]) for count in settings["benchmark_workers"]}
                workers = min(count for count, fps in medians.items() if fps >= .95 * max(medians.values()))
        env = vector_env(cfg, workers, settings)
        dev_seeds = range(*settings["development_seeds"])
        test_seeds = range(*settings["test_seeds"])
        baseline_dev, _, _ = collect(env, reference, value, dev_seeds, settings, stochastic=False, training=False)
        write_json(directory / "baseline_development.json", episode_summary(baseline_dev))
        if not resume and cfg.mode == "train":
            best_length = episode_summary(baseline_dev)["mean_length"]
            _, metrics, data = collect(env, policy, value, range(settings["train_seed_base"] - settings["episodes_per_round"], settings["train_seed_base"]), settings, stochastic=True, training=True)
            learned = update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=settings["value_warmup_epochs"], actor=False)
            counters["value_updates"] += learned["value_updates"]
            counters["env_frames"] += metrics["frames"]
            counters["episodes"] += metrics["episodes"]
            run.log({**{"warmup/" + key: value for key, value in metrics.items()}, **{"warmup/" + key: value for key, value in learned.items()}})

        def checkpoint():
            return {"model": policy.state_dict(), "value": value.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(), "value_optimizer": value_optimizer.state_dict(),
                    "counters": counters.copy(), "workers": workers, "best_length": best_length, "best_round": best_round,
                    "train_step": counters["actor_updates"], "env_steps": counters["env_frames"],
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

        if not resume and cfg.mode == "train":
            save_checkpoint(checkpoint_dir / "checkpoint_best.pth", checkpoint())
            save_checkpoint(latest, checkpoint())
        if cfg.mode == "train":
            for iteration in range(counters["round"] + 1, settings["rounds"] + 1):
                start = settings["train_seed_base"] + (iteration - 1) * settings["episodes_per_round"]
                episodes, metrics, data = collect(env, policy, value, range(start, start + settings["episodes_per_round"]), settings, stochastic=True, training=True)
                tick = time.perf_counter()
                learned = update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=settings["epochs"], actor=True)
                counters["round"] = iteration
                counters["actor_updates"] += learned["actor_updates"]
                counters["value_updates"] += learned["value_updates"]
                counters["env_frames"] += metrics["frames"]
                counters["episodes"] += len(episodes)
                record = {"round": iteration, "sampling": metrics, "learning": learned,
                          "learning_seconds": time.perf_counter() - tick, "counters": counters.copy(), "workers": workers}
                if iteration % settings["eval_every"] == 0 or iteration == settings["rounds"]:
                    evaluated, _, _ = collect(env, policy, value, dev_seeds, settings, stochastic=False, training=False)
                    record["evaluation"] = episode_summary(evaluated)
                    run.log({"budget/round": iteration, **{"eval/" + key: value for key, value in record["evaluation"].items() if key != "episodes"}})
                    if record["evaluation"]["mean_length"] > best_length:
                        best_length, best_round = record["evaluation"]["mean_length"], iteration
                        save_checkpoint(checkpoint_dir / "checkpoint_best.pth", checkpoint())
                save_checkpoint(checkpoint_dir / f"checkpoint_{iteration:04d}.pth", checkpoint())
                save_checkpoint(latest, checkpoint())
                write_json(directory / f"round_{iteration:04d}.json", record)
                run.log({**{"budget/" + key: value for key, value in counters.items()},
                         "budget/optimizer_calls": counters["actor_updates"] + counters["value_updates"],
                         **{"sampling/" + key: value for key, value in metrics.items()},
                         **{"train/" + key: value for key, value in learned.items()},
                         "perf/learning_seconds": record["learning_seconds"], "eval/best_length": best_length})
                print(json.dumps(record), flush=True)
                del data
        if cfg.mode == "train":
            state = torch.load(checkpoint_dir / "checkpoint_best.pth", map_location=device, weights_only=False)
            policy.load_state_dict(state["model"])
        baseline, _, _ = collect(env, reference, value, test_seeds, settings, stochastic=False, training=False)
        evaluated, _, _ = collect(env, policy, value, test_seeds, settings, stochastic=False, training=False)
        result = {"baseline": episode_summary(baseline), "policy": episode_summary(evaluated),
                  "comparison": comparison(evaluated, baseline), "best_round": best_round,
                  "evaluated_round": state["counters"]["round"],
                  "counters": counters, "workers": workers, "wandb_url": run.url,
                  "training_completed": counters["round"] == settings["rounds"], "evaluation_completed": True}
        write_json(directory / "result.json", result)
        run.summary.update({"final/" + key: value for key, value in result["comparison"].items()})
        write_json(directory / "run.json", {"id": run.id, "url": run.url, "status": "completed", "pid": os.getpid()})
        print(json.dumps(result), flush=True)
        exit_code = 0
    except BaseException:
        write_json(directory / "run.json", {"id": run.id, "url": run.url, "status": "failed", "pid": os.getpid(), "counters": counters})
        raise
    finally:
        if env is not None:
            env.close(terminate=True)
        run.finish(exit_code=exit_code)
    return 0
