#!/usr/bin/env python3
"""Compare conditioned H8 teacher suffixes with the deployed action generator.

Simulator, StarVLA, and plotting subcommands use separate runtime environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latency_bench.data.starvla_tasks import denormalize, normalize


def suffix_errors(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """Return errors over teacher-labelled suffix heads and the first such head."""
    difference = np.asarray(prediction) - np.asarray(target)
    valid = np.asarray(mask, dtype=bool)
    absolute = np.abs(difference[valid])
    first = int(np.flatnonzero(valid)[0])
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(difference[valid]).mean())),
        "first_head": first,
        "first_mae": float(np.abs(difference[first]).mean()),
        "first_rmse": float(np.sqrt(np.square(difference[first]).mean())),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def capture(args: argparse.Namespace) -> None:
    import torch

    from latency_bench.core.config import load_config
    from latency_bench.core.clock import EnvClock
    from latency_bench.scheduler.factory import build_action_scheduler
    from latency_bench.core.types import Action
    from scripts.mikasa.evaluate import _PpoPolicy
    from latency_bench.envs.tasks.mikasa import make_ppo_env, make_vla_env
    from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import (
        _conditioned_observation,
        _teacher_chunks,
        _uint8_rgb,
    )

    config = load_config(args.eval_config)
    prefix_settings = config["policy"]["action_prefix"]
    clock = EnvClock(env_fps=20, obs_fps=20)
    scheduler = build_action_scheduler(
        config["scheduler"], clock=clock,
        noop_action=Action(np.zeros(7, dtype=np.float32), name="noop", is_noop=True),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    episodes_dir = args.output / "episodes"
    episodes_dir.mkdir()
    rgb_env = make_vla_env(num_envs=1, simulator_device=args.device)
    branch_env = make_ppo_env(num_envs=1, simulator_device=args.device)
    teacher = _PpoPolicy(branch_env, args.teacher)
    try:
        branch_env.reset(seed=args.seed_start)
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            observation, _ = rgb_env.reset(seed=seed)
            scheduler.reset()
            state, prefix, prefix_mask = _conditioned_observation(
                observation["proprio"][0].detach().cpu().numpy(),
                scheduler, prefix_settings, 0,
            )
            chunk, loss_mask = _teacher_chunks(
                teacher, rgb_env, branch_env,
                prefix[None], prefix_mask[None], np.ones(1, dtype=bool),
            )
            rgb = _uint8_rgb(observation["rgb"])[0]
            np.savez_compressed(
                episodes_dir / f"seed_{seed}.npz",
                top=rgb[:, :, :3], wrist=rgb[:, :, 3:6], state=state,
                issued_command_chunk=chunk[0], action_prefix_mask=prefix_mask,
                action_loss_mask=loss_mask[0], episode_seed=np.asarray(seed),
            )
            print(f"captured {seed}", flush=True)
    finally:
        rgb_env.close()
        branch_env.close()
    _write_json(args.output / "capture.json", {
        "eval_config": str(args.eval_config), "teacher": str(args.teacher),
        "seed_start": args.seed_start, "episodes": args.episodes,
        "prefix": prefix_settings,
    })


def capture_batch_first(args: argparse.Namespace) -> None:
    from latency_bench.core.clock import EnvClock
    from latency_bench.core.config import load_config
    from latency_bench.core.types import Action
    from latency_bench.scheduler.factory import build_action_scheduler
    from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend
    from scripts.mikasa.evaluate import _PpoPolicy
    from latency_bench.envs.tasks.mikasa import make_ppo_env
    from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import (
        _conditioned_observation,
        _teacher_chunks,
    )

    config = load_config(args.eval_config)
    slots = config["evaluation"]["eval_parallel_envs"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "episodes").mkdir()
    backend = MikasaEnvStepBackend(config=config, num_slots=slots)
    branch = make_ppo_env(num_envs=slots, simulator_device=config["env"]["simulator_device"])
    teacher = _PpoPolicy(branch, args.teacher)
    scheduler = build_action_scheduler(
        config["scheduler"], clock=EnvClock(env_fps=20, obs_fps=20),
        noop_action=Action(np.zeros(7, dtype=np.float32), name="noop", is_noop=True),
    )
    try:
        branch.reset(seed=args.seed_start)
        for start in range(0, args.episodes, slots):
            observations = [
                backend._reset_slot_observation(slot, seed=args.seed_start + start + slot)
                for slot in range(slots)
            ]
            states, prefixes, masks = [], [], []
            for observation in observations:
                scheduler.reset()
                state, prefix, mask = _conditioned_observation(
                    observation.metadata["mikasa_proprio"], scheduler,
                    config["policy"]["action_prefix"], 0,
                )
                states.append(state)
                prefixes.append(prefix)
                masks.append(mask)
            # The older remote MIKASA env shrinks oracle_info on partial reset.
            # At reset, ball velocity is the same privileged field used by PPO.
            backend.env.unwrapped.oracle_info = backend.env.unwrapped.ball.linear_velocity.clone()
            chunks, valid = _teacher_chunks(
                teacher, backend.env, branch, np.stack(prefixes), np.stack(masks),
                np.ones(slots, dtype=bool),
            )
            for slot, observation in enumerate(observations):
                views = observation.metadata["mikasa_image_views"]
                seed = args.seed_start + start + slot
                np.savez_compressed(
                    args.output / f"episodes/seed_{seed}.npz",
                    top=views[0], wrist=views[1], state=states[slot],
                    issued_command_chunk=chunks[slot], action_prefix_mask=masks[slot],
                    action_loss_mask=valid[slot], episode_seed=np.asarray(seed),
                )
            print(f"captured batch {start // slots + 1}", flush=True)
    finally:
        backend.close()
        branch.close()
    _write_json(args.output / "capture.json", {
        "eval_config": str(args.eval_config), "teacher": str(args.teacher),
        "seed_start": args.seed_start, "episodes": args.episodes,
        "reset": "MikasaEnvStepBackend._reset_slot_observation",
    })


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    return [values[index] for index in np.linspace(0, len(values) - 1, count, dtype=int)]


def _dataset_rows(raw: Path, split: str, *, first_limit: int | None, later_limit: int) -> list[dict]:
    collection = json.loads((raw / "collection.json").read_text(encoding="utf-8"))
    episodes = collection["splits"][split]["episode_indices"]
    first = _evenly_spaced(episodes, first_limit) if first_limit is not None else episodes
    later = _evenly_spaced(episodes, later_limit)
    selected = {episode: [0] for episode in first}
    for episode in later:
        selected.setdefault(episode, []).append(-1)
    rows = []
    for episode, indices in selected.items():
        with np.load(raw / f"episodes/episode_{episode:06d}.npz") as data:
            for index in indices:
                row_index = 0 if index == 0 else max(1, len(data["state"]) // 2)
                if row_index >= len(data["state"]):
                    continue
                rows.append({
                    "split": split,
                    "position": "first" if row_index == 0 else "later",
                    "episode": int(data["episode_seed"]), "source_episode": episode,
                    "row": row_index,
                    "top": data["top"][row_index].copy(),
                    "wrist": data["wrist"][row_index].copy(),
                    "state": data["state"][row_index].copy(),
                    "target": data["issued_command_chunk"][row_index].copy(),
                    "prefix_mask": data["action_prefix_mask"][row_index].copy(),
                    "loss_mask": data["action_loss_mask"][row_index].copy(),
                })
    return rows


def _captured_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "episodes").glob("seed_*.npz")):
        with np.load(path) as data:
            rows.append({
                "split": "heldout", "position": "first", "episode": int(data["episode_seed"]),
                "row": 0, "top": data["top"].copy(), "wrist": data["wrist"].copy(),
                "state": data["state"].copy(), "target": data["issued_command_chunk"].copy(),
                "prefix_mask": data["action_prefix_mask"].copy(),
                "loss_mask": data["action_loss_mask"].copy(),
            })
    return rows


def _training_means(raw: Path) -> np.ndarray:
    collection = json.loads((raw / "collection.json").read_text(encoding="utf-8"))
    totals = np.zeros((8, 7), dtype=np.float64)
    counts = np.zeros(8, dtype=np.int64)
    for episode in collection["splits"]["train"]["episode_indices"]:
        with np.load(raw / f"episodes/episode_{episode:06d}.npz") as data:
            targets = data["issued_command_chunk"]
            mask = data["action_loss_mask"]
            totals += (targets * mask[:, :, None]).sum(0)
            counts += mask.sum(0)
    means = np.zeros((8, 7), dtype=np.float32)
    supervised = counts > 0
    means[supervised] = (totals[supervised] / counts[supervised, None]).astype(np.float32)
    return means


def _observation(row: dict, config: dict):
    from latency_bench.core.types import Observation

    policy = config["policy"]
    return Observation(
        data=None, env_step=row["row"], sim_time_ms=row["row"] * 50.0,
        metadata={
            policy["state_info_key"]: row["state"],
            policy["image_views_info_key"]: np.stack((row["top"], row["wrist"])),
            "action_prefix": row["target"],
            "action_prefix_mask": row["prefix_mask"],
        },
    )


def _clipping_fraction(values: np.ndarray, statistics: dict) -> float:
    minimum = np.asarray(statistics["min"])
    maximum = np.asarray(statistics["max"])
    return float(np.mean((values < minimum) | (values > maximum)))


def _correlations(targets: np.ndarray, predictions: np.ndarray) -> list[float | None]:
    values = []
    for dim in range(targets.shape[-1]):
        x, y = targets[:, dim], predictions[:, dim]
        values.append(
            float(np.corrcoef(x, y)[0, 1])
            if len(x) > 1 and np.std(x) > 1e-7 and np.std(y) > 1e-7 else None
        )
    return values


def _summarize(rows: list[dict], records: list[dict], means: np.ndarray) -> dict:
    errors = np.asarray([record["physical"]["mae"] for record in records])
    first_errors = np.asarray([record["physical"]["first_mae"] for record in records])
    normalized_errors = np.asarray([record["normalized"]["mae"] for record in records])
    target_valid = np.concatenate([row["target"][row["loss_mask"]] for row in rows])
    prediction_valid = np.concatenate([
        np.asarray(record["prediction"])[row["loss_mask"]] for row, record in zip(rows, records)
    ])
    head_metrics = {}
    dim_metrics = {}
    for head in range(8):
        pairs = [(row["target"][head], record["prediction"][head])
                 for row, record in zip(rows, records) if row["loss_mask"][head]]
        if pairs:
            target = np.stack([pair[0] for pair in pairs])
            prediction = np.stack([pair[1] for pair in pairs])
            head_metrics[str(head)] = {
                "count": len(pairs), "mae": float(np.abs(prediction - target).mean()),
                "correlation_by_dim": _correlations(target, prediction),
            }
    for dim in range(7):
        dim_metrics[str(dim)] = {
            "mae": float(np.abs(target_valid[:, dim] - prediction_valid[:, dim]).mean()),
            "correlation": _correlations(target_valid, prediction_valid)[dim],
        }
    constant_errors = np.asarray([
        np.abs(row["target"][row["loss_mask"]] - means[row["loss_mask"]]).mean()
        for row in rows
    ])
    episodes = sorted({row["episode"] for row in rows})
    episode_mae = np.asarray([
        errors[[row["episode"] == episode for row in rows]].mean() for episode in episodes
    ])
    episode_first = np.asarray([
        first_errors[[row["episode"] == episode for row in rows]].mean() for episode in episodes
    ])
    rng = np.random.default_rng(271828)
    draws = rng.integers(0, len(episodes), size=(1000, len(episodes)))
    return {
        "n": len(rows), "episodes": len(episodes),
        "mae": float(episode_mae.mean()),
        "mae_ci95": np.quantile(episode_mae[draws].mean(1), [0.025, 0.975]).tolist(),
        "first_mae": float(episode_first.mean()),
        "first_mae_ci95": np.quantile(episode_first[draws].mean(1), [0.025, 0.975]).tolist(),
        "normalized_mae": float(normalized_errors.mean()),
        "mae_by_noise_seed": {
            str(seed): float(np.mean([
                record["physical"]["mae"] for record in records if record["noise_seed"] == seed
            ])) for seed in sorted({record["noise_seed"] for record in records})
        },
        "mean_sampling_mae_std": float(np.mean([
            np.std([record["physical"]["mae"] for row, record in zip(rows, records)
                    if row["episode"] == episode]) for episode in episodes
        ])),
        "constant_predictor_mae": float(constant_errors.mean()),
        "objective": float(np.mean([record["objective"] for record in records])),
        "state_clipped_fraction": float(np.mean([record["state_clipped"] for record in records])),
        "prefix_clipped_fraction": float(np.mean([record["prefix_clipped"] for record in records])),
        "by_head": head_metrics, "by_dim": dim_metrics,
    }


def _plot_first_suffix(results: list[tuple[dict, dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    colors = {"train": "#287bb5", "validation": "#e29a26", "heldout": "#bb4b45"}
    for dim, axis in enumerate(axes.flat[:7]):
        for split in ("train", "validation", "heldout"):
            pairs = [
                (row["target"][record["physical"]["first_head"]],
                 np.asarray(record["prediction"])[record["physical"]["first_head"]])
                for row, record in results
                if row["split"] == split and row["position"] == "first"
                and record["noise_seed"] == 0
            ]
            if pairs:
                teacher = np.asarray([pair[0][dim] for pair in pairs])
                student = np.asarray([pair[1][dim] for pair in pairs])
                axis.scatter(teacher, student, s=8, alpha=0.45, color=colors[split], label=split)
        axis.plot([-1, 1], [-1, 1], color="black", linewidth=0.7)
        axis.set(xlabel="PPO teacher", ylabel="VLA", title=f"first suffix · action {dim}")
    axes.flat[7].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def oracle(args: argparse.Namespace) -> None:
    from PIL import Image

    from latency_bench.envs.tasks.mikasa import make_vla_env
    from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import _uint8_rgb

    env = make_vla_env(num_envs=1, simulator_device=args.device)
    rows = []
    try:
        for path in sorted((args.capture / "episodes").glob("seed_*.npz")):
            with np.load(path) as data:
                seed = int(data["episode_seed"])
                observation, _ = env.reset(seed=seed)
                rgb = _uint8_rgb(observation["rgb"])[0]
                image = np.concatenate((data["top"], data["wrist"]), axis=-1)
                head = int(np.flatnonzero(data["action_loss_mask"])[0])
                velocity = env.unwrapped.oracle_info[0].detach().cpu().numpy().copy()
                thumbnail = np.concatenate([
                    np.asarray(Image.fromarray(image[:, :, view]).resize((32, 32)), dtype=np.float32)
                    for view in (slice(0, 3), slice(3, 6))
                ], axis=-1)
                rows.append({
                    "seed": seed, "velocity": velocity, "image": image,
                    "thumbnail": thumbnail.reshape(-1),
                    "teacher_first": data["issued_command_chunk"][head].copy(),
                    "repeat_image_mae": float(np.abs(rgb.astype(np.float32) - image).mean()),
                })
    finally:
        env.close()

    features = np.stack([row["thumbnail"] for row in rows])
    distances = np.maximum(
        np.square(features).sum(1)[:, None] + np.square(features).sum(1)[None, :]
        - 2 * features @ features.T, 0,
    )
    np.fill_diagonal(distances, np.inf)
    pairs = sorted({tuple(sorted((i, j))) for i in range(len(rows))
                    for j in np.argsort(distances[i])[:5]})
    pair_metrics = []
    for i, j in pairs:
        pair_metrics.append({
            "seed_a": rows[i]["seed"], "seed_b": rows[j]["seed"],
            "image_mae_255": float(np.abs(rows[i]["image"].astype(np.float32)
                                          - rows[j]["image"].astype(np.float32)).mean()),
            "velocity_distance": float(np.linalg.norm(rows[i]["velocity"] - rows[j]["velocity"])),
            "teacher_first_mae": float(np.abs(rows[i]["teacher_first"]
                                               - rows[j]["teacher_first"]).mean()),
        })
    velocity_gap = np.asarray([row["velocity_distance"] for row in pair_metrics])
    teacher_gap = np.asarray([row["teacher_first_mae"] for row in pair_metrics])
    q25, q75 = np.quantile(velocity_gap, [0.25, 0.75])
    velocity = np.stack([row["velocity"] for row in rows])
    teacher = np.stack([row["teacher_first"] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, {
        "capture": str(args.capture), "n": len(rows),
        "repeat_image_mae_255": float(np.mean([row["repeat_image_mae"] for row in rows])),
        "velocity_min": velocity.min(0).tolist(), "velocity_max": velocity.max(0).tolist(),
        "teacher_correlation_by_velocity_dim": {
            str(dim): [
                _correlations(velocity[:, dim, None], teacher[:, action_dim, None])[0]
                for action_dim in range(7)
            ] for dim in range(3)
        },
        "nearest_pair_count": len(pairs),
        "nearest_pair_image_mae_255_median": float(np.median([row["image_mae_255"] for row in pair_metrics])),
        "teacher_gap_low_velocity_quartile": float(np.median(teacher_gap[velocity_gap <= q25])),
        "teacher_gap_high_velocity_quartile": float(np.median(teacher_gap[velocity_gap >= q75])),
        "nearest_pairs": pair_metrics,
    })


def rollout_first(args: argparse.Namespace) -> None:
    records = [json.loads(line) for line in args.actions.read_text(encoding="utf-8").splitlines()]
    first_requests = [record for record in records if record["obs_env_step"] == 0
                      and record["metadata"]["episode_id"] < args.limit]
    rows = []
    for record in first_requests:
        episode_id = record["metadata"]["episode_id"]
        seed = args.seed_start + episode_id
        with np.load(args.capture / f"episodes/seed_{seed}.npz") as data:
            prediction = np.asarray(record["action_chunk"])
            target = data["issued_command_chunk"]
            mask = data["action_loss_mask"]
            rows.append({"seed": seed, "episode_id": episode_id,
                         **suffix_errors(prediction, target, mask)})
    mae = np.asarray([row["mae"] for row in rows])
    first_mae = np.asarray([row["first_mae"] for row in rows])
    rng = np.random.default_rng(271828)
    draws = rng.integers(0, len(rows), size=(1000, len(rows)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, {
        "n": len(rows), "mae": float(mae.mean()),
        "mae_ci95": np.quantile(mae[draws].mean(1), [0.025, 0.975]).tolist(),
        "first_mae": float(first_mae.mean()),
        "first_mae_ci95": np.quantile(first_mae[draws].mean(1), [0.025, 0.975]).tolist(),
        "rows": rows,
    })


def batch_reset_parity(args: argparse.Namespace) -> None:
    from latency_bench.core.config import load_config
    from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend

    config = load_config(args.eval_config)
    slots = config["evaluation"]["eval_parallel_envs"]
    backend = MikasaEnvStepBackend(config=config, num_slots=slots)
    rows = []
    try:
        for episode_id in range(slots):
            seed = args.seed_start + episode_id
            observation = backend._reset_slot_observation(episode_id, seed=seed)
            with np.load(args.capture / f"episodes/seed_{seed}.npz") as data:
                views = observation.metadata["mikasa_image_views"]
                expected = np.stack((data["top"], data["wrist"]))
                state = observation.metadata["mikasa_proprio"]
                rows.append({
                    "slot": episode_id, "seed": seed,
                    "image_mae_255": float(np.abs(views.astype(np.float32) - expected).mean()),
                    "proprio_max_abs": float(np.abs(state - data["state"][:7]).max()),
                })
    finally:
        backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, {"slots": rows})


def corrected_eval(args: argparse.Namespace) -> None:
    import torch
    import yaml

    from latency_bench.core.config import load_config
    from latency_bench.eval.driver import run_from_config
    from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend

    class ScalarResetBackend(MikasaEnvStepBackend):
        """Use the scalar seed for the slot whose episode is starting."""

        def _reset_slot_observation(self, slot_id: int, *, seed: int | None):
            if seed is not None:
                self._episode_seeds[slot_id] = int(seed)
            env_idx = torch.tensor([slot_id], device=self.env.unwrapped.device)
            self._observation, _ = self.env.reset(
                seed=self._episode_seeds[slot_id], options={"env_idx": env_idx}
            )
            self._env_steps[slot_id] = 0
            self._success[slot_id] = False
            return self._observation_for_slot(slot_id)

    config = load_config(args.eval_config)
    config["logging"]["output_dir"] = str(args.output)
    config["experiment"]["name"] += "_scalar_reset_diagnostic"
    config["evaluation"]["eval_episodes"] = args.episodes
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "eval_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    backend = ScalarResetBackend(
        config=config, num_slots=config["evaluation"]["eval_parallel_envs"]
    )
    run_from_config(
        config, env_backend=backend,
        inference_devices=config["executor"]["inference_devices"],
    )


def rollout_relabel(args: argparse.Namespace) -> None:
    import torch

    from latency_bench.core.clock import EnvClock
    from latency_bench.core.config import load_config
    from latency_bench.core.types import Action, ActionEvent, LatencyRecord
    from latency_bench.scheduler.factory import build_action_scheduler
    from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend
    from scripts.mikasa.evaluate import _PpoPolicy
    from latency_bench.envs.tasks.mikasa import make_ppo_env
    from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import (
        _conditioned_observation,
        _teacher_chunks,
    )

    config = load_config(args.eval_config)
    slots = config["evaluation"]["eval_parallel_envs"]
    action_rows = [json.loads(line) for line in args.actions.read_text().splitlines()]
    step_rows = [json.loads(line) for line in args.steps.read_text().splitlines()]
    actions_by_request = {
        (row["metadata"]["episode_id"], row["obs_env_step"]): row
        for row in action_rows if row["metadata"]["episode_id"] < slots
    }
    actions_by_apply = {
        (row["metadata"]["episode_id"], row["apply_raw_frame"]): row
        for row in action_rows if row["metadata"]["episode_id"] < slots and row["applied"]
    }
    steps_by_frame = {
        (row["episode_id"], row["env_step"]): row
        for row in step_rows if row["episode_id"] < slots
    }
    backend = MikasaEnvStepBackend(config=config, num_slots=slots)
    branch = make_ppo_env(num_envs=slots, simulator_device=config["env"]["simulator_device"])
    teacher = _PpoPolicy(branch, args.teacher)
    clock = EnvClock(env_fps=20, obs_fps=20)
    schedulers = [
        build_action_scheduler(
            config["scheduler"], clock=clock,
            noop_action=Action(np.zeros(7, dtype=np.float32), name="noop", is_noop=True),
        ) for _ in range(slots)
    ]
    requests = []
    action_differences = []
    reward_differences = []
    counts = np.zeros(slots, dtype=np.int64)
    try:
        branch.reset(seed=config["experiment"]["seed"])
        for slot in range(slots):
            backend._reset_slot_observation(slot, seed=config["experiment"]["seed"] + slot)
        oracle_initial = backend.env.unwrapped.ball.linear_velocity.clone()
        for frame in range(config["evaluation"]["eval_max_steps"]):
            prefixes = np.zeros((slots, 8, 7), dtype=np.float32)
            masks = np.zeros((slots, 8), dtype=bool)
            submitted = np.zeros(slots, dtype=bool)
            for slot in range(slots):
                applied = actions_by_apply.get((slot, frame))
                if applied is not None:
                    chunk = np.asarray(applied["action_chunk"], dtype=np.float32)
                    schedulers[slot].enqueue(ActionEvent(
                        action_id=applied["action_id"],
                        action=Action(value=chunk[0]), action_chunk=chunk,
                        obs_env_step=applied["obs_env_step"],
                        ready_time_ms=frame * 50.0, ready_step_float=float(frame),
                        ready_raw_frame=frame,
                        latency=LatencyRecord(obs_id=0, latency_ms=0.0),
                    ))
                request = actions_by_request.get((slot, frame))
                if request is not None:
                    _, prefix, mask = _conditioned_observation(
                        backend._observation["proprio"][slot].detach().cpu().numpy(),
                        schedulers[slot], config["policy"]["action_prefix"], frame,
                    )
                    prefixes[slot] = prefix
                    masks[slot] = mask
                    submitted[slot] = True
            if submitted.any():
                backend.env.unwrapped.oracle_info = oracle_initial.clone()
                targets, valid = _teacher_chunks(
                    teacher, backend.env, branch, prefixes, masks, submitted,
                )
                for slot in np.flatnonzero(submitted):
                    if valid[slot].any():
                        prediction = np.asarray(actions_by_request[(slot, frame)]["action_chunk"])
                        metrics = suffix_errors(prediction, targets[slot], valid[slot])
                        requests.append({
                            "episode_id": int(slot), "frame": frame,
                            "request_index": int(counts[slot]), **metrics,
                        })
                    counts[slot] += 1
            commands = np.zeros((slots, 7), dtype=np.float32)
            for slot in range(slots):
                command = np.asarray(schedulers[slot].update(frame, frame * 50.0).value)
                logged = np.asarray(steps_by_frame[(slot, frame)]["active_action"]["value"])
                action_differences.append(float(np.abs(command - logged).max()))
                commands[slot] = logged
            observation, reward, _, _, _ = backend.env.step(
                torch.as_tensor(commands, device=backend.env.unwrapped.device)
            )
            backend._observation = observation
            reward_values = reward.detach().cpu().numpy().reshape(-1)
            for slot in range(slots):
                reward_differences.append(float(abs(reward_values[slot]
                                                - steps_by_frame[(slot, frame)]["reward"])))
    finally:
        backend.close()
        branch.close()
    by_index = {}
    for index in sorted({row["request_index"] for row in requests}):
        values = [row["mae"] for row in requests if row["request_index"] == index]
        by_index[str(index)] = {"n": len(values), "mae": float(np.mean(values))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, {
        "action_replay_max_abs": max(action_differences),
        "reward_replay_max_abs": max(reward_differences),
        "by_request_index": by_index,
        "requests": requests,
    })


def overfit(args: argparse.Namespace) -> None:
    import torch
    import yaml

    from latency_bench.policy.starvla_tasks import build_task_starvla_policy

    config = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    config["policy"]["backbone_path"] = str(args.backbone)
    config["policy"]["device"] = args.device
    runner = build_task_starvla_policy(config)
    statistics = runner.contract["normalization"]["action"]
    rows = _dataset_rows(args.raw, "train", first_limit=args.rows, later_limit=0)
    heldout = _captured_rows(args.capture)[:16]
    examples = []
    for row in rows:
        example = runner._example(_observation(row, config))
        example.update(
            action=normalize(row["target"], statistics).astype(np.float16),
            action_prefix_mask=row["prefix_mask"], action_loss_mask=row["loss_mask"],
        )
        examples.append(example)
    for parameter in runner.framework.parameters():
        parameter.requires_grad_(False)
    for parameter in runner.framework.action_model.parameters():
        parameter.requires_grad_(True)
    runner.framework.eval()
    runner.framework.action_model.train()
    optimizer = torch.optim.AdamW(runner.framework.action_model.parameters(), lr=1e-4)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=False)

    def endpoint_mae(samples: list[dict]) -> float:
        errors = []
        runner.framework.action_model.eval()
        for row in samples:
            example = runner._example(_observation(row, config))
            torch.manual_seed(0)
            with torch.inference_mode():
                normalized = runner.framework.predict_action(examples=[example])["normalized_actions"][0]
            prediction = denormalize(normalized, statistics)
            errors.append(suffix_errors(prediction, row["target"], row["loss_mask"])["mae"])
        runner.framework.action_model.train()
        return float(np.mean(errors))

    with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as sink:
        sink.write(json.dumps({
            "step": 0, "train_mae": endpoint_mae(rows[:32]),
            "heldout_mae": endpoint_mae(heldout),
        }) + "\n")
        losses = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = runner.framework.forward(examples=[examples[(step - 1) % len(examples)]])["action_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.framework.action_model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 100 == 0:
                metrics = {
                    "step": step, "objective": float(np.mean(losses)),
                    "train_mae": endpoint_mae(rows[:32]),
                    "heldout_mae": endpoint_mae(heldout),
                }
                sink.write(json.dumps(metrics) + "\n")
                sink.flush()
                print(metrics, flush=True)
                losses.clear()


def plot_relabel(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for entry in args.input:
        name, path = entry.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        x, y, lower, upper = [], [], [], []
        for index in range(12):
            errors = np.asarray([
                row["mae"] for row in report["requests"] if row["request_index"] == index
            ])
            if len(errors) == 0:
                break
            rng = np.random.default_rng(271828 + index)
            draws = rng.integers(0, len(errors), size=(1000, len(errors)))
            interval = np.quantile(errors[draws].mean(1), [0.025, 0.975])
            x.append(index + 1)
            y.append(errors.mean())
            lower.append(interval[0])
            upper.append(interval[1])
        axis.plot(x, y, marker="o", markersize=3, label=name)
        axis.fill_between(x, lower, upper, alpha=0.12)
    axis.set(xlabel="request index", ylabel="teacher suffix MAE", xlim=(1, 12))
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


def sampler_steps(args: argparse.Namespace) -> None:
    import torch
    import yaml

    from latency_bench.policy.starvla_tasks import build_task_starvla_policy

    config = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    config["policy"]["backbone_path"] = str(args.backbone)
    config["policy"]["device"] = args.device
    runner = build_task_starvla_policy(config)
    action_stats = runner.contract["normalization"]["action"]
    rows = (
        _dataset_rows(args.raw, "train", first_limit=args.rows, later_limit=0)
        + _dataset_rows(args.raw, "validation", first_limit=args.rows, later_limit=0)
    )
    results = {}
    for steps in (4, 8, 16, 32):
        runner.framework.action_model.num_inference_timesteps = steps
        split_errors = {"train": [], "validation": []}
        split_dim_zero = {"train": [], "validation": []}
        for row in rows:
            torch.manual_seed(0)
            with torch.inference_mode():
                normalized = runner.framework.predict_action(
                    examples=[runner._example(_observation(row, config))]
                )["normalized_actions"][0]
            prediction = denormalize(normalized, action_stats)
            split_errors[row["split"]].append(
                suffix_errors(prediction, row["target"], row["loss_mask"])["mae"]
            )
            split_dim_zero[row["split"]].append(
                float(np.abs(prediction[row["loss_mask"], 0]
                             - row["target"][row["loss_mask"], 0]).mean())
            )
        results[str(steps)] = {
            split: {
                "n": len(split_errors[split]),
                "mae": float(np.mean(split_errors[split])),
                "action_0_mae": float(np.mean(split_dim_zero[split])),
            } for split in ("train", "validation")
        }
        print(steps, results[str(steps)], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, results)


def parity(args: argparse.Namespace) -> None:
    import yaml
    from omegaconf import OmegaConf

    from latency_bench.policy.starvla_tasks import TaskStarVlaPolicyRunner
    from starVLA.dataloader.latency_bench_tasks import get_vla_dataset

    config = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    runner = TaskStarVlaPolicyRunner(
        None, policy_config=config["policy"], model_config=model_config, contract=contract
    )
    data_cfg = OmegaConf.load(args.training_config).datasets.vla_data
    results = []
    for split in ("train", "validation"):
        data_cfg.data_mix = split
        mixture = get_vla_dataset(data_cfg, mode="val")
        dataset = mixture.datasets[0]
        index_lookup = {tuple(pair): index for index, pair in enumerate(dataset.all_steps)}
        collection = json.loads((args.raw / "collection.json").read_text(encoding="utf-8"))
        episode_indices = collection["splits"][split]["episode_indices"]
        selected = _dataset_rows(args.raw, split, first_limit=8, later_limit=8)
        for row in selected:
            local_episode = episode_indices.index(row["source_episode"])
            sample = dataset[index_lookup[(local_episode, row["row"])]]
            example = runner._example(_observation(row, config))
            np.testing.assert_allclose(sample["state"], example["state"], atol=2e-3)
            np.testing.assert_allclose(
                sample["action"], normalize(row["target"], contract["normalization"]["action"]),
                atol=2e-3,
            )
            np.testing.assert_array_equal(sample["action_prefix_mask"], row["prefix_mask"])
            np.testing.assert_array_equal(sample["action_loss_mask"], row["loss_mask"])
            image_mae = [
                float(np.abs(np.asarray(train_image, dtype=np.float32)
                             - np.asarray(eval_image, dtype=np.float32)).mean())
                for train_image, eval_image in zip(sample["image"], example["image"])
            ]
            results.append({
                "split": split, "episode": row["episode"], "row": row["row"],
                "image_mae_255": image_mae,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, {
        "n": len(results), "image_mae_255_mean": np.mean(
            [row["image_mae_255"] for row in results], axis=0
        ).tolist(), "rows": results,
    })


def analyze(args: argparse.Namespace) -> None:
    import torch
    import yaml

    from latency_bench.policy.starvla_tasks import build_task_starvla_policy

    config = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    config["policy"]["backbone_path"] = str(args.backbone)
    runner = build_task_starvla_policy(config)
    contract = runner.contract
    stats = contract["normalization"]
    rows = (
        _dataset_rows(args.raw, "train", first_limit=args.train_first, later_limit=args.train_later)
        + _dataset_rows(args.raw, "validation", first_limit=args.val_first, later_limit=args.val_later)
        + _captured_rows(args.capture)
    )
    means = _training_means(args.raw)
    args.output.mkdir(parents=True, exist_ok=False)
    all_results = []
    with (args.output / "per_sample.jsonl").open("w", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            example = runner._example(_observation(row, config))
            target_norm = normalize(row["target"], stats["action"])
            example.update(
                action=target_norm.astype(np.float16),
                action_prefix_mask=row["prefix_mask"],
                action_loss_mask=row["loss_mask"],
            )
            seed_results = []
            for noise_seed in range(3 if args.model != "qwenoft" else 1):
                torch.manual_seed(noise_seed)
                with torch.inference_mode():
                    normalized = runner.framework.predict_action(examples=[example])["normalized_actions"][0]
                prediction = denormalize(normalized, stats["action"])
                torch.manual_seed(noise_seed)
                with torch.no_grad():
                    objective = runner.framework.forward(examples=[example])["action_loss"].item()
                seed_results.append({
                    "physical": suffix_errors(prediction, row["target"], row["loss_mask"]),
                    "normalized": suffix_errors(normalized, target_norm, row["loss_mask"]),
                    "objective": objective,
                    "prediction": prediction.tolist(),
                })
            for noise_seed, result in enumerate(seed_results):
                result.update(
                    split=row["split"], position=row["position"], episode=row["episode"],
                    row=row["row"], noise_seed=noise_seed,
                    state_clipped=_clipping_fraction(row["state"], stats["state"]),
                    prefix_clipped=_clipping_fraction(
                        row["target"][row["prefix_mask"]], stats["action"]
                    ),
                )
                sink.write(json.dumps({key: value for key, value in result.items() if key != "prediction"}) + "\n")
                all_results.append((row, result))
            if (index + 1) % 25 == 0:
                print(f"analyzed {index + 1}/{len(rows)}", flush=True)
    summary = {}
    for split in ("train", "validation", "heldout"):
        for position in ("first", "later"):
            selected = [(row, record) for row, record in all_results
                        if row["split"] == split and row["position"] == position]
            if selected:
                summary[f"{split}_{position}"] = _summarize(
                    [row for row, _ in selected], [record for _, record in selected], means
                )
    result = {
        "model": args.model, "checkpoint": str(config["policy"]["checkpoint_path"]),
        "raw": str(args.raw), "capture": str(args.capture), "groups": summary,
    }
    if args.actions is not None:
        logged = {
            record["metadata"]["episode_id"]: np.asarray(record["action_chunk"])
            for record in (json.loads(line) for line in args.actions.read_text().splitlines())
            if record["obs_env_step"] == 0
        }
        differences = [
            (row["episode"] - config["experiment"]["seed"],
             float(np.abs(np.asarray(record["prediction"])[row["loss_mask"]]
                          - logged[row["episode"] - config["experiment"]["seed"]][row["loss_mask"]]).mean()))
            for row, record in all_results
            if row["split"] == "heldout" and record["noise_seed"] == 0
        ]
        result["rollout_prediction_parity"] = {
            "all_mae": float(np.mean([difference for _, difference in differences])),
            "first_ten_mae": float(np.mean([difference for episode, difference in differences
                                            if episode < 10])),
        }
    _write_json(args.output / "summary.json", result)
    _plot_first_suffix(all_results, args.output / "first_suffix_scatter.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--eval-config", type=Path, required=True)
    capture_parser.add_argument("--teacher", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--device", default="cuda:0")
    capture_parser.add_argument("--seed-start", type=int, default=4242424242)
    capture_parser.add_argument("--episodes", type=int, default=200)
    batch_capture_parser = sub.add_parser("capture-batch-first")
    batch_capture_parser.add_argument("--eval-config", type=Path, required=True)
    batch_capture_parser.add_argument("--teacher", type=Path, required=True)
    batch_capture_parser.add_argument("--output", type=Path, required=True)
    batch_capture_parser.add_argument("--seed-start", type=int, default=4242424242)
    batch_capture_parser.add_argument("--episodes", type=int, default=200)
    parity_parser = sub.add_parser("parity")
    parity_parser.add_argument("--raw", type=Path, required=True)
    parity_parser.add_argument("--eval-config", type=Path, required=True)
    parity_parser.add_argument("--training-config", type=Path, required=True)
    parity_parser.add_argument("--model-config", type=Path, required=True)
    parity_parser.add_argument("--contract", type=Path, required=True)
    parity_parser.add_argument("--output", type=Path, required=True)
    oracle_parser = sub.add_parser("oracle")
    oracle_parser.add_argument("--capture", type=Path, required=True)
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.add_argument("--device", default="cuda:0")
    rollout_parser = sub.add_parser("rollout-first")
    rollout_parser.add_argument("--actions", type=Path, required=True)
    rollout_parser.add_argument("--capture", type=Path, required=True)
    rollout_parser.add_argument("--output", type=Path, required=True)
    rollout_parser.add_argument("--seed-start", type=int, default=4242424242)
    rollout_parser.add_argument("--limit", type=int, default=200)
    batch_parser = sub.add_parser("batch-reset-parity")
    batch_parser.add_argument("--eval-config", type=Path, required=True)
    batch_parser.add_argument("--capture", type=Path, required=True)
    batch_parser.add_argument("--output", type=Path, required=True)
    batch_parser.add_argument("--seed-start", type=int, default=4242424242)
    corrected_parser = sub.add_parser("corrected-eval")
    corrected_parser.add_argument("--eval-config", type=Path, required=True)
    corrected_parser.add_argument("--output", type=Path, required=True)
    corrected_parser.add_argument("--episodes", type=int, default=200)
    relabel_parser = sub.add_parser("rollout-relabel")
    relabel_parser.add_argument("--eval-config", type=Path, required=True)
    relabel_parser.add_argument("--teacher", type=Path, required=True)
    relabel_parser.add_argument("--actions", type=Path, required=True)
    relabel_parser.add_argument("--steps", type=Path, required=True)
    relabel_parser.add_argument("--output", type=Path, required=True)
    overfit_parser = sub.add_parser("overfit")
    overfit_parser.add_argument("--eval-config", type=Path, required=True)
    overfit_parser.add_argument("--raw", type=Path, required=True)
    overfit_parser.add_argument("--capture", type=Path, required=True)
    overfit_parser.add_argument("--backbone", type=Path, required=True)
    overfit_parser.add_argument("--output", type=Path, required=True)
    overfit_parser.add_argument("--device", default="cuda:0")
    overfit_parser.add_argument("--rows", type=int, default=256)
    overfit_parser.add_argument("--steps", type=int, default=1000)
    plot_parser = sub.add_parser("plot-relabel")
    plot_parser.add_argument("--input", action="append", required=True)
    plot_parser.add_argument("--output", type=Path, required=True)
    sampler_parser = sub.add_parser("sampler-steps")
    sampler_parser.add_argument("--eval-config", type=Path, required=True)
    sampler_parser.add_argument("--raw", type=Path, required=True)
    sampler_parser.add_argument("--backbone", type=Path, required=True)
    sampler_parser.add_argument("--output", type=Path, required=True)
    sampler_parser.add_argument("--device", default="cuda:0")
    sampler_parser.add_argument("--rows", type=int, default=64)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--model", choices=("qwenoft", "qwenpi_v3", "qwengr00t"), required=True)
    analyze_parser.add_argument("--eval-config", type=Path, required=True)
    analyze_parser.add_argument("--raw", type=Path, required=True)
    analyze_parser.add_argument("--capture", type=Path, required=True)
    analyze_parser.add_argument("--actions", type=Path)
    analyze_parser.add_argument("--backbone", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--train-first", type=int, default=256)
    analyze_parser.add_argument("--train-later", type=int, default=256)
    analyze_parser.add_argument("--val-first", type=int)
    analyze_parser.add_argument("--val-later", type=int, default=256)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args)
    elif args.mode == "capture-batch-first":
        capture_batch_first(args)
    elif args.mode == "parity":
        parity(args)
    elif args.mode == "oracle":
        oracle(args)
    elif args.mode == "rollout-first":
        rollout_first(args)
    elif args.mode == "batch-reset-parity":
        batch_reset_parity(args)
    elif args.mode == "corrected-eval":
        corrected_eval(args)
    elif args.mode == "rollout-relabel":
        rollout_relabel(args)
    elif args.mode == "overfit":
        overfit(args)
    elif args.mode == "plot-relabel":
        plot_relabel(args)
    elif args.mode == "sampler-steps":
        sampler_steps(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
