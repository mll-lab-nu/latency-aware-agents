#!/usr/bin/env python3
"""Collect MIKASA H8 attempts with causal fixed-prefix teacher labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.executors._simulated_timeline import (
    SimulatedResultTimeline,
    SimulatedWorkerCapacity,
    build_simulated_action_event,
)
from latency_bench.latency.samplers import build_sampler
from latency_bench.logging.metrics import latency_type_from_source, profile_metadata_from_source
from latency_bench.policy.action_prefix import resolve_action_prefix, with_action_prefix
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES
from latency_bench.scheduler.factory import build_action_scheduler
from scripts.mikasa.evaluate import (
    MIKASA_IMAGE_VIEWS_INFO_KEY,
    MIKASA_STATE_INFO_KEY,
    _PpoPolicy,
    _make_ppo_env,
    _make_vla_env,
    _reset_seeded_slots,
)


ACTION_DIM = 7
HORIZON = 8
FPS = 20
MAX_EPISODE_STEPS = 60


def _prefix_settings(condition: str, config: dict) -> dict:
    if condition == "zero-latency":
        return {"mode": "none"}
    settings = (
        {
            "mode": "profile_p95",
            "profile_path": config["latency"]["profile_path"],
            "profile_revision": config["profile_source"]["revision"],
        }
        if condition == "profile-latency"
        else {"mode": "fixed_frames", "frames": FIXED_LATENCIES[condition]}
    )
    return resolve_action_prefix(
        settings,
        env_fps=FPS,
        action_horizon=HORIZON,
        action_dim=ACTION_DIM,
    )


def _copy_branch_state(source, target) -> None:
    source_base = source.unwrapped
    target_base = target.unwrapped
    target_base.set_state_dict(source_base.get_state_dict())
    target_base._elapsed_steps.copy_(source_base.elapsed_steps)
    target_base.oracle_info = source_base.oracle_info.clone()
    target_base.reached_status.copy_(source_base.reached_status)


def _branch_observation(env) -> dict[str, torch.Tensor]:
    return {"state": env.unwrapped.get_obs()}


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _uint8_rgb(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _as_numpy(value)
    if array.dtype == np.uint8:
        return array
    return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)


def _conditioned_observation(proprio, scheduler, prefix, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = Observation(
        data=None,
        env_step=frame,
        sim_time_ms=frame * 1000.0 / FPS,
        metadata={
            MIKASA_STATE_INFO_KEY: np.asarray(proprio, dtype=np.float32),
            "returned_action_context": {
                "state_key": MIKASA_STATE_INFO_KEY,
                "horizon": HORIZON,
                "frames": None,
                "order": np.arange(ACTION_DIM),
                "low": -np.ones(ACTION_DIM, dtype=np.float32),
                "high": np.ones(ACTION_DIM, dtype=np.float32),
            },
        },
    )
    conditioned = with_action_prefix(observation, scheduler, prefix)
    return (
        conditioned.metadata[MIKASA_STATE_INFO_KEY],
        conditioned.metadata["action_prefix"],
        conditioned.metadata["action_prefix_mask"],
    )


def _teacher_chunks(teacher, main_env, branch_env, prefixes, prefix_masks, submitted):
    _copy_branch_state(main_env, branch_env)
    observation = _branch_observation(branch_env)
    batch_size = len(submitted)
    device = branch_env.unwrapped.device
    targets = np.zeros((batch_size, HORIZON, ACTION_DIM), dtype=np.float32)
    valid = np.zeros((batch_size, HORIZON), dtype=bool)
    targets[prefix_masks] = prefixes[prefix_masks]
    alive = submitted.copy()

    for head in range(HORIZON):
        if not alive.any():
            break
        actions = torch.zeros((batch_size, ACTION_DIM), device=device)
        prefix_active = alive & prefix_masks[:, head]
        suffix_active = alive & ~prefix_masks[:, head]
        if prefix_active.any():
            prefix_active_tensor = torch.as_tensor(prefix_active, device=device)
            actions[prefix_active_tensor] = torch.as_tensor(
                prefixes[prefix_active, head],
                device=device,
            )
        if suffix_active.any():
            suffix_active_tensor = torch.as_tensor(suffix_active, device=device)
            command = teacher.forward(observation).clamp(-1, 1)
            command_array = command.detach().cpu().numpy()
            actions[suffix_active_tensor] = command[suffix_active_tensor]
            targets[suffix_active, head] = command_array[suffix_active]
            valid[suffix_active, head] = True
        observation, _, terminated, truncated, info = branch_env.step(actions)
        alive &= ~(
            _as_numpy(info["success"]).reshape(batch_size).astype(bool)
            | _as_numpy(terminated).reshape(batch_size).astype(bool)
            | _as_numpy(truncated).reshape(batch_size).astype(bool)
        )
    return targets, valid


def _selected_slots(
    valid_action_counts: list[int],
    *,
    collected: int,
    target: int,
) -> list[bool]:
    selected = []
    for count in valid_action_counts:
        keep = count > 0 and collected < target
        selected.append(keep)
        if keep:
            collected += count
    return selected


def _save_episode(path: Path, rows: list[dict], *, seed: int) -> None:
    values = {
        "top": np.stack([row["top"] for row in rows]),
        "wrist": np.stack([row["wrist"] for row in rows]),
        "state": np.stack([row["state"] for row in rows]),
        "action": np.stack([row["chunk"][0] for row in rows]),
        "issued_command_chunk": np.stack([row["chunk"] for row in rows]),
        "action_prefix_mask": np.stack([row["prefix_mask"] for row in rows]),
        "action_loss_mask": np.stack([row["loss_mask"] for row in rows]),
        "latency_raw_frames": np.asarray([row["latency_raw_frames"] for row in rows], dtype=np.int64),
        "latency_ms": np.asarray([row["latency_ms"] for row in rows], dtype=np.float32),
        "observation_frame": np.asarray([row["frame"] for row in rows], dtype=np.int64),
        "episode_seed": np.asarray(seed, dtype=np.int64),
    }
    if "student_chunk" in rows[0]:
        values["student_command_chunk"] = np.stack([row["student_chunk"] for row in rows])
        values["request_index"] = np.asarray([row["request_index"] for row in rows], dtype=np.int64)
    np.savez_compressed(path, **values)


def collect(args: argparse.Namespace, *, student_pool=None, seed_start: int = 0,
            simulator_device: str = "gpu", student_source: str | None = None) -> dict:
    config = load_config(args.latency_config)
    config["scheduler"]["hold_last_chunk_action"] = True
    prefix_settings = _prefix_settings(args.condition, config)
    clock = EnvClock(env_fps=FPS, obs_fps=FPS)
    samplers = []
    schedulers = []
    timelines = []
    capacities = []
    for slot_id in range(args.num_envs):
        latency_config = dict(config["latency"])
        if "seed" in latency_config:
            latency_config["seed"] = derive_seed(
                latency_config["seed"],
                vector_index=slot_id,
            )
        samplers.append(build_sampler(latency_config, clock=clock))
        scheduler = build_action_scheduler(
            config["scheduler"],
            clock=clock,
            noop_action=Action(
                np.zeros(ACTION_DIM, dtype=np.float32),
                name="noop",
                is_noop=True,
            ),
        )
        schedulers.append(scheduler)
        timelines.append(
            SimulatedResultTimeline(ordering_policy=scheduler.ordering_policy)
        )
        capacities.append(
            SimulatedWorkerCapacity(capacity=1, busy_until_by_worker={})
        )

    args.output.mkdir(parents=True)
    episodes_dir = args.output / "episodes"
    episodes_dir.mkdir()
    attempts_path = args.output / "attempts.jsonl"
    rgb_env = _make_vla_env(num_envs=args.num_envs, simulator_device=simulator_device)
    branch_env = _make_ppo_env(num_envs=args.num_envs, simulator_device=simulator_device)
    teacher = _PpoPolicy(branch_env, args.checkpoint)
    split_targets = {
        "train": args.train_targets,
        "validation": args.validation_targets,
    }
    splits = {
        name: {
            "target_valid_actions": target,
            "episode_indices": [],
            "seeds": [],
            "rows": 0,
            "valid_actions": 0,
            "successes": 0,
            "failures": 0,
        }
        for name, target in split_targets.items()
    }
    saved_episodes = []
    action_id = 0
    obs_id = 0
    latency_counts = Counter()
    attempted_seed = seed_start
    batch_index = 0
    student_errors = {"first": [], "later": []}

    try:
        while any(splits[name]["valid_actions"] < target for name, target in split_targets.items()):
            split = (
                "train"
                if splits["train"]["valid_actions"] < split_targets["train"]
                else "validation"
            )
            seeds = list(range(attempted_seed, attempted_seed + args.num_envs))
            if student_pool is not None:
                for slot_id in range(args.num_envs):
                    student_pool.reset_state(slot_id)
            rgb_observation = _reset_seeded_slots(rgb_env, seeds)
            if student_pool is None:
                _reset_seeded_slots(branch_env, seeds)
            else:
                # Each teacher query replaces the branch with an exact copy of main state.
                branch_env.reset(seed=0)
            for scheduler, timeline, capacity in zip(
                schedulers,
                timelines,
                capacities,
            ):
                scheduler.reset()
                timeline.reset()
                capacity.reset()
            rows = [[] for _ in range(args.num_envs)]
            success = np.zeros(args.num_envs, dtype=bool)
            active = np.ones(args.num_envs, dtype=bool)
            lengths = np.full(args.num_envs, MAX_EPISODE_STEPS, dtype=np.int64)
            episode_returns = np.zeros(args.num_envs, dtype=np.float64)
            recent_drop_counts = np.zeros(args.num_envs, dtype=np.int64)
            submitted_requests = np.zeros(args.num_envs, dtype=np.int64)
            applied_heads = [Counter() for _ in range(args.num_envs)]

            for frame in range(MAX_EPISODE_STEPS):
                proprio = _as_numpy(rgb_observation["proprio"])
                submitted = np.zeros(args.num_envs, dtype=bool)
                states = [None] * args.num_envs
                prefixes = np.zeros(
                    (args.num_envs, HORIZON, ACTION_DIM),
                    dtype=np.float32,
                )
                prefix_masks = np.zeros(
                    (args.num_envs, HORIZON),
                    dtype=bool,
                )

                for slot_id in np.flatnonzero(active):
                    capacity = capacities[slot_id]
                    timeline = timelines[slot_id]
                    scheduler = schedulers[slot_id]
                    capacity.release_ready(frame)
                    released, _ = timeline.release_arrived(frame)
                    for event in released:
                        scheduler.enqueue(event)
                    if capacity.can_submit() and timeline.pending_observation_count == 0:
                        if args.condition == "zero-latency":
                            state = np.asarray(proprio[slot_id], dtype=np.float32)
                            action_prefix = prefixes[slot_id]
                            prefix_mask = prefix_masks[slot_id]
                        else:
                            state, action_prefix, prefix_mask = _conditioned_observation(
                                proprio[slot_id], scheduler, prefix_settings, frame,
                            )
                        submitted[slot_id] = True
                        states[slot_id] = state
                        prefixes[slot_id] = action_prefix
                        prefix_masks[slot_id] = prefix_mask
                    else:
                        recent_drop_counts[slot_id] += 1

                if submitted.any():
                    teacher_chunks, loss_masks = _teacher_chunks(
                        teacher,
                        rgb_env,
                        branch_env,
                        prefixes,
                        prefix_masks,
                        submitted,
                    )
                    rgb = _uint8_rgb(rgb_observation["rgb"])
                    if student_pool is None:
                        chunks = teacher_chunks
                    else:
                        slots = np.flatnonzero(submitted)
                        observations = []
                        for slot_id in slots:
                            metadata = {
                                MIKASA_IMAGE_VIEWS_INFO_KEY: np.stack((
                                    rgb[slot_id, :, :, :3], rgb[slot_id, :, :, 3:6],
                                )),
                                MIKASA_STATE_INFO_KEY: states[slot_id],
                                "slot_id": int(slot_id),
                            }
                            if args.condition != "zero-latency":
                                metadata["action_prefix"] = prefixes[slot_id]
                                metadata["action_prefix_mask"] = prefix_masks[slot_id]
                            observations.append(Observation(
                                data=None, env_step=frame,
                                sim_time_ms=clock.step_to_time_ms(frame),
                                metadata=metadata,
                            ))
                        predicted = student_pool.predict_batch(observations)
                        chunks = np.zeros_like(teacher_chunks)
                        for slot_id, output in zip(slots, predicted, strict=True):
                            chunks[slot_id] = np.asarray(output.action_chunk, dtype=np.float32)
                else:
                    chunks = np.zeros(
                        (args.num_envs, HORIZON, ACTION_DIM),
                        dtype=np.float32,
                    )
                    loss_masks = np.zeros(
                        (args.num_envs, HORIZON),
                        dtype=bool,
                    )
                    teacher_chunks = chunks

                for slot_id in np.flatnonzero(submitted):
                    sample = samplers[slot_id].sample(
                        {
                            "episode_id": seeds[slot_id],
                            "worker_slot": 0,
                            "slot_id": slot_id,
                            "obs_id": obs_id,
                            "raw_frame": frame,
                            "env_step": frame,
                            "sim_time_ms": clock.step_to_time_ms(frame),
                            "in_flight_count": 0,
                            "idle_worker_count": 1,
                            "recent_drop_count": int(recent_drop_counts[slot_id]),
                        }
                    )
                    ready_frame = frame + sample.action_ready_raw_frames
                    chunk = chunks[slot_id]
                    policy_output = PolicyOutput(
                        action=Action(chunk[0]),
                        action_chunk=[Action(value) for value in chunk],
                    )
                    _, event = build_simulated_action_event(
                        action_id=action_id,
                        obs_id=obs_id,
                        policy_output=policy_output,
                        raw_frame=frame,
                        ready_raw_frame=ready_frame,
                        ready_time_ms=clock.step_to_time_ms(ready_frame),
                        latency_sample=sample,
                        frame_ms=clock.frame_ms,
                        latency_type=latency_type_from_source(samplers[slot_id]),
                        profile_metadata=profile_metadata_from_source(samplers[slot_id]),
                        extra_event_metadata={
                            "episode_id": seeds[slot_id],
                            "slot_id": slot_id,
                            "worker_id": 0,
                        },
                    )
                    timelines[slot_id].submit(
                        obs_id=obs_id,
                        ready_raw_frame=ready_frame,
                        event=event,
                    )
                    capacities[slot_id].submit(
                        0,
                        frame + sample.worker_service_raw_frames,
                    )
                    submitted_requests[slot_id] += 1
                    latency_counts[sample.action_ready_raw_frames] += 1
                    loss_mask = loss_masks[slot_id]
                    if loss_mask.any():
                        row = {
                                "frame": frame,
                                "top": rgb[slot_id, :, :, :3],
                                "wrist": rgb[slot_id, :, :, 3:6],
                                "state": states[slot_id],
                                "chunk": teacher_chunks[slot_id],
                                "prefix_mask": prefix_masks[slot_id],
                                "loss_mask": loss_mask,
                                "latency_raw_frames": sample.action_ready_raw_frames,
                                "latency_ms": sample.latency_ms,
                            }
                        if student_pool is not None:
                            row["student_chunk"] = chunk
                            row["request_index"] = int(submitted_requests[slot_id])
                        rows[slot_id].append(row)
                    action_id += 1
                    obs_id += 1
                    recent_drop_counts[slot_id] = 0

                actions = np.zeros((args.num_envs, ACTION_DIM), dtype=np.float32)
                for slot_id in np.flatnonzero(active):
                    timeline = timelines[slot_id]
                    scheduler = schedulers[slot_id]
                    released, _ = timeline.release_arrived(frame)
                    for event in released:
                        scheduler.enqueue(event)
                    active_action = scheduler.update(
                        frame,
                        clock.step_to_time_ms(frame),
                    )
                    if scheduler.latest_applied_event is not None:
                        applied_heads[slot_id][
                            scheduler.latest_applied_event.metadata["chunk_index"]
                        ] += 1
                    actions[slot_id] = np.asarray(
                        active_action.value,
                        dtype=np.float32,
                    )

                rgb_observation, reward, terminated, truncated, info = rgb_env.step(
                    torch.as_tensor(actions, device=rgb_env.unwrapped.device)
                )
                reward_values = _as_numpy(reward).reshape(args.num_envs)
                terminated_values = (
                    _as_numpy(terminated).reshape(args.num_envs).astype(bool)
                )
                truncated_values = (
                    _as_numpy(truncated).reshape(args.num_envs).astype(bool)
                )
                success_values = (
                    _as_numpy(info["success"]).reshape(args.num_envs).astype(bool)
                )
                episode_returns[active] += reward_values[active]
                success |= active & success_values
                finished = active & (
                    success_values | terminated_values | truncated_values
                )
                lengths[finished] = frame + 1
                active[finished] = False
                if not active.any():
                    break

            valid_action_counts = [
                sum(int(row["loss_mask"].sum()) for row in episode_rows)
                for episode_rows in rows
            ]
            selected = _selected_slots(
                valid_action_counts,
                collected=splits[split]["valid_actions"],
                target=split_targets[split],
            )
            with attempts_path.open("a", encoding="utf-8") as stream:
                for slot_id, seed in enumerate(seeds):
                    attempt = {
                        "seed": seed,
                        "batch_index": batch_index,
                        "slot_id": slot_id,
                        "selected": selected[slot_id],
                        "success": bool(success[slot_id]),
                        "length": int(lengths[slot_id]),
                        "return": float(episode_returns[slot_id]),
                        "split": split,
                        "labels": len(rows[slot_id]),
                        "valid_actions": valid_action_counts[slot_id],
                        "submitted_requests": int(submitted_requests[slot_id]),
                        "dropped_requests": int(
                            lengths[slot_id] - submitted_requests[slot_id]
                        ),
                        "applied_head_counts": dict(
                            sorted(applied_heads[slot_id].items())
                        ),
                    }
                    stream.write(json.dumps(attempt) + "\n")
                    if selected[slot_id]:
                        episode_index = len(saved_episodes)
                        _save_episode(
                            episodes_dir / f"episode_{episode_index:06d}.npz",
                            rows[slot_id],
                            seed=seed,
                        )
                        if student_pool is not None:
                            for row in rows[slot_id]:
                                mask = row["loss_mask"]
                                error = np.abs(
                                    row["student_chunk"][mask] - row["chunk"][mask]
                                ).mean()
                                group = "first" if row["request_index"] == 1 else "later"
                                student_errors[group].append(float(error))
                        episode = {"episode_index": episode_index, **attempt}
                        saved_episodes.append(episode)
                        split_record = splits[split]
                        split_record["episode_indices"].append(episode_index)
                        split_record["seeds"].append(seed)
                        split_record["rows"] += len(rows[slot_id])
                        split_record["valid_actions"] += valid_action_counts[slot_id]
                        split_record[
                            "successes" if success[slot_id] else "failures"
                        ] += 1
                    print(
                        json.dumps(
                            {
                                "saved_episodes": len(saved_episodes),
                                "train_valid_actions": splits["train"]["valid_actions"],
                                "validation_valid_actions": splits["validation"]["valid_actions"],
                                **attempt,
                            }
                        ),
                        flush=True,
                    )
            attempted_seed += args.num_envs
            batch_index += 1
    finally:
        branch_env.close()
        rgb_env.close()

    result = {
        "format": ("mikasa_h8_naive_dagger_raw_v1" if args.condition == "zero-latency"
                   else "mikasa_conditioned_h8_raw_v2"),
        "condition": args.condition,
        "model": args.model,
        "collection_num_envs": args.num_envs,
        "latency_stream_scope": "collector_slot",
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "latency_config": config,
        "action_prefix": prefix_settings,
        "attempted_episodes": attempted_seed - seed_start,
        "seed_start": seed_start,
        "student_control": student_pool is not None,
        "student_source": student_source,
        "student_suffix_mae": {
            key: {"n": len(values), "mean": float(np.mean(values)) if values else None}
            for key, values in student_errors.items()
        },
        "saved_episodes": saved_episodes,
        "splits": splits,
        "latency_raw_frame_counts": dict(sorted(latency_counts.items())),
    }
    (args.output / "collection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("profile-latency", *FIXED_LATENCIES, "zero-latency"), required=True)
    parser.add_argument(
        "--model",
        choices=("qwenoft", "qwenpi_v3", "qwengr00t", "shared"),
        required=True,
    )
    parser.add_argument("--latency-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-targets", type=int, default=64_288)
    parser.add_argument("--validation-targets", type=int, default=6_884)
    parser.add_argument("--num-envs", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args), indent=2))


if __name__ == "__main__":
    main()
