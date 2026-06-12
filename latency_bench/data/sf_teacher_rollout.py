"""Collect canonical rollout rows from Sample Factory teacher checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm

from latency_bench.data.action_layouts import action_spec_for_env
from latency_bench.data.hf_parquet import (
    completed_episode_indices_from_shards,
    load_episode_row_shards,
    write_episode_row_shard,
)
from latency_bench.data.rollout_image_io import RolloutImageWriter, frame_from_env_raw_rgb_info
from latency_bench.data.rollout_schema import SCHEMA_VERSION, sort_rows
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.data.sf_teacher_drivers import (
    FlappyBatchedTeacherRolloutDriver,
    SerialTeacherRolloutDriver,
    batch_observations,
)
from latency_bench.data.sf_teacher_runtime import (
    build_env_and_policy,
    build_sample_factory_rollout_cfg,
    sample_factory_init_log,
)
from latency_bench.policy.sample_factory_common import (
    _deterministic_actions_for_space,
    _normalize_actions_for_space,
    _split_batch_actions,
)
from training.common.utils import maybe_set_cli_override


def _build_episode_plan(
    *,
    max_episodes: int,
    val_fraction: float,
    split_seed: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Create deterministic episode split and reset seed specs.

    Args:
        max_episodes: Number of episodes to collect.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        base_seed: First environment reset seed.

    Returns:
        Episode specs with index, split, and reset seed.
    """
    val_count = int(round(int(max_episodes) * float(val_fraction)))
    val_indices = set(np.random.default_rng(int(split_seed)).choice(int(max_episodes), size=val_count, replace=False).tolist())
    return [
        {
            "episode_idx": idx,
            "split": "val" if idx in val_indices else "train",
            "seed": int(base_seed) + idx,
        }
        for idx in range(int(max_episodes))
    ]


def _collect_teacher_dataset_with_driver(
    runtime: Mapping[str, Any],
    cfg: Any,
    driver: SerialTeacherRolloutDriver | FlappyBatchedTeacherRolloutDriver,
    *,
    env_name: str,
    output_dir: Path,
    max_episodes: int,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    frame_stack: int,
    return_rows: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Collect teacher rollout rows using a concrete rollout driver.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        cfg: Sample Factory rollout cfg.
        driver: Driver that abstracts serial vs batched env stepping.
        env_name: Canonical benchmark environment name.
        output_dir: Rollout export directory.
        max_episodes: Number of episodes to collect.
        max_steps_per_episode: Step cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        frame_stack: Teacher checkpoint frame stack, stored as metadata.
        return_rows: Whether to keep rows in memory while sharding.

    Returns:
        Train rows, validation rows, and rollout metadata.
    """

    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    argmax_actions = runtime["argmax_actions"]
    preprocess_actions = runtime["preprocess_actions"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
    rnn_size = runtime["rnn_size"]
    action_spec = action_spec_for_env(env_name)
    env_fps = float(cfg.env_fps)
    obs_fps = float(cfg.obs_fps)
    obs_stride_raw_frames = int(env_fps / obs_fps)
    planned_episodes = _build_episode_plan(
        max_episodes=int(max_episodes),
        val_fraction=float(val_fraction),
        split_seed=int(split_seed),
        base_seed=int(cfg.seed),
    )
    if return_rows:
        train_rows, val_rows, completed_episode_indices = load_episode_row_shards(output_dir)
    else:
        train_rows = []
        val_rows = []
        completed_episode_indices = completed_episode_indices_from_shards(output_dir)
    planned_episodes = [spec for spec in planned_episodes if int(spec["episode_idx"]) not in completed_episode_indices]

    active_specs: list[dict[str, Any] | None] = [None for _ in range(driver.slot_count)]
    active_obs: list[Any | None] = [None for _ in range(driver.slot_count)]
    active_image_frames: list[np.ndarray | None] = [None for _ in range(driver.slot_count)]
    active_episode_rows: list[list[dict[str, Any]]] = [[] for _ in range(driver.slot_count)]
    active_decision_steps = [0 for _ in range(driver.slot_count)]
    rnn_states = torch.zeros([driver.slot_count, rnn_size], dtype=torch.float32, device=device)
    image_writer = RolloutImageWriter(
        max_workers=int(image_writer_workers),
        row_buffer_size=int(row_buffer_size),
        export_image_size=export_image_size,
    )
    image_resolution: list[int] | None = None

    try:
        next_spec_idx = 0
        initial_reset_plan: dict[int, int] = {}
        for slot in range(driver.slot_count):
            if next_spec_idx >= len(planned_episodes):
                break
            spec = planned_episodes[next_spec_idx]
            active_specs[slot] = spec
            initial_reset_plan[slot] = int(spec["seed"])
            next_spec_idx += 1
        if initial_reset_plan:
            reset_results = driver.reset_slots(initial_reset_plan, [spec is not None for spec in active_specs])
            for slot, (obs, reset_info) in reset_results.items():
                active_obs[slot] = obs
                active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                active_episode_rows[slot] = []
                active_decision_steps[slot] = 0
                rnn_states[slot].zero_()

        completed = len(completed_episode_indices)
        with tqdm(total=int(max_episodes), initial=completed, unit="episode") as progress:
            while completed < int(max_episodes):
                active_slots = [slot for slot, spec in enumerate(active_specs) if spec is not None]
                if not active_slots:
                    break
                obs_batch = batch_observations([active_obs[slot] for slot in active_slots])
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs_batch)
                with torch.no_grad():
                    policy_outputs = actor_critic(normalized_obs, rnn_states[active_slots])
                    actions = policy_outputs["actions"]
                    if deterministic:
                        actions = _deterministic_actions_for_space(
                            actor_critic.action_distribution(),
                            action_space=driver.action_space,
                            argmax_actions=argmax_actions,
                            device=device,
                        )
                    actions = _normalize_actions_for_space(
                        actions,
                        batch_size=len(active_slots),
                        action_space=driver.action_space,
                        device=device,
                        unsqueeze_tensor=unsqueeze_tensor,
                    )
                    processed_actions = preprocess_actions(runtime["env_info"], actions)
                    raw_actions = _split_batch_actions(
                        processed_actions,
                        len(active_slots),
                        action_space=driver.action_space,
                    )
                    new_rnn_states = policy_outputs["new_rnn_states"]

                step_results = driver.step(active_slots, raw_actions)
                reset_plan: dict[int, int] = {}

                for batch_idx, slot in enumerate(active_slots):
                    spec = active_specs[slot]
                    split = str(spec["split"])
                    decision_step = int(active_decision_steps[slot])
                    image_path = (
                        output_dir
                        / "images"
                        / split
                        / f"episode_{int(spec['episode_idx']):06d}"
                        / f"decision_step_{decision_step:06d}.png"
                    )
                    image_frame = active_image_frames[slot]
                    if image_resolution is None:
                        if export_image_size is not None:
                            width, height = export_image_size
                            image_resolution = [height, width, image_frame.shape[2]]
                        else:
                            image_resolution = list(image_frame.shape)
                    image = image_writer.write(image_frame, image_path)
                    raw_action = raw_actions[batch_idx]
                    step_result = step_results[slot]
                    done = bool(step_result.terminated) or bool(step_result.truncated)
                    row = teacher_dataset_row(
                        env_name=env_name,
                        spec=spec,
                        decision_step=decision_step,
                        image=image,
                        raw_action=raw_action,
                        info=step_result.info,
                        reward=step_result.reward,
                        env_fps=env_fps,
                        obs_fps=obs_fps,
                    )
                    active_episode_rows[slot].append(row)
                    if return_rows:
                        if split == "val":
                            val_rows.append(row)
                        else:
                            train_rows.append(row)

                    rnn_states[slot] = new_rnn_states[batch_idx]
                    next_decision_step = decision_step + 1
                    if done or next_decision_step >= int(max_steps_per_episode):
                        image_writer.flush()
                        write_episode_row_shard(output_dir, active_episode_rows[slot])
                        progress.update(1)
                        completed += 1
                        if next_spec_idx < len(planned_episodes):
                            next_spec = planned_episodes[next_spec_idx]
                            active_specs[slot] = next_spec
                            reset_plan[slot] = int(next_spec["seed"])
                            next_spec_idx += 1
                        else:
                            active_specs[slot] = None
                            active_obs[slot] = None
                            active_image_frames[slot] = None
                            active_episode_rows[slot] = []
                            active_decision_steps[slot] = 0
                            rnn_states[slot].zero_()
                    else:
                        active_obs[slot] = step_result.next_obs
                        active_image_frames[slot] = frame_from_env_raw_rgb_info(step_result.info)
                        active_decision_steps[slot] = next_decision_step

                if reset_plan:
                    reset_results = driver.reset_slots(reset_plan, [spec is not None for spec in active_specs])
                    for slot, (obs, reset_info) in reset_results.items():
                        active_obs[slot] = obs
                        active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                        active_episode_rows[slot] = []
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                else:
                    driver.sync_active([spec is not None for spec in active_specs])
    finally:
        try:
            image_writer.close()
        finally:
            driver.close()

    return sort_rows(train_rows), sort_rows(val_rows), {
        "schema_version": SCHEMA_VERSION,
        "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "checkpoint_train_step": int(runtime["checkpoint"]["train_step"]),
        "frame_stack": frame_stack,
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": obs_stride_raw_frames,
        "image_source": "env_raw_rgb",
        "image_resolution": image_resolution,
        "train_episodes": len({row["episode_idx"] for row in train_rows}),
        "val_episodes": len({row["episode_idx"] for row in val_rows}),
    }


def collect_teacher_dataset(
    config: dict[str, Any],
    *,
    output_dir: Path,
    max_episodes: int,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    num_envs: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    return_rows: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Collect a Sample Factory teacher dataset from a checkpoint descriptor.

    Args:
        config: Rollout descriptor loaded from a checkpoint.
        output_dir: Rollout export directory.
        max_episodes: Number of episodes to collect.
        max_steps_per_episode: Step cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        num_envs: Number of rollout environments or agents.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        return_rows: Whether to keep rows in memory while sharding.

    Returns:
        Train rows, validation rows, and rollout metadata.
    """
    env_name = str(config["env_name"])
    with sample_factory_init_log(output_dir):
        module, cfg = build_sample_factory_rollout_cfg(config)
        frame_stack = cfg.frame_stack
        setattr(cfg, "export_env_raw_rgb_frames", True)
        use_flappy_batched_driver = str(env_name) == "flappy" and int(num_envs) > 1 and bool(cfg.use_gpu_render)
        if use_flappy_batched_driver:
            maybe_set_cli_override(cfg, "gpu_render_batch_size", int(num_envs))
            runtime = build_env_and_policy(module, cfg, config, 1)
        else:
            runtime = build_env_and_policy(module, cfg, config, num_envs)

    if use_flappy_batched_driver:
        driver = FlappyBatchedTeacherRolloutDriver(runtime["envs"][0], int(num_envs))
    else:
        driver = SerialTeacherRolloutDriver(runtime["envs"])
    return _collect_teacher_dataset_with_driver(
        runtime,
        cfg,
        driver,
        env_name=env_name,
        output_dir=output_dir,
        max_episodes=int(max_episodes),
        max_steps_per_episode=int(max_steps_per_episode),
        val_fraction=float(val_fraction),
        split_seed=int(split_seed),
        deterministic=bool(deterministic),
        export_image_size=export_image_size,
        image_writer_workers=int(image_writer_workers),
        row_buffer_size=int(row_buffer_size),
        frame_stack=int(frame_stack),
        return_rows=bool(return_rows),
    )
