"""Collect canonical rollout rows from Sample Factory teacher checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm

from latency_bench.data.action_layouts import action_spec_for_env
from latency_bench.data.filters import (
    episode_passes_preset,
    filter_train_rows_for_preset,
)
from latency_bench.data.hf_parquet import (
    completed_episode_indices_from_shards,
    episode_row_split_stats,
    load_episode_row_shards,
    write_episode_row_shard,
)
from latency_bench.data.rollout_image_io import RolloutImageWriter, frame_from_env_raw_rgb_info
from latency_bench.data.rollout_schema import SCHEMA_VERSION, sort_rows
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.data.sf_teacher_drivers import (
    BatchedTeacherRolloutDriver,
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


def _attempt_episode_spec(*, attempt_idx: int, base_seed: int) -> dict[str, Any]:
    """Build an online rollout attempt spec before final split assignment.

    Args:
        attempt_idx: Zero-based rollout attempt index.
        base_seed: First environment reset seed.

    Returns:
        Episode spec using an attempt-local image namespace.
    """
    return {
        "attempt_idx": attempt_idx,
        "episode_idx": attempt_idx,
        "split": "_attempts",
        "seed": base_seed + attempt_idx,
    }


def _episode_image_dir(output_dir: Path, split: str, episode_idx: int) -> Path:
    """Build the image directory for one rollout episode.

    Args:
        output_dir: Rollout export directory.
        split: Row split or attempt namespace.
        episode_idx: Episode or attempt index.

    Returns:
        Directory containing decision-step PNG files.
    """
    return output_dir / "images" / split / f"episode_{episode_idx:06d}"


def _finalize_attempt_rows(
    rows: list[dict[str, Any]],
    *,
    attempt_dir: Path,
    final_dir: Path,
    episode_idx: int,
    split: str,
) -> list[dict[str, Any]]:
    """Rewrite accepted attempt rows to their final dataset identity.

    Args:
        rows: Rows collected under the attempt namespace.
        attempt_dir: Temporary attempt image directory.
        final_dir: Final split image directory.
        episode_idx: Final accepted episode index.
        split: Final dataset split.

    Returns:
        Final row copies with updated split, episode index, and image path.
    """
    finalized_rows = []
    for row in rows:
        finalized = dict(row)
        finalized["episode_idx"] = episode_idx
        finalized["split"] = split
        finalized["image"] = str(final_dir / Path(row["image"]).relative_to(attempt_dir))
        finalized_rows.append(finalized)
    return finalized_rows


def _new_filter_report(filter_preset: str) -> dict[str, Any]:
    """Create the rollout filter report accumulator.

    Args:
        filter_preset: Empty string for no filtering, or the active preset name.

    Returns:
        Mutable report counters written to metadata and ``filter_report.json``.
    """
    return {
        "preset": filter_preset,
        "attempted_episodes": 0,
        "accepted_episodes": 0,
        "rejected_episodes": 0,
        "accepted_train_episodes": 0,
        "accepted_val_episodes": 0,
        "train_rows_before_row_filter": 0,
        "train_rows_after_row_filter": 0,
        "val_rows_before_row_filter": 0,
        "val_rows_after_row_filter": 0,
    }


def _record_accepted_episode(
    report: dict[str, Any],
    *,
    split: str,
    rows_before_filter: int,
    rows_after_filter: int,
) -> None:
    """Update filter report counters for one accepted episode.

    Args:
        report: Filter report accumulator.
        split: Final dataset split.
        rows_before_filter: Episode rows before train density filtering.
        rows_after_filter: Rows written after split-specific filtering.
    """
    report["accepted_episodes"] += 1
    report[f"accepted_{split}_episodes"] += 1
    report[f"{split}_rows_before_row_filter"] += rows_before_filter
    report[f"{split}_rows_after_row_filter"] += rows_after_filter


def _collect_teacher_dataset_with_driver(
    runtime: Mapping[str, Any],
    cfg: Any,
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
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
    filter_preset: str,
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
        filter_preset: Empty string for no filtering, or the active filter preset.

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
    use_online_filter = filter_preset != ""
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
    final_splits = {
        int(spec["episode_idx"]): str(spec["split"])
        for spec in _build_episode_plan(
            max_episodes=int(max_episodes),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            base_seed=0,
        )
    }
    filter_report = _new_filter_report(str(filter_preset))

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
        next_attempt_idx = len(completed_episode_indices)
        initial_reset_plan: dict[int, int] = {}
        for slot in range(driver.slot_count):
            if use_online_filter:
                if slot >= int(max_episodes) - len(completed_episode_indices):
                    break
                spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed))
                next_attempt_idx += 1
            else:
                if next_spec_idx >= len(planned_episodes):
                    break
                spec = planned_episodes[next_spec_idx]
                next_spec_idx += 1
            active_specs[slot] = spec
            initial_reset_plan[slot] = int(spec["seed"])
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
                    image_path = _episode_image_dir(output_dir, split, int(spec["episode_idx"])) / (
                        f"decision_step_{decision_step:06d}.png"
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
                    if return_rows and not use_online_filter:
                        if split == "val":
                            val_rows.append(row)
                        else:
                            train_rows.append(row)

                    rnn_states[slot] = new_rnn_states[batch_idx]
                    next_decision_step = decision_step + 1
                    if done or next_decision_step >= int(max_steps_per_episode):
                        image_writer.flush()
                        if use_online_filter:
                            filter_report["attempted_episodes"] += 1
                            accepted = completed < int(max_episodes) and episode_passes_preset(
                                str(filter_preset),
                                env_name,
                                active_episode_rows[slot],
                                max_steps_per_episode=int(max_steps_per_episode),
                            )
                            if accepted:
                                final_episode_idx = completed
                                final_split = final_splits[final_episode_idx]
                                attempt_dir = _episode_image_dir(output_dir, "_attempts", int(spec["episode_idx"]))
                                final_dir = _episode_image_dir(output_dir, final_split, final_episode_idx)
                                final_dir.parent.mkdir(parents=True, exist_ok=True)
                                attempt_dir.rename(final_dir)
                                final_rows = _finalize_attempt_rows(
                                    active_episode_rows[slot],
                                    attempt_dir=attempt_dir,
                                    final_dir=final_dir,
                                    episode_idx=final_episode_idx,
                                    split=final_split,
                                )
                                if final_split == "train":
                                    shard_rows = filter_train_rows_for_preset(
                                        str(filter_preset),
                                        env_name,
                                        final_rows,
                                        seed=int(spec["seed"]),
                                    )
                                    if return_rows:
                                        train_rows.extend(shard_rows)
                                else:
                                    shard_rows = [dict(row) for row in final_rows]
                                    if return_rows:
                                        val_rows.extend(shard_rows)
                                write_episode_row_shard(output_dir, shard_rows)
                                _record_accepted_episode(
                                    filter_report,
                                    split=final_split,
                                    rows_before_filter=len(final_rows),
                                    rows_after_filter=len(shard_rows),
                                )
                                progress.update(1)
                                completed += 1
                            else:
                                filter_report["rejected_episodes"] += 1
                                attempt_dir = _episode_image_dir(output_dir, "_attempts", int(spec["episode_idx"]))
                                if attempt_dir.exists():
                                    shutil.rmtree(attempt_dir)
                            if completed < int(max_episodes):
                                next_spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed))
                                next_attempt_idx += 1
                                active_specs[slot] = next_spec
                                reset_plan[slot] = int(next_spec["seed"])
                            else:
                                active_specs[slot] = None
                                active_obs[slot] = None
                                active_image_frames[slot] = None
                                active_episode_rows[slot] = []
                                active_decision_steps[slot] = 0
                                rnn_states[slot].zero_()
                        else:
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
        attempts_dir = output_dir / "images" / "_attempts"
        if attempts_dir.exists():
            shutil.rmtree(attempts_dir)

    if not use_online_filter:
        split_stats = episode_row_split_stats(output_dir)
        filter_report["attempted_episodes"] = split_stats["total_episodes"]
        filter_report["accepted_episodes"] = split_stats["total_episodes"]
        filter_report["accepted_train_episodes"] = split_stats["train_episodes"]
        filter_report["accepted_val_episodes"] = split_stats["val_episodes"]
        filter_report["train_rows_before_row_filter"] = split_stats["train_rows"]
        filter_report["train_rows_after_row_filter"] = split_stats["train_rows"]
        filter_report["val_rows_before_row_filter"] = split_stats["val_rows"]
        filter_report["val_rows_after_row_filter"] = split_stats["val_rows"]

    return sort_rows(train_rows), sort_rows(val_rows), {
        "schema_version": SCHEMA_VERSION,
        "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "reward_field": action_spec.reward_field,
        "episode_return_field": action_spec.episode_return_field,
        "reward_semantics": action_spec.reward_semantics,
        "filter_preset": str(filter_preset),
        "filter_report": filter_report,
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
    filter_preset: str = "",
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
        filter_preset: Empty string for no filtering, or the active filter preset.
        return_rows: Whether to keep rows in memory while sharding.

    Returns:
        Train rows, validation rows, and rollout metadata.
    """
    env_name = str(config["env_name"])
    with sample_factory_init_log(output_dir):
        module, cfg = build_sample_factory_rollout_cfg(config)
        frame_stack = cfg.frame_stack
        setattr(cfg, "export_env_raw_rgb_frames", True)
        use_batched_driver = (
            (env_name == "flappy" and int(num_envs) > 1 and cfg.use_gpu_render)
            or (env_name == "demon_attack" and cfg.simulator == "gpu")
        )
        if use_batched_driver:
            if env_name == "flappy":
                maybe_set_cli_override(cfg, "gpu_render_batch_size", int(num_envs))
            else:
                maybe_set_cli_override(cfg, "gpu_env_batch_size", int(num_envs))
            runtime = build_env_and_policy(module, cfg, config, 1)
        else:
            runtime = build_env_and_policy(module, cfg, config, num_envs)

    if use_batched_driver:
        driver = BatchedTeacherRolloutDriver(runtime["envs"][0], int(num_envs))
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
        filter_preset=str(filter_preset),
    )
