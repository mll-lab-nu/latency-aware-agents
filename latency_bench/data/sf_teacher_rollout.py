"""Collect canonical rollout rows from Sample Factory teacher checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm

from latency_bench.data.action_layouts import action_spec_for_env
from latency_bench.data.filters import (
    FILTER_PRESET_CLEAN_V1,
    episode_passes_clean_v1,
    filter_train_rows_clean_v1,
)
from latency_bench.data.hf_parquet import (
    completed_episode_indices_from_shards,
    episode_row_split_stats,
    write_episode_row_shard,
)
from latency_bench.data.rollout_image_io import RolloutImageWriter, frame_from_env_raw_rgb_info
from latency_bench.data.rollout_schema import SCHEMA_VERSION
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
    final_dir: Path,
    episode_idx: int,
    split: str,
) -> list[dict[str, Any]]:
    """Rewrite accepted attempt rows to their final dataset identity.

    Args:
        rows: Rows collected under the attempt namespace.
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
        finalized["image"] = str(final_dir / f"decision_step_{int(row['decision_step']):06d}.png")
        finalized_rows.append(finalized)
    return finalized_rows


def _online_filter_state_path(output_dir: Path) -> Path:
    """Build the state path used to resume online filtered rollouts.

    Args:
        output_dir: Rollout export directory.

    Returns:
        Path to the online filter state JSON file.
    """
    return output_dir / "online_filter_state.json"


def _new_filter_report() -> dict[str, Any]:
    """Create the rollout filter report accumulator.

    Returns:
        Mutable report counters written to metadata and ``filter_report.json``.
    """
    return {
        "preset": FILTER_PRESET_CLEAN_V1,
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


def _new_online_filter_state() -> dict[str, Any]:
    """Create the online filter state for a fresh export.

    Returns:
        Mutable state persisted after each completed rollout attempt.
    """
    return {
        "next_attempt_idx": 0,
        "filter_report": _new_filter_report(),
    }


def _load_online_filter_state(
    output_dir: Path,
    *,
    completed_episode_indices: set[int],
) -> dict[str, Any]:
    """Load or initialize online filter resume state.

    Args:
        output_dir: Rollout export directory.
        completed_episode_indices: Accepted episode shards already on disk.

    Returns:
        Online filter state with next attempt id and report counters.
    """
    state_path = _online_filter_state_path(output_dir)
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    if completed_episode_indices:
        raise FileNotFoundError(state_path)
    return _new_online_filter_state()


def _write_online_filter_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist online filter resume state.

    Args:
        output_dir: Rollout export directory.
        state: JSON-serializable online filter state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = _online_filter_state_path(output_dir)
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


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


def _write_filtered_episode_images(
    image_writer: RolloutImageWriter,
    rows: list[dict[str, Any]],
    frames_by_decision_step: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    """Write images for the rows that survived online filtering.

    Args:
        image_writer: Image writer shared by the rollout run.
        rows: Final rows selected for materialization.
        frames_by_decision_step: Captured frames keyed by decision step.

    Returns:
        Row copies with image paths written to disk.
    """
    written_rows = []
    for row in rows:
        written = dict(row)
        image_path = Path(written["image"])
        written["image"] = image_writer.write(frames_by_decision_step[int(written["decision_step"])], image_path)
        written_rows.append(written)
    image_writer.flush()
    return written_rows


def _update_online_filter_progress(progress: tqdm, report: Mapping[str, Any]) -> None:
    """Display raw online filter attempt counters in the progress bar.

    Args:
        progress: Active tqdm progress bar.
        report: Filter report counters.
    """
    attempted = int(report["attempted_episodes"])
    accepted = int(report["accepted_episodes"])
    accept_rate = accepted / attempted if attempted else 0.0
    progress.set_postfix(
        attempted=attempted,
        rejected=int(report["rejected_episodes"]),
        accept_rate=f"{accept_rate:.2f}",
    )


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
) -> dict[str, Any]:
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
    Returns:
        Rollout metadata.
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
    completed_episode_indices = completed_episode_indices_from_shards(output_dir)
    final_splits = {
        int(spec["episode_idx"]): str(spec["split"])
        for spec in _build_episode_plan(
            max_episodes=int(max_episodes),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            base_seed=0,
        )
    }
    online_filter_state = _load_online_filter_state(
        output_dir,
        completed_episode_indices=completed_episode_indices,
    )
    filter_report = online_filter_state["filter_report"]

    active_specs: list[dict[str, Any] | None] = [None for _ in range(driver.slot_count)]
    active_obs: list[Any | None] = [None for _ in range(driver.slot_count)]
    active_image_frames: list[np.ndarray | None] = [None for _ in range(driver.slot_count)]
    active_episode_rows: list[list[dict[str, Any]]] = [[] for _ in range(driver.slot_count)]
    active_episode_frames: list[list[np.ndarray]] = [[] for _ in range(driver.slot_count)]
    active_decision_steps = [0 for _ in range(driver.slot_count)]
    rnn_states = torch.zeros([driver.slot_count, rnn_size], dtype=torch.float32, device=device)
    image_writer = RolloutImageWriter(
        max_workers=int(image_writer_workers),
        row_buffer_size=int(row_buffer_size),
        export_image_size=export_image_size,
    )
    image_resolution: list[int] | None = None

    try:
        next_attempt_idx = int(online_filter_state["next_attempt_idx"])
        initial_reset_plan: dict[int, int] = {}
        for slot in range(driver.slot_count):
            if slot >= int(max_episodes) - len(completed_episode_indices):
                break
            spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed))
            next_attempt_idx += 1
            active_specs[slot] = spec
            initial_reset_plan[slot] = int(spec["seed"])
        if initial_reset_plan:
            reset_results = driver.reset_slots(initial_reset_plan, [spec is not None for spec in active_specs])
            for slot, (obs, reset_info) in reset_results.items():
                active_obs[slot] = obs
                active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                active_episode_rows[slot] = []
                active_episode_frames[slot] = []
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
                    image = str(image_path)
                    active_episode_frames[slot].append(np.array(image_frame, copy=True))
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

                    rnn_states[slot] = new_rnn_states[batch_idx]
                    next_decision_step = decision_step + 1
                    if done or next_decision_step >= int(max_steps_per_episode):
                        filter_report["attempted_episodes"] += 1
                        accepted = completed < int(max_episodes) and episode_passes_clean_v1(
                            env_name,
                            active_episode_rows[slot],
                            max_steps_per_episode=int(max_steps_per_episode),
                        )
                        if accepted:
                            final_episode_idx = completed
                            final_split = final_splits[final_episode_idx]
                            final_dir = _episode_image_dir(output_dir, final_split, final_episode_idx)
                            final_rows = _finalize_attempt_rows(
                                active_episode_rows[slot],
                                final_dir=final_dir,
                                episode_idx=final_episode_idx,
                                split=final_split,
                            )
                            if final_split == "train":
                                shard_rows = filter_train_rows_clean_v1(
                                    env_name,
                                    final_rows,
                                    seed=int(spec["seed"]),
                                )
                            else:
                                shard_rows = [dict(row) for row in final_rows]
                            frames_by_decision_step = {
                                int(row["decision_step"]): frame
                                for row, frame in zip(final_rows, active_episode_frames[slot])
                            }
                            shard_rows = _write_filtered_episode_images(
                                image_writer,
                                shard_rows,
                                frames_by_decision_step,
                            )
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
                            active_episode_frames[slot] = []
                            active_decision_steps[slot] = 0
                            rnn_states[slot].zero_()
                        online_filter_state["next_attempt_idx"] = next_attempt_idx
                        _write_online_filter_state(output_dir, online_filter_state)
                        _update_online_filter_progress(progress, filter_report)
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
                        active_episode_frames[slot] = []
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                else:
                    driver.sync_active([spec is not None for spec in active_specs])
    finally:
        try:
            image_writer.close()
        finally:
            driver.close()

    split_stats = episode_row_split_stats(output_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "reward_field": action_spec.reward_field,
        "episode_return_field": action_spec.episode_return_field,
        "reward_semantics": action_spec.reward_semantics,
        "filter_preset": FILTER_PRESET_CLEAN_V1,
        "filter_report": filter_report,
        "checkpoint_train_step": int(runtime["checkpoint"]["train_step"]),
        "frame_stack": frame_stack,
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": obs_stride_raw_frames,
        "image_source": "env_raw_rgb",
        "image_resolution": image_resolution,
        "train_episodes": split_stats["train_episodes"],
        "val_episodes": split_stats["val_episodes"],
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
) -> dict[str, Any]:
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
    Returns:
        Rollout metadata.
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
    )
