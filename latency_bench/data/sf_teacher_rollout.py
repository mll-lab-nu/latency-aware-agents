"""Collect canonical rollout rows from Sample Factory teacher checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from latency_bench.core.decision_action_history import DecisionActionHistory
from latency_bench.data.action_layouts import ActionSpec
from latency_bench.data.filters import (
    FILTER_PRESET_CLEAN_V1,
    filter_train_rows,
)
from latency_bench.data.hf_parquet import (
    completed_episode_indices_from_shards,
    write_episode_row_shard,
)
from latency_bench.data.ghost_trail import GhostTrailConfig, DemonGhostTrailStream
from latency_bench.data.rollout_image_io import (
    RolloutImageWriter, build_episode_image_frames,
    frame_from_env_raw_rgb_info, frames_from_env_raw_rgb_info,
)
from latency_bench.data.rollout_rows import teacher_dataset_row
from latency_bench.data import sf_teacher_probe as probe
from latency_bench.data.sf_teacher_drivers import (
    BatchedTeacherRolloutDriver,
    SerialTeacherRolloutDriver,
)
from latency_bench.data import sf_teacher_runtime as teacher_runtime


def _g1_issued_absolute_request(
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
    slot: int,
    action: Any,
) -> np.ndarray:
    """Return the absolute request issued for the current policy decision."""
    env = driver.envs[slot] if isinstance(driver, SerialTeacherRolloutDriver) else driver.control_env
    native = env.unwrapped
    control_mode = native._control_mode
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if control_mode == "raw_body_target":
        return native.g1_absolute_request_from_raw_action(action)
    return action.copy()


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


def _context_decision_steps(
    decision_step: int,
    context_window: int,
    available_decision_steps: Sequence[int],
) -> list[int]:
    """Build historical decision steps from the current contiguous replay segment.

    Args:
        decision_step: Current decision step.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.
        available_decision_steps: Materialized decision steps ordered from oldest to newest.

    Returns:
        Historical decision steps ordered from oldest to newest.
    """
    current_index = available_decision_steps.index(decision_step)
    segment_start_index = current_index
    while segment_start_index > 0:
        current_step = available_decision_steps[segment_start_index]
        previous_step = available_decision_steps[segment_start_index - 1]
        if previous_step != current_step - 1:
            break
        segment_start_index -= 1

    context_steps: list[int] = []
    cursor = current_index - 1
    while cursor >= segment_start_index and len(context_steps) < context_window - 1:
        context_steps.append(available_decision_steps[cursor])
        cursor -= 1

    first_segment_step = available_decision_steps[segment_start_index]
    while len(context_steps) < context_window - 1:
        context_steps.append(first_segment_step)

    return list(reversed(context_steps))


def _image_path_for_decision_step(row: Mapping[str, Any], decision_step: int) -> Path:
    return Path(str(row["image"])).with_name(f"decision_step_{int(decision_step):06d}.png")


def _write_filtered_episode_images(
    image_writer: RolloutImageWriter,
    rows: list[dict[str, Any]],
    frames_by_decision_step: Mapping[int, np.ndarray],
    *,
    context_window: int,
) -> list[dict[str, Any]]:
    """Write current and optional historical images for filtered rollout rows.

    Args:
        image_writer: Image writer shared by the rollout run.
        rows: Final rows selected for materialization.
        frames_by_decision_step: Captured frames keyed by decision step.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.

    Returns:
        Row copies with image paths written to disk.
    """
    if not rows:
        return []
    available_decision_steps = sorted(frames_by_decision_step)
    required_steps: set[int] = set()
    for row in rows:
        decision_step = int(row["decision_step"])
        required_steps.add(decision_step)
        if int(context_window) > 1:
            required_steps.update(
                _context_decision_steps(
                    decision_step,
                    int(context_window),
                    available_decision_steps,
                )
            )

    written_paths_by_step: dict[int, str] = {}
    first_row = rows[0]
    for decision_step in sorted(required_steps):
        image_path = _image_path_for_decision_step(first_row, decision_step)
        written_paths_by_step[decision_step] = image_writer.write(frames_by_decision_step[decision_step], image_path)
    image_writer.flush()

    written_rows = []
    for row in rows:
        written = dict(row)
        decision_step = int(written["decision_step"])
        written["image"] = written_paths_by_step[decision_step]
        if int(context_window) > 1:
            written["context_images"] = [
                written_paths_by_step[context_step]
                for context_step in _context_decision_steps(
                    decision_step,
                    int(context_window),
                    available_decision_steps,
                )
            ]
        written_rows.append(written)
    return written_rows


def _replay_accepted_specs(
    runtime: Mapping[str, Any],
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
    *,
    cfg: Any,
    env_name: str,
    output_dir: Path,
    accepted_specs: list[dict[str, Any]],
    completed_episode_indices: set[int],
    max_steps_per_episode: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    context_window: int = 1,
    image_transform_config: GhostTrailConfig | None = None,
    action_spec: ActionSpec | None = None,
    base_prompt: str | None = None,
    integration_name: str | None = None,
    export_zero_action_history: int | None = None,
) -> None:
    """Replay accepted specs with raw RGB export and write episode shards.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        driver: Driver that abstracts serial vs batched env stepping.
        cfg: Sample Factory rollout cfg.
        env_name: Canonical benchmark environment name.
        output_dir: Rollout export directory.
        accepted_specs: Accepted probe specs.
        completed_episode_indices: Episode indices already written as shards.
        max_steps_per_episode: Raw environment frame cap per episode.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        filter_preset: Filter preset name for train row density control.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.
        image_transform_config: Optional causal image transform applied before
            row filtering. Defaults to raw RGB.
        action_spec: Explicit action layout carried by a generic checkpoint.
        base_prompt: Explicit task prompt carried by a generic checkpoint.
        integration_name: Integration identity used for legacy-specific rollout behavior.
        export_zero_action_history: Export-only history; the teacher retains its
            original observation and network input contract.
    """
    image_transform_config = GhostTrailConfig() if image_transform_config is None else image_transform_config
    behavior_env_name = env_name if integration_name is None else integration_name
    specs_to_replay = [
        spec for spec in accepted_specs if int(spec["episode_idx"]) not in completed_episode_indices
    ]
    if not specs_to_replay:
        driver.close()
        return None

    rnn_size = runtime["rnn_size"]
    device = runtime["device"]
    env_fps = float(cfg.env_fps)
    obs_fps = float(cfg.obs_fps)
    obs_stride_raw_frames = int(env_fps / obs_fps)
    include_state = integration_name == "gymnasium" and bool(cfg.gym_state_labels_json)
    history_decisions = getattr(cfg, "action_history_decisions", None)
    include_decision_history = (
        export_zero_action_history is None
        and (integration_name in {"flappy", "demon_attack", "deadly_corridor"}
             or env_name in teacher_runtime._GYMNASIUM_ATARI_GPU_ENVS)
        and history_decisions is not None
    )
    zero_history = None
    if export_zero_action_history is not None:
        if cfg.latency_type != "fixed" or cfg.fixed_latency_ms != 0:
            raise ValueError("Export-only zero action history requires fixed latency 0 ms")
        zero_history = DecisionActionHistory(
            driver.action_space, num_envs=driver.slot_count, decisions=export_zero_action_history,
        )
    uses_g1_sonic_contract = env_name.startswith("g1_sonic_")
    state_info_key = "gymnasium_state" if include_state else None
    spec_cursor = 0
    replayed = 0

    # For demon_attack_ghost_trail: accumulate consecutive raw ALE frames externally.
    # frame_stack=4 gives 4 new raw frames per driver.step(); keep a maxlen=30
    # stream so each decision step sees the last 30 consecutive raw ALE frames at 60 fps.
    #
    # Efficiency: DemonGhostTrailStream maintains a parallel deque of _FrameCache objects so that
    # foreground_components (cv2.connectedComponents) is called exactly once per new
    # raw frame (4× per decision step) rather than 30× per decision step.
    _is_demon_ghost_trail = (
        behavior_env_name == "demon_attack"
        and image_transform_config.image_transform == "demon_attack_ghost_trail"
    )
    _ghost_trail_history_len = image_transform_config.history_frames


    active_specs: list[dict[str, Any] | None] = [None for _ in range(driver.slot_count)]
    active_obs: list[Any | None] = [None for _ in range(driver.slot_count)]
    active_image_frames: list[np.ndarray | None] = [None for _ in range(driver.slot_count)]
    active_states: list[np.ndarray | None] = [None for _ in range(driver.slot_count)]
    active_episode_rows: list[list[dict[str, Any]]] = [[] for _ in range(driver.slot_count)]
    active_episode_frames: list[list[np.ndarray]] = [[] for _ in range(driver.slot_count)]
    active_decision_steps = [0 for _ in range(driver.slot_count)]
    active_ghost_streams: list[Any | None] = [None for _ in range(driver.slot_count)]
    rnn_states = torch.zeros([driver.slot_count, rnn_size], dtype=torch.float32, device=device)
    image_writer = RolloutImageWriter(
        max_workers=int(image_writer_workers),
        row_buffer_size=int(row_buffer_size),
        export_image_size=export_image_size,
    )

    def assign_specs_to_slots(slots: list[int]) -> dict[int, int]:
        nonlocal spec_cursor
        reset_plan: dict[int, int] = {}
        for slot in slots:
            if spec_cursor >= len(specs_to_replay):
                active_specs[slot] = None
                continue
            spec = specs_to_replay[spec_cursor]
            spec_cursor += 1
            active_specs[slot] = spec
            reset_plan[slot] = int(spec["seed"])
        return reset_plan

    try:
        initial_slots = list(range(driver.slot_count))
        initial_reset_plan = assign_specs_to_slots(initial_slots)
        if initial_reset_plan:
            if isinstance(driver, BatchedTeacherRolloutDriver):
                reset_results = driver.reset_slots(initial_reset_plan, [spec is not None for spec in active_specs])
            else:
                reset_results = driver.reset_slots(initial_reset_plan)
            for slot, (obs, reset_info) in reset_results.items():
                active_obs[slot] = obs
                if state_info_key is not None:
                    active_states[slot] = reset_info[state_info_key]
                if _is_demon_ghost_trail:
                    active_ghost_streams[slot] = DemonGhostTrailStream(_ghost_trail_history_len)
                    active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info).copy()
                else:
                    active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                active_episode_rows[slot] = []
                active_episode_frames[slot] = []
                active_decision_steps[slot] = 0
                rnn_states[slot].zero_()

        with tqdm(total=len(specs_to_replay), initial=0, unit="episode") as progress:
            while replayed < len(specs_to_replay):
                active_slots = [slot for slot, spec in enumerate(active_specs) if spec is not None]
                if not active_slots:
                    break
                pre_decision_histories = (
                    {
                        slot: np.asarray(active_obs[slot]["transport"], dtype=np.float32)
                        .reshape(-1).copy()
                        for slot in active_slots
                    }
                    if include_decision_history else {}
                )
                if zero_history is not None:
                    histories = zero_history.observation()
                    pre_decision_histories = {slot: histories[slot] for slot in active_slots}
                raw_actions, new_rnn_states, step_results = teacher_runtime.step_teacher_policy(
                    runtime,
                    driver,
                    active_slots=active_slots,
                    active_obs=active_obs,
                    rnn_states=rnn_states,
                    deterministic=deterministic,
                    preserve_leading_batch=behavior_env_name == "gymnasium",
                    preserve_issued_actions=uses_g1_sonic_contract,
                )
                if zero_history is not None:
                    zero_history.append(active_slots, np.ones(len(active_slots)), raw_actions, raw_actions)
                reset_plan: dict[int, int] = {}

                for batch_idx, slot in enumerate(active_slots):
                    spec = active_specs[slot]
                    decision_step = int(active_decision_steps[slot])
                    final_dir = _episode_image_dir(output_dir, str(spec["split"]), int(spec["episode_idx"]))
                    image_path = final_dir / f"decision_step_{decision_step:06d}.png"
                    image_frame = active_image_frames[slot]
                    step_result = step_results[slot]
                    next_decision_step = decision_step + 1
                    life_loss_boundary, episode_done = teacher_runtime.rollout_step_flags(
                        behavior_env_name,
                        step_result,
                        next_raw_frame=next_decision_step * obs_stride_raw_frames,
                        max_steps_per_episode=int(max_steps_per_episode),
                    )
                    active_episode_frames[slot].append(np.array(image_frame, copy=True))
                    row = teacher_dataset_row(
                        env_name=env_name,
                        spec=spec,
                        decision_step=decision_step,
                        image=str(image_path),
                        raw_action=(
                            _g1_issued_absolute_request(driver, slot, raw_actions[batch_idx])
                            if uses_g1_sonic_contract else raw_actions[batch_idx]
                        ),
                        info=step_result.info | {
                            "raw_done": step_result.terminated or step_result.truncated,
                        },
                        reward=step_result.reward,
                        env_fps=env_fps,
                        obs_fps=obs_fps,
                        image_transform=image_transform_config.image_transform,
                        action_spec=action_spec,
                        base_prompt=base_prompt,
                        state=pre_decision_histories[slot] if include_decision_history or zero_history is not None else active_states[slot],
                        prompt_override=base_prompt if include_decision_history or zero_history is not None else None,
                    )
                    active_episode_rows[slot].append(row)
                    rnn_states[slot] = new_rnn_states[batch_idx]
                    if episode_done:
                        if len(active_episode_rows[slot]) != int(spec["episode_length"]):
                            raise RuntimeError(
                                f"Probe/replay length mismatch for seed={spec['seed']}: "
                                f"probe={spec['episode_length']} replay={len(active_episode_rows[slot])}"
                            )
                        final_rows = _finalize_attempt_rows(
                            active_episode_rows[slot],
                            final_dir=final_dir,
                            episode_idx=int(spec["episode_idx"]),
                            split=str(spec["split"]),
                        )
                        if (
                            uses_g1_sonic_contract
                            and final_rows
                            and "command_admitted" in final_rows[0]
                        ):
                            # Keep the raw decision trace intact.  Admission is
                            # a LeRobot anchor policy, not a replay-row filter.
                            shard_rows = [dict(row) for row in final_rows]
                        elif str(spec["split"]) == "train":
                            shard_rows = filter_train_rows(
                                filter_preset,
                                behavior_env_name,
                                final_rows,
                                seed=int(spec["seed"]),
                            )
                        else:
                            shard_rows = [dict(row) for row in final_rows]
                        if _is_demon_ghost_trail:
                            # Frames are pre-built ghost trail images; map directly.
                            ep_steps = [int(row["decision_step"]) for row in final_rows]
                            frames_by_decision_step = {
                                step: frame
                                for step, frame in zip(ep_steps, active_episode_frames[slot])
                            }
                        else:
                            frames_by_decision_step = build_episode_image_frames(
                                env_name=behavior_env_name,
                                rows=final_rows,
                                frames=active_episode_frames[slot],
                                config=image_transform_config,
                            )
                        shard_rows = _write_filtered_episode_images(
                            image_writer,
                            shard_rows,
                            frames_by_decision_step,
                            context_window=int(context_window),
                        )
                        write_episode_row_shard(output_dir, shard_rows)
                        progress.update(1)
                        replayed += 1
                        replacement_plan = assign_specs_to_slots([slot])
                        reset_plan.update(replacement_plan)
                        active_obs[slot] = None
                        active_image_frames[slot] = None
                        active_states[slot] = None
                        active_episode_rows[slot] = []
                        active_episode_frames[slot] = []
                        active_decision_steps[slot] = 0
                        if _is_demon_ghost_trail:
                            active_ghost_streams[slot] = None
                        rnn_states[slot].zero_()
                    else:
                        if life_loss_boundary:
                            if zero_history is not None:
                                zero_history.reset([slot])
                            active_obs[slot], reset_info = driver.continue_after_life_loss(slot)
                            if state_info_key is not None:
                                active_states[slot] = reset_info[state_info_key]
                            if _is_demon_ghost_trail:
                                active_ghost_streams[slot].clear()
                                active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info).copy()
                            else:
                                active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                            rnn_states[slot].zero_()
                        else:
                            active_obs[slot] = step_result.next_obs
                            if state_info_key is not None:
                                active_states[slot] = step_result.info[state_info_key]
                            if _is_demon_ghost_trail:
                                new_raw_frames = frames_from_env_raw_rgb_info(step_result.info)
                                for f in new_raw_frames:
                                    active_ghost_streams[slot].append(f)
                                active_image_frames[slot] = active_ghost_streams[slot].render()
                            else:
                                active_image_frames[slot] = frame_from_env_raw_rgb_info(step_result.info)
                        active_decision_steps[slot] = next_decision_step

                if reset_plan:
                    if zero_history is not None:
                        zero_history.reset(list(reset_plan))
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        reset_results = driver.reset_slots(reset_plan, [spec is not None for spec in active_specs])
                    else:
                        reset_results = driver.reset_slots(reset_plan)
                    for slot, (obs, reset_info) in reset_results.items():
                        active_obs[slot] = obs
                        if state_info_key is not None:
                            active_states[slot] = reset_info[state_info_key]
                        if _is_demon_ghost_trail:
                            active_ghost_streams[slot] = DemonGhostTrailStream(_ghost_trail_history_len)
                            active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info).copy()
                        else:
                            active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                        active_episode_rows[slot] = []
                        active_episode_frames[slot] = []
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                else:
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        driver.sync_active([spec is not None for spec in active_specs])
    finally:
        try:
            image_writer.close()
        finally:
            driver.close()


def collect_teacher_dataset(
    config: dict[str, Any],
    *,
    output_dir: Path,
    max_episodes: int,
    max_attempts: int | None = None,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    num_envs: int,
    replay_num_envs: int = 16,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    demon_episode_threshold: float | None = None,
    episode_return_gt: float | None = None,
    episode_acceptance: Mapping[str, Any] | None = None,
    context_window: int = 1,
    rollout_seed_offset: int = 0,
    probe_only: bool = False,
    image_transform_config: GhostTrailConfig | None = None,
) -> dict[str, Any]:
    """Collect a Sample Factory teacher dataset from a checkpoint descriptor.

    Args:
        config: Rollout descriptor loaded from a checkpoint.
        output_dir: Rollout export directory.
        max_episodes: Number of episodes to collect.
        max_steps_per_episode: Raw environment frame cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        num_envs: Number of probe rollout environments or agents.
        replay_num_envs: Number of replay environments or agents used while writing rows/images.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        filter_preset: Filter preset name (``"none"`` or ``"clean_v1"``).
        demon_episode_threshold: Optional Demon raw-score threshold.
        episode_return_gt: Optional strict lower bound for accepted episode returns.
        episode_acceptance: Explicit task-specific episode acceptance contract.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.
        image_transform_config: Optional causal image transform applied during
            replay before row filtering.
    Returns:
        Persisted probe decisions and runtime metadata; replay writes episode shards.
    """
    env_name = str(config["env_name"])
    integration_name = config["integration_name"]
    action_spec = config["action_spec"]
    base_prompt = config["base_prompt"]
    image_transform_config = GhostTrailConfig() if image_transform_config is None else image_transform_config
    completed_episode_indices = completed_episode_indices_from_shards(output_dir)
    probe_state = probe.load_probe_state(
        output_dir,
        completed_episode_indices=completed_episode_indices,
    )

    if len(probe_state["accepted_specs"]) < int(max_episodes):
        probe_cfg, probe_runtime, probe_driver, frame_stack = teacher_runtime.build_rollout_driver_runtime(
            config,
            output_dir=output_dir,
            num_envs=int(num_envs),
            export_env_raw_rgb_frames=False,
        )
        probe_state["runtime_metadata"] = {
            "checkpoint_train_step": int(probe_runtime["checkpoint"]["train_step"]),
            "frame_stack": int(frame_stack),
            "env_fps": float(probe_cfg.env_fps),
            "obs_fps": float(probe_cfg.obs_fps),
        }
        probe.write_probe_state(output_dir, probe_state)
        probe.probe_accepted_specs(
            probe_runtime,
            probe_driver,
            cfg=probe_cfg,
            env_name=env_name,
            output_dir=output_dir,
            max_episodes=int(max_episodes),
            max_attempts=max_attempts,
            max_steps_per_episode=int(max_steps_per_episode),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            deterministic=bool(deterministic),
            probe_state=probe_state,
            filter_preset=filter_preset,
            demon_episode_threshold=demon_episode_threshold,
            episode_return_gt=episode_return_gt,
            episode_acceptance=episode_acceptance,
            rollout_seed_offset=int(rollout_seed_offset),
            action_spec=action_spec,
            integration_name=integration_name,
        )

    if probe_only:
        return probe_state

    accepted_specs = probe_state["accepted_specs"]
    if len(accepted_specs) < int(max_episodes):
        report = {
            "status": "incomplete_attempt_budget_exhausted",
            "requested_episodes": int(max_episodes),
            "accepted_episodes": len(accepted_specs),
            "attempted_episodes": int(probe_state["attempted_episodes"]),
            "rejected_episodes": int(probe_state["rejected_episodes"]),
            "attempt_budget": max_attempts,
        }
        (output_dir / "incomplete_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(
            f"Probe accepted {len(accepted_specs)}/{int(max_episodes)} episodes "
            f"within max_attempts={max_attempts}; replay was not started."
        )
    completed_episode_indices = completed_episode_indices_from_shards(output_dir)
    remaining_specs = [
        spec for spec in accepted_specs if int(spec["episode_idx"]) not in completed_episode_indices
    ]
    if remaining_specs:
        matched_gpu_replay = (
            (integration_name in {"flappy", "demon_attack"}
             or config["env_name"] in teacher_runtime._GYMNASIUM_ATARI_GPU_ENVS)
            and ("export_zero_action_history" in config or (
                "flat_cfg" in config
                and config["flat_cfg"].get("action_history_decisions") == 8
            ))
        )
        if matched_gpu_replay and completed_episode_indices:
            raise RuntimeError(
                "GPU history replay must start from all probe slots; clear partial "
                "episode_rows and images before replaying again"
            )
        replay_envs = (
            int(replay_num_envs) if matched_gpu_replay
            else min(int(replay_num_envs), len(remaining_specs))
        )
        replay_cfg, replay_runtime, replay_driver, _ = teacher_runtime.build_rollout_driver_runtime(
            config,
            output_dir=output_dir,
            num_envs=int(replay_envs),
            export_env_raw_rgb_frames=True,
        )
        _replay_accepted_specs(
            replay_runtime,
            replay_driver,
            cfg=replay_cfg,
            env_name=env_name,
            output_dir=output_dir,
            accepted_specs=accepted_specs,
            completed_episode_indices=completed_episode_indices,
            max_steps_per_episode=int(max_steps_per_episode),
            deterministic=bool(deterministic),
            export_image_size=export_image_size,
            image_writer_workers=int(image_writer_workers),
            row_buffer_size=int(row_buffer_size),
            filter_preset=filter_preset,
            context_window=int(context_window),
            image_transform_config=image_transform_config,
            action_spec=action_spec,
            base_prompt=base_prompt,
            integration_name=integration_name,
            export_zero_action_history=(config["export_zero_action_history"]
                                        if "export_zero_action_history" in config else None),
        )

    return probe_state
