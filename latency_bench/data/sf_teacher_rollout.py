"""Collect canonical rollout rows from Sample Factory teacher checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from latency_bench.data.action_layouts import action_spec_for_env
from latency_bench.data.filters import (
    FILTER_PRESET_CLEAN_V1,
    accepts_episode,
    demon_raw_score_bin_counts,
    filter_train_rows,
)
from latency_bench.data.hf_parquet import (
    completed_episode_indices_from_shards,
    episode_row_split_stats,
    write_episode_row_shard,
)
from latency_bench.data.ghost_trail import GhostTrailConfig, build_episode_image_frames
from latency_bench.data.rollout_image_io import RolloutImageWriter, frame_from_env_raw_rgb_info
from latency_bench.data.rollout_schema import SCHEMA_VERSION
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.data.sf_teacher_drivers import (
    BatchedTeacherRolloutDriver,
    RolloutStepResult,
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
    _to_python_action,
)
from training.atari_episode_stats import is_true_atari_episode_end
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


def _probe_state_path(output_dir: Path) -> Path:
    """Build the state path used to resume probe/replay rollout.

    Args:
        output_dir: Rollout export directory.

    Returns:
        Path to the probe state JSON file.
    """
    return output_dir / "probe_state.json"


def _new_probe_state() -> dict[str, Any]:
    """Create probe state for a fresh export.

    Returns:
        Mutable state persisted after each completed rollout attempt.
    """
    return {
        "next_attempt_idx": 0,
        "attempted_episodes": 0,
        "rejected_episodes": 0,
        "accepted_specs": [],
    }


def _load_probe_state(
    output_dir: Path,
    *,
    completed_episode_indices: set[int],
) -> dict[str, Any]:
    """Load or initialize probe resume state.

    Args:
        output_dir: Rollout export directory.
        completed_episode_indices: Accepted episode shards already on disk.

    Returns:
        Probe state with attempt counters and accepted specs.
    """
    state_path = _probe_state_path(output_dir)
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    if completed_episode_indices:
        raise FileNotFoundError(state_path)
    return _new_probe_state()


def _write_probe_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist probe resume state.

    Args:
        output_dir: Rollout export directory.
        state: JSON-serializable probe state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = _probe_state_path(output_dir)
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def merge_probe_states(
    shard_states: Sequence[Mapping[str, Any]],
    *,
    max_episodes: int,
    val_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    """Merge sharded probe states into one resume state for single-process replay.

    Each shard explored a disjoint reset-seed range (see ``rollout_seed_offset``) and
    persisted its own accepted specs. The merged state takes the first ``max_episodes``
    accepted specs in deterministic shard order, dedupes by reset seed, and reassigns
    contiguous ``episode_idx`` plus train/val split via the same plan the single-process
    probe uses (``_build_episode_plan``) — so the downstream replay path is identical to a
    normal resume. Each spec keeps its original reset seed so replay reproduces the episode.

    Args:
        shard_states: Probe states (parsed ``probe_state.json``) from each shard, in the
            shard order that defines the deterministic merge order.
        max_episodes: Number of episodes the merged dataset must contain.
        val_fraction: Fraction assigned to validation (same plan as the probe).
        split_seed: RNG seed for split assignment (same plan as the probe).

    Returns:
        A merged probe state with exactly ``max_episodes`` accepted specs, ready to write
        to the final ``output_dir`` so ``collect_teacher_dataset`` skips probing and replays.
    """
    merged_specs: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for state in shard_states:
        for spec in state.get("accepted_specs", []):
            seed = int(spec["seed"])
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            merged_specs.append(dict(spec))

    if len(merged_specs) < int(max_episodes):
        raise ValueError(
            f"sharded probe collected {len(merged_specs)} accepted episodes, "
            f"fewer than max_episodes={int(max_episodes)}; increase per-shard quota."
        )

    final_splits = {
        int(plan_spec["episode_idx"]): str(plan_spec["split"])
        for plan_spec in _build_episode_plan(
            max_episodes=int(max_episodes),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            base_seed=0,
        )
    }

    accepted_specs: list[dict[str, Any]] = []
    for episode_idx, spec in enumerate(merged_specs[: int(max_episodes)]):
        finalized = dict(spec)
        finalized["episode_idx"] = int(episode_idx)
        finalized["split"] = final_splits[int(episode_idx)]
        accepted_specs.append(finalized)

    merged = _new_probe_state()
    merged["accepted_specs"] = accepted_specs
    merged["attempted_episodes"] = sum(int(s.get("attempted_episodes", 0)) for s in shard_states)
    merged["rejected_episodes"] = sum(int(s.get("rejected_episodes", 0)) for s in shard_states)
    merged["next_attempt_idx"] = sum(int(s.get("next_attempt_idx", 0)) for s in shard_states)
    for state in shard_states:
        if "runtime_metadata" in state:
            merged["runtime_metadata"] = dict(state["runtime_metadata"])
            break
    return merged


def _filter_report_from_probe_state(
    output_dir: Path,
    probe_state: Mapping[str, Any],
    *,
    env_name: str,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
) -> dict[str, Any]:
    """Build final filter report from probe state and written row shards.

    Args:
        output_dir: Rollout export directory containing episode row shards.
        probe_state: Persisted probe state with accepted specs and attempt counters.
        env_name: Canonical benchmark environment name.
        filter_preset: Filter preset name recorded in the report.

    Returns:
        Filter report written to metadata and ``filter_report.json``.
    """
    split_stats = episode_row_split_stats(output_dir)
    accepted_specs = probe_state["accepted_specs"]
    train_specs = [spec for spec in accepted_specs if spec["split"] == "train"]
    val_specs = [spec for spec in accepted_specs if spec["split"] == "val"]
    report = {
        "preset": filter_preset,
        "attempted_episodes": int(probe_state["attempted_episodes"]),
        "accepted_episodes": len(accepted_specs),
        "rejected_episodes": int(probe_state["rejected_episodes"]),
        "accepted_train_episodes": len(train_specs),
        "accepted_val_episodes": len(val_specs),
        "train_rows_before_row_filter": sum(int(spec["episode_length"]) for spec in train_specs),
        "train_rows_after_row_filter": split_stats["train_rows"],
        "val_rows_before_row_filter": sum(int(spec["episode_length"]) for spec in val_specs),
        "val_rows_after_row_filter": split_stats["val_rows"],
    }
    if env_name == "demon_attack":
        report["raw_score_bins"] = demon_raw_score_bin_counts(accepted_specs)
    return report


def _write_filtered_episode_images(
    image_writer: RolloutImageWriter,
    rows: list[dict[str, Any]],
    frames_by_decision_step: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    """Write images for rows selected by split-specific row filtering.

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


def _update_probe_progress(progress: tqdm, probe_state: Mapping[str, Any], *, allocated_attempts: int) -> None:
    """Display raw probe attempt counters in the progress bar.

    Args:
        progress: Active tqdm progress bar.
        probe_state: Probe attempt counters.
        allocated_attempts: Number of raw attempt ids assigned to rollout slots.
    """
    attempted = int(probe_state["attempted_episodes"])
    accepted = len(probe_state["accepted_specs"])
    accept_rate = accepted / attempted if attempted else 0.0
    # alloc counts attempt seeds assigned to active slots; unfinished attempts are dropped after target acceptance.
    progress.set_postfix(
        alloc=int(allocated_attempts),
        done=attempted,
        rej=int(probe_state["rejected_episodes"]),
        accept_rate=f"{accept_rate:.2f}",
    )


def _step_teacher_policy(
    runtime: Mapping[str, Any],
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
    *,
    active_slots: list[int],
    active_obs: list[Any | None],
    rnn_states: torch.Tensor,
    deterministic: bool,
) -> tuple[list[Any], torch.Tensor, dict[int, Any]]:
    """Run one teacher policy/env step for active rollout slots.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        driver: Driver that abstracts serial vs batched env stepping.
        active_slots: Slot ids that currently hold rollout attempts.
        active_obs: Latest observations for every driver slot.
        rnn_states: Recurrent state tensor aligned with driver slots.
        deterministic: Whether to use deterministic teacher actions.

    Returns:
        Raw actions, new RNN states for the active batch, and step results.
    """
    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    argmax_actions = runtime["argmax_actions"]
    preprocess_actions = runtime["preprocess_actions"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
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
    return raw_actions, new_rnn_states, driver.step(active_slots, raw_actions)


def _rollout_step_flags(
    env_name: str,
    step_result: RolloutStepResult,
    *,
    next_decision_step: int,
    max_steps_per_episode: int,
) -> tuple[bool, bool]:
    raw_done = step_result.terminated or step_result.truncated
    life_loss_boundary = False
    true_episode_end = raw_done
    if env_name == "demon_attack":
        true_episode_end = is_true_atari_episode_end(
            step_result.terminated,
            step_result.truncated,
            step_result.info,
        )
        life_loss_boundary = raw_done and not true_episode_end
    if next_decision_step >= max_steps_per_episode:
        return life_loss_boundary, True
    return life_loss_boundary, true_episode_end


# Persist the probe resume state on every acceptance (rare, <= max_episodes) but only
# every N rejections. Once the per-step device syncs are gone, episodes complete very
# frequently, and serializing the whole state to disk per completion dominates. The final
# state is flushed once the probe loop ends, so a clean run's persisted state is identical.
PROBE_STATE_WRITE_EVERY = 256


def _probe_accepted_specs(
    runtime: Mapping[str, Any],
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
    *,
    cfg: Any,
    env_name: str,
    output_dir: Path,
    max_episodes: int,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    deterministic: bool,
    probe_state: dict[str, Any],
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    filter_kwargs: dict[str, Any] | None = None,
    rollout_seed_offset: int = 0,
) -> None:
    """Probe rollout attempts without raw RGB until enough accepted specs exist.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        driver: Driver that abstracts serial vs batched env stepping.
        cfg: Sample Factory rollout cfg.
        env_name: Canonical benchmark environment name.
        output_dir: Rollout export directory.
        max_episodes: Number of accepted episodes requested.
        max_steps_per_episode: Step cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        deterministic: Whether to use deterministic teacher actions.
        probe_state: Mutable probe state persisted for resume.
        filter_preset: Filter preset name for episode-level gating.
        filter_kwargs: Optional overrides forwarded to the episode gate.
    """
    accepted_specs = probe_state["accepted_specs"]
    rnn_size = runtime["rnn_size"]
    device = runtime["device"]
    action_spec = action_spec_for_env(env_name)
    final_splits = {
        int(spec["episode_idx"]): str(spec["split"])
        for spec in _build_episode_plan(
            max_episodes=int(max_episodes),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            base_seed=0,
        )
    }
    active_specs: list[dict[str, Any] | None] = [None for _ in range(driver.slot_count)]
    active_obs: list[Any | None] = [None for _ in range(driver.slot_count)]
    active_episode_lengths = [0 for _ in range(driver.slot_count)]
    active_episode_returns = [0.0 for _ in range(driver.slot_count)]
    active_episode_raw_returns = [0.0 for _ in range(driver.slot_count)]
    active_decision_steps = [0 for _ in range(driver.slot_count)]
    rnn_states = torch.zeros([driver.slot_count, rnn_size], dtype=torch.float32, device=device)

    try:
        next_attempt_idx = int(probe_state["next_attempt_idx"])
        initial_reset_plan: dict[int, int] = {}
        for slot in range(driver.slot_count):
            spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed) + int(rollout_seed_offset))
            next_attempt_idx += 1
            active_specs[slot] = spec
            initial_reset_plan[slot] = int(spec["seed"])
        probe_state["next_attempt_idx"] = next_attempt_idx
        if initial_reset_plan:
            if isinstance(driver, BatchedTeacherRolloutDriver):
                reset_results = driver.reset_slots(initial_reset_plan, [spec is not None for spec in active_specs])
            else:
                reset_results = driver.reset_slots(initial_reset_plan)
            for slot, (obs, _reset_info) in reset_results.items():
                active_obs[slot] = obs
                active_episode_lengths[slot] = 0
                active_episode_returns[slot] = 0.0
                active_episode_raw_returns[slot] = 0.0
                active_decision_steps[slot] = 0
                rnn_states[slot].zero_()

        with tqdm(total=int(max_episodes), initial=len(accepted_specs), unit="episode") as progress:
            while len(accepted_specs) < int(max_episodes):
                active_slots = [slot for slot, spec in enumerate(active_specs) if spec is not None]
                if not active_slots:
                    break
                _raw_actions, new_rnn_states, step_results = _step_teacher_policy(
                    runtime,
                    driver,
                    active_slots=active_slots,
                    active_obs=active_obs,
                    rnn_states=rnn_states,
                    deterministic=deterministic,
                )
                reset_plan: dict[int, int] = {}
                target_reached = False

                # Scatter the new recurrent states for every active slot in one
                # batched assignment instead of one tiny GPU write per slot. Slots
                # that finish/reset below still get .zero_()'d afterwards, exactly
                # as when each row was written individually.
                rnn_states[active_slots] = new_rnn_states

                for batch_idx, slot in enumerate(active_slots):
                    if len(accepted_specs) >= int(max_episodes):
                        active_specs[slot] = None
                        active_obs[slot] = None
                        active_episode_lengths[slot] = 0
                        active_episode_returns[slot] = 0.0
                        active_episode_raw_returns[slot] = 0.0
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                        continue
                    spec = active_specs[slot]
                    step_result = step_results[slot]
                    active_episode_lengths[slot] += 1
                    active_episode_returns[slot] += float(_to_python_action(step_result.reward))
                    if env_name == "demon_attack":
                        active_episode_raw_returns[slot] += float(_to_python_action(step_result.info["raw_reward"]))
                    else:
                        active_episode_raw_returns[slot] += float(_to_python_action(step_result.reward))

                    next_decision_step = int(active_decision_steps[slot]) + 1
                    life_loss_boundary, episode_done = _rollout_step_flags(
                        env_name,
                        step_result,
                        next_decision_step=next_decision_step,
                        max_steps_per_episode=int(max_steps_per_episode),
                    )
                    if episode_done:
                        probe_state["attempted_episodes"] += 1
                        summary = {
                            "episode_length": int(active_episode_lengths[slot]),
                            action_spec.episode_return_field: float(active_episode_returns[slot]),
                        }
                        if env_name == "demon_attack":
                            summary["episode_raw_return"] = float(active_episode_raw_returns[slot])
                        accepted = accepts_episode(
                            filter_preset,
                            env_name,
                            summary,
                            max_steps_per_episode=int(max_steps_per_episode),
                            filter_kwargs=filter_kwargs,
                        )
                        if accepted:
                            final_episode_idx = len(accepted_specs)
                            final_split = final_splits[final_episode_idx]
                            accepted_spec = {
                                "attempt_idx": int(spec["attempt_idx"]),
                                "episode_idx": int(final_episode_idx),
                                "episode_length": int(summary["episode_length"]),
                                action_spec.episode_return_field: float(summary[action_spec.episode_return_field]),
                                "seed": int(spec["seed"]),
                                "split": str(final_split),
                            }
                            if env_name == "demon_attack":
                                accepted_spec["episode_raw_return"] = float(summary["episode_raw_return"])
                            accepted_specs.append(accepted_spec)
                            progress.update(1)
                            target_reached = len(accepted_specs) >= int(max_episodes)
                        else:
                            probe_state["rejected_episodes"] += 1
                        if len(accepted_specs) < int(max_episodes):
                            next_spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed) + int(rollout_seed_offset))
                            next_attempt_idx += 1
                            active_specs[slot] = next_spec
                            reset_plan[slot] = int(next_spec["seed"])
                        else:
                            active_specs[slot] = None
                            active_obs[slot] = None
                            active_episode_lengths[slot] = 0
                            active_episode_returns[slot] = 0.0
                            active_episode_raw_returns[slot] = 0.0
                            active_decision_steps[slot] = 0
                            rnn_states[slot].zero_()
                        probe_state["next_attempt_idx"] = next_attempt_idx
                        if accepted or int(probe_state["attempted_episodes"]) % PROBE_STATE_WRITE_EVERY == 0:
                            _write_probe_state(output_dir, probe_state)
                        _update_probe_progress(
                            progress,
                            probe_state,
                            allocated_attempts=next_attempt_idx,
                        )
                    else:
                        if life_loss_boundary:
                            active_obs[slot], _reset_info = driver.continue_after_life_loss(slot)
                            rnn_states[slot].zero_()
                        else:
                            active_obs[slot] = step_result.next_obs
                        active_decision_steps[slot] = next_decision_step

                if target_reached:
                    break
                if reset_plan:
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        reset_results = driver.reset_slots(reset_plan, [spec is not None for spec in active_specs])
                    else:
                        reset_results = driver.reset_slots(reset_plan)
                    for slot, (obs, _reset_info) in reset_results.items():
                        active_obs[slot] = obs
                        active_episode_lengths[slot] = 0
                        active_episode_returns[slot] = 0.0
                        active_episode_raw_returns[slot] = 0.0
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                else:
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        driver.sync_active([spec is not None for spec in active_specs])
        # Flush the final probe state once the loop ends so a clean run persists the
        # exact same accepted_specs/counters as writing on every completion did.
        _write_probe_state(output_dir, probe_state)
    finally:
        driver.close()


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
    image_transform_config: GhostTrailConfig | None = None,
) -> list[int] | None:
    """Replay accepted specs with raw RGB export and write episode shards.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        driver: Driver that abstracts serial vs batched env stepping.
        cfg: Sample Factory rollout cfg.
        env_name: Canonical benchmark environment name.
        output_dir: Rollout export directory.
        accepted_specs: Accepted probe specs.
        completed_episode_indices: Episode indices already written as shards.
        max_steps_per_episode: Step cap per episode.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        filter_preset: Filter preset name for train row density control.
        image_transform_config: Optional causal image transform applied before
            row filtering. Defaults to raw RGB.
    Returns:
        First observed image resolution, or ``None`` when no spec is replayed.
    """
    image_transform_config = image_transform_config or GhostTrailConfig()
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
    spec_cursor = 0
    replayed = 0

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
                raw_actions, new_rnn_states, step_results = _step_teacher_policy(
                    runtime,
                    driver,
                    active_slots=active_slots,
                    active_obs=active_obs,
                    rnn_states=rnn_states,
                    deterministic=deterministic,
                )
                reset_plan: dict[int, int] = {}

                for batch_idx, slot in enumerate(active_slots):
                    spec = active_specs[slot]
                    decision_step = int(active_decision_steps[slot])
                    final_dir = _episode_image_dir(output_dir, str(spec["split"]), int(spec["episode_idx"]))
                    image_path = final_dir / f"decision_step_{decision_step:06d}.png"
                    image_frame = active_image_frames[slot]
                    if image_resolution is None:
                        if export_image_size is not None:
                            width, height = export_image_size
                            image_resolution = [height, width, image_frame.shape[2]]
                        else:
                            image_resolution = list(image_frame.shape)
                    step_result = step_results[slot]
                    next_decision_step = decision_step + 1
                    life_loss_boundary, episode_done = _rollout_step_flags(
                        env_name,
                        step_result,
                        next_decision_step=next_decision_step,
                        max_steps_per_episode=int(max_steps_per_episode),
                    )
                    if not life_loss_boundary:
                        active_episode_frames[slot].append(np.array(image_frame, copy=True))
                        row = teacher_dataset_row(
                            env_name=env_name,
                            spec=spec,
                            decision_step=decision_step,
                            image=str(image_path),
                            raw_action=raw_actions[batch_idx],
                            info=step_result.info,
                            reward=step_result.reward,
                            env_fps=env_fps,
                            obs_fps=obs_fps,
                            image_transform=image_transform_config.normalized_transform(),
                        )
                        active_episode_rows[slot].append(row)
                    rnn_states[slot] = new_rnn_states[batch_idx]
                    if episode_done:
                        final_rows = _finalize_attempt_rows(
                            active_episode_rows[slot],
                            final_dir=final_dir,
                            episode_idx=int(spec["episode_idx"]),
                            split=str(spec["split"]),
                        )
                        if str(spec["split"]) == "train":
                            shard_rows = filter_train_rows(
                                filter_preset,
                                env_name,
                                final_rows,
                                seed=int(spec["seed"]),
                            )
                        else:
                            shard_rows = [dict(row) for row in final_rows]
                        frames_by_decision_step = build_episode_image_frames(
                            env_name=env_name,
                            rows=final_rows,
                            frames=active_episode_frames[slot],
                            config=image_transform_config,
                        )
                        shard_rows = _write_filtered_episode_images(
                            image_writer,
                            shard_rows,
                            frames_by_decision_step,
                        )
                        write_episode_row_shard(output_dir, shard_rows)
                        progress.update(1)
                        replayed += 1
                        replacement_plan = assign_specs_to_slots([slot])
                        reset_plan.update(replacement_plan)
                        active_obs[slot] = None
                        active_image_frames[slot] = None
                        active_episode_rows[slot] = []
                        active_episode_frames[slot] = []
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                    else:
                        if life_loss_boundary:
                            active_obs[slot], reset_info = driver.continue_after_life_loss(slot)
                            active_image_frames[slot] = frame_from_env_raw_rgb_info(reset_info)
                            rnn_states[slot].zero_()
                        else:
                            active_obs[slot] = step_result.next_obs
                            active_image_frames[slot] = frame_from_env_raw_rgb_info(step_result.info)
                        active_decision_steps[slot] = next_decision_step

                if reset_plan:
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        reset_results = driver.reset_slots(reset_plan, [spec is not None for spec in active_specs])
                    else:
                        reset_results = driver.reset_slots(reset_plan)
                    for slot, (obs, reset_info) in reset_results.items():
                        active_obs[slot] = obs
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
    return image_resolution


def _build_rollout_driver_runtime(
    config: dict[str, Any],
    *,
    output_dir: Path,
    num_envs: int,
    export_env_raw_rgb_frames: bool,
):
    """Build Sample Factory cfg, runtime, and driver for one rollout phase.

    Args:
        config: Rollout descriptor loaded from a checkpoint.
        output_dir: Rollout export directory.
        num_envs: Number of rollout envs or batched agents for this phase.
        export_env_raw_rgb_frames: Whether env info should include raw RGB frames.

    Returns:
        Sample Factory cfg, runtime mapping, driver, and checkpoint frame stack.
    """
    env_name = str(config["env_name"])
    with sample_factory_init_log(output_dir):
        module, cfg = build_sample_factory_rollout_cfg(config)
        frame_stack = int(cfg.frame_stack)
        setattr(cfg, "export_env_raw_rgb_frames", bool(export_env_raw_rgb_frames))
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
            runtime = build_env_and_policy(module, cfg, config, int(num_envs))

    if use_batched_driver:
        driver = BatchedTeacherRolloutDriver(runtime["envs"][0], int(num_envs))
    else:
        driver = SerialTeacherRolloutDriver(runtime["envs"])
    return cfg, runtime, driver, frame_stack


def collect_teacher_dataset(
    config: dict[str, Any],
    *,
    output_dir: Path,
    max_episodes: int,
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
    filter_kwargs: dict[str, Any] | None = None,
    rollout_seed_offset: int = 0,
    probe_only: bool = False,
    image_transform_config: GhostTrailConfig | None = None,
) -> dict[str, Any]:
    """Collect a Sample Factory teacher dataset from a checkpoint descriptor.

    Args:
        config: Rollout descriptor loaded from a checkpoint.
        output_dir: Rollout export directory.
        max_episodes: Number of episodes to collect.
        max_steps_per_episode: Step cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        num_envs: Number of probe rollout environments or agents.
        replay_num_envs: Number of replay environments or agents used while writing rows/images.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        filter_preset: Filter preset name (``"none"`` or ``"clean_v1"``).
        filter_kwargs: Temporary exporter experiment arguments retained at the public boundary.
        image_transform_config: Optional causal image transform applied during
            replay before row filtering.
    Returns:
        Rollout metadata.
    """
    env_name = str(config["env_name"])
    image_transform_config = image_transform_config or GhostTrailConfig()
    completed_episode_indices = completed_episode_indices_from_shards(output_dir)
    probe_state = _load_probe_state(
        output_dir,
        completed_episode_indices=completed_episode_indices,
    )

    if len(probe_state["accepted_specs"]) < int(max_episodes):
        probe_cfg, probe_runtime, probe_driver, frame_stack = _build_rollout_driver_runtime(
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
        _write_probe_state(output_dir, probe_state)
        _probe_accepted_specs(
            probe_runtime,
            probe_driver,
            cfg=probe_cfg,
            env_name=env_name,
            output_dir=output_dir,
            max_episodes=int(max_episodes),
            max_steps_per_episode=int(max_steps_per_episode),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            deterministic=bool(deterministic),
            probe_state=probe_state,
            filter_preset=filter_preset,
            filter_kwargs=filter_kwargs,
            rollout_seed_offset=int(rollout_seed_offset),
        )

    if probe_only:
        # Sharded-probe worker mode: the accepted_specs (with their reset seeds) are now
        # persisted in probe_state.json; skip replay/image writing. A separate merge step
        # combines shards and the single-process replay path materializes the dataset.
        return {
            "probe_only": True,
            "env_name": env_name,
            "accepted_episodes": len(probe_state["accepted_specs"]),
            "rollout_seed_offset": int(rollout_seed_offset),
            "filter_preset": filter_preset,
        }

    accepted_specs = probe_state["accepted_specs"]
    completed_episode_indices = completed_episode_indices_from_shards(output_dir)
    remaining_specs = [
        spec for spec in accepted_specs if int(spec["episode_idx"]) not in completed_episode_indices
    ]
    image_resolution = None
    if remaining_specs:
        replay_envs = min(int(replay_num_envs), len(remaining_specs))
        replay_cfg, replay_runtime, replay_driver, frame_stack = _build_rollout_driver_runtime(
            config,
            output_dir=output_dir,
            num_envs=int(replay_envs),
            export_env_raw_rgb_frames=True,
        )
        image_resolution = _replay_accepted_specs(
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
            image_transform_config=image_transform_config,
        )

    action_spec = action_spec_for_env(env_name)
    runtime_metadata = probe_state["runtime_metadata"]
    env_fps = float(runtime_metadata["env_fps"])
    obs_fps = float(runtime_metadata["obs_fps"])
    split_stats = episode_row_split_stats(output_dir)
    filter_report = _filter_report_from_probe_state(
        output_dir,
        probe_state,
        env_name=env_name,
        filter_preset=filter_preset,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "reward_field": action_spec.reward_field,
        "episode_return_field": action_spec.episode_return_field,
        "reward_semantics": action_spec.reward_semantics,
        "filter_preset": filter_preset,
        "filter_report": filter_report,
        "checkpoint_train_step": int(runtime_metadata["checkpoint_train_step"]),
        "frame_stack": int(runtime_metadata["frame_stack"]),
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": int(env_fps / obs_fps),
        "image_source": "env_raw_rgb"
        if image_transform_config.normalized_transform() == "raw_rgb"
        else f"env_raw_rgb_{image_transform_config.normalized_transform()}",
        "image_transform": image_transform_config.metadata(),
        "image_resolution": image_resolution,
        "replay_num_envs": int(replay_num_envs),
        "train_episodes": split_stats["train_episodes"],
        "val_episodes": split_stats["val_episodes"],
    }
