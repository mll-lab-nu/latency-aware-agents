"""Persist teacher probe decisions and summarize materialized episode shards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, demon_raw_score_bin_counts
from latency_bench.data.hf_parquet import episode_row_split_stats
from latency_bench.data.rollout_schema import SCHEMA_VERSION


def build_episode_plan(
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
    val_count = round(max_episodes * val_fraction)
    val_indices = set(np.random.default_rng(split_seed).choice(max_episodes, size=val_count, replace=False).tolist())
    return [
        {
            "episode_idx": idx,
            "split": "val" if idx in val_indices else "train",
            "seed": base_seed + idx,
        }
        for idx in range(max_episodes)
    ]


def probe_state_path(output_dir: Path) -> Path:
    """Build the state path used to resume probe/replay rollout.

    Args:
        output_dir: Rollout export directory.

    Returns:
        Path to the probe state JSON file.
    """
    return output_dir / "probe_state.json"


def new_probe_state() -> dict[str, Any]:
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


def load_probe_state(
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
    state_path = probe_state_path(output_dir)
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    if completed_episode_indices:
        raise FileNotFoundError(state_path)
    return new_probe_state()


def write_probe_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist probe resume state.

    Args:
        output_dir: Rollout export directory.
        state: JSON-serializable probe state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = probe_state_path(output_dir)
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
    probe uses (``build_episode_plan``) — so the downstream replay path is identical to a
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
        for spec in state["accepted_specs"]:
            seed = spec["seed"]
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            merged_specs.append(dict(spec))

    if len(merged_specs) < max_episodes:
        raise ValueError(
            f"sharded probe collected {len(merged_specs)} accepted episodes, "
            f"fewer than max_episodes={max_episodes}; increase per-shard quota."
        )

    final_splits = {
        plan_spec["episode_idx"]: plan_spec["split"]
        for plan_spec in build_episode_plan(
            max_episodes=max_episodes,
            val_fraction=val_fraction,
            split_seed=split_seed,
            base_seed=0,
        )
    }

    accepted_specs: list[dict[str, Any]] = []
    for episode_idx, spec in enumerate(merged_specs[: max_episodes]):
        finalized = dict(spec)
        finalized["episode_idx"] = episode_idx
        finalized["split"] = final_splits[episode_idx]
        accepted_specs.append(finalized)

    merged = new_probe_state()
    merged["accepted_specs"] = accepted_specs
    merged["attempted_episodes"] = sum(s["attempted_episodes"] for s in shard_states)
    merged["rejected_episodes"] = sum(s["rejected_episodes"] for s in shard_states)
    merged["next_attempt_idx"] = sum(s["next_attempt_idx"] for s in shard_states)
    for state in shard_states:
        if "runtime_metadata" in state:
            merged["runtime_metadata"] = dict(state["runtime_metadata"])
            break
    return merged


def filter_report_from_probe_state(
    output_dir: Path,
    probe_state: Mapping[str, Any],
    *,
    env_name: str,
    episode_return_field: str,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    episode_return_gt: float | None = None,
    episode_acceptance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build final filter report from probe state and written row shards.

    Args:
        output_dir: Rollout export directory containing episode row shards.
        probe_state: Persisted probe state with accepted specs and attempt counters.
        env_name: Canonical benchmark environment name.
        filter_preset: Filter preset name recorded in the report.
        episode_return_field: Canonical episode-return field used by the acceptance gate.
        episode_return_gt: Optional strict lower bound for accepted episode returns.
        episode_acceptance: Explicit task-specific episode acceptance contract.

    Returns:
        Filter report written to metadata and ``filter_report.json``.
    """
    split_stats = episode_row_split_stats(output_dir)
    accepted_specs = probe_state["accepted_specs"]
    train_specs = [spec for spec in accepted_specs if spec["split"] == "train"]
    val_specs = [spec for spec in accepted_specs if spec["split"] == "val"]
    report = {
        "preset": filter_preset,
        "attempted_episodes": probe_state["attempted_episodes"],
        "accepted_episodes": len(accepted_specs),
        "rejected_episodes": probe_state["rejected_episodes"],
        "accepted_train_episodes": len(train_specs),
        "accepted_val_episodes": len(val_specs),
        "train_rows_before_row_filter": sum(spec["episode_length"] for spec in train_specs),
        "train_rows_after_row_filter": split_stats["train_rows"],
        "val_rows_before_row_filter": sum(spec["episode_length"] for spec in val_specs),
        "val_rows_after_row_filter": split_stats["val_rows"],
    }
    if episode_acceptance is not None:
        report["episode_acceptance"] = dict(episode_acceptance)
    if episode_return_gt is not None:
        report["episode_return_gate"] = {
            "field": episode_return_field,
            "operator": ">",
            "threshold": episode_return_gt,
        }
    if env_name == "demon_attack":
        report["raw_score_bins"] = demon_raw_score_bin_counts(accepted_specs)
    return report


def finalize_metadata(*, output_dir, probe_state, config, action_spec, filter_preset,
                      demon_episode_threshold, episode_return_gt, episode_acceptance,
                      image_transform_config, image_resolution, context_window,
                      replay_num_envs):
    """Describe the accepted probe decisions and completed replay dataset."""
    runtime_metadata = probe_state["runtime_metadata"]
    env_fps = runtime_metadata["env_fps"]
    obs_fps = runtime_metadata["obs_fps"]
    split_stats = episode_row_split_stats(output_dir)
    filter_report = filter_report_from_probe_state(
        output_dir,
        probe_state,
        env_name=config["integration_name"],
        filter_preset=filter_preset,
        episode_return_field=action_spec.episode_return_field,
        episode_return_gt=episode_return_gt,
        episode_acceptance=episode_acceptance,
    )
    if demon_episode_threshold is not None:
        filter_report["demon_episode_threshold"] = demon_episode_threshold
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "reward_field": action_spec.reward_field,
        "episode_return_field": action_spec.episode_return_field,
        "reward_semantics": action_spec.reward_semantics,
        "filter_preset": filter_preset,
        "filter_report": filter_report,
        "checkpoint_train_step": runtime_metadata["checkpoint_train_step"],
        "frame_stack": runtime_metadata["frame_stack"],
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": int(env_fps / obs_fps),
        "max_steps_unit": "raw_frame",
        "image_source": "env_raw_rgb"
        if image_transform_config.normalized_transform() == "raw_rgb"
        else f"env_raw_rgb_{image_transform_config.normalized_transform()}",
        "image_transform": image_transform_config.metadata(),
        "image_resolution": image_resolution,
        "context_window": context_window,
        "context_image_count": max(0, context_window - 1),
        "replay_num_envs": replay_num_envs,
        "train_episodes": split_stats["train_episodes"],
        "val_episodes": split_stats["val_episodes"],
    }
    if "flat_cfg" in config and config["flat_cfg"].get("action_history_decisions") is not None:
        metadata["action_history_decisions"] = config["flat_cfg"]["action_history_decisions"]
    if "export_zero_action_history" in config:
        metadata["action_history_decisions"] = config["export_zero_action_history"]
    if ("flat_cfg" in config and config["flat_cfg"].get("latency_type") == "profile_sample"
            and metadata.get("action_history_decisions") == 8):
        metadata["profile_seed_by_reset"] = "env_reset_seed"
    if episode_return_gt is not None:
        metadata["episode_return_gate"] = filter_report["episode_return_gate"]
    return metadata
