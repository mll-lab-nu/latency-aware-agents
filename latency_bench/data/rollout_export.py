"""Finalize canonical splits and their collection metadata."""
from typing import Any, Mapping
from dataclasses import asdict
from latency_bench.data.hf_parquet import write_parquet_splits_from_shards, cleanup_rollout_intermediates
from latency_bench.data.rollout_schema import SCHEMA_VERSION
from latency_bench.utils.io import write_json_atomic
from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, demon_raw_score_bin_counts

def filter_report_from_probe_state(
    split_stats: Mapping[str, Any],
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
        split_stats: Counts returned by the canonical Parquet writer.
        probe_state: Persisted probe state with accepted specs and attempt counters.
        env_name: Canonical benchmark environment name.
        filter_preset: Filter preset name recorded in the report.
        episode_return_field: Canonical episode-return field used by the acceptance gate.
        episode_return_gt: Optional strict lower bound for accepted episode returns.
        episode_acceptance: Explicit task-specific episode acceptance contract.

    Returns:
        Filter report written to metadata and ``filter_report.json``.
    """
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

def finalize_dataset(*, output_dir, task, options, probe_state, source_metadata):
    """Write both canonical splits and publish metadata only after successful writes."""
    spec = task["action_spec"]
    stats = write_parquet_splits_from_shards(output_dir, action_spec=spec)
    runtime = probe_state["runtime_metadata"]
    report = filter_report_from_probe_state(
        stats, probe_state, env_name=task["integration_name"],
        episode_return_field=spec.episode_return_field, filter_preset=options["filter_preset"],
        episode_return_gt=options["episode_return_gt"], episode_acceptance=options["episode_acceptance"],
    )
    if options["demon_episode_threshold"] is not None:
        report["demon_episode_threshold"] = options["demon_episode_threshold"]
    transform = options["image_transform_config"]
    metadata = {
        **source_metadata, **runtime, **stats,
        "schema_version": SCHEMA_VERSION, "env_name": task["env_name"],
        "action_layout": spec.layout, "action_labels": list(spec.labels),
        "reward_field": spec.reward_field, "episode_return_field": spec.episode_return_field,
        "reward_semantics": spec.reward_semantics,
        "filter_preset": options["filter_preset"], "filter_report": report,
        "base_prompt": task["base_prompt"], "rows_unit": "decision_step", "max_steps_unit": "raw_frame",
        "obs_stride_raw_frames": round(runtime["env_fps"] / runtime["obs_fps"]),
        "image_source": task["image_source"] if transform.image_transform == "raw_rgb" else f"{task['image_source']}_{transform.image_transform}",
        "image_transform": asdict(transform), "context_window": options["context_window"],
        "context_image_count": max(0, options["context_window"] - 1),
        "label_provenance": {"semantics": "teacher_issued_command", "alignment": "source_observation"},
        **{key: options[key] for key in ("max_episodes", "max_steps_per_episode", "val_fraction", "split_seed", "num_envs", "replay_num_envs")},
    }
    if task["integration_name"] == "gymnasium":
        metadata["integration_name"] = "gymnasium"
    if options["episode_return_gt"] is not None:
        metadata["episode_return_gate"] = report["episode_return_gate"]
    if options["episode_acceptance"] is not None:
        metadata["episode_acceptance"] = dict(options["episode_acceptance"])
    write_json_atomic(output_dir / "filter_report.json", report)
    write_json_atomic(output_dir / "metadata.json", metadata)
    cleanup_rollout_intermediates(output_dir)
    return metadata
