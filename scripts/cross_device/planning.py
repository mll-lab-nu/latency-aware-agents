"""Planning, row selection, and skip logic for cross-device runs.

The planner converts protocol spec rows into concrete task objects. It may check
whether a profile exists, but dry-runs deliberately avoid network downloads.
"""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.analysis.cross_device_protocol_tables import slugify
from scripts.cross_device.models import PlannedTask, ProtocolTask, RuntimeSelection
from scripts.cross_device.profile_cache import ensure_profile_cached
from scripts.cross_device.runtime import (
    _normalize_name,
    _resolve_runtime_alias,
    _runtime_conda_env,
    _devices,
)


STAGE_TO_EXPERIMENTS = {
    "measure": ("exp1_latency_profile_summary", "exp2_same_device_calibration", "exp3_cross_device_prediction"),
    "realtime": ("exp2_same_device_calibration", "exp3_cross_device_prediction"),
    "sim_same": ("exp2_same_device_calibration",),
    "sim_cross": ("exp3_cross_device_prediction",),
}


def _build_plan(
    args: argparse.Namespace,
    spec: dict[str, Any],
    stage: str,
    out: Path,
    profiles_dir: Path,
    current_device: dict[str, Any],
    runtime: RuntimeSelection,
    *,
    for_doctor: bool = False,
) -> list[PlannedTask]:
    """Build concrete tasks for one stage from selected protocol rows.

    Args:
        args: Parsed CLI namespace with filters, overrides, and smoke options.
        spec: Loaded protocol spec.
        stage: Normalized stage name, such as `measure` or `sim_cross`.
        out: Protocol output root.
        profiles_dir: Local profile cache root.
        current_device: Hardware entry for the machine running this command.
        runtime: Resolved runtime environment selection.
        for_doctor: When true, report missing profiles without trying HF download.

    Returns:
        Planned tasks marked runnable or skipped with a concrete reason.
    """
    plan_by_key: dict[tuple[Any, ...], PlannedTask] = {}
    seed_set = _seed_set(spec, args)
    checkpoint_kind = str(args.checkpoint_kind or spec.get("checkpoint_kind", "best"))
    for experiment_id, experiment, row in _iter_selected_rows(spec, args, include_stage=stage):
        n_episodes = _n_episodes_for_row_stage(spec, args, row, stage)
        device_pairs = _device_pairs_for_stage(spec, args, stage, current_device, experiment_id)
        for target_device, replay_device in device_pairs:
            for seed in seed_set:
                profile_dir = profiles_dir / slugify(row.get("policy")) / slugify(row.get("env")) / str(target_device["profile_id"])
                profile_path = profile_dir / "profile.json"
                runtime_alias = str(row.get("runtime_env") or "")
                runtime_conda = _runtime_conda_env(spec, runtime_alias)
                task = _make_task(
                    stage=stage,
                    experiment_id=experiment_id,
                    experiment_ids=[experiment_id],
                    panel=str(experiment.get("panel", "")),
                    row=row,
                    target_device=target_device,
                    replay_device=replay_device,
                    seed=int(seed),
                    seed_set=seed_set,
                    n_episodes=n_episodes,
                    checkpoint_kind=checkpoint_kind,
                    out=out,
                    target_profile_path=profile_path if stage in {"realtime", "sim_same", "sim_cross"} else None,
                    profile_output_dir=profile_dir if stage == "measure" else None,
                    runtime_env_alias=runtime_alias or None,
                    runtime_conda_env=runtime_conda,
                    smoke=bool(getattr(args, "smoke", False)),
                    smoke_max_steps=getattr(args, "smoke_max_steps", None),
                )
                key = _dedup_key(task)
                skip_reason = _task_skip_reason(
                    task=task,
                    spec=spec,
                    args=args,
                    stage=stage,
                    profiles_dir=profiles_dir,
                    current_device=current_device,
                    runtime=runtime,
                    for_doctor=for_doctor,
                )
                item = PlannedTask(task=task, runnable=skip_reason is None, skip_reason=skip_reason)
                existing = plan_by_key.get(key)
                if existing is None:
                    plan_by_key[key] = item
                    continue
                if experiment_id not in existing.task.experiment_ids:
                    existing.task.experiment_ids.append(experiment_id)
                if existing.runnable:
                    continue
                if item.runnable:
                    item.task.experiment_ids = existing.task.experiment_ids
                    plan_by_key[key] = item
                elif item.skip_reason and existing.skip_reason and item.skip_reason not in existing.skip_reason:
                    existing.skip_reason = f"{existing.skip_reason}; {item.skip_reason}"
    return list(plan_by_key.values())


def _make_task(
    *,
    stage: str,
    experiment_id: str,
    experiment_ids: list[str],
    panel: str,
    row: dict[str, Any],
    target_device: dict[str, Any],
    replay_device: dict[str, Any] | None,
    seed: int,
    seed_set: list[int],
    n_episodes: int,
    checkpoint_kind: str,
    out: Path,
    target_profile_path: Path | None,
    profile_output_dir: Path | None,
    runtime_env_alias: str | None,
    runtime_conda_env: str | None,
    smoke: bool = False,
    smoke_max_steps: int | None = None,
) -> ProtocolTask:
    """Create output paths and immutable run identity for one task.

    Args:
        stage: Normalized stage name.
        experiment_id: Primary experiment that selected this row.
        experiment_ids: All experiments deduped into this task.
        panel: Paper/protocol panel label.
        row: Protocol row containing policy/env/config metadata.
        target_device: Device whose latency profile is measured or replayed.
        replay_device: Host device for cross-device replay, if any.
        seed: Evaluation seed.
        seed_set: Full seed set used in the run identity.
        n_episodes: Episode count for this task.
        checkpoint_kind: Checkpoint selector forwarded to evaluation.
        out: Protocol output root.
        target_profile_path: Target-device profile manifest consumed by realtime and replay stages.
        profile_output_dir: Profile artifact directory produced by measure.
        runtime_env_alias: Required runtime alias from the protocol row.
        runtime_conda_env: Conda env name for display and diagnostics.
        smoke: Whether this task is a smoke run.
        smoke_max_steps: Optional max-step cap for smoke evaluation.

    Returns:
        A `ProtocolTask` with run output and generated-config paths assigned.
    """
    policy_slug = slugify(row.get("policy"))
    env_slug = slugify(row.get("env"))
    stage_dir = stage
    run_id = f"smoke_seed_{seed}_n{n_episodes}" if smoke else f"{checkpoint_kind}_seed_{seed}_n{n_episodes}"
    if stage == "sim_cross" and replay_device is not None:
        device_or_pair = f"{target_device['id']}_replay_{replay_device['id']}"
        output_dir = out / "runs" / stage_dir / policy_slug / env_slug / str(target_device["id"]) / f"replay_{replay_device['id']}" / run_id
    else:
        device_or_pair = str(target_device["id"])
        output_dir = out / "runs" / stage_dir / policy_slug / env_slug / str(target_device["id"]) / run_id
    generated_config_path = out / "generated_configs" / stage_dir / policy_slug / env_slug / device_or_pair / f"{run_id}.yaml"
    return ProtocolTask(
        stage=stage,
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        panel=panel,
        row=row,
        target_device=target_device,
        replay_device=replay_device,
        seed=seed,
        seed_set=seed_set,
        n_episodes=n_episodes,
        checkpoint_kind=checkpoint_kind,
        output_dir=output_dir,
        generated_config_path=generated_config_path,
        target_profile_path=target_profile_path,
        profile_output_dir=profile_output_dir,
        runtime_env_alias=runtime_env_alias,
        runtime_conda_env=runtime_conda_env,
        smoke=smoke,
        smoke_max_steps=None if smoke_max_steps is None else int(smoke_max_steps),
    )


def _dedup_key(task: ProtocolTask) -> tuple[Any, ...]:
    policy = slugify(task.row.get("policy"))
    env = slugify(task.row.get("env"))
    seed_set = tuple(task.seed_set)
    profile_id = task.target_device.get("profile_id")
    replay_id = task.replay_device.get("id") if task.replay_device else None
    if task.stage == "measure":
        return (
            "measure",
            policy,
            env,
            task.target_device.get("id"),
            profile_id,
            task.checkpoint_kind,
            seed_set,
            task.seed,
        )
    if task.stage == "realtime":
        return ("realtime", policy, env, task.target_device.get("id"), task.checkpoint_kind, seed_set, task.seed)
    if task.stage == "sim_same":
        return (
            "sim_same",
            policy,
            env,
            task.target_device.get("id"),
            profile_id,
            task.checkpoint_kind,
            seed_set,
            task.seed,
        )
    if task.stage == "sim_cross":
        return (
            "sim_cross",
            policy,
            env,
            task.target_device.get("id"),
            replay_id,
            profile_id,
            task.checkpoint_kind,
            seed_set,
            task.seed,
        )
    return (task.stage, policy, env, task.target_device.get("id"), replay_id, task.seed)


def _task_skip_reason(
    *,
    task: ProtocolTask,
    spec: dict[str, Any],
    args: argparse.Namespace,
    stage: str,
    profiles_dir: Path,
    current_device: dict[str, Any],
    runtime: RuntimeSelection,
    for_doctor: bool,
) -> str | None:
    """Return why a planned task cannot run on this invocation.

    Args:
        task: Candidate task.
        spec: Loaded protocol spec.
        args: Parsed CLI namespace.
        stage: Normalized stage name.
        profiles_dir: Local profile cache root for HF profile resolution.
        current_device: Hardware entry for the running machine.
        runtime: Resolved runtime selection.
        for_doctor: Disables network profile fetches for read-only checks.

    Returns:
        A semicolon-separated skip reason, or None when the task is runnable.
    """
    reasons: list[str] = []
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if stage in {"measure", "realtime", "sim_same"} and not _matches_filter(
        _device_filter_candidates(current_device), only_devices
    ):
        reasons.append(f"hardware mismatch: current hardware {current_device.get('id')} not in --only-device")
    if stage == "sim_cross" and not _matches_filter(_device_filter_candidates(task.target_device), only_devices):
        reasons.append(f"hardware mismatch: target hardware {task.target_device.get('id')} not in --only-device")
    if stage == "sim_cross" and task.replay_device is not None:
        if str(task.replay_device["id"]) != str(current_device["id"]):
            reasons.append(
                "replay hardware mismatch: "
                f"current hardware {current_device['id']} is not pair replay_device {task.replay_device['id']}"
            )

    runtime_reason = _runtime_skip_reason(task, runtime)
    if runtime_reason:
        reasons.append(runtime_reason)

    # Dry-run and doctor must stay read-only; real profile-consuming stages may
    # hydrate a missing local profile from the configured Hugging Face repo.
    if (
        stage in {"realtime", "sim_same", "sim_cross"}
        and task.target_profile_path is not None
        and not task.target_profile_path.exists()
        and not for_doctor
        and not args.dry_run
    ):
        hf_profile_repo = os.environ.get("HF_PROFILE_REPO")
        if hf_profile_repo:
            _ensure_profile_cached(
                profile_path=task.target_profile_path,
                profiles_dir=profiles_dir,
                hf_repo=hf_profile_repo,
            )
    if (
        stage in {"realtime", "sim_same", "sim_cross"}
        and task.target_profile_path is not None
        and not task.target_profile_path.exists()
        and not args.dry_run
    ):
        reasons.append(f"missing profile: {task.target_profile_path}")

    if reasons:
        return "; ".join(reasons)
    return None


def _ensure_profile_cached(*, profile_path: Path, profiles_dir: Path, hf_repo: str) -> Path:
    return ensure_profile_cached(
        profile_path=profile_path,
        profiles_dir=profiles_dir,
        hf_repo=hf_repo,
        repo_type=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"),
        revision=os.environ.get("HF_PROFILE_REVISION"),
    )


def _runtime_skip_reason(task: ProtocolTask, runtime: RuntimeSelection) -> str | None:
    required = task.runtime_env_alias
    if not required:
        return "missing runtime_env on protocol row"
    if runtime.allow_all:
        if runtime.allow_mismatch:
            return None
        if runtime.matched_alias != required:
            return (
                "runtime_env mismatch: "
                f"requires {required} ({task.runtime_conda_env}), "
                f"current={runtime.detected_raw or 'unknown'}"
            )
        return None
    if runtime.selected_alias != required:
        return f"runtime_env selection mismatch: selected {runtime.selected_alias}, row requires {required}"
    if runtime.allow_mismatch:
        return None
    if runtime.matched_alias != required:
        return (
            "runtime_env mismatch: "
            f"requires {required} ({task.runtime_conda_env}), "
            f"current={runtime.detected_raw or 'unknown'}"
        )
    return None


def _iter_selected_rows(
    spec: dict[str, Any],
    args: argparse.Namespace,
    *,
    include_stage: str | None,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return protocol rows selected by stage, filters, and runtime selection.

    Args:
        spec: Loaded protocol spec.
        args: Parsed CLI namespace with `only_*` and runtime filters.
        include_stage: Optional normalized stage used to restrict experiments.

    Returns:
        Tuples of `(experiment_id, experiment_config, row_config)`.
    """
    allowed_experiments = None if include_stage is None else set(STAGE_TO_EXPERIMENTS[include_stage])
    only_experiments = _csv_filter(getattr(args, "only_experiment", None))
    only_policies = _csv_filter(getattr(args, "only_policy", None))
    only_envs = _csv_filter(getattr(args, "only_env", None))
    runtime_filter = _runtime_row_filter(spec, getattr(args, "runtime_env", "auto"))
    output: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for experiment_id, experiment in spec.get("experiments", {}).items():
        if allowed_experiments is not None and experiment_id not in allowed_experiments:
            continue
        if not _matches_filter(_experiment_filter_candidates(experiment_id), only_experiments):
            continue
        for row in experiment.get("rows", []):
            if runtime_filter and not _matches_filter(_runtime_filter_candidates(spec, row), runtime_filter):
                continue
            if not _matches_filter(_policy_filter_candidates(row), only_policies):
                continue
            if not _matches_filter(_env_filter_candidates(row), only_envs):
                continue
            output.append((experiment_id, experiment, row))
    return output


def _device_pairs_for_stage(
    spec: dict[str, Any],
    args: argparse.Namespace,
    stage: str,
    current_device: dict[str, Any],
    experiment_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Return `(target_device, replay_device)` pairs for one stage.

    Args:
        spec: Loaded protocol spec.
        args: Parsed CLI namespace with device filters.
        stage: Normalized stage name.
        current_device: Device entry for this command's execution host.
        experiment_id: Experiment whose replay pairs should be expanded.

    Returns:
        A list of device pairs. Non-cross stages use the current device and no
        replay device; sim-cross uses explicit `replay_pairs` from the spec.
    """
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if stage in {"measure", "realtime", "sim_same"}:
        if not _matches_filter(_device_filter_candidates(current_device), only_devices):
            return []
        return [(current_device, None)]
    devices_by_id = {str(device["id"]): device for device in _devices(spec)}
    output = []
    for pair in _experiment_replay_pairs(spec, experiment_id):
        target = devices_by_id[str(pair["target_device"])]
        replay = devices_by_id[str(pair["replay_device"])]
        if _matches_filter(_device_filter_candidates(target), only_devices):
            output.append((target, replay))
    return output


def _experiment_replay_pairs(spec: dict[str, Any], experiment_id: str) -> list[dict[str, Any]]:
    experiment = spec.get("experiments", {}).get(experiment_id, {})
    return list(experiment["replay_pairs"])


def _seed_set(spec: dict[str, Any], args: argparse.Namespace) -> list[int]:
    raw = args.seed_set if getattr(args, "seed_set", None) is not None else spec.get("seed_set", [0])
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    return [int(item) for item in items]


def _profiles_dir_for_args(args: argparse.Namespace, out: Path) -> Path:
    raw = getattr(args, "profiles_dir", None)
    if raw is not None:
        return Path(raw)
    if bool(getattr(args, "smoke", False)):
        return out / "profiles_smoke"
    return out / "profiles"


def _n_episodes_for_row_stage(
    spec: dict[str, Any],
    args: argparse.Namespace,
    row: dict[str, Any],
    stage: str,
) -> int:
    if bool(getattr(args, "smoke", False)):
        return int(getattr(args, "smoke_episodes", 1))
    if getattr(args, "n_episodes", None) is not None:
        return int(args.n_episodes)
    stage_overrides = row.get("stage_overrides", {})
    if isinstance(stage_overrides, dict):
        stage_cfg = stage_overrides.get(stage)
        if stage_cfg is None:
            stage_cfg = stage_overrides.get(stage.replace("_", "-"))
        if isinstance(stage_cfg, dict) and stage_cfg.get("n_episodes") is not None:
            return int(stage_cfg["n_episodes"])
    if row.get("n_episodes") is not None:
        return int(row["n_episodes"])
    return int(spec.get("n_episodes", 1))


def _spec_with_cli_overrides(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply global CLI overrides to a copy of the protocol spec.

    Args:
        spec: Loaded protocol spec.
        args: Parsed CLI namespace containing optional overrides.

    Returns:
        A copied spec with seed, episode, or checkpoint-kind overrides applied.
    """
    updated = deepcopy(spec)
    if getattr(args, "seed_set", None) is not None:
        updated["seed_set"] = _seed_set(spec, args)
    if getattr(args, "n_episodes", None) is not None:
        updated["n_episodes"] = int(args.n_episodes)
    if getattr(args, "checkpoint_kind", None) is not None:
        updated["checkpoint_kind"] = str(args.checkpoint_kind)
    return updated


def _filtered_spec_for_aggregate(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Filter a copied spec before aggregate table generation.

    Args:
        spec: Loaded protocol spec.
        args: Parsed CLI namespace with row/device/runtime filters.

    Returns:
        A copied spec containing only rows and devices selected for aggregate.
    """
    updated = deepcopy(spec)
    only_experiments = _csv_filter(getattr(args, "only_experiment", None))
    only_policies = _csv_filter(getattr(args, "only_policy", None))
    only_envs = _csv_filter(getattr(args, "only_env", None))
    only_devices = _csv_filter(getattr(args, "only_device", None))
    runtime_filter = _runtime_row_filter(updated, getattr(args, "runtime_env", "auto"))
    if only_devices:
        updated["devices"] = [
            device for device in _devices(updated) if _matches_filter(_device_filter_candidates(device), only_devices)
        ]
    experiments = updated.get("experiments", {})
    if isinstance(experiments, dict):
        for experiment_id in list(experiments):
            if not _matches_filter(_experiment_filter_candidates(experiment_id), only_experiments):
                experiments.pop(experiment_id)
                continue
            experiment = experiments[experiment_id]
            if not isinstance(experiment, dict):
                continue
            rows = []
            for row in experiment.get("rows", []):
                if runtime_filter and not _matches_filter(_runtime_filter_candidates(updated, row), runtime_filter):
                    continue
                if not _matches_filter(_policy_filter_candidates(row), only_policies):
                    continue
                if not _matches_filter(_env_filter_candidates(row), only_envs):
                    continue
                rows.append(row)
            experiment["rows"] = rows
    return updated


def _csv_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _matches_filter(candidates: list[Any], filters: set[str]) -> bool:
    if not filters:
        return True
    candidate_tokens = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_tokens.update(_filter_token_variants(candidate))
    filter_tokens = set()
    for item in filters:
        filter_tokens.update(_filter_token_variants(item))
    return bool(candidate_tokens & filter_tokens)


def _filter_token_variants(value: Any) -> set[str]:
    raw = str(value).strip()
    if not raw:
        return set()
    return {raw, raw.lower(), slugify(raw), _normalize_name(raw)}


def _policy_filter_candidates(row: dict[str, Any]) -> list[Any]:
    return [row.get("policy"), row.get("policy_type")]


def _env_filter_candidates(row: dict[str, Any]) -> list[Any]:
    return [row.get("env"), row.get("env_name")]


def _experiment_filter_candidates(experiment_id: str) -> list[Any]:
    return [experiment_id]


def _device_filter_candidates(device: dict[str, Any]) -> list[Any]:
    """Return filter tokens that can identify one hardware entry.

    Args:
        device: Hardware entry from the protocol spec.

    Returns:
        Device id, display name, profile id, and aliases.
    """
    return [
        device.get("id"),
        device.get("display_name"),
        device.get("profile_id"),
        *device.get("aliases", []),
    ]


def _runtime_row_filter(spec: dict[str, Any], requested: Any) -> set[str]:
    value = str(requested or "auto")
    if value in {"auto", "all"}:
        return set()
    alias = _resolve_runtime_alias(spec, value)
    return {alias or value}


def _runtime_filter_candidates(spec: dict[str, Any], row: dict[str, Any]) -> list[Any]:
    alias = row.get("runtime_env")
    return [alias, _runtime_conda_env(spec, str(alias)) if alias else None]


def _output_root(spec: dict[str, Any], out: Path | None) -> Path:
    return Path(out) if out is not None else Path(str(spec.get("output_root", "results/cross_device")))


def _normalize_stage(stage: str) -> str:
    return stage.replace("-", "_")
