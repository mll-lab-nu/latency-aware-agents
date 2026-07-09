"""Task models, planning, row selection, and skip logic for cross-device runs.

The planner converts protocol spec rows into concrete task objects. It may check
whether a profile exists, but dry-runs deliberately avoid network downloads.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_bench.cross_device.spec import _devices, profile_path as layout_profile_path, slugify
from latency_bench.cross_device.sync import ensure_profile_cached, hf_env


STAGE_TO_EXPERIMENTS = {
    "measure": ("exp1_latency_profile_summary", "exp2_same_device_calibration", "exp3_cross_device_prediction"),
    "sim_same": ("exp2_same_device_calibration",),
    "sim_cross": ("exp3_cross_device_prediction",),
}


@dataclass
class ProtocolTask:
    """Concrete unit of protocol work after row/device/seed expansion."""
    stage: str
    experiment_ids: list[str]
    row: dict[str, Any]
    target_device: dict[str, Any]
    replay_device: dict[str, Any] | None
    seed: int
    seed_set: list[int]
    n_episodes: int
    checkpoint_kind: str
    output_dir: Path
    generated_config_path: Path
    target_profile_path: Path | None
    profile_output_dir: Path | None
    instance_id: str | None
    instance_origin: str | None
    target_profile_instance_id: str | None
    smoke: bool = False
    smoke_max_steps: int | None = None


@dataclass
class PlannedTask:
    """A protocol task plus the planner decision about whether it can run."""
    task: ProtocolTask
    runnable: bool
    skip_reason: str | None = None


def _build_plan(
    args: argparse.Namespace,
    spec: dict[str, Any],
    stage: str,
    out: Path,
    profiles_dir: Path,
    current_device: dict[str, Any],
    *,
    instance_id: str | None = None,
    instance_origin: str | None = None,
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
        for_doctor: When true, report missing profiles without trying HF download.

    Returns:
        Planned tasks marked runnable or skipped with a concrete reason.
    """
    plan: list[PlannedTask] = []
    seed_set = _seed_set(spec, args)
    checkpoint_kind = str(args.checkpoint_kind or spec.get("checkpoint_kind", "best"))
    n_episodes = _n_episodes(spec, args)
    experiment_ids = list(STAGE_TO_EXPERIMENTS[stage])
    _, explicit_instance = _parse_target(args)
    for row in _iter_selected_rows(spec, args):
        for target_device, replay_device in _device_pairs_for_stage(spec, args, stage, current_device):
            for seed in seed_set:
                target_profile_instance_id = instance_id if stage in {"measure", "sim_same"} else explicit_instance
                profile_path = (
                    layout_profile_path(
                        profiles_dir,
                        model_type=row["model_type"],
                        env_name=row["env_name"],
                        device_id=target_device["id"],
                        instance_id=target_profile_instance_id,
                    )
                    if target_profile_instance_id is not None
                    else None
                )
                task = _make_task(
                    stage=stage,
                    experiment_ids=experiment_ids,
                    row=row,
                    target_device=target_device,
                    replay_device=replay_device,
                    seed=int(seed),
                    seed_set=seed_set,
                    n_episodes=n_episodes,
                    checkpoint_kind=checkpoint_kind,
                    out=out,
                    target_profile_path=profile_path if stage in {"sim_same", "sim_cross"} else None,
                    profile_output_dir=profile_path.parent if stage == "measure" and profile_path is not None else None,
                    instance_id=instance_id,
                    instance_origin=instance_origin,
                    target_profile_instance_id=target_profile_instance_id,
                    smoke=bool(getattr(args, "smoke", False)),
                    smoke_max_steps=getattr(args, "smoke_max_steps", None),
                )
                skip_reason = _task_skip_reason(
                    task=task,
                    args=args,
                    stage=stage,
                    profiles_dir=profiles_dir,
                    current_device=current_device,
                    for_doctor=for_doctor,
                )
                plan.append(PlannedTask(task=task, runnable=skip_reason is None, skip_reason=skip_reason))
    return plan


def _parse_target(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Split `--target <device_id>/<instance_id>` into its two parts."""
    raw = getattr(args, "target", None)
    if not raw:
        return None, None
    device_token, _, instance_token = str(raw).partition("/")
    if not device_token or not instance_token:
        raise SystemExit("--target must be <device_id>/<instance_id>")
    return device_token, instance_token


def _make_task(
    *,
    stage: str,
    experiment_ids: list[str],
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
    instance_id: str | None,
    instance_origin: str | None,
    target_profile_instance_id: str | None,
    smoke: bool = False,
    smoke_max_steps: int | None = None,
) -> ProtocolTask:
    """Create output paths and immutable run identity for one task."""
    policy_slug = slugify(row["policy"])
    count_suffix = f"n{n_episodes}"
    run_id = f"smoke_seed_{seed}_{count_suffix}" if smoke else f"{checkpoint_kind}_seed_{seed}_{count_suffix}"
    if stage == "sim_cross" and replay_device is not None:
        device_path = Path(str(target_device["id"]))
        output_dir = out / "runs" / stage / policy_slug / device_path / run_id
        generated_config_path = out / "generated_configs" / stage / policy_slug / device_path / f"{run_id}.yaml"
    else:
        output_dir = out / "runs" / stage / policy_slug / run_id
        generated_config_path = out / "generated_configs" / stage / policy_slug / f"{run_id}.yaml"
    return ProtocolTask(
        stage=stage,
        experiment_ids=experiment_ids,
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
        instance_id=instance_id,
        instance_origin=instance_origin,
        target_profile_instance_id=target_profile_instance_id,
        smoke=smoke,
        smoke_max_steps=None if smoke_max_steps is None else int(smoke_max_steps),
    )


def _task_skip_reason(
    *,
    task: ProtocolTask,
    args: argparse.Namespace,
    stage: str,
    profiles_dir: Path,
    current_device: dict[str, Any],
    for_doctor: bool,
) -> str | None:
    """Return why a planned task cannot run on this invocation, or None."""
    reasons: list[str] = []
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if stage in {"measure", "sim_same"} and not _matches_filter(
        _device_filter_candidates(current_device), only_devices
    ):
        reasons.append(f"hardware mismatch: current hardware {current_device.get('id')} not in --only-device")
    if stage == "sim_cross" and not _matches_filter(_device_filter_candidates(task.target_device), only_devices):
        reasons.append(f"hardware mismatch: target hardware {task.target_device.get('id')} not in --only-device")

    reasons.extend(
        _profile_resource_skip_reasons(
            task=task,
            args=args,
            stage=stage,
            profiles_dir=profiles_dir,
            for_doctor=for_doctor,
        )
    )

    if reasons:
        return "; ".join(reasons)
    return None


def _profile_resource_skip_reasons(
    *,
    task: ProtocolTask,
    args: argparse.Namespace,
    stage: str,
    profiles_dir: Path,
    for_doctor: bool,
) -> list[str]:
    """Return profile-resource reasons that prevent a planned task from running."""
    reasons: list[str] = []

    if stage in {"measure", "sim_same"} and task.target_profile_instance_id is None:
        reasons.append("missing local profile instance: set CROSS_DEVICE_INSTANCE_ID")
    if stage == "sim_cross" and task.target_profile_instance_id is None:
        reasons.append(
            f"missing profile: no instance for target {task.target_device['id']}; "
            "pass --target <device>/<instance>"
        )

    if (
        stage in {"sim_same", "sim_cross"}
        and task.target_profile_path is not None
        and not task.target_profile_path.exists()
        and not for_doctor
        and not args.dry_run
    ):
        hf = hf_env()
        if hf is not None:
            ensure_profile_cached(
                profile_path=task.target_profile_path,
                profiles_dir=profiles_dir,
                hf_repo=hf["repo_id"],
                repo_type=hf["repo_type"],
                revision=hf["revision"],
            )

    if (
        stage in {"sim_same", "sim_cross"}
        and task.target_profile_path is not None
        and not task.target_profile_path.exists()
        and not args.dry_run
    ):
        reasons.append(f"missing profile: {task.target_profile_path}")

    return reasons


def _iter_selected_rows(spec: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return protocol rows selected by the `--only-policy`/`--only-env` filters."""
    return [row for row in spec.get("rows", []) if _row_selected(row, args)]


def _row_selected(row: dict[str, Any], args: argparse.Namespace) -> bool:
    only_policies = _csv_filter(getattr(args, "only_policy", None))
    only_envs = _csv_filter(getattr(args, "only_env", None))
    return _matches_filter([row.get("policy"), row.get("policy_type")], only_policies) and _matches_filter(
        [row.get("env"), row.get("env_name")], only_envs
    )


def _device_pairs_for_stage(
    spec: dict[str, Any],
    args: argparse.Namespace,
    stage: str,
    current_device: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Return `(target_device, replay_device)` pairs for one stage.

    Local stages use the current device and no replay device; sim-cross replays
    the `--target` device's profile on this host.
    """
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if stage in {"measure", "sim_same"}:
        if not _matches_filter(_device_filter_candidates(current_device), only_devices):
            return []
        return [(current_device, None)]
    device_token, _ = _parse_target(args)
    if device_token is None:
        return []
    devices_by_id = {str(device["id"]): device for device in _devices(spec)}
    if device_token not in devices_by_id:
        raise SystemExit(f"--target device {device_token!r} not in spec devices: {sorted(devices_by_id)}")
    target = devices_by_id[device_token]
    if not _matches_filter(_device_filter_candidates(target), only_devices):
        return []
    return [(target, current_device)]


def _seed_set(spec: dict[str, Any], args: argparse.Namespace) -> list[int]:
    raw = args.seed_set if getattr(args, "seed_set", None) is not None else spec.get("seed_set", [0])
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    return [int(item) for item in items]


def _n_episodes(spec: dict[str, Any], args: argparse.Namespace) -> int:
    if bool(getattr(args, "smoke", False)):
        return int(getattr(args, "smoke_episodes", 1))
    if getattr(args, "n_episodes", None) is not None:
        return int(args.n_episodes)
    return int(spec.get("n_episodes", 1))


def _profiles_dir_for_args(args: argparse.Namespace, out: Path) -> Path:
    raw = getattr(args, "profiles_dir", None)
    if raw is not None:
        return Path(raw)
    if bool(getattr(args, "smoke", False)):
        return out / "profiles_smoke"
    return out / "profiles"


def _spec_with_cli_overrides(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply global CLI overrides to a copy of the protocol spec (aggregate path)."""
    updated = deepcopy(spec)
    if getattr(args, "seed_set", None) is not None:
        updated["seed_set"] = _seed_set(spec, args)
    if getattr(args, "n_episodes", None) is not None:
        updated["n_episodes"] = int(args.n_episodes)
    if getattr(args, "checkpoint_kind", None) is not None:
        updated["checkpoint_kind"] = str(args.checkpoint_kind)
    return updated


def _filtered_spec_for_aggregate(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Filter a copied spec down to the rows/devices/experiments selected for aggregate."""
    updated = deepcopy(spec)
    only_experiments = _csv_filter(getattr(args, "only_experiment", None))
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if only_devices:
        updated["devices"] = [
            device for device in _devices(updated) if _matches_filter(_device_filter_candidates(device), only_devices)
        ]
    updated["rows"] = _iter_selected_rows(updated, args)
    experiments = updated.get("experiments", {})
    for experiment_id in list(experiments):
        if not _matches_filter([experiment_id], only_experiments):
            experiments.pop(experiment_id)
    return updated


def _csv_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _matches_filter(candidates: list[Any], filters: set[str]) -> bool:
    if not filters:
        return True
    candidate_tokens: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_tokens.update({str(candidate), slugify(str(candidate))})
    filter_tokens: set[str] = set()
    for item in filters:
        filter_tokens.update({str(item), slugify(str(item))})
    return bool(candidate_tokens & filter_tokens)


def _device_filter_candidates(device: dict[str, Any]) -> list[Any]:
    """Return filter tokens that can identify one hardware entry."""
    return [
        device.get("id"),
        device.get("display_name"),
        device.get("profile_id"),
        *device.get("aliases", []),
    ]


def _output_root(spec: dict[str, Any], out: Path | None) -> Path:
    return Path(out) if out is not None else Path(str(spec.get("output_root", "results/cross_device")))
