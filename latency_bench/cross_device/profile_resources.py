"""Profile resource checks and cache hydration for cross-device planning."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from latency_bench.cross_device.models import ProtocolTask
from latency_bench.cross_device.sync import ensure_profile_cached


def profile_resource_skip_reasons(
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
        hf_profile_repo = os.environ.get("HF_PROFILE_REPO")
        if hf_profile_repo:
            ensure_profile_cached(
                profile_path=task.target_profile_path,
                profiles_dir=profiles_dir,
                hf_repo=hf_profile_repo,
                repo_type=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"),
                revision=os.environ.get("HF_PROFILE_REVISION"),
            )

    if (
        stage in {"sim_same", "sim_cross"}
        and task.target_profile_path is not None
        and not task.target_profile_path.exists()
        and not args.dry_run
    ):
        reasons.append(f"missing profile: {task.target_profile_path}")

    return reasons
