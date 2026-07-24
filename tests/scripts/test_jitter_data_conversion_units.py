from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from helpers import REPO_ROOT


def _run_with_fake_conda(tmp_path: Path, script: Path, env_name: str) -> list[str]:
    calls_path = tmp_path / "conda_calls.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text(
        f"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "{calls_path}"
""",
        encoding="utf-8",
    )
    conda.chmod(0o755)

    subprocess.run(
        ["bash", str(script), env_name],
        check=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    return calls_path.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("env_name", "raw_name", "converter_name"),
    [
        ("flappy", "flappy_fix_latency_2_200ep", "convert_flappy_to_starvla_lerobot.py"),
        (
            "demon_attack",
            "demon_attack_fix_latency_2_200ep",
            "convert_demon_attack_to_starvla_lerobot.py",
        ),
        (
            "deadly_corridor",
            "deadly_corridor_fix_latency_2_1000ep",
            "convert_deadly_corridor_to_starvla_lerobot.py",
        ),
    ],
)
def test_current_jitter_rollout_converts_raw_latency_to_observation_ids(
    tmp_path: Path,
    env_name: str,
    raw_name: str,
    converter_name: str,
) -> None:
    calls = _run_with_fake_conda(
        tmp_path,
        REPO_ROOT / "scripts/experiments/jitter_destructiveness/rollout_data.sh",
        env_name,
    )
    conversion_call = next(call for call in calls if converter_name in call)
    metadata = REPO_ROOT / "teacher_datasets/jitter_destructiveness" / raw_name / "metadata.json"

    assert f"--source-metadata {metadata}" in conversion_call
    assert "--source-latency-column latency_raw_frames" in conversion_call
    assert "--target-latency-unit observation_steps" in conversion_call


@pytest.mark.parametrize(
    ("env_name", "raw_name", "converter_name"),
    [
        ("flappy", "flappy_fix_latency_2_200ep", "convert_flappy_to_starvla_lerobot.py"),
        (
            "demon_attack",
            "demon_attack_fix_latency_2_200ep",
            "convert_demon_attack_to_starvla_lerobot.py",
        ),
        (
            "deadly_corridor",
            "deadly_corridor_fix_latency_2_1000ep",
            "convert_deadly_corridor_to_starvla_lerobot.py",
        ),
    ],
)
def test_pinned_jitter_data_uses_its_canonical_metadata(
    tmp_path: Path,
    env_name: str,
    raw_name: str,
    converter_name: str,
) -> None:
    calls = _run_with_fake_conda(
        tmp_path,
        REPO_ROOT / "scripts/experiments/jitter_destructiveness/prepare_data.sh",
        env_name,
    )
    conversion_call = next(call for call in calls if converter_name in call)
    metadata = (
        REPO_ROOT
        / "third_party/starVLA/data/jitter_destructiveness/raw"
        / raw_name
        / "metadata.json"
    )

    assert f"--source-metadata {metadata}" in conversion_call
    assert "--source-latency-column latency_raw_frames" in conversion_call
    assert "--target-latency-unit observation_steps" in conversion_call
