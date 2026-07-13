from pathlib import Path

from latency_bench.cross_device.artifacts import (
    profile_dir,
    profile_ref,
    raw_session_dir,
    raw_workload_dir,
    simulation_seed_dir,
    target_ref,
)
from latency_bench.cross_device.spec import load_workload


def test_artifact_paths_do_not_depend_on_real_or_sim_counts(tmp_path: Path):
    target = target_ref("a100-sxm4-80gb", "instance_01b95985", "openvla-flappy")
    profile = profile_ref(target, "fit-0")

    assert raw_workload_dir(tmp_path, target) == (
        tmp_path / "raw/a100-sxm4-80gb/instance_01b95985/openvla-flappy"
    )
    assert raw_session_dir(tmp_path, target, 9) == (
        tmp_path / "raw/a100-sxm4-80gb/instance_01b95985/openvla-flappy/sessions/9"
    )
    assert profile_dir(tmp_path, profile) == (
        tmp_path / "profiles/a100-sxm4-80gb/instance_01b95985/openvla-flappy/fit-0"
    )
    assert simulation_seed_dir(tmp_path, profile, "temporal", 42) == (
        tmp_path / "sim/a100-sxm4-80gb/instance_01b95985/openvla-flappy/fit-0/temporal/42"
    )


def test_workload_config_has_no_fixed_artifact_cardinality():
    config = load_workload("configs/experiments/cross_device.yaml")

    assert config["artifact_root"] == "results/cross_device"
    assert config["workload"]["id"] == "openvla-flappy"
    assert "study_id" not in config
    assert "sessions" not in config
    assert "latency_seeds" not in config
