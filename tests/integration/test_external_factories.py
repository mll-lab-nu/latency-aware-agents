"""Long-lived custom factory contract across CLI and real subprocess boundaries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from helpers import REPO_ROOT


@pytest.fixture
def user_project(tmp_path):
    project = tmp_path / "user_project"
    project.mkdir()
    shutil.copyfile(
        Path(__file__).parent / "fixtures/external_components.py",
        project / "outside_lagen.py",
    )
    return project


def _run_user_project(project, mode, policy_factory="make_policy"):
    output = project / "output"
    config = {
        "experiment": {"name": "outside_project", "seed": 7},
        "executor": {"mode": mode, "inference_devices": ["cpu"]},
        "env": {
            "name": "custom", "factory": "outside_lagen:make_env",
            "env_fps": 20, "obs_fps": 20, "noop_action": 0,
        },
        "policy": {"type": "custom", "factory": f"outside_lagen:{policy_factory}"},
        "latency": {"method": "measured" if mode == "realtime" else "zero"},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "evaluation": {"eval_episodes": 2, "eval_max_steps": 4},
        "logging": {
            "output_dir": str(output), "save_step_records": True,
            "save_action_records": True, "save_latency_records": True,
            "realtime_pipeline_profile": mode == "realtime",
        },
        "factory_audit_path": str(project / "audit.jsonl"),
    }
    config_path = project / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(project), str(REPO_ROOT)]),
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg", "LATENCY_BENCH_ENV_BACKEND_START_METHOD": "spawn",
    }
    result = subprocess.run(
        [sys.executable, "-m", "latency_bench.run", "--config", str(config_path)],
        cwd=project, env=env, capture_output=True, text=True, timeout=60,
    )
    records = [json.loads(line) for line in (project / "audit.jsonl").read_text().splitlines()]
    return result, output, records


@pytest.mark.parametrize("mode", ["paused", "simulated", "realtime"])
def test_external_factories_run_from_user_project(user_project, mode):
    result, output, records = _run_user_project(user_project, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    env_pid = next(row["pid"] for row in records if row["event"] == "env_created")
    policy_pid = next(row["pid"] for row in records if row["event"] == "policy_created")
    assert (env_pid != policy_pid) == (mode == "realtime")
    assert any(row["event"] == "env_closed" and row["pid"] == env_pid for row in records)
    assert sum(row["event"] == "policy_reset" for row in records) >= 2
    metrics = [json.loads(line) for line in (output / "episode_metrics.jsonl").read_text().splitlines()]
    assert [row["episode_return"] for row in metrics] == [4.0, 4.0]
    if mode != "realtime":
        steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        for episode in range(2):
            assert [row["active_action"]["value"] for row in steps if row["episode_id"] == episode] == [1, 2, 3, 4]
    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert resolved["env"]["factory"] == "outside_lagen:make_env"
    assert resolved["policy"]["factory"] == "outside_lagen:make_policy"


@pytest.mark.parametrize("mode", ["paused", "simulated", "realtime"])
def test_external_constructor_failure_is_reported_and_environment_closed(user_project, mode):
    result, _, records = _run_user_project(user_project, mode, policy_factory="fail_policy")
    assert result.returncode != 0
    assert "user policy construction failed" in result.stderr
    assert any(row["event"] == "env_closed" for row in records)
