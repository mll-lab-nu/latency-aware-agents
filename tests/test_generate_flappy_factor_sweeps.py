from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from latency_bench.core.config import load_config
from scripts.sweeps.generate_flappy_factor_sweeps import main as generate_flappy_factor_sweeps


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_flappy_factor_sweep_generator_emits_matrix_configs(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_flappy_factor_sweeps(
        [
            "--spec",
            "configs/experiments/flappy/factor_sweeps/sweep.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 35
    assert len({row["train_config"] for row in rows}) == 35

    fs_normal = (
        generated_root
        / "frame_stack"
        / "normal_n2_s1"
        / "fs10"
        / "seed_0"
        / "train.yaml"
    )
    fs_cfg = yaml.safe_load(fs_normal.read_text(encoding="utf-8"))
    assert fs_cfg["experiment"]["name"] == "flappy_frame_stack_normal_n2_s1_fs10_seed0"
    assert fs_cfg["env"]["frame_stack"] == 10
    assert fs_cfg["env"]["obs_fps"] == 30
    assert fs_cfg["latency"]["type"] == "normal_distribution"
    assert fs_cfg["latency"]["mean_latency_ms"] == pytest.approx(2 * 1000.0 / 30.0)
    assert fs_cfg["latency"]["std_latency_ms"] == pytest.approx(1 * 1000.0 / 30.0)
    assert fs_cfg["latency"]["max_latency_ms"] == pytest.approx(15 * 1000.0 / 30.0)
    assert fs_cfg["logging"]["wandb_group"] == "flappy-frame_stack"
    assert fs_cfg["logging"]["wandb_name"] == "flappy-fs10-normal_n2_s1-s0"
    assert fs_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "frame_stack" / "normal_n2_s1" / "fs10" / "seed_0"
    )

    obs_uniform = (
        generated_root
        / "obs_fps"
        / "uniform_u1_3"
        / "obs15"
        / "seed_0"
        / "train.yaml"
    )
    obs_cfg = yaml.safe_load(obs_uniform.read_text(encoding="utf-8"))
    assert obs_cfg["experiment"]["name"] == "flappy_obs_fps_uniform_u1_3_obs15_seed0"
    assert obs_cfg["env"]["frame_stack"] == 4
    assert obs_cfg["env"]["obs_fps"] == 15
    assert obs_cfg["latency"]["type"] == "uniform_distribution"
    assert obs_cfg["latency"]["min_latency_ms"] == pytest.approx(1 * 1000.0 / 30.0)
    assert obs_cfg["latency"]["max_latency_ms"] == pytest.approx(3 * 1000.0 / 30.0)
    assert obs_cfg["logging"]["wandb_group"] == "flappy-obs_fps"
    assert obs_cfg["logging"]["wandb_name"] == "flappy-obs15-uniform_u1_3-s0"

    fixed_cfg = yaml.safe_load(
        (
            generated_root
            / "frame_stack"
            / "fixed_l1"
            / "fs4"
            / "seed_0"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert fixed_cfg["latency"]["type"] == "fixed"
    assert fixed_cfg["latency"]["fixed_latency_ms"] == pytest.approx(1 * 1000.0 / 30.0)

    eval_cfg = yaml.safe_load(
        (
            generated_root
            / "frame_stack"
            / "fixed_l1"
            / "fs4"
            / "seed_0"
            / "latency_bench_eval.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "eval_suites" in eval_cfg["evaluation"]
    assert eval_cfg["backend"]["run_mode"] == "eval"
    loaded_eval_cfg = load_config(
        generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_0" / "latency_bench_eval.yaml"
    )
    assert loaded_eval_cfg["policy"]["type"] == "flappy_sf"


def test_flappy_factor_sweep_wrapper_uses_env_overrides(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "GENERATED_ROOT": str(generated_root),
            "OUTPUT_ROOT": str(output_root),
            "TRAIN_DIR": str(train_dir),
            "DEVICE": "cpu",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/bash_scripts/generate_flappy_factor_sweeps.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    train_cfg = yaml.safe_load(
        (
            generated_root
            / "obs_fps"
            / "fixed_l2"
            / "obs10"
            / "seed_0"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["backend"]["device"] == "cpu"
    assert train_cfg["backend"]["train_dir"] == str(train_dir)
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "obs_fps" / "fixed_l2" / "obs10" / "seed_0"
    )
