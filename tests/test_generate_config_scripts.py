from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generate_config_script_uses_env_overrides_for_one_distribution(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "ENVS": "flappy",
            "SEEDS": "0",
            "GENERATED_ROOT": str(generated_root),
            "TRAIN_DIR": str(train_dir),
            "OUTPUT_ROOT": str(output_root),
            "DEVICE": "gpu",
        }
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/bash_scripts/generate_config.sh",
            "--latency-distribution",
            "uniform",
            "--latency-min-frames",
            "0",
            "--latency-max-frames",
            "2",
            "--wandb-group",
            "flappy-uniform",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    train_path = generated_root / "flappy" / "latency_uniform_min_0_max_2" / "seed_0" / "train.yaml"
    train_cfg = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    assert train_cfg["backend"]["train_dir"] == str(train_dir)
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "flappy" / "latency_uniform_min_0_max_2" / "seed_0"
    )
    assert train_cfg["logging"]["wandb_group"] == "flappy-uniform"
    assert train_cfg["latency"]["type"] == "uniform_distribution"
    assert train_cfg["latency"]["min_latency_ms"] == 0.0
    assert train_cfg["latency"]["max_latency_ms"] == 2 * 1000.0 / 30.0
    assert "train_values" not in train_cfg["latency"]


def test_norm_and_uni_batch_script_rejects_top_level_latency_distribution_args():
    result = subprocess.run(
        [
            "bash",
            "scripts/bash_scripts/gen_config_norm_and_uni.sh",
            "--latency-distribution",
            "normal",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Do not pass latency distribution flags" in result.stderr
