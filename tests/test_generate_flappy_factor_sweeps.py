from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from latency_bench.core.config import load_config
from scripts.sweeps.generate_flappy_factor_sweeps import main as generate_flappy_factor_sweeps


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SEEDS = tuple(range(10, 20))


def _read_manifest(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_context_window_spec_emits_official_seed_matrix(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_flappy_factor_sweeps(
        [
            "--spec",
            "configs/experiments/flappy/factor_sweeps/context_window/sweep.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 250
    assert len({row["train_config"] for row in rows}) == 250
    assert {row["seed"] for row in rows} == {str(seed) for seed in OFFICIAL_SEEDS}
    assert {row["sweep"] for row in rows} == {"frame_stack"}
    assert {row["frame_stack"] for row in rows} == {"4", "5", "6", "7", "8"}
    assert {row["observation_stride"] for row in rows} == {"1"}

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "frame_stack"
            / "normal_n2_s1"
            / "fs8"
            / "seed_10"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["experiment"]["name"] == "flappy_frame_stack_normal_n2_s1_fs8_seed10"
    assert train_cfg["experiment"]["seed"] == 10
    assert train_cfg["env"]["frame_stack"] == 8
    assert train_cfg["env"]["obs_fps"] == pytest.approx(30)
    assert train_cfg["latency"]["mean_latency_ms"] == pytest.approx(2 * 1000.0 / 30.0)
    assert train_cfg["logging"]["wandb_group"] == "flappy-fs8-normal_n2_s1"
    assert train_cfg["logging"]["wandb_name"] == "flappy-fs8-normal_n2_s1-s10"
    assert train_cfg["logging"]["wandb_tags"] == [
        "factor_sweep",
        "flappy",
        "frame_stack",
        "normal",
        "normal_n2_s1",
        "fs8",
        "seed10",
    ]
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "frame_stack" / "normal_n2_s1" / "fs8" / "seed_10"
    )
    assert "eval_interval_steps" not in train_cfg["evaluation"]

    eval_cfg = load_config(
        generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert eval_cfg["backend"]["run_mode"] == "eval"
    assert eval_cfg["backend"]["extra_args"] == ["--load_checkpoint_kind", "latest"]
    assert eval_cfg["policy"]["type"] == "flappy_sf"
    assert eval_cfg["policy"]["checkpoint_kind"] == "latest"
    assert "eval_interval_steps" not in eval_cfg["evaluation"]
    assert eval_cfg["evaluation"]["eval_suites"] == {
        "fixed": [[1]],
        "normal": [],
        "uniform": [],
    }

    normal_eval_cfg = load_config(
        generated_root / "frame_stack" / "normal_n2_s1" / "fs4" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert normal_eval_cfg["evaluation"]["eval_suites"] == {
        "fixed": [],
        "normal": [[2.0, 1.0, 0, 15]],
        "uniform": [],
    }

    uniform_eval_cfg = load_config(
        generated_root / "frame_stack" / "uniform_u1_3" / "fs4" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert uniform_eval_cfg["evaluation"]["eval_suites"] == {
        "fixed": [],
        "normal": [],
        "uniform": [[1, 3]],
    }

    assert train_cfg["evaluation"]["eval_suites"] == {
        "fixed": [],
        "normal": [[2.0, 1.0, 0, 15]],
        "uniform": [],
    }


def test_factor_sweep_specs_use_base_inheritance():
    spec_paths = [
        REPO_ROOT / "configs/experiments/flappy/factor_sweeps/context_window/sweep.yaml",
        REPO_ROOT / "configs/experiments/flappy/factor_sweeps/observation_stride/sweep.yaml",
        REPO_ROOT / "configs/experiments/demon_attack/factor_sweeps/context_window/sweep.yaml",
        REPO_ROOT / "configs/experiments/demon_attack/factor_sweeps/observation_stride/sweep.yaml",
        REPO_ROOT / "configs/experiments/deadly_corridor/factor_sweeps/context_window/sweep.yaml",
        REPO_ROOT / "configs/experiments/deadly_corridor/factor_sweeps/observation_stride/sweep.yaml",
    ]

    for spec_path in spec_paths:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert "__base__" in spec
        assert "base_config" not in spec


def test_observation_stride_spec_names_axis_by_stride_and_writes_obs_fps(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_flappy_factor_sweeps(
        [
            "--spec",
            "configs/experiments/demon_attack/factor_sweeps/observation_stride/sweep.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 250
    assert {row["seed"] for row in rows} == {str(seed) for seed in OFFICIAL_SEEDS}
    assert {row["sweep"] for row in rows} == {"observation_stride"}
    assert {row["observation_stride"] for row in rows} == {"2", "3", "4", "5", "6"}

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "observation_stride"
            / "fixed_l2"
            / "stride5"
            / "seed_10"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["experiment"]["name"] == "demon_attack_observation_stride_fixed_l2_stride5_seed10"
    assert train_cfg["env"]["frame_stack"] == 4
    assert train_cfg["env"]["obs_fps"] == pytest.approx(12)
    assert train_cfg["logging"]["wandb_group"] == "demon-attack-stride5-fixed_l2"
    assert train_cfg["logging"]["wandb_name"] == "demon-attack-stride5-fixed_l2-s10"
    assert train_cfg["logging"]["wandb_tags"] == [
        "factor_sweep",
        "demon-attack",
        "observation_stride",
        "fixed",
        "fixed_l2",
        "stride5",
        "seed10",
    ]
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "observation_stride" / "fixed_l2" / "stride5" / "seed_10"
    )
    assert "eval_interval_steps" not in train_cfg["evaluation"]


def test_observation_stride_spec_supports_fractional_deadly_corridor_obs_fps(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_flappy_factor_sweeps(
        [
            "--spec",
            "configs/experiments/deadly_corridor/factor_sweeps/observation_stride/sweep.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "observation_stride"
            / "fixed_l2"
            / "stride5"
            / "seed_10"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["env"]["obs_fps"] == pytest.approx(7)
    bench_cfg = load_config(
        generated_root / "observation_stride" / "fixed_l2" / "stride5" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert bench_cfg["policy"]["type"] == "deadly_corridor_sf"
    assert bench_cfg["env"]["noop_action"] == [0, 0, 0, 0]


def test_factor_sweep_wrapper_generates_split_sweep_folders(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "ENVS": "deadly_corridor",
            "SEEDS": "10",
            "GENERATED_ROOT": str(generated_root),
            "OUTPUT_ROOT": str(output_root),
            "TRAIN_DIR": str(train_dir),
            "DEVICE": "cpu",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/bash_scripts/generate_factor_sweeps.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    context_rows = _read_manifest(generated_root / "deadly_corridor" / "context_window" / "manifest.tsv")
    stride_rows = _read_manifest(generated_root / "deadly_corridor" / "observation_stride" / "manifest.tsv")
    assert len(context_rows) == 20
    assert len(stride_rows) == 25

    train_cfg = yaml.safe_load(Path(stride_rows[0]["train_config"]).read_text(encoding="utf-8"))
    assert train_cfg["backend"]["device"] == "cpu"
    assert train_cfg["backend"]["train_dir"] == str(train_dir / "deadly_corridor" / "observation_stride")


def test_factor_sweep_wrapper_can_generate_only_selected_sweep(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "ENVS": "flappy",
            "SWEEPS": "observation_stride",
            "SEEDS": "10,11",
            "GENERATED_ROOT": str(generated_root),
            "OUTPUT_ROOT": str(output_root),
            "TRAIN_DIR": str(train_dir),
            "DEVICE": "cpu",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/bash_scripts/generate_factor_sweeps.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not (generated_root / "flappy" / "context_window" / "manifest.tsv").exists()
    rows = _read_manifest(generated_root / "flappy" / "observation_stride" / "manifest.tsv")
    assert len(rows) == 50
    assert {row["seed"] for row in rows} == {"10", "11"}


def test_flappy_factor_sweep_wrapper_uses_split_defaults(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "SEEDS": "10",
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
    assert (generated_root / "context_window" / "manifest.tsv").is_file()
    assert (generated_root / "observation_stride" / "manifest.tsv").is_file()


def test_jitter_sweeps_keep_legacy_obs_fps_axis(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_flappy_factor_sweeps(
        [
            "--spec",
            "configs/experiments/flappy/jitter/sweep.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert rows[0]["sweep"] == "jitter"
    assert rows[0]["observation_stride"] == "2"
    train_cfg = yaml.safe_load(
        (generated_root / "jitter" / "fixed_l2" / "obs15" / "seed_0" / "train.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert train_cfg["experiment"]["name"] == "flappy_jitter_fixed_l2_obs15_seed0"
    assert train_cfg["logging"]["wandb_group"] == "flappy-obs15-fixed_l2"
