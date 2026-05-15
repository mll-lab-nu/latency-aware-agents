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
    assert fs_cfg["logging"]["wandb_group"] == "flappy-fs10-normal_n2_s1"
    assert fs_cfg["logging"]["wandb_name"] == "flappy-fs10-normal_n2_s1-s0"
    assert fs_cfg["logging"]["wandb_tags"] == "factor_sweep,flappy,frame_stack,normal,normal_n2_s1,fs10,seed0"
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
    assert obs_cfg["logging"]["wandb_group"] == "flappy-obs15-uniform_u1_3"
    assert obs_cfg["logging"]["wandb_name"] == "flappy-obs15-uniform_u1_3-s0"
    assert obs_cfg["logging"]["wandb_tags"] == "factor_sweep,flappy,obs_fps,uniform,uniform_u1_3,obs15,seed0"

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
    assert eval_cfg["logging"]["wandb_group"] == "flappy-fs4-fixed_l1"
    assert "wandb_name" not in eval_cfg["logging"]
    assert eval_cfg["logging"]["wandb_tags"] == "factor_sweep,flappy,frame_stack,fixed,fixed_l1,fs4,seed0"
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


def test_flappy_factor_sweep_generator_cli_seeds_override_spec(tmp_path):
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
            "--seeds",
            "42,114514",
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 70
    assert {row["seed"] for row in rows} == {"42", "114514"}
    assert not (generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_0").exists()

    seed_42_cfg = yaml.safe_load(
        (
            generated_root
            / "frame_stack"
            / "fixed_l1"
            / "fs4"
            / "seed_42"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert seed_42_cfg["experiment"]["seed"] == 42
    assert seed_42_cfg["experiment"]["name"] == "flappy_frame_stack_fixed_l1_fs4_seed42"
    assert seed_42_cfg["logging"]["wandb_group"] == "flappy-fs4-fixed_l1"
    assert seed_42_cfg["logging"]["wandb_name"] == "flappy-fs4-fixed_l1-s42"
    assert seed_42_cfg["logging"]["wandb_tags"] == "factor_sweep,flappy,frame_stack,fixed,fixed_l1,fs4,seed42"
    assert seed_42_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "frame_stack" / "fixed_l1" / "fs4" / "seed_42"
    )

    seed_114514_cfg = yaml.safe_load(
        (
            generated_root
            / "obs_fps"
            / "uniform_u1_3"
            / "obs10"
            / "seed_114514"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert seed_114514_cfg["experiment"]["seed"] == 114514
    assert seed_114514_cfg["experiment"]["name"] == "flappy_obs_fps_uniform_u1_3_obs10_seed114514"
    assert seed_114514_cfg["logging"]["wandb_group"] == "flappy-obs10-uniform_u1_3"
    assert seed_114514_cfg["logging"]["wandb_name"] == "flappy-obs10-uniform_u1_3-s114514"
    assert seed_114514_cfg["logging"]["wandb_tags"] == (
        "factor_sweep,flappy,obs_fps,uniform,uniform_u1_3,obs10,seed114514"
    )


def test_flappy_factor_sweep_wrapper_forwards_seed_override(tmp_path):
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
            "SEEDS": "0,42",
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
    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 70
    assert {row["seed"] for row in rows} == {"0", "42"}

    seed_42_cfg = yaml.safe_load(
        (
            generated_root
            / "obs_fps"
            / "fixed_l2"
            / "obs10"
            / "seed_42"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert seed_42_cfg["experiment"]["seed"] == 42
    assert seed_42_cfg["backend"]["device"] == "cpu"
    assert seed_42_cfg["backend"]["train_dir"] == str(train_dir)
    assert seed_42_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "obs_fps" / "fixed_l2" / "obs10" / "seed_42"
    )


def test_factor_sweep_wrapper_generates_selected_envs(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "ENVS": "demon_attack,deadly_corridor",
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
    assert (generated_root / "demon_attack" / "manifest.tsv").is_file()
    assert (generated_root / "deadly_corridor" / "manifest.tsv").is_file()
    train_cfg = yaml.safe_load(
        (
            generated_root
            / "deadly_corridor"
            / "obs_fps"
            / "fixed_l2"
            / "obs7p5"
            / "seed_0"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["backend"]["device"] == "cpu"
    assert train_cfg["backend"]["train_dir"] == str(train_dir / "deadly_corridor")
    assert train_cfg["env"]["obs_fps"] == pytest.approx(7.5)
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "deadly_corridor" / "train" / "obs_fps" / "fixed_l2" / "obs7p5" / "seed_0"
    )


def test_factor_sweep_wrapper_forwards_seed_override(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.update(
        {
            "ENVS": "demon_attack",
            "GENERATED_ROOT": str(generated_root),
            "OUTPUT_ROOT": str(output_root),
            "TRAIN_DIR": str(train_dir),
            "DEVICE": "cpu",
            "SEEDS": "0,42,114514",
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
    rows = _read_manifest(generated_root / "demon_attack" / "manifest.tsv")
    assert len(rows) == 120
    assert {row["seed"] for row in rows} == {"0", "42", "114514"}

    seed_42_cfg = yaml.safe_load(
        (
            generated_root
            / "demon_attack"
            / "obs_fps"
            / "fixed_l2"
            / "obs7p5"
            / "seed_42"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert seed_42_cfg["experiment"]["seed"] == 42
    assert seed_42_cfg["backend"]["train_dir"] == str(train_dir / "demon_attack")
    assert seed_42_cfg["logging"]["output_dir"] == str(
        output_root / "demon_attack" / "train" / "obs_fps" / "fixed_l2" / "obs7p5" / "seed_42"
    )

    seed_114514_cfg = yaml.safe_load(
        (
            generated_root
            / "demon_attack"
            / "frame_stack"
            / "fixed_l2"
            / "fs4"
            / "seed_114514"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert seed_114514_cfg["experiment"]["seed"] == 114514


@pytest.mark.parametrize(
    ("env_name", "spec_path", "expected_policy"),
    [
        ("demon_attack", "configs/experiments/demon_attack/factor_sweeps/sweep.yaml", "demon_attack_sf"),
        ("deadly_corridor", "configs/experiments/deadly_corridor/factor_sweeps/sweep.yaml", "deadly_corridor_sf"),
    ],
)
def test_factor_sweep_generator_emits_non_flappy_configs(tmp_path, env_name, spec_path, expected_policy):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_flappy_factor_sweeps(
        [
            "--spec",
            spec_path,
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 40
    assert len({row["train_config"] for row in rows}) == 40

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "obs_fps"
            / "fixed_l2"
            / "obs7p5"
            / "seed_0"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["experiment"]["name"] == f"{env_name}_obs_fps_fixed_l2_obs7p5_seed0"
    assert train_cfg["env"]["name"] == env_name
    assert train_cfg["env"]["frame_stack"] == 4
    assert train_cfg["env"]["obs_fps"] == pytest.approx(7.5)
    assert train_cfg["logging"]["wandb_name"] == f"{env_name.replace('_', '-')}-obs7p5-fixed_l2-s0"
    assert train_cfg["logging"]["wandb_group"] == f"{env_name.replace('_', '-')}-obs7p5-fixed_l2"
    assert train_cfg["logging"]["wandb_tags"] == (
        f"factor_sweep,{env_name.replace('_', '-')},obs_fps,fixed,fixed_l2,obs7p5,seed0"
    )
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "obs_fps" / "fixed_l2" / "obs7p5" / "seed_0"
    )

    bench_cfg = load_config(
        generated_root / "obs_fps" / "fixed_l2" / "obs7p5" / "seed_0" / "latency_bench_eval.yaml"
    )
    assert bench_cfg["policy"]["type"] == expected_policy
    assert bench_cfg["env"]["obs_fps"] == pytest.approx(7.5)
    if env_name == "demon_attack":
        assert bench_cfg["env"]["terminal_on_life_loss"] is True
        assert bench_cfg["env"]["clip_reward"] is False
    if env_name == "deadly_corridor":
        assert bench_cfg["env"]["noop_action"] == [0, 0, 0, 0]
