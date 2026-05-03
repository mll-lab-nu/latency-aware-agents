from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_eval_bash_scripts_are_batch_eval_wrappers():
    scripts = {
        "eval_flappy.sh": "flappy",
        "eval_demon_attack.sh": "demon_attack",
        "eval_deadly_corridor.sh": "deadly_corridor",
    }

    for script_name, env_name in scripts.items():
        script_path = REPO_ROOT / "scripts" / "bash_scripts" / script_name

        assert script_path.is_file()
        assert os.access(script_path, os.X_OK)

        syntax = subprocess.run(
            ["bash", "-n", str(script_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

        text = script_path.read_text(encoding="utf-8")
        assert f'DEFAULT_CONFIG_ROOT="configs/train/{env_name}"' in text
        assert 'python -m training.train --config "${CONFIG_DIR}/eval.yaml"' in text
        assert 'python -m latency_bench.run --config "${CONFIG_DIR}/latency_bench_eval.yaml"' in text
        assert "${CONFIG_DIR}/train.yaml" not in text
        assert "sort -V" in text
        assert "/config.json" in text
        assert "Skipping" in text


def test_eval_bash_scripts_reject_manual_latency_argument():
    for script_name in ("eval_flappy.sh", "eval_demon_attack.sh", "eval_deadly_corridor.sh"):
        script_path = REPO_ROOT / "scripts" / "bash_scripts" / script_name
        result = subprocess.run(
            ["bash", str(script_path), "0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert "does not take a latency argument" in result.stderr


def test_eval_bash_script_skips_missing_checkpoint_config(tmp_path):
    config_root = tmp_path / "demon_attack"
    config_dir = config_root / "latency_8" / "seed_0"
    config_dir.mkdir(parents=True)
    eval_yaml = """
experiment:
  name: demon_attack_sf_same_latency_l8_seed0
backend:
  train_dir: results/checkpoints
""".strip()
    (config_dir / "eval.yaml").write_text(eval_yaml + "\n", encoding="utf-8")
    (config_dir / "latency_bench_eval.yaml").write_text(eval_yaml + "\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "bash_scripts" / "eval_demon_attack.sh")],
        cwd=REPO_ROOT,
        env={**os.environ, "LATENCY_BENCH_EVAL_CONFIG_ROOT": str(config_root)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Skipping" in result.stderr
    assert "demon_attack_sf_same_latency_l8_seed0" in result.stderr
    assert "no checkpoint-backed eval configs" in result.stderr


def test_eval_bash_script_continues_after_latency_bench_failure(tmp_path):
    config_root = tmp_path / "demon_attack"
    train_dir = tmp_path / "checkpoints"
    for latency in (8, 9):
        config_dir = config_root / f"latency_{latency}" / "seed_0"
        config_dir.mkdir(parents=True)
        eval_yaml = f"""
experiment:
  name: demon_attack_sf_same_latency_l{latency}_seed0
backend:
  train_dir: {train_dir}
""".strip()
        (config_dir / "eval.yaml").write_text(eval_yaml + "\n", encoding="utf-8")
        (config_dir / "latency_bench_eval.yaml").write_text(eval_yaml + "\n", encoding="utf-8")
        checkpoint_dir = train_dir / f"demon_attack_sf_same_latency_l{latency}_seed0"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "python_calls.txt"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s\\n' \"$*\" >> \"$LATENCY_BENCH_FAKE_PYTHON_CALLS\"",
                "if [[ \"$*\" == *'-m latency_bench.run'* && \"$*\" == *'latency_8'* ]]; then",
                "  exit 17",
                "fi",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "bash_scripts" / "eval_demon_attack.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "LATENCY_BENCH_EVAL_CONFIG_ROOT": str(config_root),
            "LATENCY_BENCH_FAKE_PYTHON_CALLS": str(calls_path),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    calls = calls_path.read_text(encoding="utf-8")
    assert "-m latency_bench.run --config" in calls
    assert "latency_8/seed_0/latency_bench_eval.yaml" in calls
    assert "latency_9/seed_0/eval.yaml" in calls
    assert "latency_9/seed_0/latency_bench_eval.yaml" in calls
    assert "latency_bench eval command failed" in result.stderr
