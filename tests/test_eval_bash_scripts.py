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
        assert 'CHECKPOINT_KINDS=(latest best)' in text
        assert "--checkpoint-kind" in text
        assert "--output-dir" in text
        assert 'python -m training.train \\' in text
        assert 'python -m latency_bench.run \\' in text
        assert "${CONFIG_DIR}/train.yaml" not in text
        assert "sort -V" in text
        assert "/config.json" in text
        assert "checkpoint_p0" in text
        assert "outputs/eval/${checkpoint_kind}/" in text
        assert "outputs/latency_bench/${checkpoint_kind}/" in text
        assert "Skipping" in text


def test_openvla_install_script_bootstraps_rlinf_oft_environment():
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "install_openvla.sh"

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
    assert 'RLINF_DIR="${RLINF_DIR:-${REPO_ROOT}/RL-inf}"' in text
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-latency-openvla}"' in text
    assert 'PYTHON_VERSION="${PYTHON_VERSION:-3.10}"' in text
    assert 'TORCH_SPEC="${TORCH_SPEC:-torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0}"' in text
    assert 'TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"' in text
    assert 'conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y' in text
    assert 'conda activate "${CONDA_ENV_NAME}"' in text
    assert 'python -m pip install -e "${RLINF_DIR}[embodied]"' in text
    assert 'OPENVLA_OFT_SPEC="${OPENVLA_OFT_SPEC:-git+https://github.com/moojink/openvla-oft.git}"' in text
    assert "OPENVLA_OFT_RUNTIME_DEPS=(" in text
    assert 'python -m pip install "${OPENVLA_OFT_RUNTIME_DEPS[@]}"' in text
    assert 'python -m pip install -e "${OPENVLA_OFT_DIR}" --no-build-isolation --no-deps' in text
    assert 'python -m pip install "${OPENVLA_OFT_SPEC}" --no-build-isolation --no-deps' in text
    assert "python -m pip install -e \"${REPO_ROOT}/flappy-bird-gymnasium\"" in text
    assert 'activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"' in text
    assert "PYTHONPATH=\"${REPO_ROOT}:${RLINF_DIR}" in text
    assert 'Version(torch.__version__.split("+", 1)[0]) < Version("2.5.0")' in text
    assert 'hasattr(torch, "Event")' in text
    assert "from rlinf.models.embodiment.openvla_oft.rlinf import get_model" in text
    assert "OpenVLAOFTForRLActionPrediction" in text
    assert "import latency_bench.policy.flappy_openvla" in text
    assert "torch==2.2.0" not in text
    assert "torchvision==0.17.0" not in text
    assert "torchaudio==2.2.0" not in text
    assert "OPENVLA_VENV_DIR" not in text
    assert "--venv" not in text
    assert ".venv-latency-openvla" not in text
    assert "pip install -e \"${REPO_ROOT}/openvla\"" not in text


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
logging:
  output_dir: {tmp_path}/outputs/eval/demon_attack/latency_{latency}/seed_0
""".strip()
        latency_bench_yaml = f"""
experiment:
  name: demon_attack_sf_same_latency_l{latency}_seed0
backend:
  train_dir: {train_dir}
logging:
  output_dir: {tmp_path}/outputs/latency_bench/demon_attack/train_latency_{latency}/seed_0
""".strip()
        (config_dir / "eval.yaml").write_text(eval_yaml + "\n", encoding="utf-8")
        (config_dir / "latency_bench_eval.yaml").write_text(latency_bench_yaml + "\n", encoding="utf-8")
        checkpoint_dir = train_dir / f"demon_attack_sf_same_latency_l{latency}_seed0"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")
        policy_checkpoint_dir = checkpoint_dir / "checkpoint_p0"
        policy_checkpoint_dir.mkdir()
        (policy_checkpoint_dir / "checkpoint_000000001_000000000001.pth").write_text("", encoding="utf-8")
        (policy_checkpoint_dir / "best_000000001_000000000001.pth").write_text("", encoding="utf-8")

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
    assert "--checkpoint-kind latest" in calls
    assert "--checkpoint-kind best" in calls
    assert f"--output-dir {tmp_path}/outputs/eval/latest/demon_attack/latency_8/seed_0" in calls
    assert f"--output-dir {tmp_path}/outputs/eval/best/demon_attack/latency_8/seed_0" in calls
    assert f"--output-dir {tmp_path}/outputs/latency_bench/latest/demon_attack/train_latency_8/seed_0" in calls
    assert f"--output-dir {tmp_path}/outputs/latency_bench/best/demon_attack/train_latency_8/seed_0" in calls
    assert "latency_8/seed_0/latency_bench_eval.yaml" in calls
    assert "latency_9/seed_0/eval.yaml" in calls
    assert "latency_9/seed_0/latency_bench_eval.yaml" in calls
    assert "latency_bench eval command failed" in result.stderr


def test_eval_bash_script_reports_missing_best_checkpoint_but_runs_latest(tmp_path):
    config_root = tmp_path / "demon_attack"
    train_dir = tmp_path / "checkpoints"
    config_dir = config_root / "latency_8" / "seed_0"
    config_dir.mkdir(parents=True)
    eval_yaml = f"""
experiment:
  name: demon_attack_sf_same_latency_l8_seed0
backend:
  train_dir: {train_dir}
logging:
  output_dir: {tmp_path}/outputs/eval/demon_attack/latency_8/seed_0
""".strip()
    latency_bench_yaml = f"""
experiment:
  name: demon_attack_sf_same_latency_l8_seed0
backend:
  train_dir: {train_dir}
logging:
  output_dir: {tmp_path}/outputs/latency_bench/demon_attack/train_latency_8/seed_0
""".strip()
    (config_dir / "eval.yaml").write_text(eval_yaml + "\n", encoding="utf-8")
    (config_dir / "latency_bench_eval.yaml").write_text(latency_bench_yaml + "\n", encoding="utf-8")
    checkpoint_dir = train_dir / "demon_attack_sf_same_latency_l8_seed0"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")
    policy_checkpoint_dir = checkpoint_dir / "checkpoint_p0"
    policy_checkpoint_dir.mkdir()
    (policy_checkpoint_dir / "checkpoint_000000001_000000000001.pth").write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "python_calls.txt"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s\\n' \"$*\" >> \"$LATENCY_BENCH_FAKE_PYTHON_CALLS\"",
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
    assert "--checkpoint-kind latest" in calls
    assert "--checkpoint-kind best" not in calls
    assert "missing checkpoint matching" in result.stderr
    assert "best_*" in result.stderr
