from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _prepare_install_sh_env(tmp_path: Path, *, compute_caps: str) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda_base = tmp_path / "conda_base"
    (fake_conda_base / "etc" / "profile.d").mkdir(parents=True)
    calls_path = tmp_path / "install_calls.log"

    fake_conda_bin = fake_bin / "conda"
    _write_executable(
        fake_conda_bin,
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "info" && "${2:-}" == "--base" ]]; then
  printf '%s\\n' "$FAKE_CONDA_BASE"
  exit 0
fi
printf 'conda %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
exit 0
""",
    )

    (fake_conda_base / "etc" / "profile.d" / "conda.sh").write_text(
        """conda() {
  if [[ "${1:-}" == "activate" ]]; then
    printf 'conda activate %s\\n' "${2:-}" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
    return 0
  fi
  "$FAKE_CONDA_BIN" "$@"
}
""",
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
exit 0
""",
    )
    _write_executable(
        fake_bin / "wandb",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'wandb %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
exit 0
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "--query-gpu=compute_cap" && "${{2:-}}" == "--format=csv,noheader" ]]; then
  cat <<'EOF'
{compute_caps}EOF
  exit 0
fi
exit 1
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '[install] torch=2.10.0 cuda=13.0\\n'
  exit 0
fi
printf 'python %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
exit 0
""",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_CONDA_BASE": str(fake_conda_base),
        "FAKE_CONDA_BIN": str(fake_conda_bin),
        "LATENCY_BENCH_FAKE_INSTALL_CALLS": str(calls_path),
    }
    return env, calls_path


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
        assert 'python -m training.train \\' not in text
        assert 'python -m latency_bench.run \\' in text
        assert "${CONFIG_DIR}/train.yaml" not in text
        assert "${CONFIG_DIR}/eval.yaml" not in text
        assert "${CONFIG_DIR}/latency_bench_eval.yaml" in text
        assert "sort -V" in text
        assert "/config.json" in text
        assert "checkpoint_p0" in text
        assert "outputs/latency_bench/${checkpoint_kind}/" in text
        assert "Skipping" in text


def test_openvla_install_script_bootstraps_rlinf_oft_environment():
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "install_openvla.sh"
    requirements_path = REPO_ROOT / "requirements_openvla.txt"

    assert script_path.is_file()
    assert os.access(script_path, os.X_OK)
    assert requirements_path.is_file()

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    text = script_path.read_text(encoding="utf-8")
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-latency-openvla}"' in text
    assert 'PYTHON_VERSION="${PYTHON_VERSION:-3.10}"' in text
    assert 'REQUIREMENTS_FILE="${REPO_ROOT}/requirements_openvla.txt"' in text
    assert 'conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y' in text
    assert 'conda activate "${CONDA_ENV_NAME}"' in text
    assert 'python -m pip install -r "${REQUIREMENTS_FILE}"' in text
    assert (
        'python -m pip install "git+https://github.com/moojink/openvla-oft.git" '
        "--no-deps --no-build-isolation"
    ) in text
    assert 'find_spec("prismatic")' in text
    assert '(".__init__.py", "vla/__init__.py")' not in text
    assert '("__init__.py", "vla/__init__.py")' in text
    assert "Minimal init patched for latency_bench OpenVLA eval" in text
    assert 'PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/RL-inf' in text
    assert "from rlinf.models.embodiment.openvla_oft.rlinf import get_model" in text
    assert "import latency_bench.policy.flappy_openvla" in text
    assert "OPENVLA_OFT_RUNTIME_DEPS" not in text
    assert "activate_dir=" not in text

    requirements = requirements_path.read_text(encoding="utf-8")
    assert "-e ./RL-inf[embodied]" in requirements
    assert "-e ./flappy-bird-gymnasium" in requirements
    assert "torch==2.9.0+cu130" in requirements
    assert "tensorflow" not in requirements
    assert "tensorflow-datasets" not in requirements
    assert "tensorflow-graphics" not in requirements
    assert "dlimp" not in requirements
    assert "torch==2.2.0" not in text
    assert "torch==2.2.0" not in requirements
    assert "OPENVLA_VENV_DIR" not in text
    assert "--venv" not in text
    assert ".venv-latency-openvla" not in text
    assert "pip install -e \"${REPO_ROOT}/openvla\"" not in text


def test_flappy_cuda_extension_build_script_is_executable_bash():
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "build_flappy_cuda_extension.sh"

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
    assert "load_flappy_cuda_extension" in text
    assert "torch.cuda.synchronize" in text
    assert "cuda-toolkit[nvcc,cccl]==13.0.*" in text
    assert "nvcc --list-gpu-code" in text
    assert 'for cuda_lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib"' in text
    assert "libcudart.so" in text
    assert "LIBRARY_PATH" in text


def test_install_script_selects_cu130_for_compute_cap_10_gpu(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "selecting cu130" in result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130" in calls
    assert "python -m pip install -r requirements.txt" in calls


def test_install_script_allows_torch_profile_override(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="9.0\n")
    env["LATENCY_BENCH_TORCH_PROFILE"] = "cu128"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "torch_profile=cu128" in result.stdout
    calls = calls_path.read_text(encoding="utf-8")
    assert "python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128" in calls


def test_install_script_rejects_unknown_torch_profile(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n")
    env["LATENCY_BENCH_TORCH_PROFILE"] = "bogus"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unknown LATENCY_BENCH_TORCH_PROFILE='bogus'" in result.stderr
    assert not calls_path.exists()


def test_starvla_rl_games_installers_match_flash_attn_wheel_to_runtime(tmp_path):
    install_dir = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install"
    helper_path = install_dir / "flash_attn.sh"
    eval_extra_path = install_dir / "eval_extra.sh"
    model_scripts = [
        install_dir / "model" / "openvla.sh",
        install_dir / "model" / "pi0.sh",
        install_dir / "model" / "gr00t.sh",
    ]

    fake_python = tmp_path / "python"
    fake_log = tmp_path / "python.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf 'cp310\\n2.7\\ncu12\\nFALSE\\n'
  exit 0
fi
printf '%s\\n' "$*" >> "${PYTHON_FAKE_LOG}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    for script_path in [helper_path, eval_extra_path, *model_scripts]:
        syntax = subprocess.run(
            ["bash", "-n", str(script_path)],
            cwd=REPO_ROOT / "starVLA",
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

    expected_wheel = (
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-"
        "cp310-cp310-linux_x86_64.whl"
    )

    for script_path in model_scripts:
        fake_log.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=REPO_ROOT / "starVLA",
            env={
                **os.environ,
                "PYTHON_BIN": str(fake_python),
                "PYTHON_FAKE_LOG": str(fake_log),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert expected_wheel in fake_log.read_text(encoding="utf-8")


def test_starvla_install_stack_installs_eval_extra_requirements_after_env_deps(tmp_path):
    script_path = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "install_stack.sh"
    fake_python = tmp_path / "python"
    fake_log = tmp_path / "python.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf 'cp310\\n2.7\\ncu12\\nFALSE\\n'
  exit 0
fi
printf '%s | %s\\n' "$PWD" "$*" >> "${PYTHON_FAKE_LOG}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script_path), "--no-conda", "openvla", "flappy"],
        cwd=REPO_ROOT / "starVLA",
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "PYTHON_FAKE_LOG": str(fake_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = fake_log.read_text(encoding="utf-8")
    env_install_index = log.index("-m pip install flappy-bird-gymnasium")
    eval_extra_index = log.index(
        f"{REPO_ROOT} | -m pip install -r requirements_starvla_eval_extra.txt"
    )
    validation_index = log.index("-m compileall")
    assert env_install_index < eval_extra_index < validation_index


def test_starvla_bootstrap_split_envs_installs_eval_extra_requirements_per_model(tmp_path):
    script_path = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "bootstrap.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda_base = tmp_path / "conda"
    fake_profile_dir = fake_conda_base / "etc" / "profile.d"
    fake_profile_dir.mkdir(parents=True)
    fake_log = tmp_path / "python.log"

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf 'cp310\\n2.7\\ncu12\\nFALSE\\n'
  exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
  if [[ "${2:-}" == *"sys.version_info.major"* ]]; then
    printf '3.10\\n'
  fi
fi
printf '%s | %s\\n' "$CONDA_DEFAULT_ENV" "$*" >> "${PYTHON_FAKE_LOG}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "info" && "${{2:-}}" == "--base" ]]; then
  printf '%s\\n' "{fake_conda_base}"
  exit 0
fi
printf 'fake conda: %s\\n' "$*" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)

    (fake_profile_dir / "conda.sh").write_text(
        """conda() {
  case "$1 ${2:-}" in
    "env list")
      printf '# conda environments:\\n'
      ;;
    "create -n")
      return 0
      ;;
    "activate "*)
      export CONDA_DEFAULT_ENV="$2"
      ;;
    *)
      printf 'fake conda function: %s\\n' "$*" >&2
      return 1
      ;;
  esac
}
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--split-envs",
            "--env",
            "flappy",
            "--skip-validate",
        ],
        cwd=REPO_ROOT / "starVLA",
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PYTHON_FAKE_LOG": str(fake_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = fake_log.read_text(encoding="utf-8")
    assert log.count("-m pip install -r requirements_starvla_eval_extra.txt") == 3
    for model in ("openvla", "pi0", "gr00t"):
        assert (
            f"starvla_rl_games_{model} | -m pip install -r "
            "requirements_starvla_eval_extra.txt"
        ) in log


def test_starvla_flash_attn_helper_uses_python_abi_tag(tmp_path):
    helper_path = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "flash_attn.sh"
    (tmp_path / "torch.py").write_text(
        """__version__ = "2.7.1+cu128"

class _Version:
    cuda = "12.8"

class _CType:
    _GLIBCXX_USE_CXX11_ABI = True

version = _Version()
_C = _CType()
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(helper_path), "--print-url"],
        cwd=REPO_ROOT / "starVLA",
        env={
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "PYTHONPATH": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert result.returncode == 0, result.stderr
    assert f"{py_tag}-{py_tag}-linux_x86_64.whl" in result.stdout
    assert "cpython-" not in result.stdout


def test_starvla_flash_attn_helper_rejects_unsupported_runtime(tmp_path):
    helper_path = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "flash_attn.sh"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
printf 'cp310\\n2.5\\ncu11\\nFALSE\\n'
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(helper_path), "--print-url"],
        cwd=REPO_ROOT / "starVLA",
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported torch/CUDA runtime" in result.stderr


def test_starvla_rl_games_hfd_raw_download_convert_dry_run_single_dataset():
    script_path = (
        REPO_ROOT
        / "starVLA"
        / "examples"
        / "rl_games"
        / "scripts"
        / "download_convert_rl_games_raw.sh"
    )

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT / "starVLA",
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--hfd",
            "/opt/hfd.sh",
            "--workspace-dir",
            "/tmp/ws",
            "--env",
            "demon_attack",
            "--frame-stacks",
            "2",
            "--dry-run",
        ],
        cwd=REPO_ROOT / "starVLA",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "/opt/hfd.sh latency-sensitive-bench/demon_attack_fixed_l2_fs2 --dataset "
        "--local-dir /tmp/ws/outputs/rl_games_raw/demon_attack_fixed_l2_fs2"
    ) in result.stdout
    assert (
        "python examples/rl_games/data_conversion/convert_demon_attack_to_starvla_lerobot.py "
        "--dataset-name /tmp/ws/outputs/rl_games_raw/demon_attack_fixed_l2_fs2 "
        "--output-dir /tmp/ws/playground/Datasets/rl_games/demon_attack_fixed_l2_fs2 --force"
    ) in result.stdout


def test_starvla_rl_games_hfd_raw_download_convert_dry_run_defaults_to_all_datasets():
    script_path = (
        REPO_ROOT
        / "starVLA"
        / "examples"
        / "rl_games"
        / "scripts"
        / "download_convert_rl_games_raw.sh"
    )

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--hfd",
            "/opt/hfd.sh",
            "--workspace-dir",
            "/tmp/ws",
            "--dry-run",
        ],
        cwd=REPO_ROOT / "starVLA",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("/opt/hfd.sh latency-sensitive-bench/") == 12
    assert result.stdout.count("python examples/rl_games/data_conversion/convert_") == 12
    assert "latency-sensitive-bench/flappy_fixed_l2_fs1" in result.stdout
    assert "latency-sensitive-bench/deadly_corridor_fixed_l2_fs6" in result.stdout


def test_cross_device_openvla_script_forwards_filters_and_uses_openvla_runtime():
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "run_cross_device_openvla.sh"
    common_path = REPO_ROOT / "scripts" / "bash_scripts" / "cross_device_launcher_common.sh"

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    common_syntax = subprocess.run(
        ["bash", "-n", str(common_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert common_syntax.returncode == 0, common_syntax.stderr

    script = script_path.read_text(encoding="utf-8")
    common = common_path.read_text(encoding="utf-8")
    assert "RUNTIME_ENV=${RUNTIME_ENV:-openvla}" in script
    assert '--runtime-env "${RUNTIME_ENV}"' in script
    assert "cross_device_run_protocol_stage" in script
    assert "cross_device_finish_protocol_runs" in script
    assert "OPENVLA_CHECK_DEPS_ONLY" in script
    assert "check-deps" in script
    assert "cross_device_openvla_resolve_stages" in script
    assert "sim-same" in script
    for env_var, flag in (
        ("ONLY_ENV", "--only-env"),
        ("ONLY_POLICY", "--only-policy"),
        ("ONLY_DEVICE", "--only-device"),
        ("ONLY_EXPERIMENT", "--only-experiment"),
        ("MAX_RUNS", "--max-runs"),
    ):
        assert env_var in common
        assert flag in common
    assert "CONTINUE_ON_ERROR" in common
    assert "--continue-on-error" in common


def test_cross_device_common_builds_openvla_smoke_execution_args():
    common_path = REPO_ROOT / "scripts" / "bash_scripts" / "cross_device_launcher_common.sh"
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"""
            set -euo pipefail
            source {common_path}
            DRY_RUN=1
            OPENVLA_SMOKE=1
            OPENVLA_SMOKE_EPISODES=1
            OPENVLA_SMOKE_MAX_STEPS=7
            cross_device_optional_args
            for arg in "${{OPTIONAL_ARGS[@]}}"; do
                printf 'doctor:%s\\n' "$arg"
            done
            for arg in "${{EXECUTION_OPTIONAL_ARGS[@]}}"; do
                printf 'execution:%s\\n' "$arg"
            done
            """,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    doctor_args = [line.removeprefix("doctor:") for line in result.stdout.splitlines() if line.startswith("doctor:")]
    execution_args = [
        line.removeprefix("execution:")
        for line in result.stdout.splitlines()
        if line.startswith("execution:")
    ]
    assert "--smoke" not in doctor_args
    assert "--smoke" in execution_args
    assert ["--smoke-episodes", "1"] == execution_args[
        execution_args.index("--smoke-episodes") : execution_args.index("--smoke-episodes") + 2
    ]
    assert ["--smoke-max-steps", "7"] == execution_args[
        execution_args.index("--smoke-max-steps") : execution_args.index("--smoke-max-steps") + 2
    ]


def test_cross_device_common_resolves_openvla_stage_selection():
    common_path = REPO_ROOT / "scripts" / "bash_scripts" / "cross_device_launcher_common.sh"
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"""
            set -euo pipefail
            source {common_path}
            unset OPENVLA_STAGES
            OPENVLA_SMOKE=0
            cross_device_openvla_resolve_stages
            printf 'default:%s\\n' "${{OPENVLA_SELECTED_STAGES[*]}}"
            OPENVLA_SMOKE=1
            cross_device_openvla_resolve_stages
            printf 'smoke:%s\\n' "${{OPENVLA_SELECTED_STAGES[*]}}"
            OPENVLA_STAGES=measure,realtime,sim-same
            cross_device_openvla_resolve_stages
            printf 'explicit:%s\\n' "${{OPENVLA_SELECTED_STAGES[*]}}"
            cross_device_openvla_stage_selected doctor && printf 'doctor=yes\\n' || printf 'doctor=no\\n'
            cross_device_openvla_stage_selected measure && printf 'measure=yes\\n' || printf 'measure=no\\n'
            cross_device_openvla_stage_selected sim-same && printf 'sim_same=yes\\n' || printf 'sim_same=no\\n'
            """,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "default:doctor measure realtime sim-same sim-cross" in result.stdout
    assert "smoke:doctor measure realtime" in result.stdout
    assert "explicit:measure realtime sim-same" in result.stdout
    assert "doctor=no" in result.stdout
    assert "measure=yes" in result.stdout
    assert "sim_same=yes" in result.stdout


def test_cross_device_common_keeps_continue_on_error_out_of_doctor_args():
    common_path = REPO_ROOT / "scripts" / "bash_scripts" / "cross_device_launcher_common.sh"
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"""
            set -euo pipefail
            source {common_path}
            CONTINUE_ON_ERROR=1
            DRY_RUN=1
            ONLY_POLICY=demon-attack-openvla
            ONLY_ENV=demon_attack
            MAX_RUNS=1
            cross_device_optional_args
            for arg in "${{OPTIONAL_ARGS[@]}}"; do
                printf 'doctor:%s\\n' "$arg"
            done
            for arg in "${{EXECUTION_OPTIONAL_ARGS[@]}}"; do
                printf 'execution:%s\\n' "$arg"
            done
            """,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    doctor_args = [line.removeprefix("doctor:") for line in result.stdout.splitlines() if line.startswith("doctor:")]
    execution_args = [
        line.removeprefix("execution:")
        for line in result.stdout.splitlines()
        if line.startswith("execution:")
    ]
    assert "--continue-on-error" not in doctor_args
    assert "--continue-on-error" in execution_args
    assert "--dry-run" in doctor_args
    assert "--dry-run" in execution_args
    assert ["--only-policy", "demon-attack-openvla"] == doctor_args[
        doctor_args.index("--only-policy") : doctor_args.index("--only-policy") + 2
    ]
    assert ["--only-env", "demon_attack"] == doctor_args[
        doctor_args.index("--only-env") : doctor_args.index("--only-env") + 2
    ]
    assert ["--max-runs", "1"] == doctor_args[
        doctor_args.index("--max-runs") : doctor_args.index("--max-runs") + 2
    ]


def test_cross_device_common_records_real_stage_exit_codes_and_continues():
    common_path = REPO_ROOT / "scripts" / "bash_scripts" / "cross_device_launcher_common.sh"
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"""
            set -u
            source {common_path}
            CONTINUE_ON_ERROR=1
            cross_device_reset_failures
            cross_device_run_protocol_stage measure bash -c 'exit 17'
            printf 'after_measure\\n'
            cross_device_run_protocol_stage realtime bash -c 'exit 23'
            printf 'after_realtime\\n'
            finish_rc=0
            cross_device_finish_protocol_runs || finish_rc=$?
            printf 'finish_rc=%s\\n' "$finish_rc"
            """,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "after_measure" in combined
    assert "after_realtime" in combined
    assert "stage=measure exit_code=17" in combined
    assert "stage=realtime exit_code=23" in combined
    assert "finish_rc=1" in combined


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
    latency_bench_yaml = """
experiment:
  name: demon_attack_sf_same_latency_l8_seed0
backend:
  train_dir: results/checkpoints
""".strip()
    (config_dir / "latency_bench_eval.yaml").write_text(latency_bench_yaml + "\n", encoding="utf-8")

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
        latency_bench_yaml = f"""
experiment:
  name: demon_attack_sf_same_latency_l{latency}_seed0
backend:
  train_dir: {train_dir}
logging:
  output_dir: {tmp_path}/outputs/latency_bench/demon_attack/train_latency_{latency}/seed_0
""".strip()
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
    assert f"--output-dir {tmp_path}/outputs/latency_bench/latest/demon_attack/train_latency_8/seed_0" in calls
    assert f"--output-dir {tmp_path}/outputs/latency_bench/best/demon_attack/train_latency_8/seed_0" in calls
    assert "latency_8/seed_0/latency_bench_eval.yaml" in calls
    assert "latency_9/seed_0/latency_bench_eval.yaml" in calls
    assert "latency_bench eval command failed" in result.stderr


def test_eval_bash_script_reports_missing_best_checkpoint_but_runs_latest(tmp_path):
    config_root = tmp_path / "demon_attack"
    train_dir = tmp_path / "checkpoints"
    config_dir = config_root / "latency_8" / "seed_0"
    config_dir.mkdir(parents=True)
    latency_bench_yaml = f"""
experiment:
  name: demon_attack_sf_same_latency_l8_seed0
backend:
  train_dir: {train_dir}
logging:
  output_dir: {tmp_path}/outputs/latency_bench/demon_attack/train_latency_8/seed_0
""".strip()
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
