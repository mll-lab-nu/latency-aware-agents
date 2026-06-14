from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_upload_cross_device_auto_detects_rtx_pro_6000_blackwell(tmp_path):
    script_path = REPO_ROOT / "scripts/bash_scripts/upload_cross_device_to_hf.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    out = tmp_path / "out"
    (out / "manifests").mkdir(parents=True)
    (out / "profiles").mkdir()
    (out / "runs").mkdir()
    (out / "manifests" / "runs.jsonl").write_text("{}\n", encoding="utf-8")
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition\\n'
""",
    )
    _write_executable(
        fake_bin / "hf",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "REPO_DIR": str(tmp_path),
            "OUT": str(out),
            "HF_REPO": "latency-sensitive-bench/profiles",
            "DRY_RUN": "1",
            "RUN_TAG": "test-run",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Device: rtxpro6000blackwell" in result.stdout
    assert "Remote path: raw/rtxpro6000blackwell/combined/test-run/cross_device" in result.stdout


def _write_fake_conda_installation(prefix: Path, conda_bin: Path, calls_path: Path) -> None:
    conda_bin.parent.mkdir(parents=True, exist_ok=True)
    (prefix / "etc" / "profile.d").mkdir(parents=True, exist_ok=True)
    _write_executable(
        conda_bin,
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "info" && "${{2:-}}" == "--base" ]]; then
  printf '%s\\n' "{prefix}"
  exit 0
fi
if [[ "${{1:-}}" == "tos" ]]; then
  printf "conda: error: argument COMMAND: invalid choice: 'tos'\\n" >&2
  exit 2
fi
printf 'conda %s\\n' "$*" >> "{calls_path}"
exit 0
""",
    )
    (prefix / "etc" / "profile.d" / "conda.sh").write_text(
        f"""conda() {{
  if [[ "${{1:-}}" == "activate" ]]; then
    printf 'conda activate %s\\n' "${{2:-}}" >> "{calls_path}"
    return 0
  fi
  if [[ "${{1:-}}" == "tos" ]]; then
    printf "conda: error: argument COMMAND: invalid choice: 'tos'\\n" >&2
    return 2
  fi
  "{conda_bin}" "$@"
}}
""",
        encoding="utf-8",
    )


def _prepare_install_sh_env(
    tmp_path: Path,
    *,
    compute_caps: str,
    conda_mode: str = "path",
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "install_calls.log"
    home_dir = tmp_path
    fake_conda_prefix = home_dir / "miniconda3"

    if conda_mode == "path":
        fake_conda_bin = fake_bin / "conda"
        _write_fake_conda_installation(fake_conda_prefix, fake_conda_bin, calls_path)
    elif conda_mode == "exe":
        fake_conda_bin = fake_conda_prefix / "bin" / "conda"
        _write_fake_conda_installation(fake_conda_prefix, fake_conda_bin, calls_path)
    elif conda_mode == "missing":
        fake_conda_bin = fake_conda_prefix / "bin" / "conda"
    else:
        raise ValueError(conda_mode)

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
        fake_bin / "make",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "-C envpool bazel-build" ]]; then
  printf "error: command 'swig' failed: No such file or directory\\n" >&2
  exit 1
fi
if [[ "$*" == "-C envpool bazel-release" && "${CGO_ENABLED:-}" != "0" ]]; then
  printf 'cgo: C compiler "x86_64-conda-linux-gnu-cc" not found: exec: "x86_64-conda-linux-gnu-cc": executable file not found in $PATH\\n' >&2
  exit 1
fi
if [[ "$*" == "-C envpool bazel-release" && "${USE_BAZEL_VERSION:-}" != "6.5.0" ]]; then
  printf "ERROR: /workspace/latency-sensitive-bench/envpool/BUILD:8:1: name 'py_binary' is not defined\\n" >&2
  exit 1
fi
printf 'make %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
exit 0
""",
    )
    _write_executable(
        fake_bin / "rm",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'rm %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
exit 0
""",
    )
    if conda_mode == "missing":
        downloader_script = f"""#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -O|-o)
      output="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
cat > "$output" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
prefix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b)
      shift
      ;;
    -p)
      prefix="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$prefix/bin" "$prefix/etc/profile.d"
cat > "$prefix/bin/conda" <<'EOC'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "info" && "${{2:-}}" == "--base" ]]; then
  printf '%s\\n' "__PREFIX__"
  exit 0
fi
if [[ "${{1:-}}" == "tos" ]]; then
  printf "conda: error: argument COMMAND: invalid choice: 'tos'\\n" >&2
  exit 2
fi
printf 'conda %s\\n' "$*" >> "__CALLS__"
exit 0
EOC
sed -i "s#__PREFIX__#$prefix#g; s#__CALLS__#{calls_path}#g" "$prefix/bin/conda"
chmod +x "$prefix/bin/conda"
cat > "$prefix/etc/profile.d/conda.sh" <<'EOC'
conda() {{
  if [[ "${{1:-}}" == "activate" ]]; then
    printf 'conda activate %s\\n' "${{2:-}}" >> "__CALLS__"
    return 0
  fi
  if [[ "${{1:-}}" == "tos" ]]; then
    printf "conda: error: argument COMMAND: invalid choice: 'tos'\\n" >&2
    return 2
  fi
  "__PREFIX__/bin/conda" "$@"
}}
EOC
sed -i "s#__PREFIX__#$prefix#g; s#__CALLS__#{calls_path}#g" "$prefix/etc/profile.d/conda.sh"
printf 'wget %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
EOF
chmod +x "$output"
"""
        _write_executable(fake_bin / "curl", downloader_script.replace("wget", "curl"))
        _write_executable(fake_bin / "wget", downloader_script)
    _write_executable(
        fake_bin / "bash",
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "scripts/bash_scripts/build_cuda_extensions.sh" ]]; then
  printf 'bash %s\\n' "$*" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  exit 0
fi
if [[ "${1:-}" == "starVLA/examples/rl_games/install/bootstrap.sh" ]]; then
  printf 'bash %s\\n' "$*" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  exit 0
fi
/bin/bash "$@"
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
        "PATH": f"{fake_bin}{os.pathsep}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(home_dir),
        "LATENCY_BENCH_FAKE_INSTALL_CALLS": str(calls_path),
    }
    for key in ("CONDA_EXE", "CONDA_PREFIX", "CONDA_SHLVL", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    if conda_mode == "path":
        env["FAKE_CONDA_BIN"] = str(fake_bin / "conda")
    elif conda_mode == "exe":
        env["CONDA_EXE"] = str(fake_conda_bin)
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


def test_cuda_extensions_build_script_is_executable_bash():
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "build_cuda_extensions.sh"

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
    assert "load_cule_atari_extension" in text
    assert "torch.cuda.synchronize" in text
    assert "cuda-toolkit[nvcc,cccl]==13.0.*" in text
    assert "--list-gpu-code" in text
    assert 'if [[ -n "${CONDA_PREFIX:-}" ]]; then' in text
    assert 'elif [[ -n "${VIRTUAL_ENV:-}" ]]; then' in text
    assert 'for env_lib_dir in "${env_prefix}/lib64" "${env_prefix}/lib"; do' in text
    assert "LIBRARY_PATH" in text


def test_envpool_fork_top_level_package_builds_vizdoom_only():
    build_text = (REPO_ROOT / "envpool" / "envpool" / "BUILD").read_text(encoding="utf-8")
    entry_text = (REPO_ROOT / "envpool" / "envpool" / "entry.py").read_text(encoding="utf-8")

    assert "//envpool/vizdoom" in build_text
    assert "envpool.vizdoom.registration" in entry_text
    for unsupported_env in ["atari", "box2d", "classic_control", "mujoco", "procgen", "toy_text"]:
        assert f"//envpool/{unsupported_env}" not in build_text
        assert f"envpool.{unsupported_env}" not in entry_text


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
    assert "conda create -n latency -c conda-forge --override-channels python=3.10 -y" in calls
    assert "python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130" in calls
    assert "python -m pip install -r requirements.txt" in calls
    assert "bash scripts/bash_scripts/build_cuda_extensions.sh" in calls
    assert "conda install -y -c conda-forge --override-channels go" not in calls
    assert "rm -f envpool/dist/envpool-" not in calls
    assert "make -C envpool bazel-release" not in calls


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


def test_install_script_uses_conda_exe_when_path_is_missing(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n", conda_mode="exe")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "conda tos" not in calls
    assert "conda create -n latency -c conda-forge --override-channels python=3.10 -y" in calls
    assert "conda activate latency" in calls
    assert "conda install -y -c conda-forge --override-channels go" not in calls


def test_install_script_bootstraps_miniconda_when_conda_is_missing(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n", conda_mode="missing")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "curl " in calls
    assert "conda tos" not in calls
    assert "conda create -n latency -c conda-forge --override-channels python=3.10 -y" in calls
    assert "conda activate latency" in calls
    assert "conda install -y -c conda-forge --override-channels go" not in calls


def test_install_script_persists_conda_shell_init(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, _ = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n", conda_mode="exe")
    bashrc_path = tmp_path / ".bashrc"
    bashrc_path.write_text("# existing bashrc\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    bashrc = bashrc_path.read_text(encoding="utf-8")
    assert f"source {tmp_path / 'miniconda3' / 'etc' / 'profile.d' / 'conda.sh'}" in bashrc


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


def test_cross_device_main_script_documents_single_entrypoint_and_new_protocol_module():
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "run_cross_device.sh"

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    script = script_path.read_text(encoding="utf-8")
    assert "python -m scripts.cross_device.protocol" in script
    assert "PRESET=${1:-}" in script
    assert "profile)" in script
    assert "STAGES=${STAGES:-doctor measure}" in script
    assert "sim)" in script
    assert "STAGES=${STAGES:-doctor sim-same sim-cross}" in script
    assert "custom)" in script
    assert '--runtime-env "${RUNTIME_ENV}"' in script
    assert "measure" in script
    assert "realtime" in script
    assert "sim-same" in script
    assert "sim-cross" in script
    assert "aggregate" in script
    for env_var in (
        "ONLY_ENV",
        "ONLY_POLICY",
        "ONLY_DEVICE",
        "ONLY_EXPERIMENT",
        "MAX_RUNS",
        "N_SAMPLES",
        "CONTINUE_ON_ERROR",
    ):
        assert env_var in script


def _write_fake_cross_device_python(fake_bin: Path, calls_path: Path) -> None:
    _write_executable(
        fake_bin / "python",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{calls_path}"
exit 0
""",
    )


def test_cross_device_main_script_requires_a_preset(tmp_path):
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "run_cross_device.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "python", "#!/usr/bin/env bash\nexit 0\n")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert "profile" in result.stderr


def test_cross_device_profile_preset_expands_to_doctor_measure_and_uploads(tmp_path):
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "run_cross_device.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.log"
    out = tmp_path / "out"
    profiles_dir = out / "profiles"
    profile_path = profiles_dir / "openvla-flappy" / "flappy" / "L_A100" / "profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("{}\n", encoding="utf-8")
    run_manifest = out / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "seed_0" / "manifest.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text("{}\n", encoding="utf-8")
    _write_fake_cross_device_python(fake_bin, calls_path)

    result = subprocess.run(
        ["bash", str(script_path), "profile"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PROFILES_DIR": str(profiles_dir),
            "OUT": str(out),
            "N_SAMPLES": "3",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "-m scripts.cross_device.protocol doctor" in calls
    assert "-m scripts.cross_device.protocol measure" in calls
    assert "--n-samples 3" in calls
    assert "-m scripts.cross_device.protocol realtime" not in calls
    assert "-m scripts.cross_device.profile_cache upload" in calls
    assert "-m scripts.cross_device.profile_cache upload-run" not in calls
    assert f"--folder {out}" in calls
    assert "--hf-repo latency-sensitive-bench/profiles" in calls
    assert "stages: doctor measure" in result.stdout
    assert "hf_profile_repo: latency-sensitive-bench/profiles" in result.stdout


def test_cross_device_profile_preset_dry_run_does_not_upload(tmp_path):
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "run_cross_device.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.log"
    profiles_dir = tmp_path / "profiles"
    profile_path = profiles_dir / "openvla-flappy" / "flappy" / "L_A100" / "profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("{}\n", encoding="utf-8")
    _write_fake_cross_device_python(fake_bin, calls_path)

    result = subprocess.run(
        ["bash", str(script_path), "profile"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PROFILES_DIR": str(profiles_dir),
            "OUT": str(tmp_path / "out"),
            "DRY_RUN": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "-m scripts.cross_device.protocol doctor" in calls
    assert "-m scripts.cross_device.protocol measure" in calls
    assert "-m scripts.cross_device.profile_cache upload" not in calls
    assert "profile upload: skipped for dry run" in result.stdout


def test_cross_device_sim_preset_expands_to_doctor_and_sim_stages(tmp_path):
    script_path = REPO_ROOT / "scripts" / "bash_scripts" / "run_cross_device.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.log"
    _write_fake_cross_device_python(fake_bin, calls_path)

    result = subprocess.run(
        ["bash", str(script_path), "sim"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PROFILES_DIR": str(tmp_path / "profiles"),
            "OUT": str(tmp_path / "out"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "-m scripts.cross_device.protocol doctor" in calls
    assert "-m scripts.cross_device.protocol sim-same" in calls
    assert "-m scripts.cross_device.protocol sim-cross" in calls
    assert "-m scripts.cross_device.protocol measure" not in calls
    assert "-m scripts.cross_device.profile_cache upload" not in calls
    assert "stages: doctor sim-same sim-cross" in result.stdout


def test_cross_device_start_script_runs_flappy_measure_sim_same_and_h200_sim_cross(tmp_path):
    script_path = REPO_ROOT / "start.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.log"
    _write_executable(
        fake_bin / "conda",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'conda %s\\n' "$*" >> "{calls_path}"
exit 0
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'NVIDIA H200\\n'
""",
    )
    _write_executable(
        fake_bin / "python",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'HF_PROFILE_REPO=%s HF_PROFILE_REPO_TYPE=%s CROSS_DEVICE_RUNTIME_ENV=%s :: %s\\n' \
  "${{HF_PROFILE_REPO:-}}" "${{HF_PROFILE_REPO_TYPE:-}}" "${{CROSS_DEVICE_RUNTIME_ENV:-}}" "$*" >> "{calls_path}"
exit 0
""",
    )

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "conda activate starvla_rl_games_openvla" in calls
    assert "-m scripts.cross_device.protocol measure" in calls
    assert "-m scripts.cross_device.protocol sim-same" in calls
    assert "-m scripts.cross_device.protocol sim-cross" in calls
    assert calls.count("--only-policy openvla-flappy") == 3
    assert "demon-attack-openvla" not in calls
    assert "deadly-corridor-openvla" not in calls
    assert "--profiles-dir results/cross_device/profiles" in calls
    assert "--replay-hardware auto" in calls
    assert "HF_PROFILE_REPO=latency-sensitive-bench/profiles" in calls
    assert "HF_PROFILE_REPO_TYPE=dataset" in calls
    assert "CROSS_DEVICE_RUNTIME_ENV=openvla" in calls


def test_cross_device_aggregate_script_builds_openvla_flappy_paper_outputs(tmp_path):
    script_path = REPO_ROOT / "aggregate.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.log"
    _write_executable(
        fake_bin / "python",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'HF_PROFILE_REPO=%s HF_PROFILE_REPO_TYPE=%s :: %s\\n' \
  "${{HF_PROFILE_REPO:-}}" "${{HF_PROFILE_REPO_TYPE:-}}" "$*" >> "{calls_path}"
exit 0
""",
    )

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "-m scripts.cross_device.protocol aggregate" in calls
    assert "--only-policy openvla-flappy" in calls
    assert "--root results/cross_device" in calls
    assert "--out results/cross_device/tables" in calls
    assert "--overwrite" in calls
    assert "-m scripts.cross_device.build_figure_data" in calls
    assert "--profiles-root results/cross_device" in calls
    assert "paper/latency-sensitive-vla/figures/4_1_cross_device_sim/fig6_cross_device/data.csv" in calls
    assert "paper/latency-sensitive-vla/figures/4_1_cross_device_sim/table3_latency_profiles/data.csv" in calls
    assert "paper/latency-sensitive-vla/figures/4_1_cross_device_sim/fig6_cross_device/run.py" in calls
    assert "paper/latency-sensitive-vla/figures/4_1_cross_device_sim/table3_latency_profiles/run.py" in calls
    assert "HF_PROFILE_REPO=latency-sensitive-bench/profiles" in calls
    assert "HF_PROFILE_REPO_TYPE=dataset" in calls


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
