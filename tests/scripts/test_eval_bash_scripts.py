from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import REPO_ROOT




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


def _write_fake_conda_installation(
    prefix: Path,
    conda_bin: Path,
    calls_path: Path,
    *,
    latency_env_exists: bool = False,
) -> None:
    conda_bin.parent.mkdir(parents=True, exist_ok=True)
    (prefix / "etc" / "profile.d").mkdir(parents=True, exist_ok=True)
    env_list = (
        f"base                  *  {prefix}\nlatency                  {prefix / 'envs' / 'latency'}\n"
        if latency_env_exists
        else f"base                  *  {prefix}\n"
    )
    _write_executable(
        conda_bin,
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "info" && "${{2:-}}" == "--base" ]]; then
  printf '%s\\n' "{prefix}"
  exit 0
fi
if [[ "${{1:-}}" == "env" && "${{2:-}}" == "list" ]]; then
  cat <<'EOF'
{env_list}EOF
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
  if [[ "${{1:-}}" == "env" && "${{2:-}}" == "list" ]]; then
    "{conda_bin}" "$@"
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
    latency_env_exists: bool = False,
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "install_calls.log"
    home_dir = tmp_path
    fake_conda_prefix = home_dir / "miniconda3"

    if conda_mode == "path":
        fake_conda_bin = fake_bin / "conda"
        _write_fake_conda_installation(
            fake_conda_prefix,
            fake_conda_bin,
            calls_path,
            latency_env_exists=latency_env_exists,
        )
    elif conda_mode == "exe":
        fake_conda_bin = fake_conda_prefix / "bin" / "conda"
        _write_fake_conda_installation(
            fake_conda_prefix,
            fake_conda_bin,
            calls_path,
            latency_env_exists=latency_env_exists,
        )
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
if [[ "${{1:-}}" == "env" && "${{2:-}}" == "list" ]]; then
  printf 'base                  *  __PREFIX__\\n'
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
  if [[ "${{1:-}}" == "env" && "${{2:-}}" == "list" ]]; then
    "__PREFIX__/bin/conda" "$@"
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
  printf 'UV_HTTP_TIMEOUT=%s\\n' "${UV_HTTP_TIMEOUT-__unset__}" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  printf 'UV_HTTP_CONNECT_TIMEOUT=%s\\n' "${UV_HTTP_CONNECT_TIMEOUT-__unset__}" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  printf 'UV_HTTP_RETRIES=%s\\n' "${UV_HTTP_RETRIES-__unset__}" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  exit 0
fi
if [[ "${1:-}" == "starVLA/examples/rl_games/install/bootstrap.sh" ]]; then
  printf 'bash %s\\n' "$*" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  printf 'UV_HTTP_TIMEOUT=%s\\n' "${UV_HTTP_TIMEOUT-__unset__}" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  printf 'UV_HTTP_CONNECT_TIMEOUT=%s\\n' "${UV_HTTP_CONNECT_TIMEOUT-__unset__}" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  printf 'UV_HTTP_RETRIES=%s\\n' "${UV_HTTP_RETRIES-__unset__}" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
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
  printf '[install] torch=2.12.1 cuda=13.0\\n'
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
        # Pin the package mirror so install.sh's index URLs are deterministic; the
        # network speed-test (export_mirror_env) is exercised separately, not here.
        "MIRROR_PROVIDER_OVERRIDE": "official",
    }
    for key in ("CONDA_EXE", "CONDA_PREFIX", "CONDA_SHLVL", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    if conda_mode == "path":
        env["FAKE_CONDA_BIN"] = str(fake_bin / "conda")
    elif conda_mode == "exe":
        env["CONDA_EXE"] = str(fake_conda_bin)
    return env, calls_path


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
    assert (
        "python -m pip install torch==2.12.1+cu130 torchvision==0.27.1+cu130 "
        "--index-url https://download.pytorch.org/whl/cu130"
    ) in calls
    assert "python -m pip install -r requirements.txt" in calls
    assert "bash scripts/bash_scripts/build_cuda_extensions.sh" in calls
    assert "UV_HTTP_TIMEOUT=300" in calls
    assert "UV_HTTP_CONNECT_TIMEOUT=30" in calls
    assert "UV_HTTP_RETRIES=10" in calls
    assert "conda install -y -c conda-forge --override-channels go" not in calls


def test_install_script_reuses_existing_latency_env(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(
        tmp_path,
        compute_caps="10.3\n",
        latency_env_exists=True,
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "reusing existing conda env: latency" in result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "conda create -n latency" not in calls
    assert "conda activate latency" in calls


def test_install_script_selects_cu130_for_compute_cap_12_gpu(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="12.0\n")

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
    assert (
        "python -m pip install torch==2.12.1+cu130 torchvision==0.27.1+cu130 "
        "--index-url https://download.pytorch.org/whl/cu130"
    ) in calls
    assert "torchaudio" not in calls


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
    assert (
        "python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 "
        "--index-url https://download.pytorch.org/whl/cu128"
    ) in calls


def _run_mirrors_snippet(snippet: str, env_extra: dict[str, str]) -> str:
    helper = REPO_ROOT / "scripts" / "bash_scripts" / "_mirrors.sh"
    result = subprocess.run(
        ["bash", "-c", f'source "{helper}"\n{snippet}\n'],
        cwd=REPO_ROOT,
        env={**os.environ, **env_extra},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_mirrors_provider_override_routes_torch_and_pypi():
    # An explicit override skips the network probe and routes both indexes.
    out = _run_mirrors_snippet(
        'echo "$(torch_index_for cu128)|$(pypi_index_url cu128)"',
        {"MIRROR_PROVIDER_OVERRIDE": "aliyun"},
    )
    assert out == (
        "https://mirrors.aliyun.com/pytorch-wheels/cu128|https://mirrors.aliyun.com/pypi/simple"
    )


def test_mirrors_fall_back_to_official_when_no_provider_reachable(tmp_path):
    # Unknown providers resolve to no base (no network), so selection falls back.
    out = _run_mirrors_snippet(
        'echo "$(torch_index_for cu128)|$(pypi_index_url cu128)"',
        {"MIRROR_PROVIDERS": "bogus", "UV_CACHE_DIR": str(tmp_path)},
    )
    assert out == "https://download.pytorch.org/whl/cu128|https://pypi.org/simple"


@pytest.mark.parametrize(
    "helper_path",
    [
        REPO_ROOT / "scripts" / "bash_scripts" / "_pip.sh",
        REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "_pip.sh",
    ],
)
def test_pip_install_with_explicit_index_clears_uv_index_environment(tmp_path, helper_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "uv.log"
    fake_python = fake_bin / "python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )
    _write_executable(
        fake_bin / "uv",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'UV_DEFAULT_INDEX=%s\\n' "${{UV_DEFAULT_INDEX-__unset__}}" >> "{log_path}"
printf 'UV_INDEX_URL=%s\\n' "${{UV_INDEX_URL-__unset__}}" >> "{log_path}"
printf 'uv %s\\n' "$*" >> "{log_path}"
exit 0
""",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{helper_path}"\n'
            'PYTHON_BIN=python pip_install torch==2.10.0+cu128 '
            "--index-url https://download.pytorch.org/whl/cu128\n",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin:/bin",
            "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "UV_INDEX_URL": "https://pypi.org/simple",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert "UV_DEFAULT_INDEX=__unset__" in log
    assert "UV_INDEX_URL=__unset__" in log
    assert "uv pip install --python " in log
    assert "torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128" in log


def test_pip_install_without_explicit_index_preserves_uv_index_environment(tmp_path):
    helper_path = REPO_ROOT / "scripts" / "bash_scripts" / "_pip.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "uv.log"
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("numpy==1.26.4\n", encoding="utf-8")
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )
    _write_executable(
        fake_bin / "uv",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'UV_DEFAULT_INDEX=%s\\n' "${{UV_DEFAULT_INDEX-__unset__}}" >> "{log_path}"
printf 'UV_INDEX_URL=%s\\n' "${{UV_INDEX_URL-__unset__}}" >> "{log_path}"
printf 'uv %s\\n' "$*" >> "{log_path}"
exit 0
""",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{helper_path}"\nPYTHON_BIN=python pip_install -r "{requirements_path}"\n',
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin:/bin",
            "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "UV_INDEX_URL": "https://pypi.org/simple",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert "UV_DEFAULT_INDEX=https://pypi.org/simple" in log
    assert "UV_INDEX_URL=https://pypi.org/simple" in log
    assert f"uv pip install --python {fake_bin / 'python'} -r {requirements_path}" in log


@pytest.mark.parametrize(
    "helper_path",
    [
        REPO_ROOT / "scripts" / "bash_scripts" / "_pip.sh",
        REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "_pip.sh",
    ],
)
def test_pip_install_retries_same_command_after_network_failure(tmp_path, helper_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "uv.log"
    state_path = tmp_path / "uv.state"
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )
    _write_executable(
        fake_bin / "sleep",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'sleep %s\\n' "$*" >> "{log_path}"
exit 0
""",
    )
    _write_executable(
        fake_bin / "uv",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'uv %s\\n' "$*" >> "{log_path}"
if [[ ! -f "{state_path}" ]]; then
  touch "{state_path}"
  exit 28
fi
exit 0
""",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{helper_path}"\n'
            "PIP_INSTALL_RETRY_SLEEP_SECONDS=0 "
            "PYTHON_BIN=python pip_install nvidia-curand==10.4.0.35\n",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert log.count("uv pip install --python") == 2
    assert log.count("nvidia-curand==10.4.0.35") == 2
    assert "sleep 0" in log
    assert "pip install failed on attempt 1/3; retrying" in result.stderr


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
    if not helper_path.exists():
        pytest.skip("starVLA submodule pin does not include flash_attn.sh")
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
        "flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-"
        "cp310-cp310-linux_x86_64.whl"
    )

    # Model scripts install deps via _pip.sh, which prefers uv. uv cannot introspect
    # the fake python below, so exercise _pip.sh's supported plain-pip fallback by
    # dropping any uv on PATH (the fake python then logs the pip-install invocations).
    env_path = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if not (p and os.access(os.path.join(p, "uv"), os.X_OK))
    )

    for script_path in model_scripts:
        fake_log.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=REPO_ROOT / "starVLA",
            env={
                **os.environ,
                "PATH": env_path,
                "PYTHON_BIN": str(fake_python),
                "PYTHON_FAKE_LOG": str(fake_log),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert expected_wheel in fake_log.read_text(encoding="utf-8")


def test_starvla_torch_installer_cu130_uses_torchvision_without_torchaudio(tmp_path):
    script_path = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "torch.sh"
    fake_python = tmp_path / "python"
    fake_log = tmp_path / "python.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '[install/torch] torch=2.12.1+cu130 cuda=13.0\\n'
  exit 0
fi
printf '%s\\n' "$*" >> "${PYTHON_FAKE_LOG}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env_path = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if not (p and os.access(os.path.join(p, "uv"), os.X_OK))
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT / "starVLA",
        env={
            **os.environ,
            "PATH": env_path,
            "PYTHON_BIN": str(fake_python),
            "PYTHON_FAKE_LOG": str(fake_log),
            "STARVLA_TORCH_PROFILE": "cu130",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = fake_log.read_text(encoding="utf-8")
    assert "torch==2.12.1+cu130 torchvision==0.27.1+cu130" in log
    assert "--index-url https://download.pytorch.org/whl/cu130" in log
    assert "torchaudio" not in log


def test_starvla_install_stack_installs_eval_extra_requirements_after_env_deps(tmp_path):
    script_path = REPO_ROOT / "starVLA" / "examples" / "rl_games" / "install" / "install_stack.sh"
    if "eval_extra" not in script_path.read_text(encoding="utf-8"):
        pytest.skip("starVLA submodule pin predates eval_extra integration in install_stack.sh")
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
    if "eval_extra" not in script_path.read_text(encoding="utf-8"):
        pytest.skip("starVLA submodule pin predates eval_extra integration in bootstrap.sh")
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
    if not helper_path.exists():
        pytest.skip("starVLA submodule pin does not include flash_attn.sh")
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
    if not helper_path.exists():
        pytest.skip("starVLA submodule pin does not include flash_attn.sh")
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
    if not script_path.exists():
        pytest.skip("starVLA submodule pin does not include download_convert_rl_games_raw.sh")

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
    if not script_path.exists():
        pytest.skip("starVLA submodule pin does not include download_convert_rl_games_raw.sh")

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
