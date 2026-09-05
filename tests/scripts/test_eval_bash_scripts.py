from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from helpers import REPO_ROOT




def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


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
    printf '%s' "$NVCC_PREPEND_FLAGS" >/dev/null
    export CONDA_PREFIX="{prefix}/envs/${{2:-}}"
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
    driver_version: str = "580.65.06",
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
        fake_bin / "uv",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'uv %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
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
    printf '%s' "$NVCC_PREPEND_FLAGS" >/dev/null
    export CONDA_PREFIX="__PREFIX__/envs/${{2:-}}"
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
if [[ "${1:-}" == "third_party/starVLA/examples/rl_games/install/torch.sh" ]]; then
  printf 'bash %s\\n' "$*" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  exit 0
fi
if [[ "${1:-}" == third_party/starVLA/examples/rl_games/install/env/*.sh ]]; then
  printf 'bash %s\\n' "$*" >> "${LATENCY_BENCH_FAKE_INSTALL_CALLS}"
  exit 0
fi
if [[ "${1:-}" == "third_party/starVLA/examples/rl_games/install/bootstrap.sh" ]]; then
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
if [[ "${{1:-}}" == "--query-gpu=driver_version" && "${{2:-}}" == "--format=csv,noheader" ]]; then
  printf '%s\n' "{driver_version}"
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
    assert 'export CUDA_HOME="${CONDA_PREFIX}"' in text
    assert 'export TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0"' in text
    assert "--list-gpu-code" not in text
    assert "detect_gpu_compute_cap" not in text


def test_install_script_forwards_fixed_stack_bootstrap(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n")

    result = subprocess.run(
        ["bash", str(script_path), "--model", "openvla"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed stack: python=3.10 cuda=12.8.1 torch=2.11.0+cu128 model=openvla" in result.stdout
    calls = calls_path.read_text(encoding="utf-8")
    assert (
        "conda create -n latency --override-channels "
        "-c nvidia/label/cuda-12.8.1 -c conda-forge "
        "python=3.10 cuda-toolkit=12.8.1 pip setuptools wheel packaging zlib=1.3.2 -y"
    ) in calls
    assert "conda activate latency" in calls
    assert "bash third_party/starVLA/examples/rl_games/install/torch.sh" in calls
    assert "uv pip install" in calls
    assert "-e .[dev]" in calls
    assert "-e third_party/sample-factory" in calls
    for game in ("demon_attack", "deadly_corridor", "flappy"):
        assert f"bash third_party/starVLA/examples/rl_games/install/env/{game}.sh" in calls
    assert "bash scripts/bash_scripts/build_cuda_extensions.sh" in calls
    assert (
        "bash third_party/starVLA/examples/rl_games/install/bootstrap.sh "
        "--tier dev --model openvla"
    ) in calls
    assert calls.index("conda activate latency") < calls.index(
        "bash third_party/starVLA/examples/rl_games/install/bootstrap.sh"
    )
    assert "--conda-env" not in calls
    assert "torch-profile" not in calls


def test_install_script_updates_existing_latency_environment(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(
        tmp_path,
        compute_caps="12.0\n",
        latency_env_exists=True,
    )

    result = subprocess.run(
        ["bash", str(script_path), "--tier", "use", "--model", "openvla"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert (
        "conda install -n latency --override-channels "
        "-c nvidia/label/cuda-12.8.1 -c conda-forge "
        "python=3.10 cuda-toolkit=12.8.1 pip setuptools wheel packaging zlib=1.3.2 -y"
    ) in calls
    assert "conda create -n latency" not in calls
    assert "-e .[dev]" in calls
    assert "bootstrap.sh --tier use --model openvla" in calls


def test_install_script_does_not_select_stack_from_gpu_or_driver(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(
        tmp_path,
        compute_caps="12.0\n",
        driver_version="570.153.02",
    )

    result = subprocess.run(
        ["bash", str(script_path), "--model", "openvla"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed stack: python=3.10 cuda=12.8.1 torch=2.11.0+cu128 model=openvla" in result.stdout
    calls = calls_path.read_text(encoding="utf-8")
    assert "torch-profile" not in calls
    assert "cu130" not in calls


def test_install_script_help_exposes_only_fixed_stack_options(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="12.0\n")

    result = subprocess.run(
        ["bash", str(script_path), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "torch-profile" not in result.stdout
    assert "Python 3.10" in result.stdout
    assert "CUDA Toolkit 12.8.1" in result.stdout
    assert "PyTorch 2.11.0+cu128" in result.stdout
    assert "--accept-rom-license" not in result.stdout
    assert not calls_path.exists()


def test_install_script_rejects_removed_torch_profile_option(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(
        tmp_path,
        compute_caps="12.0\n",
        driver_version="570.153.02",
    )

    result = subprocess.run(
        ["bash", str(script_path), "--torch-profile", "cu128"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unknown argument: --torch-profile" in result.stderr
    assert not calls_path.exists()


def test_install_script_ignores_removed_profile_environment_variable(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="9.0\n")
    env["LATENCY_BENCH_TORCH_PROFILE"] = "cu130"

    result = subprocess.run(
        ["bash", str(script_path), "--model", "openvla"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed stack: python=3.10 cuda=12.8.1 torch=2.11.0+cu128 model=openvla" in result.stdout
    calls = calls_path.read_text(encoding="utf-8")
    assert "cu130" not in calls


def test_install_script_forwards_all_model_selection(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n")

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--tier",
            "dev",
            "--model",
            "all",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed stack: python=3.10 cuda=12.8.1 torch=2.11.0+cu128 model=all" in result.stdout
    calls = calls_path.read_text(encoding="utf-8")
    assert (
        "bash third_party/starVLA/examples/rl_games/install/bootstrap.sh "
        "--tier dev --model all"
    ) in calls
    assert "uv sync --project third_party/Isaac-GR00T --python 3.12" in calls


def test_install_script_accepts_each_supported_model(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n")

    result = subprocess.run(
        ["bash", str(script_path), "--model", "gr00t"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "bootstrap.sh --tier dev --model gr00t" in calls


def test_install_script_sets_up_official_gr00t_without_starvla_bootstrap(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n")

    result = subprocess.run(
        ["bash", str(script_path), "--model", "official_gr00t"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "git submodule update --init --recursive" in calls
    assert "uv sync --project third_party/Isaac-GR00T --python 3.12" in calls
    assert (
        "uv run --project third_party/Isaac-GR00T python -c "
        in calls
    )
    assert "bootstrap.sh" not in calls


def test_install_script_requires_model(tmp_path):
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

    assert result.returncode != 0
    assert "--model is required" in result.stderr
    assert not calls_path.exists()


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
        REPO_ROOT / "third_party" / "starVLA" / "examples" / "rl_games" / "install" / "_pip.sh",
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


def test_pip_install_retries_same_command_after_network_failure(tmp_path):
    helper_path = REPO_ROOT / "scripts" / "bash_scripts" / "_pip.sh"
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


def test_install_script_source_has_no_profile_resolver():
    source = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

    assert "_torch_profile.sh" not in source
    assert "resolve_torch_profile" not in source
    assert "nvidia-smi" not in source


def test_install_script_uses_conda_exe_when_path_is_missing(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n", conda_mode="exe")

    result = subprocess.run(
        ["bash", str(script_path), "--model", "openvla"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "conda tos" not in calls
    assert "bootstrap.sh --tier dev --model openvla" in calls


def test_install_script_bootstraps_miniconda_when_conda_is_missing(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, calls_path = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n", conda_mode="missing")

    result = subprocess.run(
        ["bash", str(script_path), "--model", "openvla"],
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
    source = script_path.read_text(encoding="utf-8")
    assert 'MINICONDA_VERSION="py310_26.1.1-1"' in source
    assert "bootstrap.sh --tier dev --model openvla" in calls


def test_install_script_persists_conda_shell_init(tmp_path):
    script_path = REPO_ROOT / "install.sh"
    env, _ = _prepare_install_sh_env(tmp_path, compute_caps="10.3\n", conda_mode="exe")
    bashrc_path = tmp_path / ".bashrc"
    bashrc_path.write_text("# existing bashrc\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path), "--model", "openvla"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    bashrc = bashrc_path.read_text(encoding="utf-8")
    assert f"source {tmp_path / 'miniconda3' / 'etc' / 'profile.d' / 'conda.sh'}" in bashrc


def test_starvla_torch_installer_uses_fixed_cu128_stack(tmp_path):
    script_path = REPO_ROOT / "third_party" / "starVLA" / "examples" / "rl_games" / "install" / "torch.sh"
    fake_python = tmp_path / "python"
    fake_log = tmp_path / "python.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '[install/torch] torch=2.11.0+cu128 cuda=12.8\\n'
  exit 0
fi
printf '%s\\n' "$*" >> "${PYTHON_FAKE_LOG}"
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    _write_executable(
        tmp_path / "uv",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${PYTHON_FAKE_LOG}"
exit 0
""",
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=REPO_ROOT / "third_party" / "starVLA",
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "PYTHON_BIN": str(fake_python),
            "PYTHON_FAKE_LOG": str(fake_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = fake_log.read_text(encoding="utf-8")
    assert (
        "torch==2.11.0+cu128 torchvision==0.26.0+cu128 "
        "torchaudio==2.11.0+cu128 triton==3.6.0"
    ) in log
    assert "--index-url https://download.pytorch.org/whl/cu128" in log
    for removed in ("cu126", "cu130", "/cpu", "2.12.1"):
        assert removed not in log


def test_starvla_flash_attn_uses_pinned_community_wheel(tmp_path):
    helper_path = (
        REPO_ROOT
        / "third_party"
        / "starVLA"
        / "examples"
        / "rl_games"
        / "install"
        / "flash_attn.sh"
    )
    fake_log = tmp_path / "python.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'CUDA_HOME=%s ARGS=%s\\n' "${CUDA_HOME}" "$*" \
  >> "${PYTHON_FAKE_LOG}"
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  printf '%s\\n' "/tmp/fake-flash-attn.so"
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_cuobjdump = tmp_path / "cuobjdump"
    fake_cuobjdump.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 'sm_80' 'sm_90' 'sm_120'\n",
        encoding="utf-8",
    )
    fake_cuobjdump.chmod(0o755)

    result = subprocess.run(
        ["bash", str(helper_path)],
        cwd=REPO_ROOT / "third_party" / "starVLA",
        env={
            **os.environ,
            "CONDA_PREFIX": str(tmp_path / "conda"),
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PYTHON_BIN": str(fake_python),
            "PYTHON_FAKE_LOG": str(fake_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = fake_log.read_text(encoding="utf-8")
    assert (
        "flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
        "#sha256=9001c730642cdc1ea44ed8130b0dc80e763519d6efc01e4de44b0700a0dfa13d "
        "--force-reinstall --no-deps"
    ) in log
    helper_source = helper_path.read_text(encoding="utf-8")
    assert "fall back to sdpa" not in helper_source
    assert "STARVLA_FLASH_ATTN_BUILD_FROM_SOURCE" not in helper_source
    assert "--print-url" not in helper_source


def test_starvla_rl_games_hfd_raw_download_convert_dry_run_single_dataset():
    script_path = (
        REPO_ROOT
        / "third_party"
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
        cwd=REPO_ROOT / "third_party" / "starVLA",
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
        cwd=REPO_ROOT / "third_party" / "starVLA",
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
        / "third_party"
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
        cwd=REPO_ROOT / "third_party" / "starVLA",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("/opt/hfd.sh latency-sensitive-bench/") == 12
    assert result.stdout.count("python examples/rl_games/data_conversion/convert_") == 12
    assert "latency-sensitive-bench/flappy_fixed_l2_fs1" in result.stdout
    assert "latency-sensitive-bench/deadly_corridor_fixed_l2_fs6" in result.stdout


@pytest.mark.parametrize(
    ("model", "expected_environment"),
    [
        ("pi05", "starvla_rl_games_pi05"),
        ("qwenoft", "starvla_rl_games_openvla"),
        ("official_gr00t", "latency_prefix"),
    ],
)
def test_profile_launcher_activates_selected_model_environment(
    tmp_path, model, expected_environment
):
    prefix = tmp_path / "conda"
    fake_bin = tmp_path / "bin"
    calls_path = tmp_path / "calls.log"
    conda_bin = fake_bin / "conda"
    _write_fake_conda_installation(
        prefix,
        conda_bin,
        calls_path,
        latency_env_exists=model == "official_gr00t",
    )
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'python %s\\n' "$*" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
printf 'start_method=%s\\n' "${LATENCY_BENCH_ENV_BACKEND_START_METHOD:-}" >> "$LATENCY_BENCH_FAKE_INSTALL_CALLS"
""",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/profile.sh"),
            "list",
            "--model",
            model,
            "--workload",
            "flappy",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LATENCY_BENCH_FAKE_INSTALL_CALLS": str(calls_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    expected_environment = (
        str(prefix / "envs" / "latency")
        if expected_environment == "latency_prefix"
        else expected_environment
    )
    assert f"conda activate {expected_environment}" in calls
    assert (
        "python -m latency_bench.cross_device.protocol "
        f"list --model {model} --workload flappy"
    ) in calls
    assert "start_method=spawn" in calls




def test_sim2real_build_figures_builds_data_before_plotting(tmp_path):
    script_path = REPO_ROOT / "scripts/experiments/sim2real/build_figures.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "calls.log"
    python_logger = f"""#!/bin/bash
set -euo pipefail
printf '%s|%s\\n' "$PWD" "$*" >> "{calls_path}"
"""
    _write_executable(fake_bin / "python", python_logger)
    _write_executable(fake_bin / "python3", python_logger)

    subprocess.run(
        ["/bin/bash", str(script_path)],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        check=True,
    )

    calls = [
        (Path(cwd) / arguments.split(maxsplit=1)[0]).resolve()
        for cwd, arguments in (
            call.split("|", maxsplit=1)
            for call in calls_path.read_text(encoding="utf-8").splitlines()
        )
    ]
    experiment_dir = REPO_ROOT / "paper/latency-sensitive-vla/exps/sim2real"
    assert calls == [
        experiment_dir / "build_data.py",
        experiment_dir / "plot.py",
    ]
