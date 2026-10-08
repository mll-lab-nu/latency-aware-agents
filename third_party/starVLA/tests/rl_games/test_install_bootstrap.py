import os
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_bootstrap_does_not_install_latency_bench_eval_extra_dependencies() -> None:
    repo_root: Path = _repo_root()
    bootstrap_script = repo_root / "examples" / "rl_games" / "install" / "bootstrap.sh"
    bootstrap_source: str = bootstrap_script.read_text()

    assert "Installing latency-bench eval extra dependencies" not in bootstrap_source
    assert "eval_extra.sh" not in bootstrap_source


def _help_output(script: Path) -> str:
    return subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_bootstrap_requires_an_explicit_model_for_the_fixed_stack() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    bootstrap_help: str = _help_output(install_dir / "bootstrap.sh")

    assert "--tier <use|dev>" in bootstrap_help
    assert "Python 3.10" in bootstrap_help
    assert "CUDA Toolkit 12.8.1" in bootstrap_help
    assert "PyTorch 2.11.0+cu128" in bootstrap_help
    assert "openvla|pi0|pi05|gr00t|wan_oft|all" in bootstrap_help
    assert "--torch-profile" not in bootstrap_help
    assert "--python-version" not in bootstrap_help
    assert "--current-env" not in bootstrap_help
    assert "--accept-rom-license" not in bootstrap_help

    missing_model = subprocess.run(
        [
            "bash",
            str(install_dir / "bootstrap.sh"),
            "--skip-validate",
        ],
        capture_output=True,
        text=True,
    )
    assert missing_model.returncode != 0
    assert "--model is required" in missing_model.stderr


def test_install_stack_remains_a_training_compatibility_entrypoint(tmp_path: Path) -> None:
    repo_root: Path = _repo_root()
    source_install_stack: Path = repo_root / "examples" / "rl_games" / "install" / "install_stack.sh"
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    install_stack = install_dir / "install_stack.sh"
    shutil.copy2(source_install_stack, install_stack)
    fake_bootstrap = install_dir / "bootstrap.sh"
    fake_bootstrap.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_bootstrap.chmod(0o755)

    install_stack_help: str = _help_output(install_stack)
    result = subprocess.run(
        [
            "bash",
            str(install_stack),
            "openvla",
            "cross_task",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "fixed PyTorch 2.11/CUDA 12.8 model" in install_stack_help
    assert "openvla|pi0|pi05|gr00t|wan_oft|all" in install_stack_help
    assert "--torch-profile" not in install_stack_help
    assert result.stdout.splitlines() == [
        "[install_stack] legacy env selector 'cross_task' accepted; installing all games",
        "--tier",
        "dev",
        "--model",
        "openvla",
    ]


def test_bootstrap_rejects_unknown_model() -> None:
    repo_root: Path = _repo_root()
    bootstrap: Path = repo_root / "examples" / "rl_games" / "install" / "bootstrap.sh"
    result = subprocess.run(
        [
            "bash",
            str(bootstrap),
            "--model",
            "typo",
            "--skip-validate",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid model 'typo'" in result.stderr


def test_bootstrap_all_creates_one_fixed_environment_per_model(tmp_path: Path) -> None:
    source_install_dir = _repo_root() / "examples" / "rl_games" / "install"
    install_dir = tmp_path / "starvla" / "examples" / "rl_games" / "install"
    install_dir.mkdir(parents=True)
    for name in ("bootstrap.sh", "_host.sh"):
        shutil.copy2(source_install_dir / name, install_dir / name)
    (install_dir / "_pip.sh").write_text("#!/bin/bash\npip_install() { :; }\n")

    env_order_log = tmp_path / "env-order.log"
    for relative_path in (
        "common.sh",
        "model/openvla.sh",
        "model/pi0.sh",
        "model/pi05.sh",
        "model/gr00t.sh",
        "model/wan_oft.sh",
    ):
        script = install_dir / relative_path
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\nexit 0\n")
        script.chmod(0o755)
    for env_name in ("flappy", "demon_attack", "deadly_corridor"):
        script = install_dir / "env" / f"{env_name}.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            f'#!/bin/bash\nprintf "%s\\n" "{env_name}" >> "{env_order_log}"\n'
        )
        script.chmod(0o755)
    build_script = tmp_path / "latency-bench" / "scripts" / "bash_scripts" / "build_cuda_extensions.sh"
    build_script.parent.mkdir(parents=True)
    build_script.write_text("#!/bin/bash\nexit 0\n")
    build_script.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda_base = tmp_path / "conda"
    profile_dir = conda_base / "etc" / "profile.d"
    profile_dir.mkdir(parents=True)
    conda_log = tmp_path / "conda.log"

    conda = fake_bin / "conda"
    conda.write_text(
        f"""#!/bin/bash
if [[ "$1 $2" == "info --base" ]]; then
  printf '%s\\n' "{conda_base}"
fi
"""
    )
    conda.chmod(0o755)
    python = fake_bin / "python"
    python.write_text("#!/bin/bash\nprintf '3.10\\n'\n")
    python.chmod(0o755)
    (profile_dir / "conda.sh").write_text(
        f"""conda() {{
  printf '%s\\n' "$*" >> "{conda_log}"
  case "$1" in
    env)
      [[ "$2" == "list" ]]
      printf '# conda environments:\\n'
      if [[ -n "${{FAKE_EXISTING_ENV:-}}" ]]; then
        printf '%s\\n' "${{FAKE_EXISTING_ENV}}"
      fi
      ;;
    activate)
      : "${{NVCC_PREPEND_FLAGS}}"
      export CONDA_PREFIX="{conda_base}/envs/$2"
      ;;
    deactivate)
      unset CONDA_PREFIX
      ;;
  esac
}}
"""
    )

    result = subprocess.run(
        [
            "bash",
            str(install_dir / "bootstrap.sh"),
            "--model",
            "all",
            "--skip-validate",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "LATENCY_BENCH_ROOT": str(tmp_path / "latency-bench"),
        },
    )

    assert result.returncode == 0, result.stderr
    conda_calls = conda_log.read_text()
    expected_models = ("openvla", "pi0", "pi05", "gr00t", "wan_oft")
    for model in expected_models:
        assert f"create -n starvla_rl_games_{model}" in conda_calls
        assert f"activate starvla_rl_games_{model}" in conda_calls
    assert "-c nvidia/label/cuda-12.8.1" in conda_calls
    assert "python=3.10" in conda_calls
    assert "cuda-toolkit=12.8.1" in conda_calls
    assert "zlib=1.3.2" in conda_calls
    create_offsets = [
        conda_calls.index(f"create -n starvla_rl_games_{model}")
        for model in expected_models
    ]
    assert create_offsets == sorted(create_offsets)
    assert env_order_log.read_text().splitlines() == [
        game
        for _model in expected_models
        for game in ("demon_attack", "deadly_corridor", "flappy")
    ]

    conda_log.write_text("")
    existing_env_result = subprocess.run(
        [
            "bash",
            str(install_dir / "bootstrap.sh"),
            "--model",
            "openvla",
            "--skip-validate",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "LATENCY_BENCH_ROOT": str(tmp_path / "latency-bench"),
            "FAKE_EXISTING_ENV": "starvla_rl_games_openvla",
        },
    )

    assert existing_env_result.returncode == 0, existing_env_result.stderr
    existing_env_calls = conda_log.read_text()
    assert "install -n starvla_rl_games_openvla" in existing_env_calls
    assert "create -n starvla_rl_games_openvla" not in existing_env_calls
    assert "python=3.10" in existing_env_calls
    assert "cuda-toolkit=12.8.1" in existing_env_calls
    assert "zlib=1.3.2" in existing_env_calls


def test_training_dependencies_are_not_in_the_use_manifest() -> None:
    repo_root: Path = _repo_root()
    use_requirements = (repo_root / "requirements.txt").read_text().splitlines()
    dev_requirements = (repo_root / "requirements-dev.txt").read_text().splitlines()
    common_install = (repo_root / "examples/rl_games/install/common.sh").read_text()

    for dependency in ("datasets", "deepspeed", "hydra-core", "wandb"):
        assert not any(line.startswith(dependency) for line in use_requirements)
        assert any(line.startswith(dependency) for line in dev_requirements)

    assert "decord==0.6.0" not in dev_requirements
    assert "eva-decord==0.6.1" in dev_requirements
    assert "eva-decord==0.6.1" in common_install
    assert "pyarrow>=15.0.0" in dev_requirements


def test_flash_attention_validation_is_never_made_nonfatal() -> None:
    install_dir = _repo_root() / "examples" / "rl_games" / "install"
    install_sources = "\n".join(
        script.read_text() for script in install_dir.rglob("*.sh")
    )

    assert "flash_attn.sh\" --check || true" not in install_sources


def test_demon_attack_installer_uses_gymnasium_compatible_ale() -> None:
    repo_root: Path = _repo_root()
    installer_source = (
        repo_root / "examples" / "rl_games" / "install" / "env" / "demon_attack.sh"
    ).read_text()

    assert '"ale-py==0.8.1"' in installer_source
    assert '"gymnasium[atari]==0.29.1"' in installer_source
    assert "ale-py==0.10.2" not in installer_source
    assert "AutoROM --accept-license" in installer_source


def test_torch_installer_is_fixed_to_cu128() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    torch_source = (install_dir / "torch.sh").read_text()

    assert not (install_dir / "_torch_profile.sh").exists()
    assert "torch==2.11.0+cu128" in torch_source
    assert "torchvision==0.26.0+cu128" in torch_source
    assert "torchaudio==2.11.0+cu128" in torch_source
    assert "triton==3.6.0" in torch_source
    assert "https://download.pytorch.org/whl/cu128" in torch_source
    for removed in ("cu126", "cu130", "resolve_torch_profile", "nvidia-smi"):
        assert removed not in torch_source


def test_model_installers_do_not_select_torch_or_flash_versions() -> None:
    install_dir = _repo_root() / "examples" / "rl_games" / "install"
    model_sources = "\n".join(
        script.read_text() for script in (install_dir / "model").glob("*.sh")
    )
    bootstrap_source = (install_dir / "bootstrap.sh").read_text()

    assert "STARVLA_TORCH_PROFILE" not in model_sources
    assert "flash_attn4" not in model_sources
    assert not (install_dir / "flash_attn4.sh").exists()
    for model in ("pi0", "gr00t"):
        assert "flash_attn.sh" in (install_dir / "model" / f"{model}.sh").read_text()
    assert "openvla|pi0|pi05|gr00t)" in bootstrap_source


def test_flash_attention_wheel_is_hash_pinned_and_checks_target_architectures() -> None:
    flash_source = (
        _repo_root() / "examples" / "rl_games" / "install" / "flash_attn.sh"
    ).read_text()

    assert "--force-reinstall" in flash_source
    assert "flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" in flash_source
    assert "9001c730642cdc1ea44ed8130b0dc80e763519d6efc01e4de44b0700a0dfa13d" in flash_source
    assert 'code_objects="$(cuobjdump --list-elf "${extension_path}")"' in flash_source
    assert 'grep -q "sm_80" <<<"${code_objects}"' in flash_source
    assert 'grep -q "sm_90" <<<"${code_objects}"' in flash_source
    assert 'grep -q "sm_120" <<<"${code_objects}"' in flash_source


def test_every_model_has_a_validator_for_every_game() -> None:
    validate_dir = _repo_root() / "examples" / "rl_games" / "install" / "validate"
    validator_names = {
        "openvla": {
            "openvla_flappy.sh",
            "openvla_demon_attack.sh",
            "openvla_deadly_corridor.sh",
        },
        "pi0": {
            "pi0_flappy.sh",
            "pi0_demon.sh",
            "pi0_deadly_corridor.sh",
        },
        "pi05": {
            "pi05_flappy.sh",
            "pi05_demon_attack.sh",
            "pi05_deadly_corridor.sh",
        },
        "gr00t": {
            "gr00t_flappy.sh",
            "gr00t_demon_attack.sh",
            "gr00t_deadly_corridor.sh",
        },
        "wan_oft": {
            "wan_oft_flappy.sh",
            "wan_oft_demon_attack.sh",
            "wan_oft_deadly_corridor.sh",
        },
    }

    for names in validator_names.values():
        for name in names:
            assert (validate_dir / name).is_file()


def test_wan_oft_install_assets_are_registered() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    model_installer: Path = install_dir / "model" / "wan_oft.sh"
    model_installer_source: str = model_installer.read_text()
    wan_validators = [
        install_dir / "validate" / f"wan_oft_{game}.sh"
        for game in ("flappy", "demon_attack", "deadly_corridor")
    ]
    validator_sources = "\n".join(path.read_text() for path in wan_validators)

    assert model_installer.is_file()
    assert os.access(model_installer, os.X_OK)
    assert "pyarrow" in model_installer_source
    assert "huggingface-hub" in model_installer_source
    assert all(path.is_file() for path in wan_validators)
    assert "from diffusers import AutoencoderKLWan, WanTransformer3DModel" in validator_sources
    assert "from transformers import UMT5EncoderModel" in validator_sources
