from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.rl_games.scripts import launch_train


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMAND_PATH = REPO_ROOT / "commands" / "openvla" / "train_gymnasium_openvla.sh"


def _gymnasium_task_contract() -> dict:
    return {
        "task_name": "mountain_car",
        "env_id": "MountainCar-v0",
        "make_kwargs": {"render_mode": "rgb_array"},
        "registration_imports": [],
        "action_labels": ["push_left", "coast", "push_right"],
        "action_values": [0, 1, 2],
        "noop_action_id": 1,
        "base_prompt": "Drive the car up the right hill.",
        "env_fps": 30,
        "obs_fps": 10,
        "frame_stack": 1,
    }


def _compose_gymnasium_openvla(tmp_path: Path, extra_overrides=()):
    dataset_root = tmp_path / "datasets"
    mixture_path = dataset_root / "_generated_mixtures" / "mountain_car_fixed_l3.json"
    task_contract = _gymnasium_task_contract()
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="gymnasium",
        init="scratch",
        mode="single",
        overrides=[
            "rl_games.gymnasium.task_contract="
            + launch_train._hydra_value(task_contract),
            f"paths.dataset_local_dir={dataset_root}",
            "dataset.single_converted_name=mountain_car_fixed_l3",
            "datasets.vla_data.data_mix=mountain_car_fixed_l3",
            f"datasets.vla_data.custom_mixtures_path={mixture_path}",
            "framework.action_model.action_env_dim=3",
            "rl_games.env_eval.enabled=false",
            "rl_games.env_eval.mid_train.enabled=false",
            "rl_games.env_eval.post_train.enabled=false",
            *extra_overrides,
        ],
    )
    return cfg, dataset_root, mixture_path, task_contract


def test_generic_gymnasium_openvla_sft_composes_without_old_task_identity(tmp_path: Path) -> None:
    cfg, dataset_root, mixture_path, task_contract = _compose_gymnasium_openvla(tmp_path)

    assert cfg.model == "openvla"
    assert cfg.framework.name == "QwenOFT"
    assert cfg.env == "gymnasium"
    assert cfg.rl_games.task == "gymnasium"
    assert cfg.rl_games.gymnasium.task_name == "mountain_car"
    assert OmegaConf.to_container(
        cfg.rl_games.gymnasium.task_contract,
        resolve=True,
    ) == task_contract
    assert cfg.mode == "single"
    assert cfg.init == "scratch"
    assert cfg.rl_games.initialization_mode == "scratch"
    assert cfg.rl_games.action_carrier == "native"
    assert cfg.dataset.converted_name == "mountain_car_fixed_l3"
    assert cfg.dataset.single_converted_name == "mountain_car_fixed_l3"
    assert cfg.dataset.mixed_converted_name == "mountain_car_fixed_l3"
    assert cfg.datasets.vla_data.data_root_dir == str(dataset_root)
    assert cfg.datasets.vla_data.data_mix == "mountain_car_fixed_l3"
    assert cfg.datasets.vla_data.custom_mixtures_path == str(mixture_path)
    assert cfg.datasets.vla_data.eval_data_mix == "mountain_car_fixed_l3__val"
    assert OmegaConf.to_container(
        cfg.datasets.vla_data.gymnasium_task_contract,
        resolve=True,
    ) == task_contract
    assert cfg.datasets.vla_data.active_action_dim == 3
    assert cfg.framework.action_model.action_dim == 3
    assert cfg.framework.action_model.action_env_dim == 3


def test_generic_gymnasium_openvla_sft_is_basic_single_image_and_has_no_env_eval(tmp_path: Path) -> None:
    cfg, _, _, _ = _compose_gymnasium_openvla(tmp_path)

    assert cfg.datasets.vla_data.include_state is False
    assert cfg.datasets.vla_data.num_obs_frames == 1
    assert cfg.datasets.vla_data.image_mode == "single"
    assert OmegaConf.select(cfg, "datasets.vla_data.image_sequence_length") is None
    assert cfg.framework.kv_memory.enabled is False
    assert cfg.framework.action_model.action_horizon == 1
    assert cfg.framework.action_model.future_action_window_size == 0
    assert cfg.framework.action_model.past_action_window_size == 0
    assert cfg.rl_games.env_eval.enabled is False
    assert cfg.rl_games.env_eval.mid_train.enabled is False
    assert cfg.rl_games.env_eval.post_train.enabled is False


def test_generic_gymnasium_openvla_can_opt_into_continuous_state_projection(tmp_path: Path) -> None:
    cfg, _, _, _ = _compose_gymnasium_openvla(
        tmp_path,
        ["framework.action_model.state_encoding=continuous_projector"],
    )

    assert cfg.framework.action_model.state_encoding == "continuous_projector"


def test_launcher_forwards_generic_gymnasium_openvla_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "test")
    cfg, dataset_root, mixture_path, task_contract = _compose_gymnasium_openvla(tmp_path)
    setup = {
        "dataset_local_dir": str(dataset_root),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    command = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "model=openvla" in command
    assert "env=gymnasium" in command
    assert "init=scratch" in command
    assert "mode=single" in command
    assert "++rl_games.task=gymnasium" in command
    assert "++rl_games.gymnasium.task_name=mountain_car" in command
    assert "++rl_games.gymnasium.task_contract.env_id=MountainCar-v0" in command
    assert "++dataset.single_converted_name=mountain_car_fixed_l3" in command
    assert "++dataset.converted_name=mountain_car_fixed_l3" in command
    assert "++datasets.vla_data.data_mix=mountain_car_fixed_l3" in command
    assert "++datasets.vla_data.eval_data_mix=mountain_car_fixed_l3__val" in command
    assert f"++datasets.vla_data.custom_mixtures_path={mixture_path}" in command
    assert any(
        item.startswith("++datasets.vla_data.gymnasium_task_contract=")
        for item in command
    )
    assert "++datasets.vla_data.active_action_dim=3" in command
    assert "++framework.action_model.action_dim=3" in command
    assert "++framework.action_model.action_env_dim=3" in command
    assert "++datasets.vla_data.num_obs_frames=1" in command
    assert "++datasets.vla_data.image_mode=single" in command
    assert "++framework.kv_memory.enabled=false" in command
    assert "++rl_games.env_eval.enabled=false" in command
    assert "++rl_games.env_eval.mid_train.enabled=false" in command
    assert "++rl_games.env_eval.post_train.enabled=false" in command

    recomposed = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="gymnasium",
        init="scratch",
        mode="single",
        overrides=[item for item in command if item.startswith("++")],
    )

    assert recomposed.rl_games.task == "gymnasium"
    assert recomposed.rl_games.gymnasium.task_name == "mountain_car"
    assert OmegaConf.to_container(
        recomposed.rl_games.gymnasium.task_contract,
        resolve=True,
    ) == task_contract
    assert recomposed.datasets.vla_data.data_root_dir == str(dataset_root)
    assert recomposed.datasets.vla_data.data_mix == "mountain_car_fixed_l3"
    assert recomposed.datasets.vla_data.eval_data_mix == "mountain_car_fixed_l3__val"
    assert recomposed.datasets.vla_data.custom_mixtures_path == str(mixture_path)
    assert OmegaConf.to_container(
        recomposed.datasets.vla_data.gymnasium_task_contract,
        resolve=True,
    ) == task_contract
    assert recomposed.datasets.vla_data.active_action_dim == 3
    assert recomposed.framework.action_model.action_dim == 3
    assert recomposed.framework.action_model.action_env_dim == 3
    assert recomposed.rl_games.env_eval.enabled is False


@pytest.mark.parametrize(
    "model,action_dim,state_dim,uses_state",
    [("openvla", 3, 1, False), ("pi05", 6, 1, False),
     ("pi05", 5, 4, True), ("pi05", 3, 11, True),
     ("pi05", 6, 17, True), ("pi05", 6, 17, True),
     ("pi05", 8, 27, True), ("pi05", 17, 376, True)],
    ids=["openvla", "air_raid", "inverted_pendulum", "hopper", "half_cheetah",
         "walker2d", "ant", "humanoid"],
)
def test_generic_gymnasium_command_reads_handoff_contract_from_manifest(
    tmp_path: Path, model: str, action_dim: int, state_dim: int, uses_state: bool,
) -> None:
    command_path = REPO_ROOT / "commands" / model / f"train_gymnasium_{model}.sh"
    subprocess.run(["bash", "-n", str(command_path)], check=True, cwd=REPO_ROOT)

    dataset_root = tmp_path / "datasets"
    data_mix = "mountain_car_fixed_l3"
    dataset_path = dataset_root / data_mix
    dataset_path.mkdir(parents=True)
    task_contract = _gymnasium_task_contract()
    (dataset_path / "manifest.json").write_text(
        json.dumps(
                {
                    "gymnasium_task": task_contract,
                    "uses_state": uses_state,
                    "state_dim": state_dim,
                    "active_action_dim": action_dim,
                    "action_dim": action_dim,
                    "action_carrier": "native",
                }
        ),
        encoding="utf-8",
    )
    mixture_path = dataset_root / "_generated_mixtures" / f"{data_mix}.json"
    capture_path = tmp_path / "launch_args.bin"
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    python_shim = shim_dir / "python"
    python_shim.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        f'  exec "{sys.executable}" "$@"\n'
        "fi\n"
        "printf '%s\\0' \"$@\" > \"${CAPTURE_PATH}\"\n",
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)

    subprocess.run(
        [
            "bash",
            str(command_path),
            str(dataset_root),
            data_mix,
            str(mixture_path),
            "trainer.max_train_steps=17",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    launched_args = [
        value.decode()
        for value in capture_path.read_bytes().split(b"\0")
        if value
    ]

    assert (
        "rl_games.gymnasium.task_contract="
        + launch_train._hydra_value(task_contract)
    ) in launched_args
    assert f"framework.action_model.action_env_dim={action_dim}" in launched_args
    assert f"framework.action_model.action_dim={action_dim}" in launched_args
    assert "rl_games.action_carrier=native" in launched_args
    assert launched_args[launched_args.index("--init") + 1] == "scratch"
    assert f"paths.dataset_local_dir={dataset_root}" in launched_args
    assert f"dataset.single_converted_name={data_mix}" in launched_args
    assert f"datasets.vla_data.custom_mixtures_path={mixture_path}" in launched_args
    assert "trainer.max_train_steps=17" in launched_args

    cfg = launch_train.compose_training_config(
        config_name="train", model=model, env="gymnasium", init="scratch", mode="single",
        overrides=launched_args[9:],
    )
    assert cfg.framework.action_model.action_dim == action_dim
    assert cfg.framework.action_model.action_env_dim == action_dim
    assert cfg.datasets.vla_data.include_state == uses_state
    assert cfg.framework.action_model.state_dim == (state_dim if uses_state or model == "openvla" else 0)
    assert cfg.framework.action_model.action_horizon == 1
    assert cfg.framework.action_model.future_action_window_size == 0
    assert cfg.framework.action_model.past_action_window_size == 0
    assert cfg.checkpoint.load == "none"
    assert cfg.trainer.is_resume is False
    assert cfg.trainer.pretrained_checkpoint is None
    assert cfg.trainer.reload_modules is None
    assert cfg.initialization.checkpoint_local_dir is None
    assert cfg.initialization.checkpoint_hf_repo_id is None
    assert cfg.initialization.checkpoint_filename is None
    if model == "pi05":
        assert cfg.base_model.repo_id == "Qwen/Qwen3-VL-4B-Instruct"
        assert cfg.framework.name == "QwenPI_v3"
        assert cfg.framework.action_model.state_encoding == "continuous_projector"
        assert cfg.framework.action_model.num_inference_timesteps == 4


def test_generic_gymnasium_openvla_command_accepts_continuous_bridge_contract(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    data_mix = "walker2d_continuous"
    dataset_path = dataset_root / data_mix
    dataset_path.mkdir(parents=True)
    task_contract = {
        "task_name": "walker2d_continuous_torque",
        "env_id": "LatencyBench/Walker2dContinuous-v0",
        "make_kwargs": {"render_mode": "rgb_array"},
        "registration_imports": ["latency_bench.envs.gymnasium_walker2d"],
        "action_space": {
            "type": "box",
            "labels": [f"torque_{index}" for index in range(6)],
            "low": [-1.0] * 6,
            "high": [1.0] * 6,
            "dtype": "float32",
            "openvla_carrier_dim": 7,
            "openvla_padding": [0.0],
        },
        "noop_action": [0.0] * 6,
        "base_prompt": "Walk forward.",
        "env_fps": 125,
        "obs_fps": 125,
        "frame_stack": 1,
    }
    (dataset_path / "manifest.json").write_text(
        json.dumps(
            {
                "gymnasium_task": task_contract,
                    "uses_state": False,
                    "state_dim": 1,
                "active_action_dim": 6,
                "action_dim": 7,
                "action_carrier": "bridge",
            }
        ),
        encoding="utf-8",
    )
    mixture_path = dataset_root / "_generated_mixtures" / f"{data_mix}.json"
    capture_path = tmp_path / "launch_args.bin"
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    python_shim = shim_dir / "python"
    python_shim.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        f'  exec "{sys.executable}" "$@"\n'
        "fi\n"
        "printf '%s\\0' \"$@\" > \"${CAPTURE_PATH}\"\n",
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)

    subprocess.run(
        ["bash", str(COMMAND_PATH), str(dataset_root), data_mix, str(mixture_path)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    launched_args = [
        value.decode()
        for value in capture_path.read_bytes().split(b"\0")
        if value
    ]

    assert (
        "rl_games.gymnasium.task_contract="
        + launch_train._hydra_value(task_contract)
    ) in launched_args
    assert launched_args[launched_args.index("--init") + 1] == "bridge"
    assert "rl_games.action_carrier=bridge" in launched_args
    assert "framework.action_model.action_dim=7" in launched_args
    assert "framework.action_model.action_env_dim=6" in launched_args
    assert "datasets.vla_data.include_state=false" in launched_args

@pytest.mark.parametrize("carrier,action_dim", [("bridge", 7), ("native", 7)])
def test_pi05_command_rejects_non_native_dataset(tmp_path: Path, carrier: str, action_dim: int) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({
        "action_carrier": carrier, "action_dim": action_dim, "active_action_dim": 6,
    }))
    shim = tmp_path / "python"
    shim.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "commands/pi05/train_gymnasium_pi05.sh"),
         str(tmp_path), "dataset", "unused.json"],
        cwd=REPO_ROOT, env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Pi05 requires a native dataset" in result.stderr
