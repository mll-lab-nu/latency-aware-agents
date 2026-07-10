from __future__ import annotations

import importlib.util
from pathlib import Path
from helpers import REPO_ROOT


MODULE_PATH = REPO_ROOT / "scripts" / "profile_sample_factory_training.py"
SPEC = importlib.util.spec_from_file_location("profile_sample_factory_training", MODULE_PATH)


def _load_module():
    assert SPEC is not None
    module = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(module)
    return module


def _base_config(tmp_path: Path) -> dict:
    return {
        "experiment": {"name": "profile_base", "seed": 7},
        "backend": {
            "type": "sample_factory",
            "algo": "APPO",
            "device": "cpu",
            "train_dir": str(tmp_path / "train"),
            "restart_behavior": "resume",
            "run_mode": "train",
        },
        "env": {
            "name": "demon_attack",
            "env_fps": 60,
            "obs_fps": 15,
            "simulator": "cpu",
            "gpu_env_device": "auto",
        },
        "latency": {"method": "zero", "add_latency_info": False},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "training": {
            "train_for_env_steps": 25000000,
            "num_workers": 16,
            "num_envs_per_worker": 4,
            "batch_size": 1024,
            "async_rl": True,
        },
        "evaluation": {},
        "logging": {
            "output_dir": str(tmp_path / "outputs"),
            "wandb_project": "latency-sensitive-bench",
        },
    }


def test_latency_case_sets_fixed_normal_and_uniform_ms():
    profile = _load_module()

    fixed = profile._latency_config_for_case("fixed", env_fps=60.0)
    normal = profile._latency_config_for_case("normal", env_fps=60.0)
    uniform = profile._latency_config_for_case("uniform", env_fps=60.0)

    assert fixed == {"method": "fixed", "fixed_latency_ms": 4000.0 / 60.0, "add_latency_info": False}
    assert normal == {
        "method": "normal",
        "mean_latency_ms": 4000.0 / 60.0,
        "std_latency_ms": 1000.0 / 60.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 15000.0 / 60.0,
        "add_latency_info": False,
    }
    assert uniform == {
        "method": "uniform",
        "min_latency_ms": 2000.0 / 60.0,
        "max_latency_ms": 6000.0 / 60.0,
        "add_latency_info": False,
    }


def test_stage_config_sets_profile_dimensions_and_disables_wandb(tmp_path: Path):
    profile = _load_module()

    cfg = profile._stage_config(
        _base_config(tmp_path),
        env_name="demon_attack",
        simulator="gpu",
        latency_case="normal",
        async_rl=False,
        train_for_env_steps=65536,
        num_workers=1,
        sf_num_envs_per_worker=1,
        worker_num_splits=1,
        batch_size=4096,
        gpu_env_batch_size=1024,
        rollout=None,
        recurrence=None,
        num_batches_per_epoch=None,
        output_root=tmp_path / "profile",
    )

    assert cfg["experiment"]["name"] == "profile_demon_attack_gpu_g1024_normal_sync"
    assert cfg["backend"]["run_mode"] == "train"
    assert cfg["backend"]["restart_behavior"] == "overwrite"
    assert cfg["backend"]["device"] == "cuda"
    assert cfg["env"]["simulator"] == "gpu"
    assert "gpu_env_batch_size" not in cfg["env"]
    assert cfg["latency"]["method"] == "normal"
    assert cfg["training"]["async_rl"] is False
    assert cfg["training"]["train_for_env_steps"] == 65536
    assert cfg["training"]["num_workers"] == 1
    assert cfg["training"]["num_envs_per_worker"] == 1
    assert cfg["training"]["worker_num_splits"] == 1
    assert cfg["training"]["batch_size"] == 4096
    assert cfg["evaluation"]["eval_interval_steps"] is None
    assert cfg["logging"]["wandb_project"] == ""
    assert cfg["logging"]["output_dir"].endswith("demon_attack/gpu/g1024/normal/sync")


def test_stage_config_can_explicitly_override_sync_training_shape(tmp_path: Path):
    profile = _load_module()

    cfg = profile._stage_config(
        _base_config(tmp_path),
        env_name="demon_attack",
        simulator="gpu",
        latency_case="fixed",
        async_rl=False,
        train_for_env_steps=65536,
        num_workers=1,
        sf_num_envs_per_worker=1,
        worker_num_splits=1,
        batch_size=1024,
        gpu_env_batch_size=1024,
        rollout=2,
        recurrence=1,
        num_batches_per_epoch=2,
        output_root=tmp_path / "profile",
    )

    assert cfg["training"]["rollout"] == 2
    assert cfg["training"]["recurrence"] == 1
    assert cfg["training"]["num_batches_per_epoch"] == 2


def test_profile_dimensions_keep_cpu_original_and_gpu_collapsed_workers():
    profile = _load_module()
    args = profile.argparse.Namespace(
        cpu_num_workers=16,
        cpu_num_envs_per_worker=4,
        cpu_worker_num_splits=2,
        gpu_num_workers=1,
        gpu_sf_num_envs_per_worker=1,
        gpu_worker_num_splits=1,
        gpu_env_batch_sizes="512,1024",
    )

    assert profile._run_dimensions(args, "cpu") == [
        {
            "num_workers": 16,
            "sf_num_envs_per_worker": 4,
            "worker_num_splits": 2,
            "gpu_env_batch_size": 0,
        }
    ]
    assert profile._run_dimensions(args, "gpu") == [
        {
            "num_workers": 1,
            "sf_num_envs_per_worker": 512,
            "worker_num_splits": 1,
            "gpu_env_batch_size": 512,
        },
        {
            "num_workers": 1,
            "sf_num_envs_per_worker": 1024,
            "worker_num_splits": 1,
            "gpu_env_batch_size": 1024,
        },
    ]
