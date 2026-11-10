"""The three dataset launchers must preserve the published H8 student recipe."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from latency_bench.utils.io import read_json, write_json_atomic
from scripts.experiments.memory_mock import train_vla
from scripts.experiments.temporal_profiles import teacher
from scripts.tasks.decision_history.train_vla_hist8 import training_config
from test_hist8_v2 import arguments
from test_temporal_teacher import snapshot


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def collections(tmp_path, snapshot):
    root = tmp_path / "collections"
    for method in ("temporal", "ar1", "arma"):
        collection = root / method / "seed_42"
        source = teacher.PROFILE_SETS["irregular_20min"]
        shutil.copytree(snapshot / source["root"] / "profiles" / method, collection / "profile")
        write_json_atomic(collection / "protocol.json", {
            "profile_set": "irregular_20min", "run_id": method + "-collection",
            "settings": {
                "method": method, "env_fps": 10, "obs_fps": 10, "seed": 42,
                "profile_seed": 42, "max_episode_raw_frames": 18000,
                "filter_preset": "none", "val_fraction": .1, "split_seed": 0,
                "deterministic": True,
            },
        })
        write_json_atomic(collection / "collection_complete.json", {
            "status": "collection_complete", "train_rows": 320000, "val_rows": 20000,
        })
        for split, frames in (("lerobot", 320000), ("lerobot__val", 20000)):
            write_json_atomic(collection / split / "manifest.json", {
                "robot_type": "rl_games_flappy", "action_dim": 2, "state_dim": 24,
                "action_history_decisions": 8, "state_source": "decision_v1",
                "obs_fps": 10, "rows_unit": "decision_step", "frames": frames,
                "state_normalization": {"min": [0.] * 24, "max": [1.] * 24},
            })
            data = collection / split / "data/chunk-000"
            data.mkdir(parents=True)
            # Only preparation is exercised: no batch reader or GPU is invoked.
            (data / "episode_000000.parquet").touch()
        write_json_atomic(collection / "_generated_mixtures/lerobot.json", {
            name: [[name, 1., "rl_games_flappy"]] for name in ("lerobot", "lerobot__val")
        })
        (collection / "DONE").touch()
    return root


def test_all_launchers_prepare_the_same_paper_recipe(collections, tmp_path):
    output = tmp_path / "students"
    normalized = []
    for method in ("temporal", "ar1", "arma"):
        result = subprocess.run(
            ["bash", f"scripts/experiments/memory_mock/train_vla_{method}.sh", "--prepare-only"],
            cwd=ROOT, env={**os.environ, "DATASET_ROOT": str(collections),
                          "VLA_OUTPUT_ROOT": str(output), "OUTPUT_DIR": str(collections),
                          "VLA_GPUS": "0",
                          "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                          "WANDB_ENTITY": "test-team", "WANDB_PROJECT": "test-project",
                          "WANDB_MODE": "online"},
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        run = output / f"flappy-qwenoft-{method}-irregular20m-hist8-s42"
        cfg = OmegaConf.load(run / "train.yaml")
        assert cfg.datasets.vla_data.data_root_dir == str(collections / method / "seed_42")
        assert cfg.datasets.vla_data.eval_data_mix == "lerobot__val"
        assert not (run / "run_metadata.json").exists()  # preparation is not training
        record = read_json(run / "training_data.json")
        assert record["collection_run_id"] == method + "-collection"
        assert record["num_processes"] == 1 and record["global_batch"] == 128
        # Compare the entire config to the actual main-paper entrypoint, not a
        # second hand-maintained implementation of its optimizer/model settings.
        reference_args = arguments(tmp_path, task="flappy")
        reference_args.dataset_dir = collections / method / "seed_42/lerobot"
        reference_args.base_model_dir = Path(cfg.framework.qwenvl.base_vlm)
        reference_args.output_root = output
        reference_args.run_id = cfg.run_id
        reference_args.wandb_entity = "test-team"
        reference_args.wandb_project = "test-project"
        reference = training_config(reference_args)
        assert reference.trainer.save_interval == 500
        assert reference.trainer.eval_interval == 500
        reference.trainer.save_interval = 5000
        reference.trainer.eval_interval = 5000
        reference.trainer.distributed_backend = "none"
        reference.trainer.gradient_accumulation_steps = 4
        reference.datasets.vla_data.per_device_batch_size = 32
        assert cfg == reference
        assert cfg.trainer.eval_interval == 5000
        assert not cfg.checkpoint.save_best_model
        assert cfg.checkpoint.save_final_model
        assert not cfg.rl_games.env_eval.mid_train.enabled
        cfg.run_id = cfg.wandb_name = "matched-run"
        cfg.datasets.vla_data.data_root_dir = "/matched/data"
        cfg.datasets.vla_data.custom_mixtures_path = "/matched/data/_generated_mixtures/lerobot.json"
        normalized.append(OmegaConf.to_container(cfg, resolve=True))
    assert normalized[0] == normalized[1] == normalized[2]
    assert not list(collections.glob("*/seed_42/train.yaml"))


@pytest.mark.parametrize("change", ("cap", "method", "filtered", "incomplete", "history"))
def test_refuse_wrong_or_partial_collection(collections, change):
    collection = collections / "temporal/seed_42"
    if change == "incomplete":
        (collection / "DONE").unlink()
    elif change == "history":
        path = collection / "lerobot/manifest.json"
        manifest = read_json(path)
        manifest["state_dim"] = 1
        write_json_atomic(path, manifest)
    else:
        path = collection / "protocol.json"
        protocol = read_json(path)
        key, value = {
            "cap": ("max_episode_raw_frames", 7200),
            "method": ("method", "arma"),
            "filtered": ("episode_return_gt", 500),
        }[change]
        protocol["settings"][key] = value
        write_json_atomic(path, protocol)
    with pytest.raises(ValueError):
        train_vla.collection_record(collection, "temporal")


def test_launch_and_resume_use_one_gpu_without_deepspeed(collections, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(train_vla.train_vla_hist8.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(train_vla.train_vla_hist8.subprocess, "check_output", lambda *a, **k: "test-source")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    argv = ["train_vla", "--method", "temporal", "--collection-dir", str(collections / "temporal/seed_42"),
            "--base-model-dir", str(tmp_path / "backbone"), "--output-root", str(tmp_path / "students")]
    monkeypatch.setattr(sys, "argv", argv)
    train_vla.main()
    command, options = calls[0]
    assert command[:2] == [sys.executable, str(train_vla.train_vla_hist8.STARVLA / "starVLA/training/train_starvla.py")]
    assert "accelerate.commands.launch" not in command
    assert options["env"]["ACCELERATE_USE_DEEPSPEED"] == "false"
    assert options["env"]["ACCELERATE_MIXED_PRECISION"] == "bf16"
    run = tmp_path / "students/flappy-qwenoft-temporal-irregular20m-hist8-s42"
    cfg = OmegaConf.load(run / "train.yaml")
    assert cfg.trainer.max_train_steps == 5000
    assert cfg.trainer.save_interval == 5000
    assert cfg.trainer.eval_interval == 5000
    assert cfg.datasets.vla_data.per_device_batch_size == 32
    assert cfg.trainer.gradient_accumulation_steps == 4
    assert cfg.trainer.distributed_backend == "none"
    assert cfg.wandb_project is None
    with pytest.raises(ValueError, match="already exists"):
        train_vla.main()
    state = run / "checkpoints/steps_500_state"
    state.mkdir(parents=True)
    (state.parent / "steps_500_pytorch_model.pt").touch()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    train_vla.main()
    resumed = OmegaConf.load(run / "train.yaml")
    assert resumed.trainer.is_resume and resumed.trainer.resume_step == 500
    assert resumed.trainer.max_train_steps == 5000
    assert resumed.trainer.save_interval == 5000
    assert resumed.trainer.eval_interval == 5000
    assert resumed.trainer.gradient_accumulation_steps == 4
    assert resumed.trainer.distributed_backend == "none"
