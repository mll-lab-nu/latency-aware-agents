"""Long-lived contracts for the single-experiment held-out sim2real protocol."""
from __future__ import annotations

import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from latency_bench.core.config import load_config
from latency_bench.cross_device import study
from latency_bench.cross_device import study_assets
from latency_bench.utils.io import read_json, write_json, write_jsonl


def archived(task="flappy", model="qwenoft"):
    registry = load_config(study.REGISTRY)
    entry = registry["tasks"][task][model]
    return entry, load_config(study.REPO_ROOT / entry["eval_config"])


def test_asset_resolution_pins_profile_model_and_relocates_archive(tmp_path, monkeypatch):
    registry = load_config(study.REGISTRY)
    entry, original = archived("ant", "qwenoft")
    snapshot = tmp_path / "snapshot"
    training_profile = {"env_fps": 60, "summary": {"mean_ms": 90}}
    profile_path = tmp_path / "training_profile.json"
    write_json(profile_path, training_profile)
    downloads = []

    def download_profile(**kwargs):
        assert kwargs == {
            "repo_id": entry["training_profile"]["repo_id"], "repo_type": "dataset",
            "revision": entry["training_profile"]["revision"],
            "filename": entry["training_profile"]["path"],
        }
        return str(profile_path)

    def download_snapshot(**kwargs):
        downloads.append(kwargs)
        if "allow_patterns" in kwargs:
            for name in kwargs["allow_patterns"]:
                file = snapshot / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("{}")
        return str(snapshot)

    monkeypatch.setattr(study_assets, "hf_hub_download", download_profile)
    monkeypatch.setattr(study_assets, "snapshot_download", download_snapshot)
    config, profile = study_assets.prepare_assets(registry, entry)
    assert profile == training_profile
    assert config["env"] == original["env"]
    assert config["scheduler"] == original["scheduler"]
    assert config["evaluation"] == original["evaluation"]
    assert config["policy"]["latency_prompt_key"] == 1
    assert config["policy"]["prompt_mode"] == "raw"
    assert config["policy"]["checkpoint_path"] == str(snapshot / entry["bundle"] / "checkpoints/model.pt")
    assert config["policy"]["task_manifest_path"] == str(snapshot / entry["bundle"] / "manifest.json")
    model_root = Path(config["policy"]["checkpoint_path"]).parents[1]
    # StarVLA loads these beside the checkpoint even when the adapter reads config.full.yaml.
    assert (model_root / "config.yaml").is_file()
    assert (model_root / "dataset_statistics.json").is_file()
    assert downloads[0]["revision"] == registry["model_revision"]
    assert downloads[1]["revision"] == registry["backbone"]["revision"]


@pytest.mark.parametrize("task,clocks,horizons,stride", [
    ("flappy", (10, 10, 10), (3600, 3600, 3600), 1),
    ("demon_attack", (60, 60, 60), (7200, 7200, 7200), 4),
    ("asterix", (60, 60, 60), (7200, 7200, 7200), 4),
    ("ant", (10, 5, 5), (3600, 1000, 1000), 1),
    ("inverted_pendulum", (10, 5, 5), (1000, 1000, 1000), 1),
    ("mikasa_intercept_grab_fast", (20, 20, 20), (60, 60, 60), 1),
])
def test_archived_task_contract_survives_clock_calibration(tmp_path, task, clocks, horizons, stride):
    for model, fps, horizon in zip(("qwenoft", "qwenpi_v3", "qwengr00t"), clocks, horizons):
        entry, original = archived(task, model)
        before = deepcopy(original)
        assert original["env"]["env_fps"] == fps
        assert original["env"]["obs_fps"] == fps / stride
        assert original["evaluation"]["eval_max_steps"] == horizon
        config = study.run_config(
            original, tmp_path, inference_device="cuda:0", simulator_device=None,
            env_fps=17.3, run_id=6,
        )
        assert config["env"]["obs_fps"] == 17.3 / stride
        assert config["env"] == {**original["env"], "env_fps": 17.3, "obs_fps": 17.3 / stride}
        assert config["scheduler"] == original["scheduler"]
        assert config["evaluation"]["eval_max_steps"] == horizon
        assert config["evaluation"]["eval_episodes"] == 20
        assert config["executor"]["inference_devices"] == ["cuda:0"]
        assert config["policy"] == {**original["policy"], "device": "cuda:0", "worker_python_executable": study.sys.executable}
        assert original == before
        assert entry["training_profile"]["revision"]


def _attempts(count, latency_ms):
    return [{
        "obs_id": index, "episode_id": index // 4, "obs_env_step": index % 4,
        "worker_id": 0, "submitted": True,
        "observation_to_action_latency_ms": latency_ms,
        "worker_service_time_ms": latency_ms - 1,
    } for index in range(count)]


def test_heldout_experiment_resumes_without_refitting_and_pairs_validation(tmp_path, monkeypatch):
    entry, base = archived()
    contract = {"checkpoint": "pinned-checkpoint", "profile": "pinned-training-profile"}
    deployment = {"gpu_class": "1x-test", "instance_id": "instance_test"}
    output = tmp_path / "experiment"
    interrupted = False
    observed_seeds = {}

    def evaluate(command, **kwargs):
        nonlocal interrupted
        config = load_config(command[command.index("--config") + 1])
        directory = Path(config["logging"]["output_dir"])
        if directory == output / "real/7" and not interrupted:
            interrupted = True
            (directory / "incomplete").touch()
            raise subprocess.CalledProcessError(1, command)
        seed = config["experiment"]["seed"]
        relative = str(directory.relative_to(output))
        observed_seeds[relative] = seed
        count = config["evaluation"]["eval_episodes"]
        if "calibration_samples" in config["evaluation"]:
            write_jsonl(directory / "observation_attempts.jsonl", _attempts(500, 20.0))
        else:
            offset = 2 if config["executor"]["mode"] == "simulated" else 0
            write_jsonl(directory / "episode_metrics.jsonl", [{
                "episode_id": i, "episode_return_env": seed + i + offset,
                "survival_steps": 60,
                "metadata": {"task_metrics": {"success": float(i % 2 == 0)}},
            } for i in range(count)])
            write_jsonl(directory / "observation_attempts.jsonl", _attempts(80, 20.0))

    monkeypatch.setattr(study.subprocess, "run", evaluate)
    arguments = dict(
        output=output, model="qwenoft", task="flappy", deployment=deployment,
        inference_device="cuda:0", simulator_device=None, contract=contract, fingerprint={},
    )
    # This profile was measured at 30 FPS but trained at the archived 10 FPS.
    training_profile = {"env_fps": 30, "summary": {"mean_ms": 80}}
    with pytest.raises(subprocess.CalledProcessError):
        study.run_experiment(base, training_profile, entry, **arguments, resume=False)
    profile_before = read_json(output / "profile/profile.json")
    assert profile_before["n_admitted_observations"] == 5 * 80
    calibration = read_json(output / "calibration/calibration.json")
    assert calibration["target_mean_frames"] == 0.8
    assert calibration["selected_env_fps"] == 40
    assert calibration["selected_obs_fps"] == 40
    first_probe = set(range(observed_seeds["calibration/probe_0"],
                            observed_seeds["calibration/probe_0"] + study.CALIBRATION_SAMPLES))
    second_probe = set(range(observed_seeds["calibration/probe_1"],
                             observed_seeds["calibration/probe_1"] + study.CALIBRATION_SAMPLES))
    assert first_probe.isdisjoint(second_probe)
    assert max(first_probe | second_probe) < study.ENV_SEED
    completed_receipt = output / "real/0/completed_receipt"
    completed_receipt.touch()
    study.run_experiment(base, training_profile, entry, **arguments, resume=True)
    assert not (output / "real/7/incomplete").exists()
    assert completed_receipt.exists()
    assert read_json(output / "profile/profile.json") == profile_before
    assert (output / "DONE").exists()
    assert len(read_json(output / "comparison.json")["methods"]) == 4
    success = read_json(output / "comparison.json")["task_metrics"]["success"]
    assert success["real"]["mean"] == 0.5
    assert success["methods"]["temporal"]["paired_sim_minus_real"]["mean"] == 0
    for method in study.METHODS:
        for i in range(5):
            assert observed_seeds[f"sim/{method}/{i}"] == observed_seeds[f"real/{i + 5}"]
        summary = read_json(output / "comparison.json")["methods"][method]
        assert summary["paired_sim_minus_real"]["mean"] == 2
        assert summary["paired_sim_minus_real"]["ci95_low"] == 2
    assert {observed_seeds[f"real/{i}"] for i in range(5)}.isdisjoint(
        {observed_seeds[f"real/{i}"] for i in range(5, 10)}
    )
    with pytest.raises(ValueError, match="contract"):
        study.run_experiment(base, training_profile, entry,
                             **(arguments | {"contract": {"checkpoint": "different"}}), resume=True)
    # Rebuilding from identical fit data remains invariant to held-out traces.
    write_jsonl(output / "real/9/observation_attempts.jsonl", _attempts(800, 9000))
    shutil.rmtree(output / "profile")
    study.fit_profile(output, model="qwenoft", task="flappy", deployment=deployment,
                      experiment_id=output.name, calibration=calibration, contract=contract)
    assert read_json(output / "profile/profile.json") == profile_before


@pytest.mark.parametrize("task,gpu,simulator,visible,inference", [
    ("flappy", 0, None, "0", "cuda:0"),
    ("flappy", 1, None, "1", "cuda:0"),
    ("ant", 1, None, "1", "cuda:0"),
    ("mikasa_intercept_grab_fast", 1, 0, "0,1", "cuda:1"),
])
def test_cli_isolates_requested_inference_gpu(tmp_path, monkeypatch, task, gpu, simulator, visible, inference):
    _, base = archived(task)
    monkeypatch.setattr(study, "prepare_assets", lambda *_: (base, {}))
    monkeypatch.setattr(study, "collect_hardware_fingerprint", lambda: {
        "gpus": [{"uuid": "GPU-A", "name": "NVIDIA GeForce RTX 3090"},
                 {"uuid": "GPU-B", "name": "NVIDIA GeForce RTX 3090"}],
        "gpu": {"topology": None},
    })
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LATENCY_BENCH_ENV_BACKEND_START_METHOD",
                "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID"):
        monkeypatch.setenv(key, "previous")
    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        assert study.os.environ["CUDA_VISIBLE_DEVICES"] == visible
        return kwargs["output"]

    monkeypatch.setattr(study, "run_experiment", run)
    arguments = ["--task", task, "--model", "qwenoft", "--gpu", str(gpu), "--output-root", str(tmp_path)]
    if simulator is not None:
        arguments += ["--simulator-gpu", str(simulator)]
    study.main(arguments)
    assert received["inference_device"] == inference
    assert received["simulator_device"] == (None if simulator is None else "cuda:0")
    assert received["deployment"]["gpu_class"] == "1x-rtx3090"
    assert len(received["deployment"]["gpus"]) == 1
    if task == "ant":
        assert study.os.environ["MUJOCO_EGL_DEVICE_ID"] == str(gpu)
