"""Long-lived contracts for the single-experiment held-out sim2real protocol."""
from __future__ import annotations

import shutil
import csv
import random
from fnmatch import fnmatch
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub.utils import HfHubHTTPError
from requests import Response

from latency_bench.core.config import load_config
from latency_bench.policy.model_source import model_file
from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.latency.samplers import build_sampler
from scripts.experiments.sim2real import budget, budget_report, study
from scripts.experiments.sim2real import assets, publication as publication_module, replay as replay_engine, runtime
from latency_bench.utils.io import read_json, write_json, write_jsonl
from training.common.checkpoints import capture_rng_state, restore_rng_state


PROTOCOL = load_config(study.REGISTRY)["protocol"]
BUDGET_CONFIG = load_config(study.REPO_ROOT / "configs/experiments/sim2real/budget.yaml")
REPEAT_CONFIG = load_config(study.REPO_ROOT / "configs/experiments/sim2real/repeats.yaml")
HOLDOUT_CONFIG = load_config(study.REPO_ROOT / "configs/experiments/sim2real/holdout.yaml")


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

    monkeypatch.setattr(assets, "hf_hub_download", download_profile)
    monkeypatch.setattr(assets, "snapshot_download", download_snapshot)
    config, profile = assets.prepare_assets(registry, entry)
    assert profile == training_profile
    assert config["env"] == original["env"]
    assert config["scheduler"] == original["scheduler"]
    assert config["evaluation"] == original["evaluation"]
    assert config["policy"]["latency_prompt_key"] == 1
    assert config["policy"]["prompt_mode"] == "raw"
    assert model_file(config["policy"], "checkpoint") == str(snapshot / entry["bundle"] / "checkpoints/model.pt")
    assert config["policy"]["task_manifest_path"] == str(snapshot / entry["bundle"] / "manifest.json")
    model_root = Path(model_file(config["policy"], "checkpoint")).parents[1]
    # StarVLA loads these beside the checkpoint even when the adapter reads config.full.yaml.
    assert (model_root / "config.yaml").is_file()
    assert (model_root / "dataset_statistics.json").is_file()
    assert downloads[0]["revision"] == registry["model_revision"]
    assert downloads[1]["revision"] == registry["backbone"]["revision"]
    monkeypatch.setattr(assets, "hf_hub_download", lambda **_: pytest.fail("training profile unavailable"))
    frozen_eval = tmp_path / "eval_template.yaml"
    study.write_generated_config(frozen_eval, original)
    replay_config = assets.prepare_model_assets(registry, entry, eval_config_path=frozen_eval)
    assert replay_config == config
    mikasa_entry, mikasa_archive = archived("mikasa_intercept_grab_fast", "qwenoft")
    mikasa = assets.prepare_model_assets(registry, mikasa_entry)
    assert mikasa["policy"]["action_prefix"] == {"mode": "none"}
    assert mikasa["env"] == mikasa_archive["env"]
    assert mikasa["scheduler"] == mikasa_archive["scheduler"]
    assert "action_prefix" not in mikasa_archive["policy"]


@pytest.mark.parametrize("task,clocks,horizons,stride", [
    ("flappy", (10, 10, 10), (3600, 3600, 3600), 1),
    ("demon_attack", (60, 60, 60), (7200, 7200, 7200), 4),
    ("asterix", (60, 60, 60), (7200, 7200, 7200), 4),
    ("ant", (10, 5, 5), (3600, 1000, 1000), 1),
    ("inverted_pendulum", (10, 5, 5), (1000, 1000, 1000), 1),
    ("mikasa_intercept_grab_fast", (20, 20, 20), (60, 60, 60), 1),
])
def test_archived_task_contract_survives_clock_calibration(tmp_path, task, clocks, horizons, stride):
    profile_path = tmp_path / "profile.json"
    for model, fps, horizon in zip(("qwenoft", "qwenpi_v3", "qwengr00t"), clocks, horizons):
        entry, original = archived(task, model)
        before = deepcopy(original)
        assert original["env"]["env_fps"] == fps
        assert original["env"]["obs_fps"] == fps / stride
        assert original["evaluation"]["max_env_steps"] == horizon
        config = runtime.run_config(
            original, tmp_path, inference_device="cuda:0", simulator_device=None,
            env_fps=17.3, run_id=6,
            protocol=PROTOCOL,
        )
        assert config["env"]["obs_fps"] == 17.3 / stride
        assert config["env"] == {**original["env"], "env_fps": 17.3, "obs_fps": 17.3 / stride}
        assert config["scheduler"] == original["scheduler"]
        assert config["evaluation"]["max_env_steps"] == horizon
        assert config["evaluation"]["episodes"] == 2
        assert config["experiment"]["seed"] == 2_000_120
        assert config["executor"]["inference_devices"] == ["cuda:0"]
        assert config["policy"] == original["policy"]
        assert original == before
        assert entry["training_profile"]["revision"]
        for method in PROTOCOL["methods"]:
            profile = {"summary": {"mean_ms": 80, "std_ms": 4}} if method in {"fixed", "normal"} else {}
            write_json(profile_path, profile)
            simulated = runtime.run_config(
                original, tmp_path, inference_device="cuda:0", simulator_device=None,
                env_fps=17.3, run_id=6, method=method, profile_path=profile_path, latency_seed=7,
                    protocol=PROTOCOL,
            )
            assert simulated["env"]["env_fps"] == 17.3
            assert simulated["env"]["obs_fps"] == 17.3 / stride
            assert simulated["evaluation"]["max_env_steps"] == horizon
            assert simulated["experiment"]["seed"] == 2_000_120
            assert simulated["latency"]["method"] == method
            assert simulated["latency"]["profile_path"] == str(profile_path)
            assert simulated["latency"]["seed"] == 7
            if method == "fixed":
                assert simulated["latency"]["fixed_latency_ms"] == 80
            elif method == "normal":
                assert simulated["latency"]["mean_latency_ms"] == 80
                assert simulated["latency"]["std_latency_ms"] == 4
            else:
                assert "fixed_latency_ms" not in simulated["latency"]
                assert "mean_latency_ms" not in simulated["latency"]
                assert "std_latency_ms" not in simulated["latency"]
            assert original == before


def test_prepare_replay_runtime_resolves_task_contract_and_uses_config_seed(tmp_path):
    """Sim2real replay owns task-input resolution followed by run-config seeding."""
    task_contract = tmp_path / "task_contract.json"
    write_json(task_contract, {"state_labels": ["position"], "robot_type": "unit_test"})
    seed = 2_000_100
    expected_python = random.Random(seed).random()
    expected_numpy = np.random.RandomState(seed).random_sample()
    expected_torch = torch.rand((), generator=torch.Generator().manual_seed(seed))
    config = {
        'experiment': {
            'seed': seed,
        },
        'executor': {
            'mode': 'simulated',
        },
        'env': {
            'name': 'g1_sonic_balance_simple',
            'frame_stack': 4,
            'env_fps': 20,
            'obs_fps': 20,
        },
        'latency': {
            'method': 'zero',
        },
        'policy': {
            'type': 'starvla',
            'task_contract_path': str(task_contract),
        },
        'evaluation': {
            'episodes': 2,
            'max_env_steps': 60,
        },
        'logging': {
            'output_dir': str(tmp_path / 'replay'),
        },
    }

    original_rng = capture_rng_state()
    try:
        runtime.prepare_replay_runtime(config)

        assert config["policy"]["task_contract_path"] == str(task_contract)
        assert config["env"]["frame_stack"] == 1
        assert config["env"]["state_space"] == {"labels": ["position"]}
        assert config["executor"]["parallel_envs"] == 1
        assert config["evaluation"]["suites"] == []
        assert config["executor"]["inference_pool"] is False
        assert random.random() == expected_python
        assert np.random.random() == expected_numpy
        torch.testing.assert_close(torch.rand(()), expected_torch)
    finally:
        restore_rng_state(original_rng)


def _attempts(count, latency_ms):
    return [{
        "obs_id": index, "episode_id": index // 4, "obs_env_step": index % 4,
        "worker_id": 0, "submitted": True,
        "observation_to_action_latency_ms": latency_ms,
        "worker_service_time_ms": latency_ms - 1,
    } for index in range(count)]


class MemoryHub:
    """HF boundary fixture retaining immutable snapshots and atomic stage commits."""

    def __init__(self):
        self.head = "0" * 40
        self.snapshots = {self.head: {}}
        self.profile_head = "a" * 40
        self.profile_snapshots = {self.profile_head: {}}
        self.fail_stage = "real/7"

    def repo_info(self, repo, **kwargs):
        return SimpleNamespace(sha=self.profile_head if repo == publication_module.PROFILE_REPO else self.head)

    def file_exists(self, repo, path, *, repo_type, revision=None):
        if revision is None:
            revision = self.repo_info(repo).sha
        snapshots = self.profile_snapshots if repo == publication_module.PROFILE_REPO else self.snapshots
        return path in snapshots[revision]

    def get_paths_info(self, repo, paths, *, repo_type, revision):
        snapshots = self.profile_snapshots if repo == publication_module.PROFILE_REPO else self.snapshots
        return [SimpleNamespace(path=path) for path in paths if path in snapshots[revision]]

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        if commit_message.endswith(": " + self.fail_stage):
            self.fail_stage = "never"
            raise RuntimeError("upload interrupted")
        profiles = repo_id == publication_module.PROFILE_REPO
        snapshots = self.profile_snapshots if profiles else self.snapshots
        head = self.profile_head if profiles else self.head
        files = dict(snapshots[head])
        for op in operations:
            value = op.path_or_fileobj
            files[op.path_in_repo] = Path(value).read_bytes() if isinstance(value, str) else value
        revision = f"{len(snapshots):040x}" if not profiles else f"a{len(snapshots):039x}"
        snapshots[revision] = files
        if profiles:
            self.profile_head = revision
        else:
            self.head = revision
        return SimpleNamespace(oid=revision)


def test_queued_publication_batches_completed_stages_and_recovers_receipts(tmp_path, monkeypatch):
    hub = MemoryHub()
    monkeypatch.setattr(publication_module, "HfApi", lambda: hub)
    publication = publication_module.StudyPublication(tmp_path, "case", defer_uploads=True)
    for run_id in (0, 1):
        path = tmp_path / "sim/fixed" / str(run_id)
        write_jsonl(path / "episode_metrics.jsonl", [{"episode_id": i} for i in range(2)])
        write_jsonl(path / "latencies.jsonl", [{"latency_ms": 12 + run_id}])
        write_jsonl(path / "actions.jsonl", [{"action": run_id}])
        write_jsonl(path / "steps.jsonl", [{"step": 0}])
        (path / "DONE").touch()
        assert publication.publish(f"sim/fixed/{run_id}", [path]) is None
    pending = sorted(tmp_path.glob("pending_uploads/**/*.json"))
    assert len(pending) == 2
    assert not (tmp_path / "uploads").exists()
    # Long-lived publication contract: a rate-limited commit retains completed work and its queue.
    commit = hub.create_commit
    def limited(**kwargs):
        if kwargs["repo_id"] == publication_module.PROFILE_REPO:
            return commit(**kwargs)
        response = Response()
        response.status_code = 429
        response.headers["Retry-After"] = "600"
        raise HfHubHTTPError("rate limited", response=response)
    monkeypatch.setattr(hub, "create_commit", limited)
    with pytest.raises(HfHubHTTPError, match="rate limited"):
        publication_module.publish_pending(pending)
    assert all(path.exists() for path in pending)
    assert not (tmp_path / "uploads").exists()
    assert all((tmp_path / f"sim/fixed/{run_id}/DONE").exists() for run_id in (0, 1))
    assert len(hub.profile_snapshots) == 2
    assert not hub.snapshots[hub.head]
    monkeypatch.setattr(hub, "create_commit", commit)
    revision = publication_module.publish_pending(pending)
    assert len(hub.snapshots) == 2
    assert len(hub.profile_snapshots) == 2
    assert not list(tmp_path.glob("pending_uploads/**/*.json"))
    for run_id in (0, 1):
        assert read_json(tmp_path / f"uploads/sim/fixed/{run_id}.json")["revision"] == revision
        assert f"case/sim/fixed/{run_id}/DONE" in hub.snapshots[revision]
        assert f"case/sim/fixed/{run_id}/latencies.jsonl" not in hub.snapshots[revision]
        assert f"case/sim/fixed/{run_id}/latencies.jsonl" in hub.profile_snapshots[hub.profile_head]
        for name in ("actions.jsonl", "steps.jsonl"):
            assert f"case/sim/fixed/{run_id}/{name}" not in hub.snapshots[revision]
            assert f"case/sim/fixed/{run_id}/{name}" in hub.profile_snapshots[hub.profile_head]
    source = read_json(tmp_path / "uploads/sim/fixed/0.json")
    assert source["repo_id"] == publication_module.DATASET_REPO
    # Recover a published batch after its local receipt was lost, without another commit.
    (tmp_path / "uploads/sim/fixed/0.json").unlink()
    publication.publish("sim/fixed/0", [tmp_path / "sim/fixed/0"])
    assert publication_module.publish_pending(list(tmp_path.glob("pending_uploads/**/*.json"))) == revision
    assert len(hub.snapshots) == 2


@pytest.mark.parametrize("parallel_runs", [1, 5])
def test_real_upload_resume_and_pinned_cross_host_replay(tmp_path, monkeypatch, parallel_runs):
    entry, base = archived()
    registry = load_config(study.REGISTRY)
    contract = {"checkpoint": "pinned-checkpoint", "protocol": {
        **PROTOCOL,
    }}
    deployment = {"gpu_class": "1x-test", "instance_id": "instance_test",
                  "gpus": [{"uuid": "private-gpu-uuid", "name": "test"}]}
    monkeypatch.setattr(study, "public_hardware_projection", lambda *a, **kw: {
        "gpu_class": "1x-test", "gpus": [{"name": "test", "slot": 0}],
    })
    base["policy"].update(source="local", root="/private-host", checkpoint="model.pt")
    output = tmp_path / "experiment"
    hub = MemoryHub()
    monkeypatch.setattr(publication_module, "HfApi", lambda: hub)
    evaluated = []
    observed_seeds = {}

    def evaluate(command, **kwargs):
        config = load_config(command[command.index("--config") + 1])
        directory = Path(config["logging"]["output_dir"])
        evaluated.append(directory)
        seed = config["experiment"]["seed"]
        observed_seeds[directory] = seed
        count = config["evaluation"]["episodes"]
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

    monkeypatch.setattr(runtime.subprocess, "run", evaluate)
    arguments = dict(
        output=output, model="qwenoft", task="flappy", deployment=deployment,
        inference_device="cuda:0", simulator_device=None, contract=contract, fingerprint={}, registry=registry,
    )
    training_profile = {"env_fps": 30, "summary": {"mean_ms": 80}}
    with pytest.raises(RuntimeError, match="upload interrupted"):
        study.run_real(base, training_profile, entry, **arguments, resume=False)
    assert (output / "real/7/DONE").exists()
    assert not (output / "real/8").exists()
    assert not (output / "uploads/real/7.json").exists()
    profile_before = read_json(output / "profile/profile.json")
    assert profile_before["n_admitted_observations"] == 5 * 80
    calibration = read_json(output / "calibration/calibration.json")
    assert calibration["target_mean_frames"] == 0.8
    assert calibration["selected_env_fps"] == calibration["selected_obs_fps"] == 40
    probe_seeds = [observed_seeds[output / f"calibration/probe_{i}"] for i in range(2)]
    assert set(range(probe_seeds[0], probe_seeds[0] + 500)).isdisjoint(range(probe_seeds[1], probe_seeds[1] + 500))
    assert max(probe_seeds) + 500 <= PROTOCOL["seed"]
    completed = tuple(evaluated)
    study.run_real(base, training_profile, entry, **arguments, resume=True)
    assert all(evaluated.count(directory) == 1 for directory in completed)
    assert read_json(output / "profile/profile.json") == profile_before
    assert (output / "REAL_DONE").exists()
    assert not (output / "sim").exists()
    assert not (output / "comparison.json").exists()
    assert len(read_json(output / "real_summary.json")["runs"]) == 10
    receipt = read_json(output / "uploads/real_complete.json")
    source_ref, source_revision = receipt["source_ref"], receipt["revision"]
    assert f"{source_ref}/REAL_DONE" in hub.snapshots[source_revision]
    public = hub.snapshots[source_revision]
    assert f"{source_ref}/contract.json" not in public
    assert f"{source_ref}/hardware.json" not in public
    assert f"{source_ref}/real/0/generated.yaml" not in public
    assert b"/private-host/model.pt" not in b"".join(public.values())
    assert b"private-gpu-uuid" not in b"".join(public.values())
    # A lost local upload receipt is recovered from the already completed remote stage.
    (output / "uploads/real/0.json").unlink()
    study.run_real(base, training_profile, entry, **arguments, resume=True)
    assert all(evaluated.count(directory) == 1 for directory in completed)

    def download(**kwargs):
        snapshots = hub.snapshots if kwargs["repo_id"] == publication_module.DATASET_REPO else hub.profile_snapshots
        if kwargs["repo_id"] == publication_module.DATASET_REPO:
            assert kwargs["revision"] == source_revision
        cache = tmp_path / "cache" / kwargs["revision"]
        for name, data in snapshots[kwargs["revision"]].items():
            if any(fnmatch(name, pattern) for pattern in kwargs["allow_patterns"]):
                path = cache / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        return str(cache)

    monkeypatch.setattr(publication_module, "snapshot_download", download)
    replay = tmp_path / "different-host" / "replay"
    manifest = publication_module.download_real_experiment(source_ref, source_revision, replay)
    assert read_json(replay / "profile/profile.json") == profile_before
    profile_source = read_json(replay / "profile_source.json")
    assert profile_source["repo_id"] == publication_module.PROFILE_REPO
    assert f"{source_ref}/profile/profile.json" not in hub.snapshots[source_revision]
    assert f"{source_ref}/profile/profile.json" in hub.profile_snapshots[profile_source["revision"]]
    changed = deepcopy(base)
    changed["env"].update(env_fps=30, obs_fps=30)
    changed["policy"].update(source="local", root="/new-host", checkpoint="model.pt")
    if parallel_runs == 5:
        monkeypatch.setattr(replay_engine, "build_replay_executor", lambda config, slots: (
            SimpleNamespace(run_warmup=lambda: None, close=lambda: None), config,
        ))
        def replay_runs(executor, configs, run_paths, runtime_dir, completed, **kwargs):
            for run_id, path in run_paths.items():
                config = configs[run_id]
                assert config["experiment"]["seed"] == PROTOCOL["seed"] + 20 * run_id
                assert config["latency"]["seed"] == run_id - 5
                runtime.execute_stage(config, path, entry)
                completed(run_id, path)
        monkeypatch.setattr(replay_engine, "replay_runs", replay_runs)
    study.run_sim(changed, entry, output=replay, manifest=manifest, source_revision=source_revision,
                  inference_device="cuda:3", simulator_device=None,
                  fingerprint={"gpus": [{"uuid": "replay-gpu", "name": "test"}]}, parallel_runs=parallel_runs)
    completed_sim = tuple(evaluated)
    study.run_sim(changed, entry, output=replay, manifest=manifest, source_revision=source_revision,
                  inference_device="cuda:3", simulator_device=None,
                  fingerprint={"gpus": [{"uuid": "replay-gpu", "name": "test"}]}, parallel_runs=parallel_runs)
    assert tuple(evaluated) == completed_sim
    assert (replay / "SIM_DONE").exists()
    assert read_json(replay / "simulation_source.json")["revision"] == source_revision
    with pytest.raises(ValueError, match="Simulation contract"):
        study.run_sim(changed, entry, output=replay, manifest=manifest, source_revision=source_revision,
                      inference_device="cuda:0", simulator_device=None,
                      fingerprint={"gpus": [{"uuid": "another-gpu", "name": "test"}]})
    summary = read_json(replay / "comparison.json")
    assert summary["task_metrics"]["success"]["real"]["mean"] == 0.5
    for method in PROTOCOL["methods"]:
        for i in range(5):
            config = load_config(replay / f"sim/{method}/{i}/generated.yaml")
            assert config["env"]["env_fps"] == 40
            assert config["executor"]["inference_devices"] == ["cuda:3"]
            assert observed_seeds[replay / f"sim/{method}/{i}"] == observed_seeds[output / f"real/{i + 5}"]
        assert summary["methods"][method]["paired_sim_minus_real"]["mean"] == 2
    assert {observed_seeds[output / f"real/{i}"] for i in range(5)}.isdisjoint(
        {observed_seeds[output / f"real/{i}"] for i in range(5, 10)})
    with pytest.raises(ValueError, match="contract"):
        study.run_real(base, training_profile, entry,
                       **(arguments | {"contract": {"checkpoint": "different"}}), resume=True)
    write_jsonl(output / "real/9/observation_attempts.jsonl", _attempts(800, 9000))
    shutil.rmtree(output / "profile")
    study.fit_profile(output, model="qwenoft", task="flappy", deployment=deployment,
                      experiment_id=output.name, calibration=calibration,
                      contract=read_json(output / "experiment.json"))
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

    monkeypatch.setattr(study, "run_real", run)
    arguments = ["--stage", "real", "--task", task, "--model", "qwenoft", "--gpu", str(gpu), "--output-root", str(tmp_path)]
    if simulator is not None:
        arguments += ["--simulator-gpu", str(simulator)]
    study.main(arguments)
    assert received["inference_device"] == inference
    assert received["simulator_device"] == (None if simulator is None else "cuda:0")
    assert received["deployment"]["gpu_class"] == "1x-rtx3090"
    assert len(received["deployment"]["gpus"]) == 1
    if task == "ant":
        assert study.os.environ["MUJOCO_EGL_DEVICE_ID"] == str(gpu)



def test_budget_split_sizes_and_total_replay_budget(tmp_path):
    # Long-lived fit-budget protocol, owned by the sim2real experiment.
    for model in BUDGET_CONFIG["models"]:
        manifest = budget.initialize(tmp_path / model, model, BUDGET_CONFIG)
        assert manifest["source"] == {
            "ref": BUDGET_CONFIG["sources"][model]["ref"], "revision": BUDGET_CONFIG["sources"][model]["revision"],
            "reference_return": BUDGET_CONFIG["sources"][model]["reference_return"],
        }
    assert budget.initialize(tmp_path / model, model, BUDGET_CONFIG) == manifest
    changed = deepcopy(BUDGET_CONFIG)
    changed["protocol"]["seed"] += 1
    with pytest.raises(ValueError, match="Budget protocol changed"):
        budget.initialize(tmp_path / model, model, changed)
    assert read_json(tmp_path / model / "budget.json") == manifest
    splits = budget.split_manifest(BUDGET_CONFIG)
    assert splits["original"]["valid"] == [5, 6, 7, 8, 9]
    assert splits["original"]["fit_order"] == [*range(5), *range(10, 55)]
    assert splits["random"]["valid"] == [2, 12, 16, 26, 48]
    episodes = 0
    for partition in splits.values():
        previous = set()
        for count in BUDGET_CONFIG["fit_sizes"]:
            fit = partition["fit_order"][:count]
            assert len(set(fit)) == count
            assert set(fit).isdisjoint(partition["valid"])
            assert previous <= set(fit)
            previous = set(fit)
            episodes += len(partition["valid"]) * 2 * len(PROTOCOL["methods"]) * len(BUDGET_CONFIG["models"])
        assert set(partition["fit_order"]) | set(partition["valid"]) == set(range(55))
    assert episodes == 960


@pytest.mark.parametrize("replica_seed", [False, True])
def test_budget_real_resume_and_fit_data_isolation(tmp_path, monkeypatch, replica_seed):
    manifest = budget.initialize(tmp_path / "qwenoft", "qwenoft", BUDGET_CONFIG)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    entry, base = archived()
    source = tmp_path / "source"
    output = tmp_path / "qwenoft"
    for run_id in range(10):
        path = source / "real" / str(run_id)
        write_jsonl(path / "observation_attempts.jsonl", _attempts(8, 20 + run_id))
        (path / "DONE").touch()
    write_json(source / "experiment.json", {"task": "flappy", "target_deployment": {"instance_id": "test", "gpu_class": "1x-test"}})
    write_json(source / "calibration/calibration.json", {"selected_env_fps": 10})
    monkeypatch.setattr(runtime, "runtime_base", lambda *a: (base, {"assets": entry}, 10))
    monkeypatch.setattr(runtime, "collect_hardware_fingerprint", lambda: {
        "gpus": [{"uuid": "GPU-test"}],
        "gpu": {"driver_version": "same"}, "packages": {"torch": "same"}, "cpu": {"model_name": "test", "cpu_mhz": "variable"},
        "memory": {"total_kib": 1024}, "system": {"hostname": "test"}})
    evaluated = []
    def execute(config, path, entry):
        evaluated.append(config["experiment"]["seed"])
        write_jsonl(path / "observation_attempts.jsonl", _attempts(8, 20))
        (path / "DONE").touch()
    monkeypatch.setattr(runtime, "execute_stage", execute)
    publication = SimpleNamespace(publish=lambda *args: None)
    budget.measure_real(output, source, "qwenoft", "python", publication, manifest)
    budget.measure_real(output, source, "qwenoft", "python", publication, manifest)
    assert evaluated == [2000000 + 20 * i for i in range(10, 55)]
    assert (output / "REAL_DONE").exists()
    # Recover a crash after DONE but before finalizing timing, without re-evaluation.
    timing = output / "timings/real_10.json"
    started = read_json(timing)["started_at"]
    write_json(timing, {"started_at": started})
    budget.measure_real(output, source, "qwenoft", "python", publication, manifest)
    assert len(evaluated) == 45
    assert read_json(timing)["seconds"] >= 0
    monkeypatch.setattr(runtime, "collect_hardware_fingerprint", lambda: {
        "gpus": [{"uuid": "GPU-test"}],
        "gpu": {"driver_version": "changed"}, "packages": {"torch": "same"}, "cpu": {"model_name": "test", "cpu_mhz": "variable"},
        "memory": {"total_kib": 1024}, "system": {"hostname": "test"}})
    with pytest.raises(ValueError, match="execution contract changed"):
        budget.measure_real(output, source, "qwenoft", "python", publication, manifest)
    # Keep the real fitter: admission counts and provenance expose leakage.
    budget.fit_profiles(output, source, "qwenoft", publication, partitions=budget.split_manifest(BUDGET_CONFIG),
                        fit_sizes=manifest["fit_sizes"], source_ref=manifest["source"])
    for split, partition in budget.split_manifest(BUDGET_CONFIG).items():
        for count in BUDGET_CONFIG["fit_sizes"]:
            profile = read_json(output / "profiles" / split / str(count) / "profile.json")
            assert profile["n_admitted_observations"] == 8 * count
            assert profile["provenance"]["fit_session_ids"] == partition["fit_order"][:count]
            assert profile["provenance"]["validation_session_ids"] == partition["valid"]
    config = {"latency": {"method": "temporal", "seed": 0, "profile_worker_slot": 0,
                          "profile_path": str(output / "profiles/original/5/profile.json")}}
    clock = EnvClock(env_fps=10, obs_fps=10)
    groups, seeds, sources = replay_engine.grouped_inputs(
        {run_id: {**config["latency"], "seed": manifest["latency_seeds"][str(run_id)]} for run_id in [3, 1]},
        clock, protocol=PROTOCOL, replica_seed=replica_seed,
    )
    assert groups == [[60, 61], [20, 21]]
    assert seeds == [2000060, 2000061, 2000020, 2000021]
    for run_id, group in zip([3, 1], groups):
        reference = build_sampler({**config["latency"],
                                   "seed": (derive_seed(manifest["latency_seeds"][str(run_id)], vector_index=0)
                                            if replica_seed else manifest["latency_seeds"][str(run_id)])}, clock=clock)
        for episode_id in group:
            context = {"episode_id": episode_id, "worker_slot": 0}
            for _ in range(20):
                assert sources[episode_id].sample(context) == reference.sample(context)


def test_budget_error_uses_absolute_mean_difference_and_paired_runs():
    result, draws = budget_report.comparison([0, 20], [10, 10], 100, paired=True, seed=1, repetitions=2000)
    assert result["absolute_error_pp"] == 0  # Not mean absolute paired error (10).
    result, draws = budget_report.comparison([10, 20, 30], [12, 22, 32], 200, paired=True, seed=1, repetitions=2000)
    assert result["signed_error_pp"] == 1
    assert result["absolute_ci_low_pp"] == result["absolute_ci_high_pp"] == 1
    assert all(draws == 1)
    independent, _ = budget_report.comparison([10, 20, 30], [12, 22, 32], 200, paired=False, seed=1, repetitions=2000)
    assert independent["signed_ci_high_pp"] > independent["signed_ci_low_pp"]
    unequal, _ = budget_report.comparison([0, 20, 40, 60], [10, 30], 100, paired=False, seed=1, repetitions=2000)
    assert unequal["signed_error_pp"] == -10


@pytest.mark.parametrize("external_timings", [False, True])
def test_budget_report_aggregates_equal_case_weight_and_exports(tmp_path, external_timings):
    campaign_config = deepcopy(BUDGET_CONFIG)
    campaign_config["sources"]["qwenoft"]["reference_return"] *= 2
    # Long-lived report contract: two case means get equal weight irrespective
    # of run count or reference scale, and a known offset survives all exports.
    for model in campaign_config["models"]:
        budget.initialize(tmp_path / model, model, campaign_config)
        output = tmp_path / model
        write_json(output / "source/calibration/calibration.json", {"selected_env_fps": 10})
        for run_id in range(55):
            path = output / "real" / str(run_id)
            write_jsonl(path / "episode_metrics.jsonl", [{"episode_return_env": run_id + e} for e in range(2)])
            write_jsonl(path / "observation_attempts.jsonl", _attempts(8, 20))
        for split, partition in budget.split_manifest(campaign_config).items():
            for count in campaign_config["fit_sizes"]:
                for method in PROTOCOL["methods"]:
                    for run_id in partition["valid"]:
                        path = output / "sim" / split / str(count) / method / "runs" / str(run_id)
                        write_jsonl(path / "episode_metrics.jsonl", [
                            {"episode_return_env": run_id + e + (1 + run_id % 2) * campaign_config["sources"][model]["reference_return"] / 100}
                            for e in range(2)
                        ])
        write_json(output / "runtime/setups/test.json", {"stage": "sim_setup", "model": model, "seconds": 10, "gpu_count": 1})
        for run_id, seconds in ((10, 3600), (15, 7200)):
            write_json(output / "timings" / f"real_{run_id}.json", {
                "stage": "real", "model": model, "run_ids": [run_id], "seconds": seconds, "gpu_count": 1})
        for split in budget.split_manifest(campaign_config):
            for count, seconds in ((5, 1800), (10, 3600)):
                write_json(output / "sim" / split / str(count) / "fixed/timings/test.json", {
                    "stage": "sim", "model": model, "split": split, "fit_runs": count,
                    "method": "fixed", "run_ids": budget.split_manifest(campaign_config)[split]["valid"],
                    "seconds": seconds, "gpu_count": 1})
        (output / "SIM_DONE").touch()
    timing_root = tmp_path
    if external_timings:
        timing_root = tmp_path / "external_timings"
        for model in campaign_config["models"]:
            for path in budget_report._timing_paths(tmp_path / model):
                destination = timing_root / model / path.relative_to(tmp_path / model)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    # A filename is evidence, not the run identity used for cost attribution.
    for model in campaign_config["models"]:
        (timing_root / model / "timings/real_10.json").rename(timing_root / model / "timings/measurement.json")
    report = budget_report.make_report(tmp_path, timing_root=timing_root)
    with (report / "nmae.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 6 * 4 * 2
    with (report / "case_errors.csv").open() as stream:
        cases = {(r["model"], r["split"], r["fit_runs"], r["method"], r["reference_split"]): r for r in csv.DictReader(stream)}
    for row in rows:
        partition = budget.split_manifest(campaign_config)[row["split"]]
        ids = partition["valid"] if row["reference_split"] == "valid" else partition["fit_order"][:int(row["fit_runs"])]
        sim_base = sum(partition["valid"]) / 5
        real_base = sum(ids) / len(ids)
        sim_offset = 1 + sum(i % 2 for i in partition["valid"]) / 5
        expected = sum(abs(100 * (sim_base - real_base) / campaign_config["sources"][m]["reference_return"] + sim_offset)
                       for m in campaign_config["models"]) / len(campaign_config["models"])
        assert float(row["nmae_pp"]) == pytest.approx(expected)
        if row["reference_split"] == "valid":
            # Shared run IDs retain the perfectly correlated normalized case errors.
            case = cases["qwenoft", row["split"], row["fit_runs"], row["method"], "valid"]
            assert float(row["ci_low_pp"]) == pytest.approx(float(case["absolute_ci_low_pp"]))
            assert float(row["ci_high_pp"]) == pytest.approx(float(case["absolute_ci_high_pp"]))
        for model in campaign_config["models"]:
            fit = cases[model, row["split"], row["fit_runs"], row["method"], "fit"]
            valid = cases[model, row["split"], row["fit_runs"], row["method"], "valid"]
            assert fit["sim_mean"] == valid["sim_mean"]
            assert int(fit["sim_runs"]) == int(valid["sim_runs"]) == 5
            assert int(fit["real_runs"]) == int(row["fit_runs"])
            assert int(valid["real_runs"]) == 5
    assert len(list(tmp_path.glob("*/sim/*/*/*/runs/*/episode_metrics.jsonl"))) * 2 == 960
    for split in budget.split_manifest(campaign_config):
        assert (report / f"valid_{split}.png").exists()
        assert (report / f"valid_{split}.pdf").exists()
    assert (tmp_path / "REPORT_DONE").exists()
    assert all(item["descriptive_plateau_from_runs"] == 5 for item in read_json(report / "stability.json"))
    assert all(item["budget_supported_at_2pp_precision"] for item in read_json(report / "stability.json"))

    with (report / "cumulative_costs.csv").open() as stream:
        costs = {(r["model"], r["split"], int(r["fit_runs"])): r for r in csv.DictReader(stream)}
    for model in campaign_config["models"]:
        for split in budget.split_manifest(campaign_config):
            assert float(costs[model, split, 5]["cumulative_replay_gpu_hours"]) == .5
            assert float(costs[model, split, 10]["cumulative_replay_gpu_hours"]) == 1.5
            assert float(costs[model, split, 50]["full_collection_gpu_hours"]) == 3
        assert float(costs[model, "original", 5]["original_prefix_equivalent_gpu_hours"]) == 0
        assert float(costs[model, "original", 10]["original_prefix_equivalent_gpu_hours"]) == 1
        assert float(costs[model, "original", 20]["original_prefix_equivalent_gpu_hours"]) == 3



@pytest.mark.parametrize("repeated", [False, True])
def test_budget_simulates_fixed_validation_runs_once_and_resumes(tmp_path, monkeypatch, repeated):
    from scripts.experiments.sim2real import budget_repeats
    partitions = ({str(seed): partition for seed, partition in budget_repeats.random_partitions(BUDGET_CONFIG, REPEAT_CONFIG["seeds"]).items()
                   if seed != str(BUDGET_CONFIG["split_seed"])} if repeated else budget.split_manifest(BUDGET_CONFIG))
    entry, base = archived()
    monkeypatch.setattr(runtime, "runtime_base", lambda *args: (base, {"assets": entry}, 10))
    monkeypatch.setattr(runtime, "record_execution_contract", lambda *args: None)
    monkeypatch.setattr(runtime, "prepare_replay_runtime", lambda config: None)
    executor = SimpleNamespace(run_warmup=lambda: None, close=lambda: None)
    monkeypatch.setattr(budget, "build_executor", lambda *args, **kwargs: executor)
    publication = SimpleNamespace(publish=lambda *args: None)
    evaluated = []
    def replay(executor, config, path, run_ids, completed, **kwargs):
        split = path.parts[-3]
        assert run_ids == partitions[split]["valid"]
        for run_id in run_ids:
            destination = path / "runs" / str(run_id)
            write_jsonl(destination / "episode_metrics.jsonl", [{"episode_return_env": run_id}] * 2)
            (destination / "DONE").touch()
            completed(run_id, destination)
            evaluated.append((path, run_id))
    monkeypatch.setattr(replay_engine, "replay_group", replay)
    def replay_settings(executor, jobs, output, publication, **kwargs):
        assert len(jobs) == 10
        for run_id, path, config, run_latency_seed in jobs:
            assert run_latency_seed == manifest["latency_seeds"][str(run_id)]
            assert run_id in partitions[path.parents[3].name]["valid"]
            write_jsonl(path / "episode_metrics.jsonl", [{"episode_return_env": run_id}] * 2)
            (path / "DONE").touch()
            evaluated.append((path.parent.parent, run_id))
    monkeypatch.setattr(replay_engine, "replay_settings", replay_settings)
    for model in BUDGET_CONFIG["models"]:
        manifest = budget.initialize(tmp_path / model, model, BUDGET_CONFIG)
        output = tmp_path / model
        for split in partitions:
            for count in BUDGET_CONFIG["fit_sizes"]:
                write_json(output / "profiles" / split / str(count) / "profile.json", {
                    "summary": {"mean_ms": 73, "std_ms": 2},
                })
        budget.simulate(output, output / "source", model, "python", publication, 10 if repeated else 8,
                        partitions=partitions, manifest=manifest, combine_settings=repeated)
        budget.simulate(output, output / "source", model, "python", publication, 10 if repeated else 8,
                        partitions=partitions, manifest=manifest, combine_settings=repeated)
        assert (output / "SIM_DONE").exists()
    assert len(evaluated) * 2 == (1920 if repeated else 960)


def test_random_repeats_preserve_first_split_and_average_absolute_errors(tmp_path):
    # Long-lived repeated-split protocol: reuse repeat one; measure split variation without cancellation.
    from scripts.experiments.sim2real import budget_repeats
    partitions = budget_repeats.random_partitions(BUDGET_CONFIG, REPEAT_CONFIG["seeds"])
    assert partitions[str(BUDGET_CONFIG["split_seed"])] == budget.split_manifest(BUDGET_CONFIG)['random']
    priority = list(range(55))
    budget.random.Random(BUDGET_CONFIG["split_seed"]).shuffle(priority)
    for partition in partitions.values():
        assert len(partition['valid']) == 5
        assert partition['fit_order'] == [i for i in priority if i not in partition['valid']]
        assert set(partition['fit_order']).isdisjoint(partition['valid'])
        assert len(partition['fit_order']) == 50
    assert len(partitions) == 5
    assert (len(partitions) - 1) * len(BUDGET_CONFIG["models"]) * len(BUDGET_CONFIG["fit_sizes"]) * len(PROTOCOL["methods"]) * 10 == 1920
    root, previous = tmp_path / 'new', tmp_path / 'previous'
    for model in BUDGET_CONFIG["models"]:
        budget.initialize(previous / model, model, BUDGET_CONFIG)
        (previous / model / "source").mkdir()
        (previous / model / "real").mkdir()
        (previous / model / "SIM_DONE").touch()
        (previous / model / "REAL_DONE").touch()
        write_json(previous / model / "uploads/sim_complete.json", {"revision": "pinned-simulation"})
    write_json(previous / "uploads/report.json", {"revision": "pinned-report"})
    manifest = budget_repeats.prepare_inputs(root, previous, REPEAT_CONFIG)
    assert manifest["new_episodes"] == 1920
    assert manifest["random_episodes_including_previous"] == 2400
    assert manifest["partitions"] == partitions
    assert budget_repeats.prepare_inputs(root, previous, REPEAT_CONFIG) == manifest
    signed_errors = [1, -2, 3, -4, 5]
    for model in BUDGET_CONFIG["models"]:
        (root / model / 'SIM_DONE').touch()
        for run_id in range(55):
            write_jsonl(previous / model / 'real' / str(run_id) / 'episode_metrics.jsonl',
                        [{'episode_return_env': 100}] * 2)
        for index, seed in enumerate(REPEAT_CONFIG["seeds"]):
            parent, split = (previous, 'random') if index == 0 else (root, str(seed))
            for count in BUDGET_CONFIG["fit_sizes"]:
                for method in PROTOCOL["methods"]:
                    for run_id in partitions[str(seed)]['valid']:
                        write_jsonl(parent / model / 'sim' / split / str(count) / method / 'runs' / str(run_id) / 'episode_metrics.jsonl',
                                    [{'episode_return_env': 100 + signed_errors[index] * BUDGET_CONFIG["sources"][model]['reference_return'] / 100}] * 2)
    report = budget_repeats.repeated_report(root, previous)
    with (report / 'random_nmae.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    for row in rows:
        assert float(row['nmae_mean_pp']) == pytest.approx(3)
        assert float(row['nmae_sd_pp']) == pytest.approx(2.5 ** .5)
        assert int(row['n_splits']) == 5
    assert (report / 'valid_random.png').exists()
    assert (report / 'valid_random.pdf').exists()


def test_resident_replay_closes_executor_when_warmup_fails(monkeypatch, tmp_path):
    manifest = budget.initialize(tmp_path / "budget", "qwenoft", BUDGET_CONFIG)
    entry, base = archived()
    monkeypatch.setattr(runtime, "runtime_base", lambda *args: (base, {"assets": entry}, 10))
    monkeypatch.setattr(runtime, "record_execution_contract", lambda *args: None)
    monkeypatch.setattr(runtime, "prepare_replay_runtime", lambda config: None)
    closed = []
    def fail():
        raise RuntimeError("warmup failed")
    executor = SimpleNamespace(run_warmup=fail, close=lambda: closed.append(True))
    monkeypatch.setattr(budget, "build_executor", lambda *args, **kwargs: executor)
    write_json(tmp_path / "profiles/original/5/profile.json", {"summary": {"mean_ms": 1, "std_ms": 0}})
    with pytest.raises(RuntimeError, match="warmup failed"):
        budget.simulate(tmp_path, tmp_path, "qwenoft", "python", None, 2,
                        partitions=budget.split_manifest(BUDGET_CONFIG), manifest=manifest)
    assert closed == [True]


def test_holdout_budget_evaluates_four_methods_once_on_27_runs(tmp_path):
    # Long-lived holdout protocol/report contract, owned by the sim2real experiment.
    from scripts.experiments.sim2real import budget_holdout
    root, previous = tmp_path / 'holdout', tmp_path / 'previous'
    for model in BUDGET_CONFIG["models"]:
        budget.initialize(previous / model, model, BUDGET_CONFIG)
        (previous / model / 'source').mkdir(parents=True)
        (previous / model / 'REAL_DONE').touch()
        for run_id in range(55):
            write_jsonl(previous / model / 'real' / str(run_id) / 'episode_metrics.jsonl',
                        [{'episode_return_env': 100 + run_id}] * 2)
    manifest = budget_holdout.prepare(root, previous, HOLDOUT_CONFIG)
    fit, valid = manifest['partition']['fit_order'], manifest['partition']['valid']
    assert len(fit) == 28 and len(valid) == 27
    assert set(fit).isdisjoint(valid) and set(fit + valid) == set(range(55))
    assert budget_holdout.prepare(root, previous, HOLDOUT_CONFIG) == manifest
    _, base = archived()
    for model, sign in zip(BUDGET_CONFIG["models"], [1, -1]):
        output = root / model
        profile = output / 'profiles/holdout/28/profile.json'
        write_json(profile, {'summary': {'mean_ms': 73, 'std_ms': 2}})
        jobs = list(budget_holdout.simulation_jobs(output, base, 10, manifest))
        assert len(jobs) * 2 * len(BUDGET_CONFIG["models"]) == 432
        assert len({path for _, path, _, _ in jobs}) == 108
        for run_id, path, config, latency_seed in jobs:
            assert latency_seed == manifest['latency_seed_base'] + run_id
            assert config['latency']['profile_path'] == str(profile)
            # Opposite model biases must not cancel in NMAE.
            value = 100 + run_id + sign * 2 * BUDGET_CONFIG["sources"][model]['reference_return'] / 100
            write_jsonl(path / 'episode_metrics.jsonl', [{'episode_return_env': value}] * 2)
        (output / 'SIM_DONE').touch()
    destination = budget_holdout.report(root, manifest)
    with (destination / 'nmae.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(row['nmae_pp']) == pytest.approx(2) for row in rows)
    assert (destination / 'heldout_calibration.pdf').exists()
