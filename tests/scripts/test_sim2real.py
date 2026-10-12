"""Long-lived contracts for the single-experiment held-out sim2real protocol."""
from __future__ import annotations

import shutil
import csv
import random
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from latency_bench.core.config import load_config
from latency_bench.policy.model_source import model_file
from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.latency.samplers import build_sampler
from scripts.experiments.sim2real import budget, budget_report, study
from scripts.experiments.sim2real import assets, replay as replay_engine, runtime
from latency_bench.utils.io import read_json, write_json, write_jsonl, write_yaml
from training.common.checkpoints import capture_rng_state, restore_rng_state


PROTOCOL = load_config(study.REGISTRY)["protocol"]
BUDGET_CONFIG = load_config(study.REPO_ROOT / "configs/experiments/sim2real/budget.yaml")
REPEAT_CONFIG = load_config(study.REPO_ROOT / "configs/experiments/sim2real/repeats.yaml")
HOLDOUT_CONFIG = load_config(study.REPO_ROOT / "configs/experiments/sim2real/holdout.yaml")


def archived(task="flappy", model="qwenoft"):
    registry = load_config(study.REGISTRY)
    entry = registry["tasks"][task][model]
    return entry, load_config(study.REPO_ROOT / entry["eval_config"])


def test_cli_replays_completed_local_real_experiment(tmp_path, monkeypatch):
    """Long-lived local replay input contract owned by the Sim2Real recipe suite."""
    entry, base = archived()
    manifest = {"task": "flappy", "model": "qwenoft", "assets": entry, "protocol": PROTOCOL}
    write_json(tmp_path / "experiment.json", manifest)
    (tmp_path / "REAL_DONE").touch()
    received = {}
    monkeypatch.setattr(study, "prepare_model_assets", lambda *args, **kwargs: base)
    monkeypatch.setattr(study, "collect_hardware_fingerprint", lambda: {"gpus": []})
    monkeypatch.setattr(study, "HfApi", lambda: pytest.fail("Local source selection contacted the Hub"))
    monkeypatch.setattr(study, "run_sim", lambda *args, **kwargs: received.update(kwargs) or kwargs["output"])
    study.main(["--stage", "sim", "--source-dir", str(tmp_path), "--gpu", "0"])
    assert received["output"] == tmp_path
    assert received["manifest"] == manifest
    assert received["source_revision"] is None


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
    write_yaml(frozen_eval, original)
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
    budget.measure_real(output, source, "qwenoft", "python", manifest)
    budget.measure_real(output, source, "qwenoft", "python", manifest)
    assert evaluated == [2000000 + 20 * i for i in range(10, 55)]
    assert (output / "REAL_DONE").exists()
    # Recover a crash after DONE but before finalizing timing, without re-evaluation.
    timing = output / "timings/real_10.json"
    started = read_json(timing)["started_at"]
    write_json(timing, {"started_at": started})
    budget.measure_real(output, source, "qwenoft", "python", manifest)
    assert len(evaluated) == 45
    assert read_json(timing)["seconds"] >= 0
    monkeypatch.setattr(runtime, "collect_hardware_fingerprint", lambda: {
        "gpus": [{"uuid": "GPU-test"}],
        "gpu": {"driver_version": "changed"}, "packages": {"torch": "same"}, "cpu": {"model_name": "test", "cpu_mhz": "variable"},
        "memory": {"total_kib": 1024}, "system": {"hostname": "test"}})
    with pytest.raises(ValueError, match="execution contract changed"):
        budget.measure_real(output, source, "qwenoft", "python", manifest)
    # Keep the real fitter: admission counts and provenance expose leakage.
    budget.fit_profiles(output, source, "qwenoft", partitions=budget.split_manifest(BUDGET_CONFIG),
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
    evaluated = []
    def replay(executor, config, path, run_ids, **kwargs):
        split = path.parts[-3]
        assert run_ids == partitions[split]["valid"]
        for run_id in run_ids:
            destination = path / "runs" / str(run_id)
            write_jsonl(destination / "episode_metrics.jsonl", [{"episode_return_env": run_id}] * 2)
            (destination / "DONE").touch()
            evaluated.append((path, run_id))
    monkeypatch.setattr(replay_engine, "replay_group", replay)
    def replay_settings(executor, jobs, output, **kwargs):
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
        budget.simulate(output, output / "source", model, "python", 10 if repeated else 8,
                        partitions=partitions, manifest=manifest, combine_settings=repeated)
        budget.simulate(output, output / "source", model, "python", 10 if repeated else 8,
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
        budget.simulate(tmp_path, tmp_path, "qwenoft", "python", 2,
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
