"""The experiment recipe preserves fitting isolation and publishable evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.config import load_config
from latency_bench.profiling.spec import resolve_profile_config
from latency_bench.utils.io import read_json, write_json, write_jsonl, write_yaml
from scripts.experiments.temporal_profiles import profiling as recipe


def recorded_run(source, config):
    deployment = {"gpu_class": "1x-test-gpu", "instance_id": "instance_test",
                  "gpus": [], "topology_links": []}
    write_json(source / "run_contract.json", {"config": config, "deployment": deployment})
    write_json(source / "hardware.json", {"torch": {}, "gpu": {}, "gpus": []})
    rng = np.random.default_rng(731)
    for session in range(config["profiling"]["repeats"]):
        busy = np.exp(rng.normal(4 if session < 3 else 9, .2, 100))
        overhead = np.exp(rng.normal(1, .1, 100))
        rows = [{"obs_id": i, "episode_id": i // 50, "obs_env_step": i,
                 "submitted": True, "worker_id": 0,
                 "observation_to_action_latency_ms": float(b + o),
                 "worker_service_time_ms": float(b)} for i, (b, o) in enumerate(zip(busy, overhead))]
        path = source / "sessions" / str(session)
        write_jsonl(path / "observation_attempts.jsonl", rows)
        write_json(path / "warmup_summary.json", {"warmup_status": "stable"})
        (path / "DONE").touch()
    return source


@pytest.fixture
def raw_run(tmp_path):
    config = resolve_profile_config(load_config(recipe.RECIPE))
    config["provenance"] = {"model_artifact": {"source": "hf", "revision": "measured-revision"}}
    source = recorded_run(tmp_path / "raw-run", config)
    # The automatic all-session fit and stray model weights must not be exported.
    write_json(source / "profile/profile.json", {"wrong_fit_ids": [0, 1, 2, 3, 4]})
    (source / "sessions/0/model.pt").write_bytes(b"weights")
    return source


def test_prepare_downloads_only_pinned_zero_model_and_compatible_backbone(tmp_path, monkeypatch):
    import huggingface_hub

    original = load_config(recipe.RECIPE)
    policy = original["policy"]
    snapshot = tmp_path / "snapshots" / policy["revision"]
    root = snapshot / policy["root"]
    (root / "checkpoints").mkdir(parents=True)
    (root / policy["checkpoint"]).write_bytes(b"test weights")
    write_yaml(root / "config.full.yaml", {"base_model": {"repo_id": recipe.BACKBONE["repo_id"]}})
    write_yaml(root / "config.yaml", {})
    write_json(root / "dataset_statistics.json", {})
    backbone = tmp_path / "backbone" / recipe.BACKBONE["revision"]
    backbone.mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(backbone if kwargs["repo_id"] == recipe.BACKBONE["repo_id"] else snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    output = tmp_path / "experiment"
    args = ["prepare", "--output-dir", str(output)]
    recipe.main(args)
    recipe.main(args)  # Resuming preparation neither downloads nor changes settings.
    assert len(calls) == 2
    assert calls[0]["revision"] == policy["revision"]
    assert calls[0]["allow_patterns"] == [f"{policy['root']}/{name}" for name in
                                         (policy["checkpoint"], "config.full.yaml", "config.yaml", "dataset_statistics.json")]
    assert calls[1] == recipe.BACKBONE
    config = load_config(output / "measurement.yaml")
    assert config["policy"]["backbone_path"] == str(backbone)
    assert config["provenance"]["model_artifact"]["revision"] == policy["revision"]
    assert config["env"]["env_fps"] == config["env"]["obs_fps"] == 30
    assert config["evaluation"]["max_env_steps"] == 3600
    assert config["executor"]["inference_devices"] == ["cuda:0"]
    assert config["profiling"]["calibration"] is None
    assert read_json(output / "protocol.json")["held_out_session_ids"] == [3, 4]
    with pytest.raises(ValueError, match="Protocol changed"):
        recipe.main(args + ["--episodes", "11"])
    with pytest.raises(ValueError, match="reserve"):
        recipe.main(["prepare", "--output-dir", str(tmp_path / "invalid"),
                     "--fit-session-ids", "0", "1", "2", "3", "4"])


def test_export_shared_fit_split_companions_and_raw_evidence(raw_run, tmp_path, monkeypatch):
    output = tmp_path / "comparison"
    monkeypatch.setattr(recipe, "download_model", lambda *args: pytest.fail("reused recordings need no model"))
    monkeypatch.setattr(recipe, "run_profile", lambda *args, **kw: pytest.fail("no remeasurement"))
    args = ["prepare", "--output-dir", str(output), "--existing-run-dir", str(raw_run),
            "--hmm-max-iterations", "20"]
    recipe.main(args)
    recipe.measure(output)
    real_fit = recipe.fit_profile
    fail_once = True

    def fit(*args, **kwargs):
        nonlocal fail_once
        if kwargs["method"] == "hmm" and fail_once:
            fail_once = False
            raise RuntimeError("interrupted fit")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(recipe, "fit_profile", fit)
    with pytest.raises(RuntimeError, match="interrupted"):
        recipe.export_profiles(output)
    assert not (output / "bundle").exists()
    assert (output / "profiles/temporal/DONE").exists()
    recipe.export_profiles(output)
    recipe.main(args)
    recipe.measure(output)
    recipe.export_profiles(output)
    bundle = output / "bundle"
    assert (bundle / "DONE").exists()
    for method in ("temporal", "hmm", "block"):
        manifest = read_json(bundle / "profiles" / method / "profile.json")
        assert manifest["provenance"]["fit_options"]["session_ids"] == [0, 1, 2]
        assert manifest["provenance"]["model_artifact"]["revision"] == "measured-revision"
        assert manifest["n_admitted_observations"] == 300
        for key in ("sequence_model_path", "burst_model_path", "distribution_path", "visualization_path"):
            if key in manifest:
                assert (bundle / "profiles" / method / manifest[key]).is_file()
    # Held-out observations are orders of magnitude larger and cannot influence fits.
    block = read_json(bundle / "profiles/block/latency_block_model.json")
    assert block["observed_max_latency_ms"] < 200
    assert (bundle / "measurement/sessions/4/observation_attempts.jsonl").exists()
    assert (bundle / "measurement/sessions/0/warmup_summary.json").exists()
    assert not (bundle / "measurement/profile").exists()
    assert not list(bundle.rglob("*.pt"))


def test_interrupted_measurement_resumes_existing_run_without_new_collection(tmp_path, monkeypatch):
    output = tmp_path / "experiment"
    config = resolve_profile_config(load_config(recipe.RECIPE))
    config["logging"]["output_dir"] = str(output / "raw")
    write_json(output / "protocol.json", {"measurement_config": config, "existing_run_dir": None,
                                          "fit_session_ids": [0, 1, 2]})
    write_yaml(output / "measurement.yaml", config)
    source = output / "raw/runs/openvla/gpu/flappy/instance/run_1"
    calls = []

    def run(config, *, resume_run_id, fit_session_ids):
        assert fit_session_ids == [0, 1, 2]
        calls.append(resume_run_id)
        if resume_run_id is None:
            write_json(source / "run_contract.json", {"config": config})
            raise RuntimeError("interrupted measurement")
        recorded_run(source, config)
        return {"RUN_DIR": str(source), "RUN_ID": source.name}

    monkeypatch.setattr(recipe, "run_profile", run)
    with pytest.raises(RuntimeError, match="interrupted"):
        recipe.measure(output)
    recipe.measure(output)
    recipe.measure(output)
    assert calls == [None, "run_1"]
    assert read_json(output / "measurement.json")["RUN_DIR"] == str(source)


@pytest.mark.parametrize('minimum_seconds', [0, 1200])
def test_irregular_profiling_freezes_live_load_and_refuses_uncontended_reuse(raw_run, tmp_path, monkeypatch, minimum_seconds):
    monkeypatch.setattr(recipe, "download_model", lambda config, output: config)
    output = tmp_path / "irregular"
    recipe.main(["prepare", "--output-dir", str(output), "--memory-condition", "irregular",
                 "--methods", "temporal", "ar1", "arma", "--compare",
                 "--min-measurement-seconds", str(minimum_seconds)])
    protocol = read_json(output / "protocol.json")
    assert protocol["measurement_config"]["profiling"]["memory_mock"]["condition"] == "irregular"
    assert protocol["measurement_config"]["evaluation"]["episodes"] == 10
    memory = protocol["measurement_config"]["profiling"]["memory_mock"]
    assert memory.get('min_measurement_seconds', 0) == minimum_seconds
    assert protocol["comparison"]["arma_order"] == [1, 0, 1]
    assert protocol["fit_session_ids"] == [0, 1, 2]
    with pytest.raises(ValueError, match="not collected under"):
        recipe.main(["prepare", "--output-dir", str(tmp_path / "bad-reuse"),
                     "--existing-run-dir", str(raw_run), "--memory-condition", "irregular"])


@pytest.mark.parametrize('arguments,message', [
    (['--min-measurement-seconds', '-1'], 'Minimum measurement seconds'),
    (['--min-measurement-seconds', 'nan'], 'Minimum measurement seconds'),
    (['--min-measurement-seconds', '7200', '--memory-condition', 'irregular'], 'run timeout'),
    (['--min-measurement-seconds', '1200'], 'requires --memory-condition'),
    (['--min-measurement-seconds', '1200', '--memory-condition', 'irregular',
      '--coverage-seconds', '100'], 'schedule coverage'),
])
def test_invalid_duration_budget_fails_before_model_download(tmp_path, monkeypatch, arguments, message):
    monkeypatch.setattr(recipe, 'download_model', lambda *args: pytest.fail('invalid budget must not download'))
    with pytest.raises(ValueError, match=message):
        recipe.main(['prepare', '--output-dir', str(tmp_path / 'invalid'), *arguments])


def test_ar_profiles_and_comparison_use_only_the_shared_fitting_partition(raw_run, tmp_path):
    from latency_bench.core.clock import EnvClock
    from latency_bench.latency.samplers import build_sampler, temporal_profile_sampler_for_reset

    output = tmp_path / "ar-comparison"
    recipe.main(["prepare", "--output-dir", str(output), "--existing-run-dir", str(raw_run),
                 "--methods", "temporal", "ar1", "arma", "--compare"])
    recipe.measure(output)
    recipe.export_profiles(output)
    summary = read_json(output / "comparison/summary.json")
    assert summary["held_out_session_ids"] == [3, 4]
    assert summary["cutoffs_from_fit_sessions_ms"]["0.95"][0] < 200
    for method in ("ar1", "arma"):
        path = output / "profiles" / method / "profile.json"
        profile = read_json(path)
        assert profile["provenance"]["fit_options"]["session_ids"] == [0, 1, 2]
        assert profile["n_admitted_observations"] == 300
        model = read_json(path.parent / profile["sequence_model_path"])
        assert model["observed_max_latency_ms"] < 200
        sampler = build_sampler({"method": method, "profile_path": str(path), "seed": 42},
                                clock=EnvClock(env_fps=30, obs_fps=30))
        reset = temporal_profile_sampler_for_reset(sampler, 42)
        assert reset.latency_type == sampler.latency_type == method
        assert sampler.sample({"worker_slot": 0, "episode_id": 0}) == reset.sample({"worker_slot": 0, "episode_id": 0})
    assert (output / "bundle/comparison/report.html").is_file()
    assert (output / "bundle/comparison/sequences.npz").is_file()
    recipe.export_profiles(output)  # Complete fits and comparison are reused.


def test_reusing_calibrated_sessions_records_the_actual_clocks(raw_run, tmp_path):
    contract = read_json(raw_run / "run_contract.json")
    contract["config"]["profiling"]["calibration"] = {"target_latency_frames": 2}
    write_json(raw_run / "run_contract.json", contract)
    write_json(raw_run / "calibration.json", {"selected_env_fps": 12, "selected_obs_fps": 12})
    output = tmp_path / "refit"
    recipe.main(["prepare", "--output-dir", str(output), "--existing-run-dir", str(raw_run)])
    actual = read_json(output / "protocol.json")["measurement_config"]
    assert actual["env"]["env_fps"] == actual["env"]["obs_fps"] == 12
    assert read_json(raw_run / "run_contract.json")["config"]["env"]["env_fps"] == 30


@pytest.mark.parametrize("create_repo", [False, True])
def test_upload_publishes_bundle_once_and_saves_exact_revision(tmp_path, monkeypatch, create_repo):
    import huggingface_hub

    output = tmp_path / "experiment"
    bundle = output / "bundle"
    write_json(bundle / "protocol.json", {"methods": ["hmm"]})
    write_json(bundle / "profiles/hmm/profile.json", {"gpu_class": "1x-gpu", "source_run_id": "run_1"})
    (bundle / "DONE").touch()
    calls = []
    creations = []
    api = SimpleNamespace(
        create_repo=lambda *a, **kw: creations.append((a, kw)),
        repo_info=lambda *a, **kw: SimpleNamespace(sha="parent-revision"),
        get_paths_info=lambda *a, **kw: [],
        upload_folder=lambda **kw: calls.append(kw) or SimpleNamespace(oid="uploaded-revision", commit_url="commit-url"),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    receipt = recipe.upload(output, "test/profiles", "experiments/timing", create_repo=create_repo)
    assert creations == ([(('test/profiles',), {'repo_type': 'dataset', 'private': True, 'exist_ok': True})]
                         if create_repo else [])
    assert len(calls) == 1
    assert calls[0]["folder_path"] == str(bundle)
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["parent_commit"] == "parent-revision"
    assert calls[0]["path_in_repo"] == "experiments/timing/flappy/openvla/1x-gpu/run_1/experiment"
    assert receipt["revision"] == "uploaded-revision"
    assert read_json(output / "upload_receipt.json") == receipt
    assert recipe.upload(output, "test/profiles", "experiments/timing", create_repo=create_repo) == receipt
    assert len(creations) == int(create_repo)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="different destination"):
        recipe.upload(output, "test/another-repo", "experiments/timing")


def test_upload_does_not_overwrite_remote_experiment(tmp_path, monkeypatch):
    import huggingface_hub

    output = tmp_path / "experiment"
    bundle = output / "bundle"
    write_json(bundle / "protocol.json", {"methods": ["block"]})
    write_json(bundle / "profiles/block/profile.json", {"gpu_class": "gpu", "source_run_id": "run_1"})
    (bundle / "DONE").touch()
    api = SimpleNamespace(
        repo_info=lambda *a, **kw: SimpleNamespace(sha="head"),
        get_paths_info=lambda *a, **kw: ["existing folder"],
        upload_folder=lambda **kw: pytest.fail("must not overwrite a published experiment"),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    with pytest.raises(FileExistsError, match="already exists"):
        recipe.upload(output, "test/profiles", "experiments/timing")
    assert not (output / "upload_receipt.json").exists()
