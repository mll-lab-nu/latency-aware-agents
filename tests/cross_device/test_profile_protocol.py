import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from latency_bench.core.config import load_config
from latency_bench.cross_device import runner
from latency_bench.cross_device.artifacts import (
    profile_dir,
    profile_ref,
    run_dir,
    run_ref,
    simulation_seed_dir,
)
from latency_bench.cross_device.profile_resolver import (
    profile_refs_from_repo_files,
    resolve_remote_profile,
)
from latency_bench.cross_device.protocol import main
from latency_bench.cross_device.runner import _profile_lock, _replace_current
from latency_bench.cross_device.runtime_configs import (
    measure_config,
    simulation_config,
)
from latency_bench.cross_device.spec import load_profile_spec
from latency_bench.cross_device.sync import download_profile, publish_run_and_profile


def test_profile_coordinates_define_every_artifact_path(tmp_path):
    profile = profile_ref("openvla", "2x-rtx4090", "flappy", "instance_abc")
    run = run_ref(profile, "run_1")

    assert profile_dir(tmp_path, profile) == tmp_path / "profiles" / profile
    assert run_dir(tmp_path, run) == tmp_path / "runs" / run
    assert simulation_seed_dir(tmp_path, profile, "run_1", "temporal", 7) == (
        tmp_path / "simulations" / profile / "run_1" / "temporal" / "7"
    )
    assert profile_refs_from_repo_files(
        [
            f"profiles/{profile.as_posix()}/DONE",
            f"profiles/{profile.as_posix()}/profile.json",
            "profiles/too/short/DONE",
        ]
    ) == [profile]


def test_registry_has_one_openvla_flappy_preset():
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")

    assert spec["session_count"] == 5
    assert spec["eval_episodes"] == 10
    assert spec["calibrate_env_fps"] is False
    assert spec["artifact_root"] == "results/profiling"


@pytest.mark.parametrize("task", ["atlantis", "asterix"])
@pytest.mark.parametrize("model", ["qwenoft", "qwengr00t", "qwenpi_v3"])
def test_atari_profile_keeps_cpu_simulation_and_history_manifest(tmp_path, task, model):
    spec = load_profile_spec("configs/profiles.yaml", model, task)
    spec.update(checkpoint_path=str(tmp_path / "bundle/checkpoints/model.pt"),
                model_config_path=str(tmp_path / "bundle/config.yaml"))
    config = measure_config(spec, {"gpus": [{}], "gpu_class": "1x-rtx3090", "instance_id": "instance_test"},
                            tmp_path / "profile", {"selected_env_fps": 60, "selected_obs_fps": 15})
    assert spec["session_count"] == 5
    assert config["env"]["simulator"] == "cpu"
    assert config["env"]["action_history_decisions"] == 8
    assert config["executor"]["mode"] == "realtime"
    assert config["executor"]["inference_devices"] == ["cuda:0"]
    assert config["evaluation"]["eval_parallel_envs"] == 1
    assert config["evaluation"]["eval_max_steps"] == 7200
    assert config["policy"]["task_manifest_path"] == str(tmp_path / "bundle/manifest.json")
    assert config["policy"]["state_source"] == "transport"


def test_registry_uses_one_episode_count_for_profile_and_replay():
    registry = load_config("configs/profiles.yaml")
    for model in registry["models"].values():
        for preset in model["workloads"].values():
            assert "eval_episodes" in preset
            assert "n_episodes" not in preset
            assert "profile_episodes" not in preset


def test_registry_has_balance_qwengr00t_preset():
    spec = load_profile_spec(
        "configs/profiles.yaml", "qwengr00t", "g1_sonic_balance_simple"
    )

    assert spec["session_count"] == 5
    assert spec["eval_episodes"] == 50
    assert spec["evaluation_seed"] == 0
    assert spec["simulation_parallel_envs"] == 1
    assert spec["calibrate_env_fps"] is False
    assert spec["simulation_env"] == {}
    assert spec["model_artifact"]["checkpoint"] == "checkpoints/model.pt"
    assert "starvla-qwengr00t" in spec["model_artifact"]["hf_path"]
    assert spec["gpu_models"]["rpbzzz6"]["aliases"] == ["NVIDIA RPBZZZ6"]


def test_registry_has_deadly_corridor_profile_presets():
    for model_id in ("openvla", "wanoft", "pi05", "gr00t"):
        spec = load_profile_spec("configs/profiles.yaml", model_id, "deadly_corridor")
        assert spec["session_count"] == 5
        assert spec["eval_episodes"] == 10
        assert spec["calibrate_env_fps"] is False
        assert spec["simulation_parallel_envs"] == 1


def test_registry_has_openvla_defend_the_line_profile_preset():
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "defend_the_line")

    assert spec["session_count"] == 5
    assert spec["eval_episodes"] == 10
    assert spec["calibrate_env_fps"] is False
    assert spec["model_artifact"]["checkpoint"] == "checkpoints/steps_7000_model.safetensors"
    assert spec["model_artifact"]["local_root"].endswith(
        "openvla_bridge_defend_the_line_single_latency_clean_data_exp1"
    )


def test_deployment_collection_preserves_cuda_device_selector(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        runner,
        "collect_hardware_fingerprint",
        lambda: {
            "gpu": {"topology": None},
            "gpus": [
                {"uuid": "GPU-b", "name": "NVIDIA RPBZZZ6"},
                {"uuid": "GPU-a", "name": "NVIDIA RPBZZZ6"},
            ],
        },
    )

    deployment, fingerprint = runner._collect_deployment(
        {
            "gpu_models": {"rpbzzz6": {"aliases": ["NVIDIA RPBZZZ6"]}},
            "simulation_gpu": None,
        }
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert [(gpu["uuid"], gpu["cuda_index"]) for gpu in deployment["gpus"]] == [
        ("GPU-a", 1),
        ("GPU-b", 0),
    ]
    assert fingerprint["gpu_deployment"] == deployment


def test_gpu_simulator_is_excluded_from_profile_deployment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setattr(
        runner,
        "collect_hardware_fingerprint",
        lambda: {
            "gpu": {"topology": None},
            "gpus": [
                {"uuid": "GPU-sim", "name": "NVIDIA RPBZZZ6"},
                {"uuid": "GPU-a", "name": "NVIDIA RPBZZZ6"},
                {"uuid": "GPU-b", "name": "NVIDIA RPBZZZ6"},
            ],
        },
    )
    spec = {
        "gpu_models": {"rpbzzz6": {"aliases": ["NVIDIA RPBZZZ6"]}},
        "simulation_gpu": "cuda:0",
    }

    deployment, _ = runner._collect_deployment(spec)

    assert deployment["gpu_class"] == "2x-rpbzzz6"
    assert {gpu["uuid"] for gpu in deployment["gpus"]} == {"GPU-a", "GPU-b"}


def test_gpu_simulator_requires_a_separate_inference_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeError, match="separate inference GPU"):
        runner._inference_cuda_visible_devices({"simulation_gpu": "cuda:0"})


@pytest.mark.parametrize(
    "model,workload,env_fps,obs_fps",
    [
        ("qwenoft", "mikasa_intercept_grab_fast", 20, 20),
        ("qwenoft", "extreme_parkour_go2", 50, 10),
        ("qwenpi_v3", "extreme_parkour_go2", 50, 10),
        ("qwengr00t", "extreme_parkour_go2", 50, 10),
        ("qwenoft", "haic_pull_cart", 50, 50),
    ],
)
def test_measure_config_resolves_task_contract_for_model_bundle(
    tmp_path, model, workload, env_fps, obs_fps
):
    spec = load_profile_spec(
        "configs/profiles.yaml", model, workload
    )
    model_config_path = tmp_path / "bundle" / "config.yaml"
    spec["checkpoint_path"] = str(tmp_path / "bundle/checkpoints/model.pt")
    spec["model_config_path"] = str(model_config_path)
    spec["backbone_path"] = "/models/backbone/snapshots/recorded-commit"
    if workload == "extreme_parkour_go2":
        spec["runtime_checkpoint_path"] = "/models/parkour/model_14999.pt"
    elif workload == "haic_pull_cart":
        spec["runtime_checkpoint_path"] = "/models/haic/checkpoint_final.pt"

    config = measure_config(
        spec,
        {
            "gpu_class": "1x-rtx3090",
            "instance_id": "instance_abc",
            "gpus": [{}],
        },
        tmp_path,
        {"selected_env_fps": env_fps, "selected_obs_fps": obs_fps},
    )

    assert config["policy"]["model_config_path"] == str(model_config_path)
    assert config["policy"]["backbone_path"] == spec["backbone_path"]
    assert config["policy"]["task_contract_path"] == str(
        model_config_path.parent / "task_contract.json"
    )
    if workload in ("extreme_parkour_go2", "haic_pull_cart"):
        assert config["policy"]["worker_python_executable"] == (
            "/venv/starvla_rl_games_openvla/bin/python"
        )
        assert spec["session_count"] == 5
        assert config["evaluation"]["eval_episodes"] == 50
        assert config["evaluation"]["eval_parallel_envs"] == 1
        assert config["executor"]["inference_devices"] == ["cuda:0"]
        assert config["env"]["simulator_device"] == "cuda:0"
        assert config["env"]["runtime_checkpoint_path"] == spec["runtime_checkpoint_path"]
        assert config["env"]["env_fps"] == env_fps
        assert config["env"]["obs_fps"] == obs_fps
    if workload == "extreme_parkour_go2":
        run_id = {
            "qwenoft": "20260912T072315-nocache32v2",
            "qwenpi_v3": "20260912T175510",
            "qwengr00t": "20260912T180303",
        }[model]
        assert spec["model_artifact"]["hf_path"] == (
            f"zero-latency/extreme-parkour-go2/vla/starvla-{model}-h1/{run_id}"
        )
        assert config["evaluation"]["eval_max_steps"] == 1500
        assert len(config["env"]["noop_action"]) == 34
    elif workload == "haic_pull_cart":
        assert spec["model_artifact"]["hf_path"] == (
            "zero-latency/haic-pull-cart/vla/starvla-qwenoft-h1/"
            "haic-qwenoft-50hz-fresh-20260913T035850Z"
        )
        assert config["evaluation"]["eval_max_steps"] == 1000
        assert config["policy"]["state_info_key"] == "haic_state"


def test_measure_config_strips_eval_suites_for_realtime_demon_attack(tmp_path):
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "demon_attack")
    spec["checkpoint_path"] = str(tmp_path / "bundle/checkpoints/model.safetensors")
    spec["model_config_path"] = str(tmp_path / "bundle/config.full.yaml")

    config = measure_config(
        spec,
        {
            "gpu_class": "1x-rtx3090",
            "instance_id": "instance_abc",
            "gpus": [{}],
        },
        tmp_path,
        {"selected_env_fps": 60, "selected_obs_fps": 15},
    )

    assert config["executor"]["mode"] == "realtime"
    assert config["latency"]["method"] == "measured"
    assert config["evaluation"]["eval_latency_values"] is None
    assert "eval_suites" not in config["evaluation"]


def test_fixed_workload_cadence_skips_policy_calibration(tmp_path, monkeypatch):
    spec = load_profile_spec(
        "configs/profiles.yaml", "qwengr00t", "g1_sonic_balance_simple"
    )
    spec["checkpoint_path"] = "/models/balance/checkpoint-15000"
    monkeypatch.setattr(
        runner,
        "calibrate_env_fps_from_saturated_latency",
        lambda **_kwargs: pytest.fail("fixed cadence must skip policy calibration"),
    )

    calibration = runner._calibrate(
        spec,
        {
            "gpu_class": "2x-rpbzzz6",
            "instance_id": "instance_abc",
            "gpus": [{}, {}],
        },
        tmp_path,
    )

    assert calibration == {
        "selected_env_fps": 50,
        "selected_obs_fps": 50,
        "calibration_source": "fixed_workload_cadence",
    }
    assert json.loads((tmp_path / "calibration.json").read_text()) == calibration
    assert (tmp_path / "calibration.DONE").exists()
    assert not (tmp_path / "calibration.generated.yaml").exists()


def test_resume_run_id_cannot_escape_artifact_root():
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--model",
                "openvla",
                "--workload",
                "flappy",
                "--resume",
                "../run",
            ]
        )


def test_current_profile_replacement_keeps_only_new_bundle(tmp_path):
    current = tmp_path / "profiles" / "current"
    source = tmp_path / "candidate"
    current.mkdir(parents=True)
    source.mkdir()
    (current / "profile.json").write_text("old", encoding="utf-8")
    (source / "profile.json").write_text("new", encoding="utf-8")

    _replace_current(source, current)

    assert (current / "profile.json").read_text(encoding="utf-8") == "new"
    assert not current.with_name(".current.old").exists()


def test_profile_lock_serializes_current_writers(tmp_path):
    current = tmp_path / "profiles" / "current"
    entered = Event()

    def wait_for_lock():
        with _profile_lock(current):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with _profile_lock(current):
            future = executor.submit(wait_for_lock)
            assert not entered.wait(0.05)
        future.result()

    assert entered.is_set()


def test_new_runs_are_retained_while_current_moves_forward(tmp_path, monkeypatch):
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "experiment": {},
                "executor": {},
                "env": {"env_fps": 25, "obs_fps": 25},
                "latency": {},
                "scheduler": {},
                "policy": {
                    "type": "starvla",
                    "checkpoint_path": "/models/openvla",
                },
                "logging": {},
                "evaluation": {},
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "artifact_root": str(tmp_path / "artifacts"),
        "model_id": "openvla",
        "workload_id": "flappy",
        "session_count": 2,
        "eval_episodes": 1,
        "evaluation_seed": 0,
        "base_config": str(base_config),
        "checkpoint_path": "/models/openvla",
        "checkpoint_kind": "best",
        "gpu_models": {"rtx4090": {"aliases": ["NVIDIA RTX 4090"]}},
        "simulation_gpu": None,
        "calibrate_env_fps": False,
        "simulation_env": {},
    }
    deployment = {
        "gpu_class": "2x-rtx4090",
        "instance_id": "instance_abc",
        "gpus": [{}, {}],
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    run_ids = iter(["run_1", "run_2"])
    monkeypatch.setattr(
        runner,
        "_collect_deployment",
        lambda _spec: (deployment, {"private": "hardware"}),
    )
    monkeypatch.setattr(
        runner,
        "_calibrate",
        lambda _spec, _deployment, _source: {
            "selected_env_fps": 25,
            "selected_obs_fps": 25,
        },
    )
    monkeypatch.setattr(runner, "_measure_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_new_run_id", lambda: next(run_ids))
    monkeypatch.setattr(
        runner,
        "public_hardware_projection",
        lambda *args, **kwargs: {"gpu_class": "2x-rtx4090"},
    )

    def build_bundle(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir()
        (output / "profile.json").write_text(
            json.dumps({"source_run_id": kwargs["source_run_id"]}),
            encoding="utf-8",
        )
        (output / "latency_distribution.json").write_text("{}", encoding="utf-8")
        (output / "latency_burst_model.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(runner, "build_profile_from_traces", build_bundle)

    first = runner.run_profile(spec, resume_run_id=None, publish=False)
    second = runner.run_profile(spec, resume_run_id=None, publish=False)

    coordinate = Path("openvla/2x-rtx4090/flappy/instance_abc")
    runs = Path(spec["artifact_root"]) / "runs" / coordinate
    current = Path(spec["artifact_root"]) / "profiles" / coordinate
    assert first["RUN_ID"] == "run_1"
    assert second["RUN_ID"] == "run_2"
    assert (runs / "run_1" / "DONE").exists()
    assert (runs / "run_2" / "DONE").exists()
    assert json.loads((current / "profile.json").read_text())["source_run_id"] == "run_2"
    contract = json.loads((runs / "run_2" / "run_contract.json").read_text())
    assert contract["identity"]["checkpoint_path"] == "/models/openvla"
    assert contract["calibration_config"]["logging"]["output_dir"] == (
        "__latency_bench_contract_output__/calibration_probe"
    )
    assert contract["measurement_config"]["logging"]["output_dir"] == (
        "__latency_bench_contract_output__"
    )


def test_resume_requires_matching_run_contract(tmp_path):
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")
    spec["checkpoint_path"] = "/models/openvla"
    deployment = {
        "gpu_class": "2x-rtx4090",
        "instance_id": "instance_abc",
        "gpus": [{}, {}],
    }
    calibration = {"selected_env_fps": 16, "selected_obs_fps": 16}
    source_dir = tmp_path / "run"
    source_dir.mkdir()
    (source_dir / "calibration.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )
    (source_dir / "calibration.DONE").touch()
    (source_dir / "run_contract.json").write_text(
        json.dumps(runner._run_contract(spec, deployment, calibration)),
        encoding="utf-8",
    )

    runner._assert_run_contract(spec, deployment, source_dir)

    spec["checkpoint_path"] = "/models/another-openvla"
    with pytest.raises(ValueError, match="run contract does not match"):
        runner._assert_run_contract(spec, deployment, source_dir)


def test_legacy_run_without_contract_cannot_resume(tmp_path):
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")
    deployment = {
        "gpu_class": "2x-rtx4090",
        "instance_id": "instance_abc",
        "gpus": [{}, {}],
    }
    with pytest.raises(FileNotFoundError, match="run_contract.json"):
        runner._assert_run_contract(spec, deployment, tmp_path)


def test_simulation_uses_profile_slots_and_selected_artifact(tmp_path):
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")
    spec["checkpoint_path"] = "/models/openvla"
    profile = {
        "model_id": "openvla",
        "gpu_class": "2x-rtx4090",
        "workload_id": "flappy",
        "instance_id": "instance_abc",
        "source_run_id": "run_1",
        "env_fps": 20,
        "per_slot_summary": {"0": {}, "1": {}},
        "summary": {"mean_ms": 50.0, "std_ms": 4.0},
    }

    config = simulation_config(
        spec,
        output_dir=tmp_path,
        profile_path=tmp_path / "profile.json",
        profile_ref=Path("openvla/2x-rtx4090/flappy/instance_abc"),
        profile=profile,
        method="temporal",
        latency_seed=7,
    )

    assert config["executor"]["simulated_worker_capacity"] == 2
    assert config["policy"]["checkpoint_path"] == spec["checkpoint_path"]
    assert config["latency"]["source_run_id"] == "run_1"


def test_publish_uses_one_commit_and_omits_private_hardware(tmp_path, monkeypatch):
    run = tmp_path / "run"
    profile = tmp_path / "profile"
    session = run / "sessions" / "0"
    session.mkdir(parents=True)
    (run / "profile").mkdir()
    profile.mkdir()
    (run / "hardware.json").write_text("private", encoding="utf-8")
    (run / "run_contract.json").write_text("private", encoding="utf-8")
    (run / "profile" / "profile.json").write_text("candidate", encoding="utf-8")
    (session / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    (profile / "profile.json").write_text(
        json.dumps({"visualization_path": "latency_profile.png"}),
        encoding="utf-8",
    )
    (profile / "latency_profile.png").write_bytes(b"png")
    for name in (
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (profile / name).write_text("public", encoding="utf-8")

    client = SimpleNamespace(
        calls=[],
        create_commit=lambda **kwargs: (
            client.calls.append(kwargs) or SimpleNamespace(oid="commit_1")
        ),
    )

    class Add:
        def __init__(self, **kwargs):
            self.path_in_repo = kwargs["path_in_repo"]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=Add, HfApi=lambda: client),
    )

    publish_run_and_profile(
        {"hf_repo": "org/profiles"},
        local_run_dir=run,
        local_profile_dir=profile,
        run_ref=Path("openvla/2x-rtx4090/flappy/instance_abc/run_1"),
        profile_ref=Path("openvla/2x-rtx4090/flappy/instance_abc"),
    )

    uploaded = {op.path_in_repo for op in client.calls[0]["operations"]}
    assert len(client.calls) == 1
    assert not any(path.endswith("/hardware.json") and path.startswith("runs/") for path in uploaded)
    assert not any(path.endswith("/run_contract.json") for path in uploaded)
    assert not any("/profile/" in path and path.startswith("runs/") for path in uploaded)
    assert "profiles/openvla/2x-rtx4090/flappy/instance_abc/hardware.json" in uploaded
    assert "profiles/openvla/2x-rtx4090/flappy/instance_abc/latency_profile.png" in uploaded
    profile_uploads = [
        operation.path_in_repo
        for operation in client.calls[0]["operations"]
        if operation.path_in_repo.startswith("profiles/")
    ]
    assert profile_uploads[-2:] == [
        "profiles/openvla/2x-rtx4090/flappy/instance_abc/latency_profile.png",
        "profiles/openvla/2x-rtx4090/flappy/instance_abc/DONE",
    ]


def test_download_fetches_current_profile(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    profile = tmp_path / "profiles" / ref
    profile.mkdir(parents=True)
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (profile / name).touch()
    (profile / "profile.json").write_text("{}", encoding="utf-8")
    calls = {}
    def snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download),
    )

    local = download_profile(
        {"hf_repo": "org/profiles"},
        ref,
        local_dir=tmp_path / "downloads",
    )

    assert local == tmp_path / "downloads"
    assert "revision" not in calls


def test_download_requires_and_repairs_declared_visualization(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache"
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    (source / "profile.json").write_text(
        json.dumps({"visualization_path": "latency_profile.png"}),
        encoding="utf-8",
    )
    for name in (
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (source / name).write_text(name, encoding="utf-8")

    local_dir = tmp_path / "downloads"
    shutil.copytree(source, local_dir)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: str(cache)),
    )

    with pytest.raises(FileNotFoundError):
        download_profile(
            {"hf_repo": "org/profiles"},
            ref,
            local_dir=local_dir,
        )
    (source / "latency_profile.png").write_bytes(b"png")

    local = download_profile(
        {"hf_repo": "org/profiles"},
        ref,
        local_dir=local_dir,
    )

    assert local == local_dir
    assert (local / "latency_profile.png").read_bytes() == b"png"


def test_download_materializes_hf_snapshot_symlinks(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache"
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name in ("profile.json", "latency_distribution.json", "latency_burst_model.json", "hardware.json", "DONE"):
        target = blobs / name
        target.write_text("{}" if name == "profile.json" else name, encoding="utf-8")
        (source / name).symlink_to(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: str(cache),
        ),
    )

    local = download_profile(
        {"hf_repo": "org/profiles"},
        ref,
        local_dir=tmp_path / "artifact_root" / "downloads" / "profiles" / ref,
    )

    assert local == tmp_path / "artifact_root" / "downloads" / "profiles" / ref
    assert all(not (local / name).is_symlink() for name in ("profile.json", "DONE"))
    assert (local / "profile.json").read_text(encoding="utf-8") == "{}"


def test_concurrent_download_materializes_a_profile_once(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    source = tmp_path / "cache" / "profiles" / ref
    source.mkdir(parents=True)
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (source / name).write_text(
            "{}" if name == "profile.json" else name,
            encoding="utf-8",
        )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: str(tmp_path / "cache"),
        ),
    )
    copy_started = Event()
    release_copy = Event()
    copy_count = 0
    real_copyfile = shutil.copyfile

    def slow_copyfile(source_path, destination_path):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 1:
            copy_started.set()
            release_copy.wait()
        return real_copyfile(source_path, destination_path)

    monkeypatch.setattr("latency_bench.cross_device.sync.shutil.copyfile", slow_copyfile)
    destination = tmp_path / "downloads"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            download_profile,
            {"hf_repo": "org/profiles"},
            ref,
            local_dir=destination,
        )
        assert copy_started.wait(1.0)
        second = executor.submit(
            download_profile,
            {"hf_repo": "org/profiles"},
            ref,
            local_dir=destination,
        )
        try:
            assert not second.done()
        finally:
            release_copy.set()
        assert first.result() == second.result()

    assert copy_count == 5


def test_remote_profile_coordinates_must_match_request(tmp_path, monkeypatch):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "model_id": "openvla",
                "gpu_class": "1x-rtx4090",
                "workload_id": "flappy",
                "instance_id": "instance_other",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "latency_bench.cross_device.profile_resolver.download_profile",
        lambda *args, **kwargs: tmp_path,
    )

    with pytest.raises(ValueError, match="coordinates"):
        resolve_remote_profile(
            {
                "artifact_root": str(tmp_path / "artifacts"),
                "model_id": "openvla",
                "workload_id": "flappy",
            },
            gpu_class="2x-rtx4090",
            instance_id="instance_requested",
        )
