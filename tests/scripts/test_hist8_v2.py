from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("omegaconf")

from scripts.tasks.decision_history.train_vla_hist8 import training_config
from scripts.tasks.decision_history.eval_vla_hist8 import package_bundle, evaluation_config, EVAL_PROTOCOL
from latency_bench.eval.reporting import _corrected_eval_scalar_payload
from scripts.tasks.decision_history.run_hist8_v2 import available_gpus, execute, run_queue
from scripts.tasks.decision_history.make_hist8_v2_queue import add_zero_controls, build_manifest, H200_FIRST, H200_SECOND, H100, TASKS, MODELS
from scripts.tasks.decision_history.monitor_hist8_v2 import pending_events


def arguments(tmp_path, task="demon_attack", model="qwenoft"):
    return Namespace(task=task, model=model, dataset_dir=tmp_path / "data" / "lerobot",
                     base_model_dir=tmp_path / "backbone", output_root=tmp_path / "train",
                     run_id=f"{task}-{model}-v2", vlm_lr=3e-6, stop_after_steps=None,
                     preflight=False, resume=False, training_condition="profile")


@pytest.mark.parametrize("task,state_dim,action_dim", [
    ("flappy", 24, 2), ("demon_attack", 88, 6), ("deadly_corridor", 120, 7),
])
@pytest.mark.parametrize("model", ["qwenoft", "qwengr00t", "qwenpi_v3"])
@pytest.mark.parametrize("condition", ["profile", "zero"])
def test_training_configs_keep_the_task_contract(tmp_path, task, state_dim, action_dim, model, condition):
    args = arguments(tmp_path, task, model)
    args.training_condition = condition
    cfg = training_config(args)
    assert cfg.run_id == args.run_id
    assert cfg.training_latency_condition == f"{condition}-latency"
    assert cfg.framework.action_model.state_dim == state_dim
    assert cfg.framework.action_model.action_dim == action_dim
    assert cfg.framework.action_model.action_horizon == 1
    assert cfg.datasets.vla_data.include_state
    assert cfg.datasets.vla_data.per_device_batch_size == 64
    assert cfg.datasets.vla_data.num_workers == 16
    assert cfg.datasets.vla_data.eval_num_workers == 16
    assert cfg.datasets.vla_data.prefetch_factor == 4
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.learning_rate.qwen_vl_interface == 3e-6
    assert cfg.trainer.learning_rate.action_model == 1e-4
    assert cfg.trainer.learning_rate.base == 3e-5
    assert cfg.trainer.scheduler_specific_kwargs.min_lr_rate == pytest.approx(1 / 30)
    assert cfg.trainer.max_train_steps == 5000
    if model == "qwenoft":
        assert cfg.framework.action_model.loss_type == (
            "multibinary_bce" if task == "deadly_corridor" else "discrete_ce")


def test_pause_and_resume_keep_the_full_schedule_and_run_identity(tmp_path):
    args = arguments(tmp_path, model="qwenpi_v3")
    args.vlm_lr = 3e-5
    args.stop_after_steps = 1000
    paused = training_config(args)
    assert paused.trainer.stop_after_steps == 1000
    assert paused.trainer.max_train_steps == 5000
    checkpoints = args.output_root / args.run_id / "checkpoints"
    state = checkpoints / "steps_1000_state"
    state.mkdir(parents=True)
    (checkpoints / "steps_1000_pytorch_model.pt").write_bytes(b"fixture")
    args.resume = True
    args.stop_after_steps = None
    resumed = training_config(args)
    assert resumed.run_id == paused.run_id
    assert resumed.trainer.resume_step == 1000
    assert resumed.trainer.pretrained_checkpoint == str(state)
    assert resumed.trainer.learning_rate.qwen_vl_interface == 3e-5
    assert resumed.trainer.max_train_steps == 5000
    assert "stop_after_steps" not in resumed.trainer


def test_partial_checkpoint_bundle_and_eval_preserve_actual_step(tmp_path):
    run = tmp_path / "old-run"
    checkpoint = run / "checkpoints" / "steps_2877_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    config = {
        "run_id": "old-demon", "version_id": "0.21",
        "framework": {"name": "QwenOFT", "qwenvl": {"base_vlm": "/old/base"}},
        "datasets": {"vla_data": {"include_state": True, "obs_image_size": None,
                                    "image_mode": "single", "num_obs_frames": 1}},
    }
    (run / "config.full.yaml").write_text(yaml.safe_dump(config))
    (run / "dataset_statistics.json").write_text("{}")
    dataset = tmp_path / "data" / "lerobot"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps({"robot_type": "rl_games_demon_attack"}))
    (dataset.parent / "quality_report.json").write_text("{}")
    (dataset.parent / "raw").mkdir()
    (dataset.parent / "raw/metadata.json").write_text("{}")
    collection = dataset.parent / "provenance.json"
    collection.write_text("{}")
    bundle = tmp_path / "bundle"
    package_bundle(run, dataset, collection, bundle, step=2877, wandb_run_id="old-id")
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["training_step"] == 2877
    assert provenance["wandb_run_id"] == "old-id"
    assert provenance["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert provenance["eval_protocol_version"] == EVAL_PROTOCOL
    cfg = evaluation_config("demon_attack", "qwenoft", bundle, tmp_path / "profile.json",
                            tmp_path / "eval", tmp_path / "backbone", Path("/python"),
                            device="cuda:0", episodes=100)
    assert "step2877" in cfg["experiment"]["name"]
    assert cfg["policy"]["state_source"] == "transport"
    assert "state_normalization" not in cfg["policy"]
    assert cfg["evaluation"]["eval_episodes"] == 100
    assert cfg["evaluation"]["eval_parallel_envs"] == 32
    assert cfg["evaluation"]["eval_max_steps"] == 7200
    zero = evaluation_config("flappy", "qwenoft", bundle, tmp_path / "profile.json",
                             tmp_path / "eval-zero", tmp_path / "backbone", Path("/python"),
                             device="cuda:0", episodes=100, env_condition="zero",
                             inference_devices=["cuda:0", "cuda:1"])
    assert zero["latency"] == {"method": "zero"}
    assert zero["evaluation"]["eval_parallel_envs"] == 32
    assert zero["evaluation"]["eval_max_steps"] == 3600
    assert zero["executor"]["inference_batch_size"] == 16
    assert zero["executor"]["inference_devices"] == ["cuda:0", "cuda:1"]


def test_history01_metric_namespace_preserves_checkpoint_step():
    payload = _corrected_eval_scalar_payload({}, [{
        "checkpoint_global_step": 2877, "suite_name": "fixed_2", "latency_type": "fixed_sweep",
        "latency": 2, "mean_return": 1., "std_return": 0., "mean_length": 10., "returns": [1.],
    }], prefix="eval_history01_v2")
    assert payload["global_step"] == 2877
    assert payload["eval_history01_v2/fixed/latency_2/mean_return"] == 1.


def test_queue_respects_external_gpu_processes_and_its_reservations(monkeypatch):
    outputs = iter(["0, uuid0, 4\n1, uuid1, 0\n2, uuid2, 800\n3, uuid3, 0\n", "uuid2\n"])
    monkeypatch.setattr("scripts.tasks.decision_history.run_hist8_v2.subprocess.check_output",
                        lambda *args, **kwargs: next(outputs))
    assert available_gpus({1}) == [0, 3]


@pytest.mark.parametrize("kind", ["train", "resume_selected"])
def test_resume_command_keeps_selected_run_and_learning_rate(tmp_path, monkeypatch, kind):
    job = {"python": "/python", "task": "demon_attack", "model": "qwenpi_v3",
           "dataset_dir": "/dataset", "backbone": "/backbone", "train_root": str(tmp_path),
           "run_id": "pi-selected-low", "vlm_lr": 3e-6, "port": 29801,
           "kind": "train", "resume": True, "stop_after_steps": None, "training_condition": "profile"}
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return Namespace(returncode=0)
    monkeypatch.setattr("scripts.tasks.decision_history.run_hist8_v2.subprocess.run", run)
    if kind == "resume_selected":
        selection = tmp_path / "selection.json"
        selection.write_text(json.dumps({"winner": job["run_id"]}))
        specification = {"kind": kind, "selection_path": str(selection), "candidates": {job["run_id"]: job}}
    else:
        specification = job
    assert execute(specification) == 0
    command, = commands
    assert command[command.index("--run-id") + 1] == "pi-selected-low"
    assert command[command.index("--vlm-lr") + 1] == "3e-06"
    assert "--resume" in command
    assert "--stop-after-steps" not in command


def test_queue_graph_covers_nine_final_models_and_nine_old_evaluations(tmp_path):
    catalog = []
    for task in TASKS:
        for model in MODELS:
            first_host = (task == "flappy" and model != "qwenpi_v3") or (task == "deadly_corridor" and model == "qwenoft")
            catalog.append({"task": task, "model": model, "host": H200_FIRST if first_host else H200_SECOND,
                            "old_root": "/old", "profile_path": f"/old/results/{task}/{model}.json",
                            "dataset_dir": f"/old/data/{task}/{model}/lerobot", "collection_provenance": "/old/provenance.json",
                            "run_dir": f"/old/train/{task}/{model}", "step": 2877, "checkpoint_sha256": "sha",
                            "wandb_run_id": f"old-{task}-{model}"})
    manifests = [build_manifest(catalog, host=host, runtime_root=tmp_path / host,
                                python_root=Path("/envs"), old_root=Path("/old"), backbone=Path("/backbone"))
                 for host in (H200_FIRST, H200_SECOND, H100)]
    jobs = [job for manifest in manifests for job in manifest["jobs"]]
    assert len([job for job in jobs if job["id"].startswith("eval_v2__")]) == 9
    assert len([job for job in jobs if job["id"].startswith("eval_legacy__")]) == 9
    training = [job for job in jobs if job["kind"] == "train"]
    assert len(training) == 9
    assert {(job["task"], job["model"]) for job in training} == {
        (task, model) for task in TASKS for model in MODELS
    }
    for job in training:
        assert job["vlm_lr"] == 3e-6
        assert job["stop_after_steps"] is None
        assert job["gpus_required"] == 2
        assert job["depends_on"] == []
    for manifest in manifests:
        ids = {job["id"] for job in manifest["jobs"]}
        for job in manifest["jobs"]:
            assert set(job["depends_on"]) <= ids
    for manifest in manifests:
        add_zero_controls(manifest, catalog, python_root=Path("/envs"), old_root=Path("/old"),
                          backbone=Path("/backbone"), teacher_catalog=Path("/teachers.json"))
    jobs = [job for manifest in manifests for job in manifest["jobs"]]
    zero_train = [job for job in jobs if job["id"].startswith("train_zero__")]
    assert len(zero_train) == 9
    assert len([job for job in jobs if job["id"].startswith("collect_zero__")]) == 3
    assert len([job for job in jobs if job["id"].startswith("materialize_zero__")]) == 3
    assert len([job for job in jobs if job["id"].startswith("package_zero__")]) == 9
    assert len([job for job in jobs if job["id"].startswith("eval_zero_to_zero__")]) == 9
    assert len([job for job in jobs if job["id"].startswith("eval_zero_to_profile__")]) == 9
    for job in zero_train:
        assert job["training_condition"] == "zero"
        assert job["vlm_lr"] == 3e-6
        assert job["gpus_required"] == 2
        assert Path(job["ready_path"]).name == "collection.complete.json"


def test_existing_selected_job_starts_fresh_from_approved_lr(tmp_path, monkeypatch):
    label, lr = "3e-6", 3e-6
    winner = f"demon-pi-{label}"
    selection = tmp_path / "demon-selection.json"
    selection.write_text(json.dumps({"status": "selected", "winner": winner}))
    candidate = {"python": "/python", "task": "flappy", "model": "qwenpi_v3",
                 "dataset_dir": "/dataset", "backbone": "/backbone", "train_root": str(tmp_path),
                 "run_id": f"flappy-pi-{label}", "vlm_lr": lr, "port": 29812,
                 "stop_after_steps": None, "resume": False, "training_condition": "profile"}
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return Namespace(returncode=0)
    monkeypatch.setattr("scripts.tasks.decision_history.run_hist8_v2.subprocess.run", run)
    assert execute({"kind": "train_selected", "selection_path": str(selection),
                    "candidates": {winner: candidate}}) == 0
    command, = commands
    assert command[command.index("--run-id") + 1] == candidate["run_id"]
    assert command[command.index("--vlm-lr") + 1] == str(lr)
    assert "--resume" not in command
    assert "--stop-after-steps" not in command


def test_queue_keeps_free_pair_for_training_before_evaluation(tmp_path, monkeypatch):
    """Scheduling contract: a released pair stays together and ready training runs first."""
    manifest = tmp_path / "manifest.json"
    jobs = [{"id": name, "kind": name, "python": "/python", "gpus_required": 2,
             "priority": priority, "depends_on": []} for name, priority in (("eval", 1000), ("train", 100))]
    manifest.write_text(json.dumps({"jobs": jobs}))
    pair_released = False
    launches = []
    def available(reserved):
        return [gpu for gpu in ([0, 1] if pair_released else [0]) if gpu not in reserved]
    def pause(_):
        nonlocal pair_released
        if not pair_released:
            state = json.loads((tmp_path / "state.json").read_text())
            assert all(s["status"] == "pending" for s in state.values())
            pair_released = True
    def launch(command, **kwargs):
        job = json.loads(Path(command[command.index("--job") + 1]).read_text())
        launches.append((job["id"], kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        Path(command[command.index("--result") + 1]).write_text(json.dumps({
            "status": "completed", "returncode": 0, "finished": 1,
        }))
        return Namespace(pid=123, poll=lambda: 0)
    monkeypatch.setattr("scripts.tasks.decision_history.run_hist8_v2.available_gpus", available)
    monkeypatch.setattr("scripts.tasks.decision_history.run_hist8_v2.time.sleep", pause)
    monkeypatch.setattr("scripts.tasks.decision_history.run_hist8_v2.subprocess.Popen", launch)
    run_queue(manifest, [])
    assert launches == [("train", "0,1"), ("eval", "0,1")]


def test_monitor_wakes_once_for_completed_evaluation_and_waits_while_running():
    job_id = "eval_v2__demon_attack-qwenpi_v3"
    state = {"status": "running", "attempt": 1, "process_alive": True, "log_age_seconds": 5}
    snapshots = {"h200-2": {"controller_alive": True, "controller_pid": 1,
                            "jobs": {job_id: state}}}
    assert pending_events(snapshots, set()) == []
    state["status"] = "completed"
    event, = pending_events(snapshots, set())
    assert event["id"] == f"evaluation_completed/h200-2/{job_id}/1"
    assert pending_events(snapshots, {event["id"]}) == []


def test_monitor_reports_a_failed_training_attempt():
    job_id = "train__demon_attack-qwenpi_v3"
    snapshots = {"h200-2": {"controller_alive": True, "controller_pid": 1,
                            "jobs": {job_id: {"status": "failed", "attempt": 1}}}}
    event, = pending_events(snapshots, set())
    assert event["id"] == f"failed/h200-2/{job_id}/1"
