from argparse import Namespace
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

pytest.importorskip("omegaconf")

from scripts.tasks.decision_history.train_vla_hist8 import training_config
from scripts.tasks.decision_history.eval_vla_hist8 import package_bundle, evaluation_config, EVAL_PROTOCOL
from latency_bench.eval.reporting import _corrected_eval_scalar_payload
from scripts.tasks.atari_hist8.profile_teacher import teacher_configs
from scripts.tasks.decision_history.collect_vla_hist8 import audit_atari_history


def test_published_profile_records_model_revision_and_uploaded_evidence(tmp_path, monkeypatch):
    """Long-lived H8 publication contract owned by the history recipe suite."""
    from scripts.tasks.atari_hist8 import profile_published as module
    from latency_bench.core.config import load_config

    publication = {
        "task": "atlantis", "model": "qwenoft", "repo_id": "org/models",
        "path": "published/bundle", "revision": "model-revision",
    }
    publication_path = tmp_path / "publication.json"
    publication_path.write_text(json.dumps(publication))
    run_dir = tmp_path / "run"
    session = run_dir / "sessions/0"
    session.mkdir(parents=True)
    (session / "observation_attempts.jsonl").write_text("{}\n")
    (session / "generated.yaml").write_text("private: config")
    (run_dir / "hardware.json").write_text("private hardware")
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profile_path = profile_dir / "profile.json"
    profile_path.write_text("{}")
    measured, publications = [], []

    def run_profile(config, **kwargs):
        measured.append(config)
        return {"PROFILE_PATH": str(profile_path), "PROFILE_REF": "m/g/w/i",
                "RUN_DIR": str(run_dir), "RUN_ID": "run-1"}

    def create_commit(**kwargs):
        publications.append(kwargs)
        return SimpleNamespace(oid="profile-revision")

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "load_config", lambda path: load_config(root / path))
    monkeypatch.setattr(module, "run_profile", run_profile)
    monkeypatch.setattr(module, "HfApi", lambda: SimpleNamespace(create_commit=create_commit))
    monkeypatch.setattr(sys, "argv", ["profile_published", "--publication", str(publication_path)])
    module.main()
    assert (measured[0]["policy"]["hf_repo"], measured[0]["policy"]["root"],
            measured[0]["policy"]["revision"]) == ("org/models", "published/bundle", "model-revision")
    upload = publications[0]
    assert (upload["repo_id"], upload["repo_type"]) == ("MLL-Lab/LAGEN-profiles", "dataset")
    assert {operation.path_in_repo for operation in upload["operations"]} == {
        "runs/m/g/w/i/run-1/sessions/0/observation_attempts.jsonl", "profiles/m/g/w/i/profile.json",
    }
    receipt = json.loads((tmp_path / "outputs/tasks/atari_hist8_v1/profile_receipts/atlantis-qwenoft.json").read_text())
    assert (receipt["profile_repo"], receipt["profile_revision"], receipt["profile_path"]) == (
        "MLL-Lab/LAGEN-profiles", "profile-revision", "profiles/m/g/w/i/profile.json",
    )


def arguments(tmp_path, task="demon_attack", model="qwenoft"):
    return Namespace(task=task, model=model, dataset_dir=tmp_path / "data" / "lerobot",
                     base_model_dir=tmp_path / "backbone", output_root=tmp_path / "train",
                     run_id=f"{task}-{model}-v2", vlm_lr=3e-6, stop_after_steps=None,
                     preflight=False, resume=False, training_condition="profile")


@pytest.mark.parametrize("task,actions", [("atlantis", 4), ("asterix", 9)])
def test_profile_teacher_keeps_cpu_learning_and_independent_gpu_evaluation(tmp_path, task, actions):
    train, evaluation = teacher_configs(task, "qwenoft", tmp_path / "profile.json", tmp_path)
    assert train["experiment"]["seed"] == 0
    assert train["env"]["simulator"] == "cpu"
    assert train["env"]["frame_stack"] == evaluation["env"]["frame_stack"] == 4
    assert train["env"]["action_history_decisions"] == evaluation["env"]["action_history_decisions"] == 8
    assert len(train["env"]["action_order"]) == actions
    assert train["training"]["train_for_env_steps"] == 15_000_000
    assert train["training"]["learning_rate"] == 0.000303
    assert train["training"]["lr_schedule"] == "constant"
    assert train["executor"]["simulated_worker_capacity"] == 1
    assert train["latency"]["method"] == "temporal"
    assert train["latency"]["profile_worker_slot"] == 0
    assert train["latency"]["profile_path"] == evaluation["latency"]["profile_path"]
    assert evaluation["env"]["simulator"] == "gpu"
    assert evaluation["experiment"]["seed"] + evaluation["evaluation"]["episode_seed_offset"] == 1_000_000
    assert evaluation["evaluation"]["episodes"] == 100
    assert evaluation["executor"]["parallel_envs"] == 32
    assert evaluation["evaluation"]["max_env_steps"] == 7200
    assert evaluation["executor"]["env_backend"] == f"{task}_gpu_batched"
    assert evaluation["logging"]["wandb_name"] == train["logging"]["wandb_name"]
    assert train["logging"]["wandb_project"] == "latency-sensitive-bench"




def test_atari_profile_history_audit_preserves_rejected_commands_and_delayed_actions(tmp_path):
    reset = [0.0] * 56
    admitted = [1., 1., 0., 0., 0., 0., 0.]  # FIRE submitted; NOOP still applied.
    rejected = [0., 0., 0., 0., 1., 0., 0.]  # RIGHTFIRE rejected; earlier FIRE applied.
    states = [reset, reset[7:] + admitted, reset[14:] + admitted + rejected]
    for split in ("train", "val"):
        rows = [{"episode_idx": 0, "decision_step": i, "action_id": action, "state": state}
                for i, (action, state) in enumerate(zip((1, 2, 0), states))]
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / f"{split}.parquet")
    assert audit_atari_history(tmp_path, "atlantis")["checked_rows"] == 6
    rows[2]["state"][-6] = 1.0
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "val.parquet")
    with pytest.raises(AssertionError):
        audit_atari_history(tmp_path, "atlantis")


@pytest.mark.parametrize("task,state_dim,action_dim", [
    ("flappy", 24, 2), ("demon_attack", 88, 6), ("deadly_corridor", 120, 7),
    ("atlantis", 56, 4), ("asterix", 136, 9),
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
    assert cfg.wandb_entity == "saberrr-zju"
    assert cfg.wandb_project == "starvla_tasks"
    if model == "qwenoft":
        assert cfg.framework.action_model.loss_type == (
            "multibinary_bce" if task == "deadly_corridor" else "discrete_ce")


@pytest.mark.parametrize("task", ["atlantis", "asterix"])
@pytest.mark.parametrize("condition", ["zero", "profile"])
def test_atari_history_evaluation_uses_gpu_and_published_action_layout(tmp_path, task, condition):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "provenance.json").write_text(json.dumps({"training_run_id": task, "training_step": 5000}))
    cfg = evaluation_config(task, "qwenoft", bundle, tmp_path / "profile.json", tmp_path / "eval",
                            tmp_path / "backbone", Path("/python"), device="cuda:0", episodes=100,
                            env_condition=condition)
    assert cfg["env"]["simulator"] == "gpu"
    assert cfg["env"]["action_history_decisions"] == 8
    assert cfg["executor"]["env_backend"] == f"{task}_gpu_batched"
    assert cfg["executor"]["parallel_envs"] == 32
    assert cfg["evaluation"]["max_env_steps"] == 7200
    assert cfg["policy"]["task_manifest_path"] == str(bundle / "manifest.json")
    assert cfg["latency"]["method"] == ("zero" if condition == "zero" else "temporal")
    if task == "asterix":
        assert cfg["policy"]["action_layout"] == "discrete_9"


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


@pytest.mark.parametrize("task", ["demon_attack", "atlantis", "asterix"])
def test_partial_checkpoint_bundle_and_eval_preserve_actual_step(tmp_path, task):
    run = tmp_path / "old-run"
    checkpoint = run / "checkpoints" / "steps_2877_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    config = {
        "run_id": f"old-{task}", "version_id": "0.21",
        "rl_games": {"env_eval": {"image_size": 224}},
        "framework": {"name": "QwenOFT", "qwenvl": {"base_vlm": "/old/base"}},
        "datasets": {"vla_data": {"include_state": True, "obs_image_size": None,
                                    "image_mode": "single", "num_obs_frames": 1}},
    }
    (run / "config.full.yaml").write_text(yaml.safe_dump(config))
    (run / "dataset_statistics.json").write_text("{}")
    dataset = tmp_path / "data" / "lerobot"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps({"robot_type": f"rl_games_{task}"}))
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
    assert (bundle / provenance["checkpoint"]).read_bytes() == checkpoint.read_bytes()
    assert provenance["eval_protocol_version"] == EVAL_PROTOCOL
    cfg = evaluation_config(task, "qwenoft", bundle, tmp_path / "profile.json",
                            tmp_path / "eval", tmp_path / "backbone", Path("/python"),
                            device="cuda:0", episodes=100)
    assert "step2877" in cfg["experiment"]["name"]
    assert cfg["policy"]["state_source"] == "transport"
    assert "state_normalization" not in cfg["policy"]
    assert cfg["evaluation"]["episodes"] == 100
    assert cfg["executor"]["parallel_envs"] == 32
    assert cfg["evaluation"]["max_env_steps"] == 7200
    if task in {"atlantis", "asterix"}:
        from latency_bench.policy.starvla import prepare_starvla_checkpoint_input_config

        prepare_starvla_checkpoint_input_config(cfg)
        assert cfg["env"]["obs_resize"] == [224, 224]
    zero = evaluation_config("flappy", "qwenoft", bundle, tmp_path / "profile.json",
                             tmp_path / "eval-zero", tmp_path / "backbone", Path("/python"),
                             device="cuda:0", episodes=100, env_condition="zero",
                             inference_devices=["cuda:0", "cuda:1"])
    assert zero["latency"] == {"method": "zero"}
    assert zero["executor"]["parallel_envs"] == 32
    assert zero["evaluation"]["max_env_steps"] == 3600
    assert zero["executor"]["inference_batch_size"] == 16
    assert zero["executor"]["inference_devices"] == ["cuda:0", "cuda:1"]


def test_history01_metric_namespace_preserves_checkpoint_step():
    payload = _corrected_eval_scalar_payload({}, [{
        "checkpoint_global_step": 2877, "suite_name": "fixed_2", "latency_type": "fixed_sweep",
        "latency": 2, "mean_return": 1., "std_return": 0., "mean_length": 10., "returns": [1.],
    }], prefix="eval_history01_v2")
    assert payload["global_step"] == 2877
    assert payload["eval_history01_v2/fixed/latency_2/mean_return"] == 1.


def test_history_training_uses_named_run_and_resumes_recorded_wandb_id(tmp_path, monkeypatch):
    """Long-lived history training contract: resume preserves run identity and its source snapshot."""
    from omegaconf import OmegaConf
    from scripts.tasks.decision_history import train_vla_hist8 as module

    (tmp_path / "code_snapshot.json").write_text('{"revision": "training-source"}')
    config = OmegaConf.create({"run_id": "named-run", "trainer": {"max_train_steps": 5000},
        "datasets": {"vla_data": {"num_workers": 16, "eval_num_workers": 16, "prefetch_factor": 2}}})
    calls = []
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "training_config", lambda args: config)
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(kwargs["env"]))
    argv = ["train_vla_hist8", "--task", "flappy", "--model", "qwenoft", "--dataset-dir", str(tmp_path / "data"),
            "--base-model-dir", str(tmp_path / "backbone"), "--output-root", str(tmp_path),
            "--run-id", "named-run", "--vlm-lr", "0.000003", "--port", "12345"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert calls[0]["WANDB_RUN_ID"] == "named-run"
    record = tmp_path / "named-run/run_metadata.json"
    metadata = json.loads(record.read_text())
    metadata["wandb_run_id"] = "existing-wandb-run"
    record.write_text(json.dumps(metadata))
    (tmp_path / "code_snapshot.json").write_text('{"revision": "resume-source"}')
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    module.main()
    assert calls[1]["WANDB_RUN_ID"] == "existing-wandb-run"
    assert calls[1]["WANDB_RESUME"] == "allow"
    assert json.loads((tmp_path / "named-run/training_code_snapshot.json").read_text()) == {"revision": "training-source"}
