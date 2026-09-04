import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.official_gr00t.train_haic_dagger as train_haic_dagger
from scripts.official_gr00t.train_haic_dagger import (
    BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS,
    BOOTSTRAP_TRAIN_EPISODES,
    BOOTSTRAP_UPDATES,
    BOOTSTRAP_VAL_EPISODES,
    CHECKPOINT_COMPLETE_MARKER,
    COLLECTOR_RANKS,
    DAGGER_ROUNDS,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_HAIC_LATENT_LOSS_WEIGHT,
    DEFAULT_WANDB_RUN_ID,
    ENVS_PER_COLLECTOR_RANK,
    EVAL_ENVS,
    GLOBAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    HAIC_DAGGER_SHARD_ROWS,
    HAIC_DAGGER_ROWS_PER_COLLECTOR,
    HAIC_DAGGER_ROWS_PER_ROUND,
    UPDATES_PER_ROUND,
    archive_and_remove_raw_data,
    bootstrap_training_command,
    checkpoint_dir,
    collector_command,
    log_eval_metrics,
    prepare_command,
    read_eval_metrics,
    training_command,
    training_dataset_dirs,
    write_policy_config,
)


def test_main_resolves_bootstrap_paths_from_output_root(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "sys.argv", ["train_haic_dagger.py", "--output-root", "/tmp/haic-custom"]
    )
    monkeypatch.setattr(
        train_haic_dagger,
        "run_pipeline",
        lambda args: captured.update(vars(args)),
    )

    train_haic_dagger.main()

    root = Path("/tmp/haic-custom")
    assert captured["bootstrap_dataset"] == root / "bootstrap/lerobot"
    assert captured["bootstrap_checkpoint"] == (
        root
        / f"bootstrap/training/{DEFAULT_WANDB_RUN_ID}_sft/checkpoint-5000"
    )
    assert captured["actor_checkpoint"] == root / "bootstrap/actor_adapt.pt"
    assert captured["end_round"] == DAGGER_ROUNDS
    assert captured["collector_devices"] is None
    assert captured["collector_num_envs"] == ENVS_PER_COLLECTOR_RANK
    assert captured["haic_latent_loss_weight"] == DEFAULT_HAIC_LATENT_LOSS_WEIGHT


def test_haic_pipeline_constants_are_single_gpu_and_paper_scale():
    assert BOOTSTRAP_TRAIN_EPISODES == 250
    assert BOOTSTRAP_VAL_EPISODES == 25
    assert BOOTSTRAP_UPDATES == 5_000
    assert DEFAULT_HAIC_LATENT_LOSS_WEIGHT == 0.1
    assert GLOBAL_BATCH_SIZE == 64
    assert BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS == 2
    assert EVAL_ENVS == 100
    assert DAGGER_ROUNDS == 10
    assert UPDATES_PER_ROUND == 500
    assert GRADIENT_ACCUMULATION_STEPS == 2
    assert COLLECTOR_RANKS * HAIC_DAGGER_ROWS_PER_COLLECTOR == HAIC_DAGGER_ROWS_PER_ROUND
    assert HAIC_DAGGER_ROWS_PER_ROUND == 64_000
    assert HAIC_DAGGER_ROWS_PER_COLLECTOR == 32_000
    assert HAIC_DAGGER_SHARD_ROWS == 1_000
    assert COLLECTOR_RANKS * ENVS_PER_COLLECTOR_RANK == 192


def test_haic_commands_encode_smoke_and_resume_contract():
    teacher = Path("teacher.pt")
    first = collector_command(
        Path("haic-python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        output_dir=Path("raw"),
        inference_device="cuda:0",
        num_envs=8,
        seed=2,
        round_index=0,
        policy_config=Path("policy.yaml"),
        row_budget=64,
    )
    assert first[first.index("--num-envs") + 1] == "8"
    assert first[first.index("--row-budget") + 1] == "64"
    assert first[first.index("--vla-cadence") + 1] == "5"
    assert first[first.index("--inference-batch-size") + 1] == str(
        DEFAULT_INFERENCE_BATCH_SIZE
    )
    assert first[first.index("--policy-config") + 1] == "policy.yaml"
    assert "--student-actor-checkpoint" not in first

    prep = prepare_command(
        Path("raw"),
        Path("dataset"),
        dagger=True,
        round_index=0,
        expected_rows=64,
        normalization_source=Path("bootstrap/lerobot"),
    )
    assert "--dagger" in prep
    assert prep[prep.index("--expected-rows") + 1] == "64"

    train = training_command(
        bootstrap_checkpoint=Path("bootstrap/checkpoint-1"),
        replay_dataset=Path("bootstrap/lerobot"),
        dataset_dirs=[Path("output/round_00/lerobot")],
        actor_checkpoint=Path("actor.pt"),
        training_output=Path("output"),
        experiment_name="haic-test",
        resume_checkpoint=Path("output/checkpoint-500"),
        max_steps=2,
        gradient_accumulation_steps=1,
        haic_latent_loss_weight=0.25,
    )
    assert training_dataset_dirs(Path("out"), 1) == [
        Path("out/round_00/lerobot"),
        Path("out/round_01/lerobot"),
    ]
    assert train[train.index("--dataset-path") + 1] == "output/round_00/lerobot"
    assert train[train.index("--replay-bootstrap-dataset") + 1] == "bootstrap/lerobot"
    assert train[train.index("--num-gpus") + 1] == "1"
    assert train[train.index("--max-state-dim") + 1] == "605"
    assert train[train.index("--max-action-dim") + 1] == "256"
    assert train[train.index("--action-horizon") + 1] == "40"
    assert train[train.index("--num-inference-timesteps") + 1] == "4"
    assert "--allow-padding" in train
    assert "--fixed-action-noise-seed" not in train
    assert "--one-step-action-regression" not in train
    assert train[train.index("--gradient-accumulation-steps") + 1] == "1"
    assert train[-2:] == ["--resume-checkpoint-path", "output/checkpoint-500"]
    assert train[train.index("--action-objective") + 1] == "haic"
    assert train[train.index("--action-actor-checkpoint") + 1] == "actor.pt"
    assert train[train.index("--haic-latent-loss-weight") + 1] == "0.25"
    assert train[train.index("--experiment-name") + 1] == "haic-test"
    assert train[train.index("--learning-rate") + 1] == "0.00005"
    assert train[train.index("--shard-size") + 1] == str(HAIC_DAGGER_SHARD_ROWS)

    dual_device = collector_command(
        Path("haic-python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        output_dir=Path("raw/rank_01"),
        inference_device="cuda:1",
        simulation_device="cuda:1",
        num_envs=48,
        seed=2,
        round_index=0,
    )
    assert dual_device[dual_device.index("--device") + 1] == "cuda:1"
    assert dual_device[dual_device.index("--inference-device") + 1] == "cuda:1"
    assert dual_device[dual_device.index("--num-envs") + 1] == "48"

    bootstrap = bootstrap_training_command(
        base_model=Path("base"),
        dataset_dir=Path("dataset"),
        actor_checkpoint=Path("actor.pt"),
        training_output=Path("output"),
        experiment_name="haic-test_sft",
    )
    assert bootstrap[bootstrap.index("--max-steps") + 1] == str(BOOTSTRAP_UPDATES)
    assert bootstrap[
        bootstrap.index("--gradient-accumulation-steps") + 1
    ] == str(BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS)
    assert "--replay-bootstrap-dataset" not in bootstrap
    assert bootstrap[bootstrap.index("--action-objective") + 1] == "haic"
    assert bootstrap[
        bootstrap.index("--haic-latent-loss-weight") + 1
    ] == str(DEFAULT_HAIC_LATENT_LOSS_WEIGHT)
    assert bootstrap[bootstrap.index("--action-horizon") + 1] == "40"
    assert bootstrap[bootstrap.index("--num-inference-timesteps") + 1] == "4"
    assert "--allow-padding" in bootstrap
    assert "--fixed-action-noise-seed" not in bootstrap
    assert "--one-step-action-regression" not in bootstrap
    assert bootstrap[bootstrap.index("--experiment-name") + 1] == "haic-test_sft"
    assert checkpoint_dir(Path("output"), 5_000) == Path("output/checkpoint-5000")


def test_run_bootstrap_reuses_existing_dataset_for_fresh_sft(tmp_path, monkeypatch):
    existing_dataset = tmp_path / "existing-bootstrap/lerobot"
    args = SimpleNamespace(
        output_root=tmp_path / "new-run",
        bootstrap_dataset=existing_dataset,
        teacher_checkpoint=tmp_path / "teacher.pt",
        actor_checkpoint=tmp_path / "actor_adapt.pt",
        base_model=tmp_path / "base-model",
        haic_python=tmp_path / "haic-python",
        inference_device="cuda:0",
        eval_max_steps=1_000,
        inference_batch_size=32,
        wandb_run_id="latent-anchor",
        haic_latent_loss_weight=DEFAULT_HAIC_LATENT_LOSS_WEIGHT,
    )
    calls = []
    monkeypatch.setattr(
        train_haic_dagger,
        "_run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    checkpoint = train_haic_dagger.run_bootstrap(args, smoke=False)

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert str(train_haic_dagger.FINETUNE_SCRIPT) in command
    assert command[command.index("--dataset-path") + 1] == str(existing_dataset)
    assert "--haic-latent-loss-weight" in command
    assert str(train_haic_dagger.HAIC_SCRIPT) not in command
    assert kwargs["cwd"] == train_haic_dagger.REPO_ROOT
    assert checkpoint == (
        tmp_path / "new-run/bootstrap/training/latent-anchor_sft/checkpoint-5000"
    )


def test_read_eval_metrics_flattens_haic_eval_json(tmp_path):
    eval_path = tmp_path / "dagger-eval.json"
    eval_path.write_text(
        '{"episodes":100,"successes":45,"success_rate":0.45,'
        '"motion_progress":{"mean":0.6,"std":0.1},'
        '"cart_progress":{"mean":0.3,"std":0.2},'
        '"pullcart_score":{"mean":45.0,"std":15.0}}\n'
    )

    assert read_eval_metrics(eval_path) == {
        "dagger_eval/episodes": 100,
        "dagger_eval/successes": 45,
        "dagger_eval/success_rate": 0.45,
        "dagger_eval/motion_progress_mean": 0.6,
        "dagger_eval/motion_progress_std": 0.1,
        "dagger_eval/cart_progress_mean": 0.3,
        "dagger_eval/cart_progress_std": 0.2,
        "dagger_eval/pullcart_score_mean": 45.0,
        "dagger_eval/pullcart_score_std": 15.0,
    }


def test_log_eval_metrics_uses_training_step_and_run(tmp_path, monkeypatch):
    eval_path = tmp_path / "dagger-eval.json"
    eval_path.write_text(
        '{"episodes":100,"successes":45,"success_rate":0.45,'
        '"motion_progress":{"mean":0.6,"std":0.1},'
        '"cart_progress":{"mean":0.3,"std":0.2},'
        '"pullcart_score":{"mean":45.0,"std":15.0}}\n'
    )
    calls = {}

    class Run:
        def define_metric(self, name, *, step_metric):
            calls["metric"] = (name, step_metric)

        def log(self, metrics):
            calls["log"] = metrics

        def finish(self):
            pass

    monkeypatch.setattr(
        train_haic_dagger.wandb,
        "init",
        lambda **kwargs: calls.setdefault("init", kwargs) and Run(),
    )
    monkeypatch.setattr(train_haic_dagger.wandb, "teardown", lambda: None)

    log_eval_metrics(eval_path, step=500, run_id="latent-only")

    assert calls["init"] == {
        "project": "finetune-gr00t-n1d7",
        "id": "latent-only",
        "name": "latent-only",
        "resume": "allow",
    }
    assert calls["metric"] == ("dagger_eval/*", "train/global_step")
    assert calls["log"]["train/global_step"] == 500
    assert calls["log"]["dagger_eval/episodes"] == 100
    assert calls["log"]["dagger_eval/successes"] == 45
    assert calls["log"]["dagger_eval/success_rate"] == 0.45
    assert calls["log"]["dagger_eval/motion_progress_mean"] == 0.6
    assert calls["log"]["dagger_eval/motion_progress_std"] == 0.1
    assert calls["log"]["dagger_eval/cart_progress_mean"] == 0.3
    assert calls["log"]["dagger_eval/cart_progress_std"] == 0.2
    assert calls["log"]["dagger_eval/pullcart_score_mean"] == 45.0
    assert calls["log"]["dagger_eval/pullcart_score_std"] == 15.0
    assert "dagger_eval/steps" not in calls["log"]


def test_archive_and_remove_raw_data(tmp_path):
    round_dir = tmp_path / "round_00"
    raw_dir = round_dir / "raw"
    dataset_dir = round_dir / "lerobot"
    raw_dir.mkdir(parents=True)
    dataset_dir.mkdir()
    (raw_dir / "shard.npz").write_bytes(b"raw")
    (dataset_dir / "manifest.json").write_text('{"frames": 64000}\n')

    archive_and_remove_raw_data(round_dir, raw_dir, dataset_dir)

    assert (round_dir / "dataset_manifest.json").read_text() == '{"frames": 64000}\n'
    assert not raw_dir.exists()
    assert dataset_dir.exists()


def test_pipeline_restarts_partial_round_and_reuses_completed_training(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "run"
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text(
        "policy:\n  model_path: old\n  device: cuda:0\n  action_dim: 256\n"
    )
    args = SimpleNamespace(
        teacher_checkpoint=tmp_path / "teacher.pt",
        base_model=tmp_path / "base-model",
        bootstrap_checkpoint=tmp_path / "bootstrap/checkpoint-5000",
        bootstrap_dataset=tmp_path / "bootstrap/lerobot",
        actor_checkpoint=tmp_path / "bootstrap/actor_adapt.pt",
        output_root=output_root,
        policy_config=policy_config,
        haic_python=tmp_path / "haic-python",
        inference_device="cuda:0",
        collector_devices=None,
        collector_num_envs=ENVS_PER_COLLECTOR_RANK,
        inference_batch_size=32,
        eval_max_steps=1_000,
        wandb_run_id="haic-test",
        start_round=0,
        end_round=2,
        run_bootstrap=False,
        haic_latent_loss_weight=DEFAULT_HAIC_LATENT_LOSS_WEIGHT,
    )
    round_dir = output_root / "round_00"
    raw_dir = round_dir / "raw"
    dataset_dir = round_dir / "lerobot"
    trained_model = output_root / "training/haic-test/checkpoint-500"
    for path in (raw_dir, dataset_dir, trained_model):
        path.mkdir(parents=True)
        (path / "partial").write_text("stale\n")

    calls = {"collect": 0, "prepare": 0, "train": 0}

    def fake_run(command, **kwargs):
        if str(train_haic_dagger.HAIC_SCRIPT) in command:
            calls["collect"] += 1
            Path(command[command.index("--output-dir") + 1]).mkdir(parents=True)
        elif str(train_haic_dagger.PREPARE_SCRIPT) in command:
            calls["prepare"] += 1
            output = Path(
                command[command.index(str(train_haic_dagger.PREPARE_SCRIPT)) + 2]
            )
            (output / "meta").mkdir(parents=True)
            (output / "manifest.json").write_text('{"frames": 64000}\n')
        elif str(train_haic_dagger.FINETUNE_SCRIPT) in command:
            calls["train"] += 1
            assert command[command.index("--experiment-name") + 1] == "haic-test"
            dataset_path = command[command.index("--dataset-path") + 1]
            updates = 1000 if "--resume-checkpoint-path" in command else 500
            expected_datasets = [str(output_root / "round_00/lerobot")]
            if updates == 1000:
                expected_datasets.append(str(output_root / "round_01/lerobot"))
            assert dataset_path == os.pathsep.join(expected_datasets)
            assert command[command.index("--replay-bootstrap-dataset") + 1] == str(
                args.bootstrap_dataset
            )
            trained = checkpoint_dir(output_root / "training/haic-test", updates)
            trained.mkdir(parents=True)
            (trained / "trainer_state.json").write_text("{}\n")

    eval_steps = []
    fail_after_training = {"enabled": True}

    def fake_evaluate(*_args, step, **_kwargs):
        eval_steps.append(step)
        if step == 500 and fail_after_training["enabled"]:
            raise RuntimeError("interrupted after training")

    monkeypatch.setattr(train_haic_dagger.subprocess, "run", fake_run)
    monkeypatch.setattr(train_haic_dagger, "evaluate_checkpoint", fake_evaluate)

    with pytest.raises(RuntimeError, match="interrupted after training"):
        train_haic_dagger.run_pipeline(args)

    marker = trained_model / train_haic_dagger.CHECKPOINT_COMPLETE_MARKER
    assert marker.read_text() == "complete\n"
    assert not (trained_model / "partial").exists()

    fail_after_training["enabled"] = False
    train_haic_dagger.run_pipeline(args)

    assert calls == {"collect": 4, "prepare": 2, "train": 2}
    assert eval_steps == [0, 500, 500, 1000]
    for round_index, expected_datasets in enumerate(
        (
            [output_root / "round_00/lerobot"],
            [
                output_root / "round_00/lerobot",
                output_root / "round_01/lerobot",
            ],
        )
    ):
        current_round_dir = output_root / f"round_{round_index:02d}"
        assert (current_round_dir / "dataset_manifest.json").is_file()
        assert json.loads(
            (current_round_dir / "training_datasets.json").read_text()
        ) == {
            "round_index": round_index,
            "datasets": [str(path) for path in expected_datasets],
            "replay_dataset": str(args.bootstrap_dataset),
            "replay_ratio": 0.5,
        }
        assert not (current_round_dir / "raw").exists()
        assert (current_round_dir / "lerobot").exists()


def test_parallel_collectors_bind_devices_and_block_prepare_on_failure(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "run"
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text(
        "policy:\n  model_path: old\n  device: cuda:0\n  action_dim: 256\n"
    )
    args = SimpleNamespace(
        teacher_checkpoint=tmp_path / "teacher.pt",
        base_model=tmp_path / "base-model",
        bootstrap_checkpoint=tmp_path / "bootstrap/checkpoint-5000",
        bootstrap_dataset=tmp_path / "bootstrap/lerobot",
        actor_checkpoint=tmp_path / "bootstrap/actor_adapt.pt",
        output_root=output_root,
        policy_config=policy_config,
        haic_python=tmp_path / "haic-python",
        inference_device="cuda:0",
        collector_devices=["cuda:1", "cuda:0"],
        collector_num_envs=48,
        inference_batch_size=32,
        eval_max_steps=1_000,
        wandb_run_id="haic-test",
        start_round=0,
        end_round=1,
        run_bootstrap=False,
        haic_latent_loss_weight=DEFAULT_HAIC_LATENT_LOSS_WEIGHT,
    )
    launched = []

    class Process:
        def __init__(self, command, *, cwd, env):
            self.command = command
            self.env = env
            self.return_code = 1 if not launched else 0
            self.running = True
            self.terminated = False
            launched.append(self)

        def wait(self):
            self.running = False
            return self.return_code

        def poll(self):
            return None if self.running else self.return_code

        def terminate(self):
            self.terminated = True
            self.running = False

    monkeypatch.setattr(train_haic_dagger.subprocess, "Popen", Process)
    monkeypatch.setattr(
        train_haic_dagger, "evaluate_checkpoint", lambda *args, **kwargs: None
    )
    prepare_calls = []
    monkeypatch.setattr(
        train_haic_dagger,
        "_run",
        lambda *args, **kwargs: prepare_calls.append((args, kwargs)),
    )

    with pytest.raises(subprocess.CalledProcessError):
        train_haic_dagger.run_pipeline(args)

    assert len(launched) == 2
    assert [process.env["LOCAL_RANK"] for process in launched] == ["1", "0"]
    assert [
        process.command[process.command.index("--device") + 1]
        for process in launched
    ] == ["cuda:1", "cuda:0"]
    assert [
        process.command[process.command.index("--inference-device") + 1]
        for process in launched
    ] == ["cuda:1", "cuda:0"]
    assert launched[1].terminated is True
    assert prepare_calls == []


def test_existing_marker_must_match_training_dataset_manifest(tmp_path, monkeypatch):
    output_root = tmp_path / "run"
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text(
        "policy:\n  model_path: old\n  device: cuda:0\n  action_dim: 256\n"
    )
    round_dir = output_root / "round_00"
    round_dir.mkdir(parents=True)
    (round_dir / "training_datasets.json").write_text(
        '{"datasets":["old-current-only"],"round_index":0}\n'
    )
    trained_model = checkpoint_dir(output_root / "training/haic-test", 500)
    trained_model.mkdir(parents=True)
    (trained_model / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n")
    args = SimpleNamespace(
        teacher_checkpoint=tmp_path / "teacher.pt",
        base_model=tmp_path / "base-model",
        bootstrap_checkpoint=tmp_path / "bootstrap/checkpoint-5000",
        bootstrap_dataset=tmp_path / "bootstrap/lerobot",
        actor_checkpoint=tmp_path / "bootstrap/actor_adapt.pt",
        output_root=output_root,
        policy_config=policy_config,
        haic_python=tmp_path / "haic-python",
        inference_device="cuda:0",
        collector_devices=None,
        collector_num_envs=ENVS_PER_COLLECTOR_RANK,
        inference_batch_size=32,
        eval_max_steps=1_000,
        wandb_run_id="haic-test",
        start_round=0,
        end_round=1,
        run_bootstrap=False,
        haic_latent_loss_weight=DEFAULT_HAIC_LATENT_LOSS_WEIGHT,
    )

    monkeypatch.setattr(train_haic_dagger.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_haic_dagger, "evaluate_checkpoint", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="different DAgger dataset aggregate"):
        train_haic_dagger.run_pipeline(args)


def test_haic_policy_config_points_at_selected_vla(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text(
        "policy:\n  model_path: old\n  device: cuda:9\n  action_dim: 256\n"
    )
    destination = tmp_path / "runtime.yaml"
    write_policy_config(
        source,
        destination,
        model_path=Path("checkpoint-2"),
        inference_device="cuda:0",
    )
    contents = destination.read_text()
    assert "model_path: checkpoint-2" in contents
    assert "device: cuda:0" in contents
    assert "action_dim: 256" in contents
