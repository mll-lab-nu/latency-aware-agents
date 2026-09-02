import argparse
from pathlib import Path

from scripts.official_gr00t import train_parkour_dagger as dagger
from scripts.official_gr00t.train_parkour_dagger import (
    CHECKPOINT_COMPLETE_MARKER,
    DAGGER_ENVS,
    DAGGER_ROUNDS,
    DEFAULT_INFERENCE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    PARKOUR_DAGGER_ROWS_PER_ROUND,
    PARKOUR_DAGGER_SHARD_ROWS,
    checkpoint_dir,
    collector_command,
    read_eval_metrics,
    training_command,
)


def test_commands_preserve_round_resume_contract():
    output = Path("output")
    teacher = Path("teacher.pt")
    bootstrap = Path("bootstrap")

    first_collect = collector_command(
        Path("python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        policy_config=Path("policy.yaml"),
        output_dir=Path("raw"),
        inference_device="cuda:1",
        num_envs=DAGGER_ENVS,
        seed=1,
        round_index=0,
        dagger_row_budget=PARKOUR_DAGGER_ROWS_PER_ROUND,
        inference_batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    first_train = training_command(
        bootstrap_checkpoint=bootstrap,
        dataset_dir=Path("dataset"),
        training_output=output,
        round_index=0,
    )
    later_train = training_command(
        bootstrap_checkpoint=bootstrap,
        dataset_dir=Path("dataset"),
        training_output=output,
        round_index=1,
    )

    assert first_collect[first_collect.index("--num_envs") + 1] == str(DAGGER_ENVS)
    assert first_collect[first_collect.index("--seed") + 1] == "1"
    assert DAGGER_ENVS == 192
    assert DAGGER_ROUNDS * PARKOUR_DAGGER_ROWS_PER_ROUND == 640_000
    assert PARKOUR_DAGGER_ROWS_PER_ROUND == 500 * 128 * 1
    assert first_collect[first_collect.index("--inference_batch_size") + 1] == str(
        DEFAULT_INFERENCE_BATCH_SIZE
    )
    assert first_collect[first_collect.index("--dagger_shard_rows") + 1] == str(
        PARKOUR_DAGGER_SHARD_ROWS
    )
    assert first_collect[first_collect.index("--dagger_row_budget") + 1] == str(
        PARKOUR_DAGGER_ROWS_PER_ROUND
    )
    assert "--student_actor_checkpoint" not in first_collect
    assert "--resume-checkpoint-path" not in first_train
    assert "--state-history-length" not in first_train
    assert later_train[-2:] == [
        "--resume-checkpoint-path",
        "output/checkpoint-500",
    ]
    assert first_train[first_train.index("--global-batch-size") + 1] == "128"
    assert first_train[first_train.index("--gradient-accumulation-steps") + 1] == str(
        GRADIENT_ACCUMULATION_STEPS
    )
    assert first_train[first_train.index("--episode-sampling-rate") + 1] == "1.0"
    assert first_train[first_train.index("--shard-size") + 1] == str(
        PARKOUR_DAGGER_SHARD_ROWS
    )
    assert first_train[first_train.index("--dataloader-num-workers") + 1] == "8"
    assert first_train[first_train.index("--save-total-limit") + 1] == "2"
    assert "--experiment-name" not in first_train
    assert "--parkour-dagger" not in first_train
    assert "--parkour-actor-checkpoint" not in first_train
    eval_command = collector_command(
        Path("python"),
        mode="vla-eval",
        teacher_checkpoint=teacher,
        policy_config=Path("policy.yaml"),
        output_dir=Path("eval"),
        inference_device="cuda:1",
        num_envs=50,
        seed=2,
        round_index=0,
    )
    assert "--dagger_round" not in eval_command
    assert "--dagger_shard_rows" not in eval_command
    assert "--dagger_row_budget" not in eval_command
    assert checkpoint_dir(output, 10) == output / "checkpoint-5000"


def test_one_round_orchestration_evaluates_before_collection_and_after_training(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("policy:\n  model_path: old\n  device: cuda:0\n")
    events = []

    def fake_run(command, **kwargs):
        if str(dagger.PARKOUR_SCRIPT) in command:
            mode = command[2]
            num_envs = command[command.index("--num_envs") + 1]
            events.append(f"{mode}:{num_envs}")
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            if mode == "vla-eval":
                (output_dir / "vla-eval.json").write_text("{}\n")
        elif str(dagger.PREPARE_SCRIPT) in command:
            events.append("prepare")
            dataset_dir = Path(command[3])
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "manifest.json").write_text('{"frames": 64000}\n')
        elif str(dagger.FINETUNE_SCRIPT) in command:
            events.append("train")
            checkpoint_dir(output_root / "training", 1).mkdir(parents=True)

    def fake_log_eval_metrics(eval_path, *, step, run_id):
        events.append(f"wandb:{step}")

    monkeypatch.setattr(dagger.subprocess, "run", fake_run)
    monkeypatch.setattr(dagger, "log_eval_metrics", fake_log_eval_metrics)
    args = argparse.Namespace(
        teacher_checkpoint=tmp_path / "teacher.pt",
        bootstrap_checkpoint=tmp_path / "bootstrap",
        bootstrap_dataset=tmp_path / "bootstrap_dataset",
        output_root=output_root,
        policy_config=policy_config,
        parkour_python=Path("/usr/bin/python"),
        inference_device="cuda:1",
        inference_batch_size=32,
        wandb_run_id="latent-only",
        start_round=0,
        end_round=1,
    )

    dagger.run_pipeline(args)

    assert events == [
        "vla-eval:50",
        "wandb:0",
        "dagger-collect:192",
        "prepare",
        "train",
        "vla-eval:50",
        "wandb:500",
    ]
    assert (
        checkpoint_dir(output_root / "training", 1) / CHECKPOINT_COMPLETE_MARKER
    ).read_text() == "complete\n"
    round_dir = output_root / "round_00"
    assert (round_dir / "dataset_manifest.json").read_text() == '{"frames": 64000}\n'
    assert not (round_dir / "raw").exists()
    assert not (round_dir / "lerobot").exists()


def test_read_eval_metrics_flattens_vla_eval_json(tmp_path):
    eval_path = tmp_path / "vla-eval.json"
    eval_path.write_text(
        '{"reward":{"mean":1.0,"std":0.2},'
        '"episode_length":{"mean":3.0,"std":0.4},'
        '"normalized_waypoint_progress":{"mean":0.5,"std":0.1},'
        '"edge_violation":{"mean":0.0,"std":0.3}}\n'
    )

    assert read_eval_metrics(eval_path) == {
        "dagger_eval/reward_mean": 1.0,
        "dagger_eval/reward_std": 0.2,
        "dagger_eval/episode_length_mean": 3.0,
        "dagger_eval/episode_length_std": 0.4,
        "dagger_eval/normalized_waypoint_progress_mean": 0.5,
        "dagger_eval/normalized_waypoint_progress_std": 0.1,
        "dagger_eval/edge_violation_mean": 0.0,
        "dagger_eval/edge_violation_std": 0.3,
    }


def test_log_eval_metrics_uses_training_step_and_run(tmp_path, monkeypatch):
    eval_path = tmp_path / "vla-eval.json"
    eval_path.write_text(
        '{"reward":{"mean":1.0,"std":0.2},'
        '"episode_length":{"mean":3.0,"std":0.4},'
        '"normalized_waypoint_progress":{"mean":0.5,"std":0.1},'
        '"edge_violation":{"mean":0.0,"std":0.3}}\n'
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
        dagger.wandb,
        "init",
        lambda **kwargs: calls.setdefault("init", kwargs) and Run(),
    )
    monkeypatch.setattr(dagger.wandb, "teardown", lambda: None)

    dagger.log_eval_metrics(eval_path, step=500, run_id="latent-only")

    assert calls["init"] == {
        "project": "finetune-gr00t-n1d7",
        "id": "latent-only",
        "resume": "allow",
    }
    assert calls["metric"] == ("dagger_eval/*", "train/global_step")
    assert calls["log"]["train/global_step"] == 500
    assert calls["log"]["dagger_eval/normalized_waypoint_progress_mean"] == 0.5
