import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from scripts.official_gr00t import train_parkour_dagger as dagger
from scripts.official_gr00t.train_parkour_dagger import (
    BOOTSTRAP_UPDATES,
    CHECKPOINT_COMPLETE_MARKER,
    DAGGER_ENVS,
    DAGGER_ROUNDS,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_PARKOUR_STARTUP_TIMEOUT_SEC,
    GRADIENT_ACCUMULATION_STEPS,
    PARKOUR_EVAL_EPISODES,
    PARKOUR_EVAL_MAX_EPISODE_STEPS,
    PARKOUR_DAGGER_ROWS_PER_ROUND,
    PARKOUR_DAGGER_SHARD_ROWS,
    DEFAULT_BASE_MODEL,
    DEFAULT_BOOTSTRAP_DATASET,
    DEFAULT_OUTPUT_ROOT,
    bootstrap_prepare_command,
    bootstrap_checkpoint_dir,
    bootstrap_training_command,
    checkpoint_dir,
    collector_command,
    read_eval_metrics,
    prepare_command,
    training_command,
    training_dataset_dirs,
)


def test_main_defaults_to_all_dagger_rounds(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        ["train_parkour_dagger.py", "--latency-profile-path", "profile.json"],
    )
    monkeypatch.setattr(dagger, "run_pipeline", lambda args: captured.update(vars(args)))

    dagger.main()

    assert captured["end_round"] == DAGGER_ROUNDS
    assert (
        captured["wandb_run_id"]
        == "extreme_parkour_go2_vla_dagger_action_yaw_h40_official_flow_aux_5k"
    )
    assert captured["base_model"] == DEFAULT_BASE_MODEL
    assert captured["bootstrap_dataset"] == DEFAULT_BOOTSTRAP_DATASET
    assert captured["bootstrap_raw"] is None
    assert captured["bootstrap_checkpoint"] == bootstrap_checkpoint_dir(
        DEFAULT_OUTPUT_ROOT / "bootstrap" / "training",
        "extreme_parkour_go2_vla_dagger_action_yaw_h40_official_flow_aux_5k_sft",
    )
    assert captured["run_bootstrap"] is False
    assert captured["latency_profile_path"] == Path("profile.json")
    assert (
        captured["parkour_startup_timeout_sec"]
        == DEFAULT_PARKOUR_STARTUP_TIMEOUT_SEC
    )


def test_commands_preserve_round_resume_contract():
    output = Path("output")
    teacher = Path("teacher.pt")
    bootstrap = Path("bootstrap")
    bootstrap_dataset = Path("bootstrap_dataset")
    round0_dataset = output / "round_00/lerobot"
    round1_dataset = output / "round_01/lerobot"

    first_collect = collector_command(
        Path("python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        policy_config=Path("policy.yaml"),
        output_dir=Path("raw"),
        inference_device=["cuda:1"],
        num_envs=DAGGER_ENVS,
        seed=1,
        round_index=0,
        dagger_row_budget=PARKOUR_DAGGER_ROWS_PER_ROUND,
        inference_batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    first_train = training_command(
        bootstrap_checkpoint=bootstrap,
        replay_dataset=bootstrap_dataset,
        dataset_dirs=[round0_dataset],
        training_output=output,
        round_index=0,
        experiment_name="action-yaw",
        actor_checkpoint=teacher,
    )
    later_train = training_command(
        bootstrap_checkpoint=bootstrap,
        replay_dataset=bootstrap_dataset,
        dataset_dirs=[round0_dataset, round1_dataset],
        training_output=output,
        round_index=1,
        experiment_name="action-yaw",
        actor_checkpoint=teacher,
    )
    smoke_train = training_command(
        bootstrap_checkpoint=bootstrap,
        replay_dataset=None,
        dataset_dirs=[round0_dataset],
        training_output=output,
        round_index=0,
        experiment_name="smoke",
        actor_checkpoint=teacher,
        num_gpus=1,
        global_batch_size=1,
    )
    prepare = prepare_command(
        Path("raw"),
        Path("dataset"),
        round_index=0,
        normalization_source=bootstrap_dataset,
    )

    assert training_dataset_dirs(Path("out"), 1) == [
        Path("out/round_00/lerobot"),
        Path("out/round_01/lerobot"),
    ]
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
    assert prepare[prepare.index("--normalization-source") + 1] == str(
        bootstrap_dataset
    )
    assert "--student_actor_checkpoint" not in first_collect
    assert "--resume-checkpoint-path" not in first_train
    assert "--state-history-length" not in first_train
    assert first_train[first_train.index("--dataset-path") + 1] == os.pathsep.join(
        str(path) for path in [round0_dataset]
    )
    assert later_train[later_train.index("--dataset-path") + 1] == os.pathsep.join(
        str(path) for path in [round0_dataset, round1_dataset]
    )
    assert first_train[first_train.index("--replay-bootstrap-dataset") + 1] == str(
        bootstrap_dataset
    )
    assert later_train[-2:] == [
        "--resume-checkpoint-path",
        "output/action-yaw/checkpoint-500",
    ]
    assert first_train[first_train.index("--global-batch-size") + 1] == "128"
    assert first_train[first_train.index("--num-gpus") + 1] == "2"
    assert smoke_train[2] == "--nproc_per_node=1"
    assert smoke_train[smoke_train.index("--num-gpus") + 1] == "1"
    assert smoke_train[smoke_train.index("--global-batch-size") + 1] == "1"
    assert first_train[first_train.index("--gradient-accumulation-steps") + 1] == str(
        GRADIENT_ACCUMULATION_STEPS
    )
    assert first_train[first_train.index("--episode-sampling-rate") + 1] == "1.0"
    assert first_train[first_train.index("--shard-size") + 1] == str(
        PARKOUR_DAGGER_SHARD_ROWS
    )
    assert first_train[first_train.index("--dataloader-num-workers") + 1] == "8"
    assert first_train[first_train.index("--save-total-limit") + 1] == "2"
    assert first_train[first_train.index("--experiment-name") + 1] == "action-yaw"
    assert first_train[first_train.index("--action-objective") + 1] == "parkour"
    assert first_train[first_train.index("--action-horizon") + 1] == "40"
    assert first_train[first_train.index("--num-inference-timesteps") + 1] == "4"
    assert "--allow-padding" in first_train
    assert "--fixed-action-noise-seed" not in first_train
    assert "--one-step-action-regression" not in first_train
    assert first_train[first_train.index("--action-actor-checkpoint") + 1] == str(
        teacher
    )
    assert first_train[first_train.index("--learning-rate") + 1] == "0.00005"
    eval_command = collector_command(
        Path("python"),
        mode="latency-eval",
        teacher_checkpoint=teacher,
        policy_config=Path("policy.yaml"),
        output_dir=Path("eval"),
        inference_device=["cuda:1"],
        num_envs=50,
        seed=2,
        round_index=0,
    )
    assert "--dagger_round" not in eval_command
    assert "--dagger_shard_rows" not in eval_command
    assert "--dagger_row_budget" not in eval_command
    assert eval_command[eval_command.index("--eval_episodes") + 1] == str(
        PARKOUR_EVAL_EPISODES
    )
    assert eval_command[eval_command.index("--max_episode_steps") + 1] == str(
        PARKOUR_EVAL_MAX_EPISODE_STEPS
    )
    assert checkpoint_dir(output, 10) == output / "checkpoint-5000"


def test_collector_command_repeats_explicit_inference_devices():
    command = collector_command(
        Path("python"),
        mode="dagger-collect",
        teacher_checkpoint=Path("teacher.pt"),
        policy_config=Path("policy.yaml"),
        output_dir=Path("raw"),
        inference_device=["cuda:0", "cuda:1"],
        num_envs=192,
        seed=1,
        round_index=0,
    )

    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--inference_device"
    ] == ["cuda:0", "cuda:1"]


def test_parkour_startup_timeout_kills_and_retries_once(tmp_path, capsys):
    attempts = tmp_path / "attempts"
    script = """
import sys
import time
from pathlib import Path

attempts = Path(sys.argv[1])
attempt = int(attempts.read_text()) + 1 if attempts.exists() else 1
attempts.write_text(str(attempt))
ready = Path(sys.argv[sys.argv.index("--startup-ready-file") + 1])
if attempt == 1:
    time.sleep(60)
ready.touch()
time.sleep(0.5)
"""

    dagger.run_parkour_command(
        [sys.executable, "-c", script, str(attempts)],
        cwd=tmp_path,
        env=os.environ.copy(),
        startup_timeout_sec=0.3,
    )

    assert attempts.read_text() == "2"
    assert "retrying once" in capsys.readouterr().err


def test_parkour_startup_timeout_stops_after_second_attempt(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        dagger.run_parkour_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            startup_timeout_sec=0.1,
        )


def _process_is_running(pid: int) -> bool:
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text().split()[2] != "Z"


def test_parkour_startup_failure_is_not_retried_and_cleans_descendants(tmp_path):
    attempts = tmp_path / "attempts"
    descendant_pid = tmp_path / "descendant-pid"
    script = """
import subprocess
import sys
from pathlib import Path

attempts = Path(sys.argv[1])
attempts.write_text(str(int(attempts.read_text()) + 1 if attempts.exists() else 1))
descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
Path(sys.argv[2]).write_text(str(descendant.pid))
raise SystemExit(7)
"""

    with pytest.raises(subprocess.CalledProcessError) as error:
        dagger.run_parkour_command(
            [sys.executable, "-c", script, str(attempts), str(descendant_pid)],
            cwd=tmp_path,
            env=os.environ.copy(),
            startup_timeout_sec=1.0,
        )

    assert error.value.returncode == 7
    assert attempts.read_text() == "1"
    pid = int(descendant_pid.read_text())
    deadline = time.monotonic() + 2.0
    while _process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_is_running(pid)


def test_parkour_runtime_failure_after_startup_is_nonzero(tmp_path):
    script = """
from pathlib import Path
import sys

Path(sys.argv[1]).touch()
raise SystemExit(7)
"""
    ready = tmp_path / "ready"

    with pytest.raises(subprocess.CalledProcessError) as error:
        dagger.run_parkour_command(
            [sys.executable, "-c", script, str(ready)],
            cwd=tmp_path,
            env=os.environ.copy(),
            startup_timeout_sec=1.0,
        )

    assert error.value.returncode == 7


def test_parkour_parent_sigterm_cleans_child_process_group(tmp_path):
    child_pid = tmp_path / "child-pid"
    orchestrator_script = """
import os
import signal
import sys
from pathlib import Path
from scripts.official_gr00t import train_parkour_dagger as dagger

real_popen = dagger.subprocess.Popen
def tracked_popen(*args, **kwargs):
    process = real_popen(*args, **kwargs)
    Path(sys.argv[2]).write_text(str(process.pid))
    os.kill(os.getpid(), signal.SIGTERM)
    return process

dagger.subprocess.Popen = tracked_popen
dagger.run_parkour_command(
    [sys.executable, "-c", "import time; time.sleep(5)"],
    cwd=Path(sys.argv[1]),
    env=os.environ.copy(),
    startup_timeout_sec=60.0,
)
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            orchestrator_script,
            str(dagger.REPO_ROOT),
            str(child_pid),
        ],
        cwd=dagger.REPO_ROOT,
        env=os.environ.copy(),
        check=False,
    )

    assert result.returncode == 143
    pid = int(child_pid.read_text())
    assert not _process_is_running(pid)


def test_bootstrap_command_starts_from_gr00t_base_without_replay():
    command = bootstrap_training_command(
        base_model=Path("base-model"),
        dataset_dir=Path("action-bootstrap"),
        training_output=Path("output/bootstrap/training"),
        experiment_name="action-yaw_sft",
        actor_checkpoint=Path("teacher.pt"),
    )

    assert command[command.index("--base-model-path") + 1] == "base-model"
    assert command[command.index("--dataset-path") + 1] == "action-bootstrap"
    assert command[command.index("--experiment-name") + 1] == "action-yaw_sft"
    assert command[command.index("--max-steps") + 1] == str(BOOTSTRAP_UPDATES)
    assert command[command.index("--stop-every-steps") + 1] == str(BOOTSTRAP_UPDATES)
    assert "--replay-bootstrap-dataset" not in command
    assert "--resume-checkpoint-path" not in command

    prepare = bootstrap_prepare_command(Path("raw"), Path("lerobot"))
    assert prepare[-2:] == ["--force", "--video-shards"]


def test_run_bootstrap_returns_fresh_checkpoint_path(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        if len(captured) <= 2:
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True)
            mode = command[2]
            progress = 0.8 if mode == "teacher-eval" else 0.5
            (output_dir / f"{mode}.json").write_text(
                json.dumps(
                    {"normalized_waypoint_progress": {"mean": progress}}
                )
            )

    monkeypatch.setattr(dagger.subprocess, "run", fake_run)
    monkeypatch.setattr(dagger, "run_parkour_command", fake_run)
    args = argparse.Namespace(
        base_model=tmp_path / "base-model",
        bootstrap_updates=dagger.BOOTSTRAP_UPDATES,
        bootstrap_dataset=tmp_path / "action-bootstrap",
        bootstrap_raw=None,
        teacher_checkpoint=tmp_path / "teacher.pt",
        output_root=tmp_path / "output",
        policy_config=tmp_path / "policy.yaml",
        parkour_python=Path("/usr/bin/python"),
        inference_device=["cuda:1"],
        inference_batch_size=32,
        parkour_startup_timeout_sec=300.0,
        wandb_run_id="action-yaw",
        num_gpus=1,
        global_batch_size=1,
    )

    assert dagger.run_bootstrap(args) == (
        tmp_path / "output/bootstrap/training/action-yaw_sft/checkpoint-5000"
    )
    modes = [command[2] for command, _ in captured[:3]]
    assert modes == ["teacher-eval", "oracle-eval", "collect"]
    assert str(dagger.PREPARE_SCRIPT) in captured[3][0]
    training_command, training_kwargs = captured[4]
    assert training_command[training_command.index("--base-model-path") + 1] == str(
        args.base_model
    )
    assert "--replay-bootstrap-dataset" not in training_command
    assert training_kwargs["env"]["WANDB_RUN_ID"] == "action-yaw_sft"
    assert training_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_run_bootstrap_reuses_explicit_raw_without_collecting(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(
        dagger.subprocess,
        "run",
        lambda command, **kwargs: captured.append((command, kwargs)),
    )
    args = argparse.Namespace(
        base_model=tmp_path / "base-model",
        bootstrap_updates=dagger.BOOTSTRAP_UPDATES,
        bootstrap_dataset=tmp_path / "action-bootstrap",
        bootstrap_raw=tmp_path / "existing-raw",
        teacher_checkpoint=tmp_path / "teacher.pt",
        output_root=tmp_path / "output",
        policy_config=tmp_path / "policy.yaml",
        parkour_python=Path("/usr/bin/python"),
        inference_device=["cuda:1"],
        inference_batch_size=32,
        parkour_startup_timeout_sec=300.0,
        wandb_run_id="action-yaw",
        num_gpus=1,
        global_batch_size=1,
    )

    dagger.run_bootstrap(args)

    assert len(captured) == 2
    assert captured[0][0][2:4] == [
        str(args.bootstrap_raw),
        str(args.bootstrap_dataset),
    ]
    assert str(dagger.FINETUNE_SCRIPT) in captured[1][0]


def test_run_bootstrap_rejects_oracle_below_teacher_gate(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True)
        mode = command[2]
        progress = 0.8 if mode == "teacher-eval" else 0.3
        (output_dir / f"{mode}.json").write_text(
            json.dumps({"normalized_waypoint_progress": {"mean": progress}})
        )

    monkeypatch.setattr(dagger, "run_parkour_command", fake_run)
    args = argparse.Namespace(
        base_model=tmp_path / "base-model",
        bootstrap_updates=dagger.BOOTSTRAP_UPDATES,
        bootstrap_dataset=tmp_path / "action-bootstrap",
        bootstrap_raw=None,
        teacher_checkpoint=tmp_path / "teacher.pt",
        output_root=tmp_path / "output",
        policy_config=tmp_path / "policy.yaml",
        parkour_python=Path("/usr/bin/python"),
        inference_device=["cuda:1"],
        inference_batch_size=32,
        parkour_startup_timeout_sec=300.0,
        wandb_run_id="action-yaw",
        num_gpus=1,
        global_batch_size=1,
    )

    with pytest.raises(RuntimeError, match="below 50% of teacher"):
        dagger.run_bootstrap(args)


@pytest.mark.parametrize("explicit_latency", [False, True])
def test_two_round_orchestration_aggregates_datasets_and_keeps_lerobot(
    tmp_path, monkeypatch, explicit_latency
):
    output_root = tmp_path / "output"
    latency_config = tmp_path / "latency.yaml"
    latency_config.write_text(
        "latency: {method: profile_sample, seed: 17}\n"
        "executor: {simulated_worker_capacity: 3}\n"
        "scheduler: {hold_policy: noop, ordering_policy: oldest_ready}\n"
    )
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text(
        "env: {}\n"
        "policy:\n  model_path: old\n  device: cuda:0\n"
        "evaluation: {}\n"
        "logging:\n  output_dir: old\n"
    )
    events = []

    def fake_run(command, **kwargs):
        if str(dagger.PARKOUR_SCRIPT) in command:
            mode = command[2]
            num_envs = command[command.index("--num_envs") + 1]
            events.append(f"{mode}:{num_envs}")
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            if mode == "latency-eval":
                (output_dir / "queue_eval_latency_profile_sample.json").write_text(
                    json.dumps(
                        {
                            "mean_return": 1.0,
                            "std_return": 0.0,
                            "mean_length": 1.0,
                            "lengths": [1.0],
                            "task_metrics": {
                                "normalized_waypoint_progress": {"mean": 1.0, "std": 0.0},
                                "edge_violation": {"mean": 0.0, "std": 0.0},
                            },
                        }
                    )
                )
        elif str(dagger.PREPARE_SCRIPT) in command:
            events.append("prepare")
            assert kwargs["env"]["PYTHONPATH"] == str(dagger.REPO_ROOT)
            dataset_dir = Path(command[3])
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "manifest.json").write_text('{"frames": 64000}\n')
        elif str(dagger.FINETUNE_SCRIPT) in command:
            events.append(f"train:{command[command.index('--dataset-path') + 1]}")
            completed_rounds = 2 if "--resume-checkpoint-path" in command else 1
            assert command[command.index("--replay-bootstrap-dataset") + 1] == str(
                tmp_path / "bootstrap_dataset"
            )
            checkpoint_dir(output_root / "training/action-yaw", completed_rounds).mkdir(
                parents=True
            )

    def fake_log_eval_metrics(eval_path, *, step, run_id):
        events.append(f"wandb:{step}")

    monkeypatch.setattr(dagger.subprocess, "run", fake_run)
    monkeypatch.setattr(dagger, "run_parkour_command", fake_run)
    monkeypatch.setattr(dagger, "log_eval_metrics", fake_log_eval_metrics)
    args = argparse.Namespace(
        teacher_checkpoint=tmp_path / "teacher.pt",
        base_model=tmp_path / "base-model",
        bootstrap_checkpoint=tmp_path / "bootstrap",
        bootstrap_dataset=tmp_path / "bootstrap_dataset",
        output_root=output_root,
        policy_config=policy_config,
        latency_profile_path=tmp_path / "profile.json",
        latency_config=latency_config if explicit_latency else None,
        parkour_python=Path("/usr/bin/python"),
        inference_device=["cuda:1"],
        inference_batch_size=32,
        parkour_startup_timeout_sec=300.0,
        wandb_run_id="action-yaw",
        start_round=0,
        end_round=2,
        run_bootstrap=False,
        num_gpus=1,
        global_batch_size=1,
    )

    dagger.run_pipeline(args)

    for eval_config in output_root.rglob("eval_config.yaml"):
        resolved = yaml.safe_load(eval_config.read_text())
        if explicit_latency:
            assert resolved["latency"]["method"] == "profile_sample"
            assert resolved["latency"]["seed"] == 17
            assert resolved["executor"]["simulated_worker_capacity"] == 3
            assert resolved["scheduler"] == {"hold_policy": "noop", "ordering_policy": "oldest_ready"}
        else:
            assert resolved["latency"]["method"] == "temporal"

    round0_dataset = output_root / "round_00/lerobot"
    round1_dataset = output_root / "round_01/lerobot"
    assert events == [
        "latency-eval:50",
        "wandb:0",
        "dagger-collect:192",
        "prepare",
        "train:" + str(round0_dataset),
        "latency-eval:50",
        "wandb:500",
        "dagger-collect:192",
        "prepare",
        "train:"
        + os.pathsep.join(str(path) for path in [round0_dataset, round1_dataset]),
        "latency-eval:50",
        "wandb:1000",
    ]
    assert (
        checkpoint_dir(output_root / "training/action-yaw", 1)
        / CHECKPOINT_COMPLETE_MARKER
    ).read_text() == "complete\n"
    assert (
        checkpoint_dir(output_root / "training/action-yaw", 2)
        / CHECKPOINT_COMPLETE_MARKER
    ).read_text() == "complete\n"
    for round_index, expected_datasets in enumerate(
        (
            [round0_dataset],
            [round0_dataset, round1_dataset],
        )
    ):
        round_dir = output_root / f"round_{round_index:02d}"
        assert (round_dir / "dataset_manifest.json").read_text() == '{"frames": 64000}\n'
        assert json.loads((round_dir / "training_datasets.json").read_text()) == {
            "round_index": round_index,
            "datasets": [str(path) for path in expected_datasets],
            "replay_dataset": str(args.bootstrap_dataset),
            "replay_ratio": 0.5,
        }
        assert not (round_dir / "raw").exists()
        assert (round_dir / "lerobot").exists()


def test_existing_marker_must_match_training_dataset_manifest(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    round_dir = output_root / "round_00"
    (checkpoint_dir(output_root / "training/action-yaw", 1)).mkdir(parents=True)
    (
        checkpoint_dir(output_root / "training/action-yaw", 1)
        / CHECKPOINT_COMPLETE_MARKER
    ).write_text("complete\n")
    round_dir.mkdir(parents=True)
    (round_dir / "training_datasets.json").write_text(
        '{"datasets":["old-current-only"],"round_index":0}\n'
    )
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("policy:\n  model_path: old\n  device: cuda:0\n")
    events = []

    def fake_run(command, **kwargs):
        if str(dagger.PARKOUR_SCRIPT) in command:
            mode = command[2]
            events.append(mode)
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            if mode == "latency-eval":
                (output_dir / "queue_eval_latency_profile_sample.json").write_text(
                    json.dumps(
                        {
                            "mean_return": 1.0,
                            "std_return": 0.0,
                            "mean_length": 1.0,
                            "lengths": [1.0],
                            "task_metrics": {
                                "normalized_waypoint_progress": {"mean": 1.0, "std": 0.0},
                                "edge_violation": {"mean": 0.0, "std": 0.0},
                            },
                        }
                    )
                )
        elif str(dagger.PREPARE_SCRIPT) in command:
            events.append("prepare")
            dataset_dir = Path(command[3])
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "manifest.json").write_text('{"frames": 64000}\n')
        elif str(dagger.FINETUNE_SCRIPT) in command:
            events.append("train")

    monkeypatch.setattr(dagger.subprocess, "run", fake_run)
    monkeypatch.setattr(dagger, "run_parkour_command", fake_run)
    monkeypatch.setattr(dagger, "log_eval_metrics", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        teacher_checkpoint=tmp_path / "teacher.pt",
        base_model=tmp_path / "base-model",
        bootstrap_checkpoint=tmp_path / "bootstrap",
        bootstrap_dataset=tmp_path / "bootstrap_dataset",
        output_root=output_root,
        policy_config=policy_config,
        latency_profile_path=tmp_path / "profile.json",
        parkour_python=Path("/usr/bin/python"),
        inference_device=["cuda:1"],
        inference_batch_size=32,
        parkour_startup_timeout_sec=300.0,
        wandb_run_id="action-yaw",
        start_round=0,
        end_round=1,
        run_bootstrap=False,
        num_gpus=1,
        global_batch_size=1,
    )

    with pytest.raises(RuntimeError, match="different DAgger dataset aggregate"):
        dagger.run_pipeline(args)

    assert events == ["dagger-collect", "prepare"]


def test_read_eval_metrics_flattens_common_eval_summary(tmp_path):
    eval_path = tmp_path / "queue_eval_latency_0.json"
    eval_path.write_text(
        json.dumps(
            {
                "mean_return": 1.0,
                "std_return": 0.2,
                "mean_length": 3.0,
                "lengths": [2.0, 4.0],
                "task_metrics": {
                    "normalized_waypoint_progress": {"mean": 0.5, "std": 0.1},
                    "edge_violation": {"mean": 0.0, "std": 0.3},
                },
            }
        )
    )

    assert read_eval_metrics(eval_path) == {
        "dagger_eval/reward_mean": 1.0,
        "dagger_eval/reward_std": 0.2,
        "dagger_eval/episode_length_mean": 3.0,
        "dagger_eval/episode_length_std": 1.0,
        "dagger_eval/normalized_waypoint_progress_mean": 0.5,
        "dagger_eval/normalized_waypoint_progress_std": 0.1,
        "dagger_eval/edge_violation_mean": 0.0,
        "dagger_eval/edge_violation_std": 0.3,
    }


def test_log_eval_metrics_uses_training_step_and_run(tmp_path, monkeypatch):
    eval_path = tmp_path / "queue_eval_latency_0.json"
    eval_path.write_text(
        json.dumps(
            {
                "mean_return": 1.0,
                "std_return": 0.2,
                "mean_length": 3.0,
                "lengths": [2.0, 4.0],
                "task_metrics": {
                    "normalized_waypoint_progress": {"mean": 0.5, "std": 0.1},
                    "edge_violation": {"mean": 0.0, "std": 0.3},
                },
            }
        )
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

    dagger.log_eval_metrics(eval_path, step=500, run_id="flow-aux")

    assert calls["init"] == {
        "project": "finetune-gr00t-n1d7",
        "id": "flow-aux",
        "resume": "allow",
    }
    assert calls["metric"] == ("dagger_eval/*", "train/global_step")
    assert calls["log"]["train/global_step"] == 500
    assert calls["log"]["dagger_eval/normalized_waypoint_progress_mean"] == 0.5
