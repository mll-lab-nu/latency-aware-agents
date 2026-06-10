from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from latency_bench.core.types import EpisodeMetrics
import latency_bench.run as latency_run
from latency_bench.run import main


def _write_dummy_config(path: Path, output_dir: Path, num_episodes: int = 1) -> None:
    path.write_text(
        f"""
experiment:
  name: config_default
  seed: 0
executor:
  mode: simulated
env:
  name: dummy
  env_fps: 30
  obs_fps: 30
  noop_action: noop
  action_map:
    noop: noop
    jump: jump
latency:
  type: zero
scheduler:
  hold_policy: one_frame_then_noop
  ordering_policy: latest_ready
policy:
  type: constant
  action: jump
  checkpoint_path: config_checkpoint.pt
evaluation:
  eval_episodes: {num_episodes}
  eval_max_steps: 1
logging:
  output_dir: {output_dir}
  save_step_records: true
  save_action_records: true
  save_latency_records: true
""",
        encoding="utf-8",
    )


def _write_dummy_sweep_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
experiment:
  name: sweep_config
  seed: 0
executor:
  mode: simulated
env:
  name: dummy
  env_fps: 30
  obs_fps: 30
  noop_action: noop
  action_map:
    noop: noop
    jump: jump
latency:
  type: zero
scheduler:
  hold_policy: one_frame_then_noop
  ordering_policy: latest_ready
policy:
  type: constant
  action: jump
  checkpoint_path: config_checkpoint.pt
logging:
  output_dir: {output_dir}
  save_step_records: true
  save_action_records: true
  save_latency_records: true
evaluation:
  eval_episodes: 1
  eval_max_steps: 1
  eval_latency_values:
    - 0
    - 3
""",
        encoding="utf-8",
    )


def _write_dummy_realtime_measured_config(
    path: Path,
    output_dir: Path,
    num_episodes: int = 1,
) -> None:
    path.write_text(
        f"""
experiment:
  name: realtime_default
  seed: 0
executor:
  mode: realtime
env:
  name: dummy
  env_fps: 30
  obs_fps: 30
  noop_action: noop
  action_map:
    noop: noop
    jump: jump
latency:
  device: cpu
  sync_cuda: false
scheduler:
  hold_policy: one_frame_then_noop
  ordering_policy: latest_ready
policy:
  type: constant
  action: jump
evaluation:
  eval_episodes: {num_episodes}
  eval_max_steps: 1
logging:
  output_dir: {output_dir}
  save_step_records: true
  save_action_records: true
  save_latency_records: true
""",
        encoding="utf-8",
    )


def _run_config(policy_type: str, tmp_path: Path, seed: int = 7, num_episodes: int = 3) -> dict:
    return {
        "experiment": {
            "name": "seed_test",
            "seed": seed,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": "dummy",
            "env_fps": 30,
            "obs_fps": 30,
            "noop_action": "noop",
            "action_map": {
                "noop": "noop",
                "jump": "jump",
            },
        },
        "latency": {
            "type": "zero",
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "latest_ready",
        },
        "policy": {
            "type": policy_type,
            "action": "jump",
        },
        "logging": {
            "output_dir": str(tmp_path),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
        "evaluation": {
            "eval_episodes": num_episodes,
            "eval_max_steps": 1,
        },
    }


def test_run_from_config_uses_sample_factory_eval_episode_seeds(monkeypatch, tmp_path: Path):
    episode_seeds = []
    global_seed_calls = []

    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

        def run_episode(self, *, episode_id, seed, eval_max_steps):
            episode_seeds.append(seed)
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=0.0,
                survival_steps=0,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    monkeypatch.setattr(latency_run, "seed_everything", lambda seed: global_seed_calls.append(seed) or seed)
    config = _run_config("deadly_corridor_sf", tmp_path, seed=11, num_episodes=3)

    latency_run.run_from_config(config)

    assert global_seed_calls == [11]
    assert episode_seeds == [104740, 104741, 104742]


def test_run_from_config_keeps_existing_episode_seeds_for_non_sample_factory_policy(
    monkeypatch,
    tmp_path: Path,
):
    episode_seeds = []
    global_seed_calls = []

    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

        def run_episode(self, *, episode_id, seed, eval_max_steps):
            episode_seeds.append(seed)
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=0.0,
                survival_steps=0,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    monkeypatch.setattr(latency_run, "seed_everything", lambda seed: global_seed_calls.append(seed) or seed)
    config = _run_config("constant", tmp_path, seed=11, num_episodes=3)

    latency_run.run_from_config(config)

    assert global_seed_calls == [11]
    assert episode_seeds == [11, 12, 13]


def test_run_from_config_uses_evaluation_eval_max_steps(
    monkeypatch,
    tmp_path: Path,
):
    eval_steps_seen = []

    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

        def run_episode(self, *, episode_id, seed, eval_max_steps):
            eval_steps_seen.append(eval_max_steps)
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=0.0,
                survival_steps=0,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    config = _run_config("constant", tmp_path, seed=11, num_episodes=1)
    config["evaluation"] = {"eval_episodes": 1, "eval_max_steps": 7}

    latency_run.run_from_config(config)

    assert eval_steps_seen == [7]


def test_run_from_config_uses_batched_executor_run_episodes_when_available(
    monkeypatch,
    tmp_path: Path,
):
    run_calls = []

    class FakeEnv:
        def close(self):
            pass

    class FakeBatchedExecutor:
        logger = None
        env = FakeEnv()

        def run_episodes(self, *, episode_ids, seeds, eval_max_steps):
            run_calls.append((list(episode_ids), list(seeds), eval_max_steps))
            return [
                EpisodeMetrics(
                    episode_id=int(episode_id),
                    episode_return=float(episode_id),
                    survival_steps=1,
                    metadata={},
                )
                for episode_id in episode_ids
            ]

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeBatchedExecutor())
    config = _run_config("constant", tmp_path, seed=11, num_episodes=3)
    config["evaluation"]["eval_max_steps"] = 7

    metrics = latency_run.run_from_config(config)

    assert run_calls == [([0, 1, 2], [11, 12, 13], 7)]
    assert [metric.episode_id for metric in metrics] == [0, 1, 2]
    assert [metric.metadata["run_name"] for metric in metrics] == ["seed_test"] * 3


def test_run_from_config_merges_executor_warmup_metadata(
    monkeypatch,
    tmp_path: Path,
):
    run_calls = []
    warmup_calls = []
    logged_metrics = []

    class FakeEnv:
        def close(self):
            pass

    class FakeLogger:
        def log_episode_metrics(self, metrics):
            logged_metrics.append(metrics.episode_id)

        def flush(self):
            pass

    class FakeExecutor:
        env = FakeEnv()

        def __init__(self):
            self.logger = FakeLogger()

        def run_warmup(self, *, eval_max_steps):
            warmup_calls.append(eval_max_steps)
            return {"warmup_status": "stable", "warmup_samples": 200}

        def run_episode(self, *, episode_id, seed, eval_max_steps):
            run_calls.append((episode_id, seed, eval_max_steps))
            if self.logger is not None:
                self.logger.log_episode_metrics(SimpleNamespace(episode_id=episode_id))
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=float(episode_id),
                survival_steps=eval_max_steps,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    config = _run_config("constant", tmp_path, seed=11, num_episodes=2)
    config["evaluation"]["eval_max_steps"] = 10

    metrics = latency_run.run_from_config(config)

    assert run_calls == [
        (0, 11, 10),
        (1, 12, 10),
    ]
    assert warmup_calls == [10]
    assert [metric.episode_id for metric in metrics] == [0, 1]
    assert [metric.metadata["warmup_status"] for metric in metrics] == ["stable", "stable"]
    assert [metric.metadata["warmup_samples"] for metric in metrics] == [200, 200]
    assert logged_metrics == [0, 1]


def test_run_from_config_without_warmup_hook_does_not_warmup(
    monkeypatch,
    tmp_path: Path,
):
    run_calls = []

    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

        def run_episode(self, *, episode_id, seed, eval_max_steps):
            run_calls.append((episode_id, seed, eval_max_steps))
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=0.0,
                survival_steps=eval_max_steps,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    config = _run_config("constant", tmp_path, seed=11, num_episodes=1)
    config["evaluation"]["eval_max_steps"] = 10

    metrics = latency_run.run_from_config(config)

    assert run_calls == [(0, 11, 10)]
    assert [metric.episode_id for metric in metrics] == [0]


def test_run_from_config_records_stratified_replay_videos(tmp_path: Path):
    config = _run_config("constant", tmp_path, seed=7, num_episodes=4)
    config["logging"]["video"] = {
        "enabled": True,
        "num_bins": 2,
    }

    latency_run.run_from_config(config, write_summary=False)

    video_dir = tmp_path / "videos"
    manifest = [
        json.loads(line)
        for line in (video_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 2
    assert [row["bin_index"] for row in manifest] == [0, 1]
    assert [row["episode_return"] for row in manifest] == [1.0, 1.0]
    assert [row["fps"] for row in manifest] == [30.0, 30.0]
    assert all((video_dir / row["video_path"]).is_file() for row in manifest)


def test_run_from_config_requires_video_num_bins_when_enabled(tmp_path: Path):
    config = _run_config("constant", tmp_path, seed=7, num_episodes=1)
    config["logging"]["video"] = {"enabled": True}

    with pytest.raises(KeyError, match="num_bins"):
        latency_run.run_from_config(config, write_summary=False)


def test_run_from_config_requires_evaluation_eval_max_steps(monkeypatch, tmp_path: Path):
    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    config = _run_config("constant", tmp_path, seed=11, num_episodes=1)
    config.pop("evaluation")

    with pytest.raises(ValueError, match="evaluation.eval_max_steps is required"):
        latency_run.run_from_config(config)


def test_run_from_config_is_quiet_by_default(tmp_path: Path, capsys):
    config = _run_config("constant", tmp_path, seed=11, num_episodes=1)

    latency_run.run_from_config(config, write_summary=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_logs_eval_suites_to_original_wandb_run_corrected_namespace(
    monkeypatch,
    tmp_path: Path,
):
    checkpoint_root = (
        tmp_path
        / "checkpoints"
        / "demon_attack_frame_stack_fixed_l2_fs8_seed42"
        / "checkpoint_p0"
    )
    marker_root = checkpoint_root / ".latency_bench_eval_checkpoints"
    marker_root.mkdir(parents=True)
    checkpoint_100 = checkpoint_root / "checkpoint_000000100_000000010000.pth"
    checkpoint_200 = checkpoint_root / "checkpoint_000000200_000000020000.pth"
    checkpoint_100.write_text("", encoding="utf-8")
    checkpoint_200.write_text("", encoding="utf-8")
    (marker_root / f"{checkpoint_100.name}.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_100),
                "eval_phase": "train_periodic",
                "train_env_steps": 100,
            }
        ),
        encoding="utf-8",
    )
    (marker_root / f"{checkpoint_200.name}.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_200),
                "eval_phase": "train_periodic",
                "train_env_steps": 200,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "latency_bench_eval.yaml"
    config_path.write_text(
        f"""
experiment:
  name: demon_attack_frame_stack_fixed_l2_fs8_seed42
  seed: 42
backend:
  train_dir: {tmp_path / "checkpoints"}
executor:
  mode: simulated
env:
  name: demon_attack
  env_fps: 60
  obs_fps: 15
latency:
  type: fixed
scheduler:
  hold_policy: hold
  ordering_policy: latest_ready
policy:
  type: demon_attack_sf
evaluation:
  eval_episodes: 1
  eval_max_steps: 3600
  eval_suites:
    fixed:
      - [2]
logging:
  output_dir: {tmp_path / "eval_outputs"}
  save_step_records: true
  save_action_records: true
  save_latency_records: true
  wandb_project: latency-sensitive-bench
  wandb_group: demon-attack-fs8-fixed_l2
  wandb_name: demon-attack-fs8-fixed_l2-s42
  wandb_user: entity
  wandb_tags:
    - factor_sweep
    - demon-attack
""",
        encoding="utf-8",
    )

    eval_calls = []

    def fake_run_eval_suites_from_config(config, *, progress=False, extra_metadata=None):
        eval_calls.append((config, extra_metadata))
        train_env_steps = int(extra_metadata["train_env_steps"])
        return [
            {
                "suite_name": "fixed_2",
                "latency_type": "fixed_sweep",
                "latency": 2,
                "mean_return": float(train_env_steps),
                "std_return": 1.5,
                "mean_length": 3600.0,
                "completed_episodes": 1,
                "returns": [float(train_env_steps)],
                "lengths": [3600],
                "train_env_steps": train_env_steps,
            }
        ]

    init_calls = []
    log_calls = []
    finished = []
    defined_metrics = []
    fake_run = SimpleNamespace(summary={}, finish=lambda: finished.append(True))
    fake_wandb = SimpleNamespace(
        run=None,
        init=lambda **kwargs: init_calls.append(kwargs) or setattr(fake_wandb, "run", fake_run),
        define_metric=lambda *args, **kwargs: defined_metrics.append((args, kwargs)),
        log=lambda payload: log_calls.append(payload),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(latency_run, "run_eval_suites_from_config", fake_run_eval_suites_from_config)

    latency_run.main(["--config", str(config_path)])

    assert [call[0]["policy"]["checkpoint_path"] for call in eval_calls] == [
        str(checkpoint_100),
        str(checkpoint_200),
    ]
    assert [call[1]["train_env_steps"] for call in eval_calls] == [100, 200]
    assert [call[0]["logging"]["output_dir"] for call in eval_calls] == [
        str(tmp_path / "eval_outputs" / "eval_corrected_replay" / "step_100"),
        str(tmp_path / "eval_outputs" / "eval_corrected_replay" / "step_200"),
    ]
    assert init_calls == [
        {
            "project": "latency-sensitive-bench",
            "entity": "entity",
            "id": "demon-attack-fs8-fixed_l2-s42",
            "name": "demon-attack-fs8-fixed_l2-s42",
            "resume": "allow",
            "group": "demon-attack-fs8-fixed_l2",
            "job_type": None,
            "tags": ["factor_sweep", "demon-attack"],
        },
        {
            "project": "latency-sensitive-bench",
            "entity": "entity",
            "id": "demon-attack-fs8-fixed_l2-s42",
            "name": "demon-attack-fs8-fixed_l2-s42",
            "resume": "allow",
            "group": "demon-attack-fs8-fixed_l2",
            "job_type": None,
            "tags": ["factor_sweep", "demon-attack"],
        },
    ]
    assert ("eval_corrected/*",) in [args for args, _kwargs in defined_metrics]
    assert log_calls == [
        {
            "global_step": 100,
            "eval_corrected/global_steps": 100,
            "eval_corrected/fixed/latency_2/mean_return": 100.0,
            "eval_corrected/fixed/latency_2/std_return": 1.5,
            "eval_corrected/fixed/latency_2/mean_length": 3600.0,
            "eval_corrected/fixed/latency_2/completed_episodes": 1.0,
        },
        {
            "global_step": 200,
            "eval_corrected/global_steps": 200,
            "eval_corrected/fixed/latency_2/mean_return": 200.0,
            "eval_corrected/fixed/latency_2/std_return": 1.5,
            "eval_corrected/fixed/latency_2/mean_length": 3600.0,
            "eval_corrected/fixed/latency_2/completed_episodes": 1.0,
        }
    ]
    assert fake_run.summary["eval_corrected/fixed/latency_2/mean_return"] == 200.0
    assert finished == [True, True]


def test_main_runs_explicit_sample_factory_checkpoint_eval_once(
    monkeypatch,
    tmp_path: Path,
):
    checkpoint_root = (
        tmp_path
        / "checkpoints"
        / "demon_attack_frame_stack_fixed_l2_fs8_seed42"
        / "checkpoint_p0"
    )
    marker_root = checkpoint_root / ".latency_bench_eval_checkpoints"
    marker_root.mkdir(parents=True)
    checkpoint_100 = checkpoint_root / "checkpoint_000000100_000000010000.pth"
    checkpoint_100.write_text("", encoding="utf-8")
    (marker_root / f"{checkpoint_100.name}.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_100),
                "eval_phase": "train_periodic",
                "train_env_steps": 100,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "latency_bench_eval.yaml"
    config_path.write_text(
        f"""
experiment:
  name: demon_attack_frame_stack_fixed_l2_fs8_seed42
  seed: 42
backend:
  train_dir: {tmp_path / "checkpoints"}
  extra_args:
    - --load_checkpoint_kind
    - latest
executor:
  mode: simulated
env:
  name: demon_attack
  env_fps: 60
  obs_fps: 15
latency:
  type: fixed
scheduler:
  hold_policy: hold
  ordering_policy: latest_ready
policy:
  type: demon_attack_sf
  checkpoint_kind: latest
training:
  train_for_env_steps: 250
evaluation:
  eval_episodes: 1
  eval_max_steps: 3600
  eval_suites:
    fixed:
      - [2]
logging:
  output_dir: {tmp_path / "eval_outputs"}
  save_step_records: true
  save_action_records: true
  save_latency_records: true
  wandb_project: latency-sensitive-bench
  wandb_group: demon-attack-fs8-fixed_l2
  wandb_name: demon-attack-fs8-fixed_l2-s42
  wandb_user: entity
  wandb_tags:
    - factor_sweep
    - demon-attack
""",
        encoding="utf-8",
    )

    eval_calls = []

    def fake_run_eval_suites_from_config(config, *, progress=False, extra_metadata=None):
        eval_calls.append((config, progress, extra_metadata))
        return [
            {
                "suite_name": "fixed_2",
                "latency_type": "fixed_sweep",
                "latency": 2,
                "mean_return": 250.0,
                "std_return": 1.5,
                "mean_length": 3600.0,
                "completed_episodes": 1,
                "returns": [250.0],
                "lengths": [3600],
            }
        ]

    log_calls = []
    fake_run = SimpleNamespace(summary={}, finish=lambda: None)
    fake_wandb = SimpleNamespace(
        run=None,
        init=lambda **kwargs: setattr(fake_wandb, "run", fake_run),
        define_metric=lambda *args, **kwargs: None,
        log=lambda payload: log_calls.append(payload),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(latency_run, "run_eval_suites_from_config", fake_run_eval_suites_from_config)

    latency_run.main(["--config", str(config_path)])

    assert len(eval_calls) == 1
    assert eval_calls[0][0]["policy"]["checkpoint_kind"] == "latest"
    assert eval_calls[0][0]["logging"]["output_dir"] == str(tmp_path / "eval_outputs")
    assert eval_calls[0][2] is None
    assert log_calls == [
        {
            "global_step": 250,
            "eval_corrected/global_steps": 250,
            "eval_corrected/fixed/latency_2/mean_return": 250.0,
            "eval_corrected/fixed/latency_2/std_return": 1.5,
            "eval_corrected/fixed/latency_2/mean_length": 3600.0,
            "eval_corrected/fixed/latency_2/completed_episodes": 1.0,
        }
    ]
    assert fake_run.summary["eval_corrected/fixed/latency_2/mean_return"] == 250.0


def test_main_applies_runtime_overrides(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_output_dir = tmp_path / "config_output"
    override_output_dir = tmp_path / "override_output"
    _write_dummy_config(config_path, config_output_dir)

    main(
        [
            "--config",
            str(config_path),
            "--checkpoint-path",
            "override_checkpoint.pt",
            "--run-name",
            "override_run",
            "--output-dir",
            str(override_output_dir),
            "--checkpoint-kind",
            "best",
        ]
    )

    assert not config_output_dir.exists()
    metrics_path = override_output_dir / "episode_metrics.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert metrics["metadata"]["config_name"] == "override_run"
    assert metrics["metadata"]["run_name"] == "override_run"
    assert metrics["metadata"]["checkpoint_path"] == "override_checkpoint.pt"
    assert metrics["metadata"]["checkpoint_kind"] == "best"
    assert metrics["metadata"]["output_dir"] == str(override_output_dir)
    summary = json.loads((override_output_dir / "queue_eval_latency_0.json").read_text(encoding="utf-8"))
    assert summary["checkpoint_kind"] == "best"


def test_main_writes_non_sweep_queue_eval_summary(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    _write_dummy_config(config_path, output_dir, num_episodes=3)

    main(["--config", str(config_path)])

    metrics_path = output_dir / "episode_metrics.jsonl"
    summary_path = output_dir / "queue_eval_latency_0.json"
    results_path = output_dir / "queue_eval_results.jsonl"
    assert metrics_path.is_file()
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 3
    assert summary_path.is_file()
    assert results_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "config_default"
    assert summary["seed"] == 0
    assert summary["latency"] == 0
    assert summary["timestamp_utc"]
    assert summary["mean_return"] == 1.0
    assert summary["std_return"] == 0.0
    assert summary["mean_length"] == 1.0
    assert summary["returns"] == [1.0, 1.0, 1.0]
    assert summary["lengths"] == [1, 1, 1]

    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert results == [summary]
    captured = capsys.readouterr()
    assert "summary latency=0 episodes=3 mean_return=1.000 std_return=0.000 mean_length=1.0" in captured.out
    assert "[bench] run=config_default sweeps=1 episodes_per_sweep=3 total_episode_runs=3" in captured.err
    assert "[bench] sweep 1/1: eval_latency=0" in captured.err
    assert "[bench] sweep 1/1 episode 3/3 done" in captured.err
    assert "[bench] sweep 1/1 complete elapsed=" in captured.err


def test_main_writes_realtime_measured_summary_with_string_latency_label(monkeypatch, tmp_path: Path, capsys):
    import latency_bench.executors.factory as executor_factory
    from latency_bench.executors.realtime import RealtimeLatencyExecutor

    monkeypatch.setattr(executor_factory, "_resolve_realtime_inference_devices", lambda: ["cpu"])
    monkeypatch.setattr(
        RealtimeLatencyExecutor,
        "run_warmup",
        lambda self, *, eval_max_steps: {"warmup_status": "stable", "warmup_samples": 200},
    )
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "realtime_output"
    _write_dummy_realtime_measured_config(config_path, output_dir, num_episodes=2)

    main(["--config", str(config_path)])

    metrics_path = output_dir / "episode_metrics.jsonl"
    summary_path = output_dir / "queue_eval_latency_measured.json"
    results_path = output_dir / "queue_eval_results.jsonl"
    assert metrics_path.is_file()
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 2
    assert (output_dir / "eval_latency_trace.png").is_file()
    assert summary_path.is_file()
    assert results_path.is_file()
    assert not (output_dir / "queue_eval_latency_0.json").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "realtime_default"
    assert summary["seed"] == 0
    assert summary["latency"] == "measured"
    assert summary["mean_return"] == 1.0
    assert summary["std_return"] == 0.0
    assert summary["mean_length"] == 1.0
    assert summary["returns"] == [1.0, 1.0]
    assert summary["lengths"] == [1, 1]

    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert results == [summary]
    captured = capsys.readouterr()
    assert (
        "summary latency=measured episodes=2 mean_return=1.000 std_return=0.000 mean_length=1.0"
        in captured.out
    )
    assert "[bench] sweep 1/1: eval_latency=measured" in captured.err


def test_main_parallel_subproc_simulated_eval_keeps_output_schema(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "subproc_output"
    _write_dummy_config(config_path, output_dir, num_episodes=3)
    config_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace(
            "  eval_max_steps: 1\n",
            "  eval_max_steps: 1\n  eval_parallel_envs: 4\n  latency_bench_env_backend: subproc\n",
        ),
        encoding="utf-8",
    )

    main(["--config", str(config_path)])

    metrics_path = output_dir / "episode_metrics.jsonl"
    step_path = output_dir / "steps.jsonl"
    summary_path = output_dir / "queue_eval_latency_0.json"
    assert metrics_path.is_file()
    assert step_path.is_file()
    assert summary_path.is_file()
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    step = json.loads(step_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [metric["episode_id"] for metric in metrics] == [0, 1, 2]
    assert {"episode_id", "env_step", "sim_time_ms", "active_action", "reward", "done", "info"} <= set(step)
    assert summary["returns"] == [1.0, 1.0, 1.0]


def test_main_runs_eval_latency_sweep_into_separate_output_dirs(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "sweep_output"
    _write_dummy_sweep_config(config_path, output_dir)

    main(["--config", str(config_path)])

    latency_0_path = output_dir / "eval_latency_0" / "episode_metrics.jsonl"
    latency_3_path = output_dir / "eval_latency_3" / "episode_metrics.jsonl"
    assert latency_0_path.is_file()
    assert latency_3_path.is_file()
    assert not (output_dir / "eval_latency_0" / "queue_eval_latency_0.json").exists()
    assert not (output_dir / "eval_latency_0" / "queue_eval_results.jsonl").exists()
    assert not (output_dir / "eval_latency_3" / "queue_eval_latency_3.json").exists()
    assert not (output_dir / "eval_latency_3" / "queue_eval_results.jsonl").exists()

    latency_0_summary_path = output_dir / "queue_eval_latency_0.json"
    latency_3_summary_path = output_dir / "queue_eval_latency_3.json"
    results_path = output_dir / "queue_eval_results.jsonl"
    assert latency_0_summary_path.is_file()
    assert latency_3_summary_path.is_file()
    assert results_path.is_file()

    latency_0_metrics = json.loads(latency_0_path.read_text(encoding="utf-8").splitlines()[0])
    latency_3_metrics = json.loads(latency_3_path.read_text(encoding="utf-8").splitlines()[0])
    latency_0_summary = json.loads(latency_0_summary_path.read_text(encoding="utf-8"))
    latency_3_summary = json.loads(latency_3_summary_path.read_text(encoding="utf-8"))
    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]

    assert latency_0_metrics["metadata"]["train_latency_steps"] == 0
    assert latency_0_metrics["metadata"]["eval_latency_steps"] == 0
    assert latency_0_metrics["metadata"]["configured_latency_ms"] == 0.0
    assert latency_0_metrics["metadata"]["config_name"] == "sweep_config"
    assert latency_0_metrics["metadata"]["run_name"] == "sweep_config"
    assert latency_0_metrics["metadata"]["output_dir"] == str(output_dir / "eval_latency_0")

    assert latency_3_metrics["metadata"]["train_latency_steps"] == 0
    assert latency_3_metrics["metadata"]["eval_latency_steps"] == 3
    assert latency_3_metrics["metadata"]["configured_latency_ms"] == 100.0
    assert latency_3_metrics["metadata"]["config_name"] == "sweep_config"
    assert latency_3_metrics["metadata"]["run_name"] == "sweep_config"
    assert latency_3_metrics["metadata"]["output_dir"] == str(output_dir / "eval_latency_3")

    assert latency_0_summary["train_latency_steps"] == 0
    assert latency_0_summary["eval_latency_steps"] == 0
    assert latency_0_summary["configured_latency_ms"] == 0.0
    assert latency_0_summary["returns"] == [1.0]
    assert latency_0_summary["lengths"] == [1]

    assert latency_3_summary["train_latency_steps"] == 0
    assert latency_3_summary["eval_latency_steps"] == 3
    assert latency_3_summary["configured_latency_ms"] == 100.0
    assert latency_3_summary["returns"] == [1.0]
    assert latency_3_summary["lengths"] == [1]
    assert results == [latency_0_summary, latency_3_summary]

    captured = capsys.readouterr()
    assert "summary latency=0 episodes=1 mean_return=1.000 std_return=0.000 mean_length=1.0" in captured.out
    assert "summary latency=3 episodes=1 mean_return=1.000 std_return=0.000 mean_length=1.0" in captured.out
    assert "[bench] run=sweep_config sweeps=2 episodes_per_sweep=1 total_episode_runs=2" in captured.err
    assert "[bench] sweep 1/2: eval_latency=0" in captured.err
    assert "[bench] sweep 1/2 episode 1/1 done" in captured.err
    assert "[bench] sweep 1/2 complete elapsed=" in captured.err
    assert "[bench] sweep 2/2: eval_latency=3" in captured.err
    assert "[bench] sweep 2/2 episode 1/1 done" in captured.err
    assert "[bench] sweep 2/2 complete elapsed=" in captured.err
