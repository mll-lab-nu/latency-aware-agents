from __future__ import annotations

import json
from pathlib import Path

from latency_bench.core.types import EpisodeMetrics
import latency_bench.run as latency_run
from latency_bench.run import main


def _write_dummy_config(path: Path, output_dir: Path, num_episodes: int = 1) -> None:
    path.write_text(
        f"""
experiment:
  name: config_default
  seed: 0
  num_episodes: {num_episodes}
  max_env_steps: 1
executor:
  mode: simulated
env:
  name: dummy
  fps_env: 30
  noop_action: noop
  action_map:
    noop: noop
    jump: jump
control:
  frequency_hz: 30
  observe_every_n_frames: 1
latency:
  type: zero
scheduler:
  persistence_mode: noop_after_oneshot
  stale_action_mode: latest_ready
policy:
  type: constant
  action: jump
  checkpoint_path: config_checkpoint.pt
logging:
  output_dir: {output_dir}
""",
        encoding="utf-8",
    )


def _write_dummy_sweep_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
experiment:
  name: sweep_config
  seed: 0
  num_episodes: 1
  max_env_steps: 1
executor:
  mode: simulated
env:
  name: dummy
  fps_env: 30
  noop_action: noop
  action_map:
    noop: noop
    jump: jump
control:
  frequency_hz: 30
  observe_every_n_frames: 1
latency:
  type: zero
  train_values:
    - 0
scheduler:
  persistence_mode: noop_after_oneshot
  stale_action_mode: latest_ready
policy:
  type: constant
  action: jump
  checkpoint_path: config_checkpoint.pt
logging:
  output_dir: {output_dir}
evaluation:
  eval_latency_values:
    - 0
    - 3
""",
        encoding="utf-8",
    )


def _run_config(policy_type: str, tmp_path: Path, seed: int = 7, num_episodes: int = 3) -> dict:
    return {
        "experiment": {
            "name": "seed_test",
            "seed": seed,
            "num_episodes": num_episodes,
            "max_env_steps": 1,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": "dummy",
            "fps_env": 30,
            "noop_action": "noop",
            "action_map": {
                "noop": "noop",
                "jump": "jump",
            },
        },
        "control": {
            "frequency_hz": 30,
            "observe_every_n_frames": 1,
        },
        "latency": {
            "type": "zero",
        },
        "scheduler": {
            "persistence_mode": "noop_after_oneshot",
            "stale_action_mode": "latest_ready",
        },
        "policy": {
            "type": policy_type,
            "action": "jump",
        },
        "logging": {
            "output_dir": str(tmp_path),
        },
    }


def test_run_from_config_uses_sample_factory_eval_episode_seeds(monkeypatch, tmp_path: Path):
    from training.sample_factory_common import common

    episode_seeds = []
    global_seed_calls = []

    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

        def run_episode(self, *, episode_id, seed, max_env_steps):
            episode_seeds.append(seed)
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=0.0,
                survival_steps=0,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    monkeypatch.setattr(common, "seed_everything", lambda seed: global_seed_calls.append(seed) or seed)
    config = _run_config("deadly_corridor_sf", tmp_path, seed=11, num_episodes=3)

    latency_run.run_from_config(config)

    assert global_seed_calls == [11]
    assert episode_seeds == [104740, 104741, 104742]


def test_run_from_config_keeps_existing_episode_seeds_for_non_sample_factory_policy(
    monkeypatch,
    tmp_path: Path,
):
    episode_seeds = []

    class FakeEnv:
        def close(self):
            pass

    class FakeExecutor:
        logger = None
        env = FakeEnv()

        def run_episode(self, *, episode_id, seed, max_env_steps):
            episode_seeds.append(seed)
            return EpisodeMetrics(
                episode_id=episode_id,
                episode_return=0.0,
                survival_steps=0,
                metadata={},
            )

    monkeypatch.setattr(latency_run, "build_executor", lambda config: FakeExecutor())
    config = _run_config("constant", tmp_path, seed=11, num_episodes=3)

    latency_run.run_from_config(config)

    assert episode_seeds == [11, 12, 13]


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
        ]
    )

    assert not config_output_dir.exists()
    metrics_path = override_output_dir / "episode_metrics.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert metrics["metadata"]["config_name"] == "override_run"
    assert metrics["metadata"]["run_name"] == "override_run"
    assert metrics["metadata"]["checkpoint_path"] == "override_checkpoint.pt"
    assert metrics["metadata"]["output_dir"] == str(override_output_dir)


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
    assert "summary latency=0 episodes=3 mean_return=1.000 std_return=0.000 mean_length=1.0" in (
        capsys.readouterr().out
    )


def test_main_runs_eval_latency_sweep_into_separate_output_dirs(tmp_path: Path):
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
