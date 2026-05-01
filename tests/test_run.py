from __future__ import annotations

import json
from pathlib import Path

from latency_bench.run import main


def _write_dummy_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
experiment:
  name: config_default
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
