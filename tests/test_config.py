from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.core.config import load_config, merge_dicts


BASE_CONFIG = """
experiment:
  name: base
  seed: 0
executor:
  mode: simulated
env:
  name: dummy
  fps_env: 30
  noop_action: noop
control:
  frequency_hz: 30
  observe_every_n_frames: 1
latency:
  type: zero
scheduler:
  persistence_mode: noop_after_oneshot
  stale_action_mode: latest_ready
policy:
  type: random
logging:
  output_dir: outputs/base
"""


def test_merge_dicts_overrides_nested_values():
    merged = merge_dicts(
        {"env": {"fps_env": 30, "name": "base"}, "latency": {"type": "zero"}},
        {"env": {"fps_env": 60}},
    )

    assert merged["env"]["name"] == "base"
    assert merged["env"]["fps_env"] == 60
    assert merged["latency"]["type"] == "zero"


def test_load_config_without_base(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(BASE_CONFIG, encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment"]["name"] == "base"
    assert config["env"]["fps_env"] == 30


def test_load_config_with_one_base(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(BASE_CONFIG, encoding="utf-8")
    child_path.write_text(
        """
__base__: base.yaml
experiment:
  name: child
latency:
  type: fixed
  fixed_latency_ms: 100
logging:
  output_dir: outputs/child
""",
        encoding="utf-8",
    )

    config = load_config(child_path)

    assert config["experiment"]["name"] == "child"
    assert config["env"]["fps_env"] == 30
    assert config["latency"]["type"] == "fixed"
    assert config["latency"]["fixed_latency_ms"] == 100


def test_load_config_with_nested_base(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    mid_path = tmp_path / "mid.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(BASE_CONFIG, encoding="utf-8")
    mid_path.write_text(
        """
__base__: base.yaml
env:
  fps_env: 60
""",
        encoding="utf-8",
    )
    child_path.write_text(
        """
__base__: mid.yaml
control:
  frequency_hz: 15
logging:
  output_dir: outputs/nested
""",
        encoding="utf-8",
    )

    config = load_config(child_path)

    assert config["env"]["fps_env"] == 60
    assert config["env"]["name"] == "dummy"
    assert config["control"]["frequency_hz"] == 15


def test_load_config_validates_executor_mode(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(BASE_CONFIG.replace("mode: simulated", "mode: unknown"), encoding="utf-8")

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "executor.mode" in str(exc)
    else:
        raise AssertionError("invalid executor.mode should fail validation")


def test_load_config_accepts_normal_distribution_latency(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG.replace(
            "latency:\n  type: zero",
            """
latency:
  type: normal_distribution
  mean_latency_ms: 100
  std_latency_ms: 20
  seed: 0
  min_latency_ms: 0
  max_latency_ms: 200
""".strip(),
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["latency"]["type"] == "normal_distribution"
    assert config["latency"]["mean_latency_ms"] == 100
    assert config["latency"]["std_latency_ms"] == 20


def test_load_config_requires_normal_distribution_mean_and_std(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG.replace(
            "latency:\n  type: zero",
            """
latency:
  type: normal_distribution
  mean_latency_ms: 100
""".strip(),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latency.std_latency_ms"):
        load_config(config_path)


def test_load_config_rejects_empirical_distribution_latency(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG.replace(
            "latency:\n  type: zero",
            """
latency:
  type: empirical_distribution
  latency_values_ms: [1, 2, 3]
""".strip(),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latency.type"):
        load_config(config_path)


def test_load_config_rejects_negative_normal_distribution_std(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG.replace(
            "latency:\n  type: zero",
            """
latency:
  type: normal_distribution
  mean_latency_ms: 100
  std_latency_ms: -1
""".strip(),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latency.std_latency_ms"):
        load_config(config_path)


def test_load_config_rejects_nonpositive_realtime_max_pending_observations(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG.replace(
            "executor:\n  mode: simulated",
            """
executor:
  mode: realtime
  max_pending_observations: 0
""".strip(),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="executor.max_pending_observations"):
        load_config(config_path)


def test_load_config_rejects_nonpositive_realtime_max_catchup_steps(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG
        + """
realtime:
  max_catchup_steps_per_loop: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="realtime.max_catchup_steps_per_loop"):
        load_config(config_path)


def test_load_config_rejects_invalid_latency_bench_env_backend(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        BASE_CONFIG
        + """
evaluation:
  eval_max_steps: 1
  latency_bench_env_backend: invalid
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latency_bench_env_backend"):
        load_config(config_path)
