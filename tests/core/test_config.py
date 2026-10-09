from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.core.config import load_config, merge_dicts
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy


BASE_CONFIG = """
experiment:
  name: base
  seed: 0
executor:
  mode: simulated
env:
  name: dummy
  env_fps: 30
  obs_fps: 10
  noop_action: noop
latency:
  method: zero
scheduler:
  hold_policy: hold
  ordering_policy: latest_ready
policy:
  type: random
logging:
  output_dir: outputs/base
"""


def test_merge_dicts_overrides_nested_values():
    merged = merge_dicts(
        {"env": {"env_fps": 30, "name": "base"}, "latency": {"method": "zero"}},
        {"env": {"env_fps": 60}},
    )

    assert merged["env"]["name"] == "base"
    assert merged["env"]["env_fps"] == 60
    assert merged["latency"]["method"] == "zero"


def test_scheduler_public_core_types_are_clean():
    assert {item.value for item in OrderingPolicy} == {"issue_order_fifo", "latest_ready"}
    assert {item.value for item in ActionHoldPolicy} == {"one_frame_then_noop", "hold"}


def test_load_config_without_base_keeps_declared_shape(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(BASE_CONFIG, encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment"]["name"] == "base"
    assert config["env"]["env_fps"] == 30
    assert config["env"]["obs_fps"] == 10


def test_load_config_merges_single_base(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(BASE_CONFIG, encoding="utf-8")
    child_path.write_text(
        """
__base__: base.yaml
experiment:
  name: child
env:
  obs_fps: 30
scheduler:
  hold_policy: one_frame_then_noop
""",
        encoding="utf-8",
    )

    config = load_config(child_path)

    assert config["experiment"]["name"] == "child"
    assert config["env"]["env_fps"] == 30
    assert config["env"]["obs_fps"] == 30
    assert config["scheduler"]["hold_policy"] == "one_frame_then_noop"


def test_load_config_merges_multiple_bases_left_to_right(tmp_path: Path):
    base_a = tmp_path / "base_a.yaml"
    base_b = tmp_path / "base_b.yaml"
    child = tmp_path / "child.yaml"
    base_a.write_text(BASE_CONFIG, encoding="utf-8")
    base_b.write_text(
        """
env:
  env_fps: 60
  obs_fps: 15
policy:
  type: constant
""",
        encoding="utf-8",
    )
    child.write_text(
        """
__base__:
  - base_a.yaml
  - base_b.yaml
policy:
  action: noop
""",
        encoding="utf-8",
    )

    config = load_config(child)

    assert config["env"]["env_fps"] == 60
    assert config["env"]["obs_fps"] == 15
    assert config["policy"] == {"type": "constant", "action": "noop"}


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(config_path)


def test_load_config_rejects_invalid_base_reference_type(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("__base__: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="__base__"):
        load_config(config_path)
