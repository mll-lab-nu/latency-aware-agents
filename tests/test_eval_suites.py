from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from latency_bench.core.types import EpisodeMetrics


def _suite_config(tmp_path: Path) -> dict:
    return {
        "experiment": {
            "name": "suite_config",
            "seed": 0,
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
            "type": "constant",
            "action": "jump",
        },
        "logging": {
            "output_dir": str(tmp_path),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
        "evaluation": {
            "eval_episodes": 1,
            "eval_max_steps": 1,
            "eval_suites": {
                "fixed": [
                    [0, 2],
                ],
                "normal": [
                    [2.0, 1.0, 0, 15],
                    [4.0, 2.0, 0, 15],
                ],
                "uniform": [
                    [4, 12],
                ],
            },
        },
    }


def test_parse_grouped_eval_suites_supports_multiple_normal_distributions(tmp_path: Path):
    from latency_bench.eval_suites import parse_eval_suites

    suites = parse_eval_suites(_suite_config(tmp_path))

    assert [suite.name for suite in suites] == [
        "fixed_0_to_2",
        "normal_mean_2_std_1_min_0_max_15",
        "normal_mean_4_std_2_min_0_max_15",
        "uniform_min_4_max_12",
    ]
    assert suites[0].fixed_latency_values == [0, 1, 2]
    assert suites[1].latency_config == {
        "type": "normal_distribution",
        "mean_latency_ms": pytest.approx(2.0 * 1000.0 / 30.0),
        "std_latency_ms": pytest.approx(1.0 * 1000.0 / 30.0),
        "min_latency_ms": 0.0,
        "max_latency_ms": pytest.approx(15.0 * 1000.0 / 30.0),
        "seed": 0,
    }
    assert suites[2].latency_config["mean_latency_ms"] == pytest.approx(4.0 * 1000.0 / 30.0)
    assert suites[3].latency_config == {
        "type": "uniform_distribution",
        "min_latency_ms": pytest.approx(4.0 * 1000.0 / 30.0),
        "max_latency_ms": pytest.approx(12.0 * 1000.0 / 30.0),
        "seed": 0,
    }


def test_parse_fixed_eval_suite_supports_explicit_latency_list(tmp_path: Path):
    from latency_bench.eval_suites import parse_eval_suites

    config = _suite_config(tmp_path)
    config["evaluation"]["eval_suites"] = {
        "fixed": [
            [0, 1, 2, 3, 4],
        ],
    }

    suites = parse_eval_suites(config)

    assert [suite.name for suite in suites] == ["fixed_0_1_2_3_4"]
    assert suites[0].fixed_latency_values == [0, 1, 2, 3, 4]


def test_run_eval_suites_writes_per_suite_outputs(monkeypatch, tmp_path: Path):
    import latency_bench.run as latency_run

    seen_configs: list[dict] = []

    def fake_run_from_config(config, extra_metadata=None, write_summary=True, on_episode_complete=None):
        seen_configs.append(config)
        metadata = dict(extra_metadata or {})
        metadata["output_dir"] = config["logging"]["output_dir"]
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0 + len(seen_configs),
                survival_steps=5,
                metadata=metadata,
            )
        ]

    monkeypatch.setattr(latency_run, "run_from_config", fake_run_from_config)

    summaries = latency_run.run_eval_suites_from_config(_suite_config(tmp_path), progress=False)

    assert [summary["suite_name"] for summary in summaries] == [
        "fixed_0_to_2",
        "fixed_0_to_2",
        "fixed_0_to_2",
        "normal_mean_2_std_1_min_0_max_15",
        "normal_mean_4_std_2_min_0_max_15",
        "uniform_min_4_max_12",
    ]
    assert [summary["latency"] for summary in summaries] == [
        0,
        1,
        2,
        "normal_mean_2_std_1_min_0_max_15",
        "normal_mean_4_std_2_min_0_max_15",
        "uniform_min_4_max_12",
    ]
    assert seen_configs[0]["logging"]["output_dir"] == str(tmp_path / "fixed_0_to_2" / "eval_latency_0")
    assert seen_configs[3]["logging"]["output_dir"] == str(tmp_path / "normal_mean_2_std_1_min_0_max_15")
    assert seen_configs[3]["latency"]["type"] == "normal_distribution"
    assert seen_configs[5]["latency"]["type"] == "uniform_distribution"
    assert (tmp_path / "fixed_0_to_2" / "queue_eval_latency_0.json").is_file()
    normal_summary_path = (
        tmp_path
        / "normal_mean_2_std_1_min_0_max_15"
        / "queue_eval_latency_normal_mean_2_std_1_min_0_max_15.json"
    )
    assert normal_summary_path.is_file()

    normal_summary = json.loads(normal_summary_path.read_text(encoding="utf-8"))
    assert normal_summary["suite_name"] == "normal_mean_2_std_1_min_0_max_15"
    assert normal_summary["latency_type"] == "normal_distribution"


def test_cli_prints_eval_suite_summaries(monkeypatch, tmp_path: Path, capsys):
    import latency_bench.run as latency_run

    def fake_run_from_config(config, extra_metadata=None, write_summary=True, on_episode_complete=None):
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0,
                survival_steps=5,
                metadata=dict(extra_metadata or {}),
            )
        ]

    monkeypatch.setattr(latency_run, "run_from_config", fake_run_from_config)
    config_path = tmp_path / "eval_suites.yaml"
    config_path.write_text(yaml.safe_dump(_suite_config(tmp_path)), encoding="utf-8")

    latency_run.main(["--config", str(config_path)])

    output = capsys.readouterr().out
    assert "summary suite=fixed_0_to_2 latency=0" in output
    assert "summary suite=normal_mean_2_std_1_min_0_max_15 latency=normal_mean_2_std_1_min_0_max_15" in output
