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
            "method": "zero",
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


@pytest.mark.parametrize("raw_reward", [None, True])
def test_eval_raw_reward_remains_opt_in_when_resolving_config(tmp_path, raw_reward):
    from latency_bench.eval.config import resolve_evaluation_config

    config = _suite_config(tmp_path)
    if raw_reward is not None:
        config["evaluation"]["eval_raw_reward"] = raw_reward
    resolve_evaluation_config(config)
    assert config["evaluation"]["eval_raw_reward"] is (raw_reward is True)


def test_parse_grouped_eval_suites_supports_multiple_normal_distributions(tmp_path: Path):
    from latency_bench.eval.suites import parse_eval_suites

    suites = parse_eval_suites(_suite_config(tmp_path))

    assert [suite.name for suite in suites] == [
        "fixed_0_to_2",
        "normal_mean_2_std_1_min_0_max_15",
        "normal_mean_4_std_2_min_0_max_15",
        "uniform_min_4_max_12",
    ]
    assert suites[0].fixed_latency_values == [0, 1, 2]
    assert suites[1].latency_config == {
        "method": "normal",
        "mean_latency_ms": pytest.approx(2.0 * 1000.0 / 30.0),
        "std_latency_ms": pytest.approx(1.0 * 1000.0 / 30.0),
        "min_latency_ms": 0.0,
        "max_latency_ms": pytest.approx(15.0 * 1000.0 / 30.0),
        "seed": 0,
    }
    assert suites[2].latency_config["mean_latency_ms"] == pytest.approx(4.0 * 1000.0 / 30.0)
    assert suites[3].latency_config == {
        "method": "uniform",
        "min_latency_ms": pytest.approx(4.0 * 1000.0 / 30.0),
        "max_latency_ms": pytest.approx(12.0 * 1000.0 / 30.0),
        "seed": 0,
    }


def test_parse_fixed_eval_suite_supports_explicit_latency_list(tmp_path: Path):
    from latency_bench.eval.suites import parse_eval_suites

    config = _suite_config(tmp_path)
    config["evaluation"]["eval_suites"] = {
        "fixed": [
            [0, 1, 2, 3, 4],
        ],
    }

    suites = parse_eval_suites(config)

    assert [suite.name for suite in suites] == ["fixed_0_1_2_3_4"]
    assert suites[0].fixed_latency_values == [0, 1, 2, 3, 4]


def test_empty_eval_suite_mapping_uses_single_run(tmp_path: Path):
    from latency_bench.eval.suites import has_eval_suites

    config = _suite_config(tmp_path)
    config["evaluation"]["eval_suites"] = {
        "fixed": [],
        "normal": [],
        "uniform": [],
    }

    assert not has_eval_suites(config)


@pytest.mark.parametrize("method", ["iid", "temporal"])
def test_parse_profile_eval_suites_uses_explicit_name_and_profile_provenance(tmp_path: Path, method):
    from latency_bench.eval.suites import parse_eval_suites

    config = _suite_config(tmp_path)
    profile_path = tmp_path / "profiles" / "rtx_3090" / "profile.json"
    config["evaluation"]["eval_suites"] = {
        method: [
            {
                "name": "rtx_3090",
                "profile_path": str(profile_path),
                "seed": 42,
            }
        ],
    }

    suites = parse_eval_suites(config)

    assert [suite.name for suite in suites] == ["rtx_3090"]
    assert suites[0].latency_config == {
        "method": method,
        "profile_path": str(profile_path),
        "seed": 42,
    }
    assert suites[0].latency_label == "rtx_3090"
    assert suites[0].metadata == {
        "suite_name": "rtx_3090",
        "latency_type": "profile_sample",
        "latency_seed": 42,
    }


def test_iid_suite_summary_retains_profile_provenance(monkeypatch, tmp_path: Path):
    import latency_bench.eval.sweeps as eval_sweeps

    config = _suite_config(tmp_path)
    profile_path = tmp_path / "profiles" / "rtx_3090" / "profile.json"
    config["evaluation"]["eval_suites"] = {
        "iid": [
            {
                "name": "rtx_3090",
                "profile_path": str(profile_path),
                "seed": 42,
            }
        ],
    }

    def fake_run_from_config(
        config,
        extra_metadata=None,
        write_summary=True,
        on_episode_complete=None,
        episode_ids=None,
    ):
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0,
                survival_steps=5,
                metadata=dict(extra_metadata or {}),
            )
        ]

    monkeypatch.setattr(eval_sweeps, "run_from_config", fake_run_from_config)

    summaries = eval_sweeps.run_eval_suites_from_config(config, progress=False)

    assert summaries[0]["latency_type"] == "profile_sample"
    assert "profile_path" not in summaries[0]
    assert summaries[0]["latency_seed"] == 42
    summary_path = tmp_path / "rtx_3090" / "queue_eval_latency_rtx_3090.json"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["latency_seed"] == 42


@pytest.mark.parametrize("policy_type", ["flappy_sf", "starvla"])
def test_checkpoint_eval_suites_build_policy_once_and_reuse_it(
    monkeypatch,
    tmp_path: Path,
    policy_type: str,
):
    import latency_bench.eval.sweeps as eval_sweeps

    config = _suite_config(tmp_path)
    config["policy"] = {
        "type": policy_type,
        "checkpoint_path": str(tmp_path / "checkpoint.pth"),
    }
    config["evaluation"]["eval_suites"] = {
        "iid": [
            {
                "name": "rtx_3090",
                "profile_path": str(tmp_path / "profile.json"),
                "seed": 42,
            },
            {
                "name": "rtx_4090",
                "profile_path": str(tmp_path / "profile.json"),
                "seed": 42,
            },
        ],
    }
    built_policies: list[object] = []
    prepared_configs: list[dict] = []
    used_policies: list[object] = []

    def fake_build_policy(config, action_resolver):
        policy = object()
        built_policies.append(policy)
        return policy

    def fake_run_from_config(
        config,
        extra_metadata=None,
        write_summary=True,
        on_episode_complete=None,
        episode_ids=None,
        policy=None,
    ):
        used_policies.append(policy)
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0,
                survival_steps=5,
                metadata=dict(extra_metadata or {}),
            )
        ]

    monkeypatch.setattr(eval_sweeps, "build_policy", fake_build_policy)
    monkeypatch.setattr(eval_sweeps, "run_from_config", fake_run_from_config)
    if policy_type == "starvla":
        import latency_bench.policy.starvla as starvla

        monkeypatch.setattr(
            starvla,
            "prepare_starvla_checkpoint_input_config",
            prepared_configs.append,
        )

    eval_sweeps.run_eval_suites_from_config(config, progress=False)

    assert len(built_policies) == 1
    assert prepared_configs == ([config] if policy_type == "starvla" else [])
    assert used_policies == [built_policies[0], built_policies[0]]


def test_checkpoint_eval_suites_preserve_explicit_policy(monkeypatch, tmp_path: Path):
    import latency_bench.eval.sweeps as eval_sweeps

    config = _suite_config(tmp_path)
    config["policy"] = {
        "type": "flappy_sf",
        "checkpoint_path": str(tmp_path / "checkpoint.pth"),
    }
    config["evaluation"]["eval_suites"] = {
        "iid": [
            {
                "name": "rtx_3090",
                "profile_path": str(tmp_path / "profile.json"),
                "seed": 42,
            }
        ],
    }
    injected_policy = object()
    used_policies: list[object] = []

    def fake_run_from_config(config, *, extra_metadata, policy, **kwargs):
        used_policies.append(policy)
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0,
                survival_steps=5,
                metadata=dict(extra_metadata),
            )
        ]

    def fail_build_policy(config, action_resolver):
        raise AssertionError("explicit suite policy must not be rebuilt")

    monkeypatch.setattr(eval_sweeps, "build_policy", fail_build_policy)
    monkeypatch.setattr(eval_sweeps, "run_from_config", fake_run_from_config)

    eval_sweeps.run_eval_suites_from_config(
        config,
        progress=False,
        policy=injected_policy,
    )

    assert used_policies == [injected_policy]


@pytest.mark.parametrize(
    "executor_config",
    [
        {"mode": "realtime"},
        {"mode": "simulated", "simulated_inference_pool": True},
    ],
)
def test_checkpoint_eval_suites_leave_subprocess_policy_construction_to_executor(
    monkeypatch,
    tmp_path: Path,
    executor_config: dict,
):
    import latency_bench.eval.sweeps as eval_sweeps

    config = _suite_config(tmp_path)
    config["executor"] = executor_config
    config["policy"] = {
        "type": "flappy_sf",
        "checkpoint_path": str(tmp_path / "checkpoint.pth"),
    }
    config["evaluation"]["eval_suites"] = {"fixed": [[0]]}
    built_policies: list[object] = []
    used_policies: list[object | None] = []

    def fake_build_policy(config, action_resolver):
        built_policies.append(object())

    def fake_run_from_config(config, *, extra_metadata, policy=None, **kwargs):
        used_policies.append(policy)
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0,
                survival_steps=5,
                metadata=dict(extra_metadata),
            )
        ]

    monkeypatch.setattr(eval_sweeps, "build_policy", fake_build_policy)
    monkeypatch.setattr(eval_sweeps, "run_from_config", fake_run_from_config)

    eval_sweeps.run_eval_suites_from_config(config, progress=False)

    assert built_policies == []
    assert used_policies == [None]


def test_run_eval_suites_writes_per_suite_outputs(monkeypatch, tmp_path: Path):
    import latency_bench.eval.sweeps as eval_sweeps

    seen_configs: list[dict] = []

    def fake_run_from_config(config, extra_metadata=None, write_summary=True, on_episode_complete=None, episode_ids=None):
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

    monkeypatch.setattr(eval_sweeps, "run_from_config", fake_run_from_config)

    summaries = eval_sweeps.run_eval_suites_from_config(_suite_config(tmp_path), progress=False)

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
    assert seen_configs[3]["latency"]["method"] == "normal"
    assert seen_configs[5]["latency"]["method"] == "uniform"
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
    import latency_bench.eval.sweeps as eval_sweeps

    def fake_run_from_config(config, extra_metadata=None, write_summary=True, on_episode_complete=None, episode_ids=None):
        return [
            EpisodeMetrics(
                episode_id=0,
                episode_return=10.0,
                survival_steps=5,
                metadata=dict(extra_metadata or {}),
            )
        ]

    monkeypatch.setattr(eval_sweeps, "run_from_config", fake_run_from_config)
    config_path = tmp_path / "eval_suites.yaml"
    config_path.write_text(yaml.safe_dump(_suite_config(tmp_path)), encoding="utf-8")

    latency_run.main(["--config", str(config_path)])

    output = capsys.readouterr().out
    assert "summary suite=fixed_0_to_2 latency=0" in output
    assert "summary suite=normal_mean_2_std_1_min_0_max_15 latency=normal_mean_2_std_1_min_0_max_15" in output
