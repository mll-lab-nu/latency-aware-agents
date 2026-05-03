from __future__ import annotations

import json
from types import SimpleNamespace


def test_demon_attack_eval_done_ignores_life_loss_pseudo_terminal():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "is_true_episode_end": False,
            "episodic_raw_return": 30.0,
            "episodic_clipped_return": 2.0,
            "episodic_return": 30.0,
            "episodic_length": 3,
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result is None


def test_demon_attack_eval_done_uses_episode_metrics_wrapper_keys_for_raw_scoring():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "is_true_episode_end": True,
            "episodic_raw_return": 30.0,
            "episodic_clipped_return": 2.0,
            "episodic_return": 30.0,
            "episodic_length": 3,
            "episode_extra_stats": {
                "episode/raw_return": 30.0,
                "episode/clipped_return": 2.0,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result == {
        "return": 30.0,
        "raw_return": 30.0,
        "clipped_return": 2.0,
        "length": 3,
    }


def test_demon_attack_eval_done_accepts_lives_zero_game_over_with_extra_stats():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "lives": 0,
            "episode_extra_stats": {
                "episode/raw_return": 50.0,
                "episode/clipped_return": 4.0,
                "episode/length": 8,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=False,
    )

    assert result == {
        "return": 4.0,
        "raw_return": 50.0,
        "clipped_return": 4.0,
        "length": 8,
    }


def test_demon_attack_eval_done_accepts_timelimit_truncated_timeout_with_extra_stats():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "TimeLimit.truncated": True,
            "episode_extra_stats": {
                "episode/raw_return": 12.0,
                "episode/clipped_return": 3.0,
                "episode/length": 4,
                "episode/truncated": 1.0,
                "episode/timeout": 1.0,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result == {
        "return": 12.0,
        "raw_return": 12.0,
        "clipped_return": 3.0,
        "length": 4,
        "timeout": True,
    }


def test_demon_attack_timeout_count_uses_eval_budget_not_short_truncated_info():
    from training.sample_factory_common.queue_eval import (
        _demon_attack_eval_episode_result,
        _demon_attack_eval_step_budget_timeout,
    )

    short_truncated = _demon_attack_eval_episode_result(
        {
            "TimeLimit.truncated": True,
            "episode_extra_stats": {
                "episode/raw_return": 12.0,
                "episode/clipped_return": 3.0,
                "episode/length": 4,
                "episode/truncated": 1.0,
                "episode/timeout": 1.0,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )
    assert short_truncated is not None

    assert (
        _demon_attack_eval_step_budget_timeout(
            short_truncated,
            eval_max_steps=3600,
            timed_out_by_eval_budget=False,
        )
        is False
    )
    assert (
        _demon_attack_eval_step_budget_timeout(
            short_truncated,
            eval_max_steps=4,
            timed_out_by_eval_budget=True,
        )
        is True
    )


def test_queue_eval_result_keeps_raw_and_clipped_fields_with_raw_compat_returns():
    from training.sample_factory_common.queue_eval import _build_queue_eval_result

    result = _build_queue_eval_result(
        latency_value=5,
        returns=[30.0, 50.0],
        lengths=[3, 8],
        eval_episodes=2,
        eval_max_steps=3600,
        total_frames=11,
        hit_eval_step_budget=False,
        raw_returns=[30.0, 50.0],
        clipped_returns=[2.0, 4.0],
        eval_raw_reward=True,
        env_name="latency_demon_attack",
    )

    assert result["returns"] == [30.0, 50.0]
    assert result["mean_return"] == 40.0
    assert result["raw_returns"] == [30.0, 50.0]
    assert result["clipped_returns"] == [2.0, 4.0]
    assert result["mean_raw_return"] == 40.0
    assert result["mean_clipped_return"] == 3.0
    assert result["std_raw_return"] == 10.0
    assert result["std_clipped_return"] == 1.0
    assert result["completed_episodes"] == 2
    assert "warning" not in result


def test_queue_eval_result_reports_timeout_episodes_without_missing_game_over_warning():
    from training.sample_factory_common.queue_eval import _build_queue_eval_result

    result = _build_queue_eval_result(
        latency_value=5,
        returns=[12.0],
        lengths=[3600],
        eval_episodes=1,
        eval_max_steps=3600,
        total_frames=3600,
        hit_eval_step_budget=True,
        raw_returns=[12.0],
        clipped_returns=[3.0],
        eval_raw_reward=True,
        env_name="latency_demon_attack",
        timeout_episodes=1,
    )

    assert result["returns"] == [12.0]
    assert result["raw_returns"] == [12.0]
    assert result["clipped_returns"] == [3.0]
    assert result["completed_episodes"] == 1
    assert result["requested_episodes"] == 1
    assert result["hit_eval_step_budget"] is True
    assert result["timeout_episodes"] == 1
    assert "warning" not in result
    assert result["mean_return"] == 12.0
    assert result["mean_raw_return"] == 12.0
    assert result["mean_clipped_return"] == 3.0


def test_prepare_queue_eval_cfg_uses_eval_max_steps_as_eval_only_timelimit():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_flappy",
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.max_episode_steps == 123
    assert cfg.cli_args["max_episode_steps"] == 123


def test_prepare_queue_eval_cfg_does_not_convert_demon_attack_eval_budget_to_timelimit():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_demon_attack",
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.max_episode_steps == 0
    assert "max_episode_steps" not in cfg.cli_args


def test_write_and_log_queue_eval_result_preserve_raw_and_clipped_metrics(tmp_path):
    from training.sample_factory_common.queue_eval import log_queue_eval_result, write_queue_eval_result

    cfg = SimpleNamespace(output_dir=str(tmp_path), experiment="exp", seed=7, load_checkpoint_kind="best")
    result = {
        "latency": 5,
        "mean_return": 40.0,
        "std_return": 10.0,
        "mean_length": 5.5,
        "returns": [30.0, 50.0],
        "lengths": [3, 8],
        "raw_returns": [30.0, 50.0],
        "clipped_returns": [2.0, 4.0],
        "mean_raw_return": 40.0,
        "mean_clipped_return": 3.0,
        "std_raw_return": 10.0,
        "std_clipped_return": 1.0,
        "completed_episodes": 2,
        "timeout_episodes": 1,
        "requested_episodes": 2,
    }

    write_queue_eval_result(cfg, result)
    payload = json.loads((tmp_path / "queue_eval_latency_5.json").read_text(encoding="utf-8"))

    assert payload["raw_returns"] == [30.0, 50.0]
    assert payload["checkpoint_kind"] == "best"
    assert payload["clipped_returns"] == [2.0, 4.0]
    assert payload["mean_raw_return"] == 40.0
    assert payload["mean_clipped_return"] == 3.0
    assert payload["timeout_episodes"] == 1

    class _Run:
        def __init__(self):
            self.logged = None
            self.summary = {}

        def log(self, payload):
            self.logged = payload

    run = _Run()
    log_queue_eval_result(run, result)

    assert run.logged["queue_eval/mean_raw_return"] == 40.0
    assert run.logged["queue_eval/mean_clipped_return"] == 3.0
    assert run.logged["queue_eval/timeout_episodes"] == 1.0
    assert run.summary["queue_eval/latency_5/mean_raw_return"] == 40.0
    assert run.summary["queue_eval/latency_5/timeout_episodes"] == 1.0
