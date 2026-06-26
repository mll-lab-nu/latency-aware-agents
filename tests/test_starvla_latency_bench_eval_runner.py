from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from latency_bench.core.types import EpisodeMetrics
from latency_bench.integrations import starvla_rl_games_eval_runner as runner_mod
from latency_bench.integrations.starvla_rl_games_eval_runner import (
    LatencyBenchRlGamesEvalRunner,
    build_latency_bench_config_for_rl_games,
)


def test_translation_flappy_fixed_latency(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        seed=42,
        image_size=224,
        num_episodes=5,
        eval_max_steps=2000,
        latency_steps=2,
        unnorm_key="flappy",
        latency_prompt_map_path=None,
        actions=["noop", "flap"],
        output_dir=tmp_path,
    )
    assert cfg["executor"]["mode"] == "simulated"
    assert cfg["env"]["obs_resize"] == [224, 224]
    assert cfg["latency"]["type"] == "fixed"
    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(2 * 1000.0 / 30.0)
    assert cfg["policy"]["type"] == "starvla"
    assert cfg["policy"]["unnorm_key"] == "flappy"
    assert cfg["evaluation"]["eval_episodes"] == 5
    assert cfg["evaluation"]["eval_parallel_envs"] == 1
    assert "eval_suites" not in cfg["evaluation"]
    assert cfg["logging"]["video"]["enabled"] is False
    assert cfg["logging"]["save_step_records"] is False


def test_translation_zero_latency_uses_zero_type(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        seed=0,
        image_size=224,
        num_episodes=1,
        eval_max_steps=100,
        latency_steps=0,
        unnorm_key=None,
        latency_prompt_map_path=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["latency"]["type"] == "zero"
    assert "fixed_latency_ms" not in cfg["latency"]
    assert "unnorm_key" not in cfg["policy"]


def _make_cfg(num_episodes=4, latencies=(0, 2)):
    return OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "task": "flappy",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "image_size": 224,
                    "seed": 42,
                    "fixed_episode_seeds": True,
                    "latency": {"values": list(latencies), "prompt_map_path": None},
                    "mid_train": {
                        "enabled": True,
                        "interval_steps": 100,
                        "num_episodes": num_episodes,
                        "max_steps_per_episode": 50,
                        "latencies": list(latencies),
                    },
                    "post_train": {
                        "enabled": True,
                        "num_episodes": num_episodes,
                        "max_steps_per_episode": 50,
                        "latencies": list(latencies),
                    },
                },
            },
            "framework": {"action_model": {"state_dim": 7}, "kv_memory": {"enabled": False}},
            "datasets": {"vla_data": {"include_state": True, "num_obs_frames": 1, "image_mode": "single"}},
        }
    )


def test_runner_folds_episode_metrics(tmp_path, monkeypatch):
    def fake_run_from_config(config, *, write_summary, episode_ids, policy):
        return [
            EpisodeMetrics(episode_id=i, episode_return=float(i), survival_steps=10 + i)
            for i in episode_ids
        ]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(cfg=_make_cfg(), output_dir=str(tmp_path))
    result = runner.run(model=object(), step=100, stage="mid_train", save=False)

    assert set(result.per_latency) == {"flappy/latency_0", "flappy/latency_2"}
    metrics = result.per_latency["flappy/latency_2"]
    assert metrics["num_episodes"] == 4
    assert metrics["episode_rewards"] == [0.0, 1.0, 2.0, 3.0]
    assert metrics["mean_reward"] == pytest.approx(1.5)
    assert result.aggregate["total_episodes"] == 8
    assert result.aggregate["task"] == "flappy"


def test_runner_shards_episodes_across_ranks(tmp_path, monkeypatch):
    seen_ids: list[int] = []

    def fake_run_from_config(config, *, write_summary, episode_ids, policy):
        seen_ids.extend(episode_ids)
        return [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5)
            for i in episode_ids
        ]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=4, latencies=(0,)), output_dir=str(tmp_path)
    )
    result = runner.run(model=object(), step=1, stage="mid_train", shard_rank=1, shard_count=2, save=False)

    # rank 1 of 2 over episodes [0,1,2,3] -> [1,3]
    assert sorted(seen_ids) == [1, 3]
    assert result.per_latency["flappy/latency_0"]["episode_indices"] == [1, 3]


def test_runner_merge_results_combines_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner_mod,
        "run_from_config",
        lambda config, *, write_summary, episode_ids, policy: [
            EpisodeMetrics(episode_id=i, episode_return=float(i), survival_steps=5) for i in episode_ids
        ],
    )
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())
    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=4, latencies=(0,)), output_dir=str(tmp_path)
    )
    shard0 = runner.run(model=object(), step=1, stage="mid_train", shard_rank=0, shard_count=2, save=False)
    shard1 = runner.run(model=object(), step=1, stage="mid_train", shard_rank=1, shard_count=2, save=False)

    merged = runner.merge_results([shard0, shard1], step=1, stage="mid_train")
    assert merged.per_latency["flappy/latency_0"]["num_episodes"] == 4
    assert merged.per_latency["flappy/latency_0"]["episode_indices"] == [0, 1, 2, 3]
