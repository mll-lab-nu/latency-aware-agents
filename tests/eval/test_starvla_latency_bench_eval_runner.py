from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from latency_bench.core.types import EpisodeMetrics
from latency_bench.integrations import starvla_rl_games_eval_runner as runner_mod
from latency_bench.integrations.starvla_rl_games_eval_runner import (
    LatencyBenchRlGamesEvalRunner,
    build_latency_bench_config_for_rl_games,
)
from latency_bench.policy.starvla_prompts import resolve_starvla_prompt


def _model_cfg(
    *,
    num_obs_frames: int = 1,
    image_mode: str = "single",
    kv_memory_enabled: bool = False,
) -> dict:
    return {
        "framework": {"action_model": {"state_dim": 7}, "kv_memory": {"enabled": kv_memory_enabled}},
        "datasets": {
            "vla_data": {
                "include_state": True,
                "num_obs_frames": num_obs_frames,
                "image_mode": image_mode,
            }
        },
    }


def test_translation_flappy_fixed_latency(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(),
        seed=42,
        image_size=224,
        num_episodes=5,
        eval_max_steps=2000,
        latency_steps=2,
        eval_parallel_envs=8,
        unnorm_key="flappy",
        latency_prompt_map_path=None,
        image_transform_config=None,
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
    assert cfg["evaluation"]["eval_parallel_envs"] == 8
    assert "eval_suites" not in cfg["evaluation"]
    assert cfg["logging"]["video"]["enabled"] is False
    assert cfg["logging"]["save_step_records"] is False


@pytest.mark.parametrize(
    ("env_name", "obs_fps"),
    [
        ("demon_attack", 15.0),
        ("deadly_corridor", 8.75),
    ],
)
def test_translation_frame_skipped_tasks_use_observation_step_latency(tmp_path, env_name, obs_fps):
    cfg = build_latency_bench_config_for_rl_games(
        env_name=env_name,
        model_cfg=_model_cfg(),
        seed=42,
        image_size=224,
        num_episodes=5,
        eval_max_steps=2000,
        latency_steps=2,
        eval_parallel_envs=1,
        unnorm_key=env_name,
        latency_prompt_map_path=str(tmp_path / "latency_prompt_map.json"),
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )

    assert cfg["latency"]["type"] == "fixed"
    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(2 * 1000.0 / obs_fps)
    assert cfg["policy"]["latency_prompt_key"] == 2


def test_translation_zero_latency_uses_zero_type(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(),
        seed=0,
        image_size=224,
        num_episodes=1,
        eval_max_steps=100,
        latency_steps=0,
        eval_parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["latency"]["type"] == "zero"
    assert "fixed_latency_ms" not in cfg["latency"]
    assert "unnorm_key" not in cfg["policy"]


def test_resolve_starvla_prompt_latency_neutral_strips_latency_suffix():
    prompt = (
        "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. "
        "Choose the action: NOOP, FLAP. Current action latency is 2 raw frames (66.67 ms). "
        "The environment runs at 30 FPS and observations are emitted at 30 FPS. Choose the best next action."
    )
    assert (
        resolve_starvla_prompt(
            env_name="flappy",
            observation_metadata={"latency_raw_frames": 2},
            latency_prompt_map={"2": {"prompt": prompt}},
            prompt_mode="latency_neutral",
        )
        == "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. Choose the action: NOOP, FLAP."
    )


def test_translation_multiframe_expands_eval_frame_stack(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(num_obs_frames=8, image_mode="multiframe"),
        seed=0,
        image_size=224,
        num_episodes=1,
        eval_max_steps=100,
        latency_steps=0,
        eval_parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["env"]["frame_stack"] == 8


def test_translation_multiframe_keeps_larger_base_frame_stack(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(num_obs_frames=2, image_mode="multiframe"),
        seed=0,
        image_size=224,
        num_episodes=1,
        eval_max_steps=100,
        latency_steps=0,
        eval_parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["env"]["frame_stack"] == 4


def test_translation_kv_memory_does_not_expand_eval_frame_stack(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(num_obs_frames=8, image_mode="multiframe", kv_memory_enabled=True),
        seed=0,
        image_size=224,
        num_episodes=1,
        eval_max_steps=100,
        latency_steps=0,
        eval_parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["env"]["frame_stack"] == 4


def _make_cfg(
    num_episodes=4,
    latencies=(0, 2),
    eval_parallel_envs=None,
    num_obs_frames=1,
    image_mode="single",
    kv_memory_enabled=False,
):
    env_eval = {
        "enabled": True,
        "image_size": 224,
        "seed": 42,
        "fixed_episode_seeds": True,
        "latency": {"values": list(latencies), "prompt_map_path": None},
    }
    if eval_parallel_envs is not None:
        env_eval["eval_parallel_envs"] = eval_parallel_envs
    return OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "task": "flappy",
                "model_alias": "openvla",
                "env_eval": {
                    **env_eval,
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
            **_model_cfg(
                num_obs_frames=num_obs_frames,
                image_mode=image_mode,
                kv_memory_enabled=kv_memory_enabled,
            ),
        }
    )


def test_runner_folds_episode_metrics(tmp_path, monkeypatch):
    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
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


def test_runner_threads_eval_parallel_envs_from_cfg(tmp_path, monkeypatch):
    captured = {}

    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        captured["eval_parallel_envs"] = config["evaluation"]["eval_parallel_envs"]
        return [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5) for i in episode_ids
        ]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    # default (unset) -> 1
    runner = LatencyBenchRlGamesEvalRunner(cfg=_make_cfg(latencies=(0,)), output_dir=str(tmp_path))
    runner.run(model=object(), step=1, stage="mid_train", save=False)
    assert captured["eval_parallel_envs"] == 1

    # set on env_eval -> threaded into the synthesized lb config
    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(latencies=(0,), eval_parallel_envs=8), output_dir=str(tmp_path)
    )
    runner.run(model=object(), step=1, stage="mid_train", save=False)
    assert captured["eval_parallel_envs"] == 8


def test_runner_threads_multiframe_frame_stack_from_model_cfg(tmp_path, monkeypatch):
    captured = {}

    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        captured["frame_stack"] = config["env"]["frame_stack"]
        return [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5) for i in episode_ids
        ]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(latencies=(0,), num_obs_frames=8, image_mode="multiframe"),
        output_dir=str(tmp_path),
    )
    runner.run(model=object(), step=1, stage="mid_train", save=False)

    assert captured["frame_stack"] == 8


@pytest.mark.parametrize(
    ("shard_rank", "expected_episode_ids"),
    [
        (0, [0, 2, 4]),
        (1, [1, 3]),
    ],
)
def test_runner_shards_episodes_across_ranks(tmp_path, monkeypatch, shard_rank, expected_episode_ids):
    seen_ids: list[int] = []

    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        seen_ids.extend(episode_ids)
        return [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5)
            for i in episode_ids
        ]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=5, latencies=(0,)), output_dir=str(tmp_path)
    )
    result = runner.run(
        model=object(),
        step=1,
        stage="mid_train",
        shard_rank=shard_rank,
        shard_count=2,
        save=False,
    )

    assert sorted(seen_ids) == expected_episode_ids
    assert result.per_latency["flappy/latency_0"]["episode_indices"] == expected_episode_ids


def test_runner_merge_results_combines_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner_mod,
        "run_from_config",
        lambda config, *, write_summary, episode_ids, policy, on_episode_complete=None: [
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


def test_runner_prints_single_process_global_episode_progress(tmp_path, monkeypatch, capsys):
    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        metrics = [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5)
            for i in episode_ids
        ]
        for metric in metrics:
            on_episode_complete(metric)
        return metrics

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=2, latencies=(0, 2)), output_dir=str(tmp_path)
    )
    runner.run(model=object(), step=100, stage="mid_train", save=False)

    captured = capsys.readouterr()
    assert "[bench] mid_train step=100 episodes 1/4 done" in captured.err
    assert "[bench] mid_train step=100 episodes 4/4 done" in captured.err


class _FakeDist:
    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def barrier(self):
        pass


def test_runner_distributed_nonzero_rank_writes_progress_without_printing(tmp_path, monkeypatch, capsys):
    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        metrics = [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5)
            for i in episode_ids
        ]
        for metric in metrics:
            on_episode_complete(metric)
        return metrics

    monkeypatch.setattr(runner_mod, "dist", _FakeDist())
    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=4, latencies=(0,)), output_dir=str(tmp_path)
    )
    runner.run(
        model=object(),
        step=1,
        stage="mid_train",
        shard_rank=1,
        shard_count=2,
        save=False,
    )

    captured = capsys.readouterr()
    assert "[bench]" not in captured.err
    progress_path = tmp_path / "latency_bench_eval" / "mid_train" / "step_1" / "_progress" / "rank_1.json"
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {"completed": 2}


class _FakeUninitializedDist:
    def is_available(self):
        return True

    def is_initialized(self):
        return False


def test_runner_sharded_without_distributed_group_does_not_print_local_progress(tmp_path, monkeypatch, capsys):
    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        metrics = [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5)
            for i in episode_ids
        ]
        for metric in metrics:
            on_episode_complete(metric)
        return metrics

    monkeypatch.setattr(runner_mod, "dist", _FakeUninitializedDist())
    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=4, latencies=(0,)), output_dir=str(tmp_path)
    )
    runner.run(
        model=object(),
        step=1,
        stage="mid_train",
        shard_rank=1,
        shard_count=2,
        save=False,
    )

    captured = capsys.readouterr()
    assert "[bench]" not in captured.err


def test_runner_distributed_rank_zero_prints_global_episode_progress(tmp_path, monkeypatch, capsys):
    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        progress_dir = tmp_path / "latency_bench_eval" / "mid_train" / "step_1" / "_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        (progress_dir / "rank_1.json").write_text(json.dumps({"completed": 2}), encoding="utf-8")
        metrics = [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5)
            for i in episode_ids
        ]
        for metric in metrics:
            on_episode_complete(metric)
        return metrics

    monkeypatch.setattr(runner_mod, "dist", _FakeDist())
    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(num_episodes=4, latencies=(0,)), output_dir=str(tmp_path)
    )
    runner.run(
        model=object(),
        step=1,
        stage="mid_train",
        shard_rank=0,
        shard_count=2,
        save=False,
    )

    captured = capsys.readouterr()
    assert "[bench] mid_train step=1 episodes 4/4 done" in captured.err
    assert "[bench] mid_train step=1 episodes 2/4 done" not in captured.err
