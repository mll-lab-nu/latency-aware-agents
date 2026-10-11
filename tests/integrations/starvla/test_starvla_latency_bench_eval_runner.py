from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("starVLA.training.rl_games", exc_type=ImportError)
OmegaConf = pytest.importorskip("omegaconf", exc_type=ImportError).OmegaConf

from latency_bench.core.types import EpisodeMetrics
from latency_bench.integrations.starvla import rl_games_eval_runner as runner_mod
from latency_bench.integrations.starvla.rl_games_eval_runner import (
    LatencyBenchRlGamesEvalRunner,
    build_latency_bench_config_for_rl_games,
)
from latency_bench.policy.starvla_prompts import resolve_starvla_prompt


def _model_cfg(
    *,
    num_obs_frames: int = 1,
    image_mode: str = "single",
    kv_memory_enabled: bool = False,
    pack_image_sequence: bool = False,
    image_sequence_length: int = 1,
) -> dict:
    return {
        "framework": {"action_model": {"state_dim": 7}, "kv_memory": {"enabled": kv_memory_enabled}},
        "datasets": {
            "vla_data": {
                "include_state": True,
                "num_obs_frames": num_obs_frames,
                "image_mode": image_mode,
                "pack_image_sequence": pack_image_sequence,
                "image_sequence_length": image_sequence_length,
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
        max_env_steps=2000,
        latency_steps=2,
        parallel_envs=8,
        unnorm_key="flappy",
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=["noop", "flap"],
        output_dir=tmp_path,
    )
    assert cfg["executor"]["mode"] == "simulated"
    assert cfg["env"]["obs_resize"] == [224, 224]
    assert cfg["latency"]["method"] == "fixed"
    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(2 * 1000.0 / 30.0)
    assert cfg["policy"]["type"] == "starvla"
    assert cfg["policy"]["unnorm_key"] == "flappy"
    assert cfg["evaluation"]["episodes"] == 5
    assert cfg["executor"]["parallel_envs"] == 8
    assert cfg["evaluation"]["max_env_steps"] == 2000
    assert cfg["evaluation"]["eval_max_decision_steps"] == 2000
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
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"2": {"latency_raw_frames": 8, "prompt": "latency 2"}}),
        encoding="utf-8",
    )
    cfg = build_latency_bench_config_for_rl_games(
        env_name=env_name,
        model_cfg=_model_cfg(),
        seed=42,
        image_size=224,
        num_episodes=5,
        max_env_steps=2000,
        latency_steps=2,
        parallel_envs=1,
        unnorm_key=env_name,
        latency_prompt_map_path=str(prompt_map_path),
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )

    assert cfg["latency"]["method"] == "fixed"
    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(2 * 1000.0 / obs_fps)
    assert cfg["policy"]["latency_prompt_key"] == 2
    assert cfg["evaluation"]["max_env_steps"] == pytest.approx(
        2000 * float(cfg["env"]["env_fps"]) / float(cfg["env"]["obs_fps"])
    )
    assert cfg["evaluation"]["eval_max_decision_steps"] == 2000


def test_fold_episode_metrics_reports_decision_step_lengths():
    runner = LatencyBenchRlGamesEvalRunner.__new__(LatencyBenchRlGamesEvalRunner)
    metrics = runner._fold_episode_metrics(
        [
            EpisodeMetrics(episode_id=0, episode_return=40.0, survival_steps=876),
            EpisodeMetrics(episode_id=1, episode_return=400.0, survival_steps=14400),
        ],
        latency=0,
        eval_seed=42,
        obs_stride_raw_frames=4,
    )

    assert metrics["episode_lengths"] == [219, 3600]
    assert metrics["mean_length"] == pytest.approx((219 + 3600) / 2)


@pytest.mark.parametrize(
    ("env_name", "latency_field", "expected_latency_ms"),
    [
        ("demon_attack", "latency_raw_frames", 100.0),
        ("deadly_corridor", "latency_raw_frames", 6 * 1000.0 / 35.0),
        ("atlantis", "latency", 100.0),
    ],
)
def test_translation_uses_prompt_map_raw_frame_latency(
    tmp_path,
    env_name,
    latency_field,
    expected_latency_ms,
):
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"6": {latency_field: 6, "prompt": "latency 6"}}),
        encoding="utf-8",
    )

    model_cfg = _model_cfg()
    model_cfg["datasets"]["vla_data"].update(data_root_dir=str(tmp_path), data_mix=f"{env_name}_train")
    base = runner_mod.load_config(runner_mod.BASE_CONFIGS[env_name])
    if base["env"]["name"] == "gymnasium":
        model_cfg["datasets"]["vla_data"]["gymnasium_task_contract"] = runner_mod.gymnasium_task_contract(base["env"])
    cfg = build_latency_bench_config_for_rl_games(
        env_name=env_name,
        model_cfg=model_cfg,
        seed=42,
        image_size=224,
        num_episodes=1,
        max_env_steps=10,
        latency_steps=6,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=str(prompt_map_path),
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )

    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(
        expected_latency_ms
    )
    assert cfg["policy"]["latency_prompt_key"] == 6


def test_translation_zero_latency_uses_zero_type(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(),
        seed=0,
        image_size=224,
        num_episodes=1,
        max_env_steps=100,
        latency_steps=0,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["latency"]["method"] == "zero"
    assert "fixed_latency_ms" not in cfg["latency"]
    assert "unnorm_key" not in cfg["policy"]


def test_translation_gymnasium_continuous_uses_training_contract(tmp_path):
    task_contract = {
        "task_name": "walker2d_continuous_torque",
        "env_id": "LatencyBench/Walker2dContinuous-v0",
        "registration_imports": ["latency_bench.envs.tasks.walker2d"],
        "make_kwargs": {"base_env_id": "Walker2d-v4", "render_mode": "rgb_array"},
        "base_prompt": "Move forward.",
        "env_fps": 125.0,
        "obs_fps": 125.0,
        "frame_stack": 1,
        "action_space": {
            "type": "box",
            "labels": ["a", "b"],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
            "dtype": "float32",
        },
        "noop_action": [0.0, 0.0],
    }
    model_cfg = _model_cfg()
    model_cfg["rl_games"] = {"gymnasium": {"task_contract": task_contract}}
    model_cfg["datasets"]["vla_data"]["gymnasium_task_contract"] = task_contract
    model_cfg["datasets"]["vla_data"].update(
        {"data_root_dir": str(tmp_path), "data_mix": "walker_mixture"}
    )

    cfg = build_latency_bench_config_for_rl_games(
        env_name="gymnasium",
        model_cfg=model_cfg,
        seed=42,
        image_size=224,
        num_episodes=20,
        max_env_steps=3600,
        latency_steps=2,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path / "eval",
    )

    assert cfg["env"] == {"name": "gymnasium", "obs_resize": [224, 224], **task_contract}
    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(16.0)
    assert cfg["policy"]["gymnasium_training_task_contract"] == task_contract
    assert cfg["policy"]["gymnasium_active_action_dim"] == 2
    assert cfg["evaluation"]["episodes"] == 20


def test_translation_gymnasium_discrete_restores_action_catalog(tmp_path):
    task_contract = {
        "task_name": "inverted_pendulum_discrete",
        "env_id": "LatencyBench/InvertedPendulumDiscrete-v0",
        "registration_imports": ["latency_bench.envs.tasks.inverted_pendulum"],
        "make_kwargs": {
            "base_env_id": "InvertedPendulum-v4",
            "base_make_kwargs": {},
            "render_mode": "rgb_array",
        },
        "base_prompt": "Keep the pole upright.",
        "env_fps": 25.0,
        "obs_fps": 25.0,
        "frame_stack": 1,
        "action_labels": ["left", "zero", "right"],
        "action_values": [0, 1, 2],
        "noop_action_id": 1,
    }
    model_cfg = _model_cfg()
    model_cfg["rl_games"] = {
        "gymnasium": {
            "task_contract": {
                **task_contract,
                "make_kwargs": {
                    "base_env_id": "InvertedPendulum-v4",
                    "render_mode": "rgb_array",
                },
            }
        }
    }
    model_cfg["datasets"]["vla_data"]["gymnasium_task_contract"] = task_contract
    model_cfg["datasets"]["vla_data"].update(
        {"data_root_dir": str(tmp_path), "data_mix": "pendulum_train"}
    )

    cfg = build_latency_bench_config_for_rl_games(
        env_name="gymnasium",
        model_cfg=model_cfg,
        seed=42,
        image_size=224,
        num_episodes=20,
        max_env_steps=1000,
        latency_steps=0,
        parallel_envs=5,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path / "eval",
    )

    assert cfg["env"]["action_order"] == ["left", "zero", "right"]
    assert cfg["env"]["action_map"] == {"left": 0, "zero": 1, "right": 2}
    assert cfg["env"]["noop_action"] == "zero"


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

    with pytest.raises(KeyError, match="3"):
        resolve_starvla_prompt(
            env_name="flappy",
            observation_metadata={"latency_raw_frames": 3},
            latency_prompt_map={"2": {"prompt": prompt}},
        )


@pytest.mark.parametrize(
    ("env_name", "image_mode", "num_obs_frames", "expected_frame_stack"),
    [
        ("flappy", "multiframe", 8, 8),
        ("demon_attack", "multiframe", 4, 13),
        ("deadly_corridor", "stitch", 4, 13),
    ],
)
def test_translation_multiframe_expands_eval_frame_stack_at_observation_stride(
    tmp_path,
    env_name,
    image_mode,
    num_obs_frames,
    expected_frame_stack,
):
    cfg = build_latency_bench_config_for_rl_games(
        env_name=env_name,
        model_cfg=_model_cfg(num_obs_frames=num_obs_frames, image_mode=image_mode),
        seed=0,
        image_size=224,
        num_episodes=1,
        max_env_steps=100,
        latency_steps=0,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["env"]["frame_stack"] == expected_frame_stack


def test_translation_multiframe_keeps_larger_base_frame_stack(tmp_path):
    cfg = build_latency_bench_config_for_rl_games(
        env_name="flappy",
        model_cfg=_model_cfg(num_obs_frames=2, image_mode="multiframe"),
        seed=0,
        image_size=224,
        num_episodes=1,
        max_env_steps=100,
        latency_steps=0,
        parallel_envs=1,
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
        max_env_steps=100,
        latency_steps=0,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )
    assert cfg["env"]["frame_stack"] == 4


@pytest.mark.parametrize(
    ("env_name", "image_transform", "expected_frame_stack"),
    [
        ("flappy", "flappy_ghost_trail", 6),
        ("demon_attack", "demon_attack_ghost_trail", 5),
    ],
)
def test_translation_ghost_history_is_not_expanded_by_multiframe_input(
    tmp_path,
    env_name,
    image_transform,
    expected_frame_stack,
):
    cfg = build_latency_bench_config_for_rl_games(
        env_name=env_name,
        model_cfg=_model_cfg(num_obs_frames=8, image_mode="multiframe"),
        seed=0,
        image_size=224,
        num_episodes=1,
        max_env_steps=100,
        latency_steps=0,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config={
            "image_transform": image_transform,
            "history_frames": 5,
        },
        actions=None,
        output_dir=tmp_path,
    )

    assert cfg["env"]["frame_stack"] == expected_frame_stack


@pytest.mark.parametrize(
    ("env_name", "expected_frame_stack"),
    [
        ("flappy", 5),
        ("demon_attack", 17),
        ("deadly_corridor", 17),
    ],
)
def test_translation_wanoft_expands_raw_history_at_observation_stride(
    tmp_path,
    env_name,
    expected_frame_stack,
):
    cfg = build_latency_bench_config_for_rl_games(
        env_name=env_name,
        model_cfg=_model_cfg(
            pack_image_sequence=True,
            image_sequence_length=5,
        ),
        seed=0,
        image_size=224,
        num_episodes=1,
        max_env_steps=100,
        latency_steps=0,
        parallel_envs=1,
        unnorm_key=None,
        latency_prompt_map_path=None,
        image_transform_config=None,
        actions=None,
        output_dir=tmp_path,
    )

    assert cfg["env"]["frame_stack"] == expected_frame_stack


def test_translation_wanoft_rejects_single_image_ghost_trail(tmp_path):
    with pytest.raises(
        ValueError,
        match="packed image sequences require image_transform=raw_rgb",
    ):
        build_latency_bench_config_for_rl_games(
            env_name="flappy",
            model_cfg=_model_cfg(
                pack_image_sequence=True,
                image_sequence_length=5,
            ),
            seed=0,
            image_size=224,
            num_episodes=1,
            max_env_steps=100,
            latency_steps=0,
            parallel_envs=1,
            unnorm_key=None,
            latency_prompt_map_path=None,
            image_transform_config={
                "image_transform": "flappy_ghost_trail",
                "history_frames": 5,
            },
            actions=None,
            output_dir=tmp_path,
        )


def _make_cfg(
    num_episodes=4,
    latencies=(0, 2),
    parallel_envs=None,
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
    if parallel_envs is not None:
        env_eval["parallel_envs"] = parallel_envs
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
        captured["parallel_envs"] = config["executor"]["parallel_envs"]
        return [
            EpisodeMetrics(episode_id=i, episode_return=1.0, survival_steps=5) for i in episode_ids
        ]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", lambda **kwargs: object())

    # default (unset) -> 1
    runner = LatencyBenchRlGamesEvalRunner(cfg=_make_cfg(latencies=(0,)), output_dir=str(tmp_path))
    runner.run(model=object(), step=1, stage="mid_train", save=False)
    assert captured["parallel_envs"] == 1

    # set on env_eval -> threaded into the synthesized lb config
    runner = LatencyBenchRlGamesEvalRunner(
        cfg=_make_cfg(latencies=(0,), parallel_envs=8), output_dir=str(tmp_path)
    )
    runner.run(model=object(), step=1, stage="mid_train", save=False)
    assert captured["parallel_envs"] == 8


def test_runner_applies_eval_only_gymnasium_reset_kwargs(tmp_path, monkeypatch):
    captured = {}
    saved_statistics = {"new_embodiment": {
        "state": {key: [0.0] * 7 for key in ("mean", "std", "min", "max", "q01", "q99")},
        "action": {key: [0.0] * 2 for key in ("mean", "std", "min", "max", "q01", "q99")},
    }}
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(saved_statistics))
    task_contract = {
        "task_name": "air_raid",
        "env_id": "LatencyBench/AirRaid-v0",
        "registration_imports": ["latency_bench.envs.tasks.air_raid"],
        "make_kwargs": {"noop_max": 0},
        "env_fps": 50.0,
        "obs_fps": 12.5,
        "frame_stack": 4,
        "action_labels": ["noop", "fire"],
        "action_values": [0, 1],
        "noop_action_id": 0,
        "base_prompt": "Play Air Raid.",
    }
    cfg = _make_cfg(latencies=(0,))
    cfg.rl_games.task = "gymnasium"
    cfg.rl_games.env_eval.gymnasium = {
        "make_kwargs": {"noop_max": 30, "fire_reset": True}
    }
    cfg.datasets.vla_data.gymnasium_task_contract = task_contract
    cfg.datasets.vla_data.data_root_dir = str(tmp_path)
    cfg.datasets.vla_data.data_mix = "air_raid_train"

    def fake_run_from_config(config, *, write_summary, episode_ids, policy, on_episode_complete=None):
        captured["make_kwargs"] = config["env"]["make_kwargs"]
        return [EpisodeMetrics(episode_id=0, episode_return=1.0, survival_steps=5)]

    monkeypatch.setattr(runner_mod, "run_from_config", fake_run_from_config)
    def build_policy(**kwargs):
        captured["statistics"] = kwargs["normalization_statistics"]
        return object()

    monkeypatch.setattr(runner_mod, "build_live_starvla_policy", build_policy)

    LatencyBenchRlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path)).run(
        model=object(), step=5000, stage="post_train", save=False
    )

    assert captured["make_kwargs"] == {"noop_max": 30, "fire_reset": True}
    assert task_contract["make_kwargs"] == {"noop_max": 0}
    assert captured["statistics"] == saved_statistics


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
