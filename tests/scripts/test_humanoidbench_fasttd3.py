from __future__ import annotations

from pathlib import Path

import yaml

import scripts.experiments.humanoidbench_fasttd3 as fasttd3
from scripts.experiment_runner.job_manifest import load_manifest


def test_generate_fasttd3_writes_task_configs_and_budget_override(tmp_path, monkeypatch):
    def fake_task_env_config(task_name: str):
        return (
            {
                "name": "gymnasium",
                "task_name": task_name,
                "env_id": f"LatencyBench/{task_name}-v0",
                "registration_imports": ["latency_bench.envs.humanoidbench"],
                "make_kwargs": {"render_mode": None},
                "env_fps": 50.0,
                "obs_fps": 50.0,
                "frame_stack": 1,
                "action_space": {"type": "box"},
                "noop_action": [0.0],
                "base_prompt": task_name,
            },
            {"resolved_spec": {"max_episode_steps": 500 if "basketball" in task_name else 1000}},
        )

    monkeypatch.setattr(fasttd3, "_task_env_config", fake_task_env_config)
    summary = fasttd3.generate_fasttd3(
        output_root=tmp_path / "results",
        train_for_env_steps=128,
    )

    manifest_path = Path(summary["manifest"])
    rows = load_manifest(manifest_path)
    assert len(rows) == 5
    assert {row["env"] for row in rows} == set(fasttd3.DEFAULT_TASKS)

    for row in rows:
        train_config = yaml.safe_load(Path(row["train_config"]).read_text())
        eval_config = yaml.safe_load(Path(row["eval_config"]).read_text())
        training = train_config["training"]
        assert train_config["backend"]["algo"] == "FAST_TD3"
        assert train_config["backend"]["actor_worker_gpus"] == []
        assert training["train_for_env_steps"] == 128
        assert training["num_workers"] * training["num_envs_per_worker"] == 128
        assert training["async_rl"] is False
        assert training["rollout"] == 1
        assert training["batch_size"] == 128
        assert training["fasttd3_replay_capacity"] == 6_553_600
        assert training["fasttd3_replay_batch_size"] == 32_768
        assert training["fasttd3_transitions_per_update"] == 64
        assert training["fasttd3_compile"] is True
        assert training["learning_rate"] == 3e-4
        assert training["gamma"] == 0.99
        assert training["actor_critic_share_weights"] is True
        assert training["reward_clip"] == 1e9
        expected_support = (-2000.0, 2000.0) if row["env"] == "h1hand_basketball" else (-250.0, 250.0)
        assert (training["fasttd3_v_min"], training["fasttd3_v_max"]) == expected_support
        assert train_config["evaluation"]["eval_interval_steps"] == 640_000
        assert train_config["evaluation"]["eval_suites"] == {"fixed": [[0]]}
        assert train_config["logging"]["wandb_project"] == "latency-sensitive-bench"
        assert eval_config["backend"]["run_mode"] == "eval"
        assert eval_config["evaluation"]["eval_max_steps"] == (500 if row["env"] == "h1hand_basketball" else 1000)
