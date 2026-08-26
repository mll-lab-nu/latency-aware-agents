from __future__ import annotations

import json
from pathlib import Path

import yaml

import scripts.experiments.humanoidbench_10m_baseline as baseline
from scripts.experiment_runner.job_manifest import load_manifest


def test_generate_baseline_writes_five_gpu_jobs(tmp_path, monkeypatch):
    def fake_task_env_config(task_name: str):
        return (
            {
                "name": "gymnasium",
                "task_name": task_name,
                "env_id": f"LatencyBench/{task_name}-v0",
                "registration_imports": ["latency_bench.envs.humanoidbench"],
                "make_kwargs": {"render_mode": None, "width": 256, "height": 256},
                "env_fps": 50.0,
                "obs_fps": 50.0,
                "frame_stack": 1,
                "action_space": {"type": "box"},
                "noop_action": [0.0],
                "base_prompt": task_name,
            },
            {"resolved_spec": {"max_episode_steps": 500 if "basketball" in task_name else 1000}},
        )

    monkeypatch.setattr(baseline, "_task_env_config", fake_task_env_config)
    summary = baseline.generate_baseline(
        output_root=tmp_path / "results",
    )

    assert summary["tasks"] == list(baseline.DEFAULT_TASKS)
    assert summary["train_for_env_steps"] == 10_000_000
    assert summary["runner_command"].endswith("--gpus 0,1")

    manifest_path = Path(summary["manifest"])
    rows = load_manifest(manifest_path)
    assert len(rows) == 5
    assert {row["env"] for row in rows} == set(baseline.DEFAULT_TASKS)

    for row in rows:
        train_config = yaml.safe_load(Path(row["train_config"]).read_text())
        eval_config = yaml.safe_load(Path(row["eval_config"]).read_text())
        assert train_config["experiment"]["seed"] == 0
        assert train_config["backend"]["device"] == "gpu"
        assert train_config["backend"]["restart_behavior"] == "resume"
        assert train_config["backend"]["actor_worker_gpus"] == []
        assert train_config["env"]["make_kwargs"]["render_mode"] is None
        assert train_config["training"]["train_for_env_steps"] == 10_000_000
        assert train_config["training"]["num_workers"] == 32
        assert train_config["training"]["num_envs_per_worker"] == 1
        assert train_config["training"]["worker_num_splits"] == 1
        assert train_config["training"]["rollout"] == 256
        assert train_config["training"]["batch_size"] == 512
        assert train_config["training"]["num_batches_per_epoch"] == 16
        assert train_config["training"]["num_epochs"] == 10
        assert train_config["training"]["learning_rate"] == 3e-5
        assert train_config["training"]["ppo_clip_ratio"] == 0.2
        assert train_config["training"]["ppo_clip_value"] == 1e9
        assert train_config["training"]["max_grad_norm"] == 0.5
        assert train_config["training"]["kl_loss_coeff"] == 0.0
        assert train_config["training"]["reward_clip"] == 1e9
        assert train_config["training"]["adam_eps"] == 1e-5
        assert train_config["training"]["nonlinearity"] == "tanh"
        assert train_config["training"]["policy_initialization"] == "orthogonal"
        assert train_config["training"]["actor_critic_share_weights"] is False
        assert train_config["training"]["normalize_input"] is False
        assert train_config["training"]["normalize_returns"] is False
        assert train_config["training"]["value_bootstrap"] is True
        assert train_config["training"]["shuffle_minibatches"] is True
        assert train_config["training"]["adaptive_stddev"] is False
        assert train_config["training"]["async_rl"] is False
        assert train_config["training"]["batched_sampling"] is False
        assert train_config["logging"]["wandb_project"] == "latency-sensitive-bench"
        assert train_config["logging"]["wandb_group"] == "humanoidbench-sf-official-10m"
        assert train_config["logging"]["wandb_job_type"] == "sample_factory"
        assert train_config["logging"]["wandb_tags"] == [
            "humanoidbench",
            "official-recipe",
            "raw-gaussian",
            "10m",
        ]
        assert eval_config["backend"]["run_mode"] == "eval"
        assert eval_config["backend"]["extra_args"] == ["--load_checkpoint_kind", "best"]
        assert eval_config["policy"]["checkpoint_kind"] == "best"
        assert eval_config["policy"]["device"] == "cuda"
        assert eval_config["policy"]["deterministic"] is True
        assert eval_config["env"]["make_kwargs"]["render_mode"] == "rgb_array"
        assert eval_config["evaluation"]["eval_episodes"] == 10
        assert eval_config["evaluation"]["eval_latency_values"] == [0]
        expected_horizon = 500 if row["env"] == "h1hand_basketball" else 1000
        assert eval_config["evaluation"]["eval_max_steps"] == expected_horizon
        assert eval_config["logging"]["video"]["enabled"] is True
        assert eval_config["logging"]["save_step_records"] is True
        assert "-m training.train" in row["train_command"]
        assert "-m latency_bench.run" in row["eval_command"]

    assert json.loads((manifest_path.parent / "summary.json").read_text())["seed"] == 0
    json.dumps(summary)
