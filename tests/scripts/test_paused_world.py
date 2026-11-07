from __future__ import annotations

import json

import pytest

from latency_bench.core.config import load_config
from latency_bench.utils.io import read_json, write_json, write_jsonl, write_yaml
from latency_bench.utils.paths import REPO_ROOT
from scripts.experiments.paused_world.experiment import paused_config, summarize


@pytest.mark.parametrize("task,cap,stride", [("flappy", 3600, 1), ("demon_attack", 7200, 4)])
@pytest.mark.parametrize("model", ["qwenoft", "qwenpi_v3", "qwengr00t"])
def test_paused_config_preserves_published_game_and_history_protocol(tmp_path, task, cap, stride, model):
    base = load_config(REPO_ROOT / f"configs/experiments/sim2real/eval/{task}_{model}.yaml")
    config = paused_config(
        base, tmp_path, episodes=100, parallel_envs=32,
        inference_devices=["cuda:0", "cuda:1"], batch_size=16,
    )
    assert config["latency"] == {"method": "zero"}
    assert base["latency"]["method"] == "temporal"
    assert config["env"] == base["env"]
    assert config["scheduler"] == base["scheduler"]
    assert config["env"]["env_fps"] / config["env"]["obs_fps"] == stride
    assert config["evaluation"]["max_env_steps"] == cap
    assert config["experiment"]["seed"] == 1000000
    assert config["env"]["action_history_decisions"] == 8
    assert config["policy"]["state_source"] == "transport"
    assert config["policy"]["checkpoint"] == "checkpoints/model.pt"


@pytest.fixture
def completed_run(tmp_path):
    cases = []
    for task in ("flappy", "demon_attack"):
        for index, model in enumerate(("qwenoft", "qwenpi_v3", "qwengr00t")):
            case_dir = tmp_path / task / model
            base = load_config(REPO_ROOT / f"configs/experiments/sim2real/eval/{task}_{model}.yaml")
            config = paused_config(
                base, case_dir, episodes=2, parallel_envs=2,
                inference_devices=["cuda:0"], batch_size=2,
            )
            write_yaml(case_dir / "eval_config.yaml", config)
            write_jsonl(case_dir / "episode_metrics.jsonl", [
                {"episode_return": value + index * 2, "survival_steps": 20,
                 "mean_latency_ms": 0, "metadata": {"episode_seed": 1000000 + episode}}
                for episode, value in enumerate((1, 3))
            ])
            write_json(case_dir / "archived_profile_summary.json", {
                "mean_return": 20 - index * 5, "std_return": 1, "returns": [19 - index * 5, 21 - index * 5],
            })
            cases.append({"task": task, "model": model, "bundle": f"fixture/{task}/{model}",
                          "config": f"{task}/{model}/eval_config.yaml"})
    write_json(tmp_path / "experiment.json", {"cases": cases})
    return tmp_path


def test_report_uses_new_episode_scores_and_labels_archived_comparison(completed_run):
    rows = summarize(completed_run)
    assert len(rows) == 6
    assert rows[0]["paused_mean"] == 2
    assert rows[0]["paused_std"] == 1
    assert rows[0]["paused_ci95_low"] == 1
    assert rows[0]["paused_ci95_high"] == 3
    assert rows[0]["archived_profile_mean"] == 20
    assert rows[0]["paused_rank"] == 3
    assert rows[0]["archived_profile_rank"] == 1
    assert read_json(completed_run / "results.json")["status"] == "complete"
    assert (completed_run / "results.csv").is_file()


@pytest.mark.parametrize("failure,match", [("partial", "cohort"), ("delay", "Nonzero"), ("cap", "raw-frame")])
def test_report_rejects_incomplete_or_wrong_protocol_results(completed_run, failure, match):
    path = completed_run / "flappy/qwenoft/episode_metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "partial":
        records.pop()
    elif failure == "delay":
        records[0]["mean_latency_ms"] = 100
    else:
        records[0]["survival_steps"] = 3601
    write_jsonl(path, records)
    with pytest.raises(ValueError, match=match):
        summarize(completed_run)
    assert not (completed_run / "results.csv").exists()
