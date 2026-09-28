import json

import pytest

from scripts.tasks.flappy.analyze_profile_eval import analyze


def _episodes(path, *, return_offset, game_score=1, seed_offset=0):
    rows = [
        {
            "episode_id": episode,
            "episode_return": return_offset + episode / 10,
            "game_score": game_score,
            "survival_steps": 20 + episode,
            "num_actions": 10,
            "num_dropped_actions": 1,
            "metadata": {
                "episode_seed": seed_offset + episode,
                "submitted_observation_frames": 8,
                "dropped_observation_count": 2,
                "checkpoint_path": str(path.with_suffix(".pth")),
                "train_env_steps": 25_000_000,
            },
        }
        for episode in range(100)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_matched_profile_report_requires_positive_ci_and_pipe_median(tmp_path):
    best = _episodes(tmp_path / "best.jsonl", return_offset=10)
    final = _episodes(tmp_path / "final.jsonl", return_offset=9)
    noop = _episodes(tmp_path / "noop.jsonl", return_offset=0, game_score=0)
    random = _episodes(tmp_path / "random.jsonl", return_offset=2, game_score=0)
    milestone = _episodes(tmp_path / "milestone.jsonl", return_offset=7)

    report = analyze(best, final, noop, random, [milestone])

    assert report["learned_control"] is True
    assert report["best_minus_noop_mean_return_ci95"] == pytest.approx([10, 10])
    assert report["best_minus_random_mean_return_ci95"] == pytest.approx([8, 8])
    assert report["best"]["median_game_score"] == 1
    assert report["best"]["admitted_observation_fraction"] == pytest.approx(0.8)
    assert report["milestones"][0]["training_step"] == 25_000_000


def test_matched_profile_report_rejects_different_episode_seeds(tmp_path):
    best = _episodes(tmp_path / "best.jsonl", return_offset=10)
    noop = _episodes(tmp_path / "noop.jsonl", return_offset=0, seed_offset=1)
    random = _episodes(tmp_path / "random.jsonl", return_offset=2)

    with pytest.raises(ValueError, match="seeds do not match"):
        analyze(best, best, noop, random, [])


def test_trained_profile_requires_checkpoint_and_step_provenance(tmp_path):
    best = _episodes(tmp_path / "best.jsonl", return_offset=10)
    control = _episodes(tmp_path / "control.jsonl", return_offset=0)
    rows = [json.loads(line) for line in best.read_text().splitlines()]
    for row in rows:
        row["metadata"]["checkpoint_path"] = ""
        del row["metadata"]["train_env_steps"]
    best.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises((ValueError, KeyError)):
        analyze(best, control, control, control, [])
