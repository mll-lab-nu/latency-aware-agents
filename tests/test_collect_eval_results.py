from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_eval_results.py"
HEADER = [
    "game",
    "model",
    "train_latency",
    "eval_frames_advance",
    "mean_score",
    "mean_steps",
    "seed",
    "source_file",
    "raw_return_mean",
    "return_mean",
    "raw_frames_mean",
    "episode_length_mean",
]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_collect(root: Path, out: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--out",
            str(out),
            *extra_args,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_collects_summary_json_and_writes_per_game_csv(tmp_path: Path) -> None:
    root = tmp_path / "hf_eval"
    out = tmp_path / "tables" / "eval_summary_all.csv"
    result_file = (
        root
        / "outputs"
        / "latency_bench"
        / "deadly_corridor"
        / "train_latency_3"
        / "seed_7"
        / "queue_eval_latency_5.json"
    )
    _write_json(
        result_file,
        {
            "mean_score": 12.5,
            "mean_steps": 33,
            "raw_return_mean": 11.5,
            "return_mean": 12.25,
            "raw_frames_mean": 40,
            "episode_length_mean": 35,
        },
    )

    result = _run_collect(root, out)

    assert result.returncode == 0, result.stderr
    rows = _read_rows(out)
    assert rows == [
        {
            "game": "deadly_corridor",
            "model": "Sample Factory PPO latency=3",
            "train_latency": "3",
            "eval_frames_advance": "5",
            "mean_score": "12.5",
            "mean_steps": "33",
            "seed": "7",
            "source_file": "outputs/latency_bench/deadly_corridor/train_latency_3/seed_7/queue_eval_latency_5.json",
            "raw_return_mean": "11.5",
            "return_mean": "12.25",
            "raw_frames_mean": "40",
            "episode_length_mean": "35",
        }
    ]
    assert _read_rows(tmp_path / "tables" / "eval_summary_all_deadly_corridor.csv") == rows


def test_collects_episode_list_json(tmp_path: Path) -> None:
    root = tmp_path / "hf_eval"
    out = tmp_path / "tables" / "eval_summary_all.csv"
    _write_json(
        root
        / "outputs"
        / "eval"
        / "demon_attack"
        / "latency_4"
        / "seed_0"
        / "queue_eval_latency_2.json",
        {
            "episodes": [
                {"score": 1, "return": 2, "raw_return": 3, "steps": 4, "length": 5, "raw_frames": 6},
                {"score": 3, "return": 4, "raw_return": 5, "steps": 6, "length": 7, "raw_frames": 8},
            ]
        },
    )

    result = _run_collect(root, out)

    assert result.returncode == 0, result.stderr
    rows = _read_rows(out)
    assert rows[0]["game"] == "demon_attack"
    assert rows[0]["train_latency"] == "4"
    assert rows[0]["eval_frames_advance"] == "2"
    assert rows[0]["mean_score"] == "2.0"
    assert rows[0]["mean_steps"] == "5.0"
    assert rows[0]["raw_return_mean"] == "4.0"
    assert rows[0]["return_mean"] == "3.0"
    assert rows[0]["raw_frames_mean"] == "7.0"
    assert rows[0]["episode_length_mean"] == "6.0"


def test_collects_root_episode_list_and_ignores_actions_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "hf_eval"
    out = tmp_path / "tables" / "eval_summary_all.csv"
    result_dir = root / "flappy" / "train_latency_1" / "seed_2"
    _write_json(
        result_dir / "queue_eval_latency_0.json",
        [{"return": 10, "length": 20}, {"return": 14, "length": 24}],
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "queue_eval_latency_99_actions.jsonl").write_text("ignored\n", encoding="utf-8")

    result = _run_collect(root, out)

    assert result.returncode == 0, result.stderr
    rows = _read_rows(out)
    assert len(rows) == 1
    assert rows[0]["eval_frames_advance"] == "0"
    assert rows[0]["mean_score"] == "12.0"
    assert rows[0]["mean_steps"] == "22.0"
    assert rows[0]["return_mean"] == "12.0"
    assert rows[0]["episode_length_mean"] == "22.0"


def test_collects_repo_mean_return_mean_length_format(tmp_path: Path) -> None:
    root = tmp_path / "hf_eval"
    out = tmp_path / "tables" / "eval_summary_all.csv"
    _write_json(
        root / "outputs" / "eval" / "demon_attack" / "latency_15" / "seed_0" / "queue_eval_latency_5.json",
        {
            "experiment_name": "demon_attack_sf_same_latency_l15_seed0",
            "latency": 5,
            "mean_length": 300.83,
            "mean_return": 7.66,
            "returns": [24.0, 5.0],
            "lengths": [915, 217],
        },
    )

    result = _run_collect(root, out)

    assert result.returncode == 0, result.stderr
    rows = _read_rows(out)
    assert rows[0]["mean_score"] == "7.66"
    assert rows[0]["mean_steps"] == "300.83"
    assert rows[0]["return_mean"] == "7.66"
    assert rows[0]["episode_length_mean"] == "300.83"


def test_no_matches_writes_header_and_reports_warning(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    out = tmp_path / "tables" / "eval_summary_all.csv"
    root.mkdir()

    result = _run_collect(root, out)

    assert result.returncode == 0
    assert "No queue_eval_latency_*.json files found" in result.stderr
    with out.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert list(reader) == [HEADER]


def test_no_per_game_disables_per_game_outputs(tmp_path: Path) -> None:
    root = tmp_path / "hf_eval"
    out = tmp_path / "tables" / "eval_summary_all.csv"
    _write_json(
        root / "flappy" / "train_latency_0" / "seed_0" / "queue_eval_latency_1.json",
        {"mean_return": 9, "mean_length": 10},
    )

    result = _run_collect(root, out, "--no-per-game")

    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert not (tmp_path / "tables" / "eval_summary_all_flappy.csv").exists()
