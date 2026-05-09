from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.analysis.eval_stats import (
    EPISODE_METRICS_CANDIDATES,
    extract_action_age_frames_from_records,
    extract_episode_length,
    extract_episode_return,
    extract_latency_samples_from_records,
    load_jsonl_records,
    resolve_run_file,
    summarize_eval_run,
)


def _write_jsonl(path: Path, records: list[dict], *, blank_line: bool = False) -> None:
    lines: list[str] = []
    for index, record in enumerate(records):
        lines.append(json.dumps(record))
        if blank_line and index == 0:
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_jsonl_records_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [{"a": 1}, {"b": 2}], blank_line=True)

    assert load_jsonl_records(path) == [{"a": 1}, {"b": 2}]


def test_load_jsonl_records_fails_clearly_on_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": 1}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON.*bad.jsonl:2"):
        load_jsonl_records(path)


def test_load_jsonl_records_fails_clearly_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="JSONL file does not exist"):
        load_jsonl_records(tmp_path / "missing.jsonl")


def test_resolve_run_file_supports_direct_file_and_run_directory(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics_path = run_dir / "episode_metrics.jsonl"
    _write_jsonl(metrics_path, [{"episode_return_env": 1.0, "survival_steps": 2}])

    assert resolve_run_file(metrics_path, EPISODE_METRICS_CANDIDATES) == metrics_path
    assert resolve_run_file(run_dir, EPISODE_METRICS_CANDIDATES) == metrics_path


def test_extract_episode_return_defaults_to_env_return_before_legacy_or_raw():
    record = {
        "episode_return_env": 3.0,
        "episode_return": 4.0,
        "return_raw": 30.0,
    }

    assert extract_episode_return(record) == pytest.approx(3.0)
    assert extract_episode_return(record, return_field="return_raw") == pytest.approx(30.0)
    assert extract_episode_return({"return_raw": 30.0}) == pytest.approx(30.0)
    assert extract_episode_return(record, return_field="missing") is None


def test_extract_episode_length_supports_existing_survival_steps_key():
    assert extract_episode_length({"survival_steps": 12}) == 12
    assert extract_episode_length({"episode_length": 10, "survival_steps": 12}) == 10
    assert extract_episode_length({"steps": 7}, length_field="steps") == 7
    assert extract_episode_length({"survival_steps": 12}, length_field="missing") is None


def test_summarize_eval_run_computes_population_stats_and_metadata(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "episode_metrics.jsonl",
        [
            {
                "episode_return_env": 100.0,
                "survival_steps": 10,
                "metadata": {
                    "mode": "realtime",
                    "policy_id": "pi0",
                    "target_device": "RTX_3090",
                    "fps": 25.0,
                },
            },
            {
                "episode_return_env": 110.0,
                "survival_steps": 14,
                "metadata": {
                    "mode": "realtime",
                    "policy_id": "pi0",
                    "target_device": "RTX_4090",
                    "fps": 25.0,
                },
            },
        ],
    )

    summary = summarize_eval_run(run_dir)

    assert summary["n_episodes"] == 2
    assert summary["mean_return"] == pytest.approx(105.0)
    assert summary["std_return"] == pytest.approx(5.0)
    assert summary["min_return"] == pytest.approx(100.0)
    assert summary["max_return"] == pytest.approx(110.0)
    assert summary["mean_length"] == pytest.approx(12.0)
    assert summary["std_length"] == pytest.approx(2.0)
    assert summary["min_length"] == pytest.approx(10.0)
    assert summary["max_length"] == pytest.approx(14.0)
    assert summary["return_field"] == "episode_return_env"
    assert summary["length_field"] == "survival_steps"
    assert summary["mode"] == "realtime"
    assert summary["policy_id"] == "pi0"
    assert summary["target_device"] == "RTX_3090;RTX_4090"
    assert summary["fps"] == pytest.approx(25.0)


def test_extract_latency_samples_from_records_supports_auto_measured_and_sampled():
    records = [
        {"latency_ms": 10.0, "latency_ms_measured": 99.0},
        {"latency_ms_measured": 20.0},
        {"latency_ms_sampled": 30.0},
        {"other": 1},
    ]

    assert extract_latency_samples_from_records(records, source="auto") == [10.0, 20.0, 30.0]
    assert extract_latency_samples_from_records(records, source="measured") == [99.0, 20.0]
    assert extract_latency_samples_from_records(records, source="sampled") == [30.0]
    assert extract_latency_samples_from_records(records, source="latency_ms") == [10.0]


def test_extract_latency_samples_from_records_raises_when_no_valid_samples():
    with pytest.raises(ValueError, match="No valid latency samples"):
        extract_latency_samples_from_records([{"other": 1}], source="sampled")


def test_extract_action_age_frames_from_records_ignores_missing_values_by_default():
    records = [
        {"action_age_frames": 0},
        {"action_age_frames": None},
        {},
        {"action_age_frames": 2.0},
    ]

    assert extract_action_age_frames_from_records(records) == [0, 2]
    assert extract_action_age_frames_from_records([{}]) == []
    with pytest.raises(ValueError, match="No valid action_age_frames"):
        extract_action_age_frames_from_records([{}], strict=True)
