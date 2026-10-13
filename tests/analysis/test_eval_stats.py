from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.analysis.eval_stats import (
    EPISODE_METRICS_CANDIDATES,
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


def test_summarize_eval_run_uses_requested_fields(tmp_path: Path):
    """Long-lived analysis contract: the selected metric fields determine the summary."""
    path = tmp_path / "episode_metrics.jsonl"
    _write_jsonl(path, [{
        "episode_return_env": 3.0,
        "return_raw": 30.0,
        "survival_steps": 12,
        "steps": 7,
        "metadata": {},
    }])

    summary = summarize_eval_run(path, return_field="return_raw", length_field="steps")

    assert summary["mean_return"] == 30.0
    assert summary["mean_length"] == 7.0
    assert summary["return_field"] == "return_raw"
    assert summary["length_field"] == "steps"


@pytest.mark.parametrize(
    "record,fields,missing",
    [
        ({}, {}, "episode_return_env"),
        ({"episode_return_env": 3.0}, {}, "survival_steps"),
        ({"episode_return_env": 3.0}, {"return_field": "missing"}, "missing"),
        ({"episode_return_env": 3.0}, {"length_field": "missing"}, "missing"),
    ],
)
def test_summarize_eval_run_requires_selected_fields(tmp_path: Path, record, fields, missing):
    """Long-lived analysis contract: missing selected fields fail without substitution."""
    path = tmp_path / "episode_metrics.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(KeyError, match=missing):
        summarize_eval_run(path, **fields)


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
                    "gpu_class": "1x-rtx3090",
                    "env_fps": 25.0,
                },
            },
            {
                "episode_return_env": 110.0,
                "survival_steps": 14,
                "metadata": {
                    "mode": "realtime",
                    "policy_id": "pi0",
                    "gpu_class": "1x-rtx4090",
                    "env_fps": 25.0,
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
    assert summary["gpu_class"] == "1x-rtx3090;1x-rtx4090"
    assert summary["env_fps"] == pytest.approx(25.0)


def test_extract_latency_samples_from_records_uses_latency_ms():
    records = [
        {"latency_ms": 10.0},
        {"latency_ms": 20.0},
        {"latency_ms": 30.0},
        {"other": 1},
    ]

    assert extract_latency_samples_from_records(records) == [10.0, 20.0, 30.0]
