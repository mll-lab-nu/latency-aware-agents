from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_latency_bench_env_backend.py"
SPEC = importlib.util.spec_from_file_location("bench_latency_bench_env_backend", MODULE_PATH)
assert SPEC is not None
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench)


def _base_config(output_root: Path) -> dict:
    return {
        "experiment": {
            "name": "bench_test",
        },
        "env": {
            "name": "demon_attack",
        },
        "policy": {
            "type": "demon_attack_sf",
        },
        "latency": {
            "type": "zero",
        },
        "evaluation": {
            "eval_episodes": 3,
        },
        "logging": {
            "output_dir": str(output_root),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def test_parse_parallel_envs_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        bench._parse_parallel_envs("1,0,4")


def test_parse_backends_accepts_flappy_gpu_batched_backend():
    assert bench._parse_backends("serial,flappy_gpu_batched,auto") == [
        "serial",
        "flappy_gpu_batched",
        "auto",
    ]


def test_stage_configs_expand_parallel_envs_and_record_correctness_metadata(tmp_path: Path):
    stages = bench._stage_configs(
        _base_config(tmp_path),
        backend="auto",
        output_root=tmp_path / "out",
        requested_parallel_envs=[1, 4],
    )

    assert len(stages) == 2
    first_latency_steps, _first_latency_ms, first_config, first_metadata = stages[0]
    second_latency_steps, _second_latency_ms, second_config, second_metadata = stages[1]

    assert first_latency_steps is None
    assert second_latency_steps is None
    assert first_config["evaluation"]["eval_parallel_envs"] == 1
    assert second_config["evaluation"]["eval_parallel_envs"] == 4
    assert first_config["evaluation"]["latency_bench_env_backend"] == "auto"
    assert second_config["evaluation"]["latency_bench_env_backend"] == "auto"
    assert first_config["logging"]["output_dir"].endswith("parallel_1/auto/current_latency")
    assert second_config["logging"]["output_dir"].endswith("parallel_4/auto/current_latency")
    assert first_metadata == {
        "requested_parallel_envs": 1,
        "effective_parallel_envs": 1,
        "expected_episodes": 3,
    }
    assert second_metadata == {
        "requested_parallel_envs": 4,
        "effective_parallel_envs": 3,
        "expected_episodes": 3,
    }


def test_run_stage_redirects_eval_stdout_to_stderr(monkeypatch, capsys, tmp_path: Path):
    def fake_run_from_config(_config, *, extra_metadata, write_summary):
        print("latency_bench summary noise")
        return [object(), object(), object()]

    monkeypatch.setattr(bench, "run_from_config", fake_run_from_config)

    completed, expected, completed_ok, _wall_sec, _cpu_sec = bench._run_stage(
        _base_config(tmp_path),
        extra_metadata={"expected_episodes": 3},
        write_summary=True,
    )

    captured = capsys.readouterr()
    assert "latency_bench summary noise" not in captured.out
    assert "latency_bench summary noise" in captured.err
    assert completed == 3
    assert expected == 3
    assert completed_ok is True
