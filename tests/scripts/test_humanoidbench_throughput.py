import json

from scripts.experiments import humanoidbench_throughput as throughput


def test_skip_raw_reuses_only_selected_cases(tmp_path, monkeypatch):
    existing_rows = [
        {
            "measurement": "raw",
            "task_name": task,
            "env_count": env_count,
            "aggregate_fps": 100.0,
            "speedup_vs_1": env_count,
            "scaling_efficiency": 1.0,
            "realtime_env_capacity": 2.0,
        }
        for task in throughput.DEFAULT_TASKS
        for env_count in throughput.DEFAULT_ENV_COUNTS
    ]
    existing_rows.extend(
        {"measurement": "raw_rgb", "task_name": task, "env_count": 1, "aggregate_fps": 10.0}
        for task in throughput.DEFAULT_TASKS
    )
    (tmp_path / "throughput.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in existing_rows), encoding="utf-8"
    )

    monkeypatch.setattr(
        throughput,
        "run_sample_factory_case",
        lambda **kwargs: {
            "measurement": "sample_factory",
            "task_name": kwargs["task_name"],
            "env_count": kwargs["env_count"],
            "aggregate_fps": 40.0,
            "steady_state_fps_10s": 50.0,
        },
    )
    monkeypatch.setattr(
        "scripts.scaffold_humanoidbench.build_probe_report", lambda task_name: {}
    )
    monkeypatch.setattr(
        "scripts.scaffold_humanoidbench.build_env_config", lambda task_name, report: {}
    )

    rows = throughput.run_benchmark(
        output_dir=tmp_path,
        tasks=("h1hand_balance_hard",),
        env_counts=(4,),
        warmup_seconds=0.0,
        measure_seconds=1.0,
        skip_sample_factory=False,
        skip_raw=True,
    )

    assert [(row["measurement"], row["task_name"], row["env_count"]) for row in rows] == [
        ("raw", "h1hand_balance_hard", 4),
        ("raw_rgb", "h1hand_balance_hard", 1),
        ("sample_factory", "h1hand_balance_hard", 4),
    ]
    assert "speedup_vs_1" not in rows[0]
