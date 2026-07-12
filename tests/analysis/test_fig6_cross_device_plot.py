from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIG6_SCRIPT = (
    ROOT
    / "paper/latency-sensitive-vla/figures/4_1_cross_device_sim/fig6_cross_device/run.py"
)


def _load_fig6_module():
    spec = importlib.util.spec_from_file_location("fig6_cross_device_run", FIG6_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latency_row(
    *,
    instance_id: str,
    latency_ms: float,
    profile_revision: str = "fit-0",
    latency_kind: str = "e2e_latency",
) -> dict[str, object]:
    return {
        "workload_id": "openvla-flappy",
        "policy_id": "openvla-flappy",
        "env_id": "flappy",
        "device_id": "a100",
        "device_name": "A100",
        "instance_id": instance_id,
        "profile_id": "L_A100",
        "profile_revision": profile_revision,
        "profile_ref": f"a100/{instance_id}/openvla-flappy/{profile_revision}",
        "fit_session_ids": "0,1,2,3,4",
        "sample_index": 0,
        "latency_ms": latency_ms,
        "env_fps": 16,
        "frame_ms": 62.5,
        "latency_kind": latency_kind,
    }


def _result_row(
    *,
    latency_method: str,
    instance_id: str = "gpu-0",
    metric: str = "episode_return_env",
    sim_mean: float = 105.0,
    real_ci95: object = 4.0,
) -> dict[str, object]:
    return {
        "workload_id": "openvla-flappy",
        "policy_id": "openvla-flappy",
        "env_id": "flappy",
        "device_id": "a100",
        "device_name": "A100",
        "instance_id": instance_id,
        "profile_id": "L_A100",
        "profile_revision": "fit-0",
        "profile_ref": f"a100/{instance_id}/openvla-flappy/fit-0",
        "fit_session_ids": "0,1,2,3,4",
        "latency_method": latency_method,
        "metric": metric,
        "evaluation_scope": "heldout",
        "real_session_count": 5,
        "real_episode_count": 500,
        "real_mean": 100.0,
        "real_std": 3.0,
        "real_ci95": real_ci95,
        "sim_seed_count": 20,
        "sim_episode_count": 2000,
        "sim_mean": sim_mean,
        "sim_std": 5.0,
        "sim_ci95": 2.0,
        "sim_minus_real": sim_mean - 100.0,
        "difference_ci95": 5.0,
    }


def test_latency_panel_keeps_instances_separate_and_uses_e2e_samples():
    fig6 = _load_fig6_module()
    rows = [
        _latency_row(instance_id=f"gpu-{instance}", latency_ms=40.0 + instance * 10 + value)
        for instance in range(5)
        for value in (0.0, 2.0, 4.0, 6.0)
    ] + [
        _latency_row(
            instance_id="gpu-0",
            latency_ms=500.0,
            latency_kind="inference_latency",
        ),
    ]
    fig, ax = plt.subplots()

    fig6.plot_latency_profiles(ax, rows)

    assert {line.get_label() for line in ax.get_lines()} == {
        f"A100 (gpu-{instance}/fit-0)" for instance in range(5)
    }
    assert max(max(line.get_xdata()) for line in ax.get_lines()) < 110.0
    plt.close(fig)


def test_latency_panel_keeps_profile_revisions_separate():
    fig6 = _load_fig6_module()
    rows = [
        _latency_row(
            instance_id="gpu-0",
            profile_revision=revision,
            latency_ms=offset + value,
        )
        for revision, offset in (("fit-0", 40.0), ("fit-0-4", 60.0))
        for value in (0.0, 2.0, 4.0, 6.0)
    ]
    fig, ax = plt.subplots()

    fig6.plot_latency_profiles(ax, rows)

    assert {line.get_label() for line in ax.get_lines()} == {
        "A100 (gpu-0/fit-0)",
        "A100 (gpu-0/fit-0-4)",
    }
    plt.close(fig)


def test_return_panel_compares_temporal_and_iid_and_ignores_other_metrics():
    fig6 = _load_fig6_module()
    rows = [
        _result_row(latency_method="temporal", real_ci95=""),
        _result_row(latency_method="iid", sim_mean=120.0),
        _result_row(latency_method="temporal", instance_id="gpu-1", sim_mean=108.0),
        _result_row(latency_method="iid", instance_id="gpu-1", sim_mean=123.0),
        _result_row(latency_method="temporal", metric="survival_steps", sim_mean=300.0),
    ]
    fig, ax = plt.subplots()
    calls = []
    errorbar = ax.errorbar

    def capture_errorbar(x, y, **kwargs):
        calls.append((x, y, kwargs["fmt"], kwargs["xerr"]))
        return errorbar(x, y, **kwargs)

    ax.errorbar = capture_errorbar
    fig6.plot_real_vs_sim(ax, rows)

    assert calls == [
        (100.0, 105.0, "o", None),
        (100.0, 120.0, "s", 4.0),
        (100.0, 108.0, "o", 4.0),
        (100.0, 123.0, "s", 4.0),
    ]
    assert [text.get_text() for text in ax.texts] == [
        "A100/gpu-0/fit-0",
        "A100/gpu-0/fit-0",
        "A100/gpu-1/fit-0",
        "A100/gpu-1/fit-0",
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Temporal", "Iid"]
    plt.close(fig)


def test_main_reads_fixed_analysis_outputs_and_writes_only_figure(tmp_path: Path):
    fig6 = _load_fig6_module()
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "output"
    analysis_dir.mkdir()
    _write_csv(
        analysis_dir / "latency_samples.csv",
        [
            _latency_row(instance_id="gpu-0", latency_ms=value)
            for value in (40.0, 42.0, 44.0, 46.0)
        ],
    )
    _write_csv(
        analysis_dir / "real_vs_sim.csv",
        [
            _result_row(latency_method="temporal"),
            _result_row(latency_method="iid", sim_mean=110.0),
        ],
    )

    fig6.main(["--analysis-dir", str(analysis_dir), "--out-dir", str(output_dir)])

    assert sorted(path.name for path in output_dir.iterdir()) == ["fig.pdf", "fig.png"]
