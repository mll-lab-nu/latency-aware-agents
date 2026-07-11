from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from helpers import REPO_ROOT


RUN_PATH = (
    REPO_ROOT
    / "paper"
    / "latency-sensitive-vla"
    / "figures"
    / "4_1_cross_device_sim"
    / "fig6_cross_device"
    / "run.py"
)


def _load_fig6_module():
    spec = importlib.util.spec_from_file_location("fig6_cross_device_run", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_calibration_rows_use_legacy_plot_path():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "row_type": "calibration",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "A100",
            "realtime_return": "30.0",
            "simulated_return_using_L_d": "32.0",
        }
    ]
    figure, ax = plt.subplots()

    fig6.plot_same_device_calibration(ax, rows)

    assert ax.get_title() == "(b) Same-device calibration"
    assert len(ax.collections) == 1
    plt.close(figure)


def test_protocol_calibration_uses_aggregate_session_intervals():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "row_type": "calibration",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "A100",
            "profile_instance_id": "instance_a",
            "profile_build_id": "aaa111build",
            "latency_method": "temporal",
            "sim_host_device_id": "rtx5090",
            "sim_host_instance_id": "instance_5090",
            "realtime_return": "30.0",
            "simulated_return_using_L_d": "32.0",
            "realtime_return_ci95": "4.0",
            "simulated_return_ci95": "3.0",
            "realtime_session_count": "5",
            "simulated_session_count": "20",
        }
    ]
    figure, ax = plt.subplots()

    fig6.plot_same_device_calibration(ax, rows)

    assert ax.get_title() == "(b) RTX 5090-hosted simulation calibration"
    assert any("n=5/20" in text.get_text() for text in ax.texts)
    plt.close(figure)


def test_protocol_calibration_labels_multiple_sim_hosts():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "row_type": "calibration",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "A100",
            "profile_instance_id": "instance_a",
            "profile_build_id": "aaa111build",
            "latency_method": "temporal",
            "sim_host_device_id": "rtx5090",
            "sim_host_instance_id": host,
            "realtime_return": "30.0",
            "simulated_return_using_L_d": simulated_return,
            "realtime_return_ci95": "4.0",
            "simulated_return_ci95": "3.0",
            "realtime_session_count": "5",
            "simulated_session_count": "20",
        }
        for host, simulated_return in (
            ("instance_host_a", "32.0"),
            ("instance_host_b", "34.0"),
        )
    ]
    figure, ax = plt.subplots()

    fig6.plot_same_device_calibration(ax, rows)

    labels = {text.get_text() for text in ax.get_legend().get_texts()}
    assert "A100/a/aaa111@rtx5090/host_a" in labels
    assert "A100/a/aaa111@rtx5090/host_b" in labels
    plt.close(figure)


def test_sampler_ablation_plot_uses_temporal_and_iid_summary_rows():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "A100",
            "profile_instance_id": "instance_a",
            "profile_build_id": "build_a",
            "sim_host_device_id": "rtx5090",
            "sim_host_instance_id": "instance_host",
            "latency_method": method,
            "sim_minus_real_return": difference,
            "sim_minus_real_ci95": "3.0",
        }
        for method, difference in (("temporal", "2.0"), ("iid", "12.0"))
    ]
    figure, ax = plt.subplots()

    fig6.plot_sampler_ablation(ax, rows)

    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["A100"]
    assert {text.get_text() for text in ax.get_legend().get_texts()} == {"temporal", "iid"}
    plt.close(figure)


def test_sampler_ablation_keeps_multiple_instances_of_same_hardware():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "A100",
            "profile_instance_id": instance_id,
            "profile_build_id": build_id,
            "sim_host_device_id": "rtx5090",
            "sim_host_instance_id": f"instance_{instance_id.removeprefix('instance_')}_host",
            "latency_method": method,
            "sim_minus_real_return": difference,
            "sim_minus_real_ci95": "3.0",
        }
        for instance_id, build_id, difference in (
            ("instance_first", "aaa111build", "2.0"),
            ("instance_second", "bbb222build", "7.0"),
        )
        for method in ("temporal", "iid")
    ]
    figure, ax = plt.subplots()

    fig6.plot_sampler_ablation(ax, rows)

    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "A100/first/aaa111@rtx5090/first_host",
        "A100/second/bbb222@rtx5090/second_host",
    ]
    plt.close(figure)


def test_sampler_ablation_keeps_multiple_sim_hosts_for_one_profile():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "A100",
            "profile_instance_id": "instance_a",
            "profile_build_id": "aaa111build",
            "sim_host_device_id": "rtx5090",
            "sim_host_instance_id": host,
            "latency_method": method,
            "sim_minus_real_return": "2.0",
            "sim_minus_real_ci95": "3.0",
        }
        for host in ("instance_host_a", "instance_host_b")
        for method in ("temporal", "iid")
    ]
    figure, ax = plt.subplots()

    fig6.plot_sampler_ablation(ax, rows)

    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "A100/a/aaa111@rtx5090/host_a",
        "A100/a/aaa111@rtx5090/host_b",
    ]
    plt.close(figure)


def test_panel_a_keeps_multiple_instances_of_same_hardware():
    fig6 = _load_fig6_module()
    import matplotlib.pyplot as plt

    rows = [
        {
            "row_type": "e2e_latency_sample",
            "hardware": "A100",
            "profile_instance_id": instance_id,
            "profile_build_id": build_id,
            "latency_ms": str(latency_ms),
        }
        for instance_id, build_id, samples in (
            ("instance_first", "aaa111build", (40.0, 41.0, 42.0)),
            ("instance_second", "bbb222build", (70.0, 71.0, 72.0)),
        )
        for latency_ms in samples
    ]
    figure, ax = plt.subplots()

    fig6.plot_latency_profiles(
        ax,
        rows,
        row_type="e2e_latency_sample",
        xlabel="E2E",
        title="profiles",
    )

    assert {text.get_text() for text in ax.get_legend().get_texts()} == {
        "A100/first/aaa111",
        "A100/second/bbb222",
    }
    plt.close(figure)


def test_empty_sampler_ablation_removes_stale_outputs(tmp_path: Path):
    fig6 = _load_fig6_module()
    outputs = [tmp_path / "sampler_ablation.pdf", tmp_path / "sampler_ablation.png"]
    for output in outputs:
        output.write_text("stale", encoding="utf-8")

    fig6._remove_sampler_ablation_outputs(tmp_path)

    assert not any(output.exists() for output in outputs)


def test_main_renders_panels_a_b_and_sampler_ablation_without_panel_c(tmp_path: Path, monkeypatch):
    fig6 = _load_fig6_module()
    data = tmp_path / "data.csv"
    ablation = tmp_path / "sampler_ablation.csv"
    data.write_text("row_type\ne2e_latency_sample\n", encoding="utf-8")
    ablation.write_text("latency_method\ntemporal\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(fig6, "plot_latency_profiles", lambda *args, **kwargs: calls.append("a"))
    monkeypatch.setattr(fig6, "plot_same_device_calibration", lambda *args, **kwargs: calls.append("b"))
    monkeypatch.setattr(fig6, "plot_cross_device_prediction", lambda *args, **kwargs: calls.append("c"))
    monkeypatch.setattr(fig6, "plot_sampler_ablation", lambda *args, **kwargs: calls.append("ablation"))

    fig6.main([
        "--data",
        str(data),
        "--sampler-ablation",
        str(ablation),
        "--out-dir",
        str(tmp_path),
    ])

    assert calls == ["a", "b", "ablation"]
    assert all((tmp_path / f"fig.{suffix}").exists() for suffix in ("pdf", "png"))
    assert all((tmp_path / f"sampler_ablation.{suffix}").exists() for suffix in ("pdf", "png"))


def test_main_removes_stale_ablation_when_csv_has_no_rows(tmp_path: Path, monkeypatch):
    fig6 = _load_fig6_module()
    data = tmp_path / "data.csv"
    ablation = tmp_path / "sampler_ablation.csv"
    data.write_text("row_type\ne2e_latency_sample\n", encoding="utf-8")
    ablation.write_text("latency_method\n", encoding="utf-8")
    for suffix in ("pdf", "png"):
        (tmp_path / f"sampler_ablation.{suffix}").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(fig6, "plot_latency_profiles", lambda *args, **kwargs: None)
    monkeypatch.setattr(fig6, "plot_same_device_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fig6,
        "plot_sampler_ablation",
        lambda *args, **kwargs: pytest.fail("empty ablation must not render"),
    )

    fig6.main([
        "--data",
        str(data),
        "--sampler-ablation",
        str(ablation),
        "--out-dir",
        str(tmp_path),
    ])

    assert not any((tmp_path / f"sampler_ablation.{suffix}").exists() for suffix in ("pdf", "png"))
