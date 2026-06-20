from __future__ import annotations

from types import SimpleNamespace

from latency_bench.analysis.temporal_fit import roll_latency_sampler, temporal_fit_report


def test_temporal_fit_report_compares_iid_and_temporal_burst_fit():
    real = [40.0] * 10 + [80.0] * 6 + [40.0] * 10 + [80.0] * 6 + [40.0] * 10
    iid = [40.0, 80.0] * 16 + [40.0] * 10
    temporal = [40.0] * 9 + [80.0] * 7 + [40.0] * 10 + [80.0] * 6 + [40.0] * 10

    report = temporal_fit_report(
        real_e2e_latency_ms=real,
        real_worker_service_ms=[value - 1.0 for value in real],
        iid_e2e_latency_ms=iid,
        temporal_e2e_latency_ms=temporal,
        temporal_worker_service_ms=[value - 1.0 for value in temporal],
        fps=25.0,
    )

    assert set(report) == {"marginal", "temporal", "frame_level", "joint"}
    assert report["temporal"]["temporal"]["acf_abs_error_lag_1"] < report["temporal"]["iid"]["acf_abs_error_lag_1"]
    assert (
        report["temporal"]["temporal"]["high_tail_run_w1"]
        < report["temporal"]["iid"]["high_tail_run_w1"]
    )
    assert "action_ready_frames" in report["frame_level"]["temporal"]
    assert report["joint"]["temporal"]["pearson_corr"] > 0.9


class _FakeSampler:
    def __init__(self) -> None:
        self.index = 0

    def sample(self, context: dict) -> SimpleNamespace:
        self.index += 1
        return SimpleNamespace(
            latency_ms=float(10 * self.index),
            worker_service_ms=float(10 * self.index + 1),
            action_ready_raw_frames=self.index,
            worker_service_raw_frames=self.index + 1,
        )


def test_roll_latency_sampler_exports_sequences_without_env_simulation():
    rolled = roll_latency_sampler(sampler=_FakeSampler(), episode_lengths=[2, 1])

    assert rolled == {
        "e2e_latency_ms": [10.0, 20.0, 30.0],
        "worker_service_ms": [11.0, 21.0, 31.0],
        "action_ready_raw_frames": [1, 2, 3],
        "worker_service_raw_frames": [2, 3, 4],
    }
