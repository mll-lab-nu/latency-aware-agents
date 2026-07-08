from __future__ import annotations

from latency_bench.core.types import OrderingPolicy
from latency_bench.scheduler.result_coordinator import ResultCoordinator


def _latest_ready() -> ResultCoordinator[str]:
    return ResultCoordinator(ordering_policy=OrderingPolicy.LATEST_READY)


def test_latest_ready_release_voids_older_in_flight_results():
    coordinator = _latest_ready()
    coordinator.mark_submitted(0)
    coordinator.mark_submitted(1)

    assert coordinator.add_completion(1, "r1") == []
    assert coordinator.release_ready() == (["r1"], [])

    # obs 0 was voided by releasing the newer result: its late completion is
    # handed back as ignored and never buffered.
    assert coordinator.add_completion(0, "r0") == ["r0"]
    assert coordinator.release_ready() == ([], [])
    assert coordinator.ignored_late_result_count == 1


def test_latest_ready_discards_older_results_ready_in_same_batch():
    coordinator = _latest_ready()
    coordinator.mark_submitted(0)
    coordinator.mark_submitted(1)
    coordinator.mark_submitted(2)

    assert coordinator.add_completion(0, "r0") == []
    assert coordinator.add_completion(2, "r2") == []
    released, discarded = coordinator.release_ready()

    assert released == ["r2"]
    assert discarded == ["r0"]
    assert coordinator.discarded_ready_result_count == 1
    # obs 1 is still in flight but older than the released result.
    assert coordinator.add_completion(1, "r1") == ["r1"]
    assert coordinator.ignored_late_result_count == 1


def test_fifo_releases_in_issue_order_and_counts_head_of_line_blocking():
    coordinator = ResultCoordinator(ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO)
    coordinator.mark_submitted(0)
    coordinator.mark_submitted(1)

    assert coordinator.add_completion(1, "r1") == []
    assert coordinator.release_ready() == ([], [])
    assert coordinator.fifo_blocked_count == 1

    assert coordinator.add_completion(0, "r0") == []
    assert coordinator.release_ready() == (["r0", "r1"], [])
    assert coordinator.ignored_late_result_count == 0
