"""Long-lived data I/O contracts: bounded writes, ownership and worker failures."""

from threading import Event, Thread
import pytest
from latency_bench.data.background_writer import BackgroundWriter


def test_background_failure_surfaces():
    """The I/O owner drains jobs and preserves the original worker exception."""
    writer = BackgroundWriter()
    def fail():
        raise OSError("disk full")
    writer.submit(fail)
    with pytest.raises(OSError, match="disk full"):
        writer.close()


def test_background_jobs_are_bounded_and_overlap_collection():
    """I/O ownership contract: the third write waits while two unfinished jobs remain."""
    entered, release, submitted = Event(), Event(), Event()
    writer = BackgroundWriter()
    values = []
    def save(value):
        entered.set()
        release.wait()
        values.append(value)
    writer.submit(save, 0)
    assert entered.wait(5)
    writer.submit(save, 1)
    def third():
        writer.submit(save, 2)
        submitted.set()
    producer = Thread(target=third)
    producer.start()
    try:
        assert not submitted.wait(0.05)
    finally:
        release.set()
        producer.join(5)
        writer.close()
    assert submitted.is_set() and values == [0, 1, 2]
