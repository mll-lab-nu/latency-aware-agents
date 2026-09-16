"""Long-lived shared collector contract: native clocks, budgets, ownership and cleanup."""

import json
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from latency_bench.data.dagger import DaggerStep, collect_dagger
from latency_bench.data.background_writer import BackgroundWriter


@pytest.mark.parametrize("mode,budget", [("episode", 4), ("anchor", 4), ("step", 3)])
def test_collection_preserves_sampling_and_budget_semantics(tmp_path, mode, budget):
    class Adapter:
        num_envs = 2
        inference_period = 5 if mode == "anchor" else 2 if mode == "episode" else 1
        record_period = 5 if mode == "anchor" else 1
        budget_mode = mode
        shard_rows = 1000 if mode == "anchor" else None
        seed_mode = "global" if mode == "step" else "slot"
        reset = None

        def __init__(self):
            self.elapsed = [0, 0]
            self.advances = [0, 0]

        def snapshot(self, due, record_slots):
            return self.elapsed.copy()

        def observations(self, snapshot, slots, state):
            return [(slot, state.seeds[slot], state.control_step) for slot in slots]

        def start_records(self, snapshot, slots, state):
            return [{"slot": slot, "label": snapshot[slot], "seed": state.seeds[slot], "trace": []}
                    for slot in slots]

        def step(self, snapshot, active, state):
            for slot in active:
                assert state.outputs[slot].slot == slot
                self.elapsed[slot] += 1
                self.advances[slot] += 1
            done = [slot for slot in active if self.elapsed[slot] == (6 if slot == 0 else 9)]
            for slot in done:
                self.elapsed[slot] = 0
            return DaggerStep(done, None)

        def finish_record(self, row, snapshot, result, slot, finished):
            row["trace"].append(snapshot[slot])
            if slot in result.done:
                row["trace"].extend([snapshot[slot]] * (self.record_period - len(row["trace"])))

        def write_rows(self, shard_index, slot, rows, writer):
            writer.submit((tmp_path / f"{shard_index:06d}.json").write_text, json.dumps(rows))

    requests = []
    closed = []
    def predict(observations):
        requests.extend(observations)
        return [SimpleNamespace(slot=slot) for slot, _, _ in observations]

    adapter = Adapter()
    pool = SimpleNamespace(predict_batch=predict, close=lambda: closed.append(True))
    assert collect_dagger(adapter, pool, row_budget=budget, seed=100, is_running=lambda: True) == budget
    rows = [row for path in sorted(tmp_path.glob("*.json")) for row in json.loads(path.read_text())]
    assert len(rows) == budget and closed == [True]
    assert all(row["label"] == row["trace"][0] for row in rows)
    if mode == "episode":
        assert [row["slot"] for row in rows] == [0] * 4
        assert [row["label"] for row in rows] == [0, 1, 2, 3]
        assert adapter.advances == [6, 6]
    elif mode == "anchor":
        assert [row["label"] for row in rows] == [0, 5, 0, 5]
        assert [row["trace"] for row in rows] == [
            [0, 1, 2, 3, 4], [5] * 5, [0, 1, 2, 3, 4], [5, 6, 7, 8, 8],
        ]
        assert adapter.advances == [9, 9]
    else:
        assert adapter.advances == [2, 1]
        assert [row["seed"] for row in rows] == [100, 101, 100]
    assert requests[0][0] == 0 and requests[1][0] == 1


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
