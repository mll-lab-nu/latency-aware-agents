"""Task-independent online DAgger collection; native adapters own dynamics/schema."""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from latency_bench.core.latency_distribution import derive_seed
from latency_bench.data.background_writer import BackgroundWriter


@dataclass
class DaggerState:
    """Collector-owned clocks, episode identities, seeds, and current policy outputs."""

    control_step: int
    episode_steps: list[int]
    episode_counts: list[int]
    episode_ids: list[int]
    seeds: list[int]
    outputs: list[Any]


@dataclass
class DaggerStep:
    """Native step result and slots whose native episodes ended."""

    done: list[int]
    data: Any


class DaggerTaskAdapter(Protocol):
    """Translate native observations/actions/records without owning the collection loop.

    Record arrays must own their CPU storage before background handoff. Automatic
    reset tasks leave reset=None; manual reset tasks provide reset(slot, state).
    Episode budgets count submitted rows, anchor budgets reserve rows at sampling,
    and step budgets stop advancing independent slots at the exact quota.
    """

    num_envs: int
    inference_period: int
    record_period: int
    budget_mode: Literal["episode", "anchor", "step"]
    shard_rows: int | None
    seed_mode: Literal["slot", "global"]
    reset: Callable | None

    def snapshot(self, due, record_slots): ...
    def observations(self, snapshot, slots, state: DaggerState): ...
    def start_records(self, snapshot, slots, state: DaggerState): ...
    def step(self, snapshot, active, state: DaggerState) -> DaggerStep: ...
    def finish_record(self, row, snapshot, result: DaggerStep, slot, finished): ...
    def write_rows(self, shard_index, slot, rows, writer: BackgroundWriter): ...


def collect_dagger(adapter: DaggerTaskAdapter, pool, *, row_budget, seed, is_running):
    """Collect student-visited states with aligned pre-action teacher labels."""
    slots = list(range(adapter.num_envs))
    state = DaggerState(0, [0] * len(slots), [0] * len(slots), slots.copy(),
                        [0] * len(slots), [None] * len(slots))
    next_episode = len(slots)
    buffers = [[] for _ in slots]
    records = [None for _ in slots]
    started = completed = submitted = shard_index = 0

    def reset(slot):
        state.seeds[slot] = (
            derive_seed(seed, vector_index=slot, episode_idx=state.episode_counts[slot])
            if adapter.seed_mode == "slot" else seed + state.episode_ids[slot]
        )
        if adapter.reset is not None:
            adapter.reset(slot, state)
            pool.reset_state(slot)

    with ExitStack() as cleanup:
        cleanup.callback(pool.close)
        writer = BackgroundWriter()
        cleanup.callback(writer.close)

        def flush(slot):
            nonlocal submitted, shard_index
            rows = buffers[slot]
            if adapter.budget_mode == "episode":
                rows = rows[:row_budget - submitted]
            if rows:
                adapter.write_rows(shard_index, slot, rows, writer)
                submitted += len(rows)
                shard_index += 1
            buffers[slot] = []

        try:
            for slot in slots:
                reset(slot)
            while (submitted if adapter.budget_mode == "episode" else completed) < row_budget and is_running():
                writer.poll()
                active = slots[:row_budget - completed] if adapter.budget_mode == "step" else slots
                due = [slot for slot in active if state.episode_steps[slot] % adapter.inference_period == 0]
                record_slots = [slot for slot in active if state.episode_steps[slot] % adapter.record_period == 0]
                if adapter.budget_mode == "anchor":
                    record_slots = record_slots[:row_budget - started]
                snapshot = adapter.snapshot(due, record_slots)
                if due:
                    outputs = pool.predict_batch(adapter.observations(snapshot, due, state))
                    for slot, output in zip(due, outputs, strict=True):
                        state.outputs[slot] = output
                if record_slots:
                    for slot, row in zip(record_slots, adapter.start_records(snapshot, record_slots, state), strict=True):
                        records[slot] = row
                started += len(record_slots)
                result = adapter.step(snapshot, active, state)
                done = set(result.done)
                for slot in active:
                    state.episode_steps[slot] += 1
                    finished = slot in done or state.episode_steps[slot] % adapter.record_period == 0
                    row = records[slot]
                    if row is not None:
                        adapter.finish_record(row, snapshot, result, slot, finished)
                        if finished:
                            buffers[slot].append(row)
                            records[slot] = None
                            completed += 1
                    if slot in done or adapter.budget_mode == "step" or (
                        adapter.shard_rows is not None and len(buffers[slot]) == adapter.shard_rows
                    ):
                        flush(slot)
                state.control_step += 1
                for slot in result.done:
                    state.episode_counts[slot] += 1
                    state.episode_ids[slot] = next_episode
                    state.episode_steps[slot] = 0
                    next_episode += 1
                    reset(slot)
        finally:
            for slot in slots:
                flush(slot)
    return submitted
