import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments.memory_mock import load


def schedule_file(path, durations):
    phases, end = [], 0.0
    for index, duration in enumerate(durations):
        phases.append({"phase_index": index, "phase": "ON" if index % 2 else "OFF", "duration_s": duration,
                       "planned_start_offset_s": end, "planned_end_offset_s": end + duration})
        end += duration
    path.write_text(json.dumps({"phases": phases, "load_seed": 42, "numpy_version": np.__version__}))
    return path


def test_frozen_schedule_preserves_rng_and_unclipped_durations():
    old = np.random.get_state()
    result = load.frozen_schedule(42)
    new = np.random.get_state()
    np.testing.assert_array_equal(old[1], new[1])
    assert old[0] == new[0] and old[2:] == new[2:]
    phases = result["phases"]
    expected = np.random.Generator(np.random.PCG64(42)).uniform(0, 10, len(phases))
    assert [row["duration_s"] for row in phases] == expected.tolist()
    assert result["coverage_seconds"] >= 7200 > phases[-1]["planned_start_offset_s"]
    end = 0
    for index, row in enumerate(phases):
        assert row["phase_index"] == index
        assert row["phase"] == ("ON" if index % 2 else "OFF")
        assert row["planned_start_offset_s"] == end
        end += row["duration_s"]
        assert row["planned_end_offset_s"] == end


@pytest.mark.parametrize("elapsed,index", [(0, 0), (4999.99, 0), (5000, 1), (10000, 2), (15000, 3)])
def test_periodic_boundaries_do_not_accumulate_sleep_drift(elapsed, index):
    assert load.periodic_position(1000 + elapsed, 1000) == (index, bool(index % 2), 1000 + (index + 1) * 5000)


@pytest.mark.parametrize("elapsed,index", [(0, 1), (0.019, 1), (0.02, 2), (0.031, 4), (0.9, 4)])
def test_frozen_absolute_boundaries_skip_zero_and_expired_phases(elapsed, index):
    deadlines = [1000 + offset for offset in [0, 0.02, 0.03, 0.03, 1]]
    assert load.schedule_position(1000 + elapsed, deadlines) == (index, bool(index % 2), deadlines[index])
    with pytest.raises(RuntimeError, match="exhausted"):
        load.schedule_position(1001, deadlines)


@pytest.mark.parametrize("field,value", [("phase", "ON"), ("duration_s", -1), ("duration_s", float("nan")),
                                         ("planned_start_offset_s", 1), ("planned_end_offset_s", 8)])
def test_bad_external_schedule_fails_before_cuda(tmp_path, field, value):
    path = schedule_file(tmp_path / "schedule.json", [5, 5])
    data = json.loads(path.read_text())
    data["phases"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="phase 0"):
        load.run(tmp_path / "load", "schedule", 1, path, None)
    assert not (tmp_path / "load").exists()


@pytest.mark.parametrize("mode", ["idle", "periodic", "schedule"])
def test_load_keeps_equal_buffers_and_bounds_copies_to_one_synchronized_pair(tmp_path, monkeypatch, mode):
    clock = {"ms": 1000.0, "stopped": False}
    buffers, copies, synchronizations = [], [], []

    def wait(timeout=None):
        if timeout is None:
            clock["stopped"] = True
        else:
            clock["ms"] += timeout * 1000
        return clock["stopped"]

    event = SimpleNamespace(set=lambda: clock.update(stopped=True), is_set=lambda: clock["stopped"], wait=wait)

    def allocate(size, **kwargs):
        buffer = SimpleNamespace(size=size)
        buffer.fill_ = lambda value: None
        buffer.copy_ = lambda source, **kwargs: copies.append((buffer, source))
        buffers.append(buffer)
        return buffer

    def synchronize():
        synchronizations.append(len(copies))
        if copies:
            clock.update(ms=clock["ms"] + 6000, stopped=True)

    torch = SimpleNamespace(empty=allocate, empty_like=lambda first: allocate(first.size), uint8="uint8",
                            cuda=SimpleNamespace(synchronize=synchronize, get_device_name=lambda _: "test GPU"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(load, "monotonic_ms", lambda: clock["ms"])
    monkeypatch.setattr(load.threading, "Event", lambda: event)
    monkeypatch.setattr(load.signal, "signal", lambda *_: None)
    schedule = schedule_file(tmp_path / "schedule.json", [5, 5, 5]) if mode == "schedule" else None
    output = tmp_path / "load"
    load.run(output, mode, 1, schedule, None)
    records = [json.loads(line) for line in (output / "load.jsonl").read_text().splitlines()]
    assert [buffer.size for buffer in buffers] == [1024**2, 1024**2]
    assert records[0]["buffer_bytes_each"] == 1024**2 and records[0]["buffers"] == 2
    assert (output / "load.ready").exists() and records[-1]["event"] == "stop"
    if mode == "idle":
        assert not copies and synchronizations == [0]
        assert not any(row["event"] == "work" for row in records)
    else:
        assert copies == [(buffers[0], buffers[1]), (buffers[1], buffers[0])]
        assert synchronizations == [0, 2]
        assert [row["on"] for row in records if row["event"] == "switch"] == [False, True]
        work = next(row for row in records if row["event"] == "work")
        assert work["copy_operations"] == 2 and work["copied_bytes"] == 2 * 1024**2
        overrun = next(row for row in records if row["event"] == "copy_overrun")
        assert overrun["first_submit_time_ms"] < overrun["planned_boundary_ms"] < overrun["last_completed_time_ms"]


def test_sigterm_while_waiting_never_initializes_cuda(tmp_path):
    output = tmp_path / "load"
    process = subprocess.Popen([sys.executable, "-m", "scripts.experiments.memory_mock.load", "run",
                                "--mode", "periodic", "--output", str(output),
                                "--wait-for", str(tmp_path / "warmup.done")],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        deadline = time.monotonic() + 5
        while not (output / "load.jsonl").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (output / "load.jsonl").exists()
        process.terminate()
        assert process.wait(timeout=5) == 0
        assert (output / "load.jsonl").read_text() == ""
        assert not (output / "load.ready").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_existing_run_is_preserved_without_touching_cuda(tmp_path):
    (tmp_path / "load.jsonl").write_text("previous run\n")
    with pytest.raises(FileExistsError):
        load.run(tmp_path, "periodic", 512, None, None)
    assert (tmp_path / "load.jsonl").read_text() == "previous run\n"


@pytest.mark.parametrize("args", [["--buffer-mib", "0"], ["--buffer-mib", "-1"],
                                  ["--mode", "schedule"], ["--schedule", "unused.json"]])
def test_invalid_cli_fails_before_creating_output(tmp_path, monkeypatch, args):
    output = tmp_path / "load"
    monkeypatch.setattr(sys, "argv", ["memory-mock", "run", "--output", str(output), "--mode", "periodic", *args])
    with pytest.raises(SystemExit) as result:
        load.main()
    assert result.value.code == 2
    assert not output.exists()


def test_schedule_command_does_not_overwrite_files(tmp_path, monkeypatch):
    output = tmp_path / "schedule.json"
    monkeypatch.setattr(sys, "argv", ["memory-mock", "schedule", "--seed", "42", "--output", str(output)])
    load.main()
    original = output.read_text()
    assert load.read_schedule(output) == load.frozen_schedule(42)
    with pytest.raises(FileExistsError):
        load.main()
    assert output.read_text() == original


def test_irregular_schedule_is_frozen_and_covers_both_duration_scales(tmp_path):
    schedule = load.irregular_schedule(42)
    assert schedule == load.irregular_schedule(42)
    assert schedule != load.irregular_schedule(43)
    assert schedule['coverage_seconds'] >= 7200
    assert schedule['phases'][-1]['planned_start_offset_s'] < 7200
    kinds = set()
    limits = {'quiet': (5, 15), 'short': (.2, .8), 'long': (2, 5)}
    for row in schedule['phases']:
        kinds.add(row['event_kind'])
        low, high = limits[row['event_kind']]
        assert low <= row['duration_s'] <= high
        assert (row['phase'] == 'OFF') == (row['event_kind'] == 'quiet')
    assert kinds == set(limits)
    path = tmp_path / 'irregular.json'
    path.write_text(json.dumps(schedule))
    assert load.read_schedule(path) == schedule


@pytest.mark.parametrize('probability,kind', [(0, 'short'), (1, 'long')])
def test_irregular_mixture_endpoints(probability, kind):
    schedule = load.irregular_schedule(42, coverage_seconds=100, long_probability=probability)
    assert {p['event_kind'] for p in schedule['phases'] if p['phase'] == 'ON'} == {kind}


@pytest.mark.parametrize('settings', [
    {'off_seconds': (0, 1)}, {'short_on_seconds': (2, 1)},
    {'long_on_seconds': (1, float('inf'))}, {'coverage_seconds': 0},
    {'coverage_seconds': float('nan')}, {'long_probability': 1.1},
])
def test_irregular_invalid_settings_fail(settings):
    with pytest.raises(ValueError):
        load.irregular_schedule(42, **settings)
