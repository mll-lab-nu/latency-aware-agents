import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.experiments.memory_mock import measure
from scripts.experiments.memory_mock.report import report


def test_preparation_freezes_matching_seed_and_preserves_realtime_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    args = SimpleNamespace(config=measure.ROOT / 'configs/examples/flappy/profile_openvla.yaml',
        output=tmp_path / 'run', condition='irregular', episodes=300, episode_seed=11,
        load_seed=42, buffer_mib=512, startup_timeout_seconds=1800, run_timeout_seconds=7200,
        coverage_seconds=7200., off_seconds=(5., 15.), short_on_seconds=(.2, .8),
        long_on_seconds=(2., 5.), long_probability=.2)
    protocol = measure.prepare(args)
    from latency_bench.core.config import load_config
    config = load_config(args.output / 'evaluation.yaml')
    assert config['executor']['mode'] == 'realtime'
    assert config['executor']['inference_devices'] == ['cuda:0']
    assert config['env']['env_fps'] == config['env']['obs_fps'] == 30
    assert config['experiment']['seed'] == 11
    assert config['evaluation']['episodes'] == 300
    assert protocol['cuda_visible_devices'] == '2'
    assert json.loads((args.output / 'schedule.json').read_text())['load_seed'] == 42
    assert (args.output / 'code/latency_bench/eval/driver.py').is_file()
    with pytest.raises(FileExistsError):
        measure.prepare(args)


def start_child(tmp_path, name, source):
    path = tmp_path / (name + '.py')
    path.write_text(source)
    return subprocess.Popen([sys.executable, str(path)], start_new_session=True)


def test_supervisor_detects_failed_load_and_owned_children_are_stopped(tmp_path):
    evaluation = start_child(tmp_path, 'evaluation', 'import time\ntime.sleep(30)\n')
    load = start_child(tmp_path, 'load', 'raise SystemExit(3)\n')
    try:
        load.wait(timeout=5)
        with pytest.raises(RuntimeError, match='Memory workload exited unexpectedly'):
            measure.supervise(evaluation, load, tmp_path, startup_timeout=5, run_timeout=5)
    finally:
        measure.stop_process_group(evaluation)
        measure.stop_process_group(load)
    assert evaluation.poll() is not None


def test_supervisor_handles_success_and_startup_timeout(tmp_path):
    load = start_child(tmp_path, 'load', 'import time\ntime.sleep(30)\n')
    evaluation = start_child(tmp_path, 'evaluation', 'import time\ntime.sleep(30)\n')
    try:
        with pytest.raises(TimeoutError, match='startup timeout'):
            measure.supervise(evaluation, load, tmp_path, startup_timeout=0, run_timeout=5)
        measure.stop_process_group(evaluation)
        evaluation = start_child(tmp_path, 'successful_evaluation', '')
        evaluation.wait(timeout=5)
        (tmp_path / 'measurement_started.json').write_text('{}')
        measure.supervise(evaluation, load, tmp_path, startup_timeout=5, run_timeout=5)
    finally:
        measure.stop_process_group(evaluation)
        measure.stop_process_group(load)


def test_report_uses_worker_interval_overlap_and_retains_boundary_crossings(tmp_path):
    (tmp_path / 'load').mkdir()
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'protocol.json').write_text(json.dumps({'condition': 'irregular', 'env_fps': 60}))
    events = [
        {'event': 'switch', 'phase_index': 0, 'on': False},
        {'event': 'phase_end', 'phase_index': 0, 'actual_start_time_ms': 0, 'actual_end_time_ms': 100},
        {'event': 'switch', 'phase_index': 1, 'on': True},
        {'event': 'phase_end', 'phase_index': 1, 'actual_start_time_ms': 100, 'actual_end_time_ms': 200},
    ]
    (tmp_path / 'load/load.jsonl').write_text('\n'.join(map(json.dumps, events)))
    rows = [
        {'obs_id': 0, 'episode_id': 0, 'observation_start_time_ms': 10, 'submitted': True,
         'observation_to_action_latency_ms': 20, 'worker_service_time_ms': 15},
        {'obs_id': 1, 'episode_id': 0, 'observation_start_time_ms': 90, 'submitted': True,
         'observation_to_action_latency_ms': 30, 'worker_service_time_ms': 25},
        {'obs_id': 2, 'episode_id': 0, 'observation_start_time_ms': 110, 'submitted': True,
         'observation_to_action_latency_ms': 50, 'worker_service_time_ms': 40},
        {'obs_id': 3, 'episode_id': 0, 'observation_start_time_ms': 150, 'submitted': False},
    ]
    (tmp_path / 'evaluation/observation_attempts.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    result = report(tmp_path)
    assert result['groups']['OFF']['worker']['mean_ms'] == 15
    assert result['groups']['ON']['worker']['mean_ms'] == 40
    assert result['groups']['MIXED']['worker']['mean_ms'] == 25
    assert result['dropped_observations_by_load_at_capture'] == {'OFF': 0, 'ON': 1}
    assert result['measured_requests'] == 3
    assert result['on_fraction_of_observed_span'] == pytest.approx(60 / 150)
    assert result['action_raw_frame_counts'] == {'2': 2, '3': 1}


def test_idle_report_marks_missing_on_evidence(tmp_path):
    (tmp_path / 'load').mkdir()
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'protocol.json').write_text(json.dumps({'condition': 'idle', 'env_fps': 10}))
    (tmp_path / 'load/load.jsonl').write_text(json.dumps({'event': 'switch', 'phase_index': 0, 'on': False}))
    row = {'obs_id': 0, 'episode_id': 0, 'observation_start_time_ms': 10, 'submitted': True,
           'observation_to_action_latency_ms': 20, 'worker_service_time_ms': 15}
    (tmp_path / 'evaluation/observation_attempts.jsonl').write_text(json.dumps(row))
    result = report(tmp_path)
    assert result['groups']['ON']['worker']['mean_ms'] is None
    assert result['on_fraction_of_observed_span'] == 0
