"""Describe measured ON/OFF inference timing; never infer a model-comparison winner."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from latency_bench.core.clock import ms_to_raw_frames


def records(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def on_intervals(events):
    switches = {row['phase_index']: row for row in events if row['event'] == 'switch'}
    return [(row['actual_start_time_ms'], row['actual_end_time_ms']) for row in events
            if row['event'] == 'phase_end' and switches[row['phase_index']]['on']]


def overlap_ms(start, end, intervals):
    return sum(max(0., min(end, right) - max(start, left)) for left, right in intervals)


def stats(values):
    if not values:
        return {'count': 0, 'mean_ms': None, 'median_ms': None, 'p95_ms': None, 'max_ms': None}
    return {'count': len(values), 'mean_ms': float(np.mean(values)),
            'median_ms': float(np.median(values)), 'p95_ms': float(np.quantile(values, .95)),
            'max_ms': float(max(values))}


def burst_stats(values, threshold):
    mask = np.asarray(values) > threshold
    changes = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    predecessors = int(mask[:-1].sum())
    return {'threshold_ms': threshold, 'slow_requests': int(mask.sum()),
            'slow_fraction': float(mask.mean()),
            'next_slow_given_slow': float((mask[:-1] & mask[1:]).sum() / predecessors) if predecessors else None,
            'run_count': len(lengths), 'runs_ge3': int((lengths >= 3).sum()),
            'longest_run_requests': int(lengths.max()) if len(lengths) else 0}


def report(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    events = list(records(output / 'load/load.jsonl'))
    windows = on_intervals(events)
    groups = {phase: {'action': [], 'worker': []} for phase in ('OFF', 'ON', 'MIXED')}
    drops = {'OFF': 0, 'ON': 0}
    series = []
    first_ms = last_ms = None
    for row in records(output / 'evaluation/observation_attempts.jsonl'):
        observed = row['observation_start_time_ms']
        if first_ms is None:
            first_ms = observed
        last_ms = max(observed, last_ms if last_ms is not None else observed)
        if not row['submitted']:
            phase = 'ON' if any(start <= observed < end for start, end in windows) else 'OFF'
            drops[phase] += 1
            continue
        latency, busy = row['observation_to_action_latency_ms'], row['worker_service_time_ms']
        end = observed + latency
        start = end - busy
        overlap = overlap_ms(start, end, windows)
        phase = 'OFF' if overlap == 0 else ('ON' if abs(overlap - busy) < 1e-5 else 'MIXED')
        groups[phase]['action'].append(latency)
        groups[phase]['worker'].append(busy)
        last_ms = max(last_ms, end)
        series.append({'obs_id': row['obs_id'], 'episode_id': row['episode_id'],
                       'observation_time_ms': observed, 'worker_start_time_ms': start,
                       'completion_time_ms': end, 'load_phase': phase,
                       'on_overlap_ms': overlap, 'action_latency_ms': latency,
                       'worker_service_ms': busy,
                       'action_raw_frames': ms_to_raw_frames(latency, protocol['env_fps'])})
    if not series:
        raise ValueError('No completed inference requests in the measurement')
    with (output / 'timing_by_load.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    warning = []
    if not groups['ON']['worker']:
        warning.append('No request fully inside an ON interval; collect more events or longer ON intervals.')
    if not groups['OFF']['worker']:
        warning.append('No request fully inside an OFF interval; cannot estimate a quiet baseline.')
    summary = {
        'condition': protocol['condition'], 'measured_requests': len(series),
        'observed_span_seconds': (last_ms - first_ms) / 1000,
        'on_fraction_of_observed_span': overlap_ms(first_ms, last_ms, windows) / (last_ms - first_ms),
        'on_intervals_intersecting_observed_span': sum(end > first_ms and start < last_ms for start, end in windows),
        'copy_boundary_overruns': sum(row['event'] == 'copy_overrun' for row in events),
        'skipped_phases': sum(row['event'] == 'skipped_phase' for row in events),
        'dropped_observations_by_load_at_capture': drops,
        'groups': {phase: {name: stats(values) for name, values in fields.items()}
                   for phase, fields in groups.items()},
        'action_raw_frame_counts': {str(k): int(v) for k, v in zip(*np.unique(
            [row['action_raw_frames'] for row in series], return_counts=True))},
        'warnings': warning,
        'interpretation': 'ON/OFF classification uses overlap with the worker-service interval; boundary-crossing requests are MIXED. ON fraction includes inter-episode gaps in the observed span. Workload ON intervals are not latency-burst labels. These are descriptive measurements, not causal estimates or profile-comparison results.',
    }
    if groups['OFF']['worker']:
        summary['worker_slow_runs'] = {f'off_p{percentile}': burst_stats(
            [row['worker_service_ms'] for row in series], float(np.percentile(groups['OFF']['worker'], percentile)))
            for percentile in (95, 99)}
        summary['burst_definition'] = 'Consecutive admitted requests in this one measurement session, including ordinary episode boundaries. Thresholds use this session\'s OFF requests and are exploratory; they are not independent held-out tests.'
    (output / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'groups': summary['groups'], 'warnings': warning}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    report(parser.parse_args().output)
