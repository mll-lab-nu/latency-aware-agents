#!/usr/bin/env python
"""Merge sharded probe states into one resume state for single-process replay.

Each sharded probe worker (``--probe-only --rollout-seed-offset ...``) writes its own
``probe_state.json`` under its output dir. This CLI combines them with
``merge_probe_states`` (deterministic shard order, dedupe by seed, contiguous episode_idx,
same split plan as the single-process probe) and writes the merged ``probe_state.json`` to
the final output dir. Running the normal export on that dir then skips probing and replays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from latency_bench.data.rollout_probe import write_probe_state, merge_probe_states


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True,
                        help="A shard output dir containing probe_state.json. Repeat in shard order.")
    parser.add_argument("--output-dir", required=True, help="Final output dir to write merged probe_state.json.")
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)

    shard_states = []
    for shard_dir in args.shard_dir:
        state_path = Path(shard_dir) / "probe_state.json"
        shard_states.append(json.loads(state_path.read_text(encoding="utf-8")))

    merged = merge_probe_states(
        shard_states,
        max_episodes=int(args.max_episodes),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_probe_state(output_dir, merged)
    print(f"[merge] {len(args.shard_dir)} shards -> {len(merged['accepted_specs'])} accepted episodes "
          f"written to {output_dir / 'probe_state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
