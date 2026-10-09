"""Probe teacher episodes and persist accepted seeds for deterministic replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

# Accepted episodes are persisted immediately; rejected attempts are batched.
PROBE_STATE_WRITE_EVERY = 256


def build_episode_plan(
    *,
    max_episodes: int,
    val_fraction: float,
    split_seed: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Create deterministic episode split and reset seed specs.

    Args:
        max_episodes: Number of episodes to collect.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        base_seed: First environment reset seed.

    Returns:
        Episode specs with index, split, and reset seed.
    """
    val_count = round(max_episodes * val_fraction)
    val_indices = set(np.random.default_rng(split_seed).choice(max_episodes, size=val_count, replace=False).tolist())
    return [
        {
            "episode_idx": idx,
            "split": "val" if idx in val_indices else "train",
            "seed": base_seed + idx,
        }
        for idx in range(max_episodes)
    ]


def probe_state_path(output_dir: Path) -> Path:
    """Build the state path used to resume probe/replay rollout.

    Args:
        output_dir: Rollout export directory.

    Returns:
        Path to the probe state JSON file.
    """
    return output_dir / "probe_state.json"


def new_probe_state() -> dict[str, Any]:
    """Create probe state for a fresh export.

    Returns:
        Mutable state persisted after each completed rollout attempt.
    """
    return {
        "next_attempt_idx": 0,
        "attempted_episodes": 0,
        "rejected_episodes": 0,
        "accepted_specs": [],
    }


def load_probe_state(
    output_dir: Path,
    *,
    completed_episode_indices: set[int],
) -> dict[str, Any]:
    """Load or initialize probe resume state.

    Args:
        output_dir: Rollout export directory.
        completed_episode_indices: Accepted episode shards already on disk.

    Returns:
        Probe state with attempt counters and accepted specs.
    """
    state_path = probe_state_path(output_dir)
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    if completed_episode_indices:
        raise FileNotFoundError(state_path)
    return new_probe_state()


def write_probe_state(output_dir: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist probe resume state.

    Args:
        output_dir: Rollout export directory.
        state: JSON-serializable probe state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = probe_state_path(output_dir)
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def merge_probe_states(
    shard_states: Sequence[Mapping[str, Any]],
    *,
    max_episodes: int,
    val_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    """Merge sharded probe states into one resume state for single-process replay.

    Each shard explored a disjoint reset-seed range (see ``rollout_seed_offset``) and
    persisted its own accepted specs. The merged state takes the first ``max_episodes``
    accepted specs in deterministic shard order, dedupes by reset seed, and reassigns
    contiguous ``episode_idx`` plus train/val split via the same plan the single-process
    probe uses (``build_episode_plan``) — so the downstream replay path is identical to a
    normal resume. Each spec keeps its original reset seed so replay reproduces the episode.

    Args:
        shard_states: Probe states (parsed ``probe_state.json``) from each shard, in the
            shard order that defines the deterministic merge order.
        max_episodes: Number of episodes the merged dataset must contain.
        val_fraction: Fraction assigned to validation (same plan as the probe).
        split_seed: RNG seed for split assignment (same plan as the probe).

    Returns:
        A merged probe state with exactly ``max_episodes`` accepted specs, ready to write
        to the final ``output_dir`` so ``collect_teacher_dataset`` skips probing and replays.
    """
    merged_specs: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for state in shard_states:
        for spec in state["accepted_specs"]:
            seed = spec["seed"]
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            merged_specs.append(dict(spec))

    if len(merged_specs) < max_episodes:
        raise ValueError(
            f"sharded probe collected {len(merged_specs)} accepted episodes, "
            f"fewer than max_episodes={max_episodes}; increase per-shard quota."
        )

    final_splits = {
        plan_spec["episode_idx"]: plan_spec["split"]
        for plan_spec in build_episode_plan(
            max_episodes=max_episodes,
            val_fraction=val_fraction,
            split_seed=split_seed,
            base_seed=0,
        )
    }

    accepted_specs: list[dict[str, Any]] = []
    for episode_idx, spec in enumerate(merged_specs[: max_episodes]):
        finalized = dict(spec)
        finalized["episode_idx"] = episode_idx
        finalized["split"] = final_splits[episode_idx]
        accepted_specs.append(finalized)

    merged = new_probe_state()
    merged["accepted_specs"] = accepted_specs
    merged["attempted_episodes"] = sum(s["attempted_episodes"] for s in shard_states)
    merged["rejected_episodes"] = sum(s["rejected_episodes"] for s in shard_states)
    merged["next_attempt_idx"] = sum(s["next_attempt_idx"] for s in shard_states)
    for state in shard_states:
        if "runtime_metadata" in state:
            merged["runtime_metadata"] = dict(state["runtime_metadata"])
            break
    return merged


def _attempt_episode_spec(*, attempt_idx: int, base_seed: int) -> dict[str, Any]:
    """Build an online rollout attempt spec before final split assignment.

    Args:
        attempt_idx: Zero-based rollout attempt index.
        base_seed: First environment reset seed.

    Returns:
        Episode spec using an attempt-local image namespace.
    """
    return {
        "attempt_idx": attempt_idx,
        "episode_idx": attempt_idx,
        "split": "_attempts",
        "seed": base_seed + attempt_idx,
    }


def _update_probe_progress(progress: tqdm, probe_state: Mapping[str, Any], *, allocated_attempts: int) -> None:
    """Display raw probe attempt counters in the progress bar.

    Args:
        progress: Active tqdm progress bar.
        probe_state: Probe attempt counters.
        allocated_attempts: Number of raw attempt ids assigned to rollout slots.
    """
    attempted = int(probe_state["attempted_episodes"])
    accepted = len(probe_state["accepted_specs"])
    accept_rate = accepted / attempted if attempted else 0.0
    # alloc counts attempt seeds assigned to active slots; unfinished attempts are dropped after target acceptance.
    progress.set_postfix(
        alloc=int(allocated_attempts),
        done=attempted,
        rej=int(probe_state["rejected_episodes"]),
        accept_rate=f"{accept_rate:.2f}",
    )


def _passes_episode_return_gate(
    summary: Mapping[str, Any],
    *,
    episode_return_field: str,
    episode_return_gt: float | None,
) -> bool:
    if episode_return_gt is None:
        return True
    return summary[episode_return_field] > float(episode_return_gt)
