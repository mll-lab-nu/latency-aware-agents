"""Probe teacher episodes and persist accepted seeds for deterministic replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm
from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, accepts_episode, accepts_episode_contract
from latency_bench.data.action_layouts import ActionSpec, action_spec_for_env

if TYPE_CHECKING:
    from latency_bench.data.sf_teacher_drivers import BatchedTeacherRolloutDriver, SerialTeacherRolloutDriver

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


def probe_accepted_specs(
    runtime: Mapping[str, Any],
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
    *,
    cfg: Any,
    env_name: str,
    output_dir: Path,
    max_episodes: int,
    max_attempts: int | None,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    deterministic: bool,
    probe_state: dict[str, Any],
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    demon_episode_threshold: float | None = None,
    episode_return_gt: float | None = None,
    episode_acceptance: Mapping[str, Any] | None = None,
    rollout_seed_offset: int = 0,
    action_spec: ActionSpec | None = None,
    integration_name: str | None = None,
) -> None:
    """Probe rollout attempts without raw RGB until enough accepted specs exist.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        driver: Driver that abstracts serial vs batched env stepping.
        cfg: Sample Factory rollout cfg.
        env_name: Canonical benchmark environment name.
        output_dir: Rollout export directory.
        max_episodes: Number of accepted episodes requested.
        max_steps_per_episode: Raw environment frame cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        deterministic: Whether to use deterministic teacher actions.
        probe_state: Mutable probe state persisted for resume.
        filter_preset: Filter preset name for episode-level gating.
        demon_episode_threshold: Optional Demon raw-score threshold.
        episode_return_gt: Optional strict lower bound for accepted episode returns.
        episode_acceptance: Explicit task-specific episode acceptance contract.
        action_spec: Explicit action layout carried by a generic checkpoint.
        integration_name: Integration identity used for legacy-specific rollout behavior.
    """
    accepted_specs = probe_state["accepted_specs"]
    rnn_size = runtime["rnn_size"]
    device = runtime["device"]
    resolved_action_spec = action_spec if action_spec is not None else action_spec_for_env(env_name)
    # State-only shard merging does not need Torch, SF or simulator imports.
    import torch
    from latency_bench.data import sf_teacher_runtime as teacher_runtime
    from latency_bench.data.sf_teacher_drivers import BatchedTeacherRolloutDriver
    from latency_bench.integrations.sample_factory.actions import to_python_action

    behavior_env_name = env_name if integration_name is None else integration_name
    obs_stride_raw_frames = int(float(cfg.env_fps) / float(cfg.obs_fps))
    final_splits = {
        int(spec["episode_idx"]): str(spec["split"])
        for spec in build_episode_plan(
            max_episodes=int(max_episodes),
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            base_seed=0,
        )
    }
    active_specs: list[dict[str, Any] | None] = [None for _ in range(driver.slot_count)]
    active_obs: list[Any | None] = [None for _ in range(driver.slot_count)]
    active_episode_lengths = [0 for _ in range(driver.slot_count)]
    active_episode_returns = [0.0 for _ in range(driver.slot_count)]
    active_episode_raw_returns = [0.0 for _ in range(driver.slot_count)]
    active_decision_steps = [0 for _ in range(driver.slot_count)]
    rnn_states = torch.zeros([driver.slot_count, rnn_size], dtype=torch.float32, device=device)

    try:
        next_attempt_idx = int(probe_state["next_attempt_idx"])
        initial_reset_plan: dict[int, int] = {}
        for slot in range(driver.slot_count):
            if max_attempts is not None and next_attempt_idx >= max_attempts:
                break
            spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed) + int(rollout_seed_offset))
            next_attempt_idx += 1
            active_specs[slot] = spec
            initial_reset_plan[slot] = int(spec["seed"])
        probe_state["next_attempt_idx"] = next_attempt_idx
        if initial_reset_plan:
            if isinstance(driver, BatchedTeacherRolloutDriver):
                reset_results = driver.reset_slots(initial_reset_plan, [spec is not None for spec in active_specs])
            else:
                reset_results = driver.reset_slots(initial_reset_plan)
            for slot, (obs, _reset_info) in reset_results.items():
                active_obs[slot] = obs
                active_episode_lengths[slot] = 0
                active_episode_returns[slot] = 0.0
                active_episode_raw_returns[slot] = 0.0
                active_decision_steps[slot] = 0
                rnn_states[slot].zero_()

        with tqdm(total=int(max_episodes), initial=len(accepted_specs), unit="episode") as progress:
            while len(accepted_specs) < int(max_episodes):
                active_slots = [slot for slot, spec in enumerate(active_specs) if spec is not None]
                if not active_slots:
                    break
                _raw_actions, new_rnn_states, step_results = teacher_runtime.step_teacher_policy(
                    runtime,
                    driver,
                    active_slots=active_slots,
                    active_obs=active_obs,
                    rnn_states=rnn_states,
                    deterministic=deterministic,
                    preserve_leading_batch=behavior_env_name == "gymnasium",
                    preserve_issued_actions=env_name.startswith("g1_sonic_"),
                )
                reset_plan: dict[int, int] = {}
                target_reached = False

                # Scatter the new recurrent states for every active slot in one
                # batched assignment instead of one tiny GPU write per slot. Slots
                # that finish/reset below still get .zero_()'d afterwards, exactly
                # as when each row was written individually.
                rnn_states[active_slots] = new_rnn_states

                for slot in active_slots:
                    if len(accepted_specs) >= int(max_episodes):
                        active_specs[slot] = None
                        active_obs[slot] = None
                        active_episode_lengths[slot] = 0
                        active_episode_returns[slot] = 0.0
                        active_episode_raw_returns[slot] = 0.0
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                        continue
                    spec = active_specs[slot]
                    step_result = step_results[slot]
                    active_episode_lengths[slot] += 1
                    active_episode_returns[slot] += float(to_python_action(step_result.reward))
                    if behavior_env_name == "demon_attack":
                        active_episode_raw_returns[slot] += float(to_python_action(step_result.info["raw_reward"]))

                    next_decision_step = int(active_decision_steps[slot]) + 1
                    life_loss_boundary, episode_done = teacher_runtime.rollout_step_flags(
                        behavior_env_name,
                        step_result,
                        next_raw_frame=next_decision_step * obs_stride_raw_frames,
                        max_steps_per_episode=int(max_steps_per_episode),
                    )
                    if episode_done:
                        probe_state["attempted_episodes"] += 1
                        summary = {
                            "episode_length": int(active_episode_lengths[slot]),
                            "episode_raw_frames": int(active_episode_lengths[slot])
                            * obs_stride_raw_frames,
                            resolved_action_spec.episode_return_field: float(active_episode_returns[slot]),
                        }
                        if behavior_env_name == "demon_attack":
                            summary["episode_raw_return"] = float(active_episode_raw_returns[slot])
                        accepted = accepts_episode(
                            filter_preset,
                            behavior_env_name,
                            summary,
                            max_steps_per_episode=int(max_steps_per_episode),
                            demon_episode_threshold=demon_episode_threshold,
                        )
                        if episode_acceptance is not None:
                            summary["final_info"] = step_result.info
                            accepted = accepted and accepts_episode_contract(
                                summary,
                                episode_acceptance,
                                env_fps=cfg.env_fps,
                            )
                        accepted = accepted and _passes_episode_return_gate(
                            summary,
                            episode_return_field=resolved_action_spec.episode_return_field,
                            episode_return_gt=episode_return_gt,
                        )
                        if accepted:
                            final_episode_idx = len(accepted_specs)
                            final_split = final_splits[final_episode_idx]
                            accepted_spec = {
                                "attempt_idx": int(spec["attempt_idx"]),
                                "episode_idx": int(final_episode_idx),
                                "episode_length": int(summary["episode_length"]),
                                "episode_raw_frames": int(summary["episode_raw_frames"]),
                                resolved_action_spec.episode_return_field: float(
                                    summary[resolved_action_spec.episode_return_field]
                                ),
                                "seed": int(spec["seed"]),
                                "split": str(final_split),
                                "termination_reason": (
                                    "terminated" if step_result.terminated and not life_loss_boundary
                                    else "truncated" if step_result.truncated
                                    else "horizon"
                                ),
                            }
                            if behavior_env_name == "demon_attack":
                                accepted_spec["episode_raw_return"] = float(summary["episode_raw_return"])
                            accepted_specs.append(accepted_spec)
                            progress.update(1)
                            target_reached = len(accepted_specs) >= int(max_episodes)
                        else:
                            probe_state["rejected_episodes"] += 1
                        if (
                            len(accepted_specs) < int(max_episodes)
                            and (max_attempts is None or next_attempt_idx < max_attempts)
                        ):
                            next_spec = _attempt_episode_spec(attempt_idx=next_attempt_idx, base_seed=int(cfg.seed) + int(rollout_seed_offset))
                            next_attempt_idx += 1
                            active_specs[slot] = next_spec
                            reset_plan[slot] = int(next_spec["seed"])
                        else:
                            active_specs[slot] = None
                            active_obs[slot] = None
                            active_episode_lengths[slot] = 0
                            active_episode_returns[slot] = 0.0
                            active_episode_raw_returns[slot] = 0.0
                            active_decision_steps[slot] = 0
                            rnn_states[slot].zero_()
                        probe_state["next_attempt_idx"] = next_attempt_idx
                        if (
                            max_attempts is not None
                            or accepted
                            or int(probe_state["attempted_episodes"]) % PROBE_STATE_WRITE_EVERY == 0
                        ):
                            write_probe_state(output_dir, probe_state)
                        _update_probe_progress(
                            progress,
                            probe_state,
                            allocated_attempts=next_attempt_idx,
                        )
                    else:
                        if life_loss_boundary:
                            active_obs[slot], _reset_info = driver.continue_after_life_loss(slot)
                            rnn_states[slot].zero_()
                        else:
                            active_obs[slot] = step_result.next_obs
                        active_decision_steps[slot] = next_decision_step

                if target_reached:
                    break
                if reset_plan:
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        reset_results = driver.reset_slots(reset_plan, [spec is not None for spec in active_specs])
                    else:
                        reset_results = driver.reset_slots(reset_plan)
                    for slot, (obs, _reset_info) in reset_results.items():
                        active_obs[slot] = obs
                        active_episode_lengths[slot] = 0
                        active_episode_returns[slot] = 0.0
                        active_episode_raw_returns[slot] = 0.0
                        active_decision_steps[slot] = 0
                        rnn_states[slot].zero_()
                else:
                    if isinstance(driver, BatchedTeacherRolloutDriver):
                        driver.sync_active([spec is not None for spec in active_specs])
        # Flush the final probe state once the loop ends so a clean run persists the
        # exact same accepted_specs/counters as writing on every completion did.
        write_probe_state(output_dir, probe_state)
    finally:
        driver.close()
