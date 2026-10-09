from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

from latency_bench.core.clock import EnvClock, ms_to_raw_frames
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.latency.config import (
    SEEDED_STOCHASTIC_METHODS,
    latency_type_from_config,
    resolve_latency_method,
)
from latency_bench.latency.inverse_cdf import sample_inverse_cdf
from latency_bench.latency.profile import load_profile_manifest
from latency_bench.latency.sample import LatencySample
from latency_bench.utils.io import read_json


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class LatencySampler:
    """A sampling method drawing one :class:`LatencySample` per decision.

    ``latency_type`` is the stable string logged into metrics; ``profile_metadata``
    is populated for profile-backed methods and ``None`` otherwise.
    """
    latency_type: str = "zero"
    profile_metadata: dict[str, Any] | None = None

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Synthetic methods (explicit params, optionally derived from a profile summary)
# ---------------------------------------------------------------------------


class ZeroSampler(LatencySampler):
    latency_type = "zero"

    def __init__(self, *, clock: EnvClock):
        self.clock = clock

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        return LatencySample.from_latency_ms(0.0, clock=self.clock)


class FixedSampler(LatencySampler):
    latency_type = "fixed"

    def __init__(self, fixed_latency_ms: float, *, clock: EnvClock):
        self.fixed_latency_ms = float(fixed_latency_ms)
        self.clock = clock

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        return LatencySample.from_latency_ms(self.fixed_latency_ms, clock=self.clock)


class NormalSampler(LatencySampler):
    latency_type = "normal_distribution"

    def __init__(
        self,
        mean_latency_ms: float,
        std_latency_ms: float,
        *,
        clock: EnvClock,
        seed: int | None = None,
        min_latency_ms: float = 0.0,
        max_latency_ms: float | None = None,
    ):
        self.mean_latency_ms = float(mean_latency_ms)
        self.std_latency_ms = float(std_latency_ms)
        self.min_latency_ms = float(min_latency_ms)
        self.max_latency_ms = None if max_latency_ms is None else float(max_latency_ms)
        self.clock = clock
        self.rng = random.Random(seed)

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        if self.std_latency_ms == 0:
            value = self.mean_latency_ms
        else:
            value = self.rng.normalvariate(self.mean_latency_ms, self.std_latency_ms)
        value = max(value, self.min_latency_ms)
        if self.max_latency_ms is not None:
            value = min(value, self.max_latency_ms)
        return LatencySample.from_latency_ms(value, clock=self.clock)


class UniformSampler(LatencySampler):
    latency_type = "uniform_distribution"

    def __init__(
        self,
        min_latency_ms: float,
        max_latency_ms: float,
        *,
        clock: EnvClock,
        seed: int | None = None,
    ):
        self.min_latency_ms = float(min_latency_ms)
        self.max_latency_ms = float(max_latency_ms)
        self.clock = clock
        self.rng = random.Random(seed)

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        if self.min_latency_ms == self.max_latency_ms:
            value = self.min_latency_ms
        else:
            value = self.rng.uniform(self.min_latency_ms, self.max_latency_ms)
        return LatencySample.from_latency_ms(value, clock=self.clock)


# ---------------------------------------------------------------------------
# Profile methods: iid (inverse-CDF) + temporal (hidden-regime burst chain)
# ---------------------------------------------------------------------------


class IidInverseCdfSampler(LatencySampler):
    latency_type = "profile_sample"

    def __init__(self, distributions: dict[str, Any], *, clock: EnvClock, seed: int | None = None):
        self.distributions = distributions["worker_slots"]
        self.clock = clock
        self.rng = random.Random(seed)

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.rng = random.Random()
        clone.rng.setstate(self.rng.getstate())
        return clone

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        worker_slot = context["worker_slot"]
        distribution = self.distributions[str(worker_slot)]["all"]
        return _sample_joint_emission(distribution, rng=self.rng, clock=self.clock)


_STANDARD_NORMAL = NormalDist()


class HiddenRegimeSampler(LatencySampler):
    """Legacy v12 sampler for archived profiles and paired method-version controls.

    Real latency spikes arrive in clusters with normal-valued samples in
    between, so a chain keyed on the latency value itself forgets it is inside
    a burst after one normal draw; the hidden regime carries that memory. The
    chain advances once per submitted observation. Per episode, the burst
    entry probability is drawn from its Jeffreys Beta posterior over recorded
    counts; on entry the burst's dwell (in draws) is sampled fresh from the
    fitted dwell distribution — real dwells mix draw-scale blips with
    minutes-long degradations, which a geometric per-draw exit cannot produce.

    The regime is machine state, not episode state: a degraded machine stays
    degraded through env resets (real traces show one slow condition spanning
    tens of consecutive episodes). An episode boundary therefore only
    refreshes the posterior parameter draws — the chain state persists.

    Within a dwell, emission is per worker slot. Real degradations are worker
    states — one slow GPU serving every other request makes the merged stream
    alternate spike/calm — so replaying the merged pattern iid over slots
    spreads the slowness across workers and no single worker saturates; the
    real failure mode (a slow worker at utilization > 1 shedding half the
    observations) then never reproduces. At entry an empirical slot-to-rank
    template sampled from a fitted dwell fixes the rank assignment for that
    dwell. Each slot then
    draws its rank's two per-dwell latents once — severity (its fraction of
    above-threshold serves for this whole dwell) and level (this dwell's mean
    above-threshold service) — and every serve inside the dwell is above
    threshold with probability severity, emitting level plus a fitted
    asymmetric residual; severity is a dwell property, so it holds until the
    dwell ends and dies with it. Severity is drawn jointly with the dwell's
    length through the fitted Gaussian-copula rho: on every fitted device the
    host-wide events are short blips while long degradations belong to one
    worker, so an independent draw would hand a long degradation's healthy
    worker a blip's severity. Run-length closure on the fitted devices
    shows no within-dwell autocorrelation beyond severity, so the serve order
    is memoryless by evidence; constant-slow degradations and isolated blips
    are just severities near one on long and short dwells, not model classes.
    Drops stay emergent in the simulator through worker-service occupancy.
    """
    latency_type = "profile_sample"

    def __init__(
        self,
        model: dict[str, Any],
        distributions: dict[str, Any],
        *,
        clock: EnvClock,
        seed: int | None = None,
    ):
        self.model = model
        self.clock = clock
        self.rng = random.Random(seed)
        self.distributions = distributions["worker_slots"]
        self.rank_processes = model["burst_rank_processes"]
        self.pre_worker_time = model["pre_worker_time_ms"]
        self.current_episode_id: int | None = None
        self.regime = "calm"
        self._enter_burst_prob = 0.0
        self._burst_remaining = 0
        self._dwell_z = 0.0
        self._slot_to_rank: dict[int, int] = {}
        self._unassigned_ranks: list[int] = []
        self._slot_states: dict[int, dict[str, Any]] = {}

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.rng = random.Random()
        clone.rng.setstate(self.rng.getstate())
        clone._slot_to_rank = self._slot_to_rank.copy()
        clone._unassigned_ranks = self._unassigned_ranks.copy()
        clone._slot_states = {
            slot: state.copy() for slot, state in self._slot_states.items()
        }
        return clone

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        episode_id = context["episode_id"]
        worker_slot = context["worker_slot"]
        first_draw = self.current_episode_id is None
        new_episode = episode_id != self.current_episode_id
        if new_episode:
            self.current_episode_id = episode_id
            self._enter_burst_prob = self._episode_transition_prob("calm", "burst")
        if first_draw:
            pass
        elif self.regime == "calm":
            if self.rng.random() < self._enter_burst_prob:
                self.regime = "burst"
                self._burst_remaining = self._sample_burst_dwell() - 1
                self._begin_dwell(entry_slot=worker_slot)
        elif self._burst_remaining > 0:
            self._burst_remaining -= 1
        else:
            self.regime = "calm"
            self._slot_states.clear()
            self._slot_to_rank = {}
            self._unassigned_ranks = []

        return self._next_emission(worker_slot)

    def _episode_transition_prob(self, source: str, target: str) -> float:
        if not self.rank_processes:
            # No burst was ever observed: stay calm instead of hallucinating
            # spikes we have no emission distribution for.
            return 0.0 if target == "burst" else 1.0
        counts = self.model["regime_transition_counts"][source]
        switch = self.rng.gammavariate(counts[target] + 0.5, 1.0)
        stay = self.rng.gammavariate(counts[source] + 0.5, 1.0)
        return switch / (switch + stay)

    def _sample_burst_dwell(self) -> int:
        return self._sample_dwell(self.model["burst_dwell_distribution"])

    def _sample_dwell(self, dwell: dict[str, Any]) -> int:
        quantile = self.rng.random()
        # The dwell's length quantile seeds the severity copula draws of this
        # dwell's slot states.
        self._dwell_z = _STANDARD_NORMAL.inv_cdf(min(max(quantile, 1e-12), 1.0 - 1e-12))
        return max(1, round(sample_inverse_cdf(
            self.rng, dwell["quantile_levels"], dwell["latency_ms"], quantile=quantile
        )))

    def _begin_dwell(self, *, entry_slot: int) -> None:
        template = self.rng.choice(self.model["burst_slot_rank_templates"])
        self._slot_to_rank = {
            int(slot): int(rank)
            for slot, rank in template["slot_to_rank"].items()
        }
        assigned_ranks = set(self._slot_to_rank.values())
        self._unassigned_ranks = [
            rank
            for rank in range(len(self.rank_processes))
            if rank not in assigned_ranks
        ]
        self._slot_states = {}
        self._slot_state(entry_slot)

    def _new_slot_state(self, rank: int | None) -> dict[str, Any]:
        if rank is None:
            # More serving slots than fitted workers: nothing was ever
            # observed for such a slot, so it stays calm through the dwell.
            return {"rank": None, "severity": 0.0, "level": 0.0}
        process = self.rank_processes[rank]
        rho = max(-0.999999, min(0.999999, process["dwell_length_spearman_rho"]))
        z_severity = rho * self._dwell_z + math.sqrt(1.0 - rho * rho) * self.rng.normalvariate(0.0, 1.0)
        severity = sample_inverse_cdf(
            self.rng,
            process["severity"]["quantile_levels"],
            process["severity"]["latency_ms"],
            quantile=_STANDARD_NORMAL.cdf(z_severity),
        )
        level = process["level_ms"]
        return {
            "rank": rank,
            "severity": severity,
            "level": (
                sample_inverse_cdf(
                    self.rng, level["quantile_levels"], level["latency_ms"]
                )
                if level is not None
                else 0.0
            ),
        }

    def _slot_state(self, worker_slot: int) -> dict[str, Any]:
        state = self._slot_states.get(worker_slot)
        if state is None:
            if worker_slot in self._slot_to_rank:
                rank = self._slot_to_rank[worker_slot]
            elif self._unassigned_ranks:
                index = self.rng.randrange(len(self._unassigned_ranks))
                rank = self._unassigned_ranks.pop(index)
            else:
                rank = None
            state = self._new_slot_state(rank)
            self._slot_states[worker_slot] = state
        return state

    def _next_emission(self, worker_slot: int) -> LatencySample:
        steady = self.distributions[str(worker_slot)]["steady"]
        if self.regime != "burst" or not self.rank_processes:
            return _sample_joint_emission(steady, rng=self.rng, clock=self.clock)
        state = self._slot_state(worker_slot)
        if state["rank"] is not None and self.rng.random() < state["severity"]:
            return self._emit_spike(state, worker_slot)
        return _sample_joint_emission(steady, rng=self.rng, clock=self.clock)

    def _emit_spike(self, state: dict[str, Any], worker_slot: int) -> LatencySample:
        residual = self.rank_processes[state["rank"]]["level_residual_ms"]
        worker_service_ms = state["level"] + (
            sample_inverse_cdf(
                self.rng, residual["quantile_levels"], residual["latency_ms"]
            )
            if residual is not None
            else 0.0
        )
        # A spike is above-threshold by construction; keep a residual drawn
        # from a hotter dwell's pool from dipping a mild level below it.
        worker_service_ms = max(
            worker_service_ms,
            math.nextafter(
                self.model["spike_threshold_ms_by_worker_slot"][str(worker_slot)],
                math.inf,
            ),
        )
        pre_worker_time_ms = sample_inverse_cdf(
            self.rng,
            self.pre_worker_time["quantile_levels"],
            self.pre_worker_time["latency_ms"],
        )
        latency_ms = worker_service_ms + pre_worker_time_ms
        return LatencySample.from_action_ready(
            action_ready_raw_frames=self.clock.latency_ms_to_raw_frames(latency_ms),
            worker_service_raw_frames=self.clock.latency_ms_to_raw_frames(worker_service_ms),
            worker_service_ms=worker_service_ms,
            latency_ms=latency_ms,
            clock=self.clock,
        )


class JointBurstSampler(LatencySampler):
    """Resample complete burst events, preserving duration and per-slot emissions."""
    latency_type = "profile_sample"

    def __init__(self, model, distributions, *, clock, seed=None):
        self.model = model
        self.distributions = distributions["worker_slots"]
        self.clock = clock
        self.rng = random.Random(seed)
        self.current_episode_id = None
        self.regime = "calm"
        self._event = None
        self._remaining = 0
        self._enter_burst_prob = 0.0

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.rng = random.Random()
        clone.rng.setstate(self.rng.getstate())
        return clone

    def _start_burst(self, event):
        self._event = event
        self._remaining = event["length"] - 1
        self.regime = "burst"

    def sample(self, context=None):
        episode_id = context["episode_id"]
        slot = str(context["worker_slot"])
        first = self.current_episode_id is None
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            if self.model["burst_events"]:
                counts = self.model["regime_transition_counts"]["calm"]
                switch = self.rng.gammavariate(counts["burst"] + 0.5, 1.0)
                stay = self.rng.gammavariate(counts["calm"] + 0.5, 1.0)
                self._enter_burst_prob = switch / (switch + stay)
        if first:
            event = self.rng.choice(self.model["initial_events"])
            if event is not None:
                self._start_burst(event)
        elif self.regime == "calm":
            if self.rng.random() < self._enter_burst_prob:
                self._start_burst(self.rng.choice(self.model["burst_events"]))
        elif self._remaining:
            self._remaining -= 1
        else:
            self.regime = "calm"
            self._event = None

        distribution = self.distributions[slot]["steady"]
        if self._event is not None and slot in self._event["worker_slots"]:
            worker = self._event["worker_slots"][slot]
            distribution = worker["spike" if self.rng.random() < worker["severity"] else "steady"]
        return _sample_joint_emission(distribution, rng=self.rng, clock=self.clock)


def _sample_joint_emission(
    distribution: dict[str, Any],
    *,
    rng: random.Random,
    clock: EnvClock,
) -> LatencySample:
    rho = max(-0.999999, min(0.999999, distribution["spearman_rho"]))
    z_observation_to_action = rng.normalvariate(0.0, 1.0)
    z_worker = rho * z_observation_to_action + math.sqrt(
        1.0 - rho * rho
    ) * rng.normalvariate(0.0, 1.0)
    observation_to_action_latency_ms = sample_inverse_cdf(
        rng,
        distribution["observation_to_action_latency_ms"]["quantile_levels"],
        distribution["observation_to_action_latency_ms"]["latency_ms"],
        quantile=_STANDARD_NORMAL.cdf(z_observation_to_action),
    )
    worker_service_time_ms = sample_inverse_cdf(
        rng,
        distribution["worker_service_time_ms"]["quantile_levels"],
        distribution["worker_service_time_ms"]["latency_ms"],
        quantile=_STANDARD_NORMAL.cdf(z_worker),
    )
    return LatencySample.from_action_ready(
        action_ready_raw_frames=clock.latency_ms_to_raw_frames(observation_to_action_latency_ms),
        worker_service_raw_frames=clock.latency_ms_to_raw_frames(worker_service_time_ms),
        worker_service_ms=worker_service_time_ms,
        latency_ms=observation_to_action_latency_ms,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Per-episode modifier (replaces the old Mixed* sources)
# ---------------------------------------------------------------------------


class PerEpisodeSampler(LatencySampler):
    """Freeze a single draw from ``inner`` for the duration of each episode."""

    def __init__(self, inner: LatencySampler):
        self.inner = inner
        self.latency_type = inner.latency_type
        self.profile_metadata = inner.profile_metadata
        self._episode_id: int | None = None
        self._sample: LatencySample | None = None

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        episode_id = int(context["episode_id"])
        if self._episode_id != episode_id:
            self._episode_id = episode_id
            self._sample = self.inner.sample(context)
        return self._sample


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_sampler(latency_cfg: dict[str, Any], *, clock: EnvClock) -> LatencySampler:
    method = latency_cfg["method"]
    seed = latency_cfg.get("seed")
    profile_metadata: dict[str, Any] | None = None

    if method == "zero":
        sampler: LatencySampler = ZeroSampler(clock=clock)
    elif method == "fixed":
        sampler = FixedSampler(latency_cfg["fixed_latency_ms"], clock=clock)
    elif method == "normal":
        sampler = NormalSampler(
            mean_latency_ms=latency_cfg["mean_latency_ms"],
            std_latency_ms=latency_cfg["std_latency_ms"],
            clock=clock,
            seed=seed,
            min_latency_ms=latency_cfg.get("min_latency_ms", 0.0),
            max_latency_ms=latency_cfg.get("max_latency_ms"),
        )
    elif method == "uniform":
        sampler = UniformSampler(
            min_latency_ms=latency_cfg["min_latency_ms"],
            max_latency_ms=latency_cfg["max_latency_ms"],
            clock=clock,
            seed=seed,
        )
    elif method == "iid":
        distribution, profile_metadata = _load_iid_profile(latency_cfg["profile_path"])
        sampler = IidInverseCdfSampler(
            distribution,
            clock=clock,
            seed=seed,
        )
    elif method == "temporal":
        profile_path = _resolve_profile_manifest_path(latency_cfg["profile_path"])
        profile = load_profile_manifest(profile_path)
        sampler = _build_temporal(profile, profile_path=profile_path, clock=clock, seed=seed)
        profile_metadata = _profile_metadata(profile)
    else:
        raise ValueError(f"Unsupported latency method: {method}")

    return _finalize_sampler(
        sampler,
        latency_cfg=latency_cfg,
        profile_metadata=profile_metadata,
    )


def build_episode_sampler_factory(
    latency_cfg: dict[str, Any],
    clock: EnvClock,
) -> Callable[[int], LatencySampler] | None:
    """Return a per-episode sampler builder deriving a stream seed per episode.

    Seeded IID methods get independent episode streams. Temporal models retain
    machine state across episodes: one sampler per replica, reused across that
    replica's episodes.
    """
    method = resolve_latency_method(latency_cfg)
    if method == "iid":
        if "seed" not in latency_cfg:
            return None
        distribution, profile_metadata = _load_iid_profile(latency_cfg["profile_path"])
        seed = latency_cfg["seed"]

        def build_iid_for_episode(episode_id: int) -> LatencySampler:
            sampler = IidInverseCdfSampler(
                distribution,
                clock=clock,
                seed=derive_seed(seed, episode_idx=episode_id),
            )
            return _finalize_sampler(
                sampler,
                latency_cfg=latency_cfg,
                profile_metadata=profile_metadata,
            )

        return build_iid_for_episode

    if method == "temporal":
        profile_path = _resolve_profile_manifest_path(latency_cfg["profile_path"])
        profile = load_profile_manifest(profile_path)
        _load_temporal_model(profile, profile_path=profile_path)
        return None
    elif method not in SEEDED_STOCHASTIC_METHODS or "seed" not in latency_cfg:
        return None
    else:
        seed = latency_cfg["seed"]

    def build_for_episode(episode_id: int) -> LatencySampler:
        episode_latency_cfg = dict(latency_cfg)
        episode_latency_cfg["seed"] = derive_seed(seed, episode_idx=episode_id)
        return build_sampler(episode_latency_cfg, clock=clock)

    return build_for_episode


def _finalize_sampler(
    sampler: LatencySampler,
    *,
    latency_cfg: dict[str, Any],
    profile_metadata: dict[str, Any] | None,
) -> LatencySampler:
    sampler.latency_type = latency_type_from_config(latency_cfg)
    sampler.profile_metadata = profile_metadata
    if latency_cfg.get("per_episode"):
        return PerEpisodeSampler(sampler)
    return sampler


def _load_iid_profile(
    profile_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_profile_path = _resolve_profile_manifest_path(profile_path)
    profile = load_profile_manifest(resolved_profile_path)
    distribution_path = _profile_relative_path(
        resolved_profile_path, profile["distribution_path"]
    )
    distribution = read_json(distribution_path)
    if distribution["schema_version"] != 3:
        raise ValueError(
            f"Unsupported IID latency distribution schema_version={distribution['schema_version']!r} "
            f"at {distribution_path}; rebuild the profile with latency_bench.latency.profile_builder."
        )
    return distribution, _profile_metadata(profile)


def _profile_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": profile["model_id"],
        "gpu_class": profile["gpu_class"],
        "workload_id": profile["workload_id"],
        "instance_id": profile["instance_id"],
        "source_run_id": profile["source_run_id"],
    }


def _build_temporal(
    profile: dict[str, Any],
    *,
    profile_path: Path,
    clock: EnvClock,
    seed: int | None,
) -> LatencySampler:
    model = _load_temporal_model(profile, profile_path=profile_path)
    distribution_path = _profile_relative_path(
        profile_path, profile["distribution_path"]
    )
    sampler_class = {12: HiddenRegimeSampler, 13: JointBurstSampler}[model["schema_version"]]
    return sampler_class(
        model,
        read_json(distribution_path),
        clock=clock,
        seed=seed,
    )


def _load_temporal_model(
    profile: dict[str, Any], *, profile_path: Path
) -> dict[str, Any]:
    model_path = _profile_relative_path(profile_path, profile["burst_model_path"])
    model = read_json(model_path)
    contract = (model["schema_version"], model["reset_scope"])
    if model["model_type"] != "hidden_regime" or contract not in {(12, "session"), (13, "session")}:
        raise ValueError(
            f"Unsupported temporal burst model (model_type={model['model_type']!r}, "
            f"reset_scope={model['reset_scope']!r}, "
            f"schema_version={model['schema_version']!r}) at {model_path}; "
            "rebuild the profile from its measure artifacts with "
            "latency_bench.latency.profile_builder."
        )
    return model


# ---------------------------------------------------------------------------
# Profile paths
# ---------------------------------------------------------------------------


def _resolve_profile_manifest_path(profile_path: str | Path) -> Path:
    resolved = Path(profile_path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "profile.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Profile manifest not found: {resolved}")
    return resolved


def _profile_relative_path(profile_path: Path, artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        raise ValueError(f"Profile companion path must be relative: {artifact_path}")
    profile_root = profile_path.parent.resolve()
    resolved = (profile_root / path).resolve()
    if not resolved.is_relative_to(profile_root):
        raise ValueError(f"Profile companion path escapes profile directory: {artifact_path}")
    return resolved


def build_profile_sampler(
    profile_path: str,
    *,
    method: str,
    seed: int,
    env_fps: float,
    obs_fps: float,
):
    return build_sampler(
        {"method": method, "profile_path": profile_path, "seed": int(seed)},
        clock=EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps)),
    )


def temporal_profile_sampler_for_reset(
    sampler: HiddenRegimeSampler | JointBurstSampler, seed: int,
) -> HiddenRegimeSampler | JointBurstSampler:
    """Start an independent temporal stream for one explicitly seeded rollout."""
    return type(sampler)(
        sampler.model,
        {"worker_slots": sampler.distributions},
        clock=sampler.clock,
        seed=seed,
    )


def profile_sampler_max_latency_raw_frames(sampler: Any, *, worker_slot: int, env_fps: float) -> int:
    """Maximum ready delay over the sampler's generated support."""
    slot = str(worker_slot)
    distribution = sampler.distributions[slot]["steady" if isinstance(sampler, (HiddenRegimeSampler, JointBurstSampler)) else "all"]
    max_latency_ms = max(distribution["observation_to_action_latency_ms"]["latency_ms"])
    if isinstance(sampler, HiddenRegimeSampler):
        for rank in sampler.rank_processes:
            residual = rank["level_residual_ms"]
            spike_ms = max(
                max(rank["level_ms"]["latency_ms"]) + (max(residual["latency_ms"]) if residual is not None else 0.0),
                math.nextafter(sampler.model["spike_threshold_ms_by_worker_slot"][slot], math.inf),
            )
            max_latency_ms = max(max_latency_ms, spike_ms + max(sampler.pre_worker_time["latency_ms"]))
    elif isinstance(sampler, JointBurstSampler):
        for event in [*sampler.model["initial_events"], *sampler.model["burst_events"]]:
            if event is not None and slot in event["worker_slots"]:
                worker = event["worker_slots"][slot]
                for name in ("spike", "steady"):
                    if worker[name] is not None:
                        max_latency_ms = max(max_latency_ms, max(worker[name]["observation_to_action_latency_ms"]["latency_ms"]))
    return ms_to_raw_frames(float(max_latency_ms), float(env_fps))


def profile_sampler_max_latency_raw_frames_for_workers(
    sampler: Any,
    *,
    profile_worker_slot: int,
    simulated_worker_capacity: int | None,
    env_fps: float,
) -> int:
    worker_slots = (
        [profile_worker_slot]
        if simulated_worker_capacity is None
        else range(simulated_worker_capacity)
    )
    return max(
        profile_sampler_max_latency_raw_frames(
            sampler,
            worker_slot=worker_slot,
            env_fps=env_fps,
        )
        for worker_slot in worker_slots
    )
