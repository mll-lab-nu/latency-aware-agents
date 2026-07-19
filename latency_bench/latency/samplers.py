from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_latency_seed
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

    def __init__(self, distribution: dict[str, Any], *, clock: EnvClock, seed: int | None = None):
        self.distribution = distribution
        self.clock = clock
        self.rng = random.Random(seed)

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        latency_ms = sample_inverse_cdf(
            self.rng,
            self.distribution["quantile_levels"],
            self.distribution["latency_ms"],
        )
        return LatencySample.from_latency_ms(latency_ms, clock=self.clock)


_STANDARD_NORMAL = NormalDist()


class HiddenRegimeSampler(LatencySampler):
    """Two-regime (calm/burst) hidden-Markov chain over worker-service latency.

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
    observations) then never reproduces. At entry the serving slot takes the
    top severity rank (a dwell starts at its first spike, which belongs to
    the most degraded worker); any other slot draws one of the remaining
    ranks uniformly the first time it serves inside the dwell. Each slot then
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

    def __init__(self, model: dict[str, Any], *, clock: EnvClock, seed: int | None = None):
        self.model = model
        self.clock = clock
        self.rng = random.Random(seed)
        self.emissions = model["emission_distributions_by_regime"]
        self.rank_processes = model.get("burst_rank_processes") or []
        self.ready_delta = model.get("ready_delta_ms")
        self.current_episode_id: int | None = None
        self.regime = "calm"
        self._enter_burst_prob = 0.0
        self._burst_remaining = 0
        self._dwell_z = 0.0
        self._unassigned_ranks: list[int] = []
        self._slot_states: dict[int, dict[str, Any]] = {}

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        episode_id = int(context["episode_id"])
        worker_slot = int(context["worker_slot"])
        first_draw = self.current_episode_id is None
        new_episode = episode_id != self.current_episode_id
        if new_episode:
            self.current_episode_id = episode_id
            self._enter_burst_prob = self._episode_transition_prob("calm", "burst")
            if self.model["reset_scope"] == "episode":
                self.regime = "calm"
                self._burst_remaining = 0
                self._slot_states.clear()
                self._unassigned_ranks = []
                initial_counts = self.model["initial_regime_counts"]
                total = initial_counts["calm"] + initial_counts["burst"]
                if self.rng.randrange(total) < initial_counts["burst"]:
                    self.regime = "burst"
                    self._burst_remaining = self._sample_dwell(
                        self.model["initial_burst_residual_dwell_distribution"]
                    ) - 1
                    self._begin_dwell(entry_slot=worker_slot)
        if first_draw or (new_episode and self.model["reset_scope"] == "episode"):
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
            self._unassigned_ranks = []

        record = self._next_emission(worker_slot)
        action_ready_ms = float(record["action_ready_ms"])
        worker_service_ms = float(record["worker_service_ms"])
        return LatencySample.from_action_ready(
            action_ready_raw_frames=self.clock.latency_ms_to_raw_frames(action_ready_ms),
            worker_service_raw_frames=self.clock.latency_ms_to_raw_frames(worker_service_ms),
            worker_service_ms=worker_service_ms,
            latency_ms=action_ready_ms,
            clock=self.clock,
        )

    def _episode_transition_prob(self, source: str, target: str) -> float:
        if not self.rank_processes:
            # No burst was ever observed: stay calm instead of hallucinating
            # spikes we have no emission distribution for.
            return 0.0 if target == "burst" else 1.0
        counts = self.model["regime_transition_counts"][source]
        if self.model["reset_scope"] == "episode" and counts[target] == 0:
            return 0.0
        switch = self.rng.gammavariate(float(counts[target]) + 0.5, 1.0)
        stay = self.rng.gammavariate(float(counts[source]) + 0.5, 1.0)
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
        self._slot_states = {entry_slot: self._new_slot_state(0)}
        self._unassigned_ranks = list(range(1, len(self.rank_processes)))

    def _new_slot_state(self, rank: int | None) -> dict[str, Any]:
        if rank is None:
            # More serving slots than fitted workers: nothing was ever
            # observed for such a slot, so it stays calm through the dwell.
            return {"rank": None, "severity": 0.0, "level": 0.0}
        process = self.rank_processes[rank]
        rho = max(-0.999999, min(0.999999, float(process["dwell_length_spearman_rho"])))
        z_severity = rho * self._dwell_z + math.sqrt(1.0 - rho * rho) * self.rng.normalvariate(0.0, 1.0)
        severity = sample_inverse_cdf(
            self.rng,
            process["severity"]["quantile_levels"],
            process["severity"]["latency_ms"],
            quantile=_STANDARD_NORMAL.cdf(z_severity),
        )
        level = process.get("level_ms")
        return {
            "rank": rank,
            "severity": float(severity),
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
            if self._unassigned_ranks:
                index = self.rng.randrange(len(self._unassigned_ranks))
                rank = self._unassigned_ranks.pop(index)
            else:
                rank = None
            state = self._new_slot_state(rank)
            self._slot_states[worker_slot] = state
        return state

    def _next_emission(self, worker_slot: int) -> dict[str, Any]:
        if self.regime != "burst" or not self.rank_processes:
            return self._sample_emission(self.emissions["calm"])
        state = self._slot_state(worker_slot)
        if state["rank"] is not None and self.rng.random() < state["severity"]:
            return self._emit_spike(state)
        rank = state["rank"]
        if rank is not None:
            emission = self.rank_processes[rank].get("emission")
            if emission is not None:
                return self._sample_emission(emission)
        return self._sample_emission(self.emissions["calm"])

    def _emit_spike(self, state: dict[str, Any]) -> dict[str, Any]:
        residual = self.rank_processes[state["rank"]].get("level_residual_ms")
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
            math.nextafter(float(self.model["spike_threshold_ms"]), math.inf),
        )
        ready_delta_ms = sample_inverse_cdf(
            self.rng,
            self.ready_delta["quantile_levels"],
            self.ready_delta["latency_ms"],
        )
        return {
            "action_ready_ms": worker_service_ms + ready_delta_ms,
            "worker_service_ms": worker_service_ms,
        }

    def _sample_emission(self, distribution: dict[str, Any]) -> dict[str, Any]:
        rho = max(-0.999999, min(0.999999, float(distribution["spearman_rho"])))
        z_action_ready = self.rng.normalvariate(0.0, 1.0)
        z_worker = rho * z_action_ready + math.sqrt(1.0 - rho * rho) * self.rng.normalvariate(0.0, 1.0)
        action_ready_ms = sample_inverse_cdf(
            self.rng,
            distribution["action_ready_ms"]["quantile_levels"],
            distribution["action_ready_ms"]["latency_ms"],
            quantile=_STANDARD_NORMAL.cdf(z_action_ready),
        )
        worker_service_ms = sample_inverse_cdf(
            self.rng,
            distribution["worker_service_ms"]["quantile_levels"],
            distribution["worker_service_ms"]["latency_ms"],
            quantile=_STANDARD_NORMAL.cdf(z_worker),
        )
        return {
            "action_ready_ms": action_ready_ms,
            "worker_service_ms": worker_service_ms,
        }


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
    profile: dict[str, Any] | None = None
    profile_path: Path | None = None

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
    elif method in {"iid", "temporal"}:
        profile_path = _resolve_profile_manifest_path(latency_cfg["profile_path"])
        profile = load_profile_manifest(profile_path)
        if method == "iid":
            distribution_path = _profile_relative_path(profile_path, profile["distribution_path"])
            sampler = IidInverseCdfSampler(read_json(distribution_path), clock=clock, seed=seed)
        else:
            sampler = _build_temporal(profile, profile_path=profile_path, clock=clock, seed=seed)
    else:
        raise ValueError(f"Unsupported latency method: {method}")

    if profile is not None:
        sampler.profile_metadata = {
            "profile_id": profile.get("profile_id"),
            "target_device": profile.get("target_device"),
            "profile_path": str(profile_path),
        }
    if latency_cfg.get("per_episode"):
        sampler = PerEpisodeSampler(sampler)
    sampler.latency_type = latency_type_from_config(latency_cfg)
    return sampler


def build_episode_sampler_factory(
    latency_cfg: dict[str, Any],
    clock: EnvClock,
) -> Callable[[int], LatencySampler] | None:
    """Return a per-episode sampler builder deriving a stream seed per episode.

    Seeded stochastic methods get independent episode streams. Temporal models
    do so only when their fitted lifecycle is episode-scoped; session-scoped
    machine state continues to use the shared sampler.
    """
    method = resolve_latency_method(latency_cfg)
    if method == "temporal":
        profile_path = _resolve_profile_manifest_path(latency_cfg["profile_path"])
        profile = load_profile_manifest(profile_path)
        model = _load_temporal_model(profile, profile_path=profile_path)
        if model["reset_scope"] == "session":
            return None
        seed = latency_cfg["seed"]
    elif method not in SEEDED_STOCHASTIC_METHODS or "seed" not in latency_cfg:
        return None
    else:
        seed = latency_cfg["seed"]

    def build_for_episode(episode_id: int) -> LatencySampler:
        episode_latency_cfg = dict(latency_cfg)
        episode_latency_cfg["seed"] = derive_latency_seed(seed, episode_idx=episode_id)
        return build_sampler(episode_latency_cfg, clock=clock)

    return build_for_episode


def _build_temporal(
    profile: dict[str, Any],
    *,
    profile_path: Path,
    clock: EnvClock,
    seed: int | None,
) -> LatencySampler:
    model = _load_temporal_model(profile, profile_path=profile_path)
    return HiddenRegimeSampler(model, clock=clock, seed=seed)


def _load_temporal_model(
    profile: dict[str, Any], *, profile_path: Path
) -> dict[str, Any]:
    model_path = _profile_relative_path(profile_path, profile["burst_model_path"])
    model = read_json(model_path)
    contract = (model.get("schema_version"), model.get("reset_scope"))
    supported_contract = contract in {(9, "session"), (10, "episode")}
    lifecycle_matches = (
        contract == (9, "session") and "measurement" not in profile
    ) or contract == {
        "consecutive": (9, "session"),
        "isolated": (10, "episode"),
    }[profile["measurement"]["episode_lifecycle"]]
    if (
        model.get("model_type") != "hidden_regime"
        or not supported_contract
        or not lifecycle_matches
    ):
        raise ValueError(
            f"Unsupported temporal burst model (model_type={model.get('model_type')!r}, "
            f"reset_scope={model.get('reset_scope')!r}, "
            f"schema_version={model.get('schema_version')!r}) at {model_path}; "
            "rebuild the profile from its measure artifacts with "
            "latency_bench.experiments.build_latency_profile_from_trace."
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
        return path
    return profile_path.parent / path
