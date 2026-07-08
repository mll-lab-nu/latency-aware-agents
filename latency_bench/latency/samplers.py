from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_latency_seed
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
    chain advances once per submitted observation. Per episode, the two
    transition probabilities are drawn from their Jeffreys Beta posterior over
    recorded counts, and each regime's fitted joint emission distribution
    produces a fresh ``(action_ready_ms, worker_service_ms)``.

    Above-threshold latency is emitted through excursions: on a burst draw the
    sampler starts one with the fitted per-episode start probability, samples a
    fresh (duration, level) pair from the fitted joint distribution, then emits
    that many consecutive draws around the level. An in-flight excursion runs
    to completion even if the outer chain leaves burst mid-way, preserving the
    duration semantics; sustained plateaus and isolated spikes are just
    different regions of the (duration, level) density, not model classes.
    Drops stay emergent in the simulator through worker-service occupancy.
    """
    latency_type = "profile_sample"

    def __init__(self, model: dict[str, Any], *, clock: EnvClock, seed: int | None = None):
        self.model = model
        self.clock = clock
        self.rng = random.Random(seed)
        self.emissions = model["emission_distributions_by_regime"]
        self.excursions = model.get("excursions")
        self.current_episode_id: int | None = None
        self.regime = "calm"
        self._enter_burst_prob = 0.0
        self._exit_burst_prob = 1.0
        self._excursion_start_prob = 0.0
        self._excursion_remaining = 0
        self._excursion_level_ms = 0.0
        self._excursion_separator_pending = False

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        episode_id = int(context["episode_id"])
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            self.regime = "calm"
            self._excursion_remaining = 0
            self._excursion_separator_pending = False
            self._enter_burst_prob = self._episode_transition_prob("calm", "burst")
            self._exit_burst_prob = self._episode_transition_prob("burst", "calm")
            self._excursion_start_prob = self._episode_excursion_start_prob()
        elif self.regime == "calm":
            if self.rng.random() < self._enter_burst_prob:
                self.regime = "burst"
        elif self.rng.random() < self._exit_burst_prob:
            self.regime = "calm"

        record = self._next_emission()
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
        if self.emissions.get("burst") is None and self.excursions is None:
            # No burst was ever observed: stay calm instead of hallucinating
            # spikes we have no emission distribution for.
            return 0.0 if target == "burst" else 1.0
        counts = self.model["regime_transition_counts"][source]
        switch = self.rng.gammavariate(float(counts[target]) + 0.5, 1.0)
        stay = self.rng.gammavariate(float(counts[source]) + 0.5, 1.0)
        return switch / (switch + stay)

    def _episode_excursion_start_prob(self) -> float:
        if self.excursions is None:
            return 0.0
        counts = self.excursions["start_counts"]
        start = self.rng.gammavariate(float(counts["start"]) + 0.5, 1.0)
        hold = self.rng.gammavariate(float(counts["hold"]) + 0.5, 1.0)
        return start / (start + hold)

    def _next_emission(self) -> dict[str, Any]:
        if self._excursion_remaining > 0:
            return self._emit_excursion_draw()
        if self._excursion_separator_pending:
            # Fitted durations are maximal above-threshold runs, so two
            # excursions never touch: the draw right after one is always
            # sub-threshold.
            self._excursion_separator_pending = False
            if self.regime == "burst" and self.emissions.get("burst") is not None:
                return self._sample_emission("burst")
            return self._sample_emission("calm")
        if self.regime == "burst" and self.excursions is not None and (
            self.emissions.get("burst") is None
            or self.rng.random() < self._excursion_start_prob
        ):
            self._start_excursion()
            return self._emit_excursion_draw()
        return self._sample_emission(self.regime)

    def _start_excursion(self) -> None:
        rho = max(-0.999999, min(0.999999, float(self.excursions["duration_level_spearman_rho"])))
        z_duration = self.rng.normalvariate(0.0, 1.0)
        z_level = rho * z_duration + math.sqrt(1.0 - rho * rho) * self.rng.normalvariate(0.0, 1.0)
        duration = sample_inverse_cdf(
            self.rng,
            self.excursions["duration_draws"]["quantile_levels"],
            self.excursions["duration_draws"]["latency_ms"],
            quantile=_STANDARD_NORMAL.cdf(z_duration),
        )
        level_ms = sample_inverse_cdf(
            self.rng,
            self.excursions["level_ms"]["quantile_levels"],
            self.excursions["level_ms"]["latency_ms"],
            quantile=_STANDARD_NORMAL.cdf(z_level),
        )
        self._excursion_remaining = max(1, round(duration))
        self._excursion_level_ms = float(level_ms)

    def _emit_excursion_draw(self) -> dict[str, Any]:
        self._excursion_remaining -= 1
        if self._excursion_remaining == 0:
            self._excursion_separator_pending = True
        sigma_ms = float(self.excursions["within_excursion_sigma_ms"])
        worker_service_ms = self._excursion_level_ms + (
            self.rng.normalvariate(0.0, sigma_ms) if sigma_ms > 0.0 else 0.0
        )
        # An excursion is above-threshold by construction; keep the rare noise
        # draw below it from breaking the run.
        worker_service_ms = max(
            worker_service_ms,
            math.nextafter(float(self.model["spike_threshold_ms"]), math.inf),
        )
        ready_delta_ms = sample_inverse_cdf(
            self.rng,
            self.excursions["ready_delta_ms"]["quantile_levels"],
            self.excursions["ready_delta_ms"]["latency_ms"],
        )
        return {
            "action_ready_ms": worker_service_ms + ready_delta_ms,
            "worker_service_ms": worker_service_ms,
        }

    def _sample_emission(self, regime: str) -> dict[str, Any]:
        distribution = self.emissions[regime]
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
# Registry + factory
# ---------------------------------------------------------------------------


SamplerBuilder = Callable[..., LatencySampler]
SAMPLER_BUILDERS: dict[str, SamplerBuilder] = {}


def register_sampler(method: str) -> Callable[[SamplerBuilder], SamplerBuilder]:
    def decorator(builder: SamplerBuilder) -> SamplerBuilder:
        SAMPLER_BUILDERS[method] = builder
        return builder

    return decorator


def resolve_latency_method(latency_cfg: dict[str, Any]) -> str:
    """Return the normalized method name for a (possibly legacy) latency config.

    Profile-default configs return ``"profile_default"`` since the concrete method
    can only be resolved once the profile manifest is loaded.
    """
    return _normalize_latency_cfg(latency_cfg)["method"]


def build_sampler(latency_cfg: dict[str, Any], *, clock: EnvClock, seed: int | None) -> LatencySampler:
    cfg = _normalize_latency_cfg(latency_cfg)
    profile, profile_path = _maybe_load_profile(cfg)
    method = _resolve_method(cfg, profile)
    builder = SAMPLER_BUILDERS.get(method)
    if builder is None:
        raise ValueError(f"Unsupported latency method: {method}")
    sampler = builder(cfg, profile=profile, profile_path=profile_path, clock=clock, seed=seed)
    if profile is not None:
        sampler.profile_metadata = {
            "profile_id": profile.get("profile_id"),
            "target_device": profile.get("target_device"),
            "profile_path": str(profile_path),
        }
    if cfg.get("per_episode"):
        sampler = PerEpisodeSampler(sampler)
        if profile is None:
            sampler.latency_type = "mixed_normal_distribution" if method == "normal" else "mixed_uniform_distribution"
    return sampler


# Capability sets over the method names above (plus the virtual
# ``profile_default``, which resolves to a concrete method only once the
# profile manifest is loaded and therefore has no registered builder).
SEEDED_STOCHASTIC_METHODS = frozenset({"normal", "uniform", "iid", "temporal", "profile_default"})
BATCHED_SIMULATED_METHODS = frozenset({"zero", "fixed"}) | SEEDED_STOCHASTIC_METHODS


def build_episode_sampler_factory(
    latency_cfg: dict[str, Any],
    clock: EnvClock,
) -> Callable[[int], LatencySampler] | None:
    """Return a per-episode sampler builder deriving a stream seed per episode.

    Only seeded stochastic methods with an explicit ``seed`` get per-episode
    streams; other configs return ``None`` and reuse the shared sampler.
    """
    if resolve_latency_method(latency_cfg) not in SEEDED_STOCHASTIC_METHODS:
        return None
    if "seed" not in latency_cfg:
        return None

    def build_for_episode(episode_id: int) -> LatencySampler:
        episode_latency_cfg = dict(latency_cfg)
        episode_latency_cfg["seed"] = derive_latency_seed(latency_cfg["seed"], episode_idx=episode_id)
        return build_sampler(episode_latency_cfg, clock=clock, seed=episode_latency_cfg.get("seed"))

    return build_for_episode


@register_sampler("zero")
def _build_zero(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    return ZeroSampler(clock=clock)


@register_sampler("fixed")
def _build_fixed(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    if "fixed_latency_ms" in cfg:
        fixed_latency_ms = float(cfg["fixed_latency_ms"])
    else:
        fixed_latency_ms = float(_profile_summary(profile)["mean_ms"])
    return FixedSampler(fixed_latency_ms, clock=clock)


@register_sampler("normal")
def _build_normal(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    if "mean_latency_ms" in cfg or "std_latency_ms" in cfg:
        return NormalSampler(
            mean_latency_ms=float(cfg["mean_latency_ms"]),
            std_latency_ms=float(cfg["std_latency_ms"]),
            clock=clock,
            seed=seed,
            min_latency_ms=float(cfg.get("min_latency_ms", 0.0)),
            max_latency_ms=cfg.get("max_latency_ms"),
        )
    summary = _profile_summary(profile)
    return NormalSampler(
        mean_latency_ms=float(summary["mean_ms"]),
        std_latency_ms=float(summary["std_ms"]),
        clock=clock,
        seed=seed,
        min_latency_ms=float(cfg.get("min_latency_ms", summary.get("min_ms", 0.0))),
        max_latency_ms=cfg.get("max_latency_ms", summary.get("max_ms")),
    )


@register_sampler("uniform")
def _build_uniform(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    return UniformSampler(
        min_latency_ms=float(cfg["min_latency_ms"]),
        max_latency_ms=float(cfg["max_latency_ms"]),
        clock=clock,
        seed=seed,
    )


@register_sampler("iid")
def _build_iid(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    distribution_path = _profile_relative_path(profile_path, profile["distribution_path"])
    return IidInverseCdfSampler(read_json(distribution_path), clock=clock, seed=seed)


@register_sampler("temporal")
def _build_temporal(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    model_path = _profile_relative_path(profile_path, profile["burst_model_path"])
    model = read_json(model_path)
    if model.get("model_type") != "hidden_regime" or "excursions" not in model:
        raise ValueError(
            f"Unsupported temporal burst model (model_type={model.get('model_type')!r}, "
            f"schema_version={model.get('schema_version')!r}) at {model_path}; "
            "rebuild the profile from its measure artifacts with "
            "latency_bench.experiments.build_latency_profile_from_trace."
        )
    return HiddenRegimeSampler(model, clock=clock, seed=seed)


# ---------------------------------------------------------------------------
# Config normalization + profile resolution
# ---------------------------------------------------------------------------


_LEGACY_TYPE_TO_METHOD = {
    "zero": ("zero", False),
    "fixed": ("fixed", False),
    "normal_distribution": ("normal", False),
    "uniform_distribution": ("uniform", False),
    "mixed_normal_distribution": ("normal", True),
    "mixed_uniform_distribution": ("uniform", True),
}

_LEGACY_PROFILE_SAMPLER_TO_METHOD = {
    "iid_inverse_cdf": "iid",
    "bayes_markov": "temporal",
}


def _normalize_latency_cfg(latency_cfg: dict[str, Any]) -> dict[str, Any]:
    if "method" in latency_cfg:
        return dict(latency_cfg)

    cfg = dict(latency_cfg)
    latency_type = cfg.get("type", "zero")
    if latency_type == "profile_sample":
        cfg["method"] = _LEGACY_PROFILE_SAMPLER_TO_METHOD.get(
            cfg.get("sampler", "profile_default"), "profile_default"
        )
        return cfg
    method, per_episode = _LEGACY_TYPE_TO_METHOD.get(latency_type, (None, False))
    if method is None:
        raise ValueError(f"Unsupported latency.type: {latency_type}")
    cfg["method"] = method
    if per_episode:
        cfg["per_episode"] = True
    return cfg


def _resolve_method(cfg: dict[str, Any], profile: dict[str, Any] | None) -> str:
    method = cfg["method"]
    if method != "profile_default":
        return method
    if profile is not None:
        latency_method = profile.get("latency_method")
        if latency_method in ("iid", "temporal"):
            return latency_method
        sample_model_type = profile.get("sample_model_type")
        if sample_model_type in _LEGACY_PROFILE_SAMPLER_TO_METHOD:
            return _LEGACY_PROFILE_SAMPLER_TO_METHOD[sample_model_type]
    info = profile or {}
    raise ValueError(
        "latency.sampler=profile_default cannot resolve a temporal method for profile "
        f"{info.get('profile_id')!r}: latency_method={info.get('latency_method')!r}, "
        f"sample_model_type={info.get('sample_model_type')!r}, "
        f"burst_model_path={info.get('burst_model_path')!r}. Rebuild this profile's "
        "temporal burst model from measure actions, or set "
        "latency.sampler=iid_inverse_cdf explicitly to use the iid distribution."
    )


def _maybe_load_profile(cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    profile_path = cfg.get("profile_path")
    if profile_path is None:
        return None, None
    resolved = _resolve_profile_manifest_path(profile_path)
    return load_profile_manifest(resolved), resolved


def _profile_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        raise ValueError("profile_path is required to derive this latency method from a profile")
    return profile["summary"]


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
