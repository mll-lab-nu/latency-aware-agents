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
# Profile methods: iid (inverse-CDF) + temporal (Bayesian Markov)
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


class BayesMarkovSampler(LatencySampler):
    """Bayesian Markov chain over worker-service latency ms bins.

    The state is the worker-service latency bin. The chain advances once per
    submitted observation; initial/transition distributions are drawn per episode
    from the Dirichlet posterior over recorded counts, and each state's fitted
    joint emission distribution produces ``(action_ready_ms, worker_service_ms)``.
    Drops stay emergent in the simulator through worker-service occupancy.

    With ``transition_alpha``/``initial_alpha`` -> 0 and ``emission_bootstrap`` off,
    this degrades to a plain empirical Markov chain over the worker-service states.
    """
    latency_type = "profile_sample"

    def __init__(
        self,
        model: dict[str, Any],
        *,
        clock: EnvClock,
        seed: int | None = None,
        transition_alpha: float = 0.01,
        initial_alpha: float = 0.01,
        emission_bootstrap: bool = True,
        semi_markov_dwell: bool = False,
        transition_mode: str = "posterior_sample",
    ):
        self.model = model
        self.clock = clock
        self.rng = random.Random(seed)
        self.transition_alpha = float(transition_alpha)
        self.initial_alpha = float(initial_alpha)
        self.emission_bootstrap = bool(emission_bootstrap)
        self.semi_markov_dwell = bool(semi_markov_dwell)
        self.transition_mode = transition_mode
        self.states = sorted(model["state_counts"], key=lambda state: int(state))
        self.current_episode_id: int | None = None
        self.current_state: str | None = None
        self.remaining_run_length = 0
        self._episode_initial: dict[str, float] | None = None
        self._episode_transition: dict[str, dict[str, float]] | None = None

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        episode_id = int(context["episode_id"])
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            self._episode_initial = self._resolve_distribution(
                self.model["initial_state_counts"], self.initial_alpha
            )
            self._episode_transition = {
                state: self._resolve_transition_distribution(state)
                for state in self.states
            }
            self.current_state = _sample_probs(self.rng, self._episode_initial)
            self.remaining_run_length = self._sample_run_length(self.current_state)
        elif self.remaining_run_length <= 0:
            self.current_state = _sample_probs(self.rng, self._episode_transition[self.current_state])
            self.remaining_run_length = self._sample_run_length(self.current_state)

        state = self.current_state
        self.remaining_run_length -= 1
        record = self._sample_emission(state)
        action_ready_ms = float(record["action_ready_ms"])
        worker_service_ms = float(record["worker_service_ms"])
        action_ready_raw_frames = self.clock.latency_ms_to_raw_frames(action_ready_ms)
        worker_service_raw_frames = self.clock.latency_ms_to_raw_frames(worker_service_ms)
        return LatencySample.from_action_ready(
            action_ready_raw_frames=action_ready_raw_frames,
            worker_service_raw_frames=worker_service_raw_frames,
            worker_service_ms=worker_service_ms,
            latency_ms=action_ready_ms,
            clock=self.clock,
        )

    def _resolve_distribution(self, counts: dict[str, Any], alpha: float) -> dict[str, float]:
        concentration = {state: float(counts.get(state, 0)) + alpha for state in self.states}
        if self.transition_mode == "posterior_predictive":
            total = sum(concentration.values())
            return {state: value / total for state, value in concentration.items()}
        return _dirichlet_sample(self.rng, concentration)

    def _resolve_transition_distribution(self, state: str) -> dict[str, float]:
        transition_counts = self.model["transition_counts"]
        if state in transition_counts:
            return self._resolve_distribution(transition_counts[state], self.transition_alpha)
        return {
            "initial": self._episode_initial,
        }[self.model["terminal_transition_policy"]]

    def _sample_run_length(self, state: str) -> int:
        if not self.semi_markov_dwell:
            return 1
        return int(self.rng.choice(self.model["run_lengths_by_state"][state]))

    def _sample_emission(self, state: str) -> dict[str, Any]:
        distribution = self.model["emission_distributions_by_state"][state]
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


def _dirichlet_sample(rng: random.Random, concentration: dict[str, float]) -> dict[str, float]:
    weights = {
        state: (rng.gammavariate(value, 1.0) if value > 0 else 0.0)
        for state, value in concentration.items()
    }
    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(weights)
        return {state: uniform for state in weights}
    return {state: value / total for state, value in weights.items()}


def _sample_probs(rng: random.Random, probs: dict[str, float]) -> str:
    draw = rng.random()
    cumulative = 0.0
    state = ""
    for state, prob in probs.items():
        cumulative += prob
        if draw < cumulative:
            return state
    return state


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


_BAYES_MARKOV_KEYS = (
    "transition_alpha",
    "initial_alpha",
    "emission_bootstrap",
    "semi_markov_dwell",
    "transition_mode",
)


@register_sampler("temporal")
def _build_temporal(cfg, *, profile, profile_path, clock, seed) -> LatencySampler:
    model_path = _profile_relative_path(profile_path, profile["burst_model_path"])
    model = read_json(model_path)
    kwargs = {key: cfg[key] for key in _BAYES_MARKOV_KEYS if key in cfg}
    return BayesMarkovSampler(model, clock=clock, seed=seed, **kwargs)


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
