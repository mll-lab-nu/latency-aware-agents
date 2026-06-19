from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from latency_bench.core.clock import EnvClock
from latency_bench.latency.inverse_cdf import sample_empirical, sample_inverse_cdf
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
# Profile methods: iid + temporal (semi-Markov v1, queue-conditioned v2)
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


class SemiMarkovSampler(LatencySampler):
    """Run-length semi-Markov over ``latency_raw_frames`` states (burst model v1)."""

    latency_type = "profile_sample"

    def __init__(self, model: dict[str, Any], *, clock: EnvClock, seed: int | None = None):
        self.model = model
        self.clock = clock
        self.rng = random.Random(seed)
        self.current_episode_id: int | None = None
        self.current_state: str | None = None
        self.remaining_run_length = 0

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        episode_id = int(context["episode_id"])
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            self.current_state = self._sample_initial_state()
            self.remaining_run_length = self._sample_run_length(self.current_state)
        elif self.remaining_run_length <= 0:
            self.current_state = self._sample_next_state(self.current_state)
            self.remaining_run_length = self._sample_run_length(self.current_state)

        state = self.current_state
        self.remaining_run_length -= 1
        return LatencySample.from_latency_ms(
            self._sample_latency_ms(state),
            clock=self.clock,
            latency_raw_frames=int(state),
        )

    def _sample_initial_state(self) -> str:
        return _sample_counts(self.rng, self.model["initial_state_counts"])

    def _sample_next_state(self, state: str) -> str:
        transitions = self.model["transition_counts"]
        if state in transitions:
            return _sample_counts(self.rng, transitions[state])
        return self._sample_initial_state()

    def _sample_run_length(self, state: str) -> int:
        return int(self.rng.choice(self.model["run_lengths_by_state"][state]))

    def _sample_latency_ms(self, state: str) -> float:
        return sample_empirical(self.rng, self.model["latency_ms_by_state"][state])


class QueueConditionedSampler(LatencySampler):
    """Iid draw conditioned on the recent observation-drop count (burst model v2)."""

    latency_type = "profile_sample"

    def __init__(self, model: dict[str, Any], *, clock: EnvClock, seed: int | None = None):
        self.model = model
        self.clock = clock
        self.rng = random.Random(seed)
        self.recent_drop_count_cap = int(model["recent_drop_count_cap"])

    def sample(self, context: dict[str, Any] | None = None) -> LatencySample:
        recent_drop_count = min(int(context["recent_drop_count"]), self.recent_drop_count_cap)
        state = str(recent_drop_count)
        record = self.rng.choice(self.model["samples_by_state"][state])
        return LatencySample.from_action_ready(
            action_ready_raw_frames=int(record["action_ready_raw_frames"]),
            worker_service_raw_frames=int(record["worker_service_raw_frames"]),
            worker_service_ms=float(record["worker_service_ms"]),
            clock=self.clock,
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


def _sample_counts(rng: random.Random, counts: dict[str, int]) -> str:
    total = sum(int(value) for value in counts.values())
    draw = rng.randrange(total)
    cumulative = 0
    state = ""
    for state, count in counts.items():
        cumulative += int(count)
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
    if _is_queue_conditioned_model(model):
        return QueueConditionedSampler(model, clock=clock, seed=seed)
    return SemiMarkovSampler(model, clock=clock, seed=seed)


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
    "semi_markov_latency_frames": "temporal",
    "queue_conditioned": "temporal",
}


def _normalize_latency_cfg(latency_cfg: dict[str, Any]) -> dict[str, Any]:
    if "method" in latency_cfg:
        cfg = dict(latency_cfg)
        temporal = cfg.get("temporal")
        if isinstance(temporal, dict) and "variant" in temporal:
            cfg["variant"] = temporal["variant"]
        return cfg

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
        sample_model_type = profile.get("latency_method") or profile.get("sample_model_type")
        if sample_model_type in _LEGACY_PROFILE_SAMPLER_TO_METHOD:
            return _LEGACY_PROFILE_SAMPLER_TO_METHOD[sample_model_type]
        if sample_model_type in ("iid", "temporal"):
            return sample_model_type
    return "iid"


def _maybe_load_profile(cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    profile_path = cfg.get("profile_path")
    if profile_path is None:
        return None, None
    resolved = _resolve_profile_manifest_path(profile_path)
    return load_profile_manifest(resolved), resolved


def _is_queue_conditioned_model(model: dict[str, Any]) -> bool:
    return (
        int(model.get("schema_version", 1)) >= 2
        or model.get("model_type") == "queue_conditioned"
        or model.get("state_key") == "recent_drop_count_capped"
    )


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
