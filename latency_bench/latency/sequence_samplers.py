"""Stateful timing sequence replay without fitting dependencies."""
from __future__ import annotations

import copy
import numpy as np

from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import LatencySampler


class HmmSampler(LatencySampler):
    latency_type = "hmm"
    unbounded_latency = True

    def __init__(self, model, *, clock, seed=None):
        self.model, self.clock = model, clock
        self.rng = np.random.RandomState(seed)
        self.state = None
        self.startprob = np.asarray(model["startprob"])
        self.transmat = np.asarray(model["transmat"])
        self.means, self.covars = np.asarray(model["means"]), np.asarray(model["covars"])
        self.center, self.scale = np.asarray(model["center"]), np.asarray(model["scale"])

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.rng = copy.deepcopy(self.rng, memo)
        return clone

    def sample(self, context=None):
        _check_worker(context)
        probabilities = self.startprob if self.state is None else self.transmat[self.state]
        self.state = int(np.searchsorted(np.cumsum(probabilities), self.rng.rand(), side="right"))
        features = self.rng.multivariate_normal(self.means[self.state], self.covars[self.state])
        with np.errstate(over="raise", invalid="raise"):
            busy, overhead = np.exp(features * self.scale + self.center)
        return _pair_sample(float(busy + overhead), float(busy), self.clock)


class BlockResamplingSampler(LatencySampler):
    latency_type = "block"

    def __init__(self, model, *, clock, seed=None):
        self.model, self.clock = model, clock
        self.rng = np.random.default_rng(seed)
        self.sessions = [np.asarray(s, dtype=float) for s in model["sessions"]]
        self.cumulative = np.cumsum([len(s) for s in self.sessions])
        self.remaining = 0
        self.source_id = self.offset = 0

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.rng = copy.deepcopy(self.rng, memo)
        return clone

    def sample(self, context=None):
        _check_worker(context)
        if self.remaining == 0:
            flat_start = int(self.rng.integers(self.cumulative[-1]))
            self.source_id = int(np.searchsorted(self.cumulative, flat_start, side="right"))
            self.offset = flat_start - (int(self.cumulative[self.source_id - 1]) if self.source_id else 0)
            self.remaining = int(self.rng.geometric(1. / self.model["mean_block_length"]))
        source = self.sessions[self.source_id]
        latency, busy = source[self.offset]
        self.offset = (self.offset + 1) % len(source)
        self.remaining -= 1
        return _pair_sample(float(latency), float(busy), self.clock)


class AutoregressiveSampler(LatencySampler):
    unbounded_latency = True

    def __init__(self, model, *, clock, seed=None):
        self.model, self.clock = model, clock
        self.latency_type = model["model_type"]
        self.rng = np.random.default_rng(seed)
        self.ar, self.ma = np.asarray(model["ar"]), np.asarray(model["ma"])
        self.intercept = np.asarray(model["intercept"])
        self.center, self.scale = np.asarray(model["center"]), np.asarray(model["scale"])
        self.covariance = np.asarray(model["innovation_covariance"])
        # For y_t = phi*y_(t-1) + e_t + theta*e_(t-1), the MA-infinity
        # coefficients are 1, (phi+theta), phi*(phi+theta), ... .
        stationary = self.covariance * (1 + np.outer(self.ar + self.ma, self.ar + self.ma)
                                         / (1 - np.outer(self.ar, self.ar)))
        joint = np.block([[stationary, self.covariance], [self.covariance, self.covariance]])
        initial = self.rng.multivariate_normal(np.zeros(4), joint, check_valid="raise")
        self.value = self.intercept / (1 - self.ar) + initial[:2]
        self.innovation = initial[2:]

    def __deepcopy__(self, memo):
        clone = copy.copy(self)
        memo[id(self)] = clone
        clone.rng = copy.deepcopy(self.rng, memo)
        return clone

    def sample(self, context=None):
        _check_worker(context)
        innovation = self.rng.multivariate_normal(np.zeros(2), self.covariance, check_valid="raise")
        self.value = self.intercept + self.ar * self.value + self.ma * self.innovation + innovation
        self.innovation = innovation
        with np.errstate(over="raise", invalid="raise"):
            busy, overhead = np.exp(self.value * self.scale + self.center)
        return _pair_sample(float(busy + overhead), float(busy), self.clock)


def _check_worker(context):
    if context is None or context["worker_slot"] != 0:
        raise ValueError("Sequence replay requires worker_slot=0 (capacity 1)")


def _pair_sample(latency, busy, clock):
    return LatencySample.from_action_ready(
        action_ready_raw_frames=clock.latency_ms_to_raw_frames(latency),
        worker_service_raw_frames=clock.latency_ms_to_raw_frames(busy),
        worker_service_ms=busy, latency_ms=latency, clock=clock,
    )
