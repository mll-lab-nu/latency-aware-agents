"""The single-player task's scripted opponent (seat B): a fencer that sees the agent's robot (privileged, as the
referee's simulator does) with a reaction latency and plays one strategy of its level's repertoire per phrase.

Levels (the same robot and limits as the agent's; only the decisions differ):
  1  reaction 0.16 s; direct attacks, attacks on preparation, distance play and feints (no defence on the blade)
  2  reaction 0.10 s; adds beat attacks, parries with ripostes, stop-hits and running attacks
  3  reaction 0.06 s; beats (three times as often as each other action), counter-beats, derobements, parries,
     retreats, stop-hits, low and running attacks, and it prefers what has scored against this agent so far
The choice per phrase is seeded (RCB_SEED) and depends only on the bout so far, so the bout replays bit-exactly. A phrase
without a halt for STALL_S seconds makes it switch to its level's attack on the blade (feint at level 1, beat above),
so waiting does not run the clock out against it.
"""
from __future__ import annotations

import numpy as np

import fencing as F
import strategies as S

LEVELS = {
    1: dict(latency=0.16, repertoire=["lunge", "prep", "distance", "feint"], adaptive=False, breaker="feint"),
    2: dict(latency=0.10, repertoire=["lunge", "prep", "distance", "beat", "parry", "stop", "fleche"], adaptive=False,
            breaker="beat"),
    3: dict(latency=0.06, repertoire=["beat", "counterbeat", "derobe", "parry", "retreat", "lunge_low", "stop", "fleche"],
            adaptive=True, breaker="beat", weights={"beat": 3.0}),
}
STALL_S = 5.0          # a phrase without a halt for this long: the opponent switches to its level's attack on the blade


class Opponent:
    def __init__(self, level: int, seed: int, world: F.World, seat: str = "B"):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}")
        self.level, self.seed, self.w, self.seat = level, int(seed), world, seat
        self.lv = LEVELS[level]
        self.ctl = F.Controller(world, seat)
        self.lag = int(round(self.lv["latency"] / F.CONTROL_DT))
        self.score = {n: [0, 0] for n in self.lv["repertoire"]}     # strategy -> [touches scored, received]
        self.n_phrase = 0
        self.strategy = None
        self.hist = []

    def new_phrase(self):
        """after a placement: choose this phrase's strategy (seeded) and reset the controller"""
        rng = np.random.default_rng([self.seed, self.level, 101, self.n_phrase])
        names = self.lv["repertoire"]
        prior = np.array([self.lv.get("weights", {}).get(n, 1.0) for n in names])
        if self.lv["adaptive"] and self.n_phrase >= 2:
            w = prior * np.array([1.0 + 2.0 * self.score[n][0] - 1.0 * self.score[n][1] for n in names])
            w = np.clip(w, 0.25, None)
        else:
            w = prior
        p = w / w.sum()
        name = names[int(rng.choice(len(names), p=p))]
        self.ctl.reset()
        self.strategy = S.BY_NAME[name](self.ctl, np.random.default_rng([self.seed, self.level, 202, self.n_phrase]))
        self.hist = []
        self.t_switch = 0.0
        self.n_phrase += 1
        return name

    def result(self, scored: bool, received: bool):
        if self.strategy is not None:
            self.score[self.strategy.name][0] += int(scored)
            self.score[self.strategy.name][1] += int(received)

    def target(self, t_phrase: float):
        """the opponent's joint targets for the next control step (decides first, from the delayed view)"""
        self.hist.append(S.snapshot(self.w))
        snap = self.hist[max(0, len(self.hist) - 1 - self.lag)]
        o = S.view(snap, self.seat)
        o["me"] = S.view(self.hist[-1], self.seat)["me"]
        if len(self.hist) > self.lag + 2:
            del self.hist[0]
        if t_phrase - self.t_switch >= STALL_S and self.strategy.st not in ("thrust", "hit", "beat", "feint", "riposte",
                                                                             "parry"):
            # no halt for STALL_S: attack on the blade (also ends a stalling opponent's waiting game)
            self.t_switch = t_phrase
            self.strategy = S.BY_NAME[self.lv["breaker"]](self.ctl, np.random.default_rng([self.seed, 303, self.n_phrase]))
        self.strategy.act(t_phrase, o)
        return self.ctl.next_target()
