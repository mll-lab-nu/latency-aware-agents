"""Matches of the air-hockey two-mode pilot without the realtime service (lockstep: the program acts every TICK control
steps and its commands take effect at the next step, as over HTTP with no thinking time). Used by the post-match
tests and the experiments.

  Drivers (seat A):
    PlatformDriver(strategy)       HIGH: a strategy's move / strike executed by the platform (world.Seat 'high')
    PlayerDriver(player)           HIGH: a deliverable player (act(view) -> move / strike / none)
    ImplDriver(strategy, impl)     LOW: a strategy's move / strike turned into plans by an implementation of the skill
                                   API (move(view, x, y), strike(view, req), step(view) -> plan / hold) (world.Seat 'low')
  play(driver, opp_level, seed) -> the game's result (score, goals, own goals, strikes, refusals, ...)"""
from __future__ import annotations

import time

import numpy as np

import world as W

TICK = 2                     # the program acts every 2 control steps (20 ms)


class PlatformDriver:
    level = "high"

    def __init__(self, strategy_factory):
        self.f = strategy_factory
        self.s = strategy_factory()

    def reset(self):
        self.s = self.f()

    def __call__(self, view):
        return self.s.tick(view["state"])


class PlayerDriver:
    level = "high"

    def __init__(self, player):
        self.p = player

    def reset(self):
        pass

    def __call__(self, view):
        return self.p("act", view) if callable(self.p) else self.p.act(view)


class ImplDriver:
    level = "low"

    def __init__(self, strategy_factory, impl):
        self.f, self.impl = strategy_factory, impl
        self.s = strategy_factory()
        self.errors = 0

    def reset(self):
        self.s = self.f()

    def call(self, name, *a):
        if callable(self.impl):
            return self.impl(name, *a)
        return getattr(self.impl, name)(*a)

    def __call__(self, view):
        cmds = self.s.tick(view["state"])
        out = []
        for c in cmds:
            if c["type"] == "move":
                out = self.call("move", view, c["x"], c["y"]) or []
            elif c["type"] == "strike":
                out = self.call("strike", view, {k: c[k] for k in ("t", "x", "y", "dx", "dy", "speed")}) or []
        if not cmds:
            out = self.call("step", view) or []
        return out


def play(driver, opp_level=1, seed=1, first="A", max_s=None, on_step=None):
    """One game of the official format (first to 7, the 120 s cap) against the scripted opponent at opp_level."""
    h = W.ModesHockey(seed=seed, level=opp_level, duel=False, first=first)
    seat = W.Seat(h, driver.level)
    replies = []
    errors = []
    t0 = time.time()
    while not h.done:
        if h.awaiting_review:
            h.ready()
            driver.reset()
            replies = []
            continue
        if max_s is not None and h.k * W.DT >= max_s:
            break
        if h.k % TICK == 0:
            view = {"state": h.match_view("A"), "replies": replies}
            try:
                cmds = driver(view) or []
            except Exception as e:  # noqa: BLE001  (a failing implementation: the arm holds, the game goes on)
                errors.append(f"{type(e).__name__}: {e}"[:300])
                cmds = []
                if len(errors) > 50:
                    break
            if cmds:
                replies = [seat.command(c) for c in cmds]
        h.step()
        if on_step:
            on_step(h)
    sc = h.scores()
    goals = [r for r in h.results if r["how"] == "goal"]
    return {"opp_level": opp_level, "seed": seed, "score": [sc["A"]["points"], sc["B"]["points"]],
            "won": h.winner() == "A" if h.done else False, "complete": h.done, "decided_by": h.decided_by,
            "goals_for": sum(g["winner"] == "A" for g in goals), "goals_against": sum(g["winner"] == "B" for g in goals),
            "own_goals_against": sum(g["winner"] == "B" and bool(g.get("own_goal")) for g in goals),
            "rallies": len(h.results), "fouls_A": sc["A"]["fouls"], "fouls_B": sc["B"]["fouls"],
            "strikes": seat.n_strikes, "plans": seat.n_plans, "moves": seat.n_moves, "refusals": seat.refusals, "refusal_kinds": seat.reasons,
            "errors": len(errors), "error_samples": errors[:3], "t_play": round(h.k * W.DT, 2),
            "cpu_s": round(time.time() - t0, 1)}
