"""Drills of the built-in players' building blocks in the world (development aid; numbers in the players report):
  collect  a resting ball at a random spot; one robot gets it and shoots at the goal (a still keeper on the line)
  pass     robot 1 holds the ball, robot 2 stands 1.5-5 m away; robot 1 passes, robot 2 receives (no opponents)
Each trial reports success and the time it took. The team is builtin_players.Reference driven exactly as in a match
(truth view + own sensors every 4 control steps, commands 4 steps later, at most 5 per act).

    python3 sports/soccer_ssl/screen/scenarios.py collect --n 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPORT = os.path.dirname(HERE)
sys.path.insert(0, SPORT)
sys.path.insert(0, HERE)

import builtin_players as BP  # noqa: E402
from worldrun import AgentSide  # noqa: E402


def setup(seed, keep):
    from sim.world import World

    w = World(seed=seed, half_s=600, first_kickoff="blue")
    for rid in w.ids:
        if rid not in keep:
            w.remove_robot(rid)
    return w


def running(w, x, y):
    w.ball.place(x, y)
    w.mj.mj_forward(w.model, w.data)
    w.referee.force("RUNNING", ball=w.ball.pos)


def drill_collect(n, seed0):
    out = []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        w = setup(seed0 + i, keep=[("blue", 1), ("yellow", 0)])
        side = AgentSide("reference", "blue", seed0 + i)
        bx, by = rng.uniform(-1.0, 2.5), rng.uniform(-2.5, 2.5)
        w._reset_robot(("blue", 1), (rng.uniform(-3, 0), rng.uniform(-2.5, 2.5), rng.uniform(-3, 3)))
        w._reset_robot(("yellow", 0), (4.45, 0.0, math.pi))
        running(w, bx, by)
        got, res = None, "timeout"
        for k in range(1500):
            side.tick(w)
            w.step()
            if got is None and w.rigs[("blue", 1)].state()["ball_in_beam"]:
                got = round(w.data.time, 2)
            ev = [e["type"] for e in w.referee.events]
            if "GOAL" in ev:
                res = "goal"
                break
            if w.referee.state != "RUNNING":
                res = "stop:" + ",".join(e for e in ev if e not in ("FORCE_START",))[-60:]
                break
        out.append(dict(i=i, ball=[round(bx, 2), round(by, 2)], got=got, res=res, t=round(w.data.time, 2)))
    return out


class PassOnly(BP.Reference):
    """the reference whose carrier always passes to robot 2 (the drill)."""

    def decide(self, n, dd):
        o = self._pass_option(n, 2, self.v.own[2].p)
        if o is None or n == 2:
            return dict(num=n, kind="carry", target=self.v.own[n].p, speed=0.0, mode="flat", t=self.v.t)
        return dict(num=n, kind="pass", target=o["target"], speed=o["speed"], mode=o["mode"], recv=2, T=o["T"],
                    t=self.v.t, margin=o["margin"])

    def attack_spots(self, n, carrier):
        return [self.v.own[k].p for k in self.field_ids() if k != carrier][:n]


def drill_pass(n, seed0):
    out = []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        w = setup(seed0 + i, keep=[("blue", 1), ("blue", 2)])
        side = AgentSide("reference", "blue", seed0 + i)
        side.team = PassOnly(team="blue", seed=i)
        D = rng.uniform(1.5, 5.0)
        a = rng.uniform(-math.pi, math.pi)
        p1 = (-1.0, 0.0)
        p2 = (p1[0] + D * math.cos(a), max(-2.7, min(2.7, p1[1] + D * math.sin(a))))
        yaw = rng.uniform(-math.pi, math.pi)
        w._reset_robot(("blue", 1), (p1[0], p1[1], yaw))
        w._reset_robot(("blue", 2), (p2[0], p2[1], 0.0))
        kp = w.kick_point(("blue", 1))
        running(w, kp[0], kp[1])
        w.execute("blue", 1, "dribble", dict(rpm=3000))
        res, t_kick = "timeout", None
        for k in range(600):
            side.tick(w)
            w.step()
            if t_kick is None and side.team.stats["passes"]:
                t_kick = round(w.data.time, 2)
            if side.team.stats["pass_ok"]:
                res = "ok"
                break
            if w.referee.state != "RUNNING":
                res = "stop:" + ",".join(e["type"] for e in w.referee.events)[-60:]
                break
        out.append(dict(i=i, D=round(D, 2), res=res, t_kick=t_kick, t=round(w.data.time, 2),
                        lost=side.team.stats["pass_lost"]))
    return out


def drill_keeper(n, seed0, keeper="L3"):
    """shots at the goal (the ball launched at 5.5 m/s from 2.5-5 m towards a random point of the mouth, |y| <= 0.4)
    against a keeper alone: `keeper` is a scripted level ("L1".."L3") or "reference". Reports saves."""
    from worldrun import ScriptedSide

    out = []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        w = setup(seed0 + i, keep=[("yellow", 0)])
        side = ScriptedSide(int(keeper[1:]), "yellow", seed0 + i) if keeper.startswith("L") else \
            AgentSide(keeper, "yellow", seed0 + i)
        s = w.sign("yellow")  # yellow defends the goal at x = -4.5 * s
        gx = -4.5 * s
        d = rng.uniform(2.5, 5.0)
        a = rng.uniform(-1.0, 1.0)
        bx, by = gx + s * d * math.cos(a), d * math.sin(a)
        by = max(-2.8, min(2.8, by))
        ty = rng.uniform(-0.4, 0.4)
        w._reset_robot(("yellow", 0), (gx + 0.2 * s, 0.0, 0.0 if s > 0 else math.pi))
        running(w, bx, by)
        # let the keeper settle on the ball's line for 1.5 s
        for k in range(150):
            side.tick(w)
            w.step()
        dx, dy = gx - bx, ty - by
        L = math.hypot(dx, dy)
        w.ball.place(bx, by, vel=(5.5 * dx / L, 5.5 * dy / L, 0.0))
        w.mj.mj_forward(w.model, w.data)
        res = "save"
        for k in range(200):
            side.tick(w)
            w.step()
            ev = [e["type"] for e in w.referee.events]
            if "GOAL" in ev:
                res = "goal"
                break
            if w.referee.state != "RUNNING":
                break
        out.append(dict(i=i, d=round(d, 2), a=round(a, 2), ty=round(ty, 2), res=res))
    return out


def drill_penalty(n, seed0, kicker="reference", keeper="L3"):
    """penalty kicks (§5.3.5) with both teams played by their programs: `kicker` (a VARIANTS / make name) for blue,
    `keeper` (a scripted level or a name) for yellow. A penalty for blue is called from a running game; reports the
    referee's outcome of each."""
    from worldrun import ScriptedSide

    out = []
    for i in range(n):
        from sim.world import World

        w = World(seed=seed0 + i, half_s=600, first_kickoff="blue")
        sides = {"blue": AgentSide(kicker, "blue", seed0 + i),
                 "yellow": ScriptedSide(int(keeper[1:]), "yellow", seed0 + i) if keeper.startswith("L")
                 else AgentSide(keeper, "yellow", seed0 + i)}
        running(w, 0.0, 0.0)
        s = w.sign("blue")
        from sim import field as F

        w.referee.stop(("PENALTY", "blue"), F.penalty_mark(s))
        res = "none"
        n_ev = 0
        for k in range(3000):
            for sd in sides.values():
                sd.tick(w)
            w.step()
            ev = w.referee.events[n_ev:]
            n_ev = len(w.referee.events)
            done = [e for e in ev if e["type"] in ("GOAL", "PENALTY_KICK_FAILED")]
            if done:
                res = done[0]["type"] + ":" + str(done[0].get("why"))
                break
        out.append(dict(i=i, res=res, t=round(w.data.time, 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drill", choices=["collect", "pass", "keeper", "penalty"])
    ap.add_argument("--kicker", default="reference")
    ap.add_argument("--keeper", default="L3")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.drill == "penalty":
        res = drill_penalty(a.n, a.seed, a.kicker, a.keeper)
        print("goals", sum(r["res"].startswith("GOAL") for r in res), "/", len(res))
    elif a.drill == "keeper":
        res = drill_keeper(a.n, a.seed, a.keeper)
        print("saves", sum(r["res"] == "save" for r in res), "/", len(res))
    else:
        res = (drill_collect if a.drill == "collect" else drill_pass)(a.n, a.seed)
    for r in res:
        print(json.dumps(r))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
