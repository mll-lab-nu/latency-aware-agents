"""Smoke matches of the world (line B's delivery, interfaces v1 §4): (1) both teams stand still for a whole half;
(2) one robot takes the ball, dribbles it under 1 m, turns to the goal and shoots: a goal; (3) a foul in play is
judged (a robot rams an opponent at 2.5 m/s: crashing, §8.4.2.3). The scripted robot below reads the truth (it is a
test of the world, not a player). Run: python3 sports/soccer_ssl/tests/test_smoke.py [name-filter]; the idle half
takes a few hundred seconds of world time (RCB_SSL_SMOKE_HALF_S shortens it)."""

from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import Carrot, events, place, run, run_all, running, target, world  # noqa: E402

OUT = os.environ.get("RCB_SSL_SMOKE_OUT")


def _log(name, w, t0, **kw):
    rec = dict(
        test=name,
        world_time_s=round(float(w.data.time), 2),
        wall_s=round(time.time() - t0, 1),
        referee=w.referee.status(),
        events=[e for e in w.referee.events],
        **kw,
    )
    if OUT:
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(rec, f, indent=1, default=str)
    return rec


def test_smoke_idle_half():
    """both teams stand still: the kick-off is in play after 10 s (§5.4), no progress every 10 s (§8.1) restarts with
    a force start, nothing else happens, and the half ends after 300 s of playing time with 0:0."""
    half = float(os.environ.get("RCB_SSL_SMOKE_HALF_S", "300"))
    t0 = time.time()
    w = world(seed=11, half_s=half)
    while not events(w, "HALF_TIME") and w.data.time < 4 * half + 60:
        w.step()
    R = w.referee
    kinds = {e["type"] for e in R.events}
    _log("smoke_idle_half", w, t0, kinds=sorted(kinds))
    assert events(w, "HALF_TIME") and R.score == {"blue": 0, "yellow": 0}
    assert kinds <= {"KICKOFF", "BALL_IN_PLAY", "NO_PROGRESS_IN_GAME", "FORCE_START", "HALF_TIME"}, kinds
    assert R.fouls == {"blue": 0, "yellow": 0}


def _shooter(w, rid, goal=(4.5, -0.25), carry=0.6, kick_speed=5.0, log=None):
    """a scripted (truth-reading) attacker: approach the ball from behind, dribble `carry` metres, turn to the goal,
    shoot. Returns a policy for run()."""
    st = dict(phase="approach", t=0.0, start=None)

    def pol(w):
        p, _ = w.robot_pose(rid)
        if w.referee.state != "RUNNING":  # the game stopped (goal): stand still (stop rules, §5.1.1)
            if st["phase"] != "done":
                target(w, rid, p[0], p[1], p[2])
                w.apply(rid[0], rid[1], "dribble", dict(rpm=0))
                st["phase"] = "done"
            return
        bx, by = float(w.ball.pos[0]), float(w.ball.pos[1])
        rig = w.rigs[rid]
        beam = rig.state().get("ball_in_beam", False) if hasattr(rig, "state") else False
        st["t"] += 0.01
        if st["phase"] == "approach":
            yaw = math.atan2(by - p[1], bx - p[0]) if math.hypot(bx - p[0], by - p[1]) > 0.3 else p[2]
            yaw = 0.0
            target(w, rid, bx - 0.3, by, yaw)
            if math.hypot(p[0] - (bx - 0.3), p[1] - by) < 0.03:
                w.apply(rid[0], rid[1], "dribble", dict(rpm=4000))
                st.update(phase="grab", t=0.0)
        elif st["phase"] == "grab":
            target(w, rid, bx - 0.06, by, 0.0)
            if beam or st["t"] > 1.5:
                st.update(phase="carry", t=0.0, start=(p[0], p[1]))
        elif st["phase"] == "carry":
            gx, gy = goal
            yaw = math.atan2(gy - by, gx - bx)
            sx, sy = st["start"]
            target(w, rid, sx + carry, sy, yaw)
            if (
                math.hypot(p[0] - sx - carry, p[1] - sy) < 0.03
                and abs(math.atan2(math.sin(p[2] - yaw), math.cos(p[2] - yaw))) < 0.02
            ):
                w.apply(rid[0], rid[1], "kick", dict(speed=kick_speed, armed_for=2.0))
                st.update(phase="shot", t=0.0)
        elif st["phase"] == "shot":  # keep the armed kicker on the ball: it fires when the beam is blocked
            gx, gy = goal
            yaw = math.atan2(gy - by, gx - bx)
            target(w, rid, bx - 0.05 * math.cos(yaw), by - 0.05 * math.sin(yaw), yaw)
        if log is not None:
            log.append(
                (round(float(w.data.time), 2), st["phase"], [round(v, 3) for v in p], [round(bx, 3), round(by, 3)])
            )

    return pol


def test_smoke_dribble_and_shoot():
    """one robot dribbles the ball 0.6 m (under the 1 m limit), turns towards the goal past the keeper and shoots:
    a valid goal, no foul."""
    t0 = time.time()
    w = world(seed=12)
    B = ("blue", 2)
    place(w, B, 1.0, -1.6, 0.0)
    running(w, 1.6, -1.0)
    trace = []
    run(w, 6.0, _shooter(w, B, log=trace))
    g = events(w, "GOAL")
    _log("smoke_dribble_and_shoot", w, t0, trace=trace[::10])
    assert g and g[0]["team"] == "blue" and g[0]["kicked_by"] == "blue2", events(w, "GOAL", "INVALID_GOAL")
    assert w.referee.fouls == {"blue": 0, "yellow": 0}, w.referee.events


def test_smoke_foul_is_judged():
    """after the kick-off, blue3 rams the standing yellow2 at about 2.5 m/s: crashing (§8.4.2.3) is called on blue."""
    t0 = time.time()
    w = world(seed=13)
    run(w, 2.1)  # the kick-off (both teams in position)
    Bp, _ = w.robot_pose(("blue", 3))
    Yp, _ = w.robot_pose(("yellow", 2))  # straight along the y = -1.5 lane
    dx, dy = Yp[0] - Bp[0], Yp[1] - Bp[1]
    c = Carrot(w, ("blue", 3), dx, dy, 2.5)

    def ram(w):
        if not events(w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN"):
            c(w)
        else:
            p, _ = w.robot_pose(("blue", 3))
            target(w, ("blue", 3), p[0], p[1], p[2])

    run(w, 4.0, ram)
    f = events(w, "BOT_CRASH_UNIQUE")
    _log("smoke_foul_is_judged", w, t0)
    assert f and f[0]["team"] == "blue" and f[0]["robot"] == "blue3", events(w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN")
    assert w.referee.fouls["blue"] >= 1


if __name__ == "__main__":
    sys.exit(1 if run_all(globals()) else 0)
