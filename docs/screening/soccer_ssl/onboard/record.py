"""Record a stretch of a real match for the onboard experiments: two built-in teams (line D's scripted teams through
its world-level harness sports/soccer_ssl/screen/worldrun.py, or random movers) play in sim.world.World; every
control step (10 ms) the recorder keeps the full qpos (so any onboard picture can be re-drawn later, rendering is a
pure function of qpos), every robot's true pose and velocity, the base setpoint velocity of its motor controller
(line A's rig: the wheel-slip model of sensors.py needs it), the ball, robot-robot / robot-wall contacts and the
referee's teleports (a robot moved or removed by the referee).

Runs against a checkout that has line D's players (origin/ssl-players), e.g.

    SSL_TREE=/path/to/ssl-players-checkout python record.py --blue L2 --yellow L3 --seed 1 --seconds 120 \\
        --out results/rec_L2L3_s1.npz
Output: an .npz with arrays (T = number of control steps, R = 10 robots)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

TREE = os.environ.get("SSL_TREE", "/tree")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "sports", "soccer_ssl"))
sys.path.insert(0, os.path.join(TREE, "sports", "soccer_ssl", "screen"))


class RandomSide:
    """every 1-2 s a new random target per robot (team frame), sometimes a spin (the 'sharp turn' case)."""

    def __init__(self, team, seed):
        self.team, self.rng, self.next = team, np.random.default_rng(seed), {}

    def tick(self, w):
        for rid in w.ids:
            if rid[0] != self.team or w.data.time < self.next.get(rid, 0.0):
                continue
            r = self.rng
            w.execute(self.team, rid[1], "move", dict(x=float(r.uniform(-4.0, 4.0)), y=float(r.uniform(-2.7, 2.7)),
                                                      theta=float(r.uniform(-math.pi, math.pi))))
            self.next[rid] = w.data.time + float(r.uniform(1.0, 2.0))


def _cmd(w, team, num, x, y, th):
    """a move in the world frame (the drill thinks in world coordinates; blue attacks +x in the first half)."""
    if team == "yellow":
        x, y, th = -x, -y, th + math.pi
    th = (th + math.pi) % (2 * math.pi) - math.pi
    w.execute(team, num, "move", dict(x=float(x), y=float(y), theta=float(th)))


class DrillSide:
    """the bump-and-spin drill (both teams): blue3 spins on the spot at up to the 20 rad/s yaw limit for 3 s, then
    drives 1-1.5 m, every 6 s; blue2 patrols between (-2, +-1); yellow2 rams blue2 from the side at full speed every
    8 s; yellow4 drives head-on through blue4 (standing at (-1.2, -1.5)) every 10 s, starting at 5 s; the others
    stand."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.spin_goal = None

    def tick(self, w):
        t = float(w.data.time)
        k = w.k
        if k % 10:
            return
        pose = {rid: w.robot_pose(rid)[0] for rid in w.ids}
        # spinner
        ph = t % 6.0
        x, y, yaw = pose[("blue", 3)]
        if ph < 3.0:
            _cmd(w, "blue", 3, x, y, yaw + 2.0)
            self.spin_goal = None
        else:
            if self.spin_goal is None:
                a = self.rng.uniform(-math.pi, math.pi)
                r = self.rng.uniform(1.0, 1.5)
                gx, gy = np.clip(x + r * math.cos(a), -4.0, -0.7), np.clip(y + r * math.sin(a), -2.6, 2.6)
                self.spin_goal = (float(gx), float(gy), float(self.rng.uniform(-math.pi, math.pi)))
            _cmd(w, "blue", 3, *self.spin_goal)
        # victim patrol
        side = 1.0 if (t % 6.0) < 3.0 else -1.0
        _cmd(w, "blue", 2, -2.0, side, 0.0)
        # rammer
        vx, vy, _ = pose[("blue", 2)]
        ph = t % 8.0
        if ph < 2.5:
            _cmd(w, "yellow", 2, vx + 0.3, vy + 1.6 * (1 if vy < 0 else -1), -math.pi / 2 if vy < 0 else math.pi / 2)
        elif ph < 5.0:
            d = 1 if vy > pose[("yellow", 2)][1] else -1
            _cmd(w, "yellow", 2, vx, vy + 0.7 * d, d * math.pi / 2)
        else:
            _cmd(w, "yellow", 2, 0.8, 2.0 * (1 if vy > 0 else -1), math.pi)
        # head-on
        ph = (t - 5.0) % 10.0
        if t >= 5.0:
            if ph < 3.0:
                _cmd(w, "yellow", 4, -1.2, 0.5, -math.pi / 2)
            elif ph < 5.5:
                _cmd(w, "yellow", 4, -1.2, -2.2, -math.pi / 2)
            else:
                _cmd(w, "yellow", 4, 1.0, -1.0, math.pi)
        _cmd(w, "blue", 4, -1.2, -1.5, math.pi / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", default="L2")
    ap.add_argument("--yellow", default="L3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mujoco as mj
    import worldrun as WR
    from sim.world import World
    w = World(seed=a.seed, half_s=max(a.seconds + 5, 300.0), first_kickoff="blue" if a.seed % 2 else "yellow")
    sides = {}
    if a.blue == "drill":
        sides["all"] = DrillSide(a.seed)
    for team, name in (() if a.blue == "drill" else (("blue", a.blue), ("yellow", a.yellow))):
        sides[team] = RandomSide(team, a.seed * 7 + len(team)) if name == "random" else WR.side(name, team, a.seed * 7 + len(team))
    m, d = w.model, w.data
    R = len(w.ids)
    T = int(round(a.seconds / 0.01))
    # robot-robot and robot-wall contacts: the geoms of each robot / the walls and goals
    wall_geoms = set()
    for gname in w.handles["field"]["walls"] + w.handles["field"]["goals"]:
        wall_geoms.add(mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, gname))
    rec = dict(t=np.zeros(T), qpos=np.zeros((T, m.nq), np.float32), pose=np.zeros((T, R, 3)), vel=np.zeros((T, R, 3)),
               vsp=np.zeros((T, R, 3)), ball=np.zeros((T, 6)), on=np.zeros((T, R), bool), touch_robot=np.zeros((T, R), bool),
               touch_wall=np.zeros((T, R), bool), teleport=np.zeros((T, R), bool), running=np.zeros(T, bool),
               beam=np.zeros((T, R), bool))
    prev = None
    t0 = time.time()
    for k in range(T):
        for s in sides.values():
            s.tick(w)
        w.step()
        rec["t"][k] = d.time
        rec["qpos"][k] = d.qpos
        rec["ball"][k, :3], rec["ball"][k, 3:] = w.ball.pos, w.ball.vel
        rec["running"][k] = w.referee.state == "RUNNING"
        for i, rid in enumerate(w.ids):
            p, v = w.robot_pose(rid)
            rec["pose"][k, i], rec["vel"][k, i] = p, v
            rig = w.rigs[rid]
            rec["vsp"][k, i] = rig.knots[-1][1] if getattr(rig, "knots", None) else v
            rec["on"][k, i] = w.on_field[rid]
            rec["beam"][k, i] = bool(rig._beam(d)[0]) if hasattr(rig, "_beam") else False
        nc = d.ncon
        if nc:
            g = d.contact.geom[:nc]
            rr = w.geom_robot[g]
            both = (rr[:, 0] >= 0) & (rr[:, 1] >= 0) & (rr[:, 0] != rr[:, 1])
            for i, j in rr[both]:
                rec["touch_robot"][k, i] = rec["touch_robot"][k, j] = True
            for c in range(nc):
                g0, g1 = int(g[c, 0]), int(g[c, 1])
                if g0 in wall_geoms and rr[c, 1] >= 0:
                    rec["touch_wall"][k, rr[c, 1]] = True
                if g1 in wall_geoms and rr[c, 0] >= 0:
                    rec["touch_wall"][k, rr[c, 0]] = True
        if prev is not None:
            jump = np.hypot(*(rec["pose"][k, :, :2] - prev[:, :2]).T)
            rec["teleport"][k] = jump > 0.06          # 6 m/s: faster than any robot can drive in 10 ms
        prev = rec["pose"][k].copy()
    meta = dict(blue=a.blue, yellow=a.yellow, seed=a.seed, seconds=a.seconds, ids=[f"{t}{n}" for t, n in w.ids],
                score=dict(w.referee.score), wall_s=round(time.time() - t0, 1), nq=int(m.nq),
                events=[e for e in w.referee.events if e.get("type") not in ("BALL_IN_PLAY",)][:200])
    np.savez_compressed(a.out, meta=json.dumps(meta, default=str), **rec)
    print(json.dumps({k: meta[k] for k in ("blue", "yellow", "seed", "score", "wall_s")}), "teleports",
          int(rec["teleport"].sum()), "robot contacts (steps)", int(rec["touch_robot"].sum()), flush=True)


if __name__ == "__main__":
    main()
