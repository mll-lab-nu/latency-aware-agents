"""Helpers of the soccer_ssl tests (not a test module). The tests drive robots through their rig with world-frame
targets (a target moving at the wanted speed: any rig with position targets follows it) and set the referee's state
directly with Referee.force(); they read only the referee's events."""

from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPORT_DIR = os.path.dirname(HERE)
if SPORT_DIR not in sys.path:
    sys.path.insert(0, SPORT_DIR)

import mujoco  # noqa: E402

from sim import field as F  # noqa: E402,F401
from sim.world import CONTROL_DT, World  # noqa: E402


def world(seed=1, **kw) -> World:
    return World(seed=seed, **kw)


def place(w, rid, x, y, yaw=0.0):
    w._reset_robot(rid, (x, y, yaw))
    mujoco.mj_forward(w.model, w.data)


def sideline(w, keep=()):
    """take every robot not in `keep` off the field (parked outside the walls, out of the way)."""
    for rid in w.ids:
        if rid not in keep:
            w.remove_robot(rid)
    mujoco.mj_forward(w.model, w.data)


def ball(w, x, y, vel=(0.0, 0.0, 0.0), z=None):
    w.ball.place(x, y, F.BALL_R if z is None else z, vel=vel)
    mujoco.mj_forward(w.model, w.data)


def running(w, bx=None, by=None):
    if bx is not None:
        ball(w, bx, by)
    w.referee.force("RUNNING", ball=w.ball.pos)


def target(w, rid, x, y, yaw=0.0):
    w.rigs[rid].set_base_target(x, y, yaw)


class Carrot:
    """a position target moving at `speed` along (dx, dy) from the robot's position: the robot follows at that speed."""

    def __init__(self, w, rid, dx, dy, speed, yaw=None, lead=0.05):
        p, _ = w.robot_pose(rid)
        n = math.hypot(dx, dy)
        self.u = (dx / n, dy / n)
        self.w, self.rid, self.v = w, rid, speed
        self.pos = [p[0] + self.u[0] * lead, p[1] + self.u[1] * lead]
        self.yaw = p[2] if yaw is None else yaw

    def __call__(self, *_):
        self.pos[0] += self.u[0] * self.v * CONTROL_DT
        self.pos[1] += self.u[1] * self.v * CONTROL_DT
        target(self.w, self.rid, self.pos[0], self.pos[1], self.yaw)


def run(w, seconds, *policies):
    for _ in range(int(round(seconds / CONTROL_DT))):
        for p in policies:
            p(w)
        w.step()


def events(w, *types):
    return [e for e in w.referee.events if e["type"] in types]


def run_all(module_globals):
    """python3 tests/test_x.py: run every test_* function of the module, print pass / fail and the time."""
    names = sorted(k for k, v in module_globals.items() if k.startswith("test_") and callable(v))
    only = sys.argv[1:]
    bad = 0
    for k in names:
        if only and not any(o in k for o in only):
            continue
        t0 = time.time()
        try:
            module_globals[k]()
            print(f"PASS {k} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback

            traceback.print_exc()
            print(f"FAIL {k} ({time.time() - t0:.1f} s): {e!r}", flush=True)
    print(f"{len(names) - bad if not only else 'selected'} passed, {bad} failed")
    return bad
