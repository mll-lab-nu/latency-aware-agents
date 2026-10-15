"""A small test world for the SSL robot (developer tool and tests; the sport's world is sports/soccer_ssl/sim/):
a carpet, a golf ball and robots, with exactly the sport's ball and carpet (sports/soccer_ssl/sim/ball.py, field.py
on ssl-world, 2026-10-11): 1 ms, implicitfast, elliptic cone, impratio 10; ball r 0.0215, 0.046 kg, condim 6,
priority 1, friction [0.35, 0.005, 0.00138], solref [-20000, -60]; carpet condim 3, friction [1, 0.005, 0.0001],
default solref.

    from bodies.ssl_robot.tools.lab import Lab
    lab = Lab([("blue_0", "blue", 0, (0, 0, 0))], ball=(0.2, 0, None))
    lab.rigs[0].set_base_target(1, 0, 0); lab.run(1.0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bodies.ssl_robot import body as SSL  # noqa: E402

BALL = dict(radius=SSL.BALL_R, mass=0.046, friction=[0.35, 0.005, 0.00138], solref=[-20000.0, -60.0], condim=6,
            priority=1)
CARPET = dict(friction=[1.0, 0.005, 0.0001], condim=3)


def scene(robots, ball=(0.3, 0.0), ball_kw=None, decl=None):
    import mujoco as mj
    B = dict(BALL, **(ball_kw or {}))
    spec = mj.MjSpec()
    spec.option.timestep = 0.001
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    f = spec.worldbody.add_geom(name="carpet", type=mj.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1], rgba=[0.1, 0.4, 0.1, 1])
    f.friction, f.condim = CARPET["friction"], CARPET["condim"]
    hs = [SSL.mount(spec, n, t, k, pose=p, decl=decl) for n, t, k, p in robots]
    bh = None
    if ball is not None:
        bb = spec.worldbody.add_body(name="ball", pos=[ball[0], ball[1], B["radius"]])
        bb.add_freejoint(name="ball_free")
        g = bb.add_geom(name="ball", type=mj.mjtGeom.mjGEOM_SPHERE, size=[B["radius"], 0, 0], mass=B["mass"],
                        rgba=[1.0, 0.45, 0.0, 1.0])
        g.friction, g.condim, g.solref, g.priority = B["friction"], B["condim"], B["solref"], B["priority"]
        bh = "ball"
    return spec, hs, bh


class Lab:
    def __init__(self, robots, ball=(0.3, 0.0), seed=0, limits=None, **kw):
        import mujoco as mj
        self.mj = mj
        self.spec, self.handles, _ = scene(robots, ball, **kw)
        self.m = self.spec.compile()
        self.d = mj.MjData(self.m)
        mj.mj_forward(self.m, self.d)
        self.rigs = [SSL.rig(self.m, self.d, h, limits, seed=seed + i) for i, h in enumerate(self.handles)]
        self.ball = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "ball") if ball is not None else -1
        if self.ball >= 0:
            self.ball_q = self.m.jnt_qposadr[self.m.body_jntadr[self.ball]]
            self.ball_v = self.m.jnt_dofadr[self.m.body_jntadr[self.ball]]
        self.events = []

    def step(self):
        SSL.step_world(self.m, self.d, self.rigs)
        for r in self.rigs:
            self.events += r.take_events()

    def run(self, seconds, every=None):
        n = int(round(seconds / 0.01))
        for i in range(n):
            self.step()
            if every is not None:
                every(self, i)

    def ball_pos(self):
        return self.d.qpos[self.ball_q:self.ball_q + 3].copy()

    def ball_vel(self):
        return self.d.qvel[self.ball_v:self.ball_v + 3].copy()

    def set_ball(self, pos, vel=(0, 0, 0), spin=(0, 0, 0)):
        self.d.qpos[self.ball_q:self.ball_q + 3] = pos
        self.d.qpos[self.ball_q + 3:self.ball_q + 7] = (1, 0, 0, 0)
        self.d.qvel[self.ball_v:self.ball_v + 3] = vel
        self.d.qvel[self.ball_v + 3:self.ball_v + 6] = spin
        self.mj.mj_forward(self.m, self.d)

    def ball_rel(self, k=0):
        """the ball's centre in robot k's frame (x forward, y left)"""
        q = self.rigs[k].pose(self.d)
        b = self.ball_pos()
        c, s = np.cos(q[2]), np.sin(q[2])
        dx, dy = b[0] - q[0], b[1] - q[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, b[2]])
