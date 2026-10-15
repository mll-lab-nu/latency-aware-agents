"""The core-skill BOUNDARY baseline (framework rule 5): the simplest implementation of the skill API from the tools
alone, without developing the skill. move: IK to the point, a smooth joint move there. strike: IK to the contact point
(the puck's centre minus the contact distance along the drive direction) and the same smooth move, arriving at rest:
no wind-up, no timing, no speed. Its API-test success should be about 0."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

for p in (Path(__file__).resolve().parent, Path("/opt/robot/lib")):
    sys.path.insert(0, str(p))
import ahkin  # noqa: E402

CONTACT = .03165 + .04815


class Impl:
    def __init__(self, spec):
        self.kin = ahkin.Kin()

    def _goto(self, st, xy):
        q0 = np.asarray(st["q_target"], float)
        qd0 = np.asarray(st["q_target_v"], float)
        q1, _ = self.kin.ik(xy, q0, 80)
        d = np.abs(q1 - q0)
        T = max(.1, float(np.max(2.0 * d / (0.6 * ahkin.VEL))), float(np.max(np.sqrt(8.0 * d / (0.5 * ahkin.ACC)))))
        T += float(np.max(np.abs(qd0) / (0.5 * ahkin.ACC)))
        n = max(1, int(math.ceil(T / 0.01)))
        s = (np.arange(1, n + 1) / n)[:, None]
        h00, h10, h01 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2
        rows = h00 * q0 + h10 * (n * 0.01) * qd0 + h01 * q1
        return [{"type": "plan", "q": rows.round(7).tolist(), "k0": st["k"]}]

    def move(self, view, x, y):
        return self._goto(view["state"], (x, y))

    def strike(self, view, req):
        d = np.array([req["dx"], req["dy"]], float)
        d /= max(1e-9, float(np.linalg.norm(d)))
        c = np.array([req["x"], req["y"]], float) - CONTACT * d
        return self._goto(view["state"], c)

    def step(self, view):
        return []
