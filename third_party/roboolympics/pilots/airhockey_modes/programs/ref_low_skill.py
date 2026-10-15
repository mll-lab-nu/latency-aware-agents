"""Our LOW-level reference implementation of the high-level skill API (deliverable form: /app/output/skill/skill.py,
class Impl), built only from the low-level toolkit: the robot model and the kinematics library (/opt/robot/lib/ahkin.py)
and plans of joint targets. The planning is the platform's own (planner.py, copied next to this file): IK along the
drive line, the wind-up / settle / timed drive / follow-through, all within the joint velocity and acceleration limits.

    impl = Impl(spec)
    impl.move(view, x, y)  -> commands     impl.strike(view, req) -> commands     impl.step(view) -> commands
    view = {"state": the observation, "replies": the results of the previous commands}; commands: plan / hold"""
from __future__ import annotations

import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):    # the deliverable's dir / the pilot
    if (_p / "planner.py").is_file():
        sys.path.insert(0, str(_p))
        break
import planner  # noqa: E402


class Impl:
    def __init__(self, spec):
        self.pl = planner.Planner()

    def move(self, view, x, y):
        st = view["state"]
        rows = self.pl.move(st["q_target"], st["q_target_v"], (x, y))
        return [{"type": "plan", "q": rows[:planner.MAX_ROWS].round(7).tolist(), "k0": st["k"]}]

    def strike(self, view, req):
        st = view["state"]
        plan = self.pl.strike(st["q_target"], st["q_target_v"], st["t"], req)
        if "rows" not in plan:                       # no plan (e.g. the contact point outside the workspace):
            c = plan.get("contact_point") or [req["x"], req["y"]]        # get as close as possible
            return self.move(view, c[0], c[1])
        return [{"type": "plan", "q": plan["rows"].round(7).tolist(), "k0": st["k"]}]

    def step(self, view):
        return []
