"""The air-hockey two-mode pilot's world: sports/airhockey/hockey.py (revision v6: the real iiwa 14 limits, the mallet
within 2 cm of the table) with seat A driven through one of the two modes' interfaces. Seat B is the scripted opponent
(single player) or a parked arm (the post-match tests).

  HIGH (Agent as Athlete)        the platform's skills, executed here by the trusted simulator:
    {"type": "move", "x", "y"}    the mallet to (x, y) along the platform's path (the official sport's target: at most
                                  2 m/s and 10 m/s^2, retimed to the joint limits, IK, motors) and held there
    {"type": "strike", "t", "x", "y", "dx", "dy", "speed"}
                                  at game time t the puck centre is at (x, y) (your prediction); hit it with the mallet
                                  moving along the unit direction (dx, dy) at `speed` m/s (0.3-2.5). The platform plans
                                  the wind-up, the settle, the timed drive and the follow-through (planner.py) and
                                  replies with the plan (contact_t, speed, late_s, feasible, why)
    {"type": "none"}              hold where the mallet is
  LOW (Agent as Robotics Engineer)  joint plans through the robot's servo:
    {"type": "plan", "q": rows[, "k0": k]}   rows of the 9 joint targets (7 iiwa axes + 2 striker joints), row 0 =
                                  the target at the end of the next 10 ms control step (with k0: of step k0 + 1; rows
                                  already past when the plan arrives are dropped), at most 300 rows; between rows the target is
                                  interpolated linearly at 1 kHz and followed by the robot's setpoint generator
                                  (velocity / acceleration / jerk limits) and PD with inverse dynamics, torques within
                                  the motors'. Refused (with the reason) if a row leaves the joint ranges, the velocity
                                  or acceleration limits (counted from the current target and its velocity), the
                                  mallet's workspace (1 cm tolerance) or the table (the mallet centre within 2 cm of
                                  its height on the table), for the rows and (v6) for the setpoints the robot's
                                  setpoint generator would make of them (a row that asks a joint for more acceleration
                                  or jerk than it has gets limited joint by joint, which bends the path off the
                                  table). The plan replaces the one under way; after the last row the target holds
    {"type": "hold"}              stop: hold the current target

Observation (pilot: TRUE state; the official sport uses the camera): t, k, rally, score, puck, puck_v, mallet,
mallet_v, opp_mallet, opp_mallet_v, q, qd (9 joints), q_target, q_target_v (the servo's current target and its
velocity: a plan's first row must continue from them), plan_rows_left, control, puck_half / puck_half_s (the
seven-second clock), play_time_left_s."""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "sports" / "airhockey", HERE, HERE / "toolkit" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import hockey as H  # noqa: E402
import ahkin  # noqa: E402
import planner as PL  # noqa: E402

DT = H.DT
MAX_ROWS = 300
Z_TOL = 0.02
WS_TOL = 0.01
HOLD_CHECK = 30                # control steps of the hold after a plan's last row that check_setpoints also follows
PARK_B = (0.88, 0.40)          # seat B parked away from its goal mouth (the post-match tests' empty goal)


class ModesHockey(H.Hockey):
    def __init__(self, *a, **k):
        self.ctl = {"A": "path", "B": "path"}
        self.rows = {"A": deque(), "B": deque()}
        self.tv = {"A": np.zeros(9), "B": np.zeros(9)}
        self.prev_pos = {}
        self.vel = {"A": np.zeros(2), "B": np.zeros(2)}
        super().__init__(*a, **k)
        self.mallet_bid = {s: self.mallet_body[s] for s in "AB"}

    def restart(self, holder, why):
        super().restart(holder, why)
        for s in "AB":
            self.ctl[s] = "path"
            self.rows[s].clear()
            self.tv[s] = np.zeros(9)
            self.vel[s] = np.zeros(2)
            self.prev_pos[s] = self.position(s)[:2].copy()

    # ------------------------------------------------------------------ the two controllers of a seat
    def reference(self, s):
        if self.ctl[s] == "rows":
            q = np.asarray(self.rows[s].popleft(), float) if self.rows[s] else self.qtarget[s].copy()
        else:
            q = super().reference(s)
        self.tv[s] = (q - self.qtarget[s]) / DT
        return q

    def mallet_of(self, s, q):
        """the mallet body's position for joint targets q of seat s (forward kinematics on the IK data)"""
        self.ik.qpos[:] = self.data.qpos
        self.ik.qpos[self.qadr[s]] = q
        mujoco.mj_kinematics(self.model, self.ik)
        return self.ik.xpos[self.mallet_bid[s]].copy()

    def to_path(self, s, xy):
        """switch seat s to the platform's path towards xy (the path starts where the current target is)"""
        if self.ctl[s] != "path":
            m = self.mallet_of(s, self.qtarget[s])
            m0 = self.mallet_of(s, self.qtarget[s] - self.tv[s] * DT)
            self.path_reset(s, m[:2], (m[:2] - m0[:2]) / DT)    # the path starts where and as the rows left it
            self.ctl[s] = "path"
            self.rows[s].clear()
        self.set_command(s, list(xy))

    def to_rows(self, s, rows):
        self.ctl[s] = "rows"
        self.rows[s] = deque(np.asarray(r, float) for r in rows)

    def step(self):
        c = super().step()
        for s in "AB":
            p = self.position(s)[:2]
            self.vel[s] = (p - self.prev_pos.get(s, p)) / DT
            self.prev_pos[s] = p.copy()
        return c

    # ------------------------------------------------------------------ low-level plan check
    def check_plan(self, s, rows):
        try:
            Q = np.asarray(rows, float)
        except (TypeError, ValueError):
            return "q must be a list of rows of 9 numbers"
        if Q.ndim != 2 or Q.shape[1] != 9 or not 1 <= len(Q) <= MAX_ROWS:
            return f"q must be 1-{MAX_ROWS} rows of 9 joint targets (7 iiwa axes + 2 striker joints)"
        if not np.all(np.isfinite(Q)):
            return "non-finite joint target"
        bad = np.nonzero((Q < ahkin.RANGE_LO - 1e-6) | (Q > ahkin.RANGE_HI + 1e-6))
        if len(bad[0]):
            return f"row {int(bad[0][0])}: joint {int(bad[1][0]) + 1} outside its range"
        a = np.vstack([self.qtarget[s][None], Q])
        v = np.diff(a, axis=0) / DT
        over = np.nonzero(np.abs(v) > ahkin.VEL * 1.001)
        if len(over[0]):
            i, j = int(over[0][0]), int(over[1][0])
            return (f"row {i}: joint {j + 1} velocity {v[i, j]:.3f} rad/s beyond its limit {ahkin.VEL[j]:.4f}"
                    + (" (the first row must continue from q_target)" if i == 0 else ""))
        acc = np.diff(np.vstack([self.tv[s][None], v]), axis=0) / DT
        over = np.nonzero(np.abs(acc) > ahkin.ACC * 1.001)
        if len(over[0]):
            i, j = int(over[0][0]), int(over[1][0])
            return (f"row {i}: joint {j + 1} acceleration {acc[i, j]:.2f} rad/s^2 beyond its limit {ahkin.ACC[j]:.4f}"
                    + (" (counted from q_target_v)" if i == 0 else ""))
        for i, q in enumerate(Q):
            m = self.mallet_of(s, q)
            if abs(m[2] - ahkin.Z) > Z_TOL:
                return f"row {i}: the mallet leaves the table (centre height {m[2]:.3f} m, nominal {ahkin.Z})"
            if not H.in_workspace(s, m[:2], WS_TOL):
                return f"row {i}: the mallet centre ({m[0]:.3f}, {m[1]:.3f}) leaves the workspace"
        return self.check_setpoints(s, Q)

    def check_setpoints(self, s, Q):
        """the plan as the robot would follow it (v6): the 1 kHz setpoint generator (velocity / acceleration / jerk
        limits, its present state) run over the rows; refused if its setpoint would take the mallet off the table
        (the AHC end-effector height: within Z_TOL of its height), which per-joint limiting does to a plan that asks
        a joint for more than its limits allow"""
        jl = H.AL.JerkLimiter(H.ARM, H.PHYSICS_DT)
        src = self.jl[s]
        jl.reset(src.q, src.qd, src.qdd)
        prev = self.qtarget[s]
        for i, q in enumerate(list(Q) + [Q[-1]] * HOLD_CHECK):    # and the hold after the last row
            i = min(i, len(Q) - 1)
            rd = np.clip((q[:7] - prev[:7]) / DT, -H.ARM.vel, H.ARM.vel)
            for sub in range(H.SUBSTEPS):
                qs = jl.step(prev[:7] + (q[:7] - prev[:7]) * (sub + 1) / H.SUBSTEPS, rd)[0]
            z = self.mallet_of(s, np.r_[qs, q[7:]])[2]
            if abs(z - ahkin.Z) > Z_TOL:
                return (f"row {i}: the robot's setpoint generator would take the mallet off the table (centre height "
                        f"{z:.3f} m, nominal {ahkin.Z}): the plan asks a joint for more acceleration or jerk than it has")
            prev = q
        return None

    # ------------------------------------------------------------------ the observation
    def observe(self, s="A"):
        o = H.OTHER[s]
        p, w = self.puck(), self.puck_velocity()
        half = self.side
        return {"t": round(self.k * DT, 4), "k": self.k, "rally": len(self.results) + (0 if self.awaiting_review or self.done else 1),
                "score": [self.scores()[s]["points"], self.scores()[o]["points"]],
                "puck": [round(float(x), 5) for x in p], "puck_v": [round(float(x), 5) for x in w],
                "mallet": [round(float(x), 5) for x in self.position(s)[:2]],
                "mallet_v": [round(float(x), 5) for x in self.vel[s]],
                "opp_mallet": [round(float(x), 5) for x in self.position(o)[:2]],
                "opp_mallet_v": [round(float(x), 5) for x in self.vel[o]],
                "q": [round(float(x), 6) for x in self.data.qpos[self.qadr[s]]],
                "qd": [round(float(x), 6) for x in self.data.qvel[self.dadr[s]]],
                "q_target": [float(x) for x in self.qtarget[s]], "q_target_v": [float(x) for x in self.tv[s]],
                "plan_rows_left": len(self.rows[s]), "control": self.ctl[s],
                "puck_half": None if half is None else ("own" if half == s else "opponent"),
                "puck_half_s": round(self.side_steps * DT, 3) if half else 0.0,
                "play_time_left_s": round(max(0.0, H.CAP_S - self.k * DT), 3)}

    def match_view(self, s="A"):
        """the observation as the match service returns it (observe + the service's state and the rallies): the view
        the post-match tests give a deliverable, so that it sees what it saw in the match"""
        o = self.observe(s)
        o["state"] = "over" if self.done else ("review" if self.awaiting_review else "running")
        o["rallies"] = [{k: r.get(k) for k in ("rally", "how", "winner", "guilty", "own_goal")} for r in self.results]
        return o

    def park(self, s, xy):
        """put seat s's arm at rest with its mallet at xy (instantly; the post-match tests' parked opponent)"""
        q = self.solve(s, self.data.qpos[self.qadr[s]].copy(), np.array(xy, float), 120)
        self.data.qpos[self.qadr[s]] = q
        self.data.qvel[self.dadr[s]] = 0
        self.qtarget[s] = q.copy()
        self.path_reset(s, xy)
        self.commands[s] = list(xy)
        self.jl[s].reset(q[:7])
        self.ctl[s] = "path"
        self.rows[s].clear()
        self.tv[s] = np.zeros(9)
        mujoco.mj_forward(self.model, self.data)
        self.prev_pos[s] = self.position(s)[:2].copy()


class Seat:
    """One seat's command interface for a mode ('high' | 'low') on a ModesHockey. command() returns the reply (a
    dict) or the reason the command was refused (a str)."""

    def __init__(self, h: ModesHockey, level: str, seat="A", planner: PL.Planner | None = None):
        self.h, self.level, self.s = h, level, seat
        self.pl = planner if planner is not None else (PL.Planner() if level == "high" else None)
        self.refusals = 0
        self.reasons = {}
        self.n_plans = self.n_strikes = self.n_moves = 0

    def command(self, c):
        try:
            r = self._command(c)
        except (KeyError, TypeError, ValueError) as e:
            r = f"malformed command: {e}"
        if isinstance(r, str):
            self.refusals += 1
            key = r.split(":")[0] + ": " + (r.split(": ")[2].split(" ")[-1] if r.count(": ") >= 2 else r)[:60]
            if len(self.reasons) < 50 or key in self.reasons:
                self.reasons[key] = self.reasons.get(key, 0) + 1
        return r

    def _command(self, c):
        h, s = self.h, self.s
        if h.done or h.awaiting_review:
            return "the world is frozen"
        t = c.get("type") if isinstance(c, dict) else None
        if self.level == "high":
            if t == "none":
                h.to_path(s, H.workspace(s, h.position(s)[:2]))
                return {"ok": True}
            if t == "move":
                xy = [float(c["x"]), float(c["y"])]
                if not all(map(math.isfinite, xy)):
                    return "x, y must be finite"
                if not H.in_workspace(s, xy):
                    return "the target lies outside the mallet workspace"
                h.to_path(s, xy)
                self.n_moves += 1
                return {"ok": True}
            if t == "strike":
                req = {k: float(c[k]) for k in ("t", "x", "y", "dx", "dy", "speed")}
                if not all(map(math.isfinite, req.values())):
                    return "all fields must be finite numbers"
                if math.hypot(req["dx"], req["dy"]) < 1e-6:
                    return "(dx, dy) must be a direction"
                now = h.k * DT
                if not now + DT <= req["t"] <= now + 3.0:
                    return f"t must lie between the next control step and 3 s ahead (now {now:.2f})"
                if not PL.SPEED_MIN <= req["speed"] <= PL.SPEED_MAX:
                    return f"speed must lie within {PL.SPEED_MIN}-{PL.SPEED_MAX} m/s"
                plan = self.pl.strike(h.qtarget[s], h.tv[s], now, req)
                if "rows" not in plan:
                    return "strike refused: " + plan["why"]
                bad = h.check_plan(s, plan["rows"])
                if bad:
                    return "strike refused (the plan would break a limit): " + bad
                h.to_rows(s, plan["rows"])
                self.n_strikes += 1
                return {k: v for k, v in plan.items() if k != "rows"}
            return "unknown command (high level: move | strike | none)"
        if t == "hold":
            h.to_rows(s, [])
            return {"ok": True}
        if t == "plan":
            rows = c["q"]
            skip = 0
            if c.get("k0") is not None:                  # rows timed from step k0: drop the ones already past
                skip = max(0, h.k - int(c["k0"]))
                if not isinstance(rows, list) or skip >= len(rows):
                    return "plan refused: all its rows are already past (k0)"
                rows = rows[skip:]
            bad = h.check_plan(s, rows)
            if bad:
                return "plan refused: " + bad
            h.to_rows(s, rows)
            self.n_plans += 1
            return {"ok": True, "rows": len(rows), "skipped": skip, "first_row_at_k": h.k + 1}
        return "unknown command (low level: plan | hold)"
