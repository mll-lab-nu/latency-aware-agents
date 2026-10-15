"""fencing (epee) on the v0.2 framework: the four groups of contract ② (core/sport.py) around the unchanged legacy
simulator (fencing.py: world, motion controllers, phrase referee; bout.py: the pool-bout referee; opponent.py +
strategies.py: the scripted opponent), migrated from scripts/olympiad/{fencing,shared/fencing} (stage 3).

  world + referee  FencingMatch: step = one 10 ms control step of both fencers (bout.Bout.step); a halt (touch,
                   corps-a-corps, rear limit, time) freezes the world for the review (UNIT_OVER); resume() puts the
                   fencers on guard (bout.Bout.resume) and fences on. state_digest() = Bout.state_sha() (the legacy
                   hash: the legacy records replay bit-exactly). Format bout: seat A against the scripted
                   opponent (side B) inside the world; format duel: seats A and B.
  actions          low: joint space, executed by the robot's own motion controller (fencing.Controller) under the
                   real Panda's limits (bodies/franka_panda): joints {q (7 arm joints), t, rail_x, rail_vmax} (a
                   time-optimal jerk-limited move), move {x, vmax} (the cart's rail goal), hold (the arm stops), traj
                   {q: rows of 8 per-step joint targets} (100 Hz joint position targets: rail x in the own frame + 7
                   arm joints; within the velocity and acceleration limits, smoothed to the jerk limit by the servo;
                   the rail column within the rail's speed, acceleration and jerk limits, joined to its motion).
                   high: the legacy interface: act {t, tip, dir, rail} (the point pose in the cart frame; IK in the
                   seat's plan worker, lowered at the effect boundary to joints), move {rail}, guard, none.
  observation      observe(): score, clock, lamps, the last halt, the own robot; camera_job(): the 2 x 4 tile image of
                   cameras.py (each seat in its own frame, drawn in the seat's render worker).
  opponent         the single format's scripted fencer (opponent.Opponent, levels 1-3) runs inside step(), from the
                   world with its reaction latency, exactly as in the legacy loop (deterministic: the replay re-runs
                   it). Its commands are not episode rows: the core has no world-side command source (README).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

import bout as BT
import cameras as CAM
import fencing as F
from core.sport import FINISHED, PLAYING, UNIT_OVER, LowLayer, MatchBase, Phase, Refused, check_command

HERE = os.path.dirname(os.path.abspath(__file__))
SEATS = F.SEATS
DT = F.CONTROL_DT
T_MAX_S = 3.0
TRAJ_MAX_ROWS = 600                # 6 s of 100 Hz targets per traj command
LIMITS = dict(u=(0.0, 2.0), v=(-0.8, 0.8), z=(0.1, 2.0), x=(-F.RAIL_LIM, F.RAIL_LIM), vmax=(0.05, F.RAIL_VMAX))
LEVELS = (1, 2, 3)
REVIEW_PROMPT = ("Halt. Review the phrase: `game ready \"WHY\" \"PLAN\"` (why it ended this way; what you will change). "
                 "The world stays frozen until then.")
TIEBREAKS = ("touches", "passivity", "time", "deciding_touch", "priority")
PASSIVITY_NOTE = ("unwillingness to fight (FIE t.124): one minute of fencing time without a hit; the first time both "
                  "fencers get a P-red card (a penalty hit each), the second time the lower score gets the P-black card "
                  "and loses (equal scores: by lot); no card at 4-4; not in the extra minute")
_W = None                          # a kinematics-only world (joint ranges, the on-guard pose; never stepped)


def world0():
    global _W
    if _W is None:
        _W = F.World()
    return _W


def arm_range():
    w = world0()
    return w.lo["A"][1:].copy(), w.hi["A"][1:].copy()


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def own(s, p):
    sg = F.SIGN[s]
    return [round(float(sg * p[0]), 4), round(float(sg * p[1]), 4), round(float(p[2]), 4)]


# ------------------------------------------------------------------------------------------------ the match
class FencingMatch(MatchBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.single = len(cfg.side_list()) == 1
        self.dt = DT
        self.bout = BT.Bout(cfg.seed)
        self.w = self.bout.w
        self.ctl = {s: F.Controller(self.w, s) for s in SEATS}
        self.opp = None
        self.level = None
        self.strategies: list[str] = []
        if self.single:
            import opponent
            self.level = int(cfg.level or 1)
            if self.level not in LEVELS:
                raise ValueError(f"opponent level must be one of {list(LEVELS)}")
            self.opp = opponent.Opponent(self.level, cfg.seed, self.w, "B")
            self.ctl["B"] = self.opp.ctl
        self.traj = {s: [] for s in SEATS}            # per-step joint targets of a traj command in force
        self.shown: dict = {}                         # the high level's command in force (observation only)
        self.frames: list[dict] = []                  # camera frames of this phrase (cameras.py)
        self.step_ms: list[float] = []                # wall time per control step (final.json information only)
        self._fence()

    @property
    def k(self):
        return self.bout.k

    # ---------------------------------------------------------------- world + referee
    def _fence(self):
        """'Play!' (legacy service._resume): from on guard the bout runs; the controllers restart from the placed
        targets; the scripted opponent chooses this phrase's strategy."""
        self.bout.fence()
        for s in SEATS:
            self.ctl[s].reset()
            self.traj[s] = []
        if self.opp is not None:
            self.strategies.append(self.opp.new_phrase())
        self.commands = {s: {"arm": "on guard", "rail": None} for s in SEATS}
        self.t_phrase = 0.0
        self.frames = []
        self._frame()

    def _frame(self):
        self.frames.append(dict(k=self.bout.k, t=round(self.bout.k * DT, 2), qpos=self.w.d.qpos.copy(),
                                wall_mono=time.monotonic()))
        del self.frames[:-CAM.KEEP]

    def _target(self, s):
        if self.traj[s]:
            row = self.traj[s].pop(0)
            return self.ctl[s].row_target(row, self.traj[s][0] if self.traj[s] else None)
        return self.ctl[s].next_target()

    def step(self):
        t0 = time.monotonic()
        agents = ("A",) if self.single else SEATS
        targets = {s: self._target(s) for s in agents}
        if self.opp is not None:
            targets["B"] = self.opp.target(self.t_phrase)
        h = self.bout.step(targets)
        self.t_phrase += DT
        self.step_ms.append((time.monotonic() - t0) * 1000.0)
        if h is not None or self.bout.k % CAM.FRAME_STEPS == 0:
            self._frame()
        if h is None:
            return []
        if self.opp is not None and h["why"] != "passivity":     # P-red penalty hits are not its strategy's
            self.opp.result("B" in h["awarded"], "A" in h["awarded"])
        for s in SEATS:
            self.traj[s] = []
        return [{"kind": "halt", **h}]

    def resume(self):
        self.bout.resume()
        self._fence()

    def phase(self):
        b = self.bout
        if b.phase == "over":
            return Phase(FINISHED, reason=b.how or "bout_over")
        if b.phase == "halt":
            return Phase(UNIT_OVER, review=tuple(self.cfg.seats), reason=b.halts[-1]["why"])
        return Phase(PLAYING)

    def unit(self):
        return len(self.bout.halts)

    def scores(self):
        return dict(self.bout.score)

    def tiebreaks(self):
        """FIE t.38 and t.124 as bout.py decides it: 5 touches; the P-black card for unwillingness to fight; at time
        the fencer ahead; the deciding touch of the extra minute; the priority drawn by lot before it (the last rule
        always decides)."""
        rules = [(n, (lambda m, n=n: m.bout.winner if m.bout.how == n else None)) for n in TIEBREAKS[:-1]]
        return rules + [(TIEBREAKS[-1], lambda m: m.bout.winner)]

    def state_digest(self):
        return self.bout.state_sha()

    # ---------------------------------------------------------------- actions (low level)
    def schema(self, seat):
        return low_schema()

    def apply(self, seat, cmd):
        c, kind = self.ctl[seat], cmd["type"]
        shown = self.shown.pop(seat, None)
        info = {}
        if kind == "traj":
            rows = cmd["q"]
            first = np.asarray(rows[0], float)
            prev = np.r_[c.x_t, c.q_t]
            jump = np.abs(first - prev)
            if np.any(jump[1:] > F.ARM_VEL * DT * 1.001 + 1e-9):
                return {"rejected": "the first row is farther from the current targets than one step at the speed "
                                    "limits allows"}
            why = F.rail_rows_why(c.x_hist + [float(r[0]) for r in rows[:3]])
            if why:
                return {"rejected": f"the first rows do not join the rail's current motion within its limits: {why}"}
            why = F.AL.check_rows(np.asarray(rows, float)[:2, 1:], DT, F.ARM, prev=c.q_t, prev_vel=c.qd_t)
            if why:
                return {"rejected": f"the first rows do not join the current targets within the limits: {why}"}
            self.traj[seat] = [list(r) for r in rows]
            c.hold_arm()
            self.commands[seat] = {"arm": f"traj ({len(rows)} steps)", "rail": "traj"}
            return {"rows": len(rows), "arrive_s": round(len(rows) * DT, 3)}
        self.traj[seat] = []
        if kind == "hold":
            c.hold_arm()
            self.commands[seat]["arm"] = "holding"
        elif kind == "joints":
            info = c.install(cmd["q"], cmd["t"])
            self.commands[seat]["arm"] = shown or dict(q_goal=[round(float(x), 5) for x in cmd["q"]], t=cmd["t"])
            if isinstance(self.commands[seat]["arm"], dict):
                self.commands[seat]["arm"]["arrive_step"] = self.bout.k + int(round(info["arrive_s"] / DT))
            if cmd.get("rail_x") is not None:
                c.rail(cmd["rail_x"], cmd.get("rail_vmax", F.RAIL_VMAX))
                self.commands[seat]["rail"] = dict(x=cmd["rail_x"], vmax=cmd.get("rail_vmax", F.RAIL_VMAX))
        elif kind == "move":
            c.rail(cmd["x"], cmd.get("vmax", F.RAIL_VMAX))
            self.commands[seat]["rail"] = dict(x=cmd["x"], vmax=cmd.get("vmax", F.RAIL_VMAX))
        return info

    # ---------------------------------------------------------------- observation
    def _halt_view(self, h, s):
        o = BT.OTHER[s]
        v = dict(n=h["n"], why=h["why"], phrase_s=h["phrase_s"],
                 lamps=dict(you=s in h["touches"], opponent=o in h["touches"]),
                 awarded=dict(you=s in h["awarded"], opponent=o in h["awarded"]), double=h["double"],
                 annulled=h["annulled"], score=dict(you=h["score"][s], opponent=h["score"][o]),
                 fencing_time_s=round(h["k"] * DT, 2))
        if h.get("seat"):
            v["over_rear_limit"] = "you" if h["seat"] == s else "opponent"
        if h.get("priority"):
            v["priority"] = "you" if h["priority"] == s else "opponent"
            v["note"] = ("time is up at equal scores: one more minute for a deciding touch; if still equal, the fencer "
                         "with priority (drawn by lot) wins")
        if h["why"] == "passivity":
            c = h.get("cards") or {}
            v["no_hit_s"] = h.get("no_hit_s")
            v["cards"] = dict(you=c.get(s), opponent=c.get(o))
            v["note"] = PASSIVITY_NOTE
        if h.get("next_on_guard"):
            v["next_on_guard_x"] = dict(you=h["next_on_guard"][s], opponent=h["next_on_guard"][o])
        if h.get("winner"):
            v["winner"] = "you" if h["winner"] == s else "opponent"
            v["how"] = h["how"]
        return v

    def observe(self, seat, query=None):
        b, w, s = self.bout, self.w, seat
        o = BT.OTHER[s]
        st = w.state(s)
        guard = w.d.site_xpos[w.m.site(f"{s}_guard_s").id]
        c = self.ctl[s]
        moving = c.moving or bool(self.traj[s])
        out = dict(score=dict(you=b.score[s], opponent=b.score[o]), fencing_time_s=round(b.k * DT, 2),
                   time_left_s=b.time_left_s(), extra_minute=b.extra,
                   priority=None if b.priority is None else ("you" if b.priority == s else "opponent"),
                   lamps=dict(you=s in b.phrase.touches, opponent=o in b.phrase.touches) if b.phase == "fencing"
                   else None,
                   robot=dict(cart_x=round(st["x"], 4), cart_v=round(st["v"], 4), q=[round(float(x), 5) for x in st["q"]],
                              point=[round(float(x), 4) for x in st["tip"]],
                              blade_dir=[round(float(x), 5) for x in st["axis"]], guard=own(s, guard),
                              rail_goal=round(c.rail_goal, 4), rail_vmax=round(c.rail_vmax, 3), arm_moving=moving,
                              arm_arrive_in_s=round(max(c.remaining_s(), len(self.traj[s]) * DT), 2)),
                   command=_copy(self.commands.get(s)), halts=len(b.halts),
                   passivity=None if b.no_hit_s() is None else dict(no_hit_s=b.no_hit_s(), limit_s=BT.PASSIVE_S,
                                                                     p_cards=b.p_calls))
        if b.halts:
            out["last_halt"] = self._halt_view(b.halts[-1], s)
        if b.phase == "halt":
            out["review_prompt"] = REVIEW_PROMPT
        return out

    def camera_job(self, seat):
        sel = [self.frames[max(0, len(self.frames) - 1 - b // CAM.FRAME_STEPS)] for b in CAM.VIEW_BACK]
        job = {"seat": seat, "snaps": [dict(k=x["k"], t=x["t"], qpos=x["qpos"].copy()) for x in sel]}
        meta = dict(view_steps=[x["k"] for x in sel], view_sim_times=[x["t"] for x in sel],
                    image_layout=CAM.IMAGE_LAYOUT, wall_mono=sel[-1]["wall_mono"])
        return job, meta

    def render_state(self):
        return {"qpos": self.w.d.qpos.tolist()}

    def final_fields(self):
        """The sport's fields, and the arms' self-collision record (bodies/selfcollision.py Monitor: recorded only)."""
        return {**self._final_fields(), "self_collision": {s: mon.summary() for s, mon in self.bout.w.selfcol.items()}}

    def _final_fields(self):
        b = self.bout
        out = {"halts": b.halts, "extra_minute": b.extra, "priority": b.priority, "how": b.how,
               "fencing_time_s": round(b.k * DT, 2), "first_player": self.cfg.first, "p_cards": b.p_calls,
               "step_ms_max": round(max(self.step_ms, default=0.0), 2)}
        if self.single:
            out.update(level=self.level, opponent_strategies=self.strategies if b.phase == "over" else None)
        return out


def _copy(x):
    import copy
    return copy.deepcopy(x)


# ------------------------------------------------------------------------------------------------ action layers
def low_schema():
    lo, hi = arm_range()
    num = {"type": "number"}
    rail = {"rail_x": {**num, "min": -F.RAIL_LIM, "max": F.RAIL_LIM, "default": None, "unit": "m (own frame)",
                       "doc": "optional: a new rail goal for the cart"},
            "rail_vmax": {**num, "min": LIMITS["vmax"][0], "max": F.RAIL_VMAX, "default": F.RAIL_VMAX, "unit": "m/s"}}
    return {
        "joints": {"q": {"type": "vector", "len": 7, "min": float(lo.min()), "max": float(hi.max()), "unit": "rad",
                         "doc": f"the 7 arm joint goals; per joint ranges {[[round(a, 4), round(b, 4)] for a, b in zip(lo, hi)]}"},
                   "t": {**num, "min": 0.0, "max": T_MAX_S, "unit": "s",
                         "doc": "arrive T s after the effect, or as soon as the joint velocity, acceleration and jerk "
                                "limits allow (a straight joint-space move from the current joint targets, time-optimal "
                                f"at {F.PLAN_MARGIN:g} x the limits)"},
                   **rail},
        "move": {"x": {**num, "min": -F.RAIL_LIM, "max": F.RAIL_LIM, "unit": "m (own frame)"},
                 "vmax": {**num, "min": LIMITS["vmax"][0], "max": F.RAIL_VMAX, "default": F.RAIL_VMAX, "unit": "m/s"}},
        "hold": {},
        "traj": {"q": {"type": "rows", "len": 8, "max_rows": TRAJ_MAX_ROWS,
                       "doc": "per-step joint position targets from the effect step on (100 Hz): rows [x, q1..q7] "
                              "(x = the cart's rail position in your frame); within the joint ranges and the rail "
                              "limits, consecutive rows (and the first row from the current targets) within the speed "
                              f"limits (arm {[round(float(v), 3) for v in F.ARM_VEL]} rad/s) and the arm's "
                              "acceleration limits (second differences / (10 ms)^2 <= "
                              f"{[round(float(v), 2) for v in F.ARM_ACC]} rad/s^2; the robot smooths them to its jerk "
                              "limits); the rail column, joined to the rail's last three targets (its motion under "
                              f"way), within the rail's speed {F.RAIL_VMAX} m/s, acceleration {F.RAIL_AMAX} m/s^2 and "
                              f"jerk {F.RAIL_JMAX} m/s^3 (first, second and third differences / 10 ms, (10 ms)^2, "
                              "(10 ms)^3); the last row is held (the rail brakes to rest at it within its limits)"}},
    }


def validate_low(seat, body, match, ctx):
    if not isinstance(body, dict):
        raise Refused("the action must be a JSON object")
    if body.get("type") != "traj":
        sch = low_schema()
        sch.pop("traj")
        if body.get("type") not in sch:
            raise Refused(f"unknown action type {body.get('type')!r}; one of {sorted(low_schema())}")
        cmd = check_command(sch, body)
        if cmd["type"] == "joints":
            lo, hi = arm_range()
            q = np.asarray(cmd["q"], float)
            if np.any(q < lo) or np.any(q > hi):
                raise Refused(f"q out of the joint ranges {[[round(a, 4), round(b, 4)] for a, b in zip(lo, hi)]}")
            if cmd.get("rail_x") is None:
                cmd.pop("rail_x", None)
                cmd.pop("rail_vmax", None)
        return cmd
    extra = sorted(set(body) - {"type", "q"})
    if extra:
        raise Refused(f"unknown field(s) {extra} for traj")
    rows = body.get("q")
    if not isinstance(rows, list) or not 1 <= len(rows) <= TRAJ_MAX_ROWS:
        raise Refused(f"traj needs q: a list of 1..{TRAJ_MAX_ROWS} rows")
    if not all(isinstance(r, list) and len(r) == 8 and all(is_num(v) for v in r) for r in rows):
        raise Refused("each traj row is 8 finite numbers [x, q1..q7]")
    a = np.asarray(rows, float)
    lo, hi = arm_range()
    if np.any(np.abs(a[:, 0]) > F.RAIL_LIM) or np.any(a[:, 1:] < lo) or np.any(a[:, 1:] > hi):
        raise Refused("traj rows outside the rail limits or the joint ranges")
    d = np.abs(np.diff(a, axis=0))
    if len(a) > 1 and np.any(d[:, 1:] > F.ARM_VEL * DT * 1.001):
        raise Refused("consecutive traj rows farther apart than the speed limits allow in one 10 ms step")
    why = F.rail_rows_why(a[:, 0])
    if why:
        raise Refused(f"traj rows beyond the rail's limits: {why}")
    why = F.AL.check_rows(a[:, 1:], DT, F.ARM)
    if why:
        raise Refused(f"traj rows beyond the arm's limits: {why}")
    return {"type": "traj", "q": [[float(v) for v in r] for r in rows]}


def validate_high(body):
    """the legacy interface: {'type':'act','t':T,'tip':[u,v,z],'dir':[du,dv,dz],'rail':[x, vmax] (optional)} |
    {'type':'move','rail':[x] or [x, vmax]} | {'type':'guard'} | {'type':'none'}"""
    if not isinstance(body, dict):
        raise Refused("the action must be a JSON object")
    typ = body.get("type")
    allowed = {"act": {"type", "t", "tip", "dir", "rail"}, "move": {"type", "rail"}, "guard": {"type"},
               "none": {"type"}}
    if typ not in allowed:
        raise Refused("type must be act, move, guard or none")
    extra = sorted(set(body) - allowed[typ])
    if extra:
        raise Refused(f"unknown field(s) {extra} for {typ}")
    if typ in ("none", "guard"):
        return {"type": typ}

    def rail(r):
        if not isinstance(r, list) or not 1 <= len(r) <= 2 or not all(is_num(v) for v in r):
            raise Refused("rail must be [x] or [x, vmax] (finite numbers)")
        r = [float(v) for v in r]
        if not LIMITS["x"][0] <= r[0] <= LIMITS["x"][1]:
            raise Refused(f"rail x out of range: {LIMITS['x'][0]} <= x <= {LIMITS['x'][1]}")
        vmax = r[1] if len(r) == 2 else F.RAIL_VMAX
        if not LIMITS["vmax"][0] <= vmax <= LIMITS["vmax"][1]:
            raise Refused(f"rail speed out of range: {LIMITS['vmax'][0]} <= vmax <= {LIMITS['vmax'][1]} m/s")
        return [r[0], vmax]
    if typ == "move":
        return {"type": "move", "rail": rail(body.get("rail"))}
    T, tip, d = body.get("t"), body.get("tip"), body.get("dir")
    if not is_num(T) or not isinstance(tip, list) or not isinstance(d, list) or len(tip) != 3 or len(d) != 3 \
            or not all(is_num(v) for v in tip + d):
        raise Refused("act needs t (s), tip [u, v, z] (the point in your cart frame) and dir [du, dv, dz] (the blade "
                      "direction), finite numbers, optionally rail [x, vmax]")
    if not 0.0 <= T <= T_MAX_S:
        raise Refused(f"t must be in [0, {T_MAX_S}] s")
    for name, val in zip("uvz", tip):
        lo, hi = LIMITS[name]
        if not lo <= val <= hi:
            raise Refused(f"tip {name} out of range: {lo} <= {name} <= {hi}")
    if np.linalg.norm(d) < 1e-6:
        raise Refused("dir must be a non-zero vector")
    out = {"type": "act", "t": float(T), "tip": [float(v) for v in tip],
           "dir": (np.asarray(d, float) / np.linalg.norm(d)).tolist()}
    if body.get("rail") is not None:
        out["rail"] = rail(body["rail"])
    return out


HIGH_SCHEMA = {
    "act": {"t": {"type": "number", "min": 0.0, "max": T_MAX_S, "unit": "s"},
            "tip": {"type": "vector", "len": 3, "doc": "the point (u, v, z) in your cart frame: 0 <= u <= 2, |v| <= 0.8, "
                                                      "0.1 <= z <= 2 m"},
            "dir": {"type": "vector", "len": 3, "doc": "the blade direction (du, dv, dz), normalised"},
            "rail": {"type": "vector", "len": "1 or 2", "optional": True,
                     "doc": f"[x] or [x, vmax]: a new rail goal (your frame, |x| <= {F.RAIL_LIM}) and speed cap "
                            f"(0.05 <= vmax <= {F.RAIL_VMAX} m/s)"}},
    "move": {"rail": {"type": "vector", "len": "1 or 2", "doc": "[x] or [x, vmax]: the rail goal only"}},
    "guard": {}, "none": {},
}


class JointLayer(LowLayer):
    """Low level: joint-space commands (joints / move / hold / traj) straight to the robot's controller."""

    def __init__(self):
        super().__init__(validate_low)


class PointLayer:
    """High level (the legacy interface): the point pose in the cart frame. IK off the world lock in the seat's plan
    worker (from the arm targets when the request arrives, as the legacy service), lowered at the effect boundary to a
    low-level joints command; move -> move, guard -> joints to the on-guard pose, none -> hold."""
    level = "high"
    worker = "plan"
    delay_steps = 0

    def schema(self, match, seat):
        return HIGH_SCHEMA

    def parse(self, seat, body, match, ctx):
        return validate_high(body)

    def snapshot(self, seat, cmd, match, ordinal):
        if cmd["type"] != "act":
            return None
        return dict(tip=cmd["tip"], dir=cmd["dir"], q_now=match.ctl[seat].q_t.tolist())

    def prepare(self, job):
        return plan(world0(), job)

    def lower(self, seat, cmd, prepared, match):
        kind = cmd["type"]
        if kind == "none":
            return {"type": "hold"}, {}
        if kind == "move":
            return {"type": "move", "x": cmd["rail"][0], "vmax": cmd["rail"][1]}, {}
        if kind == "guard":
            match.shown[seat] = "on guard"
            return {"type": "joints", "q": match.w._guard_arm().tolist(), "t": 0.0}, {}
        low = {"type": "joints", "q": list(prepared["q_goal"]), "t": cmd["t"]}
        if cmd.get("rail"):
            low.update(rail_x=cmd["rail"][0], rail_vmax=cmd["rail"][1])
        match.shown[seat] = dict(tip=cmd["tip"], dir=[round(x, 5) for x in cmd["dir"]], t=cmd["t"])
        info = {"ik_residual": round(float(prepared["ik_residual"]), 4)}
        if prepared["ik_residual"] > F.IK_OK:
            info["note"] = "pose not reachable exactly: the arm goes to the nearest pose it can reach"
        return low, info

    def reply(self, seat, cmd, info, match):
        return info


def plan(w, job):
    """IK for an arm command (legacy workers.plan): the point at job['tip'] (cart frame), blade along job['dir']"""
    q, res = w.ik_multi(np.asarray(job["tip"], float), np.asarray(job["dir"], float), np.asarray(job["q_now"], float))
    return dict(q_goal=q.tolist(), ik_residual=float(res))


# ------------------------------------------------------------------------------------------------ the sport
_RENDER: dict = {}


def kinematics():
    """The arm's kinematic chain (own frame; the same for both seats) for low-level players: each body from the cart
    to the epee with its pose in the parent frame (pos, quat w x y z) and its hinge (axis, range); the point."""
    import mujoco as mj
    m = world0().m
    chain, b = [], mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "A_epee")
    while m.body(b).name != "A_cart":
        j = [i for i in range(m.njnt) if m.jnt_bodyid[i] == b]
        e = dict(body=m.body(b).name[2:], pos=[round(float(x), 7) for x in m.body_pos[b]],
                 quat=[round(float(x), 7) for x in m.body_quat[b]])
        if j:
            e.update(joint=m.joint(j[0]).name[2:], axis=[float(x) for x in m.jnt_axis[j[0]]],
                     range=[round(float(x), 4) for x in m.jnt_range[j[0]]])
        chain.append(e)
        b = m.body_parentid[b]
    return dict(frame="the cart frame: origin on the piste under the cart's centre, x forward along the piste, y left, "
                      "z up; the cart moves along x (rail)",
                chain=list(reversed(chain)), point_in_epee_frame=[0.0, 0.0, 1.04],
                blade_axis="the epee body's z axis",
                fk="T = T_parent * translate(pos) * rotate(quat) * rotate(axis, q) for each body in order")


class Fencing:
    name = "fencing"
    protocol = "realtime"
    revision = "fencing-6+core-0.2"
    source_dirs = [HERE]
    workers = ("plan", "render")

    def build(self, cfg):
        return FencingMatch(cfg)

    def action_layer(self, level):
        return PointLayer() if level == "high" else JointLayer()

    def worker_init(self, kind, seat, cfg):
        if kind == "plan":
            return F.World()
        r = CAM.Renderer(seat)
        r.render([dict(k=0, t=0.0, qpos=r.d.qpos.copy())] * CAM.N_FRAMES)      # the GL context before 'ready'
        return r

    def worker_call(self, obj, kind, job):
        if kind == "plan":
            return plan(obj if obj is not None else world0(), job)
        if obj is None:                               # in-process (core/local.py): one renderer per seat
            obj = _RENDER.get(job["seat"]) or _RENDER.setdefault(job["seat"], CAM.Renderer(job["seat"]))
        return {"png": obj.render(job["snaps"])}

    def scene(self, cfg):
        from pathlib import Path
        from renderers.base import Scene
        d = Path(HERE) / "scene"
        xml = (d / "fencing_scene.xml").read_text().replace(' meshdir="assets"', "")
        return Scene(xml=xml, assets={p.name: p.read_bytes() for p in (d / "assets").iterdir()
                                      if p.suffix in (".stl", ".obj")})

    def spec(self, cfg, seat, match):
        single = len(cfg.side_list()) == 1
        out = dict(cameras=CAM.cam_info(), image=CAM.IMAGE_LAYOUT,
                   frame=("your own frame: piste surface z = 0, x along the piste towards the opponent, centre line "
                          "x = 0, your on-guard line x = -2, your rear limit x = -7; y to your left; your cart on the "
                          "rail at y = 0"),
                   bout=dict(touches=BT.BOUT_TOUCHES, time_s=BT.BOUT_S, extra_minute_s=BT.EXTRA_S,
                             passivity_s=BT.PASSIVE_S),
                   robot=dict(arm="a Franka Emika Panda (7 joints) held to the manufacturer's joint limits (velocity, "
                                  "acceleration, jerk, torque, torque rate; the servo never exceeds them)",
                              arm_limits=F.ARM.table(),
                              pedestal_m=F.PED_H, shoulder_height_m=round(F.PED_H + 0.333, 3),
                              rail=dict(vmax=F.RAIL_VMAX, amax=F.RAIL_AMAX, jmax=F.RAIL_JMAX, force_n=F.RAIL_FMAX,
                                        limits=[-F.RAIL_LIM, F.RAIL_LIM], product=F.RAIL_PRODUCT),
                              on_guard=dict(tip=F.GUARD_TIP.tolist(), dir=[round(float(x), 4) for x in F.GUARD_DIR],
                                            q=[round(float(x), 6) for x in world0()._guard_arm()])),
                   apparatus=dict(force_n=F.F_TOUCH, min_contact_ms=2, double_window_ms=F.DOUBLE_MS),
                   act_rule=("an act takes effect at the first control step after its arm plan (IK) is ready (the "
                             "reply gives the step); every other command at the next step; a new arm command replaces "
                             "the one under way, the cart keeps going to its rail goal"
                             if cfg.action_level == "high" else
                             "a command takes effect at the next control step; a new arm command (joints, hold, traj) "
                             "replaces the one under way; joints and hold keep the rail goal, traj drives the rail too"))
        if cfg.action_level == "low":
            out["kinematics"] = kinematics()
        if single:
            out["opponent_level"] = int(cfg.level or 1)
        else:
            out["first_player"] = cfg.first or "A"
        return out

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = Fencing()
