"""tennis: robot singles tennis in real time (docs/framework_zh.md §4.2). Migrated from the Season-1 tasks
tennis-match-wallclock-program-standard and tennis-duel-wallclock-program-standard (docs §10 stage 3). The four groups
of contract ②, on the unchanged legacy simulator (sim/tn.py, sim/duel.py, sim/scene/):
  world + referee  format match = tn.Match (seat A against the scripted opponent, levels 1-4, which plays
                   inside the world), format duel = duel.Duel (seats A and B, mirrored robots, a neutral
                   serving machine); step = one 10 ms control step (rt_advance(1)); a point end freezes for the reviews
  actions          low = joint targets: `traj` (a list of 9 joint targets, one per control step, followed exactly,
                   then the arm returns to its ready pose), `move` (the base's rest position) and `none`; high = the
                   legacy racket command `act` (the racket state at contact T s after the act takes effect), planned
                   in the seat's plan worker (tn.PlanCtx, from a snapshot of the seat's robot at the step the act is
                   accepted) and lowered to the SAME `traj` command with its effect step k0 = accepted step + lead
                   (the lead of the legacy service: ceil(0.5 s / wall time per step)); the episode holds only `traj`,
                   `move` and `none`
  referee          phase / scores / tiebreaks (the legacy match_winner chain, split into its three rules)
  observation      observe = the legacy state (score, what was heard, the robot's own body, its command);
                   camera_job = the legacy 2 cameras x 4 frames (40 ms apart), drawn by the sport itself (sim/scene
                   with the legacy extra geometry) in the seat's render worker
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
if SIM not in sys.path:
    sys.path.insert(0, SIM)
os.environ.pop("MATCH_SEED", None)          # tn.Match reads it at construction; the seed comes from the config only

import duel as DU  # noqa: E402
import tn as TN  # noqa: E402

from core.sport import FINISHED, PLAYING, UNIT_OVER, MatchBase, Phase, Refused  # noqa: E402

REVISION = "tennis-wallclock-9+duel-7+core-0.2"
SEATS2 = ("A", "B")
OPPONENT = "opponent"                       # the scripted opponent's side in format match (no seat)
DT = TN.CONTROL_DT
PLAN_LEAD_S = 0.5                           # wall-time budget of an act's stroke planning (legacy WALLCLOCK_PLAN_LEAD_S)
T_MAX_S = 5.0
TRAJ_MAX = 600                              # joint targets in one low-level traj command
STROKE_KEYS = ("stroke", "base", "base_goal", "k_hit", "n_swing_steps_available", "n_swing_steps", "ik_residual",
               "blade_error_m", "normal_error_deg", "speed_scale", "racket_vel_delivered", "T_from_now")
VEL_LIM = TN.VEL_LIMIT * DT * (1 + 1e-6) + 1e-12
IMAGE_LAYOUT = ("2 rows x 4 columns of 640 x 480 tiles. Row 1: camera behind and above your base line looking down the "
                "court; row 2: camera at the side of the court (x < 0 in your frame) looking along +x: your half (y < 0) "
                "is on the right. Columns: four camera frames 40 ms apart, oldest left; each tile is stamped with its "
                "control step k and its time t since the ball shown was played (view_steps, view_times). The ball is "
                "drawn enlarged: a yellow-green sphere of radius 9 cm centred on its true position.")
_TOKENS = itertools.count(1)


def lead_steps(slowdown) -> int:
    """Control steps from an act's acceptance to its effect (the legacy services' LEAD)."""
    period = DT * float(slowdown or 20)
    return max(1, int(math.ceil(PLAN_LEAD_S / period - 1e-9)))


def _clean(x):
    """JSON-exact python values (what the episode log holds and the replay reads back)."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


class Tennis(MatchBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.single = len(cfg.seats) == 1
        self.dt = DT
        self.token = next(_TOKENS)              # the render workers' tile cache key (one match)
        self.review_due = False
        self.deferred = {s: None for s in cfg.seats}     # a lowered act waiting for its effect step k0
        self.rejected = {s: 0 for s in cfg.seats}
        self.level_ = None
        if self.single:
            self.level_ = 1 if cfg.level is None else int(cfg.level)
            if self.level_ not in TN.LEVELS:
                raise ValueError(f"opponent level must be one of {sorted(TN.LEVELS)}")
            self.g = TN.Match(self.level_, lv_override={"seed": int(cfg.seed)})
        else:
            self.first = cfg.first if cfg.first in SEATS2 else "A"     # the first server (ITF Rule 5 b order)
            self._ctx()
            self.g = DU.Duel()

    def _ctx(self):
        """duel.py keeps the seed and the first server in module globals: set them for this match."""
        if not self.single:
            DU.SEED, DU.FIRST_SERVER = int(self.cfg.seed), self.first

    @property
    def k(self):
        return self.g.k

    # ------------------------------------------------------------------ robot access (A/B or the single robot)
    def target(self, seat):
        return self.g.s.target if self.single else self.g.target[seat]

    def points(self):
        return self.g.points

    def plan_active(self, seat):
        if self.single:
            return self.g.plan_j < len(self.g.plan)
        sd = self.g.sides[seat]
        return sd.plan_j < len(sd.plan)

    # ------------------------------------------------------------------ world + referee
    def step(self):
        self._ctx()
        for s, item in self.deferred.items():
            if item is not None and item["k0"] == self.g.k:
                self.deferred[s] = None
                if "rejected" in self._install(s, item):
                    self.rejected[s] += 1
        n0 = len(self.g.points)
        self.g.rt_advance(1)
        if len(self.g.points) > n0:
            self.deferred = {s: None for s in self.cfg.seats}       # no command carries over to the next point
            if not self.g.done:
                self.review_due = True
            return [{"kind": "point", **self.g.points[-1]}]
        return []

    def state_digest(self):
        return self.g.s.state_sha() if self.single else self.g.state_sha()

    def phase(self):
        if self.g.done:
            return Phase(FINISHED, reason="match_limit" if self.g.limit_reached else "match_over")
        if self.review_due:
            return Phase(UNIT_OVER, review=tuple(self.cfg.seats), reason=self.g.points[-1]["how"])
        return Phase(PLAYING)

    def resume(self):
        self.review_due = False

    def unit(self):
        return len(self.g.points)

    def sides(self):
        return [self.cfg.seats[0], OPPONENT] if self.single else list(SEATS2)

    def scores(self):
        if self.single:
            return {self.cfg.seats[0]: int(self.g.score[0]), OPPONENT: int(self.g.score[1])}
        return {s: int(self.g.score[s]) for s in SEATS2}

    def tiebreaks(self):
        """tn.match_winner as three rules: the score (first to 5, from 4:4 by 2, no cap; at the match limit the
        player ahead), else the winner of the last point that was not a let, else (last resort, not from the
        rulebook) the player who received first (format match: the opponent, since you serve first)."""
        def score(m):
            sc = m.scores()
            a, b = m.sides()
            return None if sc[a] == sc[b] else (a if sc[a] > sc[b] else b)

        def last_point(m):
            names = {"you": m.sides()[0], "opponent": OPPONENT} if m.single else {s: s for s in SEATS2}
            won = [names[p["winner"]] for p in m.g.points if p["winner"] in names]
            return won[-1] if won else None

        def first_receiver(m):
            return OPPONENT if m.single else DU.OTHER[m.first]
        return [("score", score), ("last_point", last_point), ("first_receiver", first_receiver)]

    def forfeit(self, seat, why):
        return False                            # the core decides a forfeit; the world is left as it was

    # ------------------------------------------------------------------ actions (low level)
    def schema(self, seat):
        return LOW_SCHEMA

    def toss_refusal(self, seat, pos, height, k_eff):
        """Why a toss is refused now: (message, HTTP code) or None (the serve rules of tn.py / duel.py)."""
        self._ctx()
        return self.g.toss_refusal(pos, height, k_eff) if self.single else self.g.toss_refusal(seat, pos, height, k_eff)

    def apply(self, seat, cmd):
        self._ctx()
        typ = cmd.get("type")
        if typ == "toss":
            k_eff = int(cmd["k0"]) if cmd.get("k0") is not None else self.g.k
            try:
                k = self.g.toss(cmd["pos"], cmd["height"], k_eff) if self.single else \
                    self.g.toss(seat, cmd["pos"], cmd["height"], k_eff)
            except ValueError as e:
                return {"rejected": str(e)}
            return {"toss_takes_effect_at_step": int(k)}
        if typ == "move":
            bx, by = cmd["base"]
            rest = self.g.rt_move(bx, by) if self.single else self.g.rt_move(seat, bx, by)
            return {"base_rest": rest}
        if typ == "none":
            self.g.rt_clear() if self.single else self.g.rt_clear(seat)
            self.deferred[seat] = None
            return {"command": "cleared: the arm returns to the ready pose"}
        if typ != "traj":
            return {"rejected": f"unknown command {typ!r}"}
        k0 = cmd.get("k0")
        if k0 is None or k0 == self.g.k:
            return self._install(seat, cmd)
        if k0 < self.g.k:
            return {"rejected": f"late: planned to take effect at step {k0}, now {self.g.k}"}
        self.deferred[seat] = dict(cmd)
        return {"effect_step": int(k0)}

    def _install(self, seat, cmd):
        """A traj takes effect: its joint targets replace the stroke under way (the legacy rt_install). Refused (no
        change) if it breaks a joint range or speed limit, or if the targets it was planned for are not the robot's."""
        cur = np.asarray(self.target(seat), np.float64)
        exp = cmd.get("expect")
        if exp is not None and not np.array_equal(np.asarray(exp, np.float64), cur):
            return {"rejected": "the robot's joint targets at the effect step differ from the planned ones"}
        q = np.asarray(cmd.get("q") or np.zeros((0, 9)), np.float64).reshape(-1, 9)
        lo, hi = (self.g.s.lo, self.g.s.hi) if self.single else (self.g.lo, self.g.hi)
        prev = cur
        for i, qi in enumerate(q):
            if not np.all(np.isfinite(qi)) or (qi < lo - 1e-9).any() or (qi > hi + 1e-9).any():
                return {"rejected": f"joint target {i} outside the joint ranges"}
            if (np.abs(qi - prev) > VEL_LIM).any():
                return {"rejected": f"joint target {i} breaks a joint speed limit"}
            prev = qi
        q_hit = cmd.get("q_hit")
        res = {"k_effect": self.g.k, "target_at_effect": cur.copy(), "plan": [x.copy() for x in q],
               "info": copy.deepcopy(cmd.get("info") or {}),
               "q_hit": None if q_hit is None else np.asarray(q_hit, np.float64)}
        self.g.rt_install(res) if self.single else self.g.rt_install(seat, res)
        return {"took_effect_step": int(self.g.k), "n_targets": int(len(q))}

    # ------------------------------------------------------------------ observation
    def observe(self, seat, query=None):
        g = self.g
        if self.single:
            self._ensure_frame()                # display state only (the legacy ensure_frame): not in the digest
            o = g.decision_observation()
            o.pop("view_times", None)
            leg = g.leg
            if leg["kind"] == "our_serve" and leg["state"] in ("hold", "toss"):
                o["ball_in_play"] = leg["state"] == "toss"
                o["ball_state"] = "not yet served" if leg["state"] == "hold" else "tossed: yours to hit (your serve)"
                o["heard"] = [] if leg["state"] == "hold" else [{"event": "you tossed", "t": 0.0}]
                o["t_since_played"] = None if leg["state"] == "hold" else g.t_since_launch()
            elif not g.in_play():
                o["ball_in_play"] = False
                o["ball_state"] = "not yet served"
                o["heard"] = []
                o["t_since_played"] = None
            else:
                o["ball_in_play"] = leg["state"] in ("incoming", "to_opp")
                o["ball_state"] = {"incoming": "coming to you", "to_opp": "going to the opponent",
                                   "done": "dead"}.get(leg["state"], leg["state"])
            o.update(self._single_serve())
            la, base_goal = getattr(g, "_last_act", None), g.base_goal
            o["last_points"] = [{k: p[k] for k in ("point", "winner", "how", "score")} for p in g.points[-3:]]
            o["last_shots"] = [{k: r.get(k) for k in ("returned", "summary", "speed_out")} for r in g.results[-3:]]
            o["opponent_level"] = self.level_
        else:
            o = g.side_state(seat)
            sd = g.sides[seat]
            la, base_goal = sd.last_act, sd.base_goal
            other = DU.OTHER[seat]
            o["last_points"] = [{"point": p["point"], "how": p["how"],
                                 "winner": "let" if p["winner"] is None else ("you" if p["winner"] == seat else
                                                                              "opponent"),
                                 "server": "you" if p["server"] == seat else "opponent",
                                 "score": {"you": p["score"][seat], "opponent": p["score"][other]}}
                                for p in g.points[-3:]]
        k_hit = la.get("k_hit") if la else None
        o["command"] = ({"contact_in_s": round(max(0, k_hit - g.k) * DT, 3) if k_hit is not None else None,
                         "phase": "approach" if (k_hit is not None and g.k < k_hit) else "follow_through",
                         "pos": la.get("racket_pos"), "normal": la.get("racket_normal"), "vel": la.get("racket_vel"),
                         "base_at_contact": la.get("base_goal"), "stroke": la.get("stroke"),
                         "vel_delivered": la.get("racket_vel_delivered"), "speed_scale": la.get("speed_scale")}
                        if self.plan_active(seat) and la else "none (the arm returns to / holds the ready pose)")
        o["base_rest"] = ("home " + str(TN.HOME.tolist()) if base_goal is None
                          else [round(float(x), 3) for x in base_goal])
        pend = []
        if self.deferred[seat] is not None:
            pend.append({"type": "act", "takes_effect_at_step": int(self.deferred[seat]["k0"])})
        tp = g._toss_pending
        if tp is not None and (self.single or tp[0] == seat):
            pend.append({"type": "toss", "takes_effect_at_step": int(tp[-1])})
        if pend:
            o["pending_commands"] = pend
        if self.cfg.action_level == "low":
            o["q_target"] = [float(x) for x in self.target(seat)]
        return o

    def _single_serve(self):
        """Format match: the serve fields of the cross-sport convention (you serve, the phase, the court, the window)."""
        g = self.g
        sv, leg = g.serve, g.leg
        you = sv["server"] == 0
        if leg["kind"] == "our_serve":
            phase = {"hold": "open" if g.k >= leg["open"] else "waiting", "toss": "tossed"}.get(leg["state"], "done")
        else:
            phase = "waiting" if (leg["kind"] == "serve" and g.k < leg["launch"]) else "done"
        o = {"phase": phase, "court": sv["half"], "serve_number": sv["faults"] + 1}
        if leg["kind"] == "our_serve" and leg["state"] == "hold":
            o["opens_in_s"] = round(max(0.0, (leg["open"] - g.k) * DT), 3)
            o["deadline_in_s"] = round(max(0.0, (leg["deadline"] - g.k) * DT), 3)
        if leg["kind"] == "serve" and g.k < leg["launch"]:
            o["serve_in_s"] = round((leg["launch"] - g.k) * DT, 3)    # the opponent serves in ... s
        if leg["kind"] == "our_serve" and leg.get("toss"):
            o["tossed_at_step"] = int(leg["toss"]["k"])
            o["toss"] = {"pos": leg["toss"]["pos"], "height": leg["toss"]["height"]}
        if g._toss_pending is not None:
            o["toss_takes_effect_at_step"] = int(g._toss_pending[-1])
        return {"you_serve": you, "you_receive_serve": not you, "serve": o}

    def _snaps(self):
        g = self.g
        self._ensure_frame()
        hist = list(g.review_hist if (self.review_due and g.review_hist) else g.hist)
        return [dict(hist[max(0, len(hist) - 1 - b)]) for b in reversed(TN.Match.VIEW_BACK)]

    def _ensure_frame(self):
        g = self.g
        if not g.hist:
            g.hist.append({"k": g.k, "t": g.t_since_launch() if self.single else g.t_since_played(),
                           "qpos": (g.s.d.qpos if self.single else g.d.qpos).copy(),
                           "b": g.s.b_p if self.single else g.b_p, "leg_id": (len(g.legs) - 1) if self.single
                           else g.leg_id, **({"opp": g.opp_x()} if self.single else {})})

    def camera_job(self, seat):
        snaps = self._snaps()
        job = {"token": self.token, "single": self.single, "side": seat if not self.single else "A",
               "snaps": [{"qpos": np.asarray(s["qpos"], np.float64).copy(), "t": float(s["t"]),
                          "b": tuple(float(x) for x in s["b"]), "k": int(s["k"]), "leg_id": s.get("leg_id"),
                          "opp": float(s.get("opp", 0.0))} for s in snaps]}
        meta = {"view_times": [s["t"] for s in snaps], "view_steps": [s["k"] for s in snaps],
                "view_leg_ids": [s.get("leg_id") for s in snaps], "view_sim_times": [round(s["k"] * DT, 4) for s in snaps],
                "image_layout": IMAGE_LAYOUT, "wall_mono": snaps[-1].get("wall_mono", time.monotonic())}
        return job, meta

    def final_fields(self):
        """The sport's fields, and the arms' self-collision record (bodies/selfcollision.py Monitor: recorded only)."""
        return {**self._final_fields(), "self_collision": ({self.cfg.seats[0]: self.g.s.selfcol["A"].summary()} if self.single else
                {s: mon.summary() for s, mon in self.g.selfcol.items()})}

    def _final_fields(self):
        g = self.g
        if self.single:
            a = self.cfg.seats[0]
            return {"level": self.level_, "points_for": int(g.score[0]), "points_against": int(g.score[1]),
                    "won": bool(g.done and g.winner() == 0), "returns": int(sum(r["returned"] for r in g.results)),
                    "our_legs": int(g.n_our_legs), "match_limit": bool(g.limit_reached), "hits": int(g.hits),
                    "lets": sum(1 for p in g.points if p["winner"] == "let"), "points": g.points,
                    "results": g.results, "shots": g.shots, "physics_steps": int(g.s.steps_total),
                    "serves": g.serves,
                    "plan_lead_steps": lead_steps(self.cfg.slowdown), "rejected_acts": dict(self.rejected),
                    "seats": {a: {"rejected_acts": self.rejected[a]}}}
        return {"score": dict(g.score), "points": g.points, "hits": int(g.hits), "match_limit": bool(g.limit_reached),
                "first_server": self.first, "act_lead_steps": lead_steps(self.cfg.slowdown),
                "rejected_acts": dict(self.rejected), "serves": g.serves,
                "seats": {s: {"rejected_acts": self.rejected[s]} for s in SEATS2}}


# ---------------------------------------------------------------------------------------------- action layers
NUM = {"type": "number"}
VEC2 = {"type": "vector", "len": 2, "unit": "m"}
VEC3 = {"type": "vector", "len": 3}
MOVE = {"move": {"base": {**VEC2, "doc": "the base's rest position [bx, by]: |bx| <= 5.9, -15.4 <= by <= -0.4"}}}
NONE = {"none": {}}
TOSS = {"toss": {"pos": {**VEC3, "unit": "m",
                         "doc": f"your serve: the ball appears at this point (your frame) when the command takes effect "
                                f"and is projected straight up to rise `height` above it, no spin; {TN.TOSS_Z[0]} <= z <= "
                                f"{TN.TOSS_Z[1]}, within {TN.TOSS_REACH} m (horizontally) of your base, which must be at "
                                "rest behind your base line between the centre mark and the singles sideline of the "
                                "serving half (state.serve.court); only the server, once per serve, once the window is "
                                "open (state.serve)"},
                 "height": {**NUM, "min": 0.0, "max": TN.TOSS_H_MAX, "unit": "m", "default": 0.0,
                            "doc": f"how far the ball rises above pos (0: dropped from rest; at most {TN.TOSS_H_MAX})"}}}
HIGH_SCHEMA = {
    "act": {"t": {**NUM, "min": 0.0, "max": T_MAX_S, "unit": "s",
                  "doc": "contact T s after the act takes effect (0 < T <= 5)"},
            "pos": {**VEC3, "unit": "m", "doc": "racket-head centre at contact: |x| <= 6.5, -16 <= y <= 0, 0.1 <= z <= 2.4"},
            "normal": {**VEC3, "doc": "string-face normal at contact, towards the incoming ball (non-zero)"},
            "vel": {**VEC3, "unit": "m/s", "doc": "racket-head velocity at contact, |vel| <= 8"},
            "base": {**VEC2, "default": None, "doc": "optional: the base position at contact (default: the forehand "
                                                     "stance, base = contact xy - [0.62, 0.30])"}},
    **TOSS, **MOVE, **NONE}
LOW_SCHEMA = {
    "traj": {"q": {"type": "list", "doc": f"1-{TRAJ_MAX} joint-target vectors [base_x, base_y, q1..q7], one per 10 ms "
                                          "control step from the next step on, followed exactly (then the arm returns to "
                                          "its ready pose); each within the joint ranges and the joint speed limits"}},
    **TOSS, **MOVE, **NONE}


def _vec(body, key, n):
    v = body.get(key)
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise Refused(f"{key} must be a list of {n} numbers")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise Refused(f"{key} must hold {n} finite numbers")
        out.append(float(x))
    return out


def _base(body):
    b = _vec(body, "base", 2)
    if not (abs(b[0]) <= 5.9 and -15.4 <= b[1] <= -0.4):
        raise Refused("base out of range: |bx| <= 5.9, -15.4 <= by <= -0.4")
    return b


def _common(seat, body, match, allowed, lead=0):
    if not isinstance(body, dict):
        raise Refused("the action must be a JSON object")
    typ = body.get("type")
    if typ not in allowed:
        raise Refused(f"unknown action type {typ!r}; one of {sorted(allowed)}")
    extra = sorted(set(body) - set(allowed[typ]) - {"type"})
    if extra:
        raise Refused(f"unknown field(s) {extra} for {typ}")
    item = match.deferred.get(seat)
    if item is not None and item["k0"] > match.k:
        raise Refused(f"your previous act takes effect at step {item['k0']}; send the next command after that", 409)
    if typ == "none":
        return {"type": "none"}
    if typ == "move":
        return {"type": "move", "base": _base(body)}
    if typ == "toss":
        pos = _vec(body, "pos", 3)
        h = body.get("height", 0.0)
        if isinstance(h, bool) or not isinstance(h, (int, float)) or not math.isfinite(h):
            raise Refused("height must be a finite number")
        why = match.toss_refusal(seat, pos, float(h), match.k + lead)
        if why:
            raise Refused(*why)
        return {"type": "toss", "pos": pos, "height": float(h), "k0": int(match.k + lead) if lead else None}
    return None


class LowLayer:
    """Low level: joint targets (traj), the base's rest position (move), none. Effect at the next control step."""
    level = "low"
    worker = None
    delay_steps = 0

    def schema(self, match, seat):
        return LOW_SCHEMA

    def parse(self, seat, body, match, ctx):
        out = _common(seat, body, match, LOW_SCHEMA)
        if out is not None:
            return out
        q = body.get("q")
        if not isinstance(q, list) or not 1 <= len(q) <= TRAJ_MAX:
            raise Refused(f"q must be a list of 1 to {TRAJ_MAX} joint-target vectors")
        rows = [_vec({"q": r}, "q", 9) for r in q]
        return {"type": "traj", "k0": None, "q": rows}

    def snapshot(self, seat, cmd, match, ordinal):
        return None

    def prepare(self, job):
        return None

    def lower(self, seat, cmd, prepared, match):
        return cmd, {}

    def reply(self, seat, cmd, info, match):
        return info


class RacketLayer(LowLayer):
    """High level: the legacy racket command. `act` is planned in the seat's plan worker (tn.PlanCtx.plan_act, the
    legacy planner) from a snapshot of the seat's robot at the step it is accepted, for its effect step (accepted step
    + lead), and lowered to a low-level traj that takes effect there."""
    level = "high"
    worker = "plan"

    def __init__(self, lead):
        self.lead = lead
        self.ctx = None                         # an in-process planner for a late act (see lower)

    def schema(self, match, seat):
        return HIGH_SCHEMA

    def parse(self, seat, body, match, ctx):
        out = _common(seat, body, match, HIGH_SCHEMA, self.lead)   # a toss takes effect after the act lead
        if out is not None:
            return out
        T = body.get("t")
        if isinstance(T, bool) or not isinstance(T, (int, float)) or not math.isfinite(T):
            raise Refused("t must be a finite number")
        pos, nrm, vel = _vec(body, "pos", 3), _vec(body, "normal", 3), _vec(body, "vel", 3)
        if np.linalg.norm(nrm) < 1e-6:
            raise Refused("normal must be a non-zero vector")
        if not 0.0 < T <= T_MAX_S:
            raise Refused(f"t must be in (0, {T_MAX_S}] s from when the act takes effect")
        if not (abs(pos[0]) <= 6.5 and -16.0 <= pos[1] <= 0.0 and 0.1 <= pos[2] <= 2.4):
            raise Refused("pos out of range: |x| <= 6.5, -16 <= y <= 0, 0.1 <= z <= 2.4")
        if np.linalg.norm(vel) > 8.0:
            raise Refused("vel too large: |vel| <= 8 m/s")
        out = {"type": "act", "t": float(T), "pos": pos, "normal": nrm, "vel": vel}
        if body.get("base") is not None:
            out["base"] = _base(body)
        return out

    def snapshot(self, seat, cmd, match, ordinal):
        if cmd["type"] != "act":
            return None
        match._ctx()
        snap = TN.plan_snapshot(match.g) if match.single else match.g.plan_snapshot(seat)
        return {"snap": snap, "lead": self.lead, "dec": cmd}

    def prepare(self, job):
        raise RuntimeError("runs in the plan worker")

    def lower(self, seat, cmd, prepared, match):
        if cmd["type"] != "act":
            return cmd, {}
        late = match.k - int(prepared["k_effect"])
        if late > 0 and cmd["t"] - late * DT > 0:
            # the plan worker missed the effect step (the legacy service stalled the world until the plan came): plan
            # again here, from the robot's state now, for an effect now and the same contact time (the world waits)
            if self.ctx is None:
                self.ctx = TN.PlanCtx()
            job = self.snapshot(seat, cmd, match, 0)
            d = dict(cmd, t=round(cmd["t"] - late * DT, 6))
            prepared = self.ctx.plan_act(job["snap"], 0, d["t"], d["pos"], d["normal"], d["vel"], d.get("base"))
            prepared["info"]["replanned_late_steps"] = late
        info = _clean({k: v for k, v in prepared["info"].items() if k in STROKE_KEYS + ("replanned_late_steps",)})
        low = _clean({"type": "traj", "k0": int(prepared["k_effect"]), "q": prepared["plan"],
                      "expect": prepared["target_at_effect"], "q_hit": prepared["q_hit"], "info": prepared["info"]})
        return low, {"stroke": info, "effect_step": int(prepared["k_effect"])}

    def reply(self, seat, cmd, info, match):
        return info


# ---------------------------------------------------------------------------------------------- render
class Views:
    """The legacy observation images: 2 cameras x 4 frames, each tile stamped `k=.. t=..`, one PNG. Tiles are cached
    per (match, frame) so a frame shown in several observations is drawn once (same pixels)."""

    def __init__(self, single, side):
        import mujoco as mj
        self.mj, self.single = mj, single
        self.m = mj.MjModel.from_xml_path(str(TN.SCENE if single else DU.SCENE2))
        w, h = TN.Match.TILE
        self.r = mj.Renderer(self.m, height=h, width=w)
        self.rd = mj.MjData(self.m)
        self.cams = []
        for lookat, dist, az, el in TN.Match.CAMS:
            c = mj.MjvCamera()
            c.lookat[:] = lookat if side == "A" else DU.mir(lookat)
            c.distance, c.azimuth, c.elevation = dist, (az if side == "A" else az + 180.0), el
            self.cams.append(c)
        self.cache: dict = {}

    def tile(self, ci, sn):
        from PIL import Image, ImageDraw
        mj, rd = self.mj, self.rd
        rd.qpos[:] = sn["qpos"]
        rd.mocap_pos[0] = sn["b"]
        mj.mj_kinematics(self.m, rd)
        self.r.update_scene(rd, camera=self.cams[ci])
        if self.single:
            TN.Match._extra_geoms(self.r.scene, sn["opp"], sn["b"])
        im = Image.fromarray(self.r.render())
        ImageDraw.Draw(im).text((6, 4), f"k={sn['k']} t={sn['t']:.2f}s", fill=(255, 255, 255))
        return np.asarray(im)

    def draw(self, job):
        if any(key[0] != job["token"] for key in self.cache):
            self.cache = {}
        rows = []
        for ci in range(len(self.cams)):
            tiles = []
            for sn in job["snaps"]:
                key = (job["token"], ci, sn["k"], sn["leg_id"], sn["t"])
                if key not in self.cache:
                    self.cache[key] = self.tile(ci, sn)
                tiles.append(self.cache[key])
            rows.append(np.concatenate(tiles, axis=1))
        if len(self.cache) > 64:
            for key in sorted(self.cache, key=lambda x: x[2])[:len(self.cache) - 48]:
                del self.cache[key]
        return png(np.concatenate(rows, axis=0))


def png(img: np.ndarray) -> bytes:
    """RGB8 PNG without row filters, zlib level 1 (the same pixels as the legacy imageio PNG, about 3x faster)."""
    h, w = img.shape[:2]
    raw = np.zeros((h, w * 3 + 1), np.uint8)
    raw[:, 1:] = np.ascontiguousarray(img[..., :3], np.uint8).reshape(h, -1)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))


_LOCAL_VIEWS: dict = {}


class TennisSport:
    name = "tennis"
    protocol = "realtime"
    revision = REVISION
    source_dirs = [HERE]
    workers = ("plan", "render")

    def __init__(self):
        self._lead = lead_steps(None)

    def build(self, cfg):
        self._lead = lead_steps(cfg.slowdown)   # the core asks for the action layer right after build()
        return Tennis(cfg)

    def action_layer(self, level):
        return RacketLayer(self._lead) if level == "high" else LowLayer()

    def worker_init(self, kind, seat, cfg):
        if kind == "plan":
            return TN.PlanCtx()
        return Views(len(cfg.seats) == 1, seat if len(cfg.seats) > 1 else "A")

    def worker_call(self, obj, kind, job):
        if kind == "plan":
            d = job["dec"]
            return obj.plan_act(job["snap"], job["lead"], d["t"], d["pos"], d["normal"], d["vel"], d.get("base"))
        if obj is None:                         # in-process (core/local.py): one renderer per (format, side)
            key = (job["single"], job["side"])
            if key not in _LOCAL_VIEWS:
                _LOCAL_VIEWS[key] = Views(*key)
            obj = _LOCAL_VIEWS[key]
        return {"png": obj.draw(job)}

    def spec(self, cfg, seat, m):
        lead = lead_steps(cfg.slowdown)
        single = len(cfg.seats) == 1
        out = {
            "robot": (f"franka-panda on a 0.55 m pedestal, base on x-y rails (<= {TN.BASE_VMAX} m/s, <= {TN.BASE_AMAX} "
                      "m/s^2) + tennis racket (head centre 0.52 m beyond the flange, strung area an ellipse 0.26 x 0.32 m)"),
            "frame": ("your own: x to your right, y towards the opponent, z up; your half y < 0, your base's home "
                      f"{TN.HOME.tolist()}" + ("" if single else "; the opponent's robot plays from y > 0 (in its own "
                                                                  "frame it sees the game exactly as you do)")),
            "match": {"win_points": TN.WIN_POINTS, "win_by": TN.WIN_BY, "cap_points": TN.CAP_POINTS,
                      "max_hits": TN.MAX_HITS, "max_points_played": TN.MAX_POINTS,
                      **({"opponent_level": m.level_, "first_server": "you",
                          "serves": "the players serve (ITF Rules 16-22): you serve point 1, then each player serves "
                                    "two points in turn (tie-break order, Rule 5 b)"}
                         if single else
                         {"first_server": m.first, "first_server_is_you": m.first == seat,
                          "serves": "the players serve (ITF Rules 16-22): match.first_server serves point 1, then each "
                                    "player serves two points in turn (tie-break order, Rule 5 b)"}),
                      "serve": {"window_opens_s": TN.SERVE_OPEN_S, "window_s": TN.SERVE_CLOCK_S,
                                "second_serve_window_s": TN.SECOND_CLOCK_S, "toss_height_max": TN.TOSS_H_MAX,
                                "toss_reach": TN.TOSS_REACH, "toss_z": list(TN.TOSS_Z), "base_rest_speed": TN.BASE_REST_V,
                                "base_still_m": TN.BASE_STILL_M}},
            "act": {"lead_steps": lead, "lead_game_s": round(lead * DT, 4), "T_max_s": T_MAX_S,
                    "rule": (f"an act is accepted at the next 10 ms control step and takes effect {lead} control steps "
                             "later (the stroke is planned meanwhile; `pending_commands` shows its effect step, and "
                             "a command sent before then is refused with HTTP 409); `move` and `none` take effect at "
                             "the next step" if cfg.action_level == "high" else
                             "traj, move and none take effect at the next control step")},
            "cameras": [dict(c, fovy_deg=TN.Match.FOVY, tile_px=list(TN.Match.TILE), focal_px=579.4,
                             principal_point_px=[320, 240], image_right="forward x up") for c in TN.Match.cam_info()],
            "image": "2 rows (behind, side) x 4 columns (camera frames 40 ms apart, oldest left); tile 640 x 480",
        }
        if cfg.action_level == "low":
            lo, hi = (m.g.s.lo, m.g.s.hi) if single else (m.g.lo, m.g.hi)
            out["joints"] = {"names": TN.JOINTS, "low": [float(x) for x in lo], "high": [float(x) for x in hi],
                             "max_step": [float(x) for x in TN.VEL_LIMIT * DT],
                             "ready_pose_arm": [float(x) for x in TN.Q0_ARM],
                             "rule": "each target within [low, high] and within max_step of the previous target "
                                     "(the first one: of q_target in the state)"}
        return out

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = TennisSport()
