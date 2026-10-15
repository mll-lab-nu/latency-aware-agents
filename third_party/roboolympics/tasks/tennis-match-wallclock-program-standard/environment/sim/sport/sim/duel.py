"""Tennis duel: two robots, two agents, first to 5 points.

Robot A (rails at y < 0, facing +y) and robot B (rails at y > 0, facing -y) are identical Franka arms on x-y rail
bases with tennis rackets, in one MuJoCo scene (scene/tn_duel.xml, made by scripts/olympiad/tennis/mk_duel_scene.py).
Each player sees and acts in its OWN frame, the frame of the single-player task (its half y < 0, its base's home
(0, -11.4)): for B every position, velocity and spin is mirrored, (x, y, z) -> (-x, -y, z) (a turn of 180 degrees about
z). B's joint values mean the same in B's frame as A's in the world, so both players get the same instruction, cameras
and stroke interface, and each act is planned by the single-player stroke controller (tn.PlanCtx) on a robot-only copy
in that player's frame.

Physics: tn.py's (the ball is a point mass with drag, Magnus lift, the court bounce and racket impact laws, integrated at
1 ms; MuJoCo moves the robots only, contacts disabled).

Serve (revision 5): the players' robots serve by ITF Rules 16-22 (tn.py's rules, the same as in the single game). The
order is the tie-break's (Rule 5 b): FIRST_SERVER serves point 1, then each serves two points in turn; the serving half
alternates every point, point 1 from the right half (Rule 17). The window opens tn.SERVE_OPEN_S after the point starts;
the server's base must be at rest behind its base line between the centre mark and the singles sideline of the
serving half; `toss` (Rule 16: a one-armed player releases the ball with the racket) puts the ball at a point within
tn.TOSS_REACH of the base, projected straight up by `height`, no spin; the server must hit it before it lands, and it must
first land in the receiver's service box diagonally opposite. Faults give a second serve from the same half, two lose the
point (Rules 19-20): out of the box, into the net, a foot fault (the base moves > tn.BASE_STILL_M during the motion,
Rule 18), the toss missed during a stroke (19 b; let drop without a stroke: tossed again, 19 Case 1), no toss within the
window (undue delay); a serve touching the net and otherwise good is a let (22 a). The receiver's robot cannot touch
the serve before it bounces (24 e: it loses the point). Each robot draws its own execution-scatter stream (tn.scatter_rng;
stream 1 = the first server's robot, 2 = the other's; revision 7, rule A6): the duel is a pair of matches on the same
seed with the first server swapped, so the two players swap the streams and meet the same draws over the pair.
Revisions 5-6 drew one sequence for both robots (the n-th act of each got the same draw).

Scoring: first to 5 points; from 4:4 a player must lead by 2 (no cap, revision 6). The match
also ends after MAX_HITS racket hits or MAX_POINTS points played (lets included): the player ahead wins; at equal points
the winner of the last point wins (no draws). The same format as the single-player game (tn.py).

Rules (a deterministic function of the ball events, evaluated inside every control step, so the recorded joint targets
of both robots replay the match bit-exactly):
  - the serve (above);
  - any other ball may be volleyed or hit after one bounce on the hitter's half; a second bounce (miss) loses the point;
  - after a hit by P the ball must cross the net above it and first land in the other player's singles court (lines
    in), directly or after touching the net's top edge (net cord, ITF Rule 25 a): a ball into the net, out, or a
    bounce on P's own half lose P the point; a racket touches the ball at most once in a row; a racket contact beyond the
    net (y > 0 in the hitter's frame) loses the hitter the point (ITF 24 i).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np

import mujoco as mj
import tn
from tn import (ALPHA, BALL_R, CONTROL_DT, HEAD_A, HEAD_B, E_RACKET, KD, KP, MU_RACKET, OFFCENTRE_LOSS, PHYS_DT,  # noqa: F401
                RACKET_GAIN, SUBSTEPS, VEL_LIMIT, JOINTS, HOME, ARM_VEL, base_goto, ball_advance, impact, in_court,
                in_service_box)

HERE = Path(__file__).resolve().parent
SCENE2 = HERE / "scene" / "tn_duel.xml"
# the match format of the single-player game (tn.py): first to 5, from 4:4 by 2 (no cap); the match limit
from tn import WIN_POINTS, WIN_BY, CAP_POINTS, MAX_HITS, MAX_POINTS, match_won, match_winner, serve_index  # noqa: E402
READY_STEPS = 100               # 1.0 s between the resume and the serve
LEG_STEPS = 500                 # a ball not decided 5 s after it was played: the player to hit loses
CAM_EVERY = 2
SEED = int(os.environ.get("DUEL_SEED", "1"))
REVISION = "tennis-duel-wallclock-7"     # 2: volleys, receiver every point, win by 2 (cap 7), match limit; 3: net cord; 4: no draws (the match limit ends with a winner); 5: the players serve (ITF 16-22), rails over the whole half, 24 i, one shared scatter sequence; 6: no cap (win by 2); 7: one scatter stream per robot (the first server's 1, the other's 2), swapped by the pair
FIRST_SERVER = os.environ.get("FIRST_SERVER", "A")
if FIRST_SERVER not in ("A", "B"):
    raise ValueError("FIRST_SERVER must be A or B")
SIDES = ("A", "B")
OTHER = {"A": "B", "B": "A"}
MIR = np.array([-1.0, -1.0, 1.0])
Q_HOME = np.r_[HOME, tn.Q0_ARM]


def mir(v):
    return np.asarray(v, float) * MIR


def own(side, v):
    """A world vector in `side`'s own frame (and back: the mirror is its own inverse)."""
    return np.asarray(v, float).copy() if side == "A" else mir(v)


def half_of(y):
    return "A" if y < 0 else "B"


class Side:
    """One robot's live control state (its own frame)."""

    def __init__(self, name):
        self.name = name
        self.plan, self.plan_j = [], 0
        self.base_goal = None
        self.q_hit_active = None
        self.last_act = None
        self.n_snapshots = 0           # acts planned so far (the execution scatter's ordinal)
        self.hit_enabled = False
        self.prev = None


class Duel:
    CAMS = tn.Match.CAMS
    VIEW_BACK = tn.Match.VIEW_BACK
    TILE = tn.Match.TILE

    def __init__(self):
        m = self.m = mj.MjModel.from_xml_path(str(SCENE2))
        d = self.d = mj.MjData(m)
        self.selfcol = tn.SC.monitors(m, {"A": "selfcol", "B": "selfcol_b"}, "franka_panda")   # recorded only
        jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
        sfx = {"A": "", "B": "_b"}
        self.qa = {s: np.array([m.jnt_qposadr[jid(j + sfx[s])] for j in JOINTS]) for s in SIDES}
        self.va = {s: np.array([m.jnt_dofadr[jid(j + sfx[s])] for j in JOINTS]) for s in SIDES}
        aid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, n)
        anames = ["motor_bx", "motor_by"] + [f"motor{i}" for i in range(1, 8)]
        self.aa = {s: np.array([aid(a + sfx[s]) for a in anames]) for s in SIDES}
        self.lo = m.jnt_range[[jid(j) for j in JOINTS], 0].copy()
        self.hi = m.jnt_range[[jid(j) for j in JOINTS], 1].copy()
        self.site = {s: mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "racket_center" + sfx[s]) for s in SIDES}
        self.clo, self.chi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        mj.mj_resetData(m, d)
        for s in SIDES:
            d.qpos[self.qa[s]] = Q_HOME
        mj.mj_forward(m, d)
        self.target = {s: Q_HOME.copy() for s in SIDES}
        self.actions = {s: [] for s in SIDES}
        self.sides = {s: Side(s) for s in SIDES}
        self.k = 0
        self.b_p, self.b_v, self.b_w, self.b_live = (0.0, 0.0, -5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False
        self.events: list[dict] = []
        self.score = {"A": 0, "B": 0}
        self.points: list[dict] = []
        self.hits = 0
        self.done = False
        self.limit_reached = False
        self.leg_id = -1
        self.hist: deque = deque(maxlen=16)
        self.review_hist: list = []
        self._toss_pending = None
        self.serves: list[dict] = []    # every serve and its outcome
        self._start_point(0)

    # ------------------------------------------------------------------ points
    def server(self):
        """The tie-break order (ITF Rule 5 b): FIRST_SERVER serves point 1, then each player serves two points in turn."""
        return FIRST_SERVER if tn.server_of(sum(self.score.values()), 0) == 0 else OTHER[FIRST_SERVER]

    def receiver(self):
        return OTHER[self.server()]

    def won(self):
        return match_won(self.score["A"], self.score["B"])

    def winner(self):
        """'A' or 'B' once the match is over (no draws): by the score, else the winner of the last point that was not
        a let, else (last resort) the player who received first."""
        if not self.done:
            return None
        w = match_winner(self.score["A"], self.score["B"],
                         [{"A": 0, "B": 1}.get(p["winner"]) for p in self.points], 0 if FIRST_SERVER == "B" else 1)
        return SIDES[w]

    def _start_point(self, hold_from):
        self.review_hist = list(self.hist)
        self.hist.clear()
        n = len(self.points)
        srv = self.server()
        self.leg_id += 1
        self.rs = {"point": n, "server": srv, "receiver": OTHER[srv], "half": tn.serve_half(sum(self.score.values())),
                   "launch": hold_from, "played_k": hold_from, "expect": srv, "hitter": None, "bounces": 0,
                   "state": "live", "phase": "serve", "faults": 0, "toss": None, "hits_in_point": 0,
                   "open_k": hold_from + int(round(tn.SERVE_OPEN_S / CONTROL_DT)),
                   "clock_k": hold_from + int(round((tn.SERVE_OPEN_S + tn.SERVE_CLOCK_S) / CONTROL_DT)), "serve": None}
        self._park()
        for s in SIDES:
            sd = self.sides[s]
            sd.hit_enabled = False
            sd.base_goal = None
        self._pending_point = None
        self._toss_pending = None

    def _park(self):
        self.b_p, self.b_v, self.b_w = tn.PARK, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self.b_live = False
        self.events = []

    def in_play(self):
        return self.rs["phase"] != "serve"

    # ------------------------------------------------------------------ the serve
    def toss_refusal(self, s, pos, height, k_eff):
        """Why a toss by `s` (pos in its own frame) taking effect at step k_eff is refused now: (message, code) or None."""
        rs = self.rs
        if self.done or rs["server"] != s:
            return "toss: it is not your serve", 409
        if rs["phase"] != "serve" or self._toss_pending is not None:
            return "toss: the ball was already tossed (one toss per serve)", 409
        if k_eff < rs["open_k"]:
            return (f"toss: the receiver is ready at step {rs['open_k']}; this toss would take effect at step {k_eff} "
                    "(state.serve.opens_in_s)"), 409
        base = self.d.qpos[self.qa[s][:2]]               # B's joint values are in B's own frame
        why = tn.toss_problem([float(x) for x in pos], float(height), base,
                              float(np.hypot(*self.d.qvel[self.va[s][:2]])), rs["half"])
        return (why, 400) if why else None

    def toss(self, s, pos, height, k_eff):
        why = self.toss_refusal(s, pos, height, k_eff)
        if why:
            raise ValueError(why[0])
        self._toss_pending = (s, own(s, pos), float(height), int(k_eff))
        return int(k_eff)

    def _release(self):
        s, p, h, _k = self._toss_pending
        self._toss_pending = None
        rs = self.rs
        rs["phase"] = "toss"
        rs["toss"] = {"pos": [round(float(x), 4) for x in p], "height": h, "k": int(self.k)}
        rs["base_at_release"] = self.d.qpos[self.qa[s][:2]].copy()
        rs["played_k"] = rs["launch"] = self.k
        rs["expect"], rs["hitter"], rs["bounces"], rs["serve_cord"] = s, None, 0, False
        self.leg_id += 1
        self.b_p, self.b_v, self.b_w = tuple(float(x) for x in p), (0.0, 0.0, tn.toss_vz(h)), (0.0, 0.0, 0.0)
        self.b_live = True
        self.events = []
        self.sides[s].hit_enabled = True
        self.sides[OTHER[s]].hit_enabled = False

    def _log_serve(self, outcome, why):
        rs = self.rs
        self.serves.append({"point": rs["point"] + 1, "server": rs["server"], "half": rs["half"],
                            "serve_no": rs["faults"] + 1, "outcome": outcome, "why": why, "step": int(self.k),
                            "toss": rs["toss"], "speed": (rs.get("serve_hit") or {}).get("speed"),
                            "bounce": rs.get("serve_bounce")})

    def _new_serve(self, kind):
        rs = self.rs
        rs["phase"], rs["toss"], rs["hitter"], rs["bounces"] = "serve", None, None, 0
        rs["expect"] = rs["server"]
        rs["serve_hit"] = rs["serve_bounce"] = None
        rs["open_k"] = self.k
        clock = tn.SECOND_CLOCK_S if (kind == "second" or rs["faults"] > 0) else tn.SERVE_CLOCK_S
        rs["clock_k"] = self.k + int(round(clock / CONTROL_DT))
        self._toss_pending = None
        self._park()
        for s in SIDES:
            self.sides[s].hit_enabled = False

    def _fault(self, code, why):
        """A service fault (Rule 19): a second serve from the same half, or the point lost (double fault, Rule 20)."""
        rs = self.rs
        why = f"serve_fault:{code} ({why})"
        self._log_serve("fault", why)
        if rs["faults"] == 0:
            rs["faults"] = 1
            return self._new_serve("second")
        return self._end_point(rs["server"], why.replace(" (", " (double fault: ", 1))

    def _let(self, why):
        self._log_serve("let", why)
        self._new_serve("let")

    def _plan_active(self, s):
        sd = self.sides[s]
        return sd.plan_j < len(sd.plan)

    def t_since_played(self):
        return round((self.k - self.rs["played_k"]) * CONTROL_DT, 6)

    def _end_point(self, loser, how):
        if self._pending_point is None:
            self._pending_point = (loser, how)

    def _commit_point(self):
        loser, how = self._pending_point
        winner = OTHER[loser] if loser is not None else None
        if winner is not None:
            self.score[winner] += 1
        self.points.append({"point": len(self.points) + 1, "winner": winner, "loser": loser, "how": how,
                            "score": dict(self.score), "hits": self.rs["hits_in_point"], "step": self.k,
                            "server": self.rs["server"], "receiver": self.rs["receiver"], "let": loser is None})
        self.rs["state"] = "over"
        self.b_live = False
        for s in SIDES:                     # no command carries over to the next point
            sd = self.sides[s]
            sd.plan, sd.plan_j, sd.last_act, sd.q_hit_active = [], 0, None, None
        if self.won():
            self.done = True
            return
        if self.hits >= MAX_HITS or len(self.points) >= MAX_POINTS:
            self.done = True
            self.limit_reached = True
            return
        self._start_point(self.k)

    # ------------------------------------------------------------------ physics + referee (shared by live and replay)
    def _check(self, s, q):
        if q.shape != (9,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{s}: action must be 9 finite joint targets")
        if (q < self.lo - 1e-9).any() or (q > self.hi + 1e-9).any():
            raise ValueError(f"{s}: step {self.k}: joint target outside range")
        if (np.abs(q - self.target[s]) > VEL_LIMIT * CONTROL_DT * (1 + 1e-6) + 1e-12).any():
            raise ValueError(f"{s}: step {self.k}: joint speed limit")

    def step(self, qA, qB):
        qs = {"A": np.asarray(qA, np.float64).reshape(-1), "B": np.asarray(qB, np.float64).reshape(-1)}
        for s in SIDES:
            self._check(s, qs[s])
        if self._toss_pending is not None and not self.done and self.k >= self._toss_pending[3]:
            self._release()
        m, d = self.m, self.d
        prev = {s: self.target[s] for s in SIDES}
        for j in range(1, SUBSTEPS + 1):
            for s in SIDES:
                qd_des = (qs[s] - prev[s]) / CONTROL_DT
                q_des = prev[s] + (qs[s] - prev[s]) * (j / SUBSTEPS)
                tau = KP * (q_des - d.qpos[self.qa[s]]) + KD * (qd_des - d.qvel[self.va[s]]) + d.qfrc_bias[self.va[s]]
                d.ctrl[self.aa[s]] = np.clip(tau, self.clo[self.aa[s]], self.chi[self.aa[s]])
            mj.mj_step(m, d)
            for mon in self.selfcol.values():
                mon.update(d)
            poses = {s: (d.site_xpos[self.site[s]].copy(), d.site_xmat[self.site[s]].reshape(3, 3).copy()) for s in SIDES}
            if self.b_live and self._pending_point is None:
                for s in SIDES:
                    sd = self.sides[s]
                    if sd.hit_enabled and sd.prev is not None:
                        self._check_hit(s, *poses[s])
                for s in SIDES:
                    self.sides[s].prev = (poses[s][0], poses[s][1], self.b_p)
                self._advance_ball()
            else:
                for s in SIDES:
                    self.sides[s].prev = (poses[s][0], poses[s][1], self.b_p)
        for s in SIDES:
            self.target[s] = qs[s].copy()
            self.actions[s].append(qs[s].copy())
        self.k += 1
        if not self.done and self._pending_point is None:
            self._referee()
        if self.k % CAM_EVERY == 0 or self._pending_point is not None:
            self.hist.append({"k": self.k, "t": self.t_since_played(), "qpos": d.qpos.copy(), "b": self.b_p,
                              "leg_id": self.leg_id})
        if self._pending_point is not None:
            self._commit_point()

    def _advance_ball(self):
        p, v, w, e = ball_advance(self.b_p, self.b_v, self.b_w)
        self.b_p, self.b_v, self.b_w = p, v, w
        if e is None:
            return
        if e[0] == "net":
            self.b_live = False
            self.events.append({"kind": "net", "k": self.k, "x": round(e[1], 4), "z": round(e[2], 4)})
        elif e[0] == "cord":
            self.events.append({"kind": "cord", "k": self.k, "x": round(e[1], 4), "z": round(e[2], 4)})
        else:
            self.events.append({"kind": "bounce", "k": self.k, "x": round(e[1], 4), "y": round(e[2], 4),
                                "half": half_of(e[2])})
        self._on_event(self.events[-1])

    def _check_hit(self, s, c1, R1):
        """tn.Phys._check_hit for robot s."""
        sd = self.sides[s]
        c0, R0, p0 = sd.prev
        n0, n1 = R0[:, 0], R1[:, 0]
        p0a = np.array(p0); p1a = np.array(self.b_p)
        s0 = float((p0a - c0) @ n0); s1 = float((p1a - c1) @ n1)
        side = 1.0 if s0 >= 0 else -1.0
        a0, a1 = side * s0, side * s1
        if not (a0 >= BALL_R > a1):
            return
        f = (a0 - BALL_R) / (a0 - a1)
        pc = c0 + f * (c1 - c0); ps = p0a + f * (p1a - p0a)
        nn = n0 + f * (n1 - n0); nn = nn / np.linalg.norm(nn)
        r = ps - pc
        la, lb = float(r @ R1[:, 1]), float(r @ R1[:, 2])
        rho2 = (la / HEAD_A) ** 2 + (lb / HEAD_B) ** 2
        if rho2 > 1.0:
            return
        vr = (c1 - c0) / PHYS_DT * RACKET_GAIN
        n_out = side * nn
        vb = np.array(self.b_v)
        e = E_RACKET * (1.0 - OFFCENTRE_LOSS * rho2)
        u = tuple(float(x) for x in vb - vr)
        u2, w2 = impact(u, self.b_w, tuple(float(x) for x in n_out), e, MU_RACKET)
        v_out = np.array(u2) + vr
        p_new = pc + (r - (r @ nn) * nn) + n_out * (BALL_R + 0.0005)
        self.b_p = tuple(float(x) for x in p_new); self.b_v = tuple(float(x) for x in v_out)
        self.b_w = tuple(float(x) for x in w2)
        sd.hit_enabled = False
        ev = {"kind": "hit", "side": s, "k": self.k, "pos": [round(float(x), 4) for x in ps],
              "offcentre": round(math.sqrt(rho2), 3), "speed": round(float(np.linalg.norm(v_out)), 3)}
        self.events.append(ev)
        self._on_event(ev)

    def _on_event(self, ev):
        rs = self.rs
        if self.done or rs["state"] != "live" or self._pending_point is not None:
            return
        ph, kind = rs["phase"], ev["kind"]
        srv, recv = rs["server"], rs["receiver"]
        if ph == "toss":                                 # the server's toss, before the strike
            if kind == "hit":
                moved = float(np.hypot(*(self.d.qpos[self.qa[srv][:2]] - rs["base_at_release"])))
                rs["serve_hit"] = {"k": int(self.k), "speed": ev["speed"], "base_moved": round(moved, 3)}
                if moved > tn.BASE_STILL_M:
                    return self._fault("foot_fault", f"the base moved {moved:.2f} m during the service motion (Rule 18)")
                rs["phase"], rs["hitter"], rs["expect"], rs["bounces"] = "flight", srv, recv, 0
                rs["played_k"] = self.k
                rs["hits_in_point"] += 1
                self.hits += 1
                self.leg_id += 1
                self.sides[recv].hit_enabled = True
                return
            if kind == "bounce":
                if self._plan_active(srv):
                    return self._fault("missed_toss", "the server missed the toss (Rule 19 b)")
                self._log_serve("retoss", "the toss was let drop without a stroke (Rule 19 Case 1)")
                rs["phase"], rs["toss"] = "serve", None
                self._park()
                self.sides[srv].hit_enabled = False
                return
            return self._fault("net", "the toss touched the net")
        if ph == "flight":                               # the serve, until its first bounce
            if kind == "cord":
                rs["serve_cord"] = True
                return
            if kind == "net":
                return self._fault("net", "the serve hit the net")
            if kind == "hit":                            # ITF 24 e
                return self._end_point(ev["side"], f"receive_fault:volley ({ev['side']} returned the serve before it "
                                                   "bounced, Rule 24 e)")
            if kind == "bounce":
                ox, oy = own(recv, [ev["x"], ev["y"], 0.0])[:2]
                rs["serve_bounce"] = [round(float(ox), 3), round(float(oy), 3)]
                if not (ev["half"] == recv and tn.in_box(ox, oy, rs["half"])):
                    return self._fault("out_of_box", f"the serve landed out of the box at ({ox:.2f}, {oy:.2f})")
                if rs.get("serve_cord"):
                    return self._let("let: the serve touched the net (Rule 22 a)")
                self._log_serve("in", "good")
                rs["phase"], rs["bounces"] = "rally", 1
            return
        exp, hitter = rs["expect"], rs["hitter"]
        if kind == "bounce":
            x, y, half = ev["x"], ev["y"], ev["half"]
            ox, oy = own(exp, [x, y, 0.0])[:2]          # in the frame of the player to hit
            if rs["bounces"] == 0:
                if half != exp:
                    return self._end_point(hitter, f"{hitter}'s shot came down on its own half")
                if not in_court(ox, oy, -1):
                    return self._end_point(hitter, f"{hitter}'s shot landed out")
                rs["bounces"] = 1
                return
            return self._end_point(exp, f"{exp} let the ball bounce twice")
        if kind == "cord":                               # the ball touched the net's top edge: play goes on (25 a)
            return
        if kind == "net":
            return self._end_point(hitter, f"{hitter}'s shot hit the net")
        if kind == "hit":
            p = ev["side"]
            if own(p, ev["pos"])[1] > 0.0:               # ITF 24 i: the ball hit before it passed the net
                return self._end_point(p, f"{p}'s racket met the ball beyond the net (Rule 24 i)")
            rs["hitter"], rs["expect"], rs["bounces"] = p, OTHER[p], 0
            rs["played_k"] = self.k; rs["hits_in_point"] += 1; self.hits += 1
            self.leg_id += 1
            self.sides[OTHER[p]].hit_enabled = True
            return

    def _referee(self):
        rs = self.rs
        if rs["state"] != "live":
            return
        ph = rs["phase"]
        if ph == "serve":
            if self.k >= rs["clock_k"] and self._toss_pending is None:
                return self._fault("undue_delay", "no toss within the serve window")
            return
        if ph in ("toss", "flight"):
            if not self.b_live or self.k - rs["played_k"] >= LEG_STEPS:
                return self._fault("not_completed", "the serve was not completed")
            return
        if not self.b_live or self.k - rs["played_k"] >= LEG_STEPS:
            exp, hitter = rs["expect"], rs["hitter"]
            if rs["bounces"] >= 1:
                return self._end_point(exp, f"{exp} missed the ball")
            return self._end_point(hitter, f"{hitter}'s shot was not completed")

    def replay(self, qA_list, qB_list):
        """Re-run the match from both robots' recorded joint targets (they determine everything else)."""
        if len(qA_list) != len(qB_list):
            raise ValueError("seat trajectory lengths differ")
        for a, b in zip(qA_list, qB_list):
            if self.done:
                raise ValueError("trajectory continues after the match end")
            self.step(a, b)

    def state_sha(self):
        h = hashlib.sha256()
        d = self.d
        for a in (d.qpos, d.qvel, d.qacc_warmstart, d.ctrl, np.array([d.time]), np.array(self.b_p + self.b_v + self.b_w)):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(json.dumps(self.points, sort_keys=True).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ live control (each side in its own frame)
    def _target_vel(self, s):
        a = self.actions[s]
        return (a[-1] - a[-2]) / CONTROL_DT if len(a) >= 2 else np.zeros(9)

    def _ready_target(self, s):
        tgt = self.target[s]
        lim = ARM_VEL * CONTROL_DT * 0.6
        arm = tgt[2:] + np.clip(tn.Q0_ARM - tgt[2:], -lim, lim)
        sd = self.sides[s]
        goal = HOME if sd.base_goal is None else sd.base_goal
        b, _ = base_goto(tgt[:2], self._target_vel(s)[:2], goal)
        return np.r_[b, arm]

    def _arm_target(self, s):
        sd = self.sides[s]
        if sd.plan_j < len(sd.plan):
            q = sd.plan[sd.plan_j]; sd.plan_j += 1
            return q
        return self._ready_target(s)

    def rt_advance(self, n_steps=1):
        """Up to n_steps control steps with both robots' current commands; stops at the end of a point."""
        n0, n = len(self.points), 0
        while n < n_steps and not self.done and len(self.points) == n0:
            self.step(self._arm_target("A"), self._arm_target("B")); n += 1
        return n, self.points[n0:]

    def plan_snapshot(self, s) -> dict:
        """What the stroke planner needs from robot s at the step an act is accepted (its own frame; no ball)."""
        d, sd = self.d, self.sides[s]
        snap = {"k": self.k, "qpos": d.qpos[self.qa[s]].copy(), "qvel": d.qvel[self.va[s]].copy(),
                "qacc_warmstart": d.qacc_warmstart[self.va[s]].copy(), "ctrl": d.ctrl[self.aa[s]].copy(),
                "time": float(d.time), "target": self.target[s].copy(), "tail": [a.copy() for a in self.actions[s][-2:]],
                "plan": [q.copy() for q in sd.plan[sd.plan_j:]],
                "base_goal": None if sd.base_goal is None else sd.base_goal.copy(), "q_hit_active": sd.q_hit_active,
                "launch": int(self.rs["played_k"]),
                "seed": int(SEED * 1000), "stream": 1 if s == FIRST_SERVER else 2,   # one stream per robot (revision 7)
                "n_our_legs": int(sd.n_snapshots)}
        sd.n_snapshots += 1
        return snap

    def rt_install(self, s, res):
        if res["k_effect"] != self.k or not np.array_equal(res["target_at_effect"], self.target[s]):
            raise RuntimeError(f"{s}: plan for step {res['k_effect']} installed at step {self.k} (or its predicted joint "
                               "targets differ from the live ones)")
        sd = self.sides[s]
        if res["plan"]:
            sd.plan, sd.plan_j = list(res["plan"]), 0
            sd.q_hit_active = res["q_hit"]
            sd.last_act = res["info"]
        return res["info"]

    def rt_move(self, s, bx, by):
        sd = self.sides[s]
        sd.base_goal = np.clip(np.array([bx, by], float), tn.BASE_LO, tn.BASE_HI)
        return [round(float(x), 3) for x in sd.base_goal]

    def rt_clear(self, s):
        sd = self.sides[s]
        sd.plan, sd.plan_j, sd.last_act, sd.q_hit_active = [], 0, None, None

    # ------------------------------------------------------------------ observations
    def ensure_frame(self):
        if not self.hist:
            self.hist.append({"k": self.k, "t": self.t_since_played(), "qpos": self.d.qpos.copy(), "b": self.b_p,
                              "leg_id": self.leg_id})

    def side_state(self, s):
        """What player s knows without looking, in its own frame (the serve fields of the cross-sport convention)."""
        d, rs = self.d, self.rs
        other = OTHER[s]
        c = own(s, d.site_xpos[self.site[s]])
        n = own(s, d.site_xmat[self.site[s]].reshape(3, 3)[:, 0])
        ph = rs["phase"]
        heard = []
        if ph != "serve":
            first = ("you tossed" if rs["server"] == s else "the opponent tossed") if ph == "toss" else \
                    ("you hit the ball" if rs["hitter"] == s else "the opponent hit the ball")
            heard.append({"event": first, "t": 0.0})
            for e in self.events:
                if e["kind"] == "bounce" and e["k"] >= rs["played_k"]:
                    heard.append({"event": "bounce on your side" if e["half"] == s else "bounce on the opponent's side",
                                  "t": round((e["k"] - rs["played_k"]) * CONTROL_DT, 2)})
        live = ph in ("flight", "rally") and self.b_live and rs["state"] == "live"
        tz = rs["toss"]
        serve = {"phase": ("open" if self.k >= rs["open_k"] else "waiting") if ph == "serve" else
                 ("tossed" if ph == "toss" else "done"),
                 "court": rs["half"], "serve_number": rs["faults"] + 1}
        if rs["server"] == s and ph == "serve":
            serve["opens_in_s"] = round(max(0.0, (rs["open_k"] - self.k) * CONTROL_DT), 3)
            serve["deadline_in_s"] = round(max(0.0, (rs["clock_k"] - self.k) * CONTROL_DT), 3)
        if self._toss_pending is not None:
            serve["toss_takes_effect_at_step"] = int(self._toss_pending[3])
        if tz is not None:
            serve["tossed_at_step"] = int(tz["k"])
            serve["toss"] = {"pos": [round(float(x), 4) for x in own(s, tz["pos"])], "height": tz["height"]}
        return {"score": {"you": self.score[s], "opponent": self.score[other]}, "point": len(self.points) + 1,
                "you_serve": rs["server"] == s, "you_receive_serve": rs["receiver"] == s, "serve": serve,
                "ball": {"serve": "not yet served", "toss": "toss", "flight": "serve"}.get(ph) or
                ("your shot" if rs["hitter"] == s else "opponent's shot"),
                "ball_in_play": live,
                "ball_state": ("not yet served" if ph == "serve" else
                               (("tossed: yours to hit (your serve)" if rs["server"] == s else
                                 "the opponent tossed (serving)") if ph == "toss" else
                                ("coming to you" if live and rs["expect"] == s else
                                 ("going to the opponent" if live else "dead")))),
                "t_since_played": self.t_since_played() if ph != "serve" else None, "heard": heard,
                "robot_base_xy": [round(float(x), 3) for x in d.qpos[self.qa[s][:2]]],
                "q_arm": [round(float(x), 4) for x in d.qpos[self.qa[s][2:]]],
                "racket_pos": [round(float(x), 3) for x in c], "racket_normal": [round(float(x), 3) for x in n]}

    def cameras(self, s):
        cams = []
        for lookat, dist, az, el in self.CAMS:
            c = mj.MjvCamera()
            c.lookat[:] = lookat if s == "A" else mir(lookat)
            c.distance = dist; c.azimuth = az if s == "A" else az + 180.0; c.elevation = el
            cams.append(c)
        return cams


def render_video(qA, qB, out, fps=25, every=2):
    """Replay video: a high side view of the whole court, 2x slow motion, with the score."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    du = Duel()
    r = mj.Renderer(du.m, height=360, width=640)
    rd = mj.MjData(du.m)
    c = mj.MjvCamera(); c.lookat[:] = [0.0, 0.0, 0.5]; c.distance = 30.0; c.azimuth = 0.0; c.elevation = -30.0
    w = imageio.get_writer(out, fps=fps, macro_block_size=8)
    for k, (a, b) in enumerate(zip(qA, qB)):
        du.step(a, b)
        if k % (every if du.b_live else 3 * every):
            continue
        rd.qpos[:] = du.d.qpos; rd.mocap_pos[0] = du.b_p
        mj.mj_kinematics(du.m, rd)
        r.update_scene(rd, camera=c)
        im = Image.fromarray(r.render())
        ImageDraw.Draw(im).text((8, 6), f"B (left) {du.score['B']} : {du.score['A']} A (right)", fill=(255, 255, 255))
        w.append_data(np.asarray(im))
        if du.done:
            break
    w.close(); r.close()
