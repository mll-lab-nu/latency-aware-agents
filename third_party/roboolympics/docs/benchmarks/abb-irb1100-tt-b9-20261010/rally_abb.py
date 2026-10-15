"""A minimal reference player on the ABB IRB 1100 + gantry, and real rallies between two of them.

The world: the sport's ball, table, net and contact pairs (the ball-only scene of sports/tabletennis), two robots of
the library (bodies/abb_irb1100, paddle on the flange), seat A behind the y < 0 end line, seat B mirrored behind the
y > 0 end; the robots' geoms touch the ball, the table and the net; self-collision pairs on (bodies/selfcollision.py);
every robot driven only through its rig (bodies/servo.py: the real limits of body.toml).

The player sees only what the real system sees (body.toml [perception], [controller.latency]):
  ball      its POSITION from the stereo cameras: one sample every CAM_DT, delivered CAM_LAT later, with Gaussian noise
            CAM_NOISE per axis; no velocity, no spin, no contact events (it detects bounces and hits itself)
  robot     its own joint positions (encoders) and the targets it has sent
  commands  100 Hz target rows; a row computed at time t takes effect ACT_LAT later
Per incoming ball (its own frame: the ball comes towards -y; seat B's frame is seat A's turned by 180 degrees):
  1. estimate   after the last hit / bounce it detects, a least-squares fit of the ball's position, velocity and spin
                to the samples on the sport's ball model (the ball-only MuJoCo world: drag, Magnus, the table's bounce)
  2. contact    on the predicted flight, after its own bounce, contact points every 20 ms; the paddle state for an aim on
                the other half from the contact law with two landing corrections in the contact lab (lib.lab_stroke:
                on the ESTIMATED flight, so a wrong spin estimate costs a miss); IK over the gantry and the arm
                (abbrig.AbbGantry.ik), the gantry's S-curve in time, the arm's limits-aware stroke (bodies/trajopt)
                from the state its commands will have when the new rows take effect
  3. re-plan    once the bounce has been seen (the estimate from the post-bounce samples): the stroke re-planned with
                the gantry's goal kept
  4. recover    after the follow-through, the gantry and the arm return to the ready stance (rest-to-rest S-curves)
A rally starts with a library serve (its tip state) from seat B to seat A; then A and B alternate. A return counts
when after the robot's blade contact the ball's next table contact is on the other half (no robot contact, no
double hit). Recorded: returns per rally, blade impacts (peak force, impulse), self-collision records, robot-table
contacts.

    run.sh rally_abb.py OUT_PREFIX       env NPROC, N_RALLIES (24), SEEDS (first seed), RALLY_SERVES (long | all),
                                         CAM_DT, CAM_LAT, CAM_NOISE, ACT_LAT (default: body.toml, the real system)"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("RECV_DIR", "/recv") + "/prototypes/tabletennis_receive")
import lib  # noqa: E402
from lib import mj  # noqa: E402
import oracle as O  # noqa: E402
import bodyrig as BR  # noqa: E402
from bodyrig import trajopt, AL  # noqa: E402
import abbrig  # noqa: E402
from bodies import load_body  # noqa: E402  (abbrig put the library first on sys.path)

B = load_body("abb_irb1100")
LAT = B.decl["controller"]["latency"]
PER = B.decl["perception"]
CAM_DT = float(os.environ.get("CAM_DT", PER["ball_sample_s"]))
CAM_LAT = float(os.environ.get("CAM_LAT", LAT["ball_observation_s"]))
CAM_NOISE = float(os.environ.get("CAM_NOISE", PER["ball_noise_m"]))
ACT_LAT = float(os.environ.get("ACT_LAT", LAT["command_s"]))
REHEARSE = int(os.environ.get("REHEARSE", "2"))
EVERY = 4                       # re-plan at most every 40 ms
COMMIT = 15                     # no re-plan in the last 150 ms before the contact (the stroke is under way)
DRIFT = 0.015                   # re-plan when the predicted ball at the planned contact moved this far (m)
DT = 0.01
TABLE_Y = lib.TABLE_Y
AIMS = [(0.0, 0.85), (-0.30, 0.85), (0.30, 0.85)]
VYS = (4.0, 3.4)
MIR = np.array([-1.0, -1.0, 1.0])


# ------------------------------------------------------------------------------------------------ the world
def world_xml():
    root = ET.fromstring(lib._lab_xml(False))
    infos = {}
    my = abbrig.MOUNT["y"]
    for seat, (y, yaw, acc) in {"A": (-my, math.pi / 2, (0.1, 0.35, 0.85, 1)),
                                "B": (my, -math.pi / 2, (0.85, 0.15, 0.12, 1))}.items():
        infos[seat] = B.mount(root, f"{seat}_", (0.0, y, abbrig.FLOOR_Z), yaw, hand="paddle", accent=acc)
        ET.SubElement(root.find("contact"), "pair", geom1="geom_ball", geom2=infos[seat]["contact_geoms"][0],
                      solref="-600000 -12", friction="0.8 0.8 0.005 0.0001 0.0001", solimp="0.98 0.99 0.001 0.5 2",
                      solreffriction="-0.0 -200.0")
    for g in root.iter("geom"):
        if g.get("name") in ("geom_ball", "geom_table", "geom_net"):
            g.set("conaffinity", "5")
    for f in root.iter("flag"):
        assert f.get("contact") != "disable"
    return ET.tostring(root, encoding="unicode"), infos


# ------------------------------------------------------------------------------------------------ estimation
class Estimator:
    """least squares of (p0, v0, w0) at the first sample of a segment on the ball-only model"""

    def __init__(self):
        self.bs = lib.BallSim(with_paddle=True)

    def rollout(self, x, n):
        P, V, W, ev = self.bs.rollout(x[0:3], x[3:6], x[6:9], n)
        return P, V, W, ev

    def fit(self, ts, ps, w_prior=20.0):
        ts, ps = np.asarray(ts), np.asarray(ps)
        t0 = ts[0]
        k = np.round((ts - t0) / 0.001).astype(int)
        n = int(k[-1]) + 1
        # a quadratic fit (gravity) as the start
        A = np.c_[np.ones_like(ts), ts - t0]
        g = np.array([0, 0, -9.81])
        rhs = ps - 0.5 * np.outer((ts - t0) ** 2, g)
        c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        x = np.r_[c[0], c[1], np.zeros(3)]
        sig = max(CAM_NOISE, 1e-3)

        def res(x):
            P, *_ = self.rollout(x, n)
            if len(P) <= k[-1]:
                return None
            r = (P[k] - ps).ravel() / sig
            return np.r_[r, x[6:9] / w_prior]

        r0 = res(x)
        for _ in range(4):
            if r0 is None:
                break
            J = np.zeros((len(r0), 9))
            for j in range(9):
                dx = np.zeros(9); dx[j] = 1e-4 if j < 6 else 0.05
                r1 = res(x + dx)
                if r1 is None:
                    return x
                J[:, j] = (r1 - r0) / dx[j]
            step = np.linalg.lstsq(J, -r0, rcond=None)[0]
            x2 = x + step
            r2 = res(x2)
            if r2 is None or r2 @ r2 > r0 @ r0:
                break
            x, r0 = x2, r2
        return x


# ------------------------------------------------------------------------------------------------ the player
class Player:
    def __init__(self, seat, rig, kin: abbrig.AbbGantry, est: Estimator, rng):
        self.seat, self.rig, self.kin, self.est, self.rng = seat, rig, kin, est, rng
        self.F = np.ones(3) if seat == "A" else MIR             # world -> own frame
        self.stance = np.asarray(kin.readies[os.environ.get("STANCE", "near")], float)
        self.u_ready = rig.rest.copy()
        self.u_ready[0:2] = self.stance
        self.u_ready[5:11] = kin.q_ready
        self.u_start = self.u_ready.copy()                      # where the robot stands before its first command
        self.cmd = {}                                           # control step -> target u
        self.twin = None                                        # (model, data, rig) of the robot-only twin
        self.live = None                                        # the measured joint state (encoders)
        self.samples = []                                       # (t, p own frame) delivered
        self.reset_ball()

    def reset_ball(self):
        self.seg = []                     # samples since the last detected event
        self.phase = "wait"               # wait -> planned -> replanned -> swing -> recover
        self.contact_k = None
        self.b_goal = None
        self.after_bounce = False
        self.k_plan = -10 ** 9
        self.log = {}

    # observations ------------------------------------------------------------
    def observe(self, t, p_world):
        p = p_world * self.F + self.rng.normal(0.0, CAM_NOISE, 3)
        self.samples.append((t, p))
        s = self.seg
        s.append((t, p))
        if len(s) >= 3:
            (t0, p0), (t1, p1), (t2, p2) = s[-3], s[-2], s[-1]
            vy0, vy1 = (p1[1] - p0[1]) / (t1 - t0), (p2[1] - p1[1]) / (t2 - t1)
            vz0, vz1 = (p1[2] - p0[2]) / (t1 - t0), (p2[2] - p1[2]) / (t2 - t1)
            z = [q[2] for _, q in s[-5:]]
            if len(z) == 5 and z[2] == min(z) and z[2] < 0.07 and z[0] - z[2] > 0.01 and z[4] - z[2] > 0.005:
                own = s[-3][1][1] < 0                     # a bounce 2 samples ago: the next fit starts after it
                self.seg = s[-2:]
                if own and self.phase in ("planned", "wait"):
                    self.after_bounce = True
                    self.log.setdefault("bounce_seen", round(s[-3][0], 3))
                return
            y = [q[1] for _, q in s[-5:]]
            if len(y) == 5 and (y[2] == max(y) and y[2] - y[0] > 0.01 and y[2] - y[4] > 0.01 or
                                y[2] == min(y) and y[0] - y[2] > 0.01 and y[4] - y[2] > 0.01):
                incoming = y[2] == max(y)                 # a hit 2 samples ago: the ball turned round (towards
                if incoming:                              # this player: the other side's return)
                    self.reset_ball()
                self.seg = s[-2:]

    # commands ----------------------------------------------------------------
    def target(self, k):
        while k not in self.cmd:
            ks = [x for x in self.cmd if x < k]
            if not ks:
                return self.u_start.copy()
            return self.cmd[max(ks)].copy()
        return self.cmd[k].copy()

    def state_at(self, k):
        """the commanded (u, ud, udd) at step k (targets of steps k, k-1, k-2)"""
        u0, u1, u2 = self.target(k), self.target(k - 1), self.target(k - 2)
        return u0, (u0 - u1) / DT, (u0 - 2 * u1 + u2) / DT ** 2

    def write(self, k0, rows):
        for x in [x for x in self.cmd if x >= k0]:
            del self.cmd[x]
        for i, u in enumerate(rows):
            self.cmd[k0 + i] = u

    # decisions ---------------------------------------------------------------
    def decide(self, k_now, t_now):
        """called every control step; plans / re-plans the stroke"""
        moving_in = len(self.seg) >= 2 and self.seg[-1][1][1] < self.seg[0][1][1] - 0.02
        if self.phase == "wait" and moving_in and len(self.seg) >= int(os.environ.get("N_FIRST", "10")) \
                and self.seg[-1][1][1] > -0.2:
            self.plan(k_now, t_now, replan=False)
        elif self.phase in ("planned", "replanned") and self.contact_k is not None and \
                self.contact_k - k_now > COMMIT and k_now - self.k_plan >= EVERY and \
                len(self.seg) >= int(os.environ.get("N_RE", "5")):
            self.plan(k_now, t_now, replan=True)  # re-plan every EVERY steps on the newer estimate (if it moved)
        elif self.phase in ("planned", "replanned", "swing") and self.contact_k is not None and \
                k_now > self.contact_k + 30:
            self.recover(k_now)

    def plan(self, k_now, t_now, replan):
        ts = [t for t, _ in self.seg]
        ps = [p for _, p in self.seg]
        x = self.est.fit(ts, ps)
        t_fit = ts[0]
        n_pred = 1400
        P, V, W, ev = self.est.rollout(x, n_pred)
        k_s = k_now + int(math.ceil(ACT_LAT / DT - 1e-9))       # the first step the new rows can take effect
        t_s = k_s * DT
        # the predicted bounce on the own half
        tabs = [e for e in ev if e[1] == "table_rcv"]
        if self.after_bounce:                     # the fit starts after the own bounce: contacts from its start
            i_b = -15
            tabs = []
        elif not tabs:
            self.log.setdefault("why", "no own bounce predicted")
            return
        else:
            i_b = tabs[0][0]
        after = [e for e in ev if e[0] > max(i_b, 0) + 2]
        i_end = after[0][0] if after else len(P) - 1
        flight = dict(P=P, V=V, W=W)
        self.k_plan = k_now
        if replan and self.contact_k is not None:  # keep the plan when the estimate has not moved
            i_old = int(round(((self.contact_k + 1) * DT - t_fit) / 0.001))
            prev = self.log.get("replan", self.log.get("plan", {})).get("ball_pred")
            if prev is not None and 0 <= i_old < len(P) and np.linalg.norm(P[i_old] - prev) < DRIFT:
                return
        u_s, ud_s, udd_s = self.state_at(k_s - 1)
        b_now = u_s[0:2]
        # a re-plan moves the gantry again only from rest (its rest-to-rest S-curves); while it moves, it keeps going
        hold_gantry = replan and self.b_goal is not None and float(np.max(np.abs(ud_s[0:2]))) > 1e-6
        q0, v0, a0 = u_s[5:11], ud_s[5:11], udd_s[5:11]
        kin = self.kin
        tried = 0
        why = {}
        cands = []
        for i in range(max(i_b + 15, 35), i_end - 3, 10):         # the contact lab starts 30 ms before
            px, py, pz = P[i]
            if pz < 0.06 or py > -0.15:
                continue
            t_c = t_fit + i * 0.001
            n = int(math.floor((t_c - t_s) / DT + 1e-9))
            if n < 6:
                continue
            late = 0.0 if replan else 2.0 * max(0.0, 0.20 - (i - i_b) * 0.001)   # time to re-plan after the bounce
            cands.append((abs(pz - 0.28) + 2.0 * max(0.0, py + 1.0) + late, i, n))   # in the shoulder's reach first
        cands.sort()
        if replan and self.contact_k is not None:              # keep near the planned contact time
            cands.sort(key=lambda c: abs(t_fit + c[1] * 0.001 - self.contact_k * DT))
        for _, i, n in cands[:16]:
            for aim in AIMS:
                for vy in VYS:
                    tried += 1
                    res = lib.lab_stroke(self.est.bs, flight, i, aim, vy)
                    if res is None or res[3] != "in":
                        why["lab"] = why.get("lab", 0) + 1
                        continue
                    p_c, nrm, v_p = (np.asarray(r, float) for r in res[:3])
                    if hold_gantry:                        # the gantry is still on its way: keep its rows
                        b, q, err = kin.ik(p_c, nrm, self.b_goal, self.q_goal, fix_base=True)
                        if err > 2e-3 or kin.collides(b, q):
                            k_ = "ik" if err > 2e-3 else "collide"
                            why[k_] = why.get(k_, 0) + 1
                            continue
                    else:                                  # the oracle's IK: seeds, two gantry weights, no collision
                        sols, w = O.solutions(kin, {"p": p_c.tolist(), "n": nrm.tolist()})
                        O._SOL.clear()
                        if not sols:
                            why[w] = why.get(w, 0) + 1
                            continue
                        b, q = min(sols, key=lambda bq: kin.base_time(b_now, bq[0]))
                    Tb = kin.base_time(b_now, b)
                    if Tb > n * DT:
                        why["gantry"] = why.get("gantry", 0) + 1
                        continue
                    qd = kin.contact_qd(b, q, v_p)
                    if np.any(np.abs(qd) > 0.95 * kin.lim.vel):
                        why["speed"] = why.get("speed", 0) + 1
                        continue
                    rows, pl = self.stroke_rows(k_s, q0, v0, a0, n, q, qd, b_now, b, Tb, hold_gantry)
                    if rows is None:
                        why["stroke"] = why.get("stroke", 0) + 1
                        continue
                    for _ in range(REHEARSE):         # the twin (same servo, from the live state) corrects the lag
                        qn, vn = self.rehearse(k_now, k_s, rows, n)
                        e_q, e_v = q - qn, qd - vn
                        if np.max(np.abs(e_q)) < 2e-3 and np.max(np.abs(e_v)) < 0.02:
                            break
                        r2, pl2 = self.stroke_rows(k_s, q0, v0, a0, n, np.clip(pl["goal_q"] + e_q,
                                                   kin.lim.range_lo + 1e-4, kin.lim.range_hi - 1e-4),
                                                   pl["goal_qd"] + e_v, b_now, b, Tb, hold_gantry)
                        if r2 is None:
                            break
                        rows, pl = r2, pl2
                    self.write(k_s, rows)
                    self.contact_k = k_s + n - 1
                    self.b_goal, self.q_goal = b, q
                    self.phase = "replanned" if replan else "planned"
                    self.log["replan" if replan else "plan"] = dict(t=round(t_now, 3), contact_t=round((k_s + n) * DT, 3),
                                                                     ball_pred=np.round(P[i], 3).tolist(),
                                                                     paddle_plan=np.round(p_c, 3).tolist(),
                                                                     aim=aim, vy=vy, tried=tried,
                                                                     spin=np.round(x[6:9], 1).tolist())
                    return
        self.log.setdefault("fails", []).append(dict(t=round(t_now, 3), replan=replan, n_cands=len(cands), **why))
        if replan:
            self.phase = "replanned"

    def stroke_rows(self, k_s, q0, v0, a0, n, q, qd, b_now, b, Tb, hold_gantry):
        """the 10 ms rows from step k_s: the arm's limits-aware stroke to (q, qd) at row n (bodies/trajopt), the
        gantry's S-curve to b (a re-plan keeps the gantry's rows in force: the IK held the gantry at its goal)"""
        kin = self.kin
        pl = trajopt.stroke(q0, v0, a0, n, DT, kin.lim, q, qd, n_follow=25,
                            lo=kin.lim.range_lo + 1e-4, hi=kin.lim.range_hi - 1e-4)
        if pl["contact_err_q"] > 0.01 or pl["viol"] > 1.02:
            return None, None
        if O.path_collides(kin, b_now, b, Tb, pl, n):
            return None, None
        rows = []
        for kk in range(len(pl["q"])):
            if hold_gantry:
                bb = self.target(k_s + kk)[0:2] if (k_s + kk) in self.cmd else np.asarray(b, float)
            else:
                bb = kin.base_at(b_now, b, Tb, (kk + 1) * DT) if Tb > 0 else np.asarray(b, float)
            u = self.u_ready.copy(); u[0:2] = bb; u[5:11] = pl["q"][kk]
            rows.append(u)
        pl["goal_q"], pl["goal_qd"] = np.asarray(q, float), np.asarray(qd, float)
        return rows, pl

    def rehearse(self, k_now, k_s, rows, n):
        """the robot-only twin from the live joint state and the live servo state (the controller's own), running
        the targets in force until k_s and the new rows: the arm's (q, qd) at the contact row"""
        tw, td, tr = self.twin
        r = self.rig
        import mujoco as mj_
        td.qpos[tr.qadr] = self.live[0]; td.qvel[tr.vadr] = self.live[1]
        td.qacc[:] = 0.0
        mj_.mj_forward(tw, td)
        tr.servo.reset(r.servo.x, r.servo.xd)
        if r.servo.jl is not None:
            tr.servo.jl.q, tr.servo.jl.qd, tr.servo.jl.qdd = (r.servo.jl.q.copy(), r.servo.jl.qd.copy(),
                                                              r.servo.jl.qdd.copy())
        tr.servo.act.tau = r.servo.act.tau.copy()
        seq = [self.target(k) for k in range(k_now - 1, k_s)] + rows[:n]
        for a, b in zip(seq[:-1], seq[1:]):
            for j in range(1, tr.substeps + 1):
                tr.apply(td, a, b, j)
                mj_.mj_step(tw, td)
        return td.qpos[tr.qadr[2:]].copy(), td.qvel[tr.vadr[2:]].copy()

    def recover(self, k_now, k_s=None):
        k_s = k_now + int(math.ceil(ACT_LAT / DT - 1e-9)) if k_s is None else k_s
        u_s, ud_s, _ = self.state_at(k_s - 1)
        if np.max(np.abs(ud_s)) > 1e-3:                            # the follow-through has not ended yet
            return
        kin = self.kin
        lim = AL.ArmLimits("abb", "abb", np.r_[kin.b_lo, kin.lim.range_lo], np.r_[kin.b_hi, kin.lim.range_hi],
                           np.r_[kin.gV, kin.lim.vel] * 0.95, np.r_[kin.gA, kin.lim.acc] * 0.95,
                           np.r_[kin.gJ, kin.lim.jerk] * 0.95, np.ones(8), None)
        a = np.r_[u_s[0:2], u_s[5:11]]
        b = np.r_[self.u_ready[0:2], self.u_ready[5:11]]
        rows = []
        for s in AL.plan_line(a, b, lim).samples(DT):
            u = self.u_ready.copy(); u[0:2] = s[0:2]; u[5:11] = s[2:]
            rows.append(u)
        self.write(k_s, rows)
        self.phase = "recover"


# ------------------------------------------------------------------------------------------------ a rally
class Rally:
    def __init__(self):
        xml, self.infos = world_xml()
        self.m = mj.MjModel.from_xml_string(xml)
        self.d = mj.MjData(self.m)
        assert self.m.opt.disableflags & mj.mjtDisableBit.mjDSBL_CONTACT == 0
        self.rigs = {s: B.rig(self.m, f"{s}_", hand="paddle") for s in "AB"}
        g = lambda n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, n)
        self.g_ball, self.g_table, self.g_net, self.g_floor = g("geom_ball"), g("geom_table"), g("geom_net"), g("floor")
        self.g_blade = {s: g(self.infos[s]["contact_geoms"][0]) for s in "AB"}
        self.g_robot = {}
        for s in "AB":
            base = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, f"{s}_base")
            self.g_robot[s] = {i for i in range(self.m.ngeom) if self._under(self.m.geom_bodyid[i], base)}
        self.kin = abbrig.AbbGantry()
        self.est = Estimator()
        tm = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle"))
        self.twin = (tm, mj.MjData(tm), B.rig(tm, "", hand="paddle"))

    def _under(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def play(self, tip, seed, t_max=12.0):
        m, d = self.m, self.d
        rng = np.random.default_rng(seed)
        mj.mj_resetData(m, d)
        players = {s: Player(s, self.rigs[s], self.kin, self.est, np.random.default_rng(seed * 10 + i))
                   for i, s in enumerate("AB")}
        for pl in players.values():
            pl.twin = self.twin
        from bodies import selfcollision as SC
        # the server (B) has just served: its gantry stands back 0.8 m (out of the served ball's way: the serves of
        # the library were struck by another robot's paddle where B's would stand) and returns to its stance
        players["B"].u_start[0] = max(float(self.kin.b_lo[0]), players["B"].u_start[0] - 0.8)
        players["B"].recover(0, k_s=0)
        players["B"].phase = "wait"
        for s, r in self.rigs.items():
            r.selfcol = SC.Monitor(m, f"{s}_selfcol", B.decl["self_collision"].get("warn_force_n"))
            r.place(d, players[s].u_start)
        p, v, w = (np.asarray(tip[k], float) * lib.MIR for k in ("pos", "vel", "spin"))   # B serves to A
        d.qpos[0:3] = p; d.qpos[3:7] = [1, 0, 0, 0]; d.qvel[0:3] = v; d.qvel[3:6] = w
        mj.mj_forward(m, d)
        f6 = np.zeros(6)
        hits, impacts, events = [], [], []
        last_hitter, bounced_since = "B", set()
        expect = "A"                       # who must hit next
        own_bounces = 0
        prev_c = set()
        next_cam = 0.0
        pending = []                       # (deliver_time, sample_time, pos)
        result = None
        k = 0
        cur_imp = {}
        robot_table = 0.0
        while d.time < t_max and result is None:
            t = d.time
            for s, pl in players.items():
                rr = self.rigs[s]
                pl.live = (d.qpos[rr.qadr].copy(), d.qvel[rr.vadr].copy())
                pl.decide(k, t)
            for s, pl in players.items():          # the truth at the planned contact step (own frame): diagnostics
                if pl.contact_k is not None and k == pl.contact_k + 1 and "at_contact" not in pl.log:
                    pl.log["at_contact"] = dict(ball=np.round(d.qpos[0:3] * pl.F, 3).tolist(),
                                                paddle=np.round(self.rigs[s].tool(d)[0] * pl.F, 3).tolist())
            prev_u = {s: players[s].target(k - 1) for s in "AB"}
            next_u = {s: players[s].target(k) for s in "AB"}
            for j in range(1, 11):
                for s, r in self.rigs.items():
                    r.apply(d, prev_u[s], next_u[s], j)
                mj.mj_step(m, d)
                if d.time + 1e-9 >= next_cam:
                    pending.append((d.time + CAM_LAT, d.time, d.qpos[0:3].copy()))
                    next_cam += CAM_DT
                while pending and pending[0][0] <= d.time + 1e-9:
                    _, ts, pp = pending.pop(0)
                    for pl in players.values():
                        pl.observe(ts, pp)
                cur = set()
                for i in range(d.ncon):
                    c = d.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    gs = {g1, g2}
                    if self.g_ball in gs:
                        o = (gs - {self.g_ball}).pop() if len(gs) == 2 else -1
                        for s in "AB":
                            if o == self.g_blade[s]:
                                cur.add("paddle_" + s)
                                mj.mj_contactForce(m, d, i, f6)
                                ci = cur_imp.setdefault(s, dict(peak_n=0.0, impulse_ns=0.0, ms=0))
                                ci["peak_n"] = max(ci["peak_n"], abs(f6[0])); ci["impulse_ns"] += abs(f6[0]) * 1e-3
                                ci["ms"] += 1
                            elif o in self.g_robot[s]:
                                cur.add("robot_" + s)
                        if o == self.g_table:
                            cur.add("table_" + ("B" if d.qpos[1] > 0 else "A"))
                        elif o == self.g_floor:
                            cur.add("floor")
                        elif o == self.g_net:
                            cur.add("net")
                    else:
                        for s in "AB":
                            if (g1 in self.g_robot[s] and g2 in (self.g_table, self.g_net)) or \
                                    (g2 in self.g_robot[s] and g1 in (self.g_table, self.g_net)):
                                mj.mj_contactForce(m, d, i, f6)
                                robot_table = max(robot_table, abs(f6[0]))
                for e in sorted(cur - prev_c):
                    events.append((round(d.time, 3), e))
                    if e.startswith("paddle_"):
                        s = e[-1]
                        if s != expect or own_bounces != 1:
                            result = f"{s} hit out of turn (bounces on its half {own_bounces})"
                        else:
                            hits.append(dict(seat=s, t=round(d.time, 3)))
                            last_hitter, expect, own_bounces = s, ("B" if s == "A" else "A"), 0
                    elif e.startswith("table_"):
                        side = e[-1]
                        if side == expect:
                            own_bounces += 1
                            if own_bounces >= 2:
                                result = f"{expect} missed (second bounce)"
                        elif hits:
                            result = f"{last_hitter}'s return landed on its own half"
                    elif e.startswith("robot_"):
                        result = f"the ball touched robot {e[-1]} (not the blade)"
                    elif e == "floor":
                        result = (f"{expect} missed" if own_bounces == 1 else
                                  f"{last_hitter}'s ball went out" if own_bounces == 0 else "floor")
                    if result:
                        break
                for s in [s for s in cur_imp if "paddle_" + s not in cur]:
                    impacts.append(dict(seat=s, **{k2: round(v2, 4) for k2, v2 in cur_imp.pop(s).items()}))
                prev_c = cur
                if result:
                    break
            k += 1
        returns = len(hits)
        sc = {s: {k2: r.self_collisions()[k2] for k2 in ("events", "warnings", "peak_force_n")}
              for s, r in self.rigs.items()}
        return dict(returns=returns, end=result or "time", hits=hits, impacts=impacts, self_collision=sc,
                    robot_table_n=round(robot_table, 1), events=events[:60],
                    plans={s: players[s].log for s in "AB"})


def main():
    out = sys.argv[1]
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    n = int(os.environ.get("N_RALLIES", "24"))
    seed0 = int(os.environ.get("SEEDS", "1"))
    abbrig.setup_readies()
    bs = lib.BallSim()
    LIB = lib.library(sorted(glob.glob(str(lib.OUT / "serve_ext_*.json"))), bs=bs)
    if os.environ.get("RALLY_SERVES", "long") == "long":   # the serves this body can receive (B9: not the short ones)
        LIB = [x for x in LIB if x[2]["cls"].get("length") != "short"]
    R = Rally()
    rows, t0 = [], time.time()
    rng = np.random.default_rng(seed0)
    picks = rng.choice(len(LIB), size=n, replace=False)
    for j, ix in enumerate(picks):
        if j % nparts != part:
            continue
        sid, row, s = LIB[int(ix)]
        r = R.play(row["tip"], seed0 + j)
        r.update(serve=sid, seed=seed0 + j)
        rows.append(r)
        print("rally", sid, "returns", r["returns"], "end:", r["end"], round(time.time() - t0), flush=True)
        Path(out).write_text(json.dumps(rows, default=float))


if __name__ == "__main__":
    main()
