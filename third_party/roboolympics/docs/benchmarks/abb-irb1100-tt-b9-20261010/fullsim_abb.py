"""The ABB IRB 1100 + gantry upper bound in the FULL simulation: the strokes the planner-level oracle accepts
(oracle_abb.py's evaluation), played in a MuJoCo world with the robot of the library (bodies/abb_irb1100: model,
paddle, self-collision pairs) driven ONLY through its rig (rig.apply -> bodies/servo.py: jerk-limited setpoints at
body.toml's limits, PD + gravity + inertia feed-forward, the output stage at the gantry forces and arm torques,
1 kHz physics, 100 Hz targets), the sport's ball, table, net and contact pairs; the robot's geoms touch the ball and
the table. Returned = after the blade's contact the ball's next table contact is on the server's half, with no ball
contact on the robot's other geoms and no double hit (the Franka study's criterion); robot-table contacts are
recorded with their peak force (summarize.py: the strict share without one above 20 N).

The plan per stroke is the oracle's (the gantry's rest-to-rest S-curve to the contact base, the arm's limits-aware
stroke of bodies/trajopt with the table avoidance), corrected by up to two rehearsals on a robot-only twin (the same
servo; the twin's delivered joint state at the contact step corrects the stroke's goal), as the Franka sport's planner
rehearses. Recorded per play: the blade's impact (peak normal force, impulse of the ball-blade contact), the robot's
self-collision record (rig.self_collisions()), robot-table contacts.

    run.sh fullsim_abb.py OUT_PREFIX        env NPROC, READY (default all), K (default 4), REACT, LEAD, ORACLE (the
                                            prefix of the oracle's results with the same settings: skip what it refused)"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("RECV_DIR", "/recv") + "/prototypes/tabletennis_receive")
import lib  # noqa: E402
from lib import mj  # noqa: E402
import oracle as O  # noqa: E402
import bodyrig as BR  # noqa: E402
import abbrig  # noqa: E402

abbrig.setup_readies()
O.reach_box = abbrig.reach_box
K = int(os.environ.get("K", "4"))
DT = 0.01
REHEARSE = int(os.environ.get("REHEARSE", "2"))


def plan_rows(body, b_ready, b_c, q_c, qd_c, n, v_p, p_c, nrm):
    """the oracle's arm stage plan (returning it): the stroke, the table-avoiding variant when the direct one cuts
    the table. Returns (pl, approach) or (None, why)."""
    Tb = body.base_time(b_ready, b_c)
    pl = O.plan_stroke(body, b_c, q_c, qd_c, n)
    if not O.path_collides(body, b_ready, b_c, Tb, pl, n):
        return pl, "direct"
    u = np.array([v_p[0], v_p[1], 0.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 0.1 else np.array([0, 1.0, 0])
    for dz in (0.10, 0.18):
        if n < 20:
            break
        bv, qv, rv = body.ik(p_c - 0.10 * u + np.array([0, 0, dz]), nrm, b_c, q_c, fix_base=True)
        if rv > 1e-3:
            continue
        cand = O.plan_stroke(body, b_c, q_c, qd_c, n, via=qv)
        if not O.path_collides(body, b_ready, b_c, Tb, cand, n):
            return cand, f"via +{dz:.2f}"
    return None, "path"


class World:
    def __init__(self, body):
        self.body = body
        xml, self.info = abbrig.world_xml()
        self.m = mj.MjModel.from_xml_string(xml)
        self.d = mj.MjData(self.m)
        self.rig = body.B.rig(self.m, "A_", hand="paddle")
        g = lambda n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, n)
        self.g_ball, self.g_table, self.g_net, self.g_floor = g("geom_ball"), g("geom_table"), g("geom_net"), g("floor")
        self.g_blade = g(self.info["contact_geoms"][0])
        base = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "A_base")
        self.g_robot = {i for i in range(self.m.ngeom) if self._under(self.m.geom_bodyid[i], base)}
        # the robot-only twin (rehearsals): the same body, servo and limits, no world
        import xml.etree.ElementTree as ET
        twin = ET.fromstring(body.B.robot_only_xml(hand="paddle"))
        abbrig.variant(twin, "")                  # the same what-ifs as the world (default: none)
        self.tm = mj.MjModel.from_xml_string(ET.tostring(twin, encoding="unicode"))
        self.td = mj.MjData(self.tm)
        self.trig = body.B.rig(self.tm, "", hand="paddle")
        assert self.m.opt.disableflags & mj.mjtDisableBit.mjDSBL_CONTACT == 0
        assert sum(1 for i in range(self.m.npair) if "selfcol" in (mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_PAIR, i) or "")) > 0

    def _under(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def u_of(self, b, q):
        u = self.rig.rest.copy()
        u[0:2] = b
        u[5:11] = q
        return u

    def rows(self, b_ready, b_c, pl, n, k_e):
        """the 10 ms target rows from the effect step on: the gantry's S-curve to b_c (within its limits), the arm's
        planned stroke, then the follow-through rows of the plan, then hold."""
        Tb = self.body.base_time(b_ready, b_c)
        out = []
        for k in range(len(pl["q"])):
            t = (k + 1) * DT
            b = self.body.base_at(b_ready, b_c, Tb, t) if Tb > 0 else np.asarray(b_c, float)
            out.append(self.u_of(b, pl["q"][k]))
        return out

    def rehearse(self, u0, rows, n):
        """the robot-only twin runs the rows from rest at u0: the arm's joint state at the contact step (row n)"""
        m, d, r = self.tm, self.td, self.trig
        mj.mj_resetData(m, d)
        r.place(d, u0); mj.mj_forward(m, d)
        prev = u0
        for k in range(n):
            for j in range(1, r.substeps + 1):
                r.apply(d, prev, rows[k], j)
                mj.mj_step(m, d)
            prev = rows[k]
        return d.qpos[r.qadr[2:]].copy(), d.qvel[r.vadr[2:]].copy()

    def play(self, tip, b_ready, b_c, q_c, qd_c, opt, t_e):
        body = self.body
        p_c, nrm, v_p = (np.asarray(opt[k], float) for k in ("p", "n", "v"))
        T = opt["t"] - t_e
        n = int(math.floor(T / DT + 1e-9))
        u_ready = self.u_of(b_ready, body.q_ready)
        goal_q, goal_qd = np.array(q_c, float), np.array(qd_c, float)
        info = {"rehearsals": []}
        pl = None
        for it in range(REHEARSE + 1):
            pl, approach = plan_rows(body, b_ready, b_c, goal_q, goal_qd, n, v_p, p_c, nrm)
            if pl is None:
                return False, "no collision-free stroke", info
            rows = self.rows(b_ready, b_c, pl, n, 0)
            if it == REHEARSE:
                break
            qn, vn = self.rehearse(u_ready, rows, n)
            e_q, e_v = q_c - qn, qd_c - vn
            info["rehearsals"].append(dict(err_q=round(float(np.max(np.abs(e_q))), 4),
                                           err_v=round(float(np.max(np.abs(e_v))), 3)))
            if np.max(np.abs(e_q)) < 2e-3 and np.max(np.abs(e_v)) < 0.02:
                break
            goal_q = np.clip(goal_q + e_q, body.lim.range_lo + 1e-4, body.lim.range_hi - 1e-4)
            goal_qd = goal_qd + e_v
        info["approach"] = approach
        m, d, r = self.m, self.d, self.rig
        mj.mj_resetData(m, d)
        from bodies import selfcollision as SC
        r.selfcol = SC.Monitor(m, "A_selfcol", r.B.decl["self_collision"].get("warn_force_n"))   # per play
        r.place(d, u_ready)
        p, v, w = (np.asarray(tip[k], float) * lib.MIR for k in ("pos", "vel", "spin"))
        d.qpos[0:3] = p; d.qpos[3:7] = [1, 0, 0, 0]; d.qvel[0:3] = v; d.qvel[3:6] = w
        mj.mj_forward(m, d)
        k_e = int(round((t_e - lib.TIP_DT) / DT))
        all_rows = [u_ready] * max(0, k_e) + rows
        prev = u_ready
        ev, prev_c = [], set()
        f6 = np.zeros(6)
        imp = dict(peak_n=0.0, impulse=0.0, steps=0)
        table_hit = 0.0
        tg = set()
        for k in range(len(all_rows) + 150):
            tgt = all_rows[k] if k < len(all_rows) else all_rows[-1]
            for j in range(1, r.substeps + 1):
                r.apply(d, prev, tgt, j)
                mj.mj_step(m, d)
                cur = set()
                for i in range(d.ncon):
                    c = d.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    gs = {g1, g2}
                    if self.g_ball in gs:
                        o = (gs - {self.g_ball}).pop() if len(gs) == 2 else -1
                        if o == self.g_blade:
                            cur.add("paddle")
                            mj.mj_contactForce(m, d, i, f6)
                            imp["peak_n"] = max(imp["peak_n"], abs(float(f6[0])))
                            imp["impulse"] += abs(float(f6[0])) * m.opt.timestep
                            imp["steps"] += 1
                        elif o == self.g_table:
                            cur.add("table_opp" if d.qpos[1] > 0 else "table_own")
                        elif o == self.g_net:
                            cur.add("net")
                        elif o == self.g_floor:
                            cur.add("floor")
                        elif o in self.g_robot:
                            cur.add("robot")
                    elif (g1 in self.g_robot and g2 in (self.g_table, self.g_net)) or \
                            (g2 in self.g_robot and g1 in (self.g_table, self.g_net)):
                        mj.mj_contactForce(m, d, i, f6)
                        table_hit = max(table_hit, abs(float(f6[0])))
                        rg = g1 if g1 in self.g_robot else g2
                        tg.add(mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, rg) or str(rg))
                for kind in cur - prev_c:
                    ev.append(kind)
                prev_c = cur
            prev = tgt
            info.update(impact=dict(peak_n=round(imp["peak_n"], 1), impulse_ns=round(imp["impulse"], 5),
                                    contact_ms=imp["steps"]), table_contact_n=round(table_hit, 1),
                        table_contact_geoms=sorted(tg))
            res = self._verdict(ev, table_hit)
            if res is not None:
                info["self_collision"] = {k: r.self_collisions()[k] for k in ("events", "warnings", "peak_force_n",
                                                                                   "worst_pairs")}
                return res[0], res[1], info
        info["self_collision"] = {k: r.self_collisions()[k] for k in ("events", "warnings", "peak_force_n",
                                                                                   "worst_pairs")}
        return False, "undecided", info

    @staticmethod
    def _verdict(ev, table_hit):
        """the Franka study's criterion (fullsim.py): a robot touching the table is recorded (table_contact_n), not a
        fault (ITTF 2.10.1.9 faults a player only when the playing surface moves); summarize.py also reports the
        strict share without any robot-table contact above the self-collision warning force"""
        if "paddle" in ev:
            after = ev[ev.index("paddle") + 1:]
            for kind in after:
                if kind == "table_opp":
                    return True, "returned onto the server's half"
                if kind in ("table_own", "floor", "robot"):
                    return False, f"after the paddle: {kind}"
                if kind == "paddle":
                    return False, "double hit"
            return None
        if "robot" in ev:
            return False, "the ball touched the robot (not the blade)"
        if ev.count("table_own") >= 2 or "floor" in ev:
            return False, "missed (second bounce / floor)"
        return None


def main():
    path = sys.argv[1]
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    body = abbrig.AbbGantry()
    bs = lib.BallSim()
    LIB = lib.library(sorted(glob.glob(str(lib.OUT / "serve_ext_*.json"))), bs=bs)
    L = {sid: (r, s) for sid, r, s in LIB}
    ST = O.load_strokes()
    ids = sorted(set(L) & set(ST))
    if os.environ.get("SEL"):
        sel = set(json.loads(Path(os.environ["SEL"]).read_text()))
        ids = [i for i in ids if i in sel]
    ids = ids[part::nparts]
    readies = body.readies
    if os.environ.get("READY"):
        readies = {k: v for k, v in readies.items() if k in os.environ["READY"].split(",")}
    t_e = O.REACT + O.LEAD
    skip = {}
    if os.environ.get("ORACLE"):              # the planner-level results of the same settings: a (serve, stance) the
        for f in sorted(glob.glob(str(Path(os.environ.get("ABB_OUT", "/out")) / f"{os.environ['ORACLE']}_*.json"))):
            for r in json.loads(Path(f).read_text()):          # oracle could not return cannot be returned here
                skip[r["id"]] = {k for k, v in r["per"].items() if v["stage"] != "ok"}
    W = World(body)
    rows, t0 = [], time.time()
    for sid in ids:
        row, serve = L[sid]
        st = ST[sid]
        per = {}
        for rname, b_ready in readies.items():
            b_ready = np.asarray(b_ready, float)
            tries, ok, n_chk = [], False, 0
            if rname in skip.get(sid, set()):
                per[rname] = {"returned": False, "tries": [], "checked": 0, "oracle": "no"}
                continue
            for o in O.rank(body, st["opts"], b_ready, t_e):
                if len(tries) >= K or n_chk >= 4 * O.N_EVAL:
                    break
                n_chk += 1
                stage, info = O.evaluate(body, bs, serve, o, b_ready, t_e)
                if stage != "ok":
                    continue
                b_c, q_c = np.asarray(info["base"], float), np.asarray(info["q"], float)
                qd_c = body.contact_qd(b_c, q_c, np.asarray(o["v"], float))
                ret, how, finfo = W.play(row["tip"], b_ready, b_c, q_c, qd_c, o, t_e)
                tries.append({"t": o["t"], "aim": o["aim"], "vy": o["vy"], "returned": ret, "how": how, **finfo})
                if ret:
                    ok = True
                    break
            per[rname] = {"returned": ok, "tries": tries, "checked": n_chk}
        good = [r for r, v in per.items() if v["returned"]]
        rows.append({"id": sid, "cls": st["cls"], "returnable": bool(good), "ok_ready": good, "per": per})
        print("full", sid, st["cls"], "->", good or {r: [t["how"][:30] for t in v["tries"]] for r, v in per.items()},
              round(time.time() - t0), flush=True)
        Path(path).write_text(json.dumps(rows, default=float))


if __name__ == "__main__":
    main()
