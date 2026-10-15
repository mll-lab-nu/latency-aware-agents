"""The skill-API contract of the air-hockey two-mode pilot: what a `strike` request asks for, how its execution is
measured, and the API test's tolerances (shared by the platform, the low-level deliverable and the evaluator).

A strike request {t, x, y, dx, dy, speed} (table frame, seconds of game time) asks: at time t the puck's centre is at
(x, y); the mallet hits it moving along the unit direction d = (dx, dy) at `speed` m/s. So at contact the mallet's
centre is at C = (x, y) - CONTACT d and its velocity is speed x d.

Measured on the first mallet-puck contact after the request (the contact instant is interpolated inside the 10 ms
control step from the mallet's and the puck's motion):
  time error         contact time - t
  position error     |mallet centre at contact - C|
  mallet speed / direction errors   |v| - speed, angle(v, d)
  puck speed / direction errors     the puck's velocity 30 ms after the contact against the IDEAL outcome: the same
                     puck (from the case's state) hit by a perfect mallet that moves along the line through C at
                     speed x d, passing C at time t (a kinematic mallet with the arm mallet's contact parameters)
  goal               the puck enters the opponent's goal (opponent parked away from its mouth) within GOAL_S of the
                     contact; own goal = it enters the own goal
Success = a contact, |time error| <= TOL["time_s"], position error <= TOL["position_m"], puck direction error <=
TOL["puck_direction_deg"], |puck speed error| <= TOL["puck_speed_abs"] + TOL["puck_speed_rel"] x the ideal speed, and
no refused command."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import world as W

H = W.H
CONTACT = H.CONTACT
TOL = {"time_s": 0.02, "position_m": 0.02, "puck_direction_deg": 8.0, "puck_speed_abs": 0.15, "puck_speed_rel": 0.15}
OUT_DT = 0.03            # the puck's outgoing velocity is read this long after the contact
GOAL_S = 2.5

CONTRACT = {
    "request": "strike {t, x, y, dx, dy, speed}: at game time t the puck centre is at (x, y) (table frame); hit it with "
               "the mallet moving along the unit direction (dx, dy) at `speed` m/s (0.3-2.5; faster than the arm "
               "can move on that line: as fast as it can). At contact the mallet centre is at (x, y) - CONTACT (dx, dy), "
               f"CONTACT = {CONTACT:.5f} m (puck + mallet radius)",
    "measured": "on the first mallet-puck contact after the request: time error, mallet position error (vs the contact "
                "point), mallet speed and direction errors, the puck's speed and direction 30 ms after the contact vs "
                "the ideal outcome (the same puck hit by a perfect kinematic mallet that passes the contact point at t "
                "with speed x d), goal / own goal within 2.5 s (the opponent parked away from its mouth)",
    "tolerances": TOL,
    "success": "a contact, |time error| <= 0.02 s, position error <= 0.02 m, puck direction error <= 8 deg, |puck speed "
               "error| <= 0.15 m/s + 15 % of the ideal speed, no refused command",
}


def angle_deg(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))))


# ------------------------------------------------------------------------------------------------ the ideal outcome
class Ideal:
    """The puck alone on the table (the game's walls, drag, contacts) and a kinematic mallet (mocap) with the arm
    mallet's contact parameters."""

    def __init__(self):
        root = ET.fromstring(H.scene_xml())
        wb = root.find("worldbody")
        for b in list(wb):
            if b.tag == "body" and b.get("name") != "puck_base":
                wb.remove(b)
        for tag in ("actuator", "default"):
            for e in root.findall(tag):
                root.remove(e)
        mb = ET.SubElement(wb, "body", name="mallet", mocap="true", pos="0 0 1")
        ET.SubElement(mb, "geom", name="mallet", type="cylinder", size=f"{H.MALLET_R} .03", pos="0 0 .014",
                      condim="4", solref=".02 .3", priority="2", friction="0 0 0")
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.pg, self.mg = self.m.geom("puck").id, self.m.geom("mallet").id

    def path(self, puck, puck_v, T):
        """the puck's centre every 10 ms for T s with nothing touching it"""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0], d.qpos[1] = puck
        d.qvel[0], d.qvel[1] = puck_v
        d.mocap_pos[0] = [0, 0, 1.0]
        out = []
        for i in range(int(round(T / H.PHYSICS_DT))):
            mujoco.mj_step(m, d)
            if (i + 1) % 10 == 0:
                out.append([float(d.qpos[0]), float(d.qpos[1])])
        return np.array(out).reshape(-1, 2)

    def run(self, puck, puck_v, req):
        """from the puck state at time 0 and the request: {contact_t, puck_v_out, goal (+1 / -1 / 0)}"""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0], d.qpos[1] = puck
        d.qvel[0], d.qvel[1] = puck_v
        u = np.array([req["dx"], req["dy"]], float)
        u /= max(1e-9, float(np.linalg.norm(u)))
        C = np.array([req["x"], req["y"]], float) - CONTACT * u
        v, tc = float(req["speed"]), float(req["t"])
        t_on = max(0.0, tc - 0.15)
        n = int(round((tc + GOAL_S + .5) / H.PHYSICS_DT))
        contact_t, out, goal, k_out = None, None, 0, None
        for i in range(n):
            t = (i + 1) * H.PHYSICS_DT
            if t >= t_on:
                p = C + u * v * (t - tc)
                d.mocap_pos[0] = [p[0], p[1], 0.0]      # the disc's centre at z = 0.014, as the arm's mallet (ee geom)
            else:
                d.mocap_pos[0] = [0, 0, 1.0]
            mujoco.mj_step(m, d)
            if contact_t is None:
                for j in range(d.ncon):
                    c = d.contact[j]
                    if {int(c.geom1), int(c.geom2)} == {self.pg, self.mg}:
                        contact_t = t
                        k_out = i + int(round(OUT_DT / H.PHYSICS_DT))
                        break
            if k_out is not None and i == k_out:
                out = [float(d.qvel[0]), float(d.qvel[1])]
                d.mocap_pos[0] = [0, 0, 1.0]
            if k_out is not None and i > k_out:
                d.mocap_pos[0] = [0, 0, 1.0]
                x, y = float(d.qpos[0]), float(d.qpos[1])
                if abs(x) > H.LENGTH / 2 + H.PUCK_R and abs(y) <= H.GOAL / 2 - H.PUCK_R:
                    goal = 1 if x > 0 else -1
                    break
        return {"contact_t": None if contact_t is None else round(contact_t, 4), "puck_v_out": out, "goal": goal}


# ------------------------------------------------------------------------------------------------ the measurement
def contact_instant(m0, m1, p0, w0):
    """the time (s, within one control step) at which a mallet moving from m0 to m1 meets a puck at p0 moving with w0"""
    vm = (m1 - m0) / W.DT
    r0, rv = p0 - m0, w0 - vm
    a, b, c = float(rv @ rv), 2 * float(r0 @ rv), float(r0 @ r0) - CONTACT**2
    if c <= 0:
        return 0.0, vm
    disc = b * b - 4 * a * c
    if a < 1e-12 or disc < 0:
        return W.DT / 2, vm
    tau = (-b - math.sqrt(disc)) / (2 * a)
    return float(min(max(tau, 0.0), W.DT)), vm


def errors(req, meas, ideal):
    """the contract's errors and success for one strike (meas from the game, ideal from Ideal.run)"""
    u = np.array([req["dx"], req["dy"]], float)
    u /= max(1e-9, float(np.linalg.norm(u)))
    C = np.array([req["x"], req["y"]], float) - CONTACT * u
    out = {"contact": meas.get("contact_t") is not None, "refusals": meas.get("refusals", 0),
           "goal": meas.get("goal", 0), "ideal_goal": ideal.get("goal", 0)}
    if not out["contact"]:
        out["success"] = False
        return out
    vm = np.array(meas["mallet_v"])
    out["time_err_s"] = round(meas["contact_t"] - float(req["t"]), 4)
    out["pos_err_m"] = round(float(np.linalg.norm(np.array(meas["mallet_pos"]) - C)), 4)
    out["mallet_speed_err"] = round(float(np.linalg.norm(vm)) - float(req["speed"]), 4)
    out["mallet_dir_err_deg"] = round(angle_deg(vm, u), 2)
    po, pi = meas.get("puck_v_out"), ideal.get("puck_v_out")
    if po is not None and pi is not None:
        out["puck_speed"] = round(float(np.linalg.norm(po)), 4)
        out["ideal_puck_speed"] = round(float(np.linalg.norm(pi)), 4)
        out["puck_speed_err"] = round(out["puck_speed"] - out["ideal_puck_speed"], 4)
        out["puck_dir_err_deg"] = round(angle_deg(po, pi), 2)
    ok = {"time": abs(out["time_err_s"]) <= TOL["time_s"], "pos": out["pos_err_m"] <= TOL["position_m"],
          "puck_dir": out.get("puck_dir_err_deg", 180) <= TOL["puck_direction_deg"],
          "puck_speed": po is not None and pi is not None and
          abs(out["puck_speed_err"]) <= TOL["puck_speed_abs"] + TOL["puck_speed_rel"] * out["ideal_puck_speed"],
          "no_refusal": out["refusals"] == 0}
    out["ok"] = ok
    out["success"] = all(ok.values())
    return out
