"""The ABB IRB 1100 on its two-axis gantry (bodies/abb_irb1100) in the serve-receivability study's kinematic API
(bodyrig.Body of branch tt-receive-study: b = the base, q = the arm, fk / jac / ik / collides / lim / base_time).

The robot is the library's body (load_body("abb_irb1100"): model, joint ranges, limits of body.toml), mounted in the
study's world (the sport's ball-only scene: table top z = 0, net y = 0, floor z = -0.76) behind the y < 0 end line,
facing +y. Its frame: b = (gantry_x, gantry_y) = the carriage's displacement from the gantry's home along the robot's
forward (world +y) and left (world -x) axes; q = axes 1..6.

The gantry axes are limited per axis (body.toml [limits.gantry]); the arm by [limits.arm]. Nothing is relaxed: the
same numbers as the library's servo enforces in the full simulation (fullsim_abb.py).
"""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

import lib
from lib import mj
import bodyrig as BR
from bodyrig import AL

REPO = os.environ.get("ABB_REPO", "/repo")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
for k in [k for k in sys.modules if k == "bodies" or k.startswith("bodies.")]:
    del sys.modules[k]                                   # the library's package, not tt-planner's bodies/
from bodies import load_body  # noqa: E402

FLOOR_Z = -0.76
MOUNT = dict(y=float(os.environ.get("ABB_MOUNT_Y", "2.00")))   # the gantry home behind the net (world y = -MOUNT y):
# the forward axis's front limit (+0.5 m) puts the arm's axis 1 at y = -1.50 (the Franka rail's front limit), its
# carriage plate 3 cm behind the table


def world_xml(robot=True, prefix="A_", ball_hits_robot=True, robot_hits_table=True):
    """The study's world (the sport's ball, table, net, floor and their contact pairs) with the gantry robot at its
    seat; the ball-blade pair of the sport (the same parameters as the Franka paddle). The robot's geoms touch the
    ball (a ball on the arm is a fault, not a pass-through) and the table and net (a stroke cannot cut the table)."""
    root = ET.fromstring(lib._lab_xml(False))
    info = None
    if robot:
        B = load_body("abb_irb1100")
        info = B.mount(root, prefix, (0.0, -MOUNT["y"], FLOOR_Z), math.pi / 2, hand="paddle")
        variant(root, prefix)
        con = root.find("contact")
        ET.SubElement(con, "pair", geom1="geom_ball", geom2=info["contact_geoms"][0], solref="-600000 -12",
                      friction="0.8 0.8 0.005 0.0001 0.0001", solimp="0.98 0.99 0.001 0.5 2",
                      solreffriction="-0.0 -200.0")
        for g in root.iter("geom"):
            n = g.get("name")
            if n == "geom_ball" and ball_hits_robot:
                g.set("conaffinity", "5")                 # the robot's geoms are contype 4
            if n in ("geom_table", "geom_net") and robot_hits_table:
                g.set("conaffinity", "5")
    for f in root.iter("flag"):                            # contacts must be on (a disabled flag kills <pair>s too)
        assert f.get("contact") != "disable", "contacts disabled in the scene"
    return ET.tostring(root, encoding="unicode"), info


PADDLE_EXT = float(os.environ.get("ABB_PADDLE", "0"))      # sensitivity: the blade centre this far beyond the flange
MOUNT_DROP = float(os.environ.get("ABB_DROP", "0"))        # sensitivity: the arm's base this far below the table top
SELF = os.environ.get("ABB_SELFCOL", "1") != "0"           # the planner-level checks include self-collision


def variant(root, prefix):
    """The study's what-ifs (default: none): DeepMind's paddle on its extension (S3: 18.8 cm; the blade centre
    about 0.26 m beyond the flange instead of the library paddle's 0.14 m), a lower mount (the arm's base below the
    table top: the gantry's height is not published)."""
    for b in root.iter("body"):
        if PADDLE_EXT and b.get("name") == prefix + "hand":
            for el in b:
                if el.tag in ("geom", "site") and el.get("pos"):
                    x, y, z = map(float, el.get("pos").split())
                    el.set("pos", f"{x} {y} {PADDLE_EXT}")
                if el.tag == "geom" and el.get("type") == "capsule":
                    el.set("fromto", f"0 0 0 0 0 {PADDLE_EXT - 0.08}")
        if MOUNT_DROP and b.get("name") == prefix + "link_0":
            x, y, z = map(float, b.get("pos").split())
            b.set("pos", f"{x} {y} {z - MOUNT_DROP}")


class AbbGantry(BR.FrankaRail):
    """isinstance FrankaRail: the oracle's per-axis S-curve base model (reach_box, base_time, base_at) applies, with
    this gantry's per-axis limits (bV, bA, bJ: the slower axis where they differ; base_time / reach use each axis)."""
    name = "abb_gantry"
    nb = 2
    fixed = False

    def __init__(self):
        xml, info = world_xml()
        self.m = mj.MjModel.from_xml_string(xml)
        self.k = mj.MjData(self.m)
        self.k.qpos[0:3] = [3.0, 3.0, 5.0]
        self.B = load_body("abb_irb1100")
        self.rig = rig = self.B.rig(self.m, "A_", hand="paddle")
        self.site = rig.tool_site
        self.qadr, self.vadr = rig.qadr, rig.vadr
        self.Ju = np.zeros((self.m.nv, 8)); self.Ju[self.vadr, np.arange(8)] = 1.0
        D = self.B.decl
        A, G = D["limits"]["arm"], D["limits"]["gantry"]
        jr = self.m.jnt_range[[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, "A_" + j) for j in rig.joints]]
        n = 6
        self.lim = AL.ArmLimits("abb_irb1100", D["body"]["title"], jr[2:, 0].copy(), jr[2:, 1].copy(),
                                np.asarray(A["speed_rad_s"], float), np.asarray(A["accel_rad_s2"], float),
                                np.asarray(A["jerk_rad_s3"], float), np.asarray(A["torque_nm"], float), None)
        self.lim0 = self.lim
        self.gV = np.asarray(G["speed_m_s"], float)
        self.gA = np.asarray(G["accel_m_s2"], float)
        self.gJ = np.asarray(G["jerk_m_s3"], float)
        self.bV, self.bA, self.bJ = float(self.gV.min()), float(self.gA.min()), float(self.gJ.min())
        self.b_lo, self.b_hi = jr[0:2, 0].copy(), jr[0:2, 1].copy()
        self.q_ready = np.asarray(D["rest"]["arm"], float)
        self.qw = np.ones(n)
        self.vmask = np.ones(n)
        carriage = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "A_gantry_y")
        self.robot_geoms = {g for g in range(self.m.ngeom) if self._in_subtree(self.m.geom_bodyid[g], carriage)}
        base = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "A_base")
        self.robot_all = {g for g in range(self.m.ngeom) if self._in_subtree(self.m.geom_bodyid[g], base)}
        gid = lambda nm: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, nm)
        self.env = {gid("geom_table"), gid("geom_net"), gid("floor")}
        R = READIES
        self.readies = {k: list(v) for k, v in R.items()}
        self.reach = 0.6

    def _in_subtree(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def _set(self, b, q):
        self.k.qpos[self.qadr] = np.r_[b, q]
        mj.mj_kinematics(self.m, self.k)
        mj.mj_comPos(self.m, self.k)

    def collides(self, b, q):
        """the robot (paddle included) penetrating the table, net or floor, OR itself (the body's self-collision
        pairs, bodies/selfcollision.py: the arm and the paddle against the gantry and the arm's own links); the
        Franka study checked the first only (its scene had no self-collision then)"""
        self._set(b, q)
        mj.mj_collision(self.m, self.k)
        for c in self.k.contact[:self.k.ncon]:
            if c.dist >= -0.002:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.robot_geoms and g2 in self.env) or (g2 in self.robot_geoms and g1 in self.env):
                return True
            if SELF and g1 in self.robot_all and g2 in self.robot_all:
                return True
        return False

    def lim_at(self, q_list):
        return self.lim                     # no published torque rate: no extra jerk cap (the Franka's comes from it)

    def base_time(self, b0, b1):
        d = np.abs(np.asarray(b1, float) - np.asarray(b0, float))
        return max(BR.scurve_time(float(d[a]), self.gV[a], self.gA[a], self.gJ[a]) for a in range(2))

    def base_at(self, b0, b1, T, t):
        b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
        out = b0.copy()
        for a in range(2):
            D = abs(b1[a] - b0[a])
            out[a] = b0[a] + (b1[a] - b0[a]) * BR.scurve_frac(t, T, D, self.gV[a], self.gA[a], self.gJ[a])
        return out

    def world_base(self, b):
        return np.array([-b[1], -MOUNT["y"] + b[0]])


# the receiver's stances (b = gantry_x forward, gantry_y left), as the Franka's five; filled from the gantry's travel
READIES: dict = {}


def reach_box(body, b0, T):
    """the oracle's reach box per gantry axis (each axis its own S-curve limits)"""
    lo, hi = body.b_lo.copy(), body.b_hi.copy()
    b0 = np.asarray(b0, float)
    import oracle as O
    D = np.array([O.max_dist(lambda d: BR.scurve_time(d, body.gV[a], body.gA[a], body.gJ[a]), T) for a in range(2)])
    return np.maximum(lo, b0 - D), np.minimum(hi, b0 + D)


def setup_readies():
    """The receiver's five stances at the Franka study's world positions (b = gantry forward, left): home (0, -1.75),
    near (0, -1.55), near_fh / near_bh (+-0.25, -1.55), deep (0, -2.0); world +x (the forehand of a right-handed
    receiver facing +y) is the robot's right = gantry_y < 0."""
    W = {"home": (0.0, -1.75), "near": (0.0, -1.55), "near_fh": (0.25, -1.55), "near_bh": (-0.25, -1.55),
         "deep": (0.0, -2.0)}
    READIES.clear()
    for k, (x, y) in W.items():
        READIES[k] = [round(y + MOUNT["y"], 4), -x]
