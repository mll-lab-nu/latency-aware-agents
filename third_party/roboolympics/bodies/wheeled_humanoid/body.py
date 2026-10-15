"""The wheeled humanoid (body.toml): MJCF assembly, the joint servo and the controlled degrees of freedom.

    from bodies import load_body
    B = load_body("wheeled_humanoid")
    B.mount(scene_root, prefix="A_", pos=(0, -1.8, -0.76), yaw=pi / 2, hand="paddle")   # into a scene's XML tree
    rig = B.rig(model, prefix="A_", hand="paddle")      # after compiling: addresses, limits, servo, u <-> joints
    xml = B.robot_only_xml(hand="paddle")               # the robot alone (low-level seat, planner twin)

The controlled vector u (body.toml [dof]): [base_x, base_y, base_yaw, lift, waist, arm_0 .. arm_6]. base_x / base_y
are the base's displacement from its mount along the mount's forward / left axes, base_yaw its rotation; lift is the
lifting column's angle a (torso_1 = a, torso_2 = -2a, torso_3 = a); waist = torso_5; arm_i = the hand arm's joints. The
other arm holds body.toml [rest] other_arm. Positions are targets at 100 Hz; the servo (Rig.apply -> bodies/servo.py, the
robot library's one limit / servo path) runs every physics step: the targets, linearly interpolated, through the
jerk-limited setpoint generator at body.toml's limits, then tau = kp (q_sp - q) + kd (qd_sp - qd) + bias (gravity,
Coriolis) + M(q) qdd_sp, through the output stage (torque limits).
Collision bits: robot geoms contype 4 / conaffinity 0 (never the robot itself; a scene that wants contacts with the
robot gives its own geoms conaffinity 4); hand "contact" geoms contype 1 / conaffinity 1 (the sport's <pair>s).
"""
from __future__ import annotations

import copy
import math
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .. import selfcollision as SC
from .. import servo as SV

HERE = Path(__file__).resolve().parent
HANDS = HERE.parent / "hands"
NAME_ATTRS = ("name", "joint", "body", "body1", "body2", "geom", "geom1", "geom2", "site", "mesh", "material",
              "class", "childclass", "target", "texture", "tendon", "actuator")
CLASS_ATTRS = ("class", "childclass")
SECTIONS = ("default", "asset", "worldbody", "contact", "actuator")
TORSO_LIFT = ("torso_1", "torso_2", "torso_3")
LIFT_COEF = (1.0, -2.0, 1.0)
LINK = 0.35                                   # the two lift links (m): chest height = 2 LINK cos(a) above torso_1


def _toml(p):
    return tomllib.loads(Path(p).read_text())


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_z(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _fmt(v):
    return " ".join(repr(float(x)) for x in v)


class Body:
    kind = "wheeled"

    def __init__(self):
        self.decl = _toml(HERE / "body.toml")
        self.hands = _toml(HANDS / "hands.toml")
        self.name = self.decl["body"]["name"]
        self.revision = self.decl["body"]["revision"]
        self.dof = list(self.decl["dof"]["names"])
        L = self.decl["limits"]
        self.base_speed = float(np.min(L["base"]["speed_m_s"]))      # the planner's isotropic model of the base:
        self.base_accel = float(np.min(L["base"]["accel_m_s2"]))     # the smaller axis (bodies/servo.py: the real ones)
        self.yaw_rate, self.yaw_accel = L["base"]["yaw_rate_rad_s"], L["base"]["yaw_accel_rad_s2"]
        self.dt = self.decl["controller"]["step"]["control_dt_s"]
        self.substeps = int(round(self.dt / self.decl["controller"]["step"]["physics_dt_s"]))

    # ------------------------------------------------------------------ the template, renamed
    def _template(self, prefix, hand, arm, lod, accent, head_pitch, other_hand=None, head_yaw=0.0):
        root = ET.parse(HERE / self.decl["body"]["model"]).getroot()
        meshdir = HERE / self.decl["body"]["meshes"][lod]
        if lod == "fine" and not meshdir.is_dir():          # the fine meshes are not in git (tools/make_model.py)
            meshdir, lod = HERE / self.decl["body"]["meshes"]["standard"], "standard"
        for m in root.find("asset").findall("mesh"):
            stem = Path(m.get("file")).stem
            f = meshdir / (stem + (".obj" if lod == "fine" else ".stl"))
            m.set("file", str(f))
        if accent is not None:                    # the chest stripe (LINK_6_3) in a team colour
            for mat in root.find("asset").findall("material"):
                if mat.get("name") == "wh_m_178_0_0":
                    mat.set("rgba", _fmt(accent))
        if head_pitch is None:
            head_pitch = self.decl["mounts"]["head_pitch_rad"]
        for b in root.iter("body"):                  # the fixed neck: yaw about z, then pitch about y (down)
            if b.get("name") == "link_head_2" and (head_pitch or head_yaw):
                cy, sy = math.cos(head_yaw / 2), math.sin(head_yaw / 2)
                cp, sp = math.cos(head_pitch / 2), math.sin(head_pitch / 2)
                b.set("quat", _fmt((cy * cp, -sy * sp, cy * sp, sy * cp)))
        if hand:
            mount = next(b for b in root.iter("body") if b.get("name") == f"{arm}_mount")
            mount.append(self._hand(hand))
        if other_hand:                            # a second hand on the other wrist: names "hand2_..."
            other = "left" if arm == "right" else "right"
            mount = next(b for b in root.iter("body") if b.get("name") == f"{other}_mount")
            mount.append(self._hand(other_hand, "hand2"))
        # rename every name and reference (classes too) so that several robots fit in one scene
        for el in root.iter():
            for a in NAME_ATTRS:
                v = el.get(a)
                if v is not None and not (a == "class" and el.tag == "default" and False):
                    el.set(a, prefix + v)
        return root

    def _hand(self, hand, tag="hand"):
        h = self.hands[hand]
        src = ET.parse(HANDS / h["file"]).getroot()
        body = copy.deepcopy(src.find("worldbody").find("body"))
        contact = set(h.get("contact", []))
        for el in body.iter():
            if el.tag == "body" and el.get("name") == "hand":
                el.set("name", tag)
            elif el.get("name") is not None:
                if el.tag == "geom":
                    is_contact = el.get("name") in contact
                    el.set("contype", "1" if is_contact else "4")
                    el.set("conaffinity", "1" if is_contact else "0")
                el.set("name", f"{tag}_" + el.get("name"))
            elif el.tag == "geom":
                el.set("contype", "4"); el.set("conaffinity", "0")
        return body

    # ------------------------------------------------------------------ assembly into a scene
    def mount(self, scene: ET.Element, prefix: str, pos, yaw: float, hand: str | None = "paddle", arm: str = "right",
              lod: str = "standard", accent=None, head_pitch=None, exclude_world: bool = True,
              other_hand: str | None = None, head_yaw: float = 0.0, self_collision: bool = True) -> dict:
        """Add one robot to a scene's XML tree (in place). pos: the base origin (on the floor), yaw: its facing (the
        robot's forward axis = world x rotated by yaw); hand on `arm`, optionally other_hand on the other arm (its
        names start with "hand2_"). Returns the names a sport needs."""
        t = self._template(prefix, hand, arm, lod, accent, head_pitch, other_hand, head_yaw)
        sc = SC.plan(t, self.decl["self_collision"], prefix=prefix) if self_collision else None   # selfcollision.py
        for sec in ("default", "asset"):
            dst = scene.find(sec)
            if dst is None:
                dst = ET.SubElement(scene, sec)
            for el in t.find(sec):
                dst.append(el)
        base = t.find("worldbody").find("body")
        base.set("pos", _fmt(pos))
        base.set("quat", _fmt(quat_z(yaw)))
        scene.find("worldbody").append(base)
        if sc is not None:                        # the robot's self-collision pairs (body.toml [self_collision])
            SC.add_pairs(scene, sc)
        con = scene.find("contact")
        if con is None:
            con = ET.SubElement(scene, "contact")
        if exclude_world:                         # the base glides above the floor: never a contact with the world
            ET.SubElement(con, "exclude", body1="world", body2=prefix + "base")
        act = scene.find("actuator")
        if act is None:
            act = ET.SubElement(scene, "actuator")
        for j, lim in self._actuated(arm).items():
            ET.SubElement(act, "motor", name=f"{prefix}m_{j}", joint=prefix + j, ctrlrange=f"{-lim!r} {lim!r}")
        return {**SC.info(sc), "prefix": prefix, "base": prefix + "base", "hand": prefix + "hand" if hand else None,
                "tool_site": prefix + "hand_" + self.hands[hand]["site"] if hand else None,
                "contact_geoms": [prefix + "hand_" + g for g in self.hands[hand].get("contact", [])] if hand else [],
                "arm": arm, "cameras": [prefix + c for c in self.decl["mounts"]["cameras"]],
                **({"tool_site2": prefix + "hand2_" + self.hands[other_hand]["site"],
                    "contact_geoms2": [prefix + "hand2_" + g for g in self.hands[other_hand].get("contact", [])]}
                   if other_hand else {})}

    def _actuated(self, arm):
        L = self.decl["limits"]
        out = {"base_x": L["base"]["force_n"], "base_y": L["base"]["force_n"], "base_yaw": L["base"]["yaw_torque_nm"]}
        out.update(dict(zip(TORSO_LIFT, L["lift"]["torque_nm"])))
        out["torso_5"] = L["waist"]["torque_nm"]
        for side in ("right", "left"):
            for i, tq in enumerate(L["arm"]["torque_nm"]):
                out[f"{side}_arm_{i}"] = tq
        return out

    def robot_only_xml(self, hand="paddle", arm="right", lod="standard", meshdir: str | None = None,
                       option: dict | None = None, header: str = "", self_collision: bool = True) -> str:
        """The robot alone, mounted at the origin facing +x (MJCF): the body, its hand, its motors; nothing of any
        world. meshdir: write mesh files relative to it (else absolute paths)."""
        root = ET.Element("mujoco", model=f"{self.name}_{hand}_robot_only")
        if header:
            root.append(ET.Comment(header))
        ET.SubElement(root, "compiler", angle="radian", autolimits="true", **({"meshdir": meshdir} if meshdir else {}))
        ET.SubElement(root, "option", **(option or {"timestep": "0.001", "integrator": "implicitfast"}))
        for sec in ("default", "asset", "worldbody"):
            ET.SubElement(root, sec)
        self.mount(root, "", (0.0, 0.0, 0.0), 0.0, hand=hand, arm=arm, lod=lod, exclude_world=False,
                   self_collision=self_collision)
        if meshdir:
            for m in root.find("asset").findall("mesh"):
                m.set("file", Path(m.get("file")).name)
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode") + "\n"

    def rig(self, model, prefix="", arm="right", hand="paddle", rest=None):
        return Rig(self, model, prefix, arm, hand, rest)

    @staticmethod
    def guard_root(arm):
        """The forearm's link: the planner keeps it and everything below it out of a sport's keep-out boxes."""
        return f"link_{arm}_arm_3"


class Rig:
    """One mounted robot in a compiled model: its addresses, its limits in u, the u <-> joint map and the servo."""

    def __init__(self, body: Body, model, prefix="", arm="right", hand="paddle", rest=None):
        import mujoco as mj
        self.B, self.m, self.prefix, self.arm = body, model, prefix, arm
        D = body.decl
        jid = lambda n: mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix + n)
        other = "left" if arm == "right" else "right"
        self.joints = (["base_x", "base_y", "base_yaw", *TORSO_LIFT, "torso_5"] +
                       [f"{arm}_arm_{i}" for i in range(7)] + [f"{other}_arm_{i}" for i in range(7)])
        ids = [jid(j) for j in self.joints]
        if min(ids) < 0:
            raise ValueError(f"robot {prefix!r} not in the model")
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.vadr = np.array([model.jnt_dofadr[i] for i in ids])
        self.aid = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{prefix}m_{j}") for j in self.joints])
        self.tau_lim = model.actuator_ctrlrange[self.aid, 1].copy()
        nj = len(self.joints)
        # joint targets = M @ u + c
        self.M = np.zeros((nj, 12))
        for k in range(3):
            self.M[k, k] = 1.0
        for r, cf in zip((3, 4, 5), LIFT_COEF):
            self.M[r, 3] = cf
        self.M[6, 4] = 1.0
        for i in range(7):
            self.M[7 + i, 5 + i] = 1.0
        self.c = np.zeros(nj)
        R = D["rest"]
        self.c[14:21] = R["other_arm"] if arm == "right" else [x * s for x, s in zip(R["other_arm"],
                                                                                  (1, -1, -1, 1, -1, 1, -1))]
        C, L = D["controller"], D["limits"]
        self.kp = np.r_[C["kp"]["base"], C["kp"]["base"], C["kp"]["yaw"], C["kp"]["lift"], C["kp"]["waist"],
                        C["kp"]["arm"], C["kp"]["arm"]]
        self.kd = np.r_[C["kd"]["base"], C["kd"]["base"], C["kd"]["yaw"], C["kd"]["lift"], C["kd"]["waist"],
                        C["kd"]["arm"], C["kd"]["arm"]]
        jr = model.jnt_range[ids[7:14]]
        self.LO = np.r_[-np.inf, -np.inf, -np.inf, L["lift"]["range_rad"][0], L["waist"]["range_rad"][0], jr[:, 0]]
        self.HI = np.r_[np.inf, np.inf, np.inf, L["lift"]["range_rad"][1], L["waist"]["range_rad"][1], jr[:, 1]]
        self.VEL = np.r_[body.base_speed, body.base_speed, body.yaw_rate, L["lift"]["speed_rad_s"],
                         L["waist"]["speed_rad_s"], L["arm"]["speed_rad_s"]]
        self.STEP = self.VEL * body.dt
        self.rest = np.r_[0.0, 0.0, 0.0, R["lift"], R["waist"], R["arm"]]
        if arm == "left":
            self.rest[5:] = [x * s for x, s in zip(R["arm"], (1, -1, -1, 1, -1, 1, -1))]
        if rest is not None:                      # a sport's own ready pose (lift, waist, arm)
            self.rest[3:] = rest
        self.dt, self.substeps = body.dt, body.substeps
        self.tool_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, prefix + "hand_" + body.hands[hand]["site"]) \
            if hand else -1
        self.base_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, prefix + "base")
        if self.base_lim_axes():                  # per-axis base speeds (the robot's frame)
            self.VEL[0:2] = np.asarray(D["limits"]["base"]["speed_m_s"], float)
            self.STEP = self.VEL * body.dt
        self.servo, self.ACC, self.base_lim = SV.wheeled_layout(model, self, D, body.dt / body.substeps)

        self.selfcol = SC.Monitor(model, prefix + "selfcol", D["self_collision"].get("warn_force_n"))   # recorded only
    def base_lim_axes(self):
        Lb = self.B.decl["limits"]["base"]
        return Lb.get("kind") != "absent" and Lb.get("mode") == "axes"

    # u <-> joints
    def joint_targets(self, u):
        return self.M @ np.asarray(u, float) + self.c

    def u_of(self, d):
        q = d.qpos[self.qadr]
        return np.r_[q[0:3], q[3], q[6], q[7:14]]

    def ud_of(self, d):
        v = d.qvel[self.vadr]
        return np.r_[v[0:3], v[3], v[6], v[7:14]]

    def place(self, d, u):
        """Teleport to u at rest (between points; never inside a step); the servo's setpoints follow."""
        d.qpos[self.qadr] = self.joint_targets(u)
        d.qvel[self.vadr] = 0.0
        self.sync(d)

    def sync(self, d):
        """The servo's setpoint generator to the robot's measured state (after a teleport or a state copy)."""
        self.servo.reset(self.u_of(d), self.ud_of(d))

    def begin_step(self, d, u):
        """Once per control step before its substeps (the legged body runs its walking policy here): nothing."""

    def apply(self, d, u_prev, u_next, j, u_prev2=None):
        """Physics substep j (1..substeps) of the control step from target u_prev to u_next: the targets, linearly
        interpolated, through the robot's servo (bodies/servo.py: jerk-limited setpoints at the body.toml limits, PD +
        gravity/Coriolis + inertia feed-forward, the output stage). u_prev2 is accepted for the old signature."""
        self.selfcol.update(d)                    # self-collision contacts of the last physics step (recorded)
        if not self.servo.ready:
            self.sync(d)
        a, b = np.asarray(u_prev, float), np.asarray(u_next, float)
        self.servo.track(d, a + (b - a) * (j / self.substeps), (b - a) / self.dt)

    def self_collisions(self) -> dict:
        """The self-collision record since the rig was built or reset (bodies/selfcollision.py Monitor.summary):
        events, warnings, time in contact, peak force, impulse, the worst pairs. For the match record."""
        return self.selfcol.summary()

    def check(self, u, u_prev, u_prev2=None):
        """A target row against the limits (bodies/servo.py check_row: ranges, per-step speeds; with u_prev2 also the
        accelerations of the base and of every joint coordinate): None or the reason."""
        return SV.check_row(u, u_prev, u_prev2, self.LO, self.HI, self.VEL, self.ACC, self.dt, self.base_lim)

    def tool(self, d):
        """The tool site's position, rotation (columns: local axes) in the world."""
        return d.site_xpos[self.tool_site].copy(), d.site_xmat[self.tool_site].reshape(3, 3).copy()


_BODIES: dict = {}


def load() -> Body:
    if "wh" not in _BODIES:
        _BODIES["wh"] = Body()
    return _BODIES["wh"]
