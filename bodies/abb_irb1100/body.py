"""The ABB IRB 1100 on a two-axis gantry (body.toml): MJCF assembly, the servo and the controlled degrees of freedom.

    from bodies import load_body
    B = load_body("abb_irb1100")
    B.mount(scene_root, prefix="A_", pos=(0, -2.0, -0.76), yaw=pi / 2, hand="paddle")   # into a scene's XML tree
    rig = B.rig(model, prefix="A_", hand="paddle")      # after compiling: addresses, limits, servo, u <-> joints
    xml = B.robot_only_xml(hand="paddle")               # the robot alone (low-level seat, planner twin)

The same API as the other robots with a model (bodies/tianji_marvin_pro, bodies/wheeled_humanoid). The robot is a
6-axis arm on a Cartesian gantry: the carriage moves along the robot's forward axis (gantry_x) and its lateral axis
(gantry_y), the arm stands on it. The controlled vector u keeps the library's layout
[base_x, base_y, base_yaw, lift, waist, arm_0 .. arm_6] so that the conformance test, bodies/control/wholebody.py and
the sports' command layers run unchanged:

  base_x, base_y   the gantry's two linear axes (m, the displacement of the carriage from its home in the mount frame:
                   x forward, y left). Each axis is a joint of its own with its own drive, so the servo limits it
                   per axis like a joint (bodies/servo.py SetpointGenerator: speed, acceleration, jerk of
                   body.toml [limits.gantry]); there is no yaw.
  base_yaw, lift, waist, arm_6   NOT on this robot: always 0 (ranges 0 / +-0.001, speed 0 / 1e-6: a row that moves
                   them is refused by Rig.check). arm_6 is the slot of a 7th arm joint the IRB 1100 does not have.
  arm_0 .. arm_5   the arm's axes 1..6 (rad).

pos in mount() is the floor point under the gantry's home (the carriage at the centre of its travel); the mount's
yaw turns the gantry's x axis to the robot's forward direction.
"""
from __future__ import annotations

import copy
import math
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .. import limits as AL
from .. import selfcollision as SC
from .. import servo as SV

HERE = Path(__file__).resolve().parent
HANDS = HERE.parent / "hands"
NAME_ATTRS = ("name", "joint", "body", "body1", "body2", "geom", "geom1", "geom2", "site", "mesh", "material",
              "class", "childclass", "target", "texture", "tendon", "actuator")
GANTRY = ("gantry_x", "gantry_y")                       # u[0], u[1]
ARM = tuple(f"joint_{i}" for i in range(1, 7))          # u[5] .. u[10]
U_GANTRY = (0, 1)
U_ARM = tuple(range(5, 11))


def quat_z(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _fmt(v):
    return " ".join(repr(float(x)) for x in v)


class Body:
    kind = "gantry"

    def __init__(self):
        self.decl = tomllib.loads((HERE / "body.toml").read_text())
        self.hands = tomllib.loads((HANDS / "hands.toml").read_text())
        self.name = self.decl["body"]["name"]
        self.revision = self.decl["body"]["revision"]
        self.dof = list(self.decl["dof"]["names"])
        G = self.decl["limits"]["gantry"]
        # the whole-body planner's isotropic model of the "base": the slower gantry axis (the servo: the real axes)
        self.base_speed = float(np.min(G["speed_m_s"]))
        self.base_accel = float(np.min(G["accel_m_s2"]))
        self.yaw_rate, self.yaw_accel = 0.0, 1.0
        self.dt = self.decl["controller"]["step"]["control_dt_s"]
        self.substeps = int(round(self.dt / self.decl["controller"]["step"]["physics_dt_s"]))

    # ------------------------------------------------------------------ the template
    def _template(self, prefix, hand, lod, accent):
        root = ET.parse(HERE / self.decl["body"]["model"]).getroot()
        meshdir = HERE / self.decl["body"]["meshes"][lod]
        if lod == "fine" and not any(meshdir.glob("*.stl")):   # the fine meshes are not in git
            meshdir = HERE / self.decl["body"]["meshes"]["standard"]
        for m in root.find("asset").findall("mesh"):
            m.set("file", str(meshdir / (Path(m.get("file")).stem + ".stl")))
        if accent is not None:                                  # the carriage plate in a team colour
            for mat in root.find("asset").findall("material"):
                if mat.get("name") == "abb_accent":
                    mat.set("rgba", _fmt(accent))
        if hand:
            mount = next(b for b in root.iter("body") if b.get("name") == "mount")
            mount.append(self._hand(hand))
        for el in root.iter():
            for a in NAME_ATTRS:
                v = el.get(a)
                if v is not None:
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
              lod: str = "standard", accent=None, exclude_world: bool = True, self_collision: bool = True,
              **_ignored) -> dict:
        """Add one robot to a scene's XML tree (in place). pos: the floor point under the gantry's home, yaw: its
        facing (the robot's forward axis = world x rotated by yaw); the hand on the arm's flange (`arm` is accepted
        for the library's API: the robot has one arm). Returns the names a sport needs."""
        t = self._template(prefix, hand, lod, accent)
        sc = SC.plan(t, self.decl["self_collision"], prefix=prefix) if self_collision else None
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
        if sc is not None:
            SC.add_pairs(scene, sc)
        con = scene.find("contact")
        if con is None:
            con = ET.SubElement(scene, "contact")
        if exclude_world:                         # the gantry stands on the floor: never a contact with the world
            ET.SubElement(con, "exclude", body1="world", body2=prefix + "base")
        act = scene.find("actuator")
        if act is None:
            act = ET.SubElement(scene, "actuator")
        for j, lim in self._actuated().items():
            ET.SubElement(act, "motor", name=f"{prefix}m_{j}", joint=prefix + j, ctrlrange=f"{-lim!r} {lim!r}")
        return {**SC.info(sc), "prefix": prefix, "base": prefix + "base", "hand": prefix + "hand" if hand else None,
                "tool_site": prefix + "hand_" + self.hands[hand]["site"] if hand else None,
                "contact_geoms": [prefix + "hand_" + g for g in self.hands[hand].get("contact", [])] if hand else [],
                "arm": "right", "cameras": [prefix + c for c in self.decl["mounts"]["cameras"]]}

    def _actuated(self):
        L = self.decl["limits"]
        f = L["gantry"]["force_n"]
        out = {GANTRY[0]: float(f[0]), GANTRY[1]: float(f[1])}
        out.update({j: float(t) for j, t in zip(ARM, L["arm"]["torque_nm"])})
        return out

    def robot_only_xml(self, hand="paddle", arm="right", lod="standard", meshdir: str | None = None,
                       option: dict | None = None, header: str = "", self_collision: bool = True, **_ignored) -> str:
        """The robot alone, mounted at the origin facing +x (MJCF): the gantry, the arm, its hand, its motors."""
        root = ET.Element("mujoco", model=f"{self.name}_{hand}_robot_only")
        if header:
            root.append(ET.Comment(header))
        ET.SubElement(root, "compiler", angle="radian", autolimits="true", **({"meshdir": meshdir} if meshdir else {}))
        ET.SubElement(root, "option", **(option or {"timestep": "0.001", "integrator": "implicitfast"}))
        for sec in ("default", "asset", "worldbody"):
            ET.SubElement(root, sec)
        self.mount(root, "", (0.0, 0.0, 0.0), 0.0, hand=hand, lod=lod, exclude_world=False,
                   self_collision=self_collision)
        if meshdir:
            for m in root.find("asset").findall("mesh"):
                m.set("file", Path(m.get("file")).name)
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode") + "\n"

    def rig(self, model, prefix="", arm="right", hand="paddle", rest=None):
        return Rig(self, model, prefix, hand, rest)

    @staticmethod
    def guard_root(arm="right"):
        """The forearm's link (axis 4 onwards): the planner keeps it and everything below it out of keep-out boxes."""
        return "link_4"


def layout(model, rig, decl: dict, phys_dt: float):
    """The servo of the gantry robot: the gantry's two axes and the arm's six axes, each a joint coordinate of u,
    through bodies/servo.py's SetpointGenerator at body.toml's speed, acceleration and jerk ([limits.gantry] per axis,
    [limits.arm]); the output stage (bodies/servo.py Actuators) at the gantry's drive forces and the arm's torques.
    Returns (servo, ACC: the per-coordinate acceleration limits of u for check_row)."""
    L = decl["limits"]
    G, A = L["gantry"], L["arm"]
    jix = np.array(U_GANTRY + U_ARM, int)
    acc = np.full(12, np.inf)
    jerk = np.full(12, np.inf)
    acc[list(U_GANTRY)] = G["accel_m_s2"]; jerk[list(U_GANTRY)] = G["jerk_m_s3"]
    acc[list(U_ARM)] = A["accel_rad_s2"]; jerk[list(U_ARM)] = A["jerk_rad_s3"]
    lim = AL.ArmLimits(decl["body"]["name"], decl["body"]["title"] + " [u]", rig.LO[jix], rig.HI[jix], rig.VEL[jix],
                       acc[jix], jerk[jix], np.full(len(jix), np.inf), None, dict(decl.get("sources", {})))
    act = SV.Actuators(model, rig.aid, rig.vadr, rig.tau_lim, phys_dt)
    servo = SV.Servo(model, rig.qadr, rig.vadr, act, rig.M, rig.c, rig.kp, rig.kd, phys_dt, lim, jix=jix)
    return servo, acc


class Rig:
    """One mounted robot in a compiled model: addresses, limits in u, the u <-> joint map and the servo."""

    def __init__(self, body: Body, model, prefix="", hand="paddle", rest=None):
        import mujoco as mj
        self.B, self.m, self.prefix, self.arm = body, model, prefix, "right"
        D = body.decl
        jid = lambda n: mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix + n)
        self.joints = list(GANTRY + ARM)
        ids = [jid(j) for j in self.joints]
        if min(ids) < 0:
            raise ValueError(f"robot {prefix!r} not in the model")
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.vadr = np.array([model.jnt_dofadr[i] for i in ids])
        self.other_qadr = np.zeros(0, int)                     # one arm: no other arm (bodies/conformance.py 9)
        self.aid = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{prefix}m_{j}") for j in self.joints])
        self.tau_lim = model.actuator_ctrlrange[self.aid, 1].copy()
        self.M = np.zeros((8, 12))
        for r, k in enumerate(U_GANTRY + U_ARM):
            self.M[r, k] = 1.0
        self.c = np.zeros(8)
        C, L, R = D["controller"], D["limits"], D["rest"]
        self.kp = np.r_[C["kp"]["gantry"], C["kp"]["arm"]]
        self.kd = np.r_[C["kd"]["gantry"], C["kd"]["arm"]]
        jr = model.jnt_range[ids]
        absent = (-0.001, 0.001)
        self.LO = np.r_[jr[0:2, 0], 0.0, absent[0], absent[0], jr[2:, 0], absent[0]]
        self.HI = np.r_[jr[0:2, 1], 0.0, absent[1], absent[1], jr[2:, 1], absent[1]]
        self.VEL = np.r_[L["gantry"]["speed_m_s"], 0.0, 1e-6, 1e-6, L["arm"]["speed_rad_s"], 1e-6]
        self.STEP = self.VEL * body.dt
        self.rest = np.r_[R["gantry"], 0.0, 0.0, 0.0, R["arm"], 0.0]
        if rest is not None:                                   # a sport's own ready pose (lift, waist, arm[, slot])
            r = np.asarray(rest, float)
            self.rest[3:3 + len(r)] = r
            self.rest[[2, 3, 4, 11]] = 0.0
        self.dt, self.substeps = body.dt, body.substeps
        self.tool_site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, prefix + "hand_" + body.hands[hand]["site"]) \
            if hand else -1
        self.base_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, prefix + "base")
        self.servo, self.ACC = layout(model, self, D, body.dt / body.substeps)
        self.base_lim = None
        self.selfcol = SC.Monitor(model, prefix + "selfcol", D["self_collision"].get("warn_force_n"))   # recorded only

    # u <-> joints
    def joint_targets(self, u):
        return self.M @ np.asarray(u, float) + self.c

    def place(self, d, u):
        """Teleport to u at rest (between points; never inside a step); the servo's setpoints follow."""
        d.qpos[self.qadr] = self.joint_targets(u)
        d.qvel[self.vadr] = 0.0
        self.sync(d)

    def sync(self, d):
        """The servo's setpoint generator to the robot's measured state (after a teleport or a state copy)."""
        self.servo.reset(self.u_of(d), self.ud_of(d))

    def begin_step(self, d, u):
        """Once per control step before its substeps: nothing (no walking policy)."""

    def apply(self, d, u_prev, u_next, j, u_prev2=None):
        """Physics substep j (1..substeps) of the control step from target u_prev to u_next: the targets, linearly
        interpolated, through the robot's servo (bodies/servo.py: jerk-limited setpoints at the body.toml limits, PD +
        gravity/Coriolis + inertia feed-forward, the output stage)."""
        self.selfcol.update(d)
        if not self.servo.ready:
            self.sync(d)
        a, b = np.asarray(u_prev, float), np.asarray(u_next, float)
        self.servo.track(d, a + (b - a) * (j / self.substeps), (b - a) / self.dt)

    def self_collisions(self) -> dict:
        """The self-collision record since the rig was built (bodies/selfcollision.py Monitor.summary)."""
        return self.selfcol.summary()

    def check(self, u, u_prev, u_prev2=None):
        """A target row against the limits (bodies/servo.py check_row: ranges, per-step speeds per axis; with u_prev2
        also the accelerations): None or the reason."""
        return SV.check_row(u, u_prev, u_prev2, self.LO, self.HI, self.VEL, self.ACC, self.dt, None)

    def tool(self, d):
        """The tool site's position, rotation (columns: local axes) in the world."""
        return d.site_xpos[self.tool_site].copy(), d.site_xmat[self.tool_site].reshape(3, 3).copy()

    def u_of(self, d):
        q = d.qpos[self.qadr]
        return np.r_[q[0:2], 0.0, 0.0, 0.0, q[2:], 0.0]

    def ud_of(self, d):
        v = d.qvel[self.vadr]
        return np.r_[v[0:2], 0.0, 0.0, 0.0, v[2:], 0.0]


_BODIES: dict = {}


def load() -> Body:
    if "abb" not in _BODIES:
        _BODIES["abb"] = Body()
    return _BODIES["abb"]
