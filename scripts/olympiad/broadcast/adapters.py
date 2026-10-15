"""Per-sport broadcast adapters (render helper side): how to build a match's MuJoCo model from its copied sim directory,
how to put a broadcast-log frame into MjData, the broadcast cameras and any extra geometry the scene does not hold
(e.g. the scripted opponents of the single-player tasks, which the sims also draw as overlay geoms).

A frame is a record {"k": step, "q": [qpos...], ...extra} (extra = the rest of the sim's own camera snapshot, e.g. the
ball "b", the shuttle "sh"/"shv", the opponent "opp"/"o", the duel's per-robot "qA"/"qB"); between two logged frames
the helper interpolates every numeric field. Where it can, an adapter reuses the sport's own drawing code (imported
from the match's sim directory), so the broadcast shows what the sim's cameras show.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import mujoco as mj
import audience


def cam(lookat, dist, az, el):
    c = mj.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def add_geom(scn, typ, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    mj.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.asarray(size, float), np.asarray(pos, float),
                    np.eye(3).ravel() if mat is None else np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def fresh_import(simdir, names):
    """Import modules from a match's sim directory and forget them again (two matches may run different versions of
    modules with the same names). The returned modules (and functions taken from them) stay usable."""
    before = set(sys.modules)
    sys.path.insert(0, simdir)
    try:
        return [importlib.import_module(n) for n in names]
    finally:
        sys.path.remove(simdir)
        for n in set(sys.modules) - before:
            sys.modules.pop(n, None)


class Adapter:
    CAMS = ("main",)
    CAM_SPECS: dict = {}
    QUATS = ()                # qpos slices holding a free joint's quaternion (renormalized after interpolation)
    BALL = None               # how to find the ball in a record: ("q", a, b) = q[a:b], or (key,) = record[key]
    HALO = 0.045              # radius of the halo drawn around the ball (a small ball is hard to follow on screen)
    EQUIPMENT_GEOMS = ()      # only these sporting surfaces receive seat colors; preserve robot materials

    def __init__(self, simdir):
        self.cams = {}
        self.m = self.model(simdir)
        self.d = mj.MjData(self.m)
        self.player_roots = {}
        # Color equipment only in the spectator's copy. Exact geometry names and
        # robot ancestry preserve handles, strings and the original arm materials.
        for seat, root_name in (("A", "link0"), ("B", "link0_b")):
            root = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, root_name)
            if root < 0:
                continue
            self.player_roots[seat] = root
            color = np.asarray(audience.SEAT_RGB[seat]) / 255.0
            for gid in range(self.m.ngeom):
                name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_GEOM, gid) or ""
                base_name = name.removesuffix("_b") if seat == "B" else name
                if base_name not in self.EQUIPMENT_GEOMS:
                    continue
                body = int(self.m.geom_bodyid[gid])
                while body and body != root:
                    body = int(self.m.body_parentid[body])
                if body != root:
                    continue
                self.m.geom_rgba[gid, :3] = color
                self.m.geom_matid[gid] = -1

    def camera(self, name):
        """A named broadcast camera; "<name>_b" is <name> mirrored to the other end (y -> -y, azimuth + 180), e.g.
        "end_b" = behind B. A tuple (lookat, distance, azimuth, elevation) is a free camera (the director's close-up)."""
        if isinstance(name, (tuple, list)):
            return cam(*name)
        if name not in self.cams:
            spec = self.CAM_SPECS.get(name)
            if spec is None and name.endswith("_b") and name[:-2] in self.CAM_SPECS:
                (x, y, z), dist, az, el = self.CAM_SPECS[name[:-2]]
                spec = ([x, -y, z], dist, az + 180.0, el)
            self.cams[name] = cam(*(spec or self.CAM_SPECS[self.CAMS[0]]))
        return self.cams[name]

    def set_state(self, rec):
        q = np.asarray(rec.get("q", ()), float)
        for a, b in self.QUATS:
            n = np.linalg.norm(q[a:b])
            if n > 1e-9:
                q[a:b] /= n
        self.d.qpos[:len(q)] = q[:self.m.nq]
        mj.mj_forward(self.m, self.d)

    def extras(self, scn, rec):
        pass

    def ball_pos(self, rec):
        if self.BALL is None:
            return None
        if self.BALL[0] == "q":
            return np.asarray(rec["q"][self.BALL[1]:self.BALL[2]], float)
        v = rec.get(self.BALL[0])
        return None if v is None else np.asarray(v, float)

    def ball(self, scn, rec, trail, halo=None):
        """A halo around the ball and its recent path (halo: its radius for this frame, e.g. on a wide shot)."""
        halo = float(halo or self.HALO)
        p = self.ball_pos(rec)
        if p is None or p[2] < -1.0:                                  # parked out of sight
            return
        add_geom(scn, mj.mjtGeom.mjGEOM_SPHERE, [halo] * 3, p, [1.0, 0.85, 0.2, 0.35])
        n = len(trail)
        for i, tr in enumerate(trail):
            tp = self.ball_pos(tr)
            if tp is not None:
                add_geom(scn, mj.mjtGeom.mjGEOM_SPHERE, [halo * (0.25 + 0.3 * (i + 1) / n)] * 3, tp,
                         [1.0, 0.75, 0.2, 0.15 + 0.45 * (i + 1) / n])


# ---------------------------------------------------------------- table tennis
class TableTennisSingle(Adapter):
    """tasks/tabletennis-match-*: Franka (y < 0) against the scripted opponent (a kinematic paddle, not in the scene)."""
    EQUIPMENT_GEOMS = ("paddle_blade",)
    CAMS = ("main", "high", "end")
    CAM_SPECS = {"main": ([0.0, 0.0, 0.2], 3.7, 180.0, -16.0),       # TV side view: A (y < 0) on the left
                 "high": ([0.0, 0.0, 0.0], 4.6, 180.0, -50.0),
                 "end": ([0.0, 0.3, 0.1], 3.8, 90.0, -24.0)}         # behind A, down the table
    QUATS = ((3, 7),)
    BALL = ("q", 0, 3)
    OPP_Y, OPP_Z = 1.75, 0.22                                        # match.py (where the sim draws the opponent)

    def model(self, simdir):
        return mj.MjModel.from_xml_path(os.path.join(simdir, "scene", "tt_scene.xml"))

    def extras(self, scn, rec):
        o = float(rec.get("o", 0.0))
        add_geom(scn, mj.mjtGeom.mjGEOM_BOX, [0.075, 0.008, 0.075], [o, self.OPP_Y, self.OPP_Z], [0.85, 0.1, 0.1, 1])
        add_geom(scn, mj.mjtGeom.mjGEOM_CAPSULE, [0.14, 0.30, 0], [o, self.OPP_Y + 0.35, 0.30], [0.35, 0.35, 0.4, 1])


class TableTennisDuel(TableTennisSingle):
    """tasks/tabletennis-duel-*: robots A (y < 0) and B (y > 0) in one scene."""

    def model(self, simdir):
        return mj.MjModel.from_xml_path(os.path.join(simdir, "scene", "tt_duel.xml"))

    def extras(self, scn, rec):
        pass


# ---------------------------------------------------------------- tennis
class TennisSingle(Adapter):
    """tasks/tennis-match-*: the rail-mounted Franka (y < 0) against the scripted opponent; the ball is a mocap body
    ("b"), the opponent ("opp" = its x) is drawn by tn.Match._extra_geoms, as in the sim's own cameras."""
    EQUIPMENT_GEOMS = ("racket_throat", "racket_frame")
    CAMS = ("main", "high", "end")
    CAM_SPECS = {"main": ([0.0, 0.0, 0.0], 19.5, 180.0, -24.0),
                 "high": ([0.0, 0.0, 0.0], 24.0, 180.0, -48.0),
                 "end": ([0.0, -1.0, 1.0], 18.0, 90.0, -16.0)}
    BALL = ("b",)
    HALO = 0.16
    SCENE = "tn_scene.xml"

    def model(self, simdir):
        (tn,) = fresh_import(simdir, ["tn"])
        self._extra = tn.Match._extra_geoms
        return mj.MjModel.from_xml_path(os.path.join(simdir, "scene", self.SCENE))

    def set_state(self, rec):
        q = np.asarray(rec["q"], float)
        self.d.qpos[:len(q)] = q[:self.m.nq]
        if rec.get("b") is not None and self.m.nmocap:
            self.d.mocap_pos[0] = rec["b"]
        mj.mj_kinematics(self.m, self.d)

    def extras(self, scn, rec):
        if rec.get("opp") is not None:
            self._extra(scn, rec["opp"], rec.get("b"))


class TennisDuel(TennisSingle):
    """tasks/tennis-duel-*: robots A (y < 0) and B (y > 0) in one scene (tn_duel.xml), the ball a mocap body."""
    SCENE = "tn_duel.xml"

    def model(self, simdir):
        return mj.MjModel.from_xml_path(os.path.join(simdir, "scene", self.SCENE))

    def extras(self, scn, rec):
        pass


# ---------------------------------------------------------------- badminton
class BadmintonSingle(Adapter):
    """tasks/badminton-match-*: the robot (y < 0) against the scripted opponent ("opp" = its x, y); the shuttle is a
    mocap body ("sh", oriented along its velocity "shv")."""
    EQUIPMENT_GEOMS = ("racket_frame",)
    CAMS = ("main", "high", "end")
    CAM_SPECS = {"main": ([0.0, 0.0, 1.5], 10.8, 180.0, -17.0),
                 "high": ([0.0, 0.0, 0.0], 14.0, 180.0, -45.0),
                 "end": ([0.0, -1.0, 2.8], 13.5, 90.0, -6.0)}
    BALL = ("sh",)
    HALO = 0.11

    def model(self, simdir):
        (bm,) = fresh_import(simdir, ["bm"])
        self._extra, self._quat = bm.Match._extra_geoms, bm.quat_z_to
        return mj.MjModel.from_xml_path(os.path.join(simdir, "scene", "bm_scene.xml"))

    def _shuttle(self, rec):
        if rec.get("sh") is None or not self.m.nmocap:
            return
        self.d.mocap_pos[0] = rec["sh"]
        v = np.asarray(rec.get("shv") or (0.0, 0.0, -1.0), float)
        self.d.mocap_quat[0] = self._quat(v if np.linalg.norm(v) > 1e-6 else np.array([0.0, 0.0, -1.0]))

    def set_state(self, rec):
        q = np.asarray(rec["q"], float)
        self.d.qpos[:len(q)] = q[:self.m.nq]
        self._shuttle(rec)
        mj.mj_kinematics(self.m, self.d)

    def extras(self, scn, rec):
        if rec.get("opp") is not None:
            self._extra(scn, rec["opp"], rec.get("sh"))


class BadmintonDuel(BadmintonSingle):
    """tasks/badminton-duel-*: one robot model per side in the sim; the broadcast draws bm_duel.xml with both robots
    ("qA", "qB": each robot's own qpos, as bduel.Renderer.set_state does)."""

    def model(self, simdir):
        (bm,) = fresh_import(simdir, ["bm"])
        self._quat = bm.quat_z_to
        m = mj.MjModel.from_xml_path(os.path.join(simdir, "scene", "bm_duel.xml"))
        jid = lambda n: m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]
        self.qa = {"A": np.array([jid(j) for j in bm.JOINTS]), "B": np.array([jid(j + "_b") for j in bm.JOINTS])}
        return m

    def set_state(self, rec):
        for s in ("A", "B"):
            if rec.get("q" + s) is not None:
                self.d.qpos[self.qa[s]] = rec["q" + s]
        self._shuttle(rec)
        mj.mj_kinematics(self.m, self.d)

    def extras(self, scn, rec):
        pass


# ---------------------------------------------------------------- baseball
class BaseballSingle(Adapter):
    """tasks/baseball-*: no dynamics in MuJoCo (bb.scene_xml(): the field, the figures, mocap ball, bat and hands);
    a frame holds the pitch clock's display data: the ball ("ball"), the swing's parameters ("sw") and the time since
    release ("t"), drawn with bb.Renderer.set as in the sim's own cameras. Free broadcast cameras (centre field) plus
    the scene's fixed cameras by name (catcher, side, plate_side, wide, behind)."""
    CAMS = ("main", "wide", "behind", "plate_side")
    BALL = ("ball",)
    HALO = 0.10
    FREE = {"main": ((1.3, 36.0, 4.4), (-0.45, 0.0, 0.95), 13.0)}    # centre-field TV camera over the pitcher

    def model(self, simdir):
        (bb,) = fresh_import(simdir, ["bb"])
        self._bb = bb
        return mj.MjModel.from_xml_string(bb.scene_xml())

    def __init__(self, simdir):
        super().__init__(simdir)
        rd = object.__new__(self._bb.Renderer)          # bb.Renderer.set without its own GL context
        rd.mj, rd.m, rd.d = mj, self.m, self.d
        self.rd = rd

    def camera(self, name):
        if isinstance(name, (tuple, list)):
            return cam(*name)
        if name in self.cams:
            return self.cams[name]
        cid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA, name)
        c = mj.MjvCamera()
        if cid >= 0 and name not in self.FREE:
            c.type, c.fixedcamid = mj.mjtCamera.mjCAMERA_FIXED, cid
        else:
            pos, tg, fovy = self.FREE.get(name, self.FREE["main"])
            f = np.asarray(tg, float) - np.asarray(pos, float)
            dist = float(np.linalg.norm(f)); f /= dist
            c.lookat[:] = tg; c.distance = dist
            c.azimuth = float(np.degrees(np.arctan2(f[1], f[0]))); c.elevation = float(np.degrees(np.arcsin(f[2])))
            self.m.vis.global_.fovy = fovy                 # the free camera's field of view
        self.cams[name] = c
        return c

    def set_state(self, rec):
        sw = None if rec.get("sw") is None else self._bb.Swing(*rec["sw"])
        self.rd.set(tuple(rec["ball"]) if rec.get("ball") is not None else None, sw, float(rec.get("t", 0.0)))


class BaseballDuel(BaseballSingle):
    """tasks/baseball-duel-*: the same field and rig (bbduel uses bb's scene); the seats take turns batting."""


# ---------------------------------------------------------------- soccer (RoboOlympics 2 v 2, Booster T1)
class Soccer(Adapter):
    """tasks/soccer-{match,duel,team}-*: four Booster T1 humanoids (A1, A2 attack +x; B1, B2) and the ball, all free
    joints in one scene (soccer_scene.build_model(), T1 meshes from the task image's /opt/sai_t1). A frame's "q" is the
    full qpos (ball q[0:7], the robots' roots at 7, 26, 45, 64); "look" is the host's smoothed ball point that the
    main (side-high, ball-following) camera looks at. Other cameras arrive as free cameras from the director."""
    CAMS = ("main", "high")
    QUATS = ((3, 7), (10, 14), (29, 33), (48, 52), (67, 71))
    BALL = ("q", 0, 3)
    HALO = 0.16
    MAIN = (7.4, 90.0, -30.0)                    # distance, azimuth (from the -y touchline), elevation

    def model(self, simdir):
        (sc,) = fresh_import(simdir, ["soccer_scene"])
        m = sc.build_model()
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, 1280), max(m.vis.global_.offheight, 720)
        self.look = np.zeros(3)
        return m

    def set_state(self, rec):
        super().set_state(rec)
        lk = rec.get("look")
        if lk is not None and len(lk) >= 3:
            self.look = np.asarray(lk[:3], float)
        else:
            self.look = np.asarray([np.clip(rec["q"][0], -3.0, 3.0), np.clip(rec["q"][1], -1.2, 1.2), 0.0], float)

    def camera(self, name):
        if isinstance(name, (tuple, list)):
            return cam(*name)
        if name == "high":
            return cam([0.0, 0.0, 0.0], 11.2, 90.0, -89.0)
        return cam(self.look, *self.MAIN)


# ---------------------------------------------------------------- 3D racing (two wheeled cars on a stadium track)
class Racing(Adapter):
    """tasks/racing-{match,duel}-*: two cars (A blue, B orange) on a procedural stadium track; the scene has no file,
    race.scene_xml(seed) builds it (the seed is in the broadcast log's header, passed as a scene option). A frame's "q"
    is the full qpos (each car's free joint at the header's qpos_layout, then wheels and steering); "look" / "az" /
    "dist" are the host's smoothed chase camera (behind the leading car, both cars in view while they are close).
    Other cameras: the scene's fixed "overview", "high" (the whole track from above) and the director's free cameras."""
    CAMS = ("main", "overview", "high")
    EL = -25.0
    SKY = ('<asset><texture name="bc_sky" type="skybox" builtin="gradient" rgb1=".58 .72 .9" rgb2=".93 .95 .98" '
           'width="256" height="256"/></asset>')           # spectator copy only: a sky instead of the black void

    def __init__(self, simdir, seed=1):
        self.seed = int(seed)
        super().__init__(simdir)
        self.QUATS = tuple((a + 3, a + 7) for a in (int(self.m.jnt_qposadr[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT,
                                                                                      f"root_{s}")]) for s in "AB"))
        self.look, self.az, self.dist = np.zeros(3), 90.0, 6.0

    def model(self, simdir):
        (race,) = fresh_import(simdir, ["race"])
        self.track = race.track(self.seed)[:2]                         # straight half length, radius
        xml = race.scene_xml(self.seed)
        i = xml.index(">", xml.index("<mujoco")) + 1
        xml = xml[:i] + self.SKY + xml[i:]
        j = xml.rindex("</worldbody>")
        m = mj.MjModel.from_xml_string(xml[:j] + self.finish_line(race) + xml[j:])
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, 1920), max(m.vis.global_.offheight, 1080)
        m.vis.quality.shadowsize = max(m.vis.quality.shadowsize, 2048)
        return m

    def finish_line(self, race):
        """Spectator copy only (visual, no contacts): a chequered finish line at s = 0 (x = 0 on the -y straight),
        where race.py's last gate is."""
        r, w, n = self.track[1], race.WIDTH, 12
        sq = w / n / 2
        out = []
        for row in (0, 1):
            for c in range(n):
                col = "1 1 1 1" if (row + c) % 2 else ".05 .05 .05 1"
                out.append(f'<geom type="box" pos="{(row - 0.5) * 2 * sq:.4f} {-r - w / 2 + (2 * c + 1) * sq:.4f} .0045" '
                           f'size="{sq:.4f} {sq:.4f} .0015" rgba="{col}" contype="0" conaffinity="0"/>')
        return "".join(out)

    def set_state(self, rec):
        super().set_state(rec)
        lk = rec.get("look")
        if lk is not None and len(lk) >= 3:
            self.look = np.asarray(lk[:3], float)
            self.az, self.dist = float(rec.get("az", 90.0)), float(rec.get("dist", 6.0))
        else:
            self.look, self.az, self.dist = np.asarray(rec["q"][0:3], float), 90.0, 6.0

    def camera(self, name):
        if isinstance(name, (tuple, list)):
            return cam(*name)
        if name == "overview":
            c = mj.MjvCamera()
            c.type, c.fixedcamid = mj.mjtCamera.mjCAMERA_FIXED, mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA, "overview")
            return c
        if name == "high":
            a, r = self.track
            return cam([0.0, 0.0, 0.0], 2.05 * (a + r + 1.4), 90.0, -89.0)
        return cam(self.look, self.dist, self.az, self.EL)


# ---------------------------------------------------------------- air hockey (two KUKA iiwa arms)
class AirHockey(Adapter):
    """tasks/airhockey-{match,duel}-*: two iiwa arms (A at x < 0, B at x > 0) and the puck on one table, built with the
    sim's own hockey.scene_xml() (robot files in the sim's scene/). A frame's "q" is the full qpos: puck x, y, yaw, then
    A's 9 and B's 9 joints (hinges only, nothing to renormalize). The mallets get the seat colours (spectator copy
    only). Main camera: the table from a side-high angle; "high": overhead; the director adds free cameras."""
    CAMS = ("main", "overview", "high")
    HALO = 0.055
    BALL = ("q", 0, 2)
    MALLETS = {"A": "iiwa_1/ee", "B": "iiwa_2/ee"}

    def model(self, simdir):
        (hk,) = fresh_import(simdir, ["hockey"])
        m = mj.MjModel.from_xml_string(hk.scene_xml())
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, 1920), max(m.vis.global_.offheight, 1080)
        for seat, name in self.MALLETS.items():
            gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                m.geom_rgba[gid, :3] = np.asarray(audience.SEAT_RGB[seat]) / 255.0
                m.geom_matid[gid] = -1
        return m

    def ball_pos(self, rec):
        q = rec.get("q") or ()
        return None if len(q) < 2 else np.asarray([q[0], q[1], 0.012], float)

    def ball(self, scn, rec, trail, halo=None):
        """A thin glowing disc under the puck and its recent path (a sphere would hide the red puck)."""
        r = float(halo or self.HALO)
        p = self.ball_pos(rec)
        if p is None:
            return
        add_geom(scn, mj.mjtGeom.mjGEOM_CYLINDER, [r, 0.0012, 0], [p[0], p[1], 0.0013], [1.0, 0.85, 0.2, 0.55])
        n = len(trail)
        for i, tr in enumerate(trail):
            tp = self.ball_pos(tr)
            if tp is not None:
                add_geom(scn, mj.mjtGeom.mjGEOM_CYLINDER, [r * (0.3 + 0.35 * (i + 1) / n), 0.001, 0],
                         [tp[0], tp[1], 0.0012], [1.0, 0.75, 0.2, 0.12 + 0.3 * (i + 1) / n])

    def camera(self, name):
        if isinstance(name, (tuple, list)):
            return cam(*name)
        if name == "overview":
            c = mj.MjvCamera()
            c.type, c.fixedcamid = mj.mjtCamera.mjCAMERA_FIXED, mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA, "overview")
            return c
        if name == "high":
            return cam([0.0, 0.0, 0.0], 2.6, 90.0, -89.0)
        return cam([0.0, 0.04, -0.03], 2.2, 90.0, -42.0)


ADAPTERS = {("tabletennis", "single"): TableTennisSingle, ("tabletennis", "duel"): TableTennisDuel,
            ("tennis", "single"): TennisSingle, ("tennis", "duel"): TennisDuel,
            ("badminton", "single"): BadmintonSingle, ("badminton", "duel"): BadmintonDuel,
            ("baseball", "single"): BaseballSingle, ("baseball", "duel"): BaseballDuel,
            ("soccer", "single"): Soccer, ("soccer", "duel"): Soccer, ("soccer", "team"): Soccer,
            ("racing", "single"): Racing, ("racing", "duel"): Racing,
            ("airhockey", "single"): AirHockey, ("airhockey", "duel"): AirHockey}


def get(sport, mode, simdir, **opts):
    """The adapter of a match; opts are per-match scene options (racing: the track seed from the log's header)."""
    return ADAPTERS[(sport, mode)](simdir, **opts)
