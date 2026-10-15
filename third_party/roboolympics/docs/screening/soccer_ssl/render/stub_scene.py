"""A stand-in SSL division B scene for the camera tests (C line), until B's field (sim/field.py) and A's robot
(bodies/ssl_robot) are ready: 9 x 6 m playing area on a 10.4 x 7.4 m green carpet, 0.01 m white lines (touch, goal,
halfway, goal-to-goal lines, centre circle of 1 m diameter, 1 x 2 m defense areas), division B goals (1.0 m wide,
0.18 m deep, 0.16 m high walls), 0.1 m boundary walls 0.3 m outside the lines, ten robots (a 0.18 m wide, 0.146 m high
body with a flat front, dark grey matte top, the standard vision pattern of camera.PATTERN) and an orange golf ball.
Every robot and the ball have a free joint, so a state is just qpos (random_state)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]                      # the repository
sys.path.insert(0, str(ROOT / "sports" / "soccer_ssl" / "sim"))
import camera  # noqa: E402

L, W = 9.0, 6.0
LINE = 0.01
ROBOT_R, ROBOT_H, FLAT = 0.09, 0.146, 0.07   # flat front at 0.07 m from the centre: estimate (A's model decides)
BALL_R = 0.02135
TEAMS = ("blue", "yellow")
N_PER_TEAM = 5


def _robot_mesh(n=48):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = ROBOT_R * math.cos(a), ROBOT_R * math.sin(a)
        if x > FLAT:
            continue
        pts.append((x, y))
    yf = math.sqrt(ROBOT_R ** 2 - FLAT ** 2)
    pts += [(FLAT, yf), (FLAT, -yf)]
    verts = []
    for x, y in pts:
        verts += [x, y, 0.0, x, y, ROBOT_H]
    return verts


def ssl_body():
    """A's robot module (bodies/ssl_robot, line A) from the checkout."""
    sys.path.insert(0, str(ROOT))
    from bodies.ssl_robot import body
    return body


def build(layout="ssl4", size=None, lights=True, offsamples=None, shadows=False, robot="stub"):
    """-> (MjSpec, camera names, robots [(team, number)]). robot: "stub" (this file's stand-in) or "ssl" (A's
    bodies/ssl_robot mounted at the origin; random_state sets its planar joints)."""
    import mujoco as mj
    spec = mj.MjSpec()
    spec.option.timestep = 0.001
    wb = spec.worldbody
    carpet = (0.08, 0.40, 0.12, 1)
    wb.add_geom(name="carpet", type=mj.mjtGeom.mjGEOM_PLANE, size=[5.2, 3.7, 0.1], rgba=carpet)
    white = (0.95, 0.95, 0.95, 1)

    def line(name, x0, y0, x1, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ln = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        wb.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, size=[ln / 2 + LINE / 2, LINE / 2, 0.0005],
                    pos=[cx, cy, 0.0005], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], rgba=white, contype=0, conaffinity=0)

    hl, hw = L / 2, W / 2
    line("touch_l", -hl, hw, hl, hw); line("touch_r", -hl, -hw, hl, -hw)
    line("goal_n", hl, -hw, hl, hw); line("goal_s", -hl, -hw, -hl, hw)
    line("halfway", 0, -hw, 0, hw); line("g2g", -hl, 0, hl, 0)
    for s in (-1, 1):                                       # defense areas 1 m deep, 2 m wide
        line(f"def{s}_a", s * hl, 1.0, s * (hl - 1.0), 1.0)
        line(f"def{s}_b", s * hl, -1.0, s * (hl - 1.0), -1.0)
        line(f"def{s}_c", s * (hl - 1.0), -1.0, s * (hl - 1.0), 1.0)
    n = 48
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        line(f"circ{i}", 0.5 * math.cos(a0), 0.5 * math.sin(a0), 0.5 * math.cos(a1), 0.5 * math.sin(a1))
    black = (0.05, 0.05, 0.05, 1)
    for s in (-1, 1):                                       # goals: inner width 1.0, depth 0.18, walls 0.02 x 0.16
        gx = s * (hl + 0.18 / 2 + LINE / 2)
        wb.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[0.01, 0.52, 0.08], pos=[s * (hl + 0.19 + LINE / 2), 0, 0.08], rgba=black)
        for t in (-1, 1):
            wb.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[0.10, 0.01, 0.08], pos=[gx + s * 0.01, t * 0.51, 0.08], rgba=black)
    for s in (-1, 1):                                       # boundary walls 0.3 m outside the lines
        wb.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[hl + 0.31, 0.01, 0.05], pos=[0, s * (hw + 0.31), 0.05], rgba=black)
        wb.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[0.01, hw + 0.31, 0.05], pos=[s * (hl + 0.31), 0, 0.05], rgba=black)

    pat = camera.PATTERN
    robots = []
    if robot == "ssl":
        body = ssl_body()
        for team in TEAMS:
            for k in range(N_PER_TEAM):
                body.mount(spec, f"{team}{k}", team, k, pose=(0.0, 0.0, 0.0))
                robots.append((team, k))
    else:
        spec.add_mesh(name="robot_body", uservert=_robot_mesh())
    for team in TEAMS if robot != "ssl" else ():
        for k in range(N_PER_TEAM):
            b = wb.add_body(name=f"{team}{k}", pos=[0, 0, 0])
            b.add_freejoint(name=f"{team}{k}_free")
            b.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname="robot_body", rgba=(0.12, 0.12, 0.12, 1), mass=2.3)
            z = ROBOT_H + 0.0005
            b.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER, size=[pat["center_radius"], 0.0005, 0], pos=[0, 0, z],
                       rgba=(*pat["rgb"][team], 1), contype=0, conaffinity=0, mass=0)
            code = pat["ids"][k]
            for key, c in zip(("FL", "FR", "BL", "BR"), code):
                x, y = pat["dots"][key]
                b.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER, size=[pat["dot_radius"], 0.0005, 0], pos=[x, y, z],
                           rgba=(*pat["rgb"][c], 1), contype=0, conaffinity=0, mass=0)
            robots.append((team, k))
    ball = wb.add_body(name="ball")
    ball.add_freejoint(name="ball_free")
    ball.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], rgba=(1.0, 0.45, 0.0, 1), mass=0.046)
    names = camera.add_cameras(spec, None, layout=layout, size=size, lights=lights)
    spec.visual.quality.offsamples = 4 if offsamples is None else int(offsamples)   # None: MuJoCo's default 4 (the sweeps' "msaa4")
    if shadows:
        for lt in spec.lights:
            lt.castshadow = True
    return spec, names, robots


SSL_FRONT = 0.0715            # A's robot: the front chord's distance from the centre (body.toml geometry.front)


def _has_joint(model, name):
    try:
        model.joint(name)
        return True
    except KeyError:
        return False


def random_state(model, rng, ball_at_robot=0.3):
    """qpos of a random legal-looking scene: robots >= 0.2 m apart inside the walls, the ball free or (probability
    ball_at_robot) touching some robot's front (the dribbler spot). -> (qpos, truth dict)."""
    qpos = model.qpos0.copy()
    planar = _has_joint(model, f"{TEAMS[0]}0_x")
    poses = []
    while len(poses) < 2 * N_PER_TEAM:
        x, y = rng.uniform(-4.65, 4.65), rng.uniform(-3.15, 3.15)
        if all(math.hypot(x - p[0], y - p[1]) > 0.2 for p in poses):
            poses.append((x, y, rng.uniform(-math.pi, math.pi)))
    truth = {"robots": [], "ball": None}
    for i, (x, y, yaw) in enumerate(poses):
        name = f"{TEAMS[i // N_PER_TEAM]}{i % N_PER_TEAM}"
        if planar:                                          # A's robot: qpos of <name>_x, _y, _yaw = world pose
            for suffix, v in (("x", x), ("y", y), ("yaw", yaw)):
                qpos[model.jnt_qposadr[model.joint(f"{name}_{suffix}").id]] = v
        else:
            a = model.jnt_qposadr[model.joint(f"{name}_free").id]
            qpos[a: a + 7] = [x, y, 0.0, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        truth["robots"].append({"team": TEAMS[i // N_PER_TEAM], "number": i % N_PER_TEAM, "x": x, "y": y, "yaw": yaw})
    if rng.random() < ball_at_robot:
        x, y, yaw = poses[rng.integers(len(poses))]
        d = (SSL_FRONT if planar else FLAT) + BALL_R + 0.002
        bx, by = x + d * math.cos(yaw), y + d * math.sin(yaw)
    else:
        while True:
            bx, by = rng.uniform(-4.7, 4.7), rng.uniform(-3.2, 3.2)
            if all(math.hypot(bx - p[0], by - p[1]) > ROBOT_R + BALL_R + 0.002 for p in poses):
                break
    j = model.jnt_qposadr[model.joint("ball_free").id]
    qpos[j: j + 7] = [bx, by, BALL_R, 1, 0, 0, 0]
    truth["ball"] = {"x": bx, "y": by}
    return qpos, truth
