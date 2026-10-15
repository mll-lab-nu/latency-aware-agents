#!/usr/bin/env python3
"""Provenance of the ABB IRB 1100 + gantry body (bodies/abb_irb1100/): model.xml and the meshes, written here from
ABB's published dimensions. No upstream model is used.

    python -I tools/make_model.py          (needs numpy, trimesh==5.1.1; not in the image; then
                                            python -m bodies.lod build abb_irb1100 25k)

Why no upstream model (survey 2026-10-10, olympiad_review/abb_irb1100_zh.md): there is no IRB 1100 in MuJoCo
Menagerie, ros-industrial/abb or abb_experimental, PickNikRobotics/abb_ros2 or robot_descriptions; ABB's CAD
downloads carry no redistribution licence ("must not be reproduced or copied without ABB's written permission"); the
one open model (github.com/work-r-labs/robots, MIT) has meshes that are almost certainly derived from ABB's CAD
(package.xml license TODO), wrong inertials (91-96 kg for a 21.1 kg robot) and mirrored joint axes. So this model
is a CLEARLY LABELLED APPROXIMATION from public specifications (the library's rule for a robot without a licensed
model): the kinematics are exact, the shapes are simple solids of the published outer dimensions, the masses
estimates. Everything here is our own work (the repository's licence).

Kinematics: IRB 1100-4/0.58 (S1 = ABB 3HAC064993-001 rev V, section 2.3.2 "Main dimensions of IRB 1100-4/0.58"):
axis 2 at 327 mm above the base's mounting surface, lower arm 280 mm, a 10 mm offset up to axis 4, upper arm 300 mm
to the wrist centre (axis 5), 64 mm from the wrist centre to the tool flange (flange 364 mm ahead of axis 1, 617 mm
above the base at the zero pose). Axis directions and signs as ABB's (S1 section 2.6.1 "Positions at wrist center
and angle of axes 2 and 3": axis 3 at -88 degrees points the upper arm up, axis 2 at +90 the lower arm forward):
axes 1 about +z, 2, 3, 5 about +y, 4, 6 about +x (the base frame: x forward, z up). Ranges S1 2.6.1: +-230, -115/+113,
-205/+55, +-230, -125/+120, +-400 degrees (axis 6 default).

Gantry (DeepMind's table-tennis robot: "a 6 DoF ABB 1100 arm mounted on top of two Festo linear gantries"; "The x
gantry, which moves side to side across the table, is 4m long and the y gantry, which moves towards and away from
the table, is 2m long", arXiv 2408.03906 Fig. 1): a fixed frame (two lateral rails on legs, 4.2 m), the lateral
carriage with the forward beam (2.3 m) [joint gantry_y: the robot's left], the forward carriage plate the arm stands
on [joint gantry_x: the robot's forward]. Travel: forward -1.5 .. +0.5 m, lateral +-1.85 m from the home (DeepMind's
simulated ranges, Table VI: 2.06 m and 3.77 m). The plate's top (the arm's mounting surface) is MOUNT_Z = 0.76 m above
the floor: the height of a table-tennis table's top, as the Franka sports' pedestal puts the arm's base at table
height (DeepMind does not publish its height). The moving masses are estimates (the carriage plate 8 kg, the forward
beam with the lateral carriages 20 kg; no Festo model numbers are published).

Arm masses: 21.1 kg in total (S1 2.1), split over the links by their volume (an estimate): base 6.0, link 1 4.5,
link 2 3.5, link 3 3.6, link 4 2.1, link 5 1.2, link 6 0.2 kg; inertias from the collision solids.

Meshes: per link one STL made of the same solids as the collision geoms, a little larger and smoother (the
appearance: ABB white, the motors' grey caps); meshes/ (simulation) = meshes_fine/ (not in git) = these files.
Collision: capsules, cylinders and boxes (contype 4 / conaffinity 0 as every library robot; bodies/selfcollision.py
pairs them).
"""
from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DEG = math.pi / 180.0
MOUNT_Z = 0.76
# arm (S1): joint positions in the parent link's frame at the zero pose (all link frames parallel to the base frame)
J = [  # name, parent, pos, axis, range (deg)
    ("joint_1", "link_0", (0.0, 0.0, 0.0), (0, 0, 1), (-230, 230)),
    ("joint_2", "link_1", (0.0, 0.0, 0.327), (0, 1, 0), (-115, 113)),
    ("joint_3", "link_2", (0.0, 0.0, 0.280), (0, 1, 0), (-205, 55)),
    ("joint_4", "link_3", (0.0, 0.0, 0.010), (1, 0, 0), (-230, 230)),
    ("joint_5", "link_4", (0.300, 0.0, 0.0), (0, 1, 0), (-125, 120)),
    ("joint_6", "link_5", (0.0, 0.0, 0.0), (1, 0, 0), (-400, 400)),
]
FLANGE = (0.064, 0.0, 0.0)                     # link_6 frame -> the tool flange (S1: 64 mm)
MASS = {"link_0": 6.0, "link_1": 4.5, "link_2": 3.5, "link_3": 3.6, "link_4": 2.1, "link_5": 1.2, "link_6": 0.2}
ARMATURE = [0.10, 0.10, 0.05, 0.01, 0.01, 0.005]   # kg m^2 (reflected motor inertia; not published: estimates)
DAMPING = [1.0, 1.0, 0.5, 0.1, 0.1, 0.05]          # N m s / rad (not published: estimates)
# collision solids per link: (name, type, params) in the link frame
SOLIDS = {
    "link_0": [("foot", "box", dict(pos=(0, 0, 0.012), size=(0.08, 0.08, 0.012))),
               ("column", "cylinder", dict(pos=(0, 0, 0.07), size=(0.072, 0.05)))],
    "link_1": [("turret", "cylinder", dict(pos=(0, 0, 0.19), size=(0.070, 0.07))),
               ("shoulder", "capsule", dict(fromto=(0, -0.075, 0.327, 0, 0.075, 0.327), size=(0.062,)))],
    "link_2": [("lower_arm", "capsule", dict(fromto=(0, 0.0, 0.06, 0, 0.0, 0.22), size=(0.045,)))],
    "link_3": [("elbow", "capsule", dict(fromto=(0, -0.06, 0.0, 0, 0.06, 0.0), size=(0.058,))),
               ("elbow_back", "box", dict(pos=(-0.04, 0, 0.01), size=(0.04, 0.05, 0.05)))],
    "link_4": [("upper_arm", "capsule", dict(fromto=(0.07, 0, 0.0, 0.235, 0, 0.0), size=(0.042,)))],
    "link_5": [("wrist", "capsule", dict(fromto=(0, -0.042, 0.0, 0, 0.042, 0.0), size=(0.036,)))],
    "link_6": [("flange", "cylinder", dict(fromto=(0.035, 0, 0, 0.064, 0, 0), size=(0.032,)))],
}
GANTRY = {  # body: solids (in the body's frame: base = the floor under the home; the others ride on it)
    "base": [("rail_back", "box", dict(pos=(-1.62, 0, 0.55), size=(0.06, 2.10, 0.05))),
             ("rail_front", "box", dict(pos=(0.62, 0, 0.55), size=(0.06, 2.10, 0.05)))],
    "gantry_y": [("beam", "box", dict(pos=(-0.52, 0, 0.64), size=(1.14, 0.09, 0.04)))],   # top 0.68: clear of a
    "gantry_x": [("saddle", "box", dict(pos=(0, 0, 0.705), size=(0.08, 0.08, 0.025))),   # table top's underside (0.72)
                 ("plate", "box", dict(pos=(0, 0, 0.745), size=(0.10, 0.10, 0.015)))],   # the arm's 160 mm foot + 2 cm
}
LEGS = [(x, y) for x in (-1.62, 0.62) for y in (-2.05, 0.0, 2.05)]     # visual only (on the floor, outside play)
GANTRY_MASS = {"gantry_y": 20.0, "gantry_x": 8.0}
GANTRY_RANGE = {"gantry_x": (-1.5, 0.5), "gantry_y": (-1.85, 1.85)}
MAT = {"link_0": "abb_grey", "link_1": "abb_white", "link_2": "abb_white", "link_3": "abb_white",
       "link_4": "abb_white", "link_5": "abb_white", "link_6": "abb_grey", "base": "abb_frame",
       "gantry_y": "abb_frame", "gantry_x": "abb_accent"}
MATERIALS = {"abb_white": "0.93 0.93 0.92 1", "abb_grey": "0.32 0.33 0.35 1", "abb_frame": "0.62 0.64 0.67 1",
             "abb_accent": "0.20 0.40 0.85 1"}


def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def _segment_mesh(a, b, r, cap):
    import trimesh
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    m = trimesh.creation.capsule(height=L, radius=r, count=[24, 12]) if cap else \
        trimesh.creation.cylinder(radius=r, height=L, sections=32)
    if cap:                                         # trimesh's capsule runs from z = 0 to L (v4+: centred); centre it
        m.apply_translation(-m.bounds.mean(axis=0))
    z = np.array([0, 0, 1.0]); d = (b - a) / L
    T = trimesh.geometry.align_vectors(z, d)
    T[:3, 3] = (a + b) / 2
    m.apply_transform(T)
    return m


def solid_mesh(kind, p, grow=0.004):
    """the visual shape of one collision solid, `grow` larger"""
    import trimesh
    if kind == "box":
        m = trimesh.creation.box(extents=2 * (np.asarray(p["size"]) + grow))
        m.apply_translation(p["pos"])
        return m
    if kind == "cylinder" and "fromto" not in p:
        r, h = p["size"]
        a = np.asarray(p["pos"]) - (0, 0, h); b = np.asarray(p["pos"]) + (0, 0, h)
        return _segment_mesh(a, b, r + grow, cap=False)
    a, b = np.asarray(p["fromto"][:3]), np.asarray(p["fromto"][3:])
    return _segment_mesh(a, b, p["size"][0] + grow, cap=(kind == "capsule"))


def link_mesh(solids, extra=()):
    import trimesh
    parts = [solid_mesh(k, p) for _, k, p in solids] + list(extra)
    return trimesh.util.concatenate(parts)


def geom_attrs(kind, p):
    a = {"type": kind}
    if "fromto" in p:
        a["fromto"] = fmt(p["fromto"]); a["size"] = fmt(p["size"])
    else:
        a["pos"] = fmt(p["pos"]); a["size"] = fmt(p["size"])
    return a


def main():
    import trimesh
    fine, std = HERE / "meshes_fine", HERE / "meshes"
    for d in (fine, std):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    meshes = {}
    for name, solids in {**GANTRY, **SOLIDS}.items():
        extra = []
        if name == "base":                          # legs and their feet, drawn only
            for x, y in LEGS:
                extra.append(_segment_mesh((x, y, 0.0), (x, y, 0.50), 0.035, cap=False))
        meshes[name] = link_mesh(solids, extra)
    for name, m in meshes.items():
        m.export(fine / f"{name}.stl")
        shutil.copy2(fine / f"{name}.stl", std / f"{name}.stl")

    root = ET.Element("mujoco", model="abb_irb1100")
    root.append(ET.Comment(
        " Generated by bodies/abb_irb1100/tools/make_model.py from ABB's published dimensions (Product specification "
        "IRB 1100, 3HAC064993-001 rev V) and DeepMind's gantry description: an approximation from public "
        "specifications (exact kinematics, simple solids, estimated masses), no upstream model. Template of the "
        "RoboOlympics ABB IRB 1100 + gantry body: read by body.py, which prefixes every name, picks the mesh level and "
        "mounts the hand. Do not edit. "))
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    default = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": "abb"})
    ET.SubElement(default, "joint", limited="true")
    dv = ET.SubElement(default, "default", {"class": "abb_visual"})
    ET.SubElement(dv, "geom", type="mesh", group="2", contype="0", conaffinity="0", mass="0", density="0")
    dc = ET.SubElement(default, "default", {"class": "abb_collision"})
    ET.SubElement(dc, "geom", group="3", contype="4", conaffinity="0", rgba="0.9 0.2 0.2 0.3")
    asset = ET.SubElement(root, "asset")
    for n, rgba in MATERIALS.items():
        ET.SubElement(asset, "material", name=n, specular="0.3", shininess="0.5", rgba=rgba)
    for n in meshes:
        ET.SubElement(asset, "mesh", name=f"abb_{n}", file=f"{n}.stl")
    wb = ET.SubElement(root, "worldbody")

    def add_geoms(body, name, mass):
        ET.SubElement(body, "geom", {"class": "abb_visual", "name": f"{name}_vis", "mesh": f"abb_{name}",
                                     "material": MAT[name]})
        solids = {**GANTRY, **SOLIDS}[name]
        vols = []
        for _, k, p in solids:
            m = solid_mesh(k, p, grow=0.0)
            vols.append(abs(float(m.volume)))
        for (sn, k, p), v in zip(solids, vols):
            a = {"class": "abb_collision", "name": f"{name}_{sn}", **geom_attrs(k, p)}
            a["mass"] = f"{(mass * v / sum(vols)) if mass else 0.0:.6g}"
            if not mass:
                a["density"] = "0"
            ET.SubElement(body, "geom", a)

    base = ET.SubElement(wb, "body", name="base", childclass="abb")
    ET.SubElement(base, "inertial", pos="0 0 0.3", mass="60", diaginertia="40 40 40")   # fixed: never moves
    add_geoms(base, "base", None)
    gy = ET.SubElement(base, "body", name="gantry_y")
    ET.SubElement(gy, "joint", name="gantry_y", type="slide", axis="0 1 0", range=fmt(GANTRY_RANGE["gantry_y"]),
                  armature="0", damping="20")
    add_geoms(gy, "gantry_y", GANTRY_MASS["gantry_y"])
    gx = ET.SubElement(gy, "body", name="gantry_x")
    ET.SubElement(gx, "joint", name="gantry_x", type="slide", axis="1 0 0", range=fmt(GANTRY_RANGE["gantry_x"]),
                  armature="0", damping="20")
    add_geoms(gx, "gantry_x", GANTRY_MASS["gantry_x"])
    for cam, pos, xy in (("overview", (-3.2, 0.0, 2.2), "0 -1 0 0.55 0 0.83"),):
        ET.SubElement(base, "camera", name=cam, pos=fmt(pos), xyaxes=xy, fovy="60")
    parent = ET.SubElement(gx, "body", name="link_0", pos=fmt((0, 0, MOUNT_Z)))
    add_geoms(parent, "link_0", MASS["link_0"])
    bodies = {"link_0": parent}
    for i, (jn, par, pos, axis, (lo, hi)) in enumerate(J):
        ln = f"link_{i + 1}"
        b = ET.SubElement(bodies[par], "body", name=ln, pos=fmt(pos))
        ET.SubElement(b, "joint", name=jn, axis=fmt(axis), range=f"{lo * DEG:.6g} {hi * DEG:.6g}",
                      armature=f"{ARMATURE[i]}", damping=f"{DAMPING[i]}")
        add_geoms(b, ln, MASS[ln])
        bodies[ln] = b
    ET.SubElement(bodies["link_6"], "body", name="mount", pos=fmt(FLANGE), quat=fmt((math.sqrt(0.5), 0, math.sqrt(0.5), 0)))
    ET.indent(root, space="  ")
    (HERE / "model.xml").write_text(ET.tostring(root, encoding="unicode") + "\n")
    print("meshes", {n: len(m.faces) for n, m in meshes.items()})


if __name__ == "__main__":
    main()
