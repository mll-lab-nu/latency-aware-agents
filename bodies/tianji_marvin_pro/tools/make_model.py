#!/usr/bin/env python3
"""Provenance of the Tianji Marvin Pro body (bodies/tianji_marvin_pro/): the vendor's RoboTwin embodiment of the Marvin
Pro dual-arm robot (KLMmotion/marvin_pro_robotwin, assets/embodiments/tianji: marvin_robot.urdf + STL meshes, MIT, see
LICENSE) -> model.xml (the body template body.py reads), meshes/ (decimated STL: simulation, players' cameras, in-task
videos) and meshes_fine/ (the source meshes, unchanged; not in git: regenerate them with this script). Run once when
the source changes.

    python -I tools/make_model.py MARVIN_PRO_ROBOTWIN_DIR COMMIT        (needs numpy, trimesh, networkx, fast-simplification;
                                                                       not in the image)

There is no MJCF of the Marvin Pro (no MuJoCo Menagerie entry; the robot is not in BEHAVIOR-1K / OmniGibson): the MJCF
is written here from the URDF, joint by joint. What changes against the URDF (each item is in body.toml too):
  stand    base_link (the "ZJ_Robot" stand: a 0.75 x 0.75 m floor plate, a column, the chest block with the two arm
           flanges, the head camera block) is the fixed root body "base"; its origin is put on the floor (the URDF
           origin is 0.082 m above the plate's underside); tracking_base_link is dropped;
  arms     both 7-DOF Marvin M6 arms as in the URDF (link frames, joint axes, inertials), renamed: Joint<i>_L / _R ->
           left_arm_<i-1> / right_arm_<i-1>, Link<i>_L -> link_left_arm_<i-1>, Base_L -> a geom of the stand;
           joint ranges = the intersection of the URDF's and the vendor's controller configuration (tianjizn
           tianji-robot-SDK03 CommonConfig/ccs_m6_40.MvKDCfg: J1, J3, J5 +-170 deg are narrower than the URDF's
           +-178 deg; J2 and J4 keep the URDF's one-sided ranges, which lie inside the SDK's +-120 and -145..60 deg);
  gripper  the URDF's parallel gripper (gripper_base, two fingers, tool frame) and its wrist cameras are removed: the
           RoboOlympics wrist mount (body <arm>_mount) sits at the URDF's flange frame flange_<L/R> (0.088 m along -y
           of Link7, its +z pointing away from the wrist), where bodies/hands/ attach;
  head     a head camera pair (head_left / head_right: the Marvin Pro's Intel RealSense D435 head camera, 50 mm
           stereo baseline, 58 degree vertical field of view) at the front of the stand's head block; the head is
           rigid on the real robot, its tilt is a mount parameter here;
  dynamics joint armature 0.05 kg m^2 (J1-J4) / 0.01 (J5-J7) and damping 0.5 / 0.1 N m s/rad: not in the URDF (none
           published; derived, as the wheeled humanoid's);
  collision one capsule per arm link fitted to its visual mesh, four boxes for the stand (plate, column, chest, head):
           they only serve the ball-robot and robot-table contacts of a referee.
"""
from __future__ import annotations

import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
EMB = "assets/embodiments/tianji"
FLOOR_OFFSET = 0.082                 # base_link origin above the stand plate's underside (mesh ZJ_Robot_link min z)
DEG = math.pi / 180.0
SDK_RANGE = [(-170, 170), (-120, 120), (-170, 170), (-145, 60), (-170, 170), (-60, 60), (-90, 90)]   # MvKDCfg (deg)
ARMATURE = [0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
DAMPING = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
DECIMATE = {"ZJ_Robot_link": 14000}  # target faces per mesh (others: ARM_RATIO of the source, at least ARM_MIN: fewer
ARM_RATIO = 0.08                     # faces than that leave spikes on the forearm and wrist shells)
ARM_MIN = {"Base": 2000, "Link6": 2000, "Link4": 4000, "Link5": 4000}
ARM_MIN_DEFAULT = 2500
DROP_BELOW = {"ZJ_Robot_link": 0.02}
STAND_BOXES = {                      # (lo, hi) in the base_link frame, from the stand mesh's height slices
    "plate": ((-0.367, -0.377, -0.082), (0.387, 0.377, 0.100)),
    "column": ((-0.070, -0.060, 0.100), (0.070, 0.060, 0.900)),
    "chest": ((-0.081, -0.126, 0.900), (0.081, 0.126, 1.200)),
    "head": ((-0.043, -0.074, 1.200), (0.104, 0.074, 1.341)),
}
HEAD_POS = (0.110, 0.0, 1.290)       # the head camera's centre in the base_link frame (front face of the head block)
MATERIALS = {
    "mp_white": "0.90 0.90 0.91 1",
    "mp_grey": "0.30 0.31 0.33 1",
    "mp_stand": "0.20 0.21 0.23 1",
    "mp_accent": "0.20 0.40 0.85 1",   # the arm flanges: body.py recolours it with a team accent
}
LINK_MAT = {"Base": "mp_accent", "Link6": "mp_grey", "Link7": "mp_grey"}


def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_quat(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def origin(el):
    o = el.find("origin")
    xyz = [float(x) for x in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
    rpy = [float(x) for x in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
    return np.array(xyz), rpy_mat(*rpy)


def mesh_faces(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def decimate(src: Path, dst: Path, target: int, drop_below: float = 0.0):
    """drop_below: first drop the mesh's connected parts smaller than this (m; the stand's 4,000 screws and washers:
    with them the decimation tears the floor plate apart)."""
    import fast_simplification
    import trimesh
    v, f = mesh_faces(src)
    m = trimesh.Trimesh(v, f, process=True)
    if drop_below:
        parts = [c for c in m.split(only_watertight=False) if float(np.ptp(c.vertices, 0).max()) >= drop_below]
        m = trimesh.util.concatenate(parts)
    if len(m.faces) > target:
        v2, f2 = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                                              target_reduction=1.0 - target / len(m.faces))
        m = trimesh.Trimesh(v2, f2, process=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    m.export(dst)
    return len(m.faces)


def capsule_for(points):
    """A capsule (fromto, radius) around a point cloud: principal axis, radius = 85th percentile of the distances to
    the axis (as the wheeled humanoid's make_model.py)."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    ax = vt[0]
    t = (points - c) @ ax
    d = np.linalg.norm((points - c) - np.outer(t, ax), axis=1)
    r = float(np.percentile(d, 85))
    a, b = np.percentile(t, 2) + r * 0.5, np.percentile(t, 98) - r * 0.5
    if b < a:
        a = b = (a + b) / 2
    return c + a * ax, c + b * ax, r


def inertial(link):
    i = link.find("inertial")
    p, R = origin(i)
    I = i.find("inertia")
    g = {k: float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    M = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
    M = R @ M @ R.T
    return dict(pos=fmt(p), mass=f"{float(i.find('mass').get('value')):.6g}",
                fullinertia=fmt([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]]))


def main(src_dir: str, commit: str):
    src = Path(src_dir) / EMB
    urdf = ET.parse(src / "marvin_robot.urdf").getroot()
    links = {l.get("name"): l for l in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    def mesh_of(link):
        m = links[link].find("visual/geometry/mesh")
        return src / m.get("filename")

    # meshes: fine copies and decimated ones
    fine, coarse = HERE / "meshes_fine", HERE / "meshes"
    for d in (fine, coarse):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    used = ["base_link"] + [f"{n}_{s}" for s in "LR" for n in ("Base", *(f"Link{i}" for i in range(1, 8)))]
    nf, nf_fine = {}, {}
    for ln in used:
        p = mesh_of(ln)
        stem = p.stem
        shutil.copy2(p, fine / f"{stem}.stl")
        nf_fine[stem] = len(mesh_faces(p)[1])
        target = DECIMATE.get(stem, max(ARM_MIN.get(stem[:-2], ARM_MIN_DEFAULT), int(nf_fine[stem] * ARM_RATIO)))
        nf[stem] = decimate(p, coarse / f"{stem}.stl", target, DROP_BELOW.get(stem, 0.0))

    root = ET.Element("mujoco", model="tianji_marvin_pro")
    root.append(ET.Comment(
        f" Generated by bodies/tianji_marvin_pro/tools/make_model.py from KLMmotion/marvin_pro_robotwin {EMB}/"
        f"marvin_robot.urdf (commit {commit}, MIT, LICENSE). Template of the RoboOlympics Tianji Marvin Pro body: read "
        "by body.py, which prefixes every name, picks the mesh level and mounts the hand. Do not edit. "))
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    default = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": "mp"})
    ET.SubElement(default, "joint", armature="0.05", damping="0.5", limited="true")
    dv = ET.SubElement(default, "default", {"class": "mp_visual"})
    ET.SubElement(dv, "geom", type="mesh", group="2", contype="0", conaffinity="0", mass="0", density="0")
    dc = ET.SubElement(default, "default", {"class": "mp_collision"})
    ET.SubElement(dc, "geom", group="3", contype="4", conaffinity="0", mass="0", density="0", rgba="0.9 0.2 0.2 0.3")
    asset = ET.SubElement(root, "asset")
    for n, rgba in MATERIALS.items():
        ET.SubElement(asset, "material", name=n, specular="0.4", shininess="0.4", rgba=rgba)
    for stem in nf:
        ET.SubElement(asset, "mesh", name=f"mp_{stem}", file=f"{stem}.stl")
    wb = ET.SubElement(root, "worldbody")
    base = ET.SubElement(wb, "body", name="base", childclass="mp")
    off = np.array([0.0, 0.0, FLOOR_OFFSET])
    ET.SubElement(base, "inertial", **{**inertial(links["base_link"]), "pos": fmt(off)})
    ET.SubElement(base, "geom", {"class": "mp_visual", "name": "stand_vis", "mesh": "mp_ZJ_Robot_link",
                                 "material": "mp_stand", "pos": fmt(off)})
    for n, (lo, hi) in STAND_BOXES.items():
        lo, hi = np.array(lo) + off, np.array(hi) + off
        ET.SubElement(base, "geom", {"class": "mp_collision", "name": f"stand_{n}_col", "type": "box",
                                     "pos": fmt((lo + hi) / 2), "size": fmt((hi - lo) / 2)})
    head = ET.SubElement(base, "body", name="head", pos=fmt(np.array(HEAD_POS) + off))
    # camera axes: looking along +x (forward), image right = -y, image up = +z
    for n, y in (("head_left", 0.025), ("head_right", -0.025)):
        ET.SubElement(head, "camera", name=n, pos=fmt((0, y, 0)), xyaxes="0 -1 0 0 0 1", fovy="58")
    ET.SubElement(head, "site", name="head_center", pos="0 0 0", size="0.01", group="5")

    side_name = {"L": "left", "R": "right"}
    stats = {}
    for s in "LR":
        side = side_name[s]
        # the arm flange (Base_<s>) is fixed on the stand: a geom of the base at its frame
        p0, R0 = origin(joints[f"base_link_to_Base_{s}"])
        p0 = p0 + off
        ET.SubElement(base, "geom", {"class": "mp_visual", "name": f"{side}_arm_base_vis", "mesh": f"mp_Base_{s}",
                                     "material": "mp_accent", "pos": fmt(p0), "quat": fmt(mat_quat(R0))})
        parent, parent_p, parent_R = base, p0, R0          # Link1's joint origin is relative to Base_<s>
        for i in range(1, 8):
            j = joints[f"Joint{i}_{s}"]
            p, R = origin(j)
            if i == 1:                                     # compose Base_<s> into the first body's frame
                p, R = parent_p + parent_R @ p, parent_R @ R
            link = f"Link{i}_{s}"
            b = ET.SubElement(parent, "body", name=f"link_{side}_arm_{i - 1}", pos=fmt(p), quat=fmt(mat_quat(R)))
            ET.SubElement(b, "inertial", **inertial(links[link]))
            lo_u, hi_u = float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))
            lo = max(lo_u, SDK_RANGE[i - 1][0] * DEG)
            hi = min(hi_u, SDK_RANGE[i - 1][1] * DEG)
            stats[f"{side}_arm_{i - 1}"] = (lo_u, hi_u, lo, hi)
            ET.SubElement(b, "joint", name=f"{side}_arm_{i - 1}", axis=j.find("axis").get("xyz"),
                          range=f"{lo:.6g} {hi:.6g}", armature=f"{ARMATURE[i - 1]}", damping=f"{DAMPING[i - 1]}")
            stem = mesh_of(link).stem
            ET.SubElement(b, "geom", {"class": "mp_visual", "name": f"link_{side}_arm_{i - 1}_vis",
                                      "mesh": f"mp_{stem}", "material": LINK_MAT.get(f"Link{i}", "mp_white")})
            v, _ = mesh_faces(mesh_of(link))
            a, c, r = capsule_for(v)
            ET.SubElement(b, "geom", {"class": "mp_collision", "name": f"link_{side}_arm_{i - 1}_col",
                                      "type": "capsule", "fromto": fmt([*a, *c]), "size": f"{r:.5g}"})
            parent = b
        pf, Rf = origin(joints[f"Link7_{s}_to_flange_{s}"])
        ET.SubElement(parent, "body", name=f"{side}_mount", pos=fmt(pf), quat=fmt(mat_quat(Rf)))

    ET.indent(root, space="  ")
    (HERE / "model.xml").write_text(ET.tostring(root, encoding="unicode") + "\n")
    print("faces standard", sum(nf.values()), "fine", sum(nf_fine.values()))
    for k, v in stats.items():
        print(k, "urdf [%.4f, %.4f] -> [%.4f, %.4f]" % v)


if __name__ == "__main__":
    main(*sys.argv[1:3])
