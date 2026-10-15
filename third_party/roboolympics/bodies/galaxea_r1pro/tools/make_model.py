#!/usr/bin/env python3
"""Provenance of the Galaxea R1 Pro body (bodies/galaxea_r1pro/): Galaxea's own R1 Pro URDF from the GalaxeaManipSim
repository (OpenGalaxea/GalaxeaManipSim, galaxea_sim/assets/r1_pro/configs/urdfs/r1pro.urdf and its meshes/*.STL;
LICENSE: Apache-2.0, "Copyright 2025 Galaxea") -> model.xml (the body template body.py reads), meshes/ (decimated
copies, checked in) and meshes_fine/ (the source meshes unchanged, NOT checked in: 31 MB; this script writes them, and
body.py falls back to meshes/ when they are missing). There is no MJCF of the R1 Pro in MuJoCo Menagerie or from
Galaxea, so this script is the URDF -> MJCF conversion. Run once when the source changes; the outputs are checked in.

    python -I tools/make_model.py GALAXEA_MANIPSIM_DIR COMMIT       (needs numpy, trimesh, fast-simplification)

What changes against the URDF (each item is documented in body.toml and in the design note):
  base     the URDF's floating base_link gets planar joints base_x, base_y (slide) and base_yaw (hinge), driven by
           force-limited servos: the three-wheel vector-steering chassis (steer_motor_joint1-3, wheel_motor_joint1-3)
           is not simulated, its six joints are fixed (the wheels are visual and do not turn or steer);
  torso    torso_joint1-3 (three pitch joints: "knee", "hip", the chest's counter-pitch) stay and are driven as one
           lift DOF by body.py (torso_joint1 = a, torso_joint2 = -2a, torso_joint3 = -a keeps the chest upright);
           torso_joint4 (yaw) is the waist; ranges are the URDF's;
  head     the R1 Pro head has no joint; zed_link (the head stereo camera, a fixed link) becomes the body "head" at the
           same place with an upright frame (x forward, z up; the zed mesh keeps its pose) so that a sport can tilt and
           turn the fixed head as a mount parameter (body.toml [mounts]); two head cameras are added (12 cm apart, 70
           degrees vertical: the R1 Pro head camera's datasheet field of view, the 2026 URDF's camera spacing);
  arms     both 7-DOF Galaxea A2 arms as in the URDF (joint ranges, axes, inertias); the G1 gripper (gripper_link, the
           fingers) and the wrist RealSense (optional on the real robot) are removed and replaced by the RoboOlympics
           wrist mount (body <side>_mount where the gripper's housing meets arm_link7: on the wrist axis, x = -0.0295,
           z = the lowest point of arm_link7's mesh; +z points away from the wrist);
  dynamics joint armature 0.02 and damping 0.3 (wrist joints 5-7: 0.005 and 0.05), as the wheeled humanoid's; the URDF
           gives none;
  inertia  the URDF's inertials, the tensors rotated into the link frame (fullinertia); the removed gripper's mass is
           not carried over (a hand brings its own);
  colour   the URDF gives every link the same lavender; here white shells, dark grey joints, a black head and chassis,
           and a team accent on the shoulder caps and the waist ring (material r1_accent);
  collision the URDF's collision meshes (the visual meshes) are replaced by one capsule per link fitted to its visual
           mesh, and boxes for the chassis and the chest: they only serve the ball-robot and robot-table contacts of
           a referee.
"""
from __future__ import annotations

import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
URDF_REL = "galaxea_sim/assets/r1_pro/configs/urdfs/r1pro.urdf"
MESH_REL = "galaxea_sim/assets/r1_pro/meshes"
DROP = ("gripper", "realsense")                      # links removed with everything below them
FACE_RATIO = 0.085                                   # meshes/: keep about 8.5 % of the faces (at least 300)
HEAD_TILT = 0.3491                                   # the URDF's zed_link looks 20 degrees down
MATERIALS = {                                        # name: rgba
    "r1_white": (0.90, 0.90, 0.91, 1.0), "r1_grey": (0.58, 0.59, 0.62, 1.0), "r1_dark": (0.16, 0.16, 0.18, 1.0),
    "r1_black": (0.06, 0.06, 0.07, 1.0), "r1_accent": (0.95, 0.45, 0.10, 1.0),
}


def material_of(link: str) -> str:
    if link in ("left_arm_base_link", "right_arm_base_link", "torso_link3"):
        return "r1_accent"
    if link.startswith(("steer_motor", "wheel_motor")) or link == "zed_link":
        return "r1_black"
    if link == "base_link":
        return "r1_dark"
    if "_arm_link" in link:
        return "r1_dark" if int(link[-1]) % 2 == 0 else "r1_white"
    return "r1_white"


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def origin(el):
    o = el.find("origin")
    xyz = [float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return np.array(xyz), rpy_mat(*rpy)


def mesh_faces(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def decimate(src: Path, dst: Path):
    import fast_simplification
    import trimesh
    v, f = mesh_faces(src)
    m = trimesh.Trimesh(v, f, process=True)
    target = max(300, int(len(m.faces) * FACE_RATIO))
    if len(m.faces) > target:
        v2, f2 = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                                              target_reduction=1.0 - target / len(m.faces))
        m = trimesh.Trimesh(v2, f2, process=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    m.export(dst)
    return len(m.faces)


def capsule_for(points):
    """A capsule (from, to, radius) around a point cloud: principal axis, radius = 85th percentile of the distances
    to the axis (the wheeled humanoid's fit)."""
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


def main(src_dir: str, commit: str):
    src_dir = Path(src_dir)
    urdf = ET.parse(src_dir / URDF_REL).getroot()
    links = {l.get("name"): l for l in urdf.findall("link")}
    joints = urdf.findall("joint")
    children: dict[str, list] = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    kept = [n for n in links if not any(k in n for k in DROP)]

    fine, coarse = HERE / "meshes_fine", HERE / "meshes"
    for d in (fine, coarse):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    nfaces, verts = {}, {}
    for n in kept:
        f = src_dir / MESH_REL / f"{n}.STL"
        shutil.copy2(f, fine / f"{n}.stl")
        nfaces[n] = decimate(f, coarse / f"{n}.stl")
        verts[n] = mesh_faces(f)[0]
    shutil.copy2(src_dir / "LICENSE", HERE / "LICENSE")

    out = ET.Element("mujoco", model="galaxea_r1pro")
    out.append(ET.Comment(
        f" Generated by bodies/galaxea_r1pro/tools/make_model.py from Galaxea's R1 Pro URDF in OpenGalaxea/"
        f"GalaxeaManipSim {URDF_REL} (commit {commit}, Apache-2.0, LICENSE). Template of the RoboOlympics Galaxea R1 "
        "Pro body: read by body.py, which prefixes every name, picks the mesh level and mounts the hand. Do not edit. "))
    ET.SubElement(out, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    df = ET.SubElement(out, "default")
    top = ET.SubElement(df, "default", {"class": "r1"})
    ET.SubElement(top, "joint", armature="0.02", damping="0.3", limited="true")
    vis = ET.SubElement(top, "default", {"class": "r1_visual"})
    ET.SubElement(vis, "geom", type="mesh", group="2", contype="0", conaffinity="0", mass="0", density="0")
    col = ET.SubElement(top, "default", {"class": "r1_collision"})
    ET.SubElement(col, "geom", group="3", contype="4", conaffinity="0", mass="0", density="0", rgba="0.9 0.2 0.2 0.3")
    wr = ET.SubElement(top, "default", {"class": "r1_wrist"})
    ET.SubElement(wr, "joint", armature="0.005", damping="0.05")
    asset = ET.SubElement(out, "asset")
    for name, rgba in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, rgba=fmt(rgba), specular="0.3", shininess="0.4")
    for n in kept:
        ET.SubElement(asset, "mesh", name=f"r1_{n}", file=f"{n}.stl")

    def inertial(b, link, R_extra=None, p_extra=None):
        inn = link.find("inertial")
        if inn is None:
            return
        p, R = origin(inn)
        I = inn.find("inertia")
        It = np.array([[float(I.get("ixx")), float(I.get("ixy")), float(I.get("ixz"))],
                       [float(I.get("ixy")), float(I.get("iyy")), float(I.get("iyz"))],
                       [float(I.get("ixz")), float(I.get("iyz")), float(I.get("izz"))]])
        It = R @ It @ R.T
        if R_extra is not None:                      # the link's frame expressed in the MJCF body's frame
            p, It = R_extra @ p + p_extra, R_extra @ It @ R_extra.T
        mass = float(inn.find("mass").get("value"))
        ET.SubElement(b, "inertial", pos=fmt(p), mass=f"{mass:.6g}",
                      fullinertia=fmt([It[0, 0], It[1, 1], It[2, 2], It[0, 1], It[0, 2], It[1, 2]]))

    def collision(b, n, P):
        if n.startswith(("steer_motor", "wheel_motor")):
            return
        if n in ("base_link", "torso_link4"):
            lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
            if n == "base_link":
                lo[2] = max(lo[2], 0.05)             # 5 cm above the floor: the base never touches it
            ET.SubElement(b, "geom", {"class": "r1_collision", "name": n + "_col", "type": "box",
                                      "pos": fmt((lo + hi) / 2), "size": fmt((hi - lo) / 2)})
        else:
            a, c, r = capsule_for(P)
            ET.SubElement(b, "geom", {"class": "r1_collision", "name": n + "_col", "type": "capsule",
                                      "fromto": fmt(np.r_[a, c]), "size": f"{r:.4f}"})

    def conv(name, parent, joint=None):
        if name == "base_link":
            b = ET.SubElement(parent, "body", name="base", childclass="r1")
            for jn, typ, ax in (("base_x", "slide", "1 0 0"), ("base_y", "slide", "0 1 0"),
                                ("base_yaw", "hinge", "0 0 1")):
                ET.SubElement(b, "joint", name=jn, type=typ, axis=ax, limited="false", damping="0", armature="0")
        else:
            p, R = origin(joint)
            if name == "zed_link":                   # the head: upright frame at the camera, tilted by the mount
                # zed_link's origin is the left camera (y = +0.06): the head's origin is the pair's centre
                off = np.array([0.0, 0.060003, 0.0])
                b = ET.SubElement(parent, "body", name="head", pos=fmt(p - off))
                Rh = np.array([[math.cos(HEAD_TILT), 0, math.sin(HEAD_TILT)], [0, 1, 0],
                               [-math.sin(HEAD_TILT), 0, math.cos(HEAD_TILT)]])
                b.set("quat", fmt(mat_quat(Rh)))
                Rg = Rh.T @ R                         # the zed mesh in the head's frame
                inertial(b, links[name], Rg, Rh.T @ off)
                ET.SubElement(b, "geom", {"class": "r1_visual", "mesh": "r1_" + name, "material": material_of(name),
                                          "pos": fmt(Rh.T @ off), "quat": fmt(mat_quat(Rg))})
                Pw = verts[name] @ Rg.T + Rh.T @ off
                a, c, r = capsule_for(Pw)
                ET.SubElement(b, "geom", {"class": "r1_collision", "name": "head_col", "type": "capsule",
                                          "fromto": fmt(np.r_[a, c]), "size": f"{r:.4f}"})
                # the stereo pair (MuJoCo cameras look along their local -z; xyaxes: x right, y up): 12 cm apart
                # (2026 URDF camera_head_left/right_link: 0.1202 m), 70 degree vertical field of view (datasheet)
                for cam, y in (("head_left", 0.06), ("head_right", -0.06)):
                    ET.SubElement(b, "camera", name=cam, pos=f"0 {y} 0", xyaxes="0 -1 0 0 0 1", fovy="70")
                ET.SubElement(b, "site", name="head_center", pos="0 0 0", size="0.01", group="5")
                return b
            b = ET.SubElement(parent, "body", name=name, pos=fmt(p))
            if not np.allclose(R, np.eye(3)):
                b.set("quat", fmt(mat_quat(R)))
            jt = joint.get("type")
            if jt in ("revolute", "continuous", "prismatic") and not name.startswith(("steer_motor", "wheel_motor")):
                lim = joint.find("limit")
                e = ET.SubElement(b, "joint", name=joint.get("name"), axis=joint.find("axis").get("xyz"),
                                  range=f"{float(lim.get('lower')):.9g} {float(lim.get('upper')):.9g}")
                if "arm_joint" in joint.get("name") and int(joint.get("name")[-1]) >= 5:
                    e.set("class", "r1_wrist")
        inertial(b, links[name])
        ET.SubElement(b, "geom", {"class": "r1_visual", "mesh": "r1_" + name, "material": material_of(name)})
        collision(b, name, verts[name])
        for j in children.get(name, []):
            c = j.find("child").get("link")
            if c in kept:
                conv(c, b, j)
        if name in ("left_arm_link7", "right_arm_link7"):
            side = name.split("_")[0]
            V = verts[name]
            near = V[np.hypot(V[:, 0] + 0.0295, V[:, 1]) < 0.03]
            z = float(near[:, 2].min()) if len(near) else float(V[:, 2].min())
            ET.SubElement(b, "body", name=f"{side}_mount", pos=fmt([-0.0295, 0.0, z]), quat="0 1 0 0")
        return b

    wb = ET.SubElement(out, "worldbody")
    conv("base_link", wb)
    ET.indent(out, space="  ")
    (HERE / "model.xml").write_text(ET.tostring(out, encoding="unicode") + "\n")
    print("faces (meshes/):", sum(nfaces.values()), " fine:", sum(len(mesh_faces(fine / f"{n}.stl")[1]) for n in kept))


if __name__ == "__main__":
    main(*sys.argv[1:3])
