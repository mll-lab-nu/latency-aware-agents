#!/usr/bin/env python3
"""Provenance of the wheeled humanoid's model (bodies/wheeled_humanoid/): MuJoCo Menagerie's Rainbow Robotics RB-Y1
(rby1m_1.2_no_gripper.xml: the RBY1-M mecanum base, Apache-2.0) -> model.xml (the body template the assembler in
body.py reads), meshes_fine/ (the Menagerie meshes, unchanged) and meshes/ (decimated copies for the simulation, the
players' cameras and the in-task videos). Run once when the source changes; the outputs are checked in.

    python -I tools/make_model.py MENAGERIE_RBY1_DIR MENAGERIE_COMMIT      (needs mujoco, numpy, trimesh,
                                                                         fast-simplification; not in the image)

What changes against the Menagerie model (each item is documented in body.toml and the design note):
  base     the free joint becomes planar joints base_x, base_y (slide) and base_yaw (hinge), driven by force-limited
           servos: the mecanum rollers are not simulated (the wheels are visual and do not turn);
  torso    torso_0 and torso_4 (the roll joints) are fixed at 0; torso_1/2/3 stay (the lift: driven as one DOF, see
           body.py) and torso_5 (waist yaw);
  head     head_0 / head_1 are fixed (the head pitch is a mount parameter); two head cameras are added;
  arms     both 7-DOF arms as in the source, joint ranges from Rainbow's URDF (rby1-sdk models/rby1m/urdf);
           the gripper is replaced by the RoboOlympics wrist mount (body <arm>_mount at the URDF's tool frame
           tool_<arm>: 0.1087 m along -z of link_<arm>_arm_6, its +z pointing away from the wrist);
  dynamics joint damping 5 N m s/rad and armature 0.01 of the Menagerie ("simulation-oriented") are replaced by
           damping 0.3 (0.05 on the wrist) and armature 0.02 (0.005 on the wrist): with damping 5 the wrist joints
           (10 N m) could not reach their datasheet speed of 6.28 rad/s;
  collision the Menagerie's collision meshes and spheres are replaced by one capsule per link fitted to its visual
           meshes (and a box for the base): they only serve the ball-robot and robot-table contacts of the referee.
"""
from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
FIXED_JOINTS = {"torso_0", "torso_4", "head_0", "head_1", "wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl"}
URDF_RANGE = {   # rby1-sdk models/rby1m/urdf/model.urdf <limit lower upper> (rad)
    "torso_1": (-0.523598776, 1.570796327), "torso_2": (-2.617993878, 1.570796327),
    "torso_3": (-0.785398163, 1.570796327), "torso_5": (-2.35619449, 2.35619449),
    **{f"{s}_arm_0": (-3.141592654, 3.141592654) for s in ("right", "left")},
    "right_arm_1": (-3.141592654, 0.017453293), "left_arm_1": (-0.017453293, 3.141592654),
    **{f"{s}_arm_2": (-3.141592654, 3.141592654) for s in ("right", "left")},
    **{f"{s}_arm_3": (-2.617993878, 0.017453293) for s in ("right", "left")},
    **{f"{s}_arm_4": (-3.141592654, 3.141592654) for s in ("right", "left")},
    **{f"{s}_arm_5": (-1.570796327, 1.919862177) for s in ("right", "left")},
    **{f"{s}_arm_6": (-2.705260340, 2.705260340) for s in ("right", "left")},
}
DECIMATE = 0.30          # meshes/: keep about 30 % of the faces (at least 300)


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def mesh_faces(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def decimate(src: Path, dst: Path):
    import fast_simplification
    import trimesh
    v, f = mesh_faces(src)
    m = trimesh.Trimesh(v, f, process=True)          # merge duplicate vertices (the OBJ repeats them per face)
    target = max(300, int(len(m.faces) * DECIMATE))
    if len(m.faces) > target:
        v2, f2 = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
                                              target_reduction=1.0 - target / len(m.faces))
        m = trimesh.Trimesh(v2, f2, process=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    m.export(dst)                                    # binary STL
    return len(m.faces)


def capsule_for(points):
    """A capsule (fromto, radius) around a point cloud: principal axis, radius = 85th percentile of the distances to
    the axis (the visual shells are thin: a tight fit, not a hull)."""
    c = points.mean(0)
    u, s, vt = np.linalg.svd(points - c, full_matrices=False)
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
    src = ET.parse(src_dir / "rby1m_1.2_no_gripper.xml").getroot()
    fine, coarse = HERE / "meshes_fine", HERE / "meshes"
    for d in (fine, coarse):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    meshes = {m.get("name"): m.get("file") for m in src.find("asset").findall("mesh")}
    nfaces = {}
    for name, f in sorted(meshes.items()):
        shutil.copy2(src_dir / "assets" / f, fine / f)
        nfaces[name] = decimate(src_dir / "assets" / f, coarse / (Path(f).stem + ".stl"))
    shutil.copy2(src_dir / "LICENSE", HERE / "LICENSE")

    out = ET.Element("mujoco", model="wheeled_humanoid")
    out.append(ET.Comment(
        f" Generated by bodies/wheeled_humanoid/tools/make_model.py from MuJoCo Menagerie rainbow_robotics_rby1/"
        f"rby1m_1.2_no_gripper.xml (commit {commit}, Apache-2.0, LICENSE). Template of the RoboOlympics wheeled "
        "humanoid: read by body.py, which prefixes every name, picks the mesh level and mounts the hand. Do not edit. "))
    ET.SubElement(out, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    df = ET.SubElement(out, "default")
    top = ET.SubElement(df, "default", {"class": "wh"})
    ET.SubElement(top, "joint", armature="0.02", damping="0.3", limited="true")
    vis = ET.SubElement(top, "default", {"class": "wh_visual"})
    ET.SubElement(vis, "geom", type="mesh", group="2", contype="0", conaffinity="0", mass="0", density="0")
    col = ET.SubElement(top, "default", {"class": "wh_collision"})
    ET.SubElement(col, "geom", group="3", contype="4", conaffinity="0", mass="0", density="0", rgba="0.9 0.2 0.2 0.3")
    wr = ET.SubElement(top, "default", {"class": "wh_wrist"})
    ET.SubElement(wr, "joint", armature="0.005", damping="0.05")
    asset = ET.SubElement(out, "asset")
    for m in src.find("asset").findall("material"):
        e = ET.SubElement(asset, "material", {k: v for k, v in m.attrib.items()})
        e.set("name", "wh_m_" + m.get("name").replace(",", "_"))
    for name, f in sorted(meshes.items()):
        ET.SubElement(asset, "mesh", name="wh_" + name, file=Path(f).stem + ".stl")

    # mesh vertices, for the collision capsules
    verts = {name: mesh_faces(src_dir / "assets" / f)[0] for name, f in meshes.items()}

    def conv(sb, parent):
        """Copy body `sb` (Menagerie) into `parent` with the changes listed in the module docstring."""
        name = sb.get("name")
        b = ET.SubElement(parent, "body", name=name)
        for k in ("pos", "quat"):
            if sb.get(k):
                b.set(k, sb.get(k))
        if name == "base":
            b.set("childclass", "wh")
        inn = sb.find("inertial")
        if inn is not None and float(inn.get("mass", "0")) > 0:
            ET.SubElement(b, "inertial", dict(inn.attrib))
        if name == "base":
            ET.SubElement(b, "joint", name="base_x", type="slide", axis="1 0 0", limited="false", damping="0",
                          armature="0")
            ET.SubElement(b, "joint", name="base_y", type="slide", axis="0 1 0", limited="false", damping="0",
                          armature="0")
            ET.SubElement(b, "joint", name="base_yaw", type="hinge", axis="0 0 1", limited="false", damping="0",
                          armature="0")
        for j in sb.findall("joint"):
            jn = j.get("name")
            if jn in FIXED_JOINTS or j.get("type") == "free":
                continue
            e = ET.SubElement(b, "joint", name=jn, axis=j.get("axis"))
            lo, hi = URDF_RANGE[jn]
            e.set("range", f"{lo:.9g} {hi:.9g}")
            if jn.endswith(("_arm_4", "_arm_5", "_arm_6")):
                e.set("class", "wh_wrist")
        pts = []
        for g in sb.findall("geom"):
            if g.get("class") == "visual" and g.get("mesh"):
                e = ET.SubElement(b, "geom", {"class": "wh_visual", "mesh": "wh_" + g.get("mesh")})
                if g.get("material"):
                    e.set("material", "wh_m_" + g.get("material").replace(",", "_"))
                for k in ("pos", "quat"):
                    if g.get(k):
                        e.set(k, g.get(k))
                pts.append(verts[g.get("mesh")])
        if pts and not name.startswith("wheel"):
            P = np.vstack(pts)
            if name == "base":
                lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
                lo[2] = max(lo[2], 0.02)                  # 2 cm above the floor: the base never touches it
                ET.SubElement(b, "geom", {"class": "wh_collision", "name": "base_col", "type": "box",
                                          "pos": fmt((lo + hi) / 2), "size": fmt((hi - lo) / 2)})
            else:
                a, c, r = capsule_for(P)
                ET.SubElement(b, "geom", {"class": "wh_collision", "name": name + "_col", "type": "capsule",
                                          "fromto": fmt(np.r_[a, c]), "size": f"{r:.4f}"})
        for c in sb.findall("body"):
            conv(c, b)
        if name in ("link_right_arm_6", "link_left_arm_6"):
            side = name.split("_")[1]
            ET.SubElement(b, "body", name=f"{side}_mount", pos="0 0 -0.1087", quat="0 1 0 0")
        if name == "link_head_2":
            # a stereo head camera: two cameras 12 cm apart, 70 degree vertical field of view (a ZED 2i class camera:
            # 120 mm baseline, 110 x 70 degrees), in front of the face (looking along the head's +x); RB-Y1's vision
            # bracket carries a stereo camera; MuJoCo cameras look along their local -z (xyaxes: x right, y up)
            for cam, y in (("head_left", 0.06), ("head_right", -0.06)):
                ET.SubElement(b, "camera", name=cam, pos=f"0.09 {y} 0.06", xyaxes="0 -1 0 0 0 1", fovy="70")
            ET.SubElement(b, "site", name="head_center", pos="0.09 0 0.06", size="0.01", group="5")
        return b

    wb = ET.SubElement(out, "worldbody")
    conv(src.find("worldbody").find("body"), wb)
    ET.indent(out, space="  ")
    (HERE / "model.xml").write_text(ET.tostring(out, encoding="unicode") + "\n")
    print("faces (meshes/):", sum(nfaces.values()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
