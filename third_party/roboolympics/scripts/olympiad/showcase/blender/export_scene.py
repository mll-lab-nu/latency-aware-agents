"""Showcase path-traced variant, step 1 (runs in the sport's MuJoCo image, CPU): export the scene geometry and the
per-frame poses of a captured replay window (capture.py) for Blender.

    python3 export_scene.py <frames.npz> <simdir> <outdir> [--fps 50] [--t0 S] [--dur S] [--slow 1]

Writes <outdir>/scene.json (geoms, materials, lights, camera per frame, sky colours), meshes.npz (one triangle mesh
per geom, in the geom frame) and poses.npz (per frame: world position and rotation of every geom). The geometry is
the model's own: MuJoCo meshes as compiled (mesh_vert / mesh_face), primitives tessellated from geom_size. The
camera is render.Director's (the same broadcast camera as the MuJoCo renders); materials and lights come from the
showcase model (looks.py), so the Blender frame shows the same scene, geometry and viewpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mujoco as mj  # noqa: E402
import looks  # noqa: E402
import render  # noqa: E402

G = mj.mjtGeom


def uv_sphere(n=24):
    th = np.linspace(0, np.pi, n + 1)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    v = np.array([[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph])
    f = []
    m = 2 * n
    for i in range(n):
        for j in range(m):
            a, b, c, d = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            if i > 0:
                f.append((a, c, b))
            if i < n - 1:
                f.append((b, c, d))
    return v, np.array(f)


def cylinder(r, h, n=40, caps=True):
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(ph), r * np.sin(ph)], 1)
    v = np.concatenate([np.c_[ring, -h * np.ones(n)], np.c_[ring, h * np.ones(n)], [[0, 0, -h], [0, 0, h]]])
    f = []
    for j in range(n):
        k = (j + 1) % n
        f += [(j, k, n + j), (k, n + k, n + j)]
        if caps:
            f += [(2 * n, k, j), (2 * n + 1, n + j, n + k)]
    return v, np.array(f)


def capsule(r, h, n=32):
    sv, sf = uv_sphere(n // 2)
    v = sv * r
    v[:, 2] += np.where(v[:, 2] >= 0, h, -h)            # split the sphere at the equator and stretch
    return v, sf


def box(s):
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
         (1, 5, 7), (1, 7, 3)]
    return c, np.array(f)


def geom_mesh(m, g):
    t, s = m.geom_type[g], m.geom_size[g]
    if t == G.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        return m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy(), "flat"
    if t == G.mjGEOM_BOX:
        v, f = box(s)
        return v, f, "flat"
    if t == G.mjGEOM_SPHERE:
        v, f = uv_sphere()
        return v * s[0], f, "smooth"
    if t == G.mjGEOM_ELLIPSOID:
        v, f = uv_sphere()
        return v * s, f, "smooth"
    if t == G.mjGEOM_CYLINDER:
        v, f = cylinder(s[0], s[1])
        return v, f, "auto"
    if t == G.mjGEOM_CAPSULE:
        v, f = capsule(s[0], s[1])
        return v, f, "smooth"
    if t == G.mjGEOM_PLANE:
        sx, sy = (s[0] or 50.0), (s[1] or 50.0)
        return np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]], float), np.array([(0, 1, 2), (0, 2, 3)]), "flat"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames"); ap.add_argument("simdir"); ap.add_argument("outdir")
    ap.add_argument("--fps", type=float, default=50.0); ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th video frame (previews)")
    ap.add_argument("--decor", default=None, help="promo-only surroundings (looks.DECOR), see render.py")
    a = ap.parse_args()
    z = np.load(a.frames)
    q, dt, sport = z["q"], float(z["dt"]), str(z["sport"])
    mocap = z["mocap"] if "mocap" in z else None
    std = looks.build(sport, a.simdir, "standard", (1920, 1080))
    m = looks.build(sport, a.simdir, "showcase", (1920, 1080), decor=a.decor)
    if looks.same_scene(std, m, decor=sport if a.decor else None):
        raise SystemExit("showcase scene differs from the standard scene")
    d = mj.MjData(m)
    os.makedirs(a.outdir, exist_ok=True)
    vis_groups = {0, 1, 2}                                   # what the default MuJoCo render shows
    geoms, meshes = [], {}
    for g in range(m.ngeom):
        if m.geom_group[g] not in vis_groups:
            continue
        rgba = looks.effective_rgba(m, g).astype(float)
        if rgba[3] <= 0.01:
            continue
        gm = geom_mesh(m, g)
        if gm is None:
            continue
        v, f, shade = gm
        meshes[f"v{g}"], meshes[f"f{g}"] = v.astype(np.float32), f.astype(np.int32)
        mid = int(m.geom_matid[g])
        mat = dict(specular=0.5, shininess=0.5, reflectance=0.0, emission=0.0, tex=None, texrepeat=[1, 1])
        if mid >= 0:
            mat.update(specular=float(m.mat_specular[mid]), shininess=float(m.mat_shininess[mid]),
                       emission=float(m.mat_emission[mid]),
                       reflectance=float(m.mat_reflectance[mid]), texrepeat=[float(x) for x in m.mat_texrepeat[mid]])
            tid = int(m.mat_texid[mid, mj.mjtTextureRole.mjTEXROLE_RGB])
            if tid >= 0 and m.tex_type[tid] == mj.mjtTexture.mjTEXTURE_2D:
                from PIL import Image
                h, w, c = m.tex_height[tid], m.tex_width[tid], m.tex_nchannel[tid]
                px = m.tex_data[m.tex_adr[tid]:m.tex_adr[tid] + h * w * c].reshape(h, w, c)[..., :3]
                name = f"tex{tid}.png"
                if not os.path.exists(os.path.join(a.outdir, name)):
                    Image.fromarray(px).save(os.path.join(a.outdir, name))
                mat["tex"] = name
                rgba = m.mat_rgba[mid].astype(float) if np.allclose(m.geom_rgba[g], looks.DEFAULT_RGBA) else rgba
        geoms.append(dict(id=g, name=mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
                          type=int(m.geom_type[g]), size=[float(x) for x in m.geom_size[g]], rgba=[float(x) for x in rgba],
                          shade=shade, static=bool(m.body_weldid[m.geom_bodyid[g]] == 0), mat=mat))
    np.savez_compressed(os.path.join(a.outdir, "meshes.npz"), **meshes)
    # frames
    director = render.Director(sport, q, dt)
    n = len(q)
    i0 = int(round(a.t0 / dt))
    total = (n - i0) * dt * a.slow if a.dur is None else a.dur
    nf = int(total * a.fps)
    idx = [min(n - 1, i0 + int(round(i / a.fps / a.slow / dt))) for i in range(0, nf, a.every)]
    ids = [gg["id"] for gg in geoms]
    pos = np.zeros((len(idx), len(ids), 3), np.float32)
    rot = np.zeros((len(idx), len(ids), 9), np.float32)
    cams = []
    for j, i in enumerate(idx):
        d.qpos[:] = q[i]
        if mocap is not None and m.nmocap:
            d.mocap_pos[:] = mocap[i].reshape(-1, 3)
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_camlight(m, d)
        pos[j], rot[j] = d.geom_xpos[ids], d.geom_xmat[ids]
        cams.append(director.params(i))
    np.savez_compressed(os.path.join(a.outdir, "poses.npz"), pos=pos, rot=rot, step=np.array(idx))
    lights = []
    mj.mj_kinematics(m, d); mj.mj_camlight(m, d)
    for li in range(m.nlight):
        lights.append(dict(name=mj.mj_id2name(m, mj.mjtObj.mjOBJ_LIGHT, li) or f"light{li}",
                           type=int(m.light_type[li]), pos=[float(x) for x in d.light_xpos[li]],
                           dir=[float(x) for x in d.light_xdir[li]], diffuse=[float(x) for x in m.light_diffuse[li]],
                           cutoff=float(m.light_cutoff[li]), exponent=float(m.light_exponent[li]),
                           castshadow=bool(m.light_castshadow[li]), bulbradius=float(m.light_bulbradius[li])))
    sky = None
    for t in range(m.ntex):
        if m.tex_type[t] == mj.mjtTexture.mjTEXTURE_SKYBOX:
            h, w, c = m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
            px = m.tex_data[m.tex_adr[t]:m.tex_adr[t] + h * w * c].reshape(-1, w, c)
            sky = dict(top=(px[:h // 6].reshape(-1, c)[:, :3].mean(0) / 255).tolist(),
                       horizon=(px[h * 5 // 6 - h // 12: h * 5 // 6].reshape(-1, c)[:, :3].mean(0) / 255).tolist())
    json.dump(dict(sport=sport, decor=a.decor, fps=a.fps / a.every, fovy=float(m.vis.global_.fovy), geoms=geoms, lights=lights,
                   headlight=dict(ambient=[float(x) for x in m.vis.headlight.ambient],
                                  diffuse=[float(x) for x in m.vis.headlight.diffuse]),
                   sky=sky, cameras=cams, steps=[int(x) for x in idx]),
              open(os.path.join(a.outdir, "scene.json"), "w"))
    print(json.dumps(dict(geoms=len(geoms), frames=len(idx), lights=len(lights))))


if __name__ == "__main__":
    main()
