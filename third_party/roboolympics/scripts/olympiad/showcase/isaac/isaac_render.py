"""Showcase path-traced variant on NVIDIA Isaac Sim / Omniverse RTX (the robosuite/RoboCasa route), runs in
nvcr.io/nvidia/isaac-sim:4.5.0 with its python.sh, headless, one GPU.

    /isaac-sim/python.sh isaac_render.py <export dir> <out dir> [--size 1920x1080] [--mode PathTracing|RayTracedLighting]
                         [--spp 64] [--frames a:b]

Input is showcase/blender/export_scene.py's export (not MuJoCo's USD exporter, whose colours, boxes, ball and lights
come out wrong for our scenes): the model's own triangle meshes, per-frame geom poses, the broadcast camera, the
showcase materials and lights. The USD stage is built here with pxr (UsdPreviewSurface materials with linear base
colours, UsdLux lights, a camera with the model's vertical field of view) and rendered with Replicator.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--size", default="1920x1080"); ap.add_argument("--mode", default="PathTracing")
ap.add_argument("--spp", type=int, default=64); ap.add_argument("--frames", default="")
ap.add_argument("--sun", type=float, default=2.2); ap.add_argument("--spot", type=float, default=2600.0)
ap.add_argument("--dome", type=float, default=600.0)
a = ap.parse_args()
W, H = (int(x) for x in a.size.split("x"))

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": a.mode, "width": W, "height": H, "anti_aliasing": 3})
t_start = time.time() - t0

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

S = json.load(open(os.path.join(a.src, "scene.json")))
M = np.load(os.path.join(a.src, "meshes.npz"))
P = np.load(os.path.join(a.src, "poses.npz"))
os.makedirs(a.out, exist_ok=True)

settings = carb.settings.get_settings()
if a.mode == "PathTracing":
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", a.spp)
    settings.set("/rtx/pathtracing/totalSpp", a.spp)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/pathtracing/maxBounces", 6)
else:
    settings.set("/rtx/rendermode", "RaytracedLighting")
settings.set("/rtx/post/aa/op", 3)                       # DLAA-like accumulation in RTX real-time
settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/World")


def lin(c):
    return [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c[:3]]


mats = {}


def material(gd):
    mt, rgba = gd["mat"], gd["rgba"]
    key = (tuple(round(x, 4) for x in rgba), mt["tex"], round(mt["specular"], 3), round(mt["shininess"], 3),
           round(mt.get("emission", 0), 3), round(mt["reflectance"], 3))
    if key in mats:
        return mats[key]
    path = f"/World/Looks/m{len(mats)}"
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    col = Gf.Vec3f(*lin(rgba))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        max(0.08, 0.85 - 0.75 * mt["shininess"] - 0.6 * mt["reflectance"]))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[min(1, 0.25 + 0.6 * mt["specular"])] * 3))
    if rgba[3] < 0.999:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
    if mt.get("emission", 0) > 0:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * 4 * mt["emission"] for c in lin(rgba)]))
    if mt["tex"]:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.join(a.src, mt["tex"]))
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        if not all(abs(x - 1) < 1e-3 for x in rgba[:3]):
            tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*lin(rgba), 1))
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(col)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    mats[key] = m
    return m


xforms = []
for gd in S["geoms"]:
    g = gd["id"]
    v, f = M[f"v{g}"].astype(float), M[f"f{g}"]
    path = f"/World/g{g}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(f.ravel().tolist()))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    if gd["shade"] == "smooth":                          # vertex normals: average of the adjacent face normals
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vn.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if gd["mat"]["tex"] and gd["type"] == 0:             # plane: UVs repeat the texture texrepeat times
        rx, ry = gd["mat"]["texrepeat"]
        sx, sy = gd["size"][0] or 50.0, gd["size"][1] or 50.0
        uv = np.stack([(v[:, 0] / sx + 1) / 2 * rx, (v[:, 1] / sy + 1) / 2 * ry], 1).astype(np.float32)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(gd))
    xforms.append(UsdGeom.Xformable(mesh).AddTransformOp())

# lights: the showcase lights (unitless MuJoCo colours scaled to RTX intensities) + a dome for the sky
for i, li in enumerate(S["lights"]):
    e = max(li["diffuse"])
    if e <= 0:
        continue
    d = Gf.Vec3d(*li["dir"]).GetNormalized()
    rot = Gf.Rotation(Gf.Vec3d(0, 0, -1), d)
    if li["type"] == 1:
        L = UsdLux.DistantLight.Define(stage, f"/World/light{i}")
        L.CreateIntensityAttr(a.sun * 1000 * e)
        L.CreateAngleAttr(1.0 + 20 * li["bulbradius"])
    else:
        L = UsdLux.SphereLight.Define(stage, f"/World/light{i}")
        L.CreateIntensityAttr(a.spot * e)
        L.CreateRadiusAttr(0.25)
        sh = UsdLux.ShapingAPI.Apply(L.GetPrim())
        sh.CreateShapingConeAngleAttr(min(89.0, li["cutoff"]))
        sh.CreateShapingConeSoftnessAttr(0.6)
    L.CreateColorAttr(Gf.Vec3f(*[c / e for c in li["diffuse"]]))
    xf = UsdGeom.Xformable(L)
    m4 = Gf.Matrix4d().SetRotate(rot) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*li["pos"]))
    xf.AddTransformOp().Set(m4)
sky = S.get("sky") or dict(top=[0.3, 0.35, 0.45], horizon=[0.55, 0.6, 0.7])
dome = UsdLux.DomeLight.Define(stage, "/World/sky")
dome.CreateIntensityAttr(a.dome)
dome.CreateColorAttr(Gf.Vec3f(*lin([(t + h) / 2 for t, h in zip(sky["top"], sky["horizon"])])))

# camera
cam = UsdGeom.Camera.Define(stage, "/World/cam")
focal = 24.0
vap = 2 * focal * math.tan(math.radians(S["fovy"]) / 2)
cam.CreateFocalLengthAttr(focal)
cam.CreateVerticalApertureAttr(vap)
cam.CreateHorizontalApertureAttr(vap * W / H)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 200.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def place_camera(lookat, dist, az, el):
    az, el = math.radians(az), math.radians(el)
    fwd = Gf.Vec3d(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    pos = Gf.Vec3d(*lookat) - dist * fwd
    z = -fwd                                             # USD camera looks down -Z, +Y up
    x = Gf.Cross(Gf.Vec3d(0, 0, 1), z).GetNormalized()
    y = Gf.Cross(z, x)
    m = Gf.Matrix4d(x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, pos[0], pos[1], pos[2], 1)
    cam_op.Set(m)


rp = rep.create.render_product("/World/cam", (W, H))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)

pos, rot = P["pos"], P["rot"]
nf = len(pos)
fa, fb = 0, nf
if a.frames:
    fa, fb = (int(x) for x in a.frames.split(":"))
    fb = min(fb, nf)
from PIL import Image  # noqa: E402

times = []
for j in range(fa, fb):
    for k, op in enumerate(xforms):
        R = rot[j, k].reshape(3, 3).astype(float).tolist()
        t = pos[j, k].astype(float).tolist()
        op.Set(Gf.Matrix4d(R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0,
                           R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1))
    place_camera(*S["cameras"][j])
    ts = time.time()
    rep.orchestrator.step(rt_subframes=max(1, a.spp // 16) if a.mode == "PathTracing" else 4)
    img = ann.get_data()
    times.append(time.time() - ts)
    Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(a.out, f"frame_{j:05d}.png"))
    print(f"frame {j} {times[-1]:.2f}s", flush=True)
rep_ = dict(engine="isaac-sim-4.5.0 RTX", mode=a.mode, spp=a.spp, size=[W, H], frames=len(times),
            startup_s=round(t_start, 1), first_frame_s=round(times[0], 2) if times else None,
            mean_frame_s=round(float(np.mean(times[1:] if len(times) > 1 else times)), 2) if times else None)
json.dump(rep_, open(os.path.join(a.out, "isaac_report.json"), "w"))
print(json.dumps(rep_), flush=True)
app.close()
