"""Showcase path-traced variant, step 2 (runs in rcb-showcase-blender, GPU): build the exported scene in Blender and
path-trace every frame with Cycles.

    blender -b --factory-startup -P cycles_render.py -- <export dir> <out dir> [--size 1920x1080] [--samples 64]
                 [--engine CYCLES|BLENDER_EEVEE_NEXT] [--frames a:b] [--device OPTIX|CUDA|CPU]

Geometry, poses and camera are taken verbatim from export_scene.py (the MuJoCo model's own meshes and the broadcast
camera, vertical field of view = the model's fovy); only the light transport differs from the MuJoCo rasterizer.
Frames are written as PNG (out/frame_00000.png ...); timings in out/blender_report.json.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
opt = dict(size="1920x1080", samples="64", engine="CYCLES", frames="", device="OPTIX")
for i in range(2, len(argv) - 1, 2):
    opt[argv[i].lstrip("-")] = argv[i + 1]
W, H = (int(x) for x in opt["size"].split("x"))
os.makedirs(out, exist_ok=True)
t_setup = time.time()
S = json.load(open(os.path.join(src, "scene.json")))
M = np.load(os.path.join(src, "meshes.npz"))
P = np.load(os.path.join(src, "poses.npz"))

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = W, H, 100
scn.render.image_settings.file_format = "PNG"
scn.view_settings.view_transform = "AgX"
scn.view_settings.look = "AgX - Medium High Contrast"
scn.render.engine = opt["engine"]
if opt["engine"] == "CYCLES":
    cp = bpy.context.preferences.addons["cycles"].preferences
    dev = opt["device"]
    if dev != "CPU":
        cp.compute_device_type = dev
        cp.get_devices()
        for dv in cp.devices:
            dv.use = dv.type == dev
        scn.cycles.device = "GPU"
    scn.cycles.samples = int(opt["samples"])
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPENIMAGEDENOISE"           # (the OptiX denoiser fails to load in the container)
    scn.cycles.denoising_use_gpu = dev != "CPU"
    scn.cycles.max_bounces = 6
    scn.cycles.use_adaptive_sampling = True
    scn.render.use_persistent_data = True


def lin(c):
    """MuJoCo colours are display (sRGB) values; Blender's base colours are linear."""
    return [((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c[:3]]


# ---------------------------------------------------------------------------------------------- world
world = bpy.data.worlds.new("sky")
scn.world = world
world.use_nodes = True
nt = world.node_tree
bg = nt.nodes["Background"]
sky = S.get("sky") or dict(top=[0.3, 0.35, 0.45], horizon=[0.55, 0.6, 0.7])
tc = nt.nodes.new("ShaderNodeTexCoord")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
ramp = nt.nodes.new("ShaderNodeValToRGB")
nt.links.new(tc.outputs["Generated"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.5, 1.0
ramp.color_ramp.elements[0].color = lin(sky["horizon"]) + [1]
ramp.color_ramp.elements[1].color = lin(sky["top"]) + [1]
nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 0.5

# ---------------------------------------------------------------------------------------------- materials
mats = {}


def material(gd):
    mt, rgba = gd["mat"], gd["rgba"]
    key = (tuple(round(x, 4) for x in rgba), mt["tex"], round(mt["specular"], 3), round(mt["shininess"], 3),
           round(mt.get("emission", 0), 3),
           round(mt["reflectance"], 3), tuple(mt["texrepeat"]))
    if key in mats:
        return mats[key]
    m = bpy.data.materials.new(f"m{len(mats)}")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin(rgba) + [1]
    b.inputs["Roughness"].default_value = max(0.08, 0.85 - 0.75 * mt["shininess"] - 0.6 * mt["reflectance"])
    b.inputs["Specular IOR Level"].default_value = min(1.0, 0.25 + 0.6 * mt["specular"])
    if mt.get("emission", 0) > 0:                          # LED boards, lamps (decor)
        b.inputs["Emission Color"].default_value = lin(rgba) + [1]
        b.inputs["Emission Strength"].default_value = 4.0 * mt["emission"]
    if rgba[3] < 0.999:
        b.inputs["Alpha"].default_value = rgba[3]
        m.blend_method = "BLEND"
    if mt["tex"]:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(os.path.join(src, mt["tex"]))
        mix = m.node_tree.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs["A"].default_value = [1, 1, 1, 1] if all(abs(x - 1) < 1e-3 for x in rgba[:3]) else lin(rgba) + [1]
        m.node_tree.links.new(tex.outputs["Color"], mix.inputs["B"])
        m.node_tree.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    mats[key] = m
    return m


# ---------------------------------------------------------------------------------------------- objects
objs = []
for gd in S["geoms"]:
    g = gd["id"]
    v, f = M[f"v{g}"], M[f"f{g}"]
    me = bpy.data.meshes.new(gd["name"])
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    if gd["mat"]["tex"] and gd["type"] == 0:               # plane: UVs repeat the texture texrepeat times
        rx, ry = gd["mat"]["texrepeat"]
        uv = me.uv_layers.new()
        sx, sy = gd["size"][0] or 50.0, gd["size"][1] or 50.0
        for loop in me.loops:
            x, y, _ = v[loop.vertex_index]
            uv.data[loop.index].uv = ((x / sx + 1) / 2 * rx, (y / sy + 1) / 2 * ry)
    if gd["shade"] in ("smooth", "auto"):
        me.shade_smooth()
    if gd["shade"] == "auto":                              # cylinders: smooth sides, sharp cap edges
        me.set_sharp_from_angle(angle=math.radians(40))
    ob = bpy.data.objects.new(gd["name"], me)
    ob.data.materials.append(material(gd))
    scn.collection.objects.link(ob)
    objs.append(ob)

# ---------------------------------------------------------------------------------------------- lights
# MuJoCo light colours are unitless; scaled to Blender watts (spot / area) and irradiance (sun) by hand-tuned
# factors so the mean brightness of the table / pitch roughly matches the MuJoCo showcase frame.
K_SUN, K_SPOT = 2.6, 900.0
for li in S["lights"]:
    col = li["diffuse"]
    e = max(col)
    if e <= 0:
        continue
    d = Vector(li["dir"]).normalized()
    rot = d.to_track_quat("-Z", "Y")
    if li["type"] == 1:                                    # directional
        ld = bpy.data.lights.new(li["name"], "SUN")
        ld.energy = K_SUN * e
        ld.angle = math.radians(2.0 + 40 * li["bulbradius"])
    else:
        ld = bpy.data.lights.new(li["name"], "SPOT")
        ld.energy = K_SPOT * e
        ld.spot_size = math.radians(min(170, 2 * li["cutoff"]))
        ld.spot_blend = 0.6
        ld.shadow_soft_size = 0.25
    ld.color = [c / e for c in col]
    ld.use_shadow = True                                   # path tracing: every light casts shadows
    ob = bpy.data.objects.new(li["name"], ld)
    ob.location = li["pos"]
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = rot
    scn.collection.objects.link(ob)

# ---------------------------------------------------------------------------------------------- camera
cd = bpy.data.cameras.new("cam")
cd.sensor_fit = "VERTICAL"
cd.angle_y = math.radians(S["fovy"])
cd.clip_start, cd.clip_end = 0.02, 200.0
cam = bpy.data.objects.new("cam", cd)
scn.collection.objects.link(cam)
scn.camera = cam


def place_camera(lookat, dist, az, el):
    a, e = math.radians(az), math.radians(el)
    fwd = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    cam.location = Vector(lookat) - dist * fwd
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = fwd.to_track_quat("-Z", "Y")


pos, rot = P["pos"], P["rot"]
nf = len(pos)
a, b = 0, nf
if opt["frames"]:
    a, b = (int(x) for x in opt["frames"].split(":"))
    b = min(b, nf)
t_setup = time.time() - t_setup
times = []
for j in range(a, b):
    for k, ob in enumerate(objs):
        R = Matrix(rot[j, k].reshape(3, 3).tolist())
        ob.matrix_world = Matrix.Translation(Vector(pos[j, k].tolist())) @ R.to_4x4()
    place_camera(*S["cameras"][j])
    scn.render.filepath = os.path.join(out, f"frame_{j:05d}.png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    times.append(time.time() - t)
    print(f"frame {j} {times[-1]:.2f}s", flush=True)
rep = dict(engine=opt["engine"], device=opt["device"], samples=int(opt["samples"]), size=[W, H], frames=len(times),
           setup_s=round(t_setup, 2), first_frame_s=round(times[0], 2) if times else None,
           mean_frame_s=round(float(np.mean(times[1:] if len(times) > 1 else times)), 2) if times else None)
json.dump(rep, open(os.path.join(out, "blender_report.json"), "w"))
print(json.dumps(rep))
