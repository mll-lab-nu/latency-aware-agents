"""Render quality looks: `standard` (the competition scene, unchanged) and `showcase` (the same scene with better
lighting, materials, resolution and anti-aliasing, for demos and the blog).

The showcase model is compiled from the SAME scene files of the match (MjSpec) and only gets visual changes:
  - <visual>: off-screen size, MSAA samples, shadow map size, headlight, haze
  - lights: the scene's lights re-aimed / softened / given shadows, plus extra key and fill lights (lights are not
    part of the dynamics: no mass, no contact)
  - materials: specular, shininess, reflectance and detail textures; every textured surface keeps the mean colour of
    the standard surface (`texture_mean_error` reports the difference)
Nothing else: same bodies, geoms (type, size, pose, rgba), meshes, joints and cameras. `same_scene()` checks that
against the standard model before every showcase render, and the renderers refuse to draw if it fails.

This module is used only by the offline replay renderer (showcase/render.py) and, optionally, by the broadcast helper.
It is never imported by a sim, so the agents' camera observations cannot change with it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import mujoco as mj

QUALITIES = ("standard", "showcase")


# ------------------------------------------------------------------------------------------------ texture helpers
def _noise(h, w, scale, rng):
    """Smooth value noise in [0, 1] (bilinear upsampling of a coarse random grid)."""
    gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
    g = rng.random((gh, gw))
    y = np.linspace(0, gh - 1.001, h)
    x = np.linspace(0, gw - 1.001, w)
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _fit_mean(img, rgb):
    """Scale an RGB float image so that its mean colour is `rgb` (the standard surface colour)."""
    img = img * (np.asarray(rgb, float) / np.maximum(img.reshape(-1, 3).mean(0), 1e-6))
    return np.clip(img, 0, 1)


def _texture(spec, name, img):
    t = spec.add_texture(name=name, type=mj.mjtTexture.mjTEXTURE_2D)
    h, w, _ = img.shape
    t.width, t.height, t.nchannel = w, h, 3
    t.data = (np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes()
    return t


def sport_floor(rgb, size=1024, seed=7):
    """An indoor sports floor (fine grain + soft large-scale variation), mean colour = rgb."""
    rng = np.random.default_rng(seed)
    n = 0.55 * _noise(size, size, 128, rng) + 0.3 * _noise(size, size, 16, rng) + 0.15 * rng.random((size, size))
    img = np.repeat((0.86 + 0.14 * n)[..., None], 3, 2)
    return _fit_mean(img, rgb)


def grass(rgb1, rgb2, size=1024, seed=11):
    """Mown grass: two stripes (the standard checker's two greens) with blade noise; mean = mean of rgb1, rgb2."""
    rng = np.random.default_rng(seed)
    x = np.arange(size)[None, :]
    stripe = (x // (size // 2)) % 2 == 0
    base = np.where(stripe[..., None], np.asarray(rgb1, float), np.asarray(rgb2, float))
    base = np.broadcast_to(base, (size, size, 3)).copy()
    blades = 0.5 * rng.random((size, size)) + 0.3 * _noise(size, size, 6, rng) + 0.2 * _noise(size, size, 96, rng)
    img = base * (0.82 + 0.36 * blades)[..., None]
    return _fit_mean(img, (np.asarray(rgb1, float) + np.asarray(rgb2, float)) / 2)


# ------------------------------------------------------------------------------------------------ scene builders
def _soccer_module(simdir):
    sys.path.insert(0, simdir)
    try:
        import soccer_scene
        return soccer_scene
    finally:
        sys.path.remove(simdir)


def standard_spec(sport, simdir):
    if sport == "tabletennis":
        return mj.MjSpec.from_file(os.path.join(simdir, "scene", "tt_duel.xml"))
    if sport == "tennis":
        return mj.MjSpec.from_file(os.path.join(simdir, "scene", "tn_duel.xml"))
    if sport == "soccer":
        return _soccer_module(simdir).build_spec()
    raise ValueError(f"no scene for {sport}")


def _post_compile(sport, simdir, m):
    """What the sport does to the compiled model before drawing (soccer: team colours, head cameras)."""
    if sport == "soccer":
        sc = _soccer_module(simdir)
        for r in sc.ROBOTS:
            m.geom_rgba[m.geom(f"{r}:Trunk").id] = sc.TEAM_RGBA[r]
            c = m.camera(f"{r}:egocentric").id
            m.cam_pos[c] = sc.CAM_POS_TRUNK
            m.cam_quat[c] = sc.cam_quat()
            m.cam_fovy[c] = sc.CAM_FOVY_DEG
    return m


def _vis(spec, size, samples=8, shadow=8192):
    v = spec.visual
    v.global_.offwidth, v.global_.offheight = max(v.global_.offwidth, size[0]), max(v.global_.offheight, size[1])
    v.quality.offsamples = samples
    v.quality.shadowsize = shadow


def _mat(spec, name, **kw):
    mt = spec.add_material(name=name)
    for k, val in kw.items():
        setattr(mt, k, val)
    return mt


def _geoms(spec):
    return list(spec.geoms)


DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


def _assign(spec, g, name):
    """Give geom g the showcase material `name` without changing its colour: a geom with an explicit rgba keeps it
    (MuJoCo draws a non-default geom rgba over the material's), a geom with the default rgba gets a copy of the
    material carrying that default colour."""
    if tuple(np.round(g.rgba, 6)) != DEFAULT_RGBA:
        g.material = name
        return
    copy = name + "_default_rgba"
    if spec.material(copy) is None:
        base = spec.material(name)
        _mat(spec, copy, specular=base.specular, shininess=base.shininess, reflectance=base.reflectance,
             rgba=list(DEFAULT_RGBA))
    g.material = copy


def showcase_tabletennis(spec, size):
    """Arena look: a dark hall lit from above (a spot over the table, a shadow-casting key, robot spots, fill and rim), semi-gloss table,
    a sports floor with a faint reflection, painted-metal robots."""
    _vis(spec, size)
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse, hl.specular = [0.1, 0.1, 0.11], [0.22, 0.22, 0.22], [0.05, 0.05, 0.05]
    for li in list(spec.lights):                         # the scene's overhead light becomes the shadowed key spot
        li.type = mj.mjtLightType.mjLIGHT_SPOT
        li.pos, li.dir = [0.0, 0.0, 3.2], [0.0, 0.0, -1.0]
        li.diffuse, li.specular, li.ambient = [0.3, 0.29, 0.28], [0.3, 0.3, 0.3], [0.0, 0.0, 0.0]
        li.castshadow = False                            # (a spot's shadow map ends in a visible frustum edge)
        li.cutoff, li.exponent = 50.0, 6.0
    wb = spec.worldbody
    spec.visual.map.shadowclip = 1.6                     # the directional shadow box covers the visible floor
    wb.add_light(name="show_key", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 4], dir=[0.12, 0.08, -1.0],
                 diffuse=[0.22, 0.22, 0.21], specular=[0.15, 0.15, 0.15], castshadow=True, bulbradius=0.04)
    for y in (-1.6, 1.6):                                # spots over the two robots (no shadows)
        wb.add_light(name=f"show_spot_{y:+.0f}", type=mj.mjtLightType.mjLIGHT_SPOT, pos=[0.4, y, 3.0],
                     dir=[-0.1, 0.0, -1.0], diffuse=[0.22, 0.21, 0.2], specular=[0.2, 0.2, 0.2],
                     castshadow=False, cutoff=55.0, exponent=1.0)
    wb.add_light(name="show_fill", type=mj.mjtLightType.mjLIGHT_SPOT, pos=[3.4, 0.0, 2.0],      # camera side
                 dir=[-1.0, 0.0, -0.55], diffuse=[0.16, 0.17, 0.19], specular=[0.1, 0.1, 0.1],
                 castshadow=False, cutoff=60.0, exponent=1.0)
    wb.add_light(name="show_rim", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[-3, 0, 3], dir=[1.0, 0.0, -0.8],
                 diffuse=[0.25, 0.25, 0.3], specular=[0.3, 0.3, 0.3], castshadow=False)
    # floor: the standard checker's mean colour, as a sports floor with a faint reflection
    _texture(spec, "show_floor_tex", sport_floor((0.29, 0.29, 0.33)))
    floor = spec.material("floor")
    floor.textures = ["", "show_floor_tex"]
    floor.texrepeat = [6, 6]
    floor.specular, floor.shininess, floor.reflectance = 0.2, 0.5, 0.08
    _mat(spec, "show_table", specular=0.25, shininess=0.6, reflectance=0.04)
    _mat(spec, "show_paint", specular=0.5, shininess=0.8, reflectance=0.0)
    _mat(spec, "show_rubber", specular=0.15, shininess=0.25)
    _mat(spec, "show_ball", specular=0.5, shininess=0.9)
    _mat(spec, "show_dark", specular=0.25, shininess=0.4)
    for g in _geoms(spec):
        n = g.name or ""
        if n == "geom_table":
            _assign(spec, g, "show_table")
        elif n.startswith("paddle_blade"):
            _assign(spec, g, "show_rubber")
        elif n == "geom_ball":
            _assign(spec, g, "show_ball")
        elif g.type == mj.mjtGeom.mjGEOM_MESH:
            _assign(spec, g, "show_paint")
        elif n != "floor" and not g.material:
            _assign(spec, g, "show_dark")


def showcase_soccer(spec, size):
    """Stadium look: a low sun with soft shadows, a sky fill, mown grass (the same two greens), satin robots."""
    _vis(spec, size)
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse, hl.specular = [0.25, 0.27, 0.3], [0.24, 0.24, 0.24], [0.0, 0.0, 0.0]
    spec.visual.map.haze = 0.4
    spec.visual.rgba.haze = [0.62, 0.72, 0.85, 1.0]
    for li in list(spec.lights):                         # the sun: lower, warmer, with soft shadows
        li.dir = [0.45, 0.35, -1.0]
        li.diffuse, li.specular = [0.78, 0.74, 0.66], [0.3, 0.3, 0.3]
        li.castshadow, li.bulbradius = True, 0.06
    spec.visual.map.shadowclip = 0.6
    spec.worldbody.add_light(name="show_sky", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 10],
                             dir=[-0.4, -0.3, -1.0], diffuse=[0.16, 0.19, 0.24], specular=[0.0, 0.0, 0.0],
                             castshadow=False)
    _texture(spec, "show_grass_tex", grass((0.16, 0.52, 0.18), (0.14, 0.47, 0.16)))
    gm = spec.material("grass")
    gm.textures = ["", "show_grass_tex"]
    gm.texrepeat = [8, 1]
    gm.specular, gm.shininess, gm.reflectance = 0.08, 0.15, 0.0
    _mat(spec, "show_satin", specular=0.45, shininess=0.7)
    _mat(spec, "show_ball", specular=0.4, shininess=0.8)
    _mat(spec, "show_white", specular=0.2, shininess=0.3)
    for g in _geoms(spec):
        n = g.name or ""
        if n == "ball":
            _assign(spec, g, "show_ball")
        elif n.startswith(("post_", "bar_")):
            _assign(spec, g, "show_white")
        elif ":" in n and not g.material and not n.split(":", 1)[1].startswith("jersey"):
            _assign(spec, g, "show_satin")


def showcase_tennis(spec, size):
    """Outdoor hard court: the two scene lights become a low sun with soft shadows and a sky fill; a textured
    acrylic court (same blue) and apron (same green), satin robots."""
    _vis(spec, size)
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse, hl.specular = [0.22, 0.23, 0.26], [0.24, 0.24, 0.24], [0.0, 0.0, 0.0]
    spec.visual.map.haze = 0.5
    spec.visual.map.shadowclip = 0.8
    lights = list(spec.lights)
    sun = lights[0]
    sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.dir, sun.diffuse, sun.specular = [0.35, 0.25, -1.0], [0.5, 0.48, 0.44], [0.3, 0.3, 0.3]
    sun.castshadow, sun.bulbradius = True, 0.05
    for li in lights[1:]:
        li.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
        li.diffuse, li.castshadow = [0.22, 0.25, 0.3], False
    _texture(spec, "show_court_tex", sport_floor((0.16, 0.33, 0.60), seed=3))
    _mat(spec, "show_court", textures=["", "show_court_tex"], texrepeat=[4, 8], specular=0.2, shininess=0.35)
    _texture(spec, "show_apron_tex", sport_floor((0.195, 0.41, 0.295), seed=5))
    ap = spec.material("apron")
    ap.textures, ap.texrepeat = ["", "show_apron_tex"], [12, 12]
    ap.specular, ap.shininess = 0.15, 0.3
    _mat(spec, "show_paint", specular=0.5, shininess=0.8)
    _mat(spec, "show_dark", specular=0.25, shininess=0.4)
    for g in _geoms(spec):
        n = g.name or ""
        if n == "court":
            g.material = "show_court"                      # textured: mean colour kept (texture_mean_error)
            g.rgba = [1.0, 1.0, 1.0, 1.0]
        elif g.type == mj.mjtGeom.mjGEOM_MESH and not g.material:
            _assign(spec, g, "show_paint")
        elif n not in ("apron",) and not g.material and g.type != mj.mjtGeom.mjGEOM_PLANE:
            _assign(spec, g, "show_dark")


SHOWCASE = {"tabletennis": showcase_tabletennis, "tennis": showcase_tennis, "soccer": showcase_soccer}


def build(sport, simdir, quality="standard", size=(1920, 1080), decor=None):
    """The model to draw: `standard` = the scene as the match sees it (only the off-screen buffer enlarged so the
    requested video size fits), `showcase` = the same scene with the showcase look. `decor` (showcase only, promo
    material): a named set of surroundings OUTSIDE the play area (DECOR[sport]), added as one static, non-colliding
    body after every scene body, so all scene ids stay the same (checked by same_scene(..., decor=True))."""
    if quality not in QUALITIES:
        raise ValueError(f"render_quality must be one of {QUALITIES}")
    if decor and quality != "showcase":
        raise ValueError("decor is only available with the showcase look")
    spec = standard_spec(sport, simdir)
    if quality == "showcase":
        SHOWCASE[sport](spec, size)
        if decor:
            if decor not in DECOR.get(sport, {}):
                raise ValueError(f"no decor {decor!r} for {sport}; have {sorted(DECOR.get(sport, {}))}")
            DECOR[sport][decor](spec)
    else:
        v = spec.visual.global_
        v.offwidth, v.offheight = max(v.offwidth, size[0]), max(v.offheight, size[1])
    return _post_compile(sport, simdir, spec.compile())


# ------------------------------------------------------------------------------------------------ decor (promo only)
# Play volumes (world frame, metres: |x| < X, |y| < Y, z < Z): no decor geom may reach into them.
PLAY_AREA = {"tabletennis": (2.6, 3.2, 3.0), "soccer": (6.2, 4.9, 4.0)}   # soccer: incl. the benches (|y| 4.3)
CROWD = [(0.55, 0.16, 0.16), (0.16, 0.25, 0.55), (0.75, 0.72, 0.68), (0.2, 0.2, 0.22), (0.62, 0.5, 0.2),
         (0.3, 0.45, 0.3), (0.85, 0.85, 0.85), (0.45, 0.3, 0.45), (0.12, 0.12, 0.14), (0.7, 0.4, 0.25)]


def _decor_body(spec):
    return spec.worldbody.add_body(name="decor", pos=[0, 0, 0])


def _dbox(b, name, pos, size, rgba, mat=None, euler=None):
    g = b.add_geom(name="decor_" + name, type=mj.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(size), rgba=list(rgba),
                   contype=0, conaffinity=0, mass=0, group=0)
    if mat:
        g.material = mat
    if euler is not None:
        g.alt.euler = list(euler)
    return g


def _stand(b, tag, along, lo, hi, front, out, zf, rows, rng, colour=(0.2, 0.22, 0.27, 1), density=0.85):
    """A tiered stand with seated spectators. along='x': rows run along x from lo to hi, the front row at y=front,
    tiers step outward (y += out * 0.8 per row); along='y' likewise with x and y swapped. A spectator is a capsule
    torso and a sphere head in a random muted colour."""
    for r in range(rows):
        c = front + out * (r * 0.8 + 0.4)
        hz = (r + 1) * 0.225
        if along == "x":
            _dbox(b, f"{tag}_step{r}", ((lo + hi) / 2, c, zf + hz), ((hi - lo) / 2, 0.4, hz), colour)
        else:
            _dbox(b, f"{tag}_step{r}", (c, (lo + hi) / 2, zf + hz), (0.4, (hi - lo) / 2, hz), colour)
        n = int((hi - lo) / 0.55)
        z = zf + 2 * hz
        for i in range(n):
            if rng.random() > density:
                continue
            u = lo + (i + 0.5) * (hi - lo) / n + rng.uniform(-0.08, 0.08)
            v = c + out * rng.uniform(-0.05, 0.1)
            x, y = (u, v) if along == "x" else (v, u)
            col = CROWD[rng.integers(len(CROWD))]
            h = rng.uniform(0.9, 1.08)
            b.add_geom(name=f"decor_{tag}_p{r}_{i}", type=mj.mjtGeom.mjGEOM_CAPSULE, pos=[x, y, z + 0.42 * h],
                       size=[0.17, 0.17 * h, 0], rgba=[*col, 1], contype=0, conaffinity=0, mass=0)
            skin = [0.55, 0.42, 0.33, 1] if rng.random() < 0.6 else [0.35, 0.25, 0.2, 1]
            b.add_geom(name=f"decor_{tag}_h{r}_{i}", type=mj.mjtGeom.mjGEOM_SPHERE, pos=[x, y, z + 0.82 * h],
                       size=[0.105, 0, 0], rgba=skin, contype=0, conaffinity=0, mass=0)


def decor_tt_arena(spec):
    """Table-tennis hall: the dark blue court barriers around the playing area, an end wall with a lit LED band,
    tiered stands with spectators behind the far end and both sides, and lamp rigs. Floor level z = -0.76."""
    rng = np.random.default_rng(2026)
    _mat(spec, "decor_led", emission=0.45, specular=0.1)
    _mat(spec, "decor_lamp", emission=1.0)
    b = _decor_body(spec)
    zf = -0.76
    navy = (0.08, 0.13, 0.32, 1)
    for i, y in enumerate(np.arange(-3.0, 3.01, 1.5)):                      # court barriers (far end)
        _dbox(b, f"bar_x{i}", (-2.9, y, zf + 0.35), (0.02, 0.74, 0.35), navy)
    for s in (-1, 1):
        for i, x in enumerate(np.arange(-2.2, 2.21, 1.5)):                  # court barriers (sides)
            _dbox(b, f"bar_y{s}_{i}", (x, s * 3.5, zf + 0.35), (0.74, 0.02, 0.35), navy)
    _dbox(b, "endwall", (-11.5, 0, zf + 3.5), (0.1, 10, 3.5), (0.13, 0.14, 0.18, 1))
    for i, y in enumerate(np.arange(-8.0, 8.01, 2.0)):                      # LED band, alternating panels
        col = (0.08, 0.25, 0.6, 1) if i % 2 else (0.65, 0.15, 0.12, 1)
        _dbox(b, f"led{i}", (-6.35, y, zf + 0.45), (0.03, 0.98, 0.3), col, mat="decor_led")
    _stand(b, "far", "y", -7.5, 7.5, -6.6, -1, zf, 6, rng)                   # behind the far end
    for s in (-1, 1):
        _stand(b, f"side{s}", "x", -6.0, 4.0, s * 6.6, s, zf, 5, rng)
    for i, y in enumerate((-5.5, 5.5)):                                     # lamp rigs over the side stands
        _dbox(b, f"truss{i}", (-2.0, y, 5.2), (5.0, 0.06, 0.06), (0.25, 0.25, 0.27, 1))
        for j, x in enumerate((-6.0, -3.5, -1.0, 1.5)):
            _dbox(b, f"lamp{i}_{j}", (x, y, 5.1), (0.3, 0.3, 0.05), (1.0, 0.97, 0.9, 1), mat="decor_lamp")


def decor_soccer_stadium(spec):
    """Small stadium: LED advertising boards outside the pitch boards, stands with spectators along the far
    touchline and behind both goals, floodlight masts at the corners."""
    rng = np.random.default_rng(2026)
    _mat(spec, "decor_led", emission=0.45, specular=0.1)
    _mat(spec, "decor_lamp", emission=1.0)
    b = _decor_body(spec)
    for i, x in enumerate(np.arange(-5.0, 5.01, 2.0)):                      # LED boards along the far touchline
        col = (0.08, 0.25, 0.6, 1) if i % 2 else (0.7, 0.55, 0.08, 1)
        _dbox(b, f"led_n{i}", (x, 5.2, 0.45), (0.98, 0.04, 0.45), col, mat="decor_led")
    for s in (-1, 1):
        for i, y in enumerate((-2.8, 2.8)):
            _dbox(b, f"led_e{s}{i}", (s * 6.6, y, 0.45), (0.04, 0.98, 0.45), (0.85, 0.2, 0.15, 1), mat="decor_led")
    _stand(b, "north", "x", -9.0, 9.0, 5.8, 1, 0.0, 6, rng)
    for s in (-1, 1):
        _stand(b, f"end{s}", "y", -6.0, 6.0, s * 7.4, s, 0.0, 5, rng)
    for sx in (-1, 1):                                                     # floodlight masts
        for sy in (-1, 1):
            x, y = sx * 11.5, sy * 9.5
            b.add_geom(name=f"decor_mast{sx}{sy}", type=mj.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, 7.5],
                       size=[0.18, 7.5, 0], rgba=[0.6, 0.62, 0.65, 1], contype=0, conaffinity=0, mass=0)
            _dbox(b, f"flood{sx}{sy}", (x - sx * 0.3, y - sy * 0.3, 15.2), (0.9, 0.9, 0.5), (1.0, 0.98, 0.92, 1),
                  mat="decor_lamp")


DECOR = {"tabletennis": {"arena": decor_tt_arena}, "soccer": {"stadium": decor_soccer_stadium}}
DECOR_LABEL = "re-rendered for presentation"
ENGINE_NAMES = {"mujoco-showcase": "MuJoCo showcase", "cycles": "Blender Cycles", "isaac": "Isaac Sim RTX"}


def output_label(engine, decor=None):
    """The tag on every frame of a re-rendered (non-competition) output."""
    t = f"re-rendered: {ENGINE_NAMES.get(engine, engine)}"
    return t + (" | " + DECOR_LABEL if decor else "")


def _half_extent(m, g):
    t, sz = m.geom_type[g], m.geom_size[g]
    if t == mj.mjtGeom.mjGEOM_BOX:
        return sz.copy()
    if t == mj.mjtGeom.mjGEOM_SPHERE:
        return np.array([sz[0]] * 3)
    if t == mj.mjtGeom.mjGEOM_CAPSULE:
        return np.array([sz[0], sz[0], sz[1] + sz[0]])
    if t == mj.mjtGeom.mjGEOM_CYLINDER:
        return np.array([sz[0], sz[0], sz[1]])
    return np.array([m.geom_rbound[g]] * 3)


def decor_ok(m0, m1, sport):
    """The decor geoms (ids >= m0.ngeom): all on the static 'decor' body (the last body), no contact, no mass, named
    decor_*, unrotated, and outside the sport's play volume. Every m0 body/geom and all cameras are compared by
    same_scene."""
    bad = []
    bid = mj.mj_name2id(m1, mj.mjtObj.mjOBJ_BODY, "decor")
    if m1.ngeom > m0.ngeom and (bid < 0 or bid != m1.nbody - 1 or m1.body_weldid[bid] != 0
                                or np.any(m1.body_pos[bid]) or m1.body_mass[bid] > 0):
        bad.append("decor body")
    X, Y, Z = PLAY_AREA[sport]
    for g in range(m0.ngeom, m1.ngeom):
        n = mj.mj_id2name(m1, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if not n.startswith("decor_") or m1.geom_bodyid[g] != bid or m1.geom_contype[g] or m1.geom_conaffinity[g]:
            bad.append(f"decor geom {n or g}")
            continue
        p, h = m1.geom_pos[g], _half_extent(m1, g)
        if not np.allclose(m1.geom_quat[g], [1, 0, 0, 0]):
            h = np.array([m1.geom_rbound[g]] * 3)
        if abs(p[0]) - h[0] < X and abs(p[1]) - h[1] < Y and p[2] - h[2] < Z:
            bad.append(f"decor geom {n} inside the play volume")
    return bad


# ------------------------------------------------------------------------------------------------ invariants
SAME = ("nq", "nv", "nbody", "njnt", "ngeom", "nmesh", "ncam", "nmocap",
        "body_parentid", "body_pos", "body_quat", "body_mass", "jnt_type", "jnt_qposadr", "jnt_pos", "jnt_axis",
        "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_group", "geom_dataid",
        "mesh_vertnum", "mesh_vert", "mesh_face",
        "cam_mode", "cam_bodyid", "cam_pos", "cam_quat", "cam_fovy")


TEXTURE_TOL = 0.02     # max |mean colour difference| (0..1 RGB) of a re-textured surface


PER_BODY, PER_GEOM = ("body_",), ("geom_",)


def same_scene(m0, m1, decor=None):
    """The showcase model has the standard model's geometry, colours, kinematics and cameras. Returns a list of
    differences (empty = identical). Colours: every untextured geom is drawn with exactly the same rgba; a textured
    surface keeps its mean colour within TEXTURE_TOL. decor=<sport>: m1 may have extra bodies/geoms after all of m0's
    (the decor body); m0's bodies and geoms must be unchanged and the extras must pass decor_ok()."""
    bad = []
    for name in SAME:
        a, b = np.asarray(getattr(m0, name)), np.asarray(getattr(m1, name))
        if decor and name in ("nbody", "ngeom"):
            if int(b) < int(a):
                bad.append(name)
            continue
        if decor and name.startswith(PER_BODY + PER_GEOM):
            b = b[:len(a)]
        if np.shape(a) != np.shape(b) or not np.array_equal(a, b):
            bad.append(name)
    if decor:
        bad += decor_ok(m0, m1, decor)
    if m0.vis.global_.fovy != m1.vis.global_.fovy:
        bad.append("vis.global.fovy")
    for g in range(min(m0.ngeom, m1.ngeom)):          # the colour a viewer sees of every untextured geom
        if not _textured(m0, g) and not _textured(m1, g) and not np.allclose(effective_rgba(m0, g), effective_rgba(m1, g)):
            bad.append(f"colour of geom {mj.mj_id2name(m0, mj.mjtObj.mjOBJ_GEOM, g) or g}")
    for name, err in texture_mean_error(m0, m1).items():
        if err > TEXTURE_TOL:
            bad.append(f"mean colour of textured geom {name} ({err})")
    return bad


def effective_rgba(m, g):
    """The colour MuJoCo draws a geom with: its own rgba, or its material's when the geom rgba is the default."""
    rgba = m.geom_rgba[g]
    if m.geom_matid[g] >= 0 and np.allclose(rgba, DEFAULT_RGBA):
        return m.mat_rgba[m.geom_matid[g]]
    return rgba


def _textured(m, g):
    mid = m.geom_matid[g]
    return mid >= 0 and m.mat_texid[mid, mj.mjtTextureRole.mjTEXROLE_RGB] >= 0


def texture_mean_error(m0, m1):
    """Per textured geom: |mean colour(showcase) - mean colour(standard)| (max over RGB), what a viewer would see as
    a different colour of the surface. Only geoms whose material has a 2D texture in either model."""
    def mean_rgb(m, g):
        mid = m.geom_matid[g]
        if mid < 0:
            return None
        tid = m.mat_texid[mid, mj.mjtTextureRole.mjTEXROLE_RGB]
        if tid < 0:
            return None
        a, n = m.tex_adr[tid], m.tex_height[tid] * m.tex_width[tid] * m.tex_nchannel[tid]
        px = m.tex_data[a:a + n].reshape(-1, m.tex_nchannel[tid])[:, :3] / 255.0
        return px.mean(0) * effective_rgba(m, g)[:3]
    out = {}
    for g in range(m0.ngeom):
        c0, c1 = mean_rgb(m0, g), mean_rgb(m1, g)
        if c0 is not None or c1 is not None:
            c0 = c0 if c0 is not None else effective_rgba(m0, g)[:3]
            c1 = c1 if c1 is not None else effective_rgba(m1, g)[:3]
            out[mj.mj_id2name(m0, mj.mjtObj.mjOBJ_GEOM, g) or str(g)] = round(float(np.abs(c0 - c1).max()), 4)
    return out
