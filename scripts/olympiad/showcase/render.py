"""Showcase step 2 (MuJoCo rasterizer; EGL on a GPU or OSMesa on CPU): draw captured replay states (capture.py) with
the standard and/or the showcase look from the same broadcast camera, and write clips, a side-by-side clip and stills.

    python3 render.py <frames.npz> <simdir> <outdir> [--quality standard|showcase|both] [--size 1920x1080]
                      [--fps 50] [--ssaa 2] [--t0 S] [--dur S] [--stills 3] [--slow 1.0]

Only the model's visual properties differ between the two looks (looks.py, checked by looks.same_scene); the states,
the camera (pose, field of view, aspect) and the frame times are shared, so every showcase frame shows exactly the
moment and viewpoint of the standard frame next to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco as mj  # noqa: E402
import looks  # noqa: E402


# ------------------------------------------------------------------------------------------------ cameras
def free_cam(lookat, dist, az, el):
    c = mj.MjvCamera()
    c.type = mj.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


class Director:
    """The broadcast camera of each sport (the 'main' camera of scripts/olympiad/broadcast/adapters.py; soccer: the
    ball-following side-high camera of the broadcast program). Deterministic: a function of the replayed states."""

    def __init__(self, sport, q, dt):
        self.sport = sport
        if sport == "tabletennis":
            self.spec = ([0.0, 0.0, 0.2], 3.7, 180.0, -16.0)
        elif sport == "tennis":
            self.spec = ([0.0, 0.0, 0.0], 19.5, 180.0, -24.0)
        elif sport == "soccer":
            look = np.zeros((len(q), 3))
            cur = np.array([np.clip(q[0, 0], -3.0, 3.0), np.clip(q[0, 1], -1.2, 1.2), 0.0])
            a = 1.0 - np.exp(-dt / 0.6)                                   # 0.6 s time constant
            for i in range(len(q)):
                tgt = np.array([np.clip(q[i, 0], -3.0, 3.0), np.clip(q[i, 1], -1.2, 1.2), 0.0])
                cur = cur + a * (tgt - cur)
                look[i] = cur
            self.look = look
        else:
            raise ValueError(sport)

    def camera(self, i):
        if self.sport == "soccer":
            return free_cam(self.look[i], 7.4, 90.0, -30.0)
        return free_cam(*self.spec)

    def params(self, i):
        """(lookat, distance, azimuth, elevation) for exporters (Blender)."""
        c = self.camera(i)
        return [float(x) for x in c.lookat], float(c.distance), float(c.azimuth), float(c.elevation)


# ------------------------------------------------------------------------------------------------ post-processing
_VIGNETTE = {}


def _curve(contrast):
    x = np.arange(256) / 255.0
    y = np.clip(x + contrast * (x - x * x) * (2 * x - 1) * 2, 0, 1)        # S-curve around mid-grey
    return [int(v * 255 + 0.5) for v in y] * 3


def post(img, bloom=0.18, vignette=0.22, contrast=0.12, sat=1.06):
    """Showcase grade on a uint8 RGB frame: soft bloom of highlights, gentle S-curve, light saturation, vignette.
    All in PIL's C routines (about 10-20 ms at 1080p)."""
    from PIL import ImageChops, ImageEnhance
    im = Image.fromarray(img)
    W, H = im.size
    if bloom > 0:
        lut = [0 if v < 178 else int(min(255, (v - 178) * 255 / 77 * bloom)) for v in range(256)] * 3
        small = im.resize((W // 4, H // 4), Image.BILINEAR).point(lut)
        small = small.filter(ImageFilter.GaussianBlur(radius=max(2, H // 160)))
        im = ImageChops.add(im, small.resize((W, H), Image.BILINEAR))
    if contrast > 0:
        im = im.point(_curve(contrast))
    if sat != 1.0:
        im = ImageEnhance.Color(im).enhance(sat)
    if vignette > 0:
        key = (W, H, vignette)
        if key not in _VIGNETTE:
            y, x = np.ogrid[-1:1:H * 1j, -1:1:W * 1j]
            v = 1.0 - vignette * np.clip((x * x * 0.8 + y * y) - 0.35, 0, None)
            _VIGNETTE[key] = Image.fromarray((np.repeat(v[..., None], 3, 2) * 255).astype(np.uint8))
        im = ImageChops.multiply(im, _VIGNETTE[key])
    return np.asarray(im)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label(im, text, sub=None):
    d = ImageDraw.Draw(im, "RGBA")
    s = max(14, im.size[1] // 30)
    f, fs = font(s), font(int(s * 0.7))
    w = d.textlength(text, font=f)
    d.rectangle([14, 14, 14 + w + 20, 14 + int(s * 1.6)], fill=(10, 14, 30, 200))
    d.text((24, 14 + int(s * 0.25)), text, font=f, fill=(255, 255, 255, 255))
    if sub:
        d.text((16, im.size[1] - int(s * 1.2)), sub, font=fs, fill=(255, 255, 255, 230), stroke_width=2,
               stroke_fill=(0, 0, 0, 180))
    return im


# ------------------------------------------------------------------------------------------------ rendering
TAG_FRAMES = True          # render.py --no-tag turns the renderer tag off (internal comparisons only)


class Look:
    def __init__(self, sport, simdir, quality, size, ssaa, decor=None):
        self.quality, self.size, self.ssaa = quality, size, ssaa if quality == "showcase" else 1
        self.decor = decor if quality == "showcase" else None
        self.tag_frames = TAG_FRAMES
        rs = (size[0] * self.ssaa, size[1] * self.ssaa)
        self.m = looks.build(sport, simdir, quality, rs, decor=self.decor)
        self.d = mj.MjData(self.m)
        self.r = mj.Renderer(self.m, height=rs[1], width=rs[0])
        if quality == "showcase":
            self.r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
            self.r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = True
            self.r.scene.flags[mj.mjtRndFlag.mjRND_HAZE] = True

    def draw(self, q, cam, mocap=None):
        self.d.qpos[:] = q
        if mocap is not None and self.m.nmocap:
            self.d.mocap_pos[:] = mocap.reshape(-1, 3)
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)
        mj.mj_camlight(self.m, self.d)                 # light and camera poses (mj_kinematics leaves them unset)
        self.r.update_scene(self.d, camera=cam)
        img = self.r.render()
        if self.ssaa > 1:
            img = np.asarray(Image.fromarray(img).resize(self.size, Image.BOX))
        if self.quality == "showcase":
            img = post(img)
        if self.quality == "showcase" and self.tag_frames:
            img = presentation_label(img, looks.output_label("mujoco-showcase", self.decor))
        return img


def presentation_label(img, text=None):
    """The tag every re-rendered frame carries (bottom right): which renderer drew it, and, with decor, that the
    surroundings are for presentation and not part of the match scene."""
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    d = ImageDraw.Draw(im, "RGBA")
    s = max(12, im.size[1] // 54)
    f = font(s)
    t = text or looks.DECOR_LABEL
    w = d.textlength(t, font=f)
    W, H = im.size
    d.rectangle([W - w - 30, H - s * 2 - 14, W - 12, H - 12], fill=(0, 0, 0, 150))
    d.text((W - w - 21, H - s * 2 - 14 + s * 0.45), t, font=f, fill=(255, 255, 255, 235))
    return np.asarray(im) if isinstance(img, np.ndarray) else im


def writer(path, fps):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2, quality=None,
                              ffmpeg_params=["-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                                             "-movflags", "+faststart"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames"); ap.add_argument("simdir"); ap.add_argument("outdir")
    ap.add_argument("--quality", default="both", choices=("standard", "showcase", "both"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--slow", type=float, default=1.0, help="slow motion factor (2 = half speed)")
    ap.add_argument("--ssaa", type=int, default=2, help="showcase supersampling factor")
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--stills", default="", help="comma-separated clip times (s) for PNG stills")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-tag", action="store_true", help="no renderer tag on showcase frames")
    ap.add_argument("--decor", default=None, help="promo-only surroundings outside the play area (looks.DECOR), "
                    "showcase look only; frames get a 're-rendered for presentation' tag")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    size = tuple(int(x) for x in a.size.split("x"))
    global TAG_FRAMES
    TAG_FRAMES = not a.no_tag
    z = np.load(a.frames)
    q, dt, sport = z["q"], float(z["dt"]), str(z["sport"])
    mocap = z["mocap"] if "mocap" in z else None
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag or (f"{sport}_{a.decor}" if a.decor else sport)
    qualities = ("standard", "showcase") if a.quality == "both" else (a.quality,)
    t_build = time.time()
    L = {qq: Look(sport, a.simdir, qq, size, a.ssaa, a.decor) for qq in qualities}
    t_build = time.time() - t_build
    report = dict(sport=sport, decor=a.decor, size=size, fps=a.fps, ssaa=a.ssaa, build_s=round(t_build, 2),
                  gl=os.environ.get("MUJOCO_GL", "?"))
    if "showcase" in L and "standard" in L:
        diff = looks.same_scene(L["standard"].m, L["showcase"].m, decor=sport if a.decor else None)
        if diff:
            raise SystemExit(f"showcase scene differs from the standard scene in: {diff}")
        report["same_scene"] = True
        report["texture_mean_error"] = looks.texture_mean_error(L["standard"].m, L["showcase"].m)
    else:                                   # still check against a freshly built standard model
        ref = looks.build(sport, a.simdir, "standard", size)
        for qq, lk in L.items():
            if lk.m is not ref and looks.same_scene(ref, lk.m, decor=sport if lk.decor else None):
                raise SystemExit("showcase scene differs from the standard scene")
        report["same_scene"] = True
    try:
        from OpenGL import GL
        report["renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
    except Exception:  # noqa: BLE001
        pass
    director = Director(sport, q, dt)
    n_steps = len(q)
    i0 = int(round(a.t0 / dt))
    total = (n_steps - i0) * dt * a.slow if a.dur is None else a.dur
    nf = int(total * a.fps)
    idx = [min(n_steps - 1, i0 + int(round(i / a.fps / a.slow / dt))) for i in range(nf)]
    stills = [float(x) for x in a.stills.split(",") if x.strip()]
    times = {qq: [] for qq in L}
    ws = {}
    if not a.no_video:
        for qq in L:
            ws[qq] = writer(os.path.join(a.outdir, f"{tag}_{qq}.mp4"), a.fps)
        if len(L) == 2:
            ws["sbs"] = writer(os.path.join(a.outdir, f"{tag}_side_by_side.mp4"), a.fps)
    for fi, i in enumerate(idx):
        cam = director.camera(i)
        mc = None if mocap is None else mocap[i]
        imgs = {}
        for qq, lk in L.items():
            t = time.time()
            imgs[qq] = lk.draw(q[i], cam, mc)
            times[qq].append(time.time() - t)
        want_still = any(abs(fi / a.fps - s) < 0.5 / a.fps for s in stills)
        if ws or want_still:
            for qq in L:
                if qq in ws:
                    ws[qq].append_data(imgs[qq])
            if len(L) == 2:
                half = (size[0] // 2, size[1] // 2)
                left = label(Image.fromarray(imgs["standard"]).resize(half, Image.LANCZOS), "standard (competition)")
                right = label(Image.fromarray(imgs["showcase"]).resize(half, Image.LANCZOS), "showcase")
                sbs = Image.new("RGB", (size[0], size[1]), (0, 0, 0))
                sbs.paste(left, (0, size[1] // 4)); sbs.paste(right, (size[0] // 2, size[1] // 4))
                d = ImageDraw.Draw(sbs)
                d.text((20, 40), f"RoboOlympics {sport}: same replay, same camera, same scene geometry",
                       font=font(size[1] // 28), fill=(235, 235, 235))
                d.text((20, size[1] - size[1] // 6), "left: competition render   right: showcase re-render "
                       "(lighting, materials, resolution, anti-aliasing, grade)", font=font(size[1] // 40),
                       fill=(180, 180, 180))
                if "sbs" in ws:
                    ws["sbs"].append_data(np.asarray(sbs))
            if want_still:
                st = int(round(fi / a.fps * 100))
                for qq in L:
                    Image.fromarray(imgs[qq]).save(os.path.join(a.outdir, f"{tag}_t{st:04d}_{qq}.png"))
                if len(L) == 2:
                    both = Image.new("RGB", (size[0], size[1] * 2))
                    both.paste(label(Image.fromarray(imgs["standard"]), "standard (competition)"), (0, 0))
                    both.paste(label(Image.fromarray(imgs["showcase"]), "showcase"), (0, size[1]))
                    both.save(os.path.join(a.outdir, f"{tag}_t{st:04d}_compare.jpg"), quality=92)
    for w in ws.values():
        w.close()
    for qq, lk in L.items():
        lk.r.close()
        ts = np.asarray(times[qq][3:] or times[qq])
        report[f"{qq}_ms_per_frame"] = round(1000 * float(ts.mean()), 1) if len(ts) else None
    report["frames"] = nf
    json.dump(report, open(os.path.join(a.outdir, f"{tag}_render_report.json"), "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
