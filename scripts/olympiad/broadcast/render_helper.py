"""Broadcast render helper: runs in a helper container from the eai-olympiad image (MuJoCo + OSMesa CPU or NVIDIA EGL), started
and fed by scripts/olympiad/broadcast.py over stdin/stdout (length-prefixed pickles). It never talks to a match: it gets
the copied sim directory of a match (scene files) and the broadcast-log frames the host read from the sim's
/data/broadcast.jsonl, rebuilds the MuJoCo state (qpos + mj_forward) and draws the broadcast camera.

Requests (tuples) and replies:
  ("scene", key, simdir, sport, mode[, opts])              -> ("ok", info)    load a match's scene (opts: e.g. {"seed": 1})
  ("frame", key, rec_a, rec_b, alpha, overlay, (w, h), cam) -> ("ok", jpeg)    one live frame, rec_a..rec_b interpolated
  ("clip", key, [rec...], overlay, (w, h), fps, cam)        -> ("ok", mp4)     a replay clip, one record per video frame
An overlay with "program": True gets the program feed's graphics (program_gfx.py) instead of the plain ones; its "cam"
may be a free camera (lookat, distance, azimuth, elevation) and "blank": True skips the scene (stinger fully closed).
"""
from __future__ import annotations

import io
import os
import pickle
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapters  # noqa: E402
import program_gfx  # noqa: E402

import mujoco as mj  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def _read_exact(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def recv(f):
    n = struct.unpack("<Q", _read_exact(f, 8))[0]
    return pickle.loads(_read_exact(f, n))


def send(f, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack("<Q", len(b)) + b)
    f.flush()


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


FONTS = {}


def F(size):
    if size not in FONTS:
        FONTS[size] = font(size)
    return FONTS[size]


def mix(ra, rb, alpha):
    """The record between two logged frames: every numeric field (or list of numbers) interpolated."""
    if rb is None or alpha <= 0:
        return ra
    out = dict(ra)
    for key, v in ra.items():
        w = rb.get(key)
        if isinstance(v, list) and isinstance(w, list) and len(v) == len(w) and v and \
                all(isinstance(x, (int, float)) for x in v) and all(isinstance(x, (int, float)) for x in w):
            out[key] = (np.asarray(v, float) + (np.asarray(w, float) - np.asarray(v, float)) * alpha).tolist()
        elif isinstance(v, float) and isinstance(w, (int, float)):
            out[key] = v + (w - v) * alpha
    return out


RENDER_QUALITY = os.environ.get("RENDER_QUALITY", "standard")      # standard | showcase (scripts/olympiad/showcase)


def showcase_model(ad, simdir, sport, mode):
    """The showcase look of this match's scene (showcase/looks.py; viewer-only), or None where there is none. The
    showcase model must have the standard model's geometry, colours and cameras (looks.same_scene)."""
    if RENDER_QUALITY != "showcase" or mode != "duel":
        return None
    sys.path.insert(0, os.environ.get("RCB_SHOWCASE", "/showcase"))
    try:
        import looks
    except ImportError:
        return None
    if sport not in looks.SHOWCASE:
        return None
    m = looks.build(sport, simdir, "showcase", (ad.m.vis.global_.offwidth, ad.m.vis.global_.offheight))
    return None if looks.same_scene(ad.m, m) else m


class Scene:
    def __init__(self, simdir, sport, mode, opts=None):
        self.ad = adapters.get(sport, mode, simdir, **(opts or {}))
        m = showcase_model(self.ad, simdir, sport, mode)
        if m is not None:                       # same scene, better lights/materials: swap the adapter's model
            self.ad.m, self.ad.d = m, mj.MjData(m)
        self.quality = "showcase" if m is not None else "standard"
        self.m = self.ad.m
        self.renderers = {}
        self.sport, self.mode = sport, mode

    def renderer(self, size):
        if size not in self.renderers:
            for r in self.renderers.values():
                r.close()
            self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, size[0])
            self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, size[1])
            self.renderers = {size: mj.Renderer(self.m, height=size[1], width=size[0])}
        return self.renderers[size]

    def draw(self, rec, size, cam_name, overlay):
        if overlay.get("blank"):                       # program feed: the stinger fully covers the frame
            return program_gfx.decorate_program(program_gfx.blank(size), overlay)
        self.ad.set_state(rec)
        if self.quality == "showcase":
            mj.mj_camlight(self.m, self.ad.d)  # light poses (some adapters only run mj_kinematics)
        r = self.renderer(size)
        r.update_scene(self.ad.d, camera=self.ad.camera(overlay.get("cam") or cam_name))
        self.ad.extras(r.scene, rec)
        self.ad.ball(r.scene, rec, overlay.get("trail") or [], overlay.get("halo"))
        im = Image.fromarray(r.render())
        if overlay.get("program"):
            program_gfx.decorate_program(im, overlay)
        else:
            decorate(im, overlay)
        return im

    def frame(self, ra, rb, alpha, overlay, size, cam):
        return self.draw(mix(ra, rb, alpha), size, cam, overlay)


def decorate(im, ov):
    """The on-screen graphics: a scoreboard (top left), the mode tag (top right) and a caption (bottom)."""
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    s = max(12, H // 26)
    f, fb, fs = F(s), F(int(s * 1.25)), F(max(10, int(s * 0.8)))
    # scoreboard
    rows = ov.get("rows") or []
    if rows:
        x0, y0 = 12, 10
        wname = max(dr.textlength(n, font=f) for n, _ in rows) + 16
        wsc = max(dr.textlength(str(v), font=fb) for _, v in rows) + 16
        rh = int(s * 1.7)
        dr.rectangle([x0, y0, x0 + wname + wsc, y0 + rh * len(rows)], fill=(15, 20, 40, 210))
        dr.rectangle([x0 + wname, y0, x0 + wname + wsc, y0 + rh * len(rows)], fill=(200, 160, 30, 230))
        for i, (n, v) in enumerate(rows):
            dr.text((x0 + 8, y0 + i * rh + (rh - s) // 2 - 1), n, font=f, fill=(255, 255, 255, 255))
            dr.text((x0 + wname + 8, y0 + i * rh + (rh - int(s * 1.25)) // 2 - 2), str(v), font=fb, fill=(10, 10, 10, 255))
    tag = ov.get("tag")
    if tag:
        col = (200, 30, 30, 230) if ov.get("live") else (40, 90, 170, 230)
        tw = dr.textlength(tag, font=f)
        dr.rectangle([W - tw - 28, 10, W - 12, 10 + int(s * 1.7)], fill=col)
        dr.text((W - tw - 20, 10 + int(s * 0.35) - 1), tag, font=f, fill=(255, 255, 255, 255))
    sub = ov.get("sub")
    if sub:
        tw = dr.textlength(sub, font=fs)
        dr.text((W - tw - 14, 14 + int(s * 1.7)), sub, font=fs, fill=(255, 255, 255, 235),
                stroke_width=2, stroke_fill=(0, 0, 0, 200))
    note = ov.get("note")
    if note and rows:
        y = 10 + int(s * 1.7) * len(rows) + 4
        tw = dr.textlength(note, font=fs)
        dr.rectangle([12, y, 12 + tw + 16, y + int(s * 1.25)], fill=(15, 20, 40, 190))
        dr.text((20, y + int(s * 0.18)), note, font=fs, fill=(255, 255, 255, 255))
    cap = ov.get("caption")
    if cap:
        room = W - 48 - (dr.textlength(ov["clock"], font=fs) + 24 if ov.get("clock") else 0)
        while len(cap) > 8 and dr.textlength(cap, font=f) > room:       # fit the frame, beside the clock
            cap = cap[:-2].rstrip() + "…" if not cap.endswith("…") else cap[:-2].rstrip() + "…"
        tw = dr.textlength(cap, font=f)
        dr.rectangle([12, H - int(s * 1.9) - 10, 12 + tw + 16, H - 10], fill=(15, 20, 40, 200))
        dr.text((20, H - int(s * 1.9) - 10 + int(s * 0.4)), cap, font=f, fill=(255, 255, 255, 255))
    clock = ov.get("clock")
    if clock:
        tw = dr.textlength(clock, font=fs)
        dr.text((W - tw - 14, H - int(s * 1.2) - 10), clock, font=fs, fill=(255, 255, 255, 235),
                stroke_width=2, stroke_fill=(0, 0, 0, 200))


def jpeg(im, q=90):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=0)
    return b.getvalue()


def clip(sc, recs, overlays, size, fps, cam):
    import imageio.v2 as imageio
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        w = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2, quality=None,
                               ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                                              "-movflags", "+faststart", "-threads", "1"])
        for r, ov in zip(recs, overlays):
            w.append_data(np.asarray(sc.draw(r, size, cam, ov)))
        w.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def main():
    fin = sys.stdin.buffer
    fout = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                              # anything printed goes to stderr, the protocol keeps the real stdout
    sys.stdout = sys.stderr
    scenes = {}
    send(fout, ("ok", "ready"))
    while True:
        try:
            req = recv(fin)
        except EOFError:
            return
        t0 = time.time()
        try:
            op = req[0]
            if op == "scene":
                _, key, simdir, sport, mode, *opts = req
                scenes[key] = Scene(simdir, sport, mode, opts[0] if opts else None)
                out = {"nq": int(scenes[key].m.nq), "cams": list(scenes[key].ad.CAMS), "quality": scenes[key].quality}
            elif op == "drop":
                scenes.pop(req[1], None)
                out = None
            elif op == "frame":
                _, key, ra, rb, alpha, ov, size, cam = req
                out = jpeg(scenes[key].frame(ra, rb, alpha, ov, tuple(size), cam))
            elif op == "still":                # a PNG for checks
                _, key, ra, ov, size, cam = req
                b = io.BytesIO()
                scenes[key].frame(ra, None, 0.0, ov, tuple(size), cam).save(b, "PNG")
                out = b.getvalue()
            elif op == "clip":
                _, key, recs, ovs, size, fps, cam = req
                out = clip(scenes[key], recs, ovs, tuple(size), fps, cam)
            else:
                raise ValueError(f"unknown request {op}")
            send(fout, ("ok", out, round(time.time() - t0, 4)))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            send(fout, ("error", repr(exc)[:300], round(time.time() - t0, 4)))


if __name__ == "__main__":
    main()
