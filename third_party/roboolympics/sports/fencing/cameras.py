"""The seats' cameras (each in its own frame) and the observation image (the legacy views.py, drawn in the seat's render
worker of the v0.2 core).

Each seat has two fixed cameras, placed the same way in its own frame (fencing.py: the seat stands at x < 0 and faces
+x): `side` beside the piste (y = -9 m) looking across it, the seat's robot on the left and the opponent on the right;
`top` above and beside the piste looking down at it. A camera frame is kept every FRAME_STEPS = 2 control steps (and
at every halt and placement); the observation image has 2 rows (side, top) x 4 columns (camera frames 20 ms apart, oldest
left) of 800 x 360 pixel tiles, as in the legacy task. Every seat sees its own robot red and the opponent blue.

A Renderer keeps the tiles it drew (by camera, step and the frame's joint positions) and its last image, so a new
observation draws only the camera frames that are new since the last one: the pixels are the same as drawing every
tile again, only cheaper (CPU rendering, osmesa).
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np

import fencing as F

TILE = (800, 360)                  # width, height (as the legacy views.py)
FOVY = 45.0
FOCAL = TILE[1] / 2 / math.tan(math.radians(FOVY / 2))      # 434.6 px
CAMS = {   # own frame: lookat, distance, azimuth (deg), elevation (deg)  [MuJoCo free camera] (as the legacy views.py)
    "side": ((0.0, 0.0, 0.9), 9.0, 90.0, -5.0),
    "top": ((0.0, 0.0, 0.8), 7.5, 90.0, -70.0),
}
FRAME_STEPS = 2                    # a camera frame every 2 control steps (as the legacy service)
VIEW_BACK = (6, 4, 2, 0)           # the image's frames: control steps back from the newest one (legacy views.py)
N_FRAMES = len(VIEW_BACK)          # columns of the observation image
KEEP = 8                           # camera frames kept (legacy service: 8)
IMAGE_LAYOUT = ("2 rows x 4 columns of 800 x 360 tiles. Row 1: the side camera, row 2: the top camera (poses in `game "
                "spec`). Columns: camera frames 20 ms apart, oldest left. Your robot is red, the opponent blue; the points "
                "are yellow.")


def cam_info():
    """camera pose (own frame) and intrinsics, for /spec"""
    out = []
    for name, (la, dist, az, el) in CAMS.items():
        a, e = math.radians(az), math.radians(el)
        fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        pos = np.asarray(la) - dist * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        out.append(dict(name=name, pos=[round(float(x), 4) for x in pos], forward=[round(float(x), 5) for x in fwd],
                        right=[round(float(x), 5) for x in right], up=[round(float(x), 5) for x in up],
                        fovy_deg=FOVY, tile_px=list(TILE), focal_px=round(FOCAL, 2),
                        principal_point_px=[TILE[0] / 2, TILE[1] / 2],
                        model="pinhole: pixel u = cx + f (d.right)/(d.forward), v = cy - f (d.up)/(d.forward) for a "
                              "point at offset d from pos (u to the right, v down)"))
    return out


class Renderer:
    """Draws one seat's observation image from camera frames (qpos); owns its own MjData and GL context."""

    def __init__(self, seat: str, cache: int = 32):
        import mujoco as mj
        self.mj, self.seat = mj, seat
        self.w = F.World()
        m = self.w.m
        if seat == "B":            # B sees itself red and A blue: swap the two fencers' colours
            for g in range(m.ngeom):
                name = m.geom(g).name
                if name.startswith("A_"):
                    other = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "B_" + name[2:])
                    if other >= 0:
                        ca, cb = m.geom_rgba[g].copy(), m.geom_rgba[other].copy()
                        m.geom_rgba[g], m.geom_rgba[other] = cb, ca
        self.d = mj.MjData(m)
        self.r = mj.Renderer(m, height=TILE[1], width=TILE[0])
        self.cams = []
        sg = F.SIGN[seat]
        for la, dist, az, el in CAMS.values():
            c = mj.MjvCamera()
            c.lookat[:] = (sg * la[0], sg * la[1], la[2])
            c.distance, c.azimuth, c.elevation = dist, az + (0.0 if seat == "A" else 180.0), el
            self.cams.append(c)
        self.cache: OrderedDict = OrderedDict()
        self.size = cache
        self.drawn = 0             # tiles actually rendered (the rest came from the cache)
        self.last = (None, None)   # (the frames' keys, the PNG) of the last image

    def tile(self, ci, snap):
        from PIL import Image, ImageDraw
        q = np.ascontiguousarray(snap["qpos"], dtype=np.float64)
        key = (ci, snap["k"], hashlib.sha1(q.tobytes()).hexdigest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.w.m, self.d)
        self.r.update_scene(self.d, camera=self.cams[ci])
        im = Image.fromarray(self.r.render())
        ImageDraw.Draw(im).text((6, 4), f"k={snap['k']} t={snap['t']:.2f}s", fill=(255, 255, 255))
        img = np.asarray(im)
        self.drawn += 1
        self.cache[key] = img
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return img

    def render(self, snaps):
        """snaps: [{k, qpos, t (fencing time s)}], oldest first -> PNG bytes"""
        from core.observe import png
        keys = tuple((sn["k"], hashlib.sha1(np.ascontiguousarray(sn["qpos"], dtype=np.float64).tobytes()).hexdigest())
                     for sn in snaps)
        if self.last[0] == keys:
            return self.last[1]
        rows = [np.concatenate([self.tile(ci, sn) for sn in snaps], axis=1) for ci in range(len(self.cams))]
        out = png(np.concatenate(rows, axis=0))
        self.last = (keys, out)
        return out
