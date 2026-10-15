"""What one robot's front camera picture says (prototype detector of the onboard mode; numpy only, as the built-in
players may import): colour classes, the floor boundary per column (the bottom of the first thing standing on the
carpet: wall, goal, robot), white line points on the carpet, the ball, and robots (dark objects that reach above
the lens height; the team colour when the robots carry the optional band).

All geometry comes from sports/soccer_ssl/sim/onboard.py (pinhole, mount, pitch). Robot frame: x forward, y left."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from sports.soccer_ssl.sim import onboard as OB  # noqa: E402

BALL_R = 0.0215
ROBOT_H = 0.148
ROBOT_R = 0.089


class Camera:
    def __init__(self, size=OB.DEFAULT_SIZE, hfov=OB.HFOV_DEG, pitch=OB.PITCH_DEG, height=OB.CAM_HEIGHT,
                 forward=OB.CAM_FORWARD):
        self.size, self.hfov, self.pitch, self.h, self.fwd = tuple(size), hfov, pitch, height, forward
        K = OB.intrinsics(size, hfov)
        self.f, self.cx, self.cy, self.W, self.H = K["f"], K["cx"], K["cy"], K["w"], K["h"]
        self.hrow = OB.horizon_row(size, hfov, pitch)
        self.kw = dict(size=size, hfov_deg=hfov, pitch_deg=pitch, height=height, forward=forward)

    def ground(self, u, v, z=0.0):
        return OB.pixel_to_robot(u, v, z, **self.kw)

    def project(self, x, y, z):
        return OB.robot_to_pixel(x, y, z, **self.kw)


def classes(img):
    """per-pixel colour classes (the rendered picture is noise-free and evenly lit: plain thresholds suffice)."""
    a = img.astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = np.maximum(np.maximum(R, G), B), np.minimum(np.minimum(R, G), B)
    return dict(
        green=(G > R + 40) & (G > B + 30),
        white=mn > 170,
        orange=(R > 45) & (R * 10 > G * 16) & (B * 2 < R) & (R > G + 20),   # incl. the shaded lower half
        blue=(B > 70) & (B > R + 50) & (B > G + 40),
        yellow=(R > 70) & (G > 60) & (B < 50) & (np.abs(R - G) < 45),
        dark=mx < 70,
        hall=(mn > 95) & (mx - mn < 25) & (mx < 175),
    )


def floor_boundary(C, cam: Camera, step=8):
    """per sampled column: the row where the carpet ends (the bottom of the first object standing on it) and what
    it is: 'wall' (dark, below the lens height: the 0.1 m walls), 'tall' (dark reaching above the lens height:
    a robot or a goal's black frame), 'goal' (white reaching above the lens height: a goal's foam), or None.
    Returns (u, v, kind) arrays."""
    H, W = C["green"].shape
    cols = np.arange(step // 2, W, step)
    hr = int(round(cam.hrow))
    floor = C["green"] | C["orange"]
    white = C["white"]
    # white that is connected to the rows above the lens height in its column = a goal's foam (lines are flat)
    us, vs, kinds = [], [], []
    for u in cols:
        fl = floor[:, u]
        wh = white[:, u]
        nf = ~(fl | wh)
        # tall white: the white run that contains row hr-2
        tallw = np.zeros(H, bool)
        r0 = max(hr - 2, 0)
        if wh[r0]:
            v = r0
            while v < H and wh[v]:
                tallw[v] = True
                v += 1
        nf = nf | tallw
        # lowest row v (largest index) with nf[v] and nf[v-1] (a 2-pixel run, robust to anti-aliasing)
        both = nf[1:] & nf[:-1]
        idx = np.nonzero(both)[0]
        if len(idx) == 0:
            continue
        v = int(idx[-1] + 1)
        if v < hr - 1:
            continue        # nothing standing on the carpet below the lens height (hall seen over the carpet)
        if tallw[v]:
            kind = "goal"
        else:
            # does the dark/non-floor run starting at v reach above the lens height?
            top = v
            while top > 0 and nf[top - 1] and not C["hall"][top - 1, u]:
                top -= 1
            kind = "tall" if top < hr - 1 else "wall"
        us.append(int(u))
        vs.append(v + 0.5)
        kinds.append(kind)
    return np.array(us), np.array(vs, float), kinds


def line_points(C, cam: Camera, max_range=2.5, max_n=60, rng=None):
    """white pixels on the carpet below the lens height (field lines), as robot-frame ground points within
    max_range of the lens; at most max_n (a random subset)."""
    hr = cam.hrow
    W = C["white"].copy()
    v0 = int(math.ceil(hr + 6))
    W[:v0] = False
    # only white with carpet nearby (a line) - excludes the goal foam and the robots' white rollers
    g = C["green"]
    near = np.zeros_like(W)
    near[:, 3:] |= g[:, :-3]
    near[:, :-3] |= g[:, 3:]
    near[3:, :] |= g[:-3, :]
    near[:-3, :] |= g[3:, :]
    W &= near
    vs, us = np.nonzero(W)
    if len(us) == 0:
        return np.zeros((0, 2))
    x, y = cam.ground(us + 0.5, vs + 0.5, 0.0)
    rr = np.hypot(x - cam.fwd, y)
    ok = np.isfinite(x) & (rr < max_range)
    pts = np.stack([x[ok], y[ok]], 1)
    if len(pts) > max_n:
        rng = rng or np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), max_n, replace=False)]
    return pts


def _components_1d(mask_cols):
    """runs of True in a 1-D array: list of (start, end_inclusive)."""
    out, start = [], None
    for i, m in enumerate(mask_cols):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask_cols) - 1))
    return out


def ball(C, cam: Camera):
    """the ball (the largest orange blob): robot-frame (x, y), pixel size, or None. The centre from the blob's
    centroid column and the middle of its top and bottom rows, back-projected at the ball centre's height."""
    m = C["orange"]
    vs, us = np.nonzero(m)
    if len(us) < 3:
        return None
    # the biggest cluster along u (only one ball exists; a split blob = partly hidden)
    colhit = np.zeros(m.shape[1], bool)
    colhit[np.unique(us)] = True
    runs = _components_1d(colhit)
    best = max(runs, key=lambda r: ((us >= r[0]) & (us <= r[1])).sum())
    sel = (us >= best[0]) & (us <= best[1])
    us, vs = us[sel], vs[sel]
    if len(us) < 3:
        return None
    uc = 0.5 * (us.min() + us.max() + 1)
    vtop, vbot = vs.min(), vs.max() + 1
    width = best[1] - best[0] + 1
    # if the blob is clipped at the picture's edge, the vertical extent is still right: use rows
    vc = 0.5 * (vtop + vbot)
    x, y = cam.ground(uc, vc, BALL_R)
    if not np.isfinite(x):
        return None
    # cross-check with the apparent size (diameter in px -> range)
    rng_size = 2 * BALL_R * cam.f / max(vbot - vtop, 1)
    return dict(x=float(x), y=float(y), px=int(len(us)), w=int(width), h=int(vbot - vtop),
                range_size=float(rng_size), u=float(uc), v=float(vc))


def robots(C, cam: Camera, min_cols=3, clean=True):
    """robots: column runs where something dark (or team coloured) stands above the lens height; each run split
    where its bottom row jumps (one robot in front of another). Per robot: robot-frame (x, y) of its centre (the
    bottom edge's nearest point + one radius), pixel width, the team colour seen (blue / yellow / None), and
    whether it touches the picture's edge (partly outside the view)."""
    H, W = C["dark"].shape
    hr = int(round(cam.hrow))
    band = (C["blue"] | C["yellow"])
    obj = C["dark"] | band
    # something dark at or just above the lens height in this column. A robot (0.148 m) reaches above the lens
    # (0.13 m) by 0.018 m: about 1 px at 6 m; a wall (0.1 m) stays 0.03 m below it: about 1 px under it at 9.6 m
    hf = int(math.floor(cam.hrow))
    r0, r1 = max(hf - 3, 0), hf + 1
    above = obj[r0:r1].any(0)
    out = []
    # the bottom scan stops at carpet or ball, not at white: the white dribbler bar of a robot facing the camera
    # would otherwise end the scan 3 cm above the carpet (v1 put such robots 0.5-0.9 m too far)
    floor = (C["green"] | C["orange"]) if clean else (C["green"] | C["orange"] | C["white"])
    for a, b in _components_1d(above):
        if b - a + 1 < min_cols:
            continue
        # bottom row per column: scan down from the lens height to the carpet
        bottoms = np.full(b - a + 1, -1)
        for i, u in enumerate(range(a, b + 1)):
            col = floor[hr:, u]
            idx = np.nonzero(col)[0]
            bottoms[i] = hr + (idx[0] if len(idx) else H - hr)
        # split where the bottom row jumps by > 3 px
        cuts = [0] + [i for i in range(1, len(bottoms)) if abs(int(bottoms[i]) - int(bottoms[i - 1])) > 3] + [len(bottoms)]
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e - s < min_cols:
                continue
            ua, ub = a + s, a + e - 1
            vb = float(np.median(bottoms[s:e]))
            uc = 0.5 * (ua + ub + 1)
            x, y = cam.ground(uc, vb, 0.005)     # the hull's bottom edge (5 mm ground clearance)
            if not np.isfinite(x):
                continue
            rr = math.hypot(x - cam.fwd, y)
            # the nearest point of the hull -> its centre one radius further along the ray
            ux, uy = (x - cam.fwd) / rr, y / rr
            cxr, cyr = x + ROBOT_R * ux, y + ROBOT_R * uy
            nb = int(band[hr:int(vb), ua:ub + 1].sum())
            nbl = int(C["blue"][hr:int(vb), ua:ub + 1].sum())
            team = None
            if nb >= 4:
                team = "blue" if nbl >= nb / 2 else "yellow"
            exp_w = 2 * ROBOT_R * cam.f / max(rr, 0.05)
            out.append(dict(x=float(cxr), y=float(cyr), range=float(rr), u0=int(ua), u1=int(ub), vb=vb,
                            width=int(ub - ua + 1), exp_width=float(exp_w), team=team,
                            edge=bool(ua == 0 or ub == W - 1)))
    if not clean:
        return out
    # v2 (after the first runs: 2.1 unmatched detections per picture, mostly slivers of one robot split where its
    # shovel or roller changes the bottom row, or of a robot partly hidden behind another): keep a blob only if it is
    # at least 40 % of the width a whole robot would have at that range (or touches the picture's edge), then keep the
    # widest of detections closer than 0.2 m to each other
    keep = [r for r in out if r["edge"] or r["width"] >= 0.4 * r["exp_width"]]
    keep.sort(key=lambda r: -r["width"])
    final = []
    for r in keep:
        if all(math.hypot(r["x"] - q["x"], r["y"] - q["y"]) > 0.2 for q in final):
            final.append(r)
    return final
