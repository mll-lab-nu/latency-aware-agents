"""Disc golf course: holes, trees, out of bounds, mandatories, the basket, the throw (flight.py), MuJoCo rendering.

Everything that decides a result is pure Python floats with a fixed step and seeded draws (numpy Generators seeded by
[seed, ...]), so a recorded list of throws replays bit-exactly in any process. MuJoCo (3.3.7) holds the scene for
rendering only.

A hole has its own frame: the tee at (0, 0), x roughly down the fairway, y to the left, z up; the ground is a smooth
height field z = h(x, y) (a tilt plus Gaussian hills, given on the tee sign). Trees are vertical cylinders: a trunk
(radius TRUNK_R) from the ground to the top, and a canopy (radius r) from CANOPY_BASE above the ground to the top.
A throw:
  * flight (flight.py) from the release point RELEASE_H above the lie, with the seeded execution scatter, in the wind
    of that throw (uniform, horizontal);
  * a tree: the disc bounces off the trunk (TRUNK_E normal, TRUNK_T tangential) or is caught and slowed by the canopy
    (CANOPY_KEEP of its velocity kept, reflected off the canopy's surface), and flies on;
  * the basket: the disc is holed when its centre enters the chain zone (a cylinder of radius CATCH_R around the pole,
    from CHAIN_LO to CHAIN_HI above the ground) or the tray from above, at no more than CATCH_V m/s; faster it is spat
    out (it drops at the basket); the cage below the tray and the lid bounce it;
  * the ground: the disc lands and skids along its horizontal velocity: the impact takes MU_I x the vertical speed,
    then kinetic friction MU_K stops it (skid length (v_h - MU_I v_z)^2 / (2 MU_K g)); a trunk or the basket stops it;
  * rules: the disc rests out of bounds (outside the course limits or in an OB area: PDGA 806.02) -> one penalty
    throw, the next lie is where it was last in bounds (806.02.B); a mandatory passed on the wrong side (804.01) ->
    one penalty throw, the next lie is the drop zone.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os

import numpy as np

import flight as F

BASE_SEED = 20261005
SEED = int(os.environ.get("DISCGOLF_SEED", BASE_SEED))

RELEASE_H = 1.0         # the launcher releases the disc 1.0 m above the ground at the lie
TRUNK_R = 0.30
CANOPY_BASE = 2.5
TRUNK_E, TRUNK_T = 0.35, 0.6
CANOPY_KEEP = 0.25
CATCH_R = 0.25          # chain zone radius (disc centre), m
CHAIN_LO, CHAIN_HI = 0.70, 1.40
TRAY_LO = 0.50
CAGE_R = 0.30
LID_R = 0.33
CATCH_V = 11.0          # m/s: faster, the chains spit the disc out
MU_I, MU_K = 0.6, 0.7
MAX_EVENTS = 25
PUTT_CIRCLE = 10.0      # PDGA 806.01: the putting area (circle 1), 10 m

LIMITS = {"speed": (5.0, 30.0), "launch_deg": (-5.0, 35.0), "nose_deg": (-10.0, 15.0), "bank_deg": (-60.0, 60.0),
          "spin_rps": (-20.0, 20.0), "aim_deg": (-180.0, 180.0)}
MIN_SPIN = 4.0          # |spin_rps| >= 4: a disc without spin does not fly (the launcher's wheel always spins it)
SCATTER = {"aim_deg": 1.5, "launch_deg": 1.5, "nose_deg": 1.5, "bank_deg": 3.0, "speed_frac": 0.04, "spin_frac": 0.05}
MAX_OVER_PAR = 3        # a hole not holed out after par + 3 throws (penalties included) ends and scores par + 4
WIND_MAX = 7.0


# ----------------------------------------------------------------------------------------------- the course
def _rect(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# Base holes (the default course). forests: (polygon, spacing m); trees: (x, y, canopy r, height); ob: polygons;
# mando: dict(x, y, side, ...): the disc must pass the mando point on `side` ("right": at the crossing of the line
# x = mx, y < my); dz: the drop zone.
BASE_HOLES = [
    dict(name="Hole 1", par=3, basket=(78.0, 4.0), bounds=(-15.0, 105.0, -34.0, 34.0), tilt=(0.004, 0.006),
         hills=[(45.0, -14.0, 1.2, 14.0)], forests=[(_rect(-15, 105, 22, 34), 5.0), (_rect(-15, 105, -34, -22), 5.0),
                                                     (_rect(96, 105, -22, 22), 5.0)],
         trees=[(38.0, 9.0, 2.6, 13.0), (52.0, -7.0, 2.8, 15.0), (66.0, 13.0, 2.4, 12.0)], ob=[], mando=None,
         wind_scale=1.0),
    dict(name="Hole 2", par=3, basket=(76.0, 32.0), bounds=(-15.0, 100.0, -28.0, 55.0), tilt=(0.0, 0.008),
         hills=[(70.0, 30.0, 1.0, 12.0)], forests=[(_rect(24, 60, 5, 42), 4.5), (_rect(-15, 100, -28, -18), 5.0),
                                                    (_rect(-15, 100, 47, 55), 5.0), (_rect(93, 100, -18, 47), 5.0),
                                                    (_rect(-15, 24, 20, 47), 5.0)],
         trees=[], ob=[], mando=None, wind_scale=1.0),
    dict(name="Hole 3", par=4, basket=(118.0, -42.0), bounds=(-15.0, 140.0, -70.0, 28.0), tilt=(-0.003, 0.0),
         hills=[(100.0, -30.0, 1.5, 18.0)], forests=[(_rect(28, 96, -70, -18), 4.5), (_rect(-15, 140, 18, 28), 5.0),
                                                      (_rect(132, 140, -70, 18), 5.0), (_rect(-15, 28, -70, -14), 5.0)],
         trees=[(75.0, 4.0, 2.6, 14.0)], ob=[], mando=None, wind_scale=1.0),
    dict(name="Hole 4", par=3, basket=(66.0, -4.0), bounds=(-15.0, 92.0, -32.0, 32.0), tilt=(0.0, 0.0),
         hills=[(68.0, -4.0, 0.8, 8.0)], forests=[(_rect(-15, 92, 24, 32), 5.0), (_rect(-15, 92, -32, -24), 5.0)],
         trees=[(72.0, 10.0, 2.4, 12.0)], ob=[_rect(32, 57, -24, 24), _rect(80, 92, -24, 24)], mando=None,
         wind_scale=1.0),
    dict(name="Hole 5", par=3, basket=(92.0, 12.0), bounds=(-15.0, 112.0, -32.0, 38.0), tilt=(0.003, -0.004),
         hills=[(80.0, 20.0, 1.0, 14.0)], forests=[(_rect(-15, 112, -32, -20), 5.0), (_rect(-15, 112, 30, 38), 5.0),
                                                    (_rect(104, 112, -20, 30), 5.0)],
         trees=[(50.0, -2.0, 2.2, 16.0), (62.0, 6.0, 2.5, 14.0)], ob=[],
         mando=dict(x=50.0, y=-2.0 - 2.2, side="right", dz=(42.0, -9.0)), wind_scale=1.0),
    dict(name="Hole 6", par=4, basket=(148.0, -4.0), bounds=(-15.0, 170.0, -30.0, 30.0), tilt=(0.002, 0.0),
         hills=[(90.0, 8.0, 1.5, 22.0), (130.0, -10.0, -1.0, 16.0)], forests=[(_rect(-15, 170, 21, 30), 5.0),
                                                                              (_rect(162, 170, -18, 21), 5.0)],
         trees=[(68.0, 6.0, 2.6, 15.0), (102.0, -9.0, 2.6, 14.0), (126.0, 9.0, 2.4, 13.0)],
         ob=[_rect(-15, 170, -30, -18)], mando=None, wind_scale=1.4),
]
N_HOLES = len(BASE_HOLES)
PAR_TOTAL = sum(H["par"] for H in BASE_HOLES)


def _inside(poly, x, y):
    """Point in polygon (ray casting)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xc > x:
                inside = not inside
    return inside


def make_holes(seed):
    """The course for a seed: the forests are filled with trees (jittered grid), the single trees, the basket and the
    hills move a little; layout, pars, OB areas and mandatories stay."""
    holes = copy.deepcopy(BASE_HOLES)
    for i, H in enumerate(holes):
        rng = np.random.default_rng([seed, 7, i])
        u = lambda a, b: round(float(rng.uniform(a, b)), 3)
        if seed != BASE_SEED:
            bx, by = H["basket"]
            H["basket"] = (round(bx + u(-3, 3), 3), round(by + u(-2.5, 2.5), 3))
            H["trees"] = [(round(x + u(-2.5, 2.5), 3), round(y + u(-2.0, 2.0), 3), r, h) for x, y, r, h in H["trees"]]
            H["hills"] = [(round(x + u(-5, 5), 3), round(y + u(-4, 4), 3), round(a * u(0.7, 1.3), 3), s)
                          for x, y, a, s in H["hills"]]
        trees = [dict(x=float(x), y=float(y), r=float(r), h=float(h), kind="single") for x, y, r, h in H["trees"]]
        bx, by = H["basket"]
        for poly, sp in H["forests"]:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            gx = np.arange(min(xs) + sp / 2, max(xs), sp)
            gy = np.arange(min(ys) + sp / 2, max(ys), sp)
            for x in gx:
                for y in gy:
                    tx, ty = round(float(x) + u(-1.2, 1.2), 3), round(float(y) + u(-1.2, 1.2), 3)
                    if not _inside(poly, tx, ty) or math.hypot(tx, ty) < 6 or math.hypot(tx - bx, ty - by) < 8:
                        continue
                    trees.append(dict(x=tx, y=ty, r=u(2.0, 3.0), h=u(14.0, 22.0), kind="forest"))
        H["tree_list"] = trees
    return holes


HOLES = make_holes(SEED)


def ground(H, x, y):
    sx, sy = H["tilt"]
    z = sx * x + sy * y
    for x0, y0, a, s in H["hills"]:
        z += a * math.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s * s))
    return z


def in_bounds(H, x, y):
    x0, x1, y0, y1 = H["bounds"]
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    return not any(_inside(p, x, y) for p in H["ob"])


# ----------------------------------------------------------------------------------------------- wind and scatter
def wind_of(hole_key, k):
    """The wind of the k-th throw on a hole (hole_key: the hole's index, or 100 + n for the n-th play-off hole):
    the hole's mean wind plus a per-throw gust. Returns (wx, wy) m/s (the direction the air moves to)."""
    hi = hole_key % 100 if hole_key < 100 else (hole_key - 100) % N_HOLES
    H = HOLES[hi]
    rng = np.random.default_rng([SEED, 11, hole_key])
    mean = float(rng.uniform(0.8, 4.5)) * H["wind_scale"]
    ang = float(rng.uniform(-math.pi, math.pi))
    g = np.random.default_rng([SEED, 13, hole_key, k])
    sp = max(0.0, min(WIND_MAX, mean * (1.0 + 0.15 * float(g.standard_normal()))))
    a = ang + math.radians(10.0) * float(g.standard_normal())
    return (round(sp * math.cos(a), 4), round(sp * math.sin(a), 4))


def scatter_of(hole_key, k):
    g = np.random.default_rng([SEED, 17, hole_key, k])
    z = [float(v) for v in g.standard_normal(6)]
    S = SCATTER
    return {"aim_deg": z[0] * S["aim_deg"], "launch_deg": z[1] * S["launch_deg"], "nose_deg": z[2] * S["nose_deg"],
            "bank_deg": z[3] * S["bank_deg"], "speed_frac": z[4] * S["speed_frac"], "spin_frac": z[5] * S["spin_frac"]}


# ----------------------------------------------------------------------------------------------- one throw
def _seg_circle(p0, p1, cx, cy, R):
    """First parameter t in [0, 1] where the horizontal segment p0 -> p1 enters the circle (or None; None also when
    p0 is already inside)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    fx, fy = p0[0] - cx, p0[1] - cy
    c = fx * fx + fy * fy - R * R
    if c <= 0:
        return None
    a = dx * dx + dy * dy
    if a < 1e-14:
        return None
    b = 2 * (fx * dx + fy * dy)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    return t if 0.0 <= t <= 1.0 else None


def _near_trees(H, p0, p1, pad):
    xlo, xhi = min(p0[0], p1[0]) - pad, max(p0[0], p1[0]) + pad
    ylo, yhi = min(p0[1], p1[1]) - pad, max(p0[1], p1[1]) + pad
    return [T for T in H["tree_list"] if xlo <= T["x"] <= xhi and ylo <= T["y"] <= yhi]


def _events(H, p0, p1, zb_ground):
    """Collision candidates on the segment p0 -> p1 (3D): list of (t, kind, data)."""
    ev = []
    bx, by = H["basket"]
    zb = zb_ground
    # basket: chain zone / tray (holed), lid, cage
    t = _seg_circle(p0, p1, bx, by, CATCH_R)
    if t is not None:
        z = p0[2] + t * (p1[2] - p0[2])
        if zb + CHAIN_LO <= z <= zb + CHAIN_HI:
            ev.append((t, "chains", None))
        elif zb <= z < zb + CHAIN_LO:
            ev.append((t, "cage", (bx, by, CAGE_R)))
    for zplane, kind, R in ((zb + CHAIN_HI + 0.12, "lid", LID_R), (zb + CHAIN_LO, "tray", CATCH_R)):
        if (p0[2] - zplane) * (p1[2] - zplane) < 0:
            t = (p0[2] - zplane) / (p0[2] - p1[2])
            x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
            if math.hypot(x - bx, y - by) <= R:
                if kind == "lid" and p1[2] < p0[2]:
                    ev.append((t, "lid", None))
                elif kind == "tray" and p1[2] < p0[2]:
                    ev.append((t, "chains", None))
    # trees
    for T in _near_trees(H, p0, p1, 3.5):
        g = ground(H, T["x"], T["y"])
        for R, zlo, kind in ((TRUNK_R + F.R_DISC, g - 1.0, "trunk"), (T["r"] + F.R_DISC, g + CANOPY_BASE, "canopy")):
            t = _seg_circle(p0, p1, T["x"], T["y"], R)
            if t is not None:
                z = p0[2] + t * (p1[2] - p0[2])
                if zlo <= z <= g + T["h"]:
                    ev.append((t, kind, (T["x"], T["y"], R)))
        top = g + T["h"]
        if p0[2] > top >= p1[2]:
            t = (p0[2] - top) / (p0[2] - p1[2])
            x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
            if math.hypot(x - T["x"], y - T["y"]) <= T["r"]:
                ev.append((t, "canopy_top", None))
    # ground
    g0 = p0[2] - ground(H, p0[0], p0[1])
    g1 = p1[2] - ground(H, p1[0], p1[1])
    if g0 > 0 >= g1:
        ev.append((g0 / (g0 - g1), "ground", None))
    return ev


def _reflect(s, cx, cy, R, e_n, e_t, keep_z):
    x, y = s[0], s[1]
    nx, ny = x - cx, y - cy
    d = math.hypot(nx, ny) or 1.0
    nx, ny = nx / d, ny / d
    vx, vy, vz = s[3], s[4], s[5]
    vn = vx * nx + vy * ny
    tx, ty = vx - vn * nx, vy - vn * ny
    if vn < 0:
        vn = -e_n * vn
    vx, vy = vn * nx + e_t * tx, vn * ny + e_t * ty
    px, py = cx + nx * (R + 0.01), cy + ny * (R + 0.01)
    return (px, py, s[2], vx, vy, vz * keep_z) + tuple(s[6:9]) + (s[9] * 0.5,)


def throw(H, lie, act, wind, sc, trail=True):
    """Simulate one throw from lie (x, y). act: the commanded release (aim_deg, speed, launch_deg, nose_deg, bank_deg,
    spin_rps, disc); sc: the execution scatter. Returns dict(rest, holed, result, events, track (ground track samples
    (x, y)), trail [(t, x, y, z)], ...)."""
    aim = act["aim_deg"] + sc["aim_deg"]
    spd = act["speed"] * (1.0 + sc["speed_frac"])
    s = F.release_state((lie[0], lie[1], ground(H, *lie) + RELEASE_H), aim, spd, act["launch_deg"] + sc["launch_deg"],
                        act["nose_deg"] + sc["nose_deg"], act["bank_deg"] + sc["bank_deg"],
                        act["spin_rps"] * (1.0 + sc["spin_frac"]))
    disc = act["disc"]
    wind = (float(wind[0]), float(wind[1]), 0.0)
    bx, by = H["basket"]
    zb = ground(H, bx, by)
    t = 0.0
    hits = {"trunk": 0, "canopy": 0, "basket": 0}
    out_tr = [(0.0, s[0], s[1], s[2])]
    track = [(s[0], s[1])]
    holed = False
    result = None
    n_ev = 0
    apex = s[2] - ground(H, s[0], s[1])
    landed = None
    while t < F.MAX_T:
        s1, _ = F.rk4(s, disc, wind)
        p0, p1 = s[:3], s1[:3]
        ev = _events(H, p0, p1, zb)
        if ev:
            te, kind, data = min(ev, key=lambda e: e[0])
            sp = tuple(s[i] + te * (s1[i] - s[i]) for i in range(10))
            t += te * F.DT
            out_tr.append((round(t, 4), sp[0], sp[1], sp[2]))
            track.append((sp[0], sp[1]))
            n_ev += 1
            if kind == "chains":
                spd_rel = math.sqrt(sp[3] ** 2 + sp[4] ** 2 + sp[5] ** 2)
                hits["basket"] += 1
                if spd_rel <= CATCH_V:
                    holed = True
                    result = "holed"
                    s = (bx, by, zb + 0.75) + sp[3:]
                    break
                # spat out: drops at the basket with little speed
                s = (sp[0], sp[1], sp[2], -0.1 * sp[3], -0.1 * sp[4], 0.0) + tuple(sp[6:])
                result = "spat out of the chains"
                n_ev = max(n_ev, MAX_EVENTS - 3)
            elif kind == "ground":
                landed = sp
                break
            elif kind in ("trunk", "cage"):
                hits["basket" if kind == "cage" else "trunk"] += 1
                s = _reflect(sp, data[0], data[1], data[2], TRUNK_E, TRUNK_T, 0.6)
            elif kind == "canopy":
                hits["canopy"] += 1
                s = _reflect(sp, data[0], data[1], data[2], CANOPY_KEEP, CANOPY_KEEP, CANOPY_KEEP)
            elif kind in ("canopy_top", "lid"):
                hits["canopy" if kind == "canopy_top" else "basket"] += 1
                k = CANOPY_KEEP
                s = (sp[0], sp[1], sp[2] + 0.01, sp[3] * k, sp[4] * k, abs(sp[5]) * 0.2) + tuple(sp[6:9]) + (sp[9] * 0.5,)
            if n_ev >= MAX_EVENTS:          # tangled in the trees: it drops
                s = (s[0], s[1], s[2], 0.0, 0.0, -1.0) + tuple(s[6:])
            continue
        s = s1
        t += F.DT
        apex = max(apex, s[2] - ground(H, s[0], s[1]))
        if trail and int(round(t / F.DT)) % 5 == 0:
            out_tr.append((round(t, 4), s[0], s[1], s[2]))
        track.append((s[0], s[1]))
    flight_t = round(t, 3)
    skid = 0.0
    if holed:
        rest = (bx, by)
    else:
        if landed is None:
            landed = s
        vh = math.hypot(landed[3], landed[4])
        vz = abs(landed[5])
        skid = max(0.0, vh - MU_I * vz) ** 2 / (2 * MU_K * F.G)
        x, y = landed[0], landed[1]
        if skid > 1e-6 and vh > 1e-9:
            ex, ey = landed[3] / vh * skid, landed[4] / vh * skid
            p0, p1 = (x, y), (x + ex, y + ey)
            tmin = 1.0
            for T in _near_trees(H, p0, p1, 1.0):
                tt = _seg_circle(p0, p1, T["x"], T["y"], TRUNK_R + F.R_DISC)
                if tt is not None:
                    tmin = min(tmin, tt)
            tt = _seg_circle(p0, p1, bx, by, CAGE_R + F.R_DISC)
            if tt is not None:
                tmin = min(tmin, tt)
            skid *= tmin
            nsub = max(1, int(skid / 0.5))
            for k in range(1, nsub + 1):
                track.append((x + ex * tmin * k / nsub, y + ey * tmin * k / nsub))
            x, y = x + ex * tmin, y + ey * tmin
        rest = (x, y)
        if trail:
            out_tr.append((round(t + 0.5, 4), x, y, ground(H, x, y) + 0.02))
    return dict(rest=rest, holed=holed, result=result or ("holed" if holed else "landed"), hits=hits,
                flight_time_s=flight_t, skid_m=round(skid, 3), apex_m=round(apex, 2), track=track,
                trail=out_tr if trail else None, release=dict(aim_deg=aim, speed=spd))


def mando_check(H, track):
    """For a hole with a mandatory: 'cleared' if the track first crosses the mando line x = mx on the right side,
    'missed' if on the wrong side, None if it does not cross."""
    M = H.get("mando")
    if not M:
        return None
    mx, my = M["x"], M["y"]
    for (x0, y0), (x1, y1) in zip(track, track[1:]):
        if (x0 - mx) * (x1 - mx) < 0 or (x1 == mx and x0 != mx):
            yc = y0 + (mx - x0) / (x1 - x0) * (y1 - y0)
            ok = yc < my if M["side"] == "right" else yc > my
            return "cleared" if ok else "missed"
    return None


def last_in_bounds(H, track):
    """PDGA 806.02.B: the point where the disc was last in bounds (along its ground track; the start if never)."""
    last = track[0]
    inb = True
    for p in track:
        if in_bounds(H, *p):
            last = p
    return last


class Course:
    """A round (or a single play-off hole): hole index, the lie, throws and penalties per hole."""

    def __init__(self, holes=None, key_base=0):
        self.order = list(range(N_HOLES)) if holes is None else list(holes)
        self.key_base = key_base          # 0: the round (wind key = hole index); 100 + n: play-off hole n
        self.i = 0
        self.hole = self.order[0]
        self.lie = (0.0, 0.0)
        self.throws = [0] * len(self.order)       # throws actually made
        self.penalties = [0] * len(self.order)
        self.ended = [False] * len(self.order)    # ended unfinished (par + 3 throws): scores par + 4
        self.holed = [False] * len(self.order)
        self.mando_cleared = False
        self.done = False
        self.last = None
        self.log = []

    def H(self):
        return HOLES[self.hole]

    def key(self):
        return self.i if self.key_base == 0 else self.key_base

    def count(self, i=None):
        i = self.i if i is None else i
        return self.throws[i] + self.penalties[i]

    def play(self, act):
        H = self.H()
        k = self.throws[self.i]
        wind = wind_of(self.key(), k)
        sc = scatter_of(self.key(), k)
        start = self.lie
        r = throw(H, start, act, wind, sc)
        self.throws[self.i] += 1
        rest = r["rest"]
        status = "holed" if r["holed"] else "in bounds"
        penalty = 0
        next_lie = rest
        mando = None
        if not r["holed"] and H.get("mando") and not self.mando_cleared:
            mando = mando_check(H, r["track"])
            if mando == "cleared":
                self.mando_cleared = True
        if r["holed"]:
            pass
        elif mando == "missed":
            status, penalty, next_lie = "missed the mandatory", 1, tuple(H["mando"]["dz"])
        elif not in_bounds(H, *rest):
            status, penalty, next_lie = "out of bounds", 1, last_in_bounds(H, r["track"])
        self.penalties[self.i] += penalty
        bx, by = H["basket"]
        rec = dict(hole=self.hole + 1, throw=self.throws[self.i], from_lie=[round(start[0], 3), round(start[1], 3)],
                   disc=act["disc"], result=r["result"], status=status, penalty=penalty,
                   rest=[round(rest[0], 3), round(rest[1], 3)], next_lie=[round(next_lie[0], 3), round(next_lie[1], 3)],
                   distance_to_basket_m=round(math.hypot(next_lie[0] - bx, next_lie[1] - by), 2),
                   tree_hits=r["hits"]["trunk"] + r["hits"]["canopy"], basket_hits=r["hits"]["basket"],
                   flight_time_s=r["flight_time_s"], apex_m=r["apex_m"], skid_m=r["skid_m"])
        if mando is not None:
            rec["mandatory"] = mando
        self.log.append(dict(rec, rest_exact=list(rest)))
        self.last = dict(rec, trail=r["trail"], wind=wind)
        self.lie = next_lie
        rec["hole_finished"] = False
        if r["holed"]:
            self.holed[self.i] = True
            rec["hole_finished"] = True
        elif self.count() >= H["par"] + MAX_OVER_PAR:
            self.ended[self.i] = True
            rec["hole_finished"] = True
        if rec["hole_finished"]:
            rec["hole_score"] = self.hole_score(self.i)
            self.next_hole()
        return rec, r

    def next_hole(self):
        self.i += 1
        if self.i >= len(self.order):
            self.done = True
            self.i = len(self.order) - 1
            return
        self.hole = self.order[self.i]
        self.lie = (0.0, 0.0)
        self.mando_cleared = False
        self.last = None

    def hole_score(self, i):
        if self.ended[i]:
            return HOLES[self.order[i]]["par"] + MAX_OVER_PAR + 1
        return self.throws[i] + self.penalties[i]

    def finished(self, i):
        return self.done or i < self.i

    def state_sha(self):
        s = json.dumps([[r["hole"], r["throw"], repr(r["rest_exact"][0]), repr(r["rest_exact"][1]), r["status"],
                         r["result"]] for r in self.log] + [self.throws, self.penalties, self.ended, self.i, self.done])
        return hashlib.sha256(s.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- rendering
TEE_W, TEE_H = 1000, 600
OV_W, OV_H = 1400, 700
WIND_W, WIND_H = 400, 400
FOVY_TEE = 50.0
TEE_BACK, TEE_UP, TEE_PITCH = 4.0, 2.2, 3.0       # the throw camera: 4 m behind the lie, 2.2 m up, 3 deg up
WINDSOCK = (-3.0, 5.0)        # the tee's windsock pole (hole frame), 4.0 m tall
WINDSOCK_H = 4.0
SOCK_L = 1.2
WIND_CAM_H = 3.0              # the wind camera: 3 m above the pole top, straight down, orthographic
WIND_EXTENT = 3.2             # m across the wind tile (vertical)
TEX_PPM = 5


def ov_extent(H):
    """Orthographic overview: (centre x, centre y, vertical extent m) fitting the course limits."""
    x0, x1, y0, y1 = H["bounds"]
    ext = max(y1 - y0, (x1 - x0) * OV_H / OV_W) * 1.02
    return (x0 + x1) / 2, (y0 + y1) / 2, round(ext, 3)


def ov_pixel(H, x, y):
    cx, cy, ext = ov_extent(H)
    s = OV_H / ext
    return OV_W / 2 + (x - cx) * s, OV_H / 2 - (y - cy) * s


def sock_angle_deg(speed):
    """The windsock's angle below horizontal: 90 deg in calm air, rising linearly, horizontal at 10 m/s."""
    return 90.0 * (1.0 - min(speed, 10.0) / 10.0)


def _texture(H):
    from PIL import Image, ImageDraw
    x0, x1, y0, y1 = H["bounds"]
    m = 10.0
    W, Hh = int((x1 - x0 + 2 * m) * TEX_PPM), int((y1 - y0 + 2 * m) * TEX_PPM)
    pix = lambda x, y: ((x - x0 + m) * TEX_PPM, (y1 + m - y) * TEX_PPM)
    im = Image.new("RGB", (W, Hh), (52, 112, 46))
    dr = ImageDraw.Draw(im)
    # the fairway (lighter, mowing stripes every 10 m)
    for k in range(int(math.ceil((x1 - x0) / 10))):
        xa = x0 + 10 * k
        col = (92, 160, 70) if k % 2 == 0 else (86, 152, 66)
        dr.rectangle([pix(xa, y1), pix(min(xa + 10, x1), y0)], fill=col)
    for poly in H["ob"]:
        dr.polygon([pix(*p) for p in poly], fill=(40, 90, 170))
    # OB lines (white) around the course limits and the OB areas
    lw = max(1, int(0.25 * TEX_PPM))
    dr.line([pix(x0, y0), pix(x1, y0), pix(x1, y1), pix(x0, y1), pix(x0, y0)], fill=(245, 245, 245), width=lw)
    for poly in H["ob"]:
        dr.line([pix(*p) for p in poly + [poly[0]]], fill=(245, 245, 245), width=lw)
    # grid every 10 m (faint)
    for gx in range(int(math.ceil(x0 / 10)) * 10, int(x1) + 1, 10):
        dr.line([pix(gx, y0), pix(gx, y1)], fill=(120, 175, 105), width=1)
    for gy in range(int(math.ceil(y0 / 10)) * 10, int(y1) + 1, 10):
        dr.line([pix(x0, gy), pix(x1, gy)], fill=(120, 175, 105), width=1)
    # tee pad, putting circle
    dr.rectangle([pix(-1.5, 0.75), pix(1.5, -0.75)], fill=(150, 150, 150))
    bx, by = H["basket"]
    a, b = pix(bx - PUTT_CIRCLE, by + PUTT_CIRCLE), pix(bx + PUTT_CIRCLE, by - PUTT_CIRCLE)
    dr.ellipse([a, b], outline=(245, 245, 245), width=lw)
    M = H.get("mando")
    if M:
        dx, dy = M["dz"]
        dr.rectangle([pix(dx - 1.0, dy + 1.0), pix(dx + 1.0, dy - 1.0)], outline=(250, 220, 40), width=lw)
        # the restricted line: from the mando point to the wrong side, red
        far = y1 if M["side"] == "right" else y0
        dr.line([pix(M["x"], M["y"]), pix(M["x"], far)], fill=(220, 30, 30), width=lw)
    b = io.BytesIO()
    im.transpose(Image.FLIP_TOP_BOTTOM).save(b, format="PNG")
    return b.getvalue(), (x0 - m, x1 + m, y0 - m, y1 + m)


def _mesh(H, ext, step=2.0):
    x0, x1, y0, y1 = ext
    nx, ny = int(round((x1 - x0) / step)) + 1, int(round((y1 - y0) / step)) + 1
    V, T, Fc = [], [], []
    for j in range(ny):
        for i in range(nx):
            x, y = x0 + i * (x1 - x0) / (nx - 1), y0 + j * (y1 - y0) / (ny - 1)
            V += [x, y, ground(H, x, y)]
            T += [(x - x0) / (x1 - x0), (y - y0) / (y1 - y0)]
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            Fc += [a, a + 1, a + nx + 1, a, a + nx + 1, a + nx]
    return " ".join(f"{v:.3f}" for v in V), " ".join(f"{v:.5f}" for v in T), " ".join(map(str, Fc))


def scene_xml(H, ext):
    v, t, f = _mesh(H, ext)
    g = []
    for k, T in enumerate(H["tree_list"]):
        z0 = ground(H, T["x"], T["y"])
        shade = 0.85 + 0.3 * ((k * 37) % 11) / 10.0
        col = f"{0.12 * shade:.3f} {0.36 * shade:.3f} {0.12 * shade:.3f} 1"
        g.append(f'<geom type="cylinder" size="{TRUNK_R} {(T["h"] + 0.5) / 2:.3f}" pos="{T["x"]} {T["y"]} {z0 - 1.0 + (T["h"] + 0.5) / 2:.3f}" rgba="0.36 0.24 0.14 1"/>')
        hc = T["h"] - CANOPY_BASE
        g.append(f'<geom type="cylinder" size="{T["r"]} {hc / 2:.3f}" pos="{T["x"]} {T["y"]} {z0 + CANOPY_BASE + hc / 2:.3f}" rgba="{col}"/>')
    bx, by = H["basket"]
    zb = ground(H, bx, by)
    g.append(f'<geom type="cylinder" size="0.025 {(CHAIN_HI + 0.15) / 2:.3f}" pos="{bx} {by} {zb + (CHAIN_HI + 0.15) / 2:.3f}" rgba="0.75 0.75 0.78 1"/>')
    g.append(f'<geom type="cylinder" size="{CAGE_R} 0.10" pos="{bx} {by} {zb + TRAY_LO + 0.1:.3f}" rgba="0.95 0.85 0.10 1"/>')
    g.append(f'<geom type="cylinder" size="{CATCH_R - 0.03} {(CHAIN_HI - CHAIN_LO) / 2:.3f}" pos="{bx} {by} {zb + (CHAIN_HI + CHAIN_LO) / 2:.3f}" rgba="0.80 0.80 0.84 0.55"/>')
    g.append(f'<geom type="cylinder" size="{LID_R} 0.04" pos="{bx} {by} {zb + CHAIN_HI + 0.12:.3f}" rgba="0.95 0.85 0.10 1"/>')
    wx, wy = WINDSOCK
    zw = ground(H, wx, wy)
    g.append(f'<geom type="cylinder" size="0.03 {WINDSOCK_H / 2}" pos="{wx} {wy} {zw + WINDSOCK_H / 2:.3f}" rgba="0.85 0.85 0.85 1"/>')
    M = H.get("mando")
    if M:
        g.append(f'<geom type="box" size="0.05 0.6 0.4" pos="{M["x"] - 2.6} {M["y"] + 2.2} {ground(H, M["x"] - 2.6, M["y"] + 2.2) + 1.6:.3f}" rgba="0.9 0.1 0.1 1"/>')
    return f"""
<mujoco model="discgolf">
  <visual><global offwidth="1400" offheight="700"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3" specular="0 0 0"/><map znear="0.002" zfar="6"/></visual>
  <statistic extent="100" center="{(ext[0] + ext[1]) / 2:.2f} {(ext[2] + ext[3]) / 2:.2f} 0"/>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.75 0.95" rgb2="0.95 0.97 1.0" width="256" height="256"/>
    <texture name="grass" type="2d" file="course.png"/>
    <material name="course" texture="grass" specular="0.02"/>
    <mesh name="ground" vertex="{v}" texcoord="{t}" face="{f}"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 50" dir="0.2 0.3 -1" diffuse="0.7 0.7 0.65" castshadow="false"/>
    <geom type="plane" size="400 400 1" pos="0 0 -3" rgba="0.17 0.34 0.15 1"/>
    <geom type="mesh" mesh="ground" material="course"/>
    {''.join(g)}
    <camera name="overview" mode="fixed" orthographic="true" fovy="{ov_extent(H)[2]}" pos="{ov_extent(H)[0]} {ov_extent(H)[1]} 150" xyaxes="1 0 0 0 1 0"/>
    <camera name="wind" mode="fixed" orthographic="true" fovy="{WIND_EXTENT}" pos="{wx} {wy} {zw + WINDSOCK_H + WIND_CAM_H:.3f}" xyaxes="1 0 0 0 1 0"/>
    <body name="launcher" mocap="true" pos="0 0 0">
      <geom type="box" size="0.35 0.35 0.25" pos="0 0 0.25" rgba="0.95 0.45 0.05 1"/>
      <geom type="cylinder" size="0.06 0.35" pos="0 0 0.85" rgba="0.2 0.2 0.2 1"/>
      <geom type="box" size="0.25 0.12 0.04" pos="0.15 0 {RELEASE_H:.2f}" rgba="0.95 0.45 0.05 1"/>
    </body>
    <body name="sock" mocap="true" pos="0 0 0">
      <geom type="capsule" fromto="0 0 0 {SOCK_L * 0.2:.3f} 0 0" size="0.10" rgba="1 0.45 0 1"/>
      <geom type="capsule" fromto="{SOCK_L * 0.2:.3f} 0 0 {SOCK_L * 0.4:.3f} 0 0" size="0.09" rgba="1 1 1 1"/>
      <geom type="capsule" fromto="{SOCK_L * 0.4:.3f} 0 0 {SOCK_L * 0.6:.3f} 0 0" size="0.08" rgba="1 0.45 0 1"/>
      <geom type="capsule" fromto="{SOCK_L * 0.6:.3f} 0 0 {SOCK_L * 0.8:.3f} 0 0" size="0.07" rgba="1 1 1 1"/>
      <geom type="capsule" fromto="{SOCK_L * 0.8:.3f} 0 0 {SOCK_L:.3f} 0 0" size="0.06" rgba="1 0.45 0 1"/>
    </body>
    <body name="disc" mocap="true" pos="0 0 -5">
      <geom type="cylinder" size="{F.R_DISC} 0.012" rgba="0.95 0.1 0.75 1"/>
    </body>
  </worldbody>
</mujoco>"""


def _quat_from_z(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    ax = np.cross([0, 0, 1.0], n)
    s = np.linalg.norm(ax)
    ang = math.atan2(s, n[2])
    return [math.cos(ang / 2)] + list((ax / s if s > 1e-12 else np.zeros(3)) * math.sin(ang / 2))


def heading_to_basket(H, lie):
    bx, by = H["basket"]
    return math.degrees(math.atan2(by - lie[1], bx - lie[0]))


def tee_cam_pose(H, lie):
    yaw = heading_to_basket(H, lie)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    p = (lie[0] - TEE_BACK * c, lie[1] - TEE_BACK * s)
    return (p[0], p[1], ground(H, *lie) + TEE_UP), yaw


class Views:
    def __init__(self):
        self.models = {}
        self.rend = {}

    def model(self, hi):
        import mujoco as mj
        if hi not in self.models:
            H = HOLES[hi]
            png, ext = _texture(H)
            m = mj.MjModel.from_xml_string(scene_xml(H, ext), assets={"course.png": png})
            self.models[hi] = (m, mj.MjData(m))
        return self.models[hi]

    def renderer(self, hi, w, h):
        import mujoco as mj
        k = (hi, w, h)
        if k not in self.rend:
            self.rend[k] = mj.Renderer(self.model(hi)[0], height=h, width=w)
        return self.rend[k]

    def place(self, hi, lie, yaw, wind, disc_pos=None, disc_n=(0, 0, 1)):
        import mujoco as mj
        H = HOLES[hi]
        m, d = self.model(hi)
        d.mocap_pos[0] = [lie[0], lie[1], ground(H, *lie)]
        d.mocap_quat[0] = [math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)]
        wx, wy = WINDSOCK
        sp = math.hypot(*wind)
        ang = math.radians(sock_angle_deg(sp))
        az = math.atan2(wind[1], wind[0]) if sp > 1e-6 else 0.0
        # the sock hangs from the pole top: rotate +x down by ang about the horizontal axis, then to the azimuth
        qz = [math.cos(az / 2), 0, 0, math.sin(az / 2)]
        qy = [math.cos(ang / 2), 0, math.sin(ang / 2), 0]
        q = np.zeros(4)
        mj.mju_mulQuat(q, np.array(qz), np.array(qy))
        d.mocap_pos[1] = [wx, wy, ground(H, wx, wy) + WINDSOCK_H]
        d.mocap_quat[1] = q
        if disc_pos is None:
            d.mocap_pos[2] = [0, 0, -50]
        else:
            d.mocap_pos[2] = disc_pos
            d.mocap_quat[2] = _quat_from_z(disc_n)
        mj.mj_forward(m, d)

    @staticmethod
    def cam(pos, yaw, pitch):
        import mujoco as mj
        c = mj.MjvCamera()
        fw = np.array([math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)),
                       math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)), math.sin(math.radians(pitch))])
        c.lookat[:] = np.asarray(pos) + fw
        c.distance = 1.0
        c.azimuth = yaw
        c.elevation = pitch
        return c

    def ortho(self, hi, w, h, name):
        """A fixed orthographic camera of the scene (overview / wind; see scene_xml)."""
        m, d = self.model(hi)
        r = self.renderer(hi, w, h)
        r.update_scene(d, camera=name)
        return r.render().copy()

    def render(self, hi, lie, wind, trail=None):
        """dict name -> RGB uint8: tee (throw camera at the lie), overview (orthographic, the whole hole), wind
        (orthographic, straight down on the tee's windsock)."""
        H = HOLES[hi]
        yaw = heading_to_basket(H, lie)
        self.place(hi, lie, yaw, wind)
        m, d = self.model(hi)
        out = {}
        pos, yaw = tee_cam_pose(H, lie)
        r = self.renderer(hi, TEE_W, TEE_H)
        m.vis.global_.fovy = FOVY_TEE
        r.update_scene(d, camera=self.cam(pos, yaw, TEE_PITCH))
        out["tee"] = r.render().copy()
        ov = self.ortho(hi, OV_W, OV_H, "overview")
        if trail:
            ov = draw_trail(H, ov, [(x, y) for _, x, y, _ in trail])
        out["overview"] = ov
        out["wind"] = self.ortho(hi, WIND_W, WIND_H, "wind")
        m.vis.global_.fovy = 45.0
        return out


def draw_trail(H, img, pts, rad=3):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for x, y in pts:
        u, v = ov_pixel(H, x, y)
        dr.ellipse([u - rad, v - rad, u + rad, v + rad], fill=(255, 225, 30))
    return np.asarray(im).copy()


# ----------------------------------------------------------------------------------------------- the tee sign (spec)
def spec_holes():
    out = []
    for i, H in enumerate(HOLES):
        bx, by = H["basket"]
        e = dict(hole=i + 1, par=H["par"], tee=[0.0, 0.0], basket=[bx, by],
                 distance_m=round(math.hypot(bx, by), 1),
                 course_limits={"x": [H["bounds"][0], H["bounds"][1]], "y": [H["bounds"][2], H["bounds"][3]]},
                 ob_areas=[[list(p) for p in poly] for poly in H["ob"]],
                 ground={"tilt": list(H["tilt"]), "hills": [dict(x=x, y=y, height=a, sigma=s) for x, y, a, s in H["hills"]],
                         "formula": "h(x, y) = tilt[0] x + tilt[1] y + sum height exp(-((x-hx)^2 + (y-hy)^2) / (2 sigma^2))"},
                 overview={"centre": list(ov_extent(H)[:2]), "extent_m": ov_extent(H)[2]})
        if H.get("mando"):
            M = H["mando"]
            e["mandatory"] = dict(point=[M["x"], M["y"]], pass_on=M["side"], drop_zone=list(M["dz"]),
                                  rule=(f"the disc must cross the line x = {M['x']} at y < {M['y']} (pass the mandatory "
                                        f"on its {M['side']}); crossing it at y > {M['y']} misses it (PDGA 804.01)"))
        out.append(e)
    return out


def spec_cameras():
    t = math.tan(math.radians(FOVY_TEE / 2))
    return {
        "tee": {"tile_px": [TEE_W, TEE_H], "projection": "pinhole", "fovy_deg": FOVY_TEE,
                "focal_px": round(TEE_H / 2 / t, 2), "principal_point_px": [TEE_W / 2, TEE_H / 2],
                "pose": f"{TEE_BACK} m behind your lie on the line from the basket through the lie, {TEE_UP} m above "
                        f"the ground at the lie, looking along the heading from the lie to the basket, pitched "
                        f"{TEE_PITCH} deg up, no roll; your launcher stands in front of it"},
        "overview": {"tile_px": [OV_W, OV_H], "projection": "orthographic, straight down",
                     "scale": "extent_m (per hole, holes[].overview) metres across the tile's height, square pixels; "
                              "the tile centre is holes[].overview.centre; image right = +x, image up = +y",
                     "shows": "the whole hole: trees (green discs = canopies), the basket (yellow), the putting circle "
                              "(white, 10 m), OB lines (white) and OB water (blue), the tee pad (grey), a mandatory's "
                              "restricted line (red) and drop zone (yellow square), your launcher (orange) at your "
                              "lie; the flight of your previous throw on this hole as yellow dots (every 0.05 s)"},
        "wind": {"tile_px": [WIND_W, WIND_H], "projection": "orthographic, straight down",
                 "scale": f"{WIND_EXTENT} m across the tile's height; the tile centre is the top of the windsock pole "
                          f"(at the hole's windsock, {list(WINDSOCK)}, {WINDSOCK_H} m tall); image right = +x, up = +y",
                 "windsock": f"{SOCK_L} m long, 5 stripes, orange at the mouth (the pole end); it points downwind and "
                             f"hangs at 90 deg x (1 - speed / 10 m/s) below horizontal (horizontal at 10 m/s or more)"},
        "pinhole": "u = cx + f X/Z, v = cy - f Y/Z in the camera frame (X right, Y up, Z forward)",
    }


# ----------------------------------------------------------------------------------------------- video
def render_video(actions, out, fps=20):
    """Replay: a chase camera behind the disc and the overview inset with the flight; a score line."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    v = Views()
    c = Course()
    w = imageio.get_writer(out, fps=fps, macro_block_size=8)
    for a in actions:
        if c.done:
            break
        hi, lie = c.hole, c.lie
        H = HOLES[hi]
        n = c.count()
        rec, r = c.play(a)
        tr = r["trail"]
        wind = c.last["wind"] if c.last else (0, 0)
        yaw = heading_to_basket(H, lie)
        for k in range(0, len(tr), 2):
            _, x, y, z = tr[k]
            v.place(hi, lie, yaw, wind, disc_pos=[x, y, z])
            m, d = v.model(hi)
            rr = v.renderer(hi, 960, 544)
            m.vis.global_.fovy = 50.0
            import mujoco as mj
            cam = mj.MjvCamera()
            cam.lookat[:] = [x, y, z]
            cam.distance = 9.0
            cam.azimuth = yaw
            cam.elevation = -15.0
            rr.update_scene(d, camera=cam)
            im = Image.fromarray(rr.render().copy())
            cx, cy, ext = ov_extent(H)
            ov = v.ortho(hi, 420, 210, "overview")
            ov = Image.fromarray(ov)
            dr2 = ImageDraw.Draw(ov)
            sc = 210 / ext
            for _, px, py, _ in tr[:k + 1:3]:
                u, vv = 210 + (px - cx) * sc, 105 - (py - cy) * sc
                dr2.ellipse([u - 1.5, vv - 1.5, u + 1.5, vv + 1.5], fill=(255, 225, 30))
            im.paste(ov, (im.width - 426, 32))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
            dr.text((8, 7), f"{H['name']} (par {H['par']})  throw {n + 1}  {a['disc']} {a['speed']:.1f} m/s  "
                            f"-> {rec['status']} ({rec['distance_to_basket_m']:.1f} m)", fill=(255, 255, 255))
            w.append_data(np.asarray(im))
    w.close()
