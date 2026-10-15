"""Self-authored 3D wheel/contact racing and its referee. No downloaded models or race tracks.

Rules: RoboRacer Rules (racing format, crash penalties, obstacle and border crashes, warnings, tie-breaks), adapted as
listed in sport.toml [[deviations]] (sports/racing/); the multi-car race (3-6 cars) adds a points classification
(FIA Formula 1 Sporting Regulations, points and count-back) that the RoboRacer rules do not have. Only the trusted
world/referee imports this module. Public observations are built explicitly in plugin.py; the camera reference
(builtin_players.py) never imports it.

Track (v3): a seeded closed circuit built from a curvature profile (straights and turns with cosine curvature ramps,
so the centre line is C2): one left hairpin, an S-bend, a chicane, left turns and usually a right-hand sweeper, in a
seeded order; the straight lengths are solved (non-negative least squares) so the loop closes exactly; then the loop is
checked for its length, extent and self-clearance. Fixed obstacles (barrels and blocks) stand on straight-ish
sections, each leaving a free gap of at least 1.5 m. Everything is a pure function of the seed.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

DT, PHYSICS_DT = 0.01, 0.002               # 10 ms control step (eai-standard-realtime/1.0), 500 Hz physics
SUBSTEPS = round(DT / PHYSICS_DT)
HEAT_PACE = 1.25                           # m/s: the heat time limit is the lap length at this pace, rounded up to 5 s
MAX_HEATS, HEATS_TO_WIN = 3, 2             # head-to-head: up to three heats; two heat wins win it
GATES = 16                                 # ordered referee lines across the full track width; gate 16 = finish line
WIDTH, WHEEL_RADIUS, WHEELBASE = 2.4, 0.10, 0.46   # WIDTH: the v2 constant, kept for imports; v3 width is per track
MAX_SPEED, MIN_SPEED, MAX_STEER = 4.0, -1.5, 0.48
LANE = 0.42                                # grid positions: centre line +-0.42 m (inner = left +, outer = right -)
GRID_STEP = 0.9                            # multi-car grid: staggered slots 0.9 m apart (1.8 m on each side)
CAM_W, CAM_H = 240, 160
SEATS = ("A", "B")                         # the single-player / duel seats
ALL_SEATS = ("A", "B", "C", "D", "E", "F")
RACE_CARS = (3, 6)                         # multi-car race: 3-6 cars (default 4)
POINTS = (25, 18, 15, 12, 10, 8)           # FIA F1 points for places 1-6 (multi-car race)
FINISH_WINDOW_S = 10.0                     # multi-car race: the heat ends this long after the first car finished
# Car-to-car crash rule (RoboRacer "Race penalties" / "Race violations"), decided from physics only:
CRASH_DV = 0.30         # a contact episode whose impulse could change a car's speed by >= 0.30 m/s is a crash
STUCK_S = 2.0           # contact for 2.0 s is a crash whatever the impulse (wedged / pushing / excessive touching)
FAULT_MARGIN = 0.10     # m/s: closing-speed difference below which the trailing car is at fault
BEHIND_M = 2.0          # the at-fault car is placed 2 m behind the other car, at rest
WARNINGS_DQ = 3         # every at-fault crash is a warning; the third disqualifies
# Obstacle crash (RoboRacer "Race penalties": stop the car, move it to the side of the track next to the latest
# position before the crash): the same impulse / contact-time test as a car-car crash; no warning.
OBSTACLE_HOLD_S = 1.0   # the car is placed at rest beside the obstacle and held for 1 s (the stop and the move)
SCRIPTED_SPEED = [1.75, 2.2, 2.7, 3.2]     # level 1-4 cruise speed of the single-player opponent (m/s)
R_MIN = 2.5                                # smallest centre-line turn radius (car limit 0.46 / tan 0.48 = 0.89 m)
DS = 0.02                                  # dense centre-line spacing (m)
CAR_COLORS = {'A': '.05 .35 .95 1', 'B': '.95 .25 .04 1', 'C': '0 .75 .8 1', 'D': '.5 .2 .8 1',
              'E': '.5 .3 .12 1', 'F': '.72 .72 .75 1'}
OBSTACLE_COLOR = '1 0 .7 1'                # magenta: no car, wall, road or stripe has it


def referee_sha():
    """Fingerprint of the physics + referee code (legacy: race.py + replay.py; the v0.2 core's source_sha covers the
    whole sport directory). Not part of the state digest."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    h.update(b"race.py"); h.update((here / "race.py").read_bytes())
    return h.hexdigest()


# ---- the circuit -------------------------------------------------------------------------------------------------
def _turn(phi, radius):
    """A turn of signed angle phi: cosine curvature ramps (<= 1.2 m) around a plateau of radius `radius`."""
    k = math.copysign(1.0 / radius, phi); ramp = min(1.2, 0.5 * abs(phi) * radius)
    return [('up', ramp, k), ('flat', abs(phi) * radius - ramp, k), ('down', ramp, k)]


def _integrate(pieces, h=0.005):
    """Integration of a curvature profile (midpoint heading): points, curvature, final heading."""
    ks, hs = [], []
    for kind, length, k in pieces:
        n = max(1, int(round(length / h))); u = (np.arange(n) + .5) / n
        if kind == 'up': kk = k * (1 - np.cos(np.pi * u)) / 2
        elif kind == 'down': kk = k * (1 + np.cos(np.pi * u)) / 2
        else: kk = np.full(n, float(k))
        ks.append(kk); hs.append(np.full(n, length / n))
    k = np.concatenate(ks); hh = np.concatenate(hs)
    th_end = np.cumsum(k * hh); th_mid = th_end - k * hh / 2
    xy = np.concatenate([[[0., 0.]], np.cumsum(np.column_stack([hh * np.cos(th_mid), hh * np.sin(th_mid)]), axis=0)])
    return xy, np.concatenate([[0.], k]), float(th_end[-1])


def _nnls(A, b):
    """Lawson-Hanson non-negative least squares (tiny systems: 2 x <= 9)."""
    n = A.shape[1]; P = []; x = np.zeros(n)
    for _ in range(4 * n):
        w = A.T @ (b - A @ x); Z = [i for i in range(n) if i not in P]
        if not Z or max(w[Z]) <= 1e-10: break
        P.append(max(Z, key=lambda i: w[i]))
        while True:
            z = np.zeros(n); z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if all(z[P] > 0): x = z; break
            q = [i for i in P if z[i] <= 0]; a = min(x[i] / (x[i] - z[i]) for i in q)
            x = x + a * (z - x); P = [i for i in P if x[i] > 1e-12]
    return x


def _blocks(rng):
    d = math.radians
    blocks = [('hairpin', [('T', d(rng.uniform(170, 190)), rng.uniform(2.5, 3.0))])]
    sg = float(rng.choice([-1, 1])); a = d(rng.uniform(50, 80)); r = rng.uniform(2.8, 3.6)
    blocks.append(('s_bend', [('T', sg * a, r), ('S', rng.uniform(0.3, 1.2)), ('T', -sg * a, r)]))
    sg = float(rng.choice([-1, 1])); a = d(rng.uniform(25, 40)); r = rng.uniform(2.6, 3.4)
    blocks.append(('chicane', [('T', sg * a, r), ('S', 0.4), ('T', -2 * sg * a, r), ('S', 0.4), ('T', sg * a, r)]))
    right = d(rng.uniform(35, 70)) if rng.random() < 0.7 else 0.0
    if right: blocks.append(('right_sweeper', [('T', -right, rng.uniform(3.0, 5.0))]))
    rest = 2 * math.pi - blocks[0][1][0][1] + right
    k = int(rng.integers(2, 4)); w = rng.uniform(0.6, 1.4, k); w = w / w.sum() * rest
    for x in w: blocks.append(('left_turn', [('T', float(x), rng.uniform(2.7, 4.8))]))
    return blocks


def _pieces(block):
    out = []
    for p in block: out += [('flat', p[1], 0.)] if p[0] == 'S' else _turn(p[1], p[2])
    return out


class Circuit:
    """The seeded closed circuit: dense centre line (s = 0 is the start/finish line, counter-clockwise), width,
    exact-enough projection, features and obstacles."""
    START_IN_STRAIGHT = 6.0                # the start/finish line lies 6 m into the main straight (grid behind it)

    def __init__(self, seed):
        rng = np.random.default_rng([int(seed), 2026])
        for attempt in range(1000):
            if self._attempt(rng): break
        else: raise RuntimeError(f'no circuit for seed {seed}')
        self.seed, self.attempt = int(seed), attempt
        self._obstacles(rng)

    def _attempt(self, rng):
        blocks = _blocks(rng)
        local = []
        for _, b in blocks:
            xy, _, th = _integrate(_pieces(b)); local.append((xy[-1], th))
        lmins = [9.0] + list(rng.uniform(1.0, 3.5, len(blocks) - 1))
        for _ in range(150):
            order = rng.permutation(len(blocks)); th = 0.; D = np.zeros(2); U = []
            for j, bi in enumerate(order):
                u = np.array([math.cos(th), math.sin(th)]); U.append(u); D += lmins[j] * u
                v, dth = local[bi]; c, s = math.cos(th), math.sin(th); D += [c * v[0] - s * v[1], s * v[0] + c * v[1]]; th += dth
            A = np.array(U).T; x = _nnls(A, -D)
            if np.linalg.norm(A @ x + D) > 1e-8: continue
            L = np.round(x + np.array(lmins), 6)
            if L[0] <= 15 and L[1:].max() <= 9: break
        else: return False
        pieces, marks, pos = [], [], 0.
        for j, bi in enumerate(order):
            pieces.append(('flat', float(L[j]), 0.)); pos += float(L[j])
            p = _pieces(blocks[bi][1]); ln = sum(x[1] for x in p); marks.append((blocks[bi][0], pos, pos + ln)); pos += ln
            pieces += p
        xy, k, th = _integrate(pieces)
        if np.linalg.norm(xy[-1] - xy[0]) > 1e-4 or abs(th - 2 * math.pi) > 1e-6: return False
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1); s = np.concatenate([[0.], np.cumsum(seg)]); length = float(s[-1])
        if not 45 <= length <= 70: return False
        n = int(round(length / DS)); su = np.arange(n) * length / n
        shift = self.START_IN_STRAIGHT              # re-origin: s = 0 at the start/finish line
        sq = (su + shift) % length
        pts = np.column_stack([np.interp(sq, s, xy[:, 0]), np.interp(sq, s, xy[:, 1])])
        curv = np.interp(sq, s, k)
        w0, dw, m, ph = rng.uniform(2.6, 2.8), rng.uniform(0., .15), int(rng.integers(2, 4)), rng.uniform(0, 2 * math.pi)
        width = w0 + dw * np.sin(2 * math.pi * m * su / length + ph)
        # self-clearance: parts of the loop more than 6.5 m apart along it stay a corridor width + 1.2 m apart
        sub, ss, ww = pts[::10], su[::10], width[::10]
        Dm = np.linalg.norm(sub[:, None] - sub[None], axis=2); Am = np.abs(ss[:, None] - ss[None]); Am = np.minimum(Am, length - Am)
        if np.any((Am > 6.5) & (Dm < (ww[:, None] + ww[None]) / 2 + 1.2)): return False
        lo, hi = pts.min(0), pts.max(0)
        if (hi - lo).max() > 26: return False
        centre = (lo + hi) / 2
        self.pts = np.round(pts - centre, 6); self.curv = curv; self.width = np.round(width, 6)
        self.length, self.n, self.ds = length, n, length / n
        t = np.roll(self.pts, -1, axis=0) - np.roll(self.pts, 1, axis=0); self.tan = t / np.linalg.norm(t, axis=1)[:, None]
        self.nor = np.column_stack([-self.tan[:, 1], self.tan[:, 0]])
        self.extent = (hi - lo) / 2
        self.features = [dict(kind=kname, s0=round((a - shift) % length, 3), s1=round((b - shift) % length, 3)) for kname, a, b in marks]
        return True

    # geometry on the dense centre line
    def point(self, s):
        f = (s % self.length) / self.ds; i = int(f) % self.n; j = (i + 1) % self.n; a = f - int(f)
        p = self.pts[i] * (1 - a) + self.pts[j] * a; t = self.tan[i] * (1 - a) + self.tan[j] * a
        return p, t / np.linalg.norm(t)

    def width_at(self, s):
        f = (s % self.length) / self.ds; i = int(f) % self.n; a = f - int(f)
        return float(self.width[i] * (1 - a) + self.width[(i + 1) % self.n] * a)

    def project(self, p):
        """s (0..length) and lateral (> 0: left of the driving direction) of a point: nearest dense sample, then the
        orthogonal projection onto the neighbouring segment."""
        p = np.asarray(p, dtype=float)[:2]
        d = self.pts - p; i = int(np.argmin(np.einsum('ij,ij->i', d, d)))
        best = None
        for a, b in (((i - 1) % self.n, i), (i, (i + 1) % self.n)):
            e = self.pts[b] - self.pts[a]; L2 = float(e @ e); t = min(1., max(0., float((p - self.pts[a]) @ e) / L2))
            q = self.pts[a] + t * e; dist = float(np.linalg.norm(p - q))
            if best is None or dist < best[0]:
                lat = float(e[0] * (p[1] - q[1]) - e[1] * (p[0] - q[0])) / math.sqrt(L2)
                best = (dist, (a + t) * self.ds, lat)
        return best[1] % self.length, best[2]

    # obstacles: fixed barrels and blocks on straight-ish sections, each leaving a free gap of >= 1.5 m
    def _obstacles(self, rng):
        count = int(rng.integers(3, 5)); self.obstacles = []
        for attempt in range(3000):
            if len(self.obstacles) == count: break
            # straight-ish sections first; after 2000 tries (fewer than two placed) gentle bends too
            straight = np.abs(self.curv) <= (1 / 4.5 if attempt < 2000 or len(self.obstacles) >= 2 else 1 / 3.2)
            if attempt >= 2000 and len(self.obstacles) >= 2: break
            s = float(rng.uniform(9.0, self.length - 7.0))
            i0, i1 = int((s - 2.0) / self.ds), int((s + 1.0) / self.ds)
            if not straight[np.arange(i0, i1) % self.n].all(): continue
            if any(min(abs(s - o['s']), self.length - abs(s - o['s'])) < 7.0 for o in self.obstacles): continue
            kind = 'barrel' if rng.random() < .5 else 'block'
            half_lat = .17 if kind == 'barrel' else .20
            side = float(rng.choice([-1., 1.])); w = self.width_at(s); gap = float(rng.uniform(.05, .55))
            lat = side * (w / 2 - half_lat - gap)
            if w - 2 * half_lat - gap < 1.5: continue
            p, t = self.point(s); xy = p + lat * np.array([-t[1], t[0]])
            self.obstacles.append(dict(id=len(self.obstacles), kind=kind, s=round(s, 4), lat=round(lat, 4),
                                       x=round(float(xy[0]), 4), y=round(float(xy[1]), 4),
                                       yaw=round(math.atan2(t[1], t[0]), 5), half_lat=half_lat,
                                       half_len=.17 if kind == 'barrel' else .25, side=side))
        self.obstacles.sort(key=lambda o: o['s'])
        for i, o in enumerate(self.obstacles): o['id'] = i

    def free_gap(self, o):
        """The obstacle's free side: the (low, high) lateral interval of the road beside it."""
        w = self.width_at(o['s'])
        return (-w / 2, o['lat'] - o['half_lat']) if o['side'] > 0 else (o['lat'] + o['half_lat'], w / 2)

    def line(self, s, lane):
        """A collision-free driving line: lateral `lane`, detouring through each obstacle's free gap (trusted
        drivers only: the scripted car and the tests)."""
        target = lane
        for o in self.obstacles:
            d = (s - o['s'] + self.length / 2) % self.length - self.length / 2      # signed, -L/2..L/2
            if d < -5.0 or d > 3.8: continue
            lo, hi = self.free_gap(o); m = .23 + .30                                 # car half width + margin
            want = float(np.clip(lane, lo + m, hi - m))
            if abs(want - lane) < 1e-9: continue
            # cosine ramps of 4 m in and 3 m out: the line's radius stays above 1.2 m
            if d < -1.0: wgt = .5 - .5 * math.cos(math.pi * (d + 5.0) / 4.0)
            elif d <= 0.8: wgt = 1.
            else: wgt = .5 + .5 * math.cos(math.pi * (d - 0.8) / 3.0)
            target = target + wgt * (want - target)
        return target

    def summary(self, every=0.5):
        """Public geometry (the broadcast header): centre line every 0.5 m with the width, features, obstacles."""
        k = max(1, int(round(every / self.ds)))
        return dict(kind='circuit', length_m=round(self.length, 3),
                    width_m=[round(float(self.width.min()), 3), round(float(self.width.max()), 3)],
                    extent_m=[round(float(x), 3) for x in self.extent], gates=GATES, min_radius_m=R_MIN,
                    features=self.features,
                    centreline=[[round(float(x), 3), round(float(y), 3), round(float(w), 3)]
                                for (x, y), w in zip(self.pts[::k], self.width[::k])],
                    obstacles=[dict(id=o['id'], kind=o['kind'], x=o['x'], y=o['y'], yaw=o['yaw']) for o in self.obstacles])


@functools.lru_cache(maxsize=8)
def circuit(seed):
    return Circuit(seed)


def heat_seconds(seed):
    return 5.0 * math.ceil(circuit(seed).length / HEAT_PACE / 5.0)


def track(seed):
    """v2 interface, kept for the broadcast adapter: (a, r, length, points, tangents). For a v3 circuit (a, r) are the
    oval-equivalent extents (a + r = half width of the bounding box, r = half height; the circuit is centred at the
    origin); points / tangents are 256 samples of the centre line."""
    c = circuit(seed); idx = (np.arange(256) * c.n // 256)
    hx, hy = float(c.extent[0]), float(c.extent[1])
    return max(0., hx - hy), hy, c.length, c.pts[idx], c.tan[idx]


def grid_slots(cars):
    """(s, lateral) of each grid slot: two cars side by side on the line; 3+ cars staggered behind it."""
    if cars == 2: return [(0., LANE), (0., -LANE)]
    return [(-GRID_STEP * k, LANE if k % 2 == 0 else -LANE) for k in range(cars)]


def scene_xml(seed, cars=2):
    """The MJCF scene: circuit, walls, obstacles and `cars` cars (A, B, ...). v2 call scene_xml(seed) = 2 cars."""
    c = circuit(seed); seats = ALL_SEATS[:cars]
    pieces = []
    idx = list(range(0, c.n, 12))               # visual / wall segments every 12 dense samples (0.24 m)
    for a, i in enumerate(idx):
        j = idx[(a + 1) % len(idx)]
        p, q = c.pts[i], c.pts[j]; mid = (p + q) / 2; seg = float(np.linalg.norm(q - p)); yaw = math.atan2(q[1] - p[1], q[0] - p[0])
        w = (c.width[i] + c.width[j]) / 2
        dth = abs(math.atan2(c.tan[j][1], c.tan[j][0]) - math.atan2(c.tan[i][1], c.tan[i][0])); dth = min(dth, 2 * math.pi - dth)
        over = w / 2 * dth / 2 + .01
        # Visual-only road tiles. All wheel contacts use the single flat ground.
        pieces.append(f'<geom type="box" pos="{mid[0]:.5f} {mid[1]:.5f} .002" size="{seg/2+over:.5f} {w/2:.5f} .002" euler="0 0 {yaw:.6f}" rgba=".13 .32 .17 1" contype="0" conaffinity="0"/>')
        # A narrow yellow center stripe, visible but not a policy/waypoint service.
        pieces.append(f'<geom type="box" pos="{mid[0]:.5f} {mid[1]:.5f} .006" size="{seg/2+.005:.5f} .025 .002" euler="0 0 {yaw:.6f}" rgba="1 .85 .05 1" contype="0" conaffinity="0"/>')
    for side in (1, -1):                        # walls: boxes along the offset polylines (continuous, rigid)
        off = [c.pts[i] + side * (c.width[i] / 2 + .06) * c.nor[i] for i in idx]
        for a in range(len(off)):
            p, q = off[a], off[(a + 1) % len(off)]; mid = (p + q) / 2; seg = float(np.linalg.norm(q - p)); yaw = math.atan2(q[1] - p[1], q[0] - p[0])
            color = '.9 .9 .9 1' if (a // 4) % 2 else '.8 .14 .12 1'
            pieces.append(f'<geom name="wall_{a}_{side}" type="box" pos="{mid[0]:.5f} {mid[1]:.5f} .14" size="{seg/2+.03:.5f} .06 .14" euler="0 0 {yaw:.6f}" rgba="{color}" friction=".6 .01 .001"/>')
    p, t = c.point(0.); yaw = math.atan2(t[1], t[0]); w0 = c.width_at(0.)
    pieces.append(f'<geom name="finish_line" type="box" pos="{p[0]:.5f} {p[1]:.5f} .004" size=".05 {w0/2:.5f} .002" euler="0 0 {yaw:.6f}" rgba=".95 .95 .95 1" contype="0" conaffinity="0"/>')
    for o in c.obstacles:                       # static world geoms: they cannot be pushed
        if o['kind'] == 'barrel':
            pieces.append(f'<geom name="obstacle_{o["id"]}" type="cylinder" pos="{o["x"]} {o["y"]} .2" size=".17 .2" rgba="{OBSTACLE_COLOR}" friction=".6 .01 .001"/>')
        else:
            pieces.append(f'<geom name="obstacle_{o["id"]}" type="box" pos="{o["x"]} {o["y"]} .15" size=".25 .2 .15" euler="0 0 {o["yaw"]}" rgba="{OBSTACLE_COLOR}" friction=".6 .01 .001"/>')
    cars_xml, acts, excludes = [], [], []
    for seat, (s0, lat) in zip(seats, grid_slots(cars)):
        p, t = c.point(s0); xy = p + lat * np.array([-t[1], t[0]]); yaw = math.atan2(t[1], t[0])
        car = [f'<body name="car_{seat}" pos="{xy[0]:.5f} {xy[1]:.5f} .155" euler="0 0 {yaw:.6f}"><freejoint name="root_{seat}"/>',
               f'<geom name="chassis_{seat}" type="box" size=".30 .15 .055" mass="1.8" rgba="{CAR_COLORS[seat]}"/>',
               '<geom type="box" pos="-.035 0 .085" size=".13 .12 .04" mass=".1" rgba=".16 .18 .23 1"/>',
               # camera local x points to the car's right; optical -z points forward/down.
               f'<camera name="front_{seat}" pos=".22 0 .24" xyaxes="0 -1 0 .342 0 .940" fovy="70"/>',
               f'<camera name="chase_{seat}" pos="-1.5 0 1.35" xyaxes="0 -1 0 .58 0 .815" fovy="70"/>']
        for axle, x in [('front', .23), ('rear', -.23)]:
            for side, yoff in [('left', .19), ('right', -.19)]:
                name = f'{seat}_{axle}_{side}'
                if axle == 'front':
                    car += [f'<body name="steer_{name}" pos="{x} {yoff} -.05">',
                            '<inertial pos="0 0 0" mass=".01" diaginertia=".0001 .0001 .0001"/>',
                            f'<joint name="steer_{name}" axis="0 0 1" range="-.48 .48" damping=".1"/>',
                            f'<body name="wheelbody_{name}" pos="0 0 0">']
                    acts.append(f'<position name="steer_{name}" joint="steer_{name}" kp="15" kv="1" ctrlrange="-.48 .48" forcerange="-3 3"/>')
                else:
                    car += [f'<body name="wheelbody_{name}" pos="{x} {yoff} -.05">']
                excludes.append(f'<exclude body1="car_{seat}" body2="wheelbody_{name}"/>')
                car += [f'<joint name="wheel_{name}" axis="0 1 0" damping=".001"/>',
                        f'<geom name="wheel_{name}" type="cylinder" size=".1 .035" euler="{math.pi/2} 0 0" mass=".10" rgba=".08 .08 .09 1" friction="1.3 .01 .001" condim="4"/>', '</body>']
                if axle == 'front': car += ['</body>']
                else: acts.append(f'<velocity name="drive_{name}" joint="wheel_{name}" kv=".08" ctrlrange="-15 40" forcerange="-.7 .7"/>')
        car += ['</body>']; cars_xml += car
    ex, ey = float(c.extent[0]), float(c.extent[1]); ground = max(ex, ey) + 8
    height = 1.15 * max(ex, ey * 1.6) + 2
    return f'''<mujoco model="visual_wheel_racing">
      <compiler angle="radian"/><option timestep="{PHYSICS_DT}" integrator="implicitfast" iterations="40" cone="elliptic"/>
      <visual><global offwidth="640" offheight="480"/><quality shadowsize="1024"/><headlight ambient=".5 .5 .5" diffuse=".5 .5 .5"/></visual>
      <default><geom solref=".008 1" solimp=".95 .99 .001"/></default>
      <worldbody><light pos="0 0 12" directional="true" dir="0 0 -1" castshadow="false"/>
      <geom name="ground" type="plane" size="{ground:.2f} {ground:.2f} .1" rgba=".22 .24 .25 1" friction="1.3 .01 .001"/>
      <camera name="overview" pos="0 {-0.75 * height:.3f} {height:.3f}" xyaxes="1 0 0 0 .8 .6" fovy="65"/>
      {''.join(pieces)}{''.join(cars_xml)}</worldbody><contact>{''.join(excludes)}</contact><actuator>{''.join(acts)}</actuator></mujoco>'''


def validate_action(body):
    if body.get('type') == 'none': return [0.0, 0.0]
    if body.get('type', 'drive') != 'drive': raise ValueError('action type must be drive or none')
    v, steer = body.get('speed'), body.get('steering')
    if type(v) not in (int, float) or type(steer) not in (int, float): raise ValueError('speed and steering must be numbers')
    if not math.isfinite(v) or not math.isfinite(steer): raise ValueError('action must be finite')
    if not MIN_SPEED <= v <= MAX_SPEED or not -MAX_STEER <= steer <= MAX_STEER: raise ValueError('action outside speed/steering limits')
    return [float(v), float(steer)]


def other(seat): return 'B' if seat == 'A' else 'A'


def lot(seed, heat, what, seats=SEATS):
    """The stated last resort (RoboRacer: 'a random method, e.g. coin flip'): a fixed draw from the instance seed."""
    return seats[int(hashlib.sha256(f'{seed}:{heat}:{what}'.encode()).hexdigest(), 16) % len(seats)]


def lot_key(seed, heat, what, seat):
    """A fixed draw that orders several cars (the last resort of a multi-car ordering)."""
    return hashlib.sha256(f'{seed}:{heat}:{what}:{seat}'.encode()).hexdigest()


def yaw_quat(yaw): return [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]


class Race:
    def __init__(self, seed=101, level=1, duel=False, first='A', cars=2):
        if level not in (1, 2, 3, 4): raise ValueError('invalid level')
        if cars != 2 and not (duel and RACE_CARS[0] <= cars <= RACE_CARS[1]): raise ValueError('invalid number of cars')
        self.seats = ALL_SEATS[:cars]
        if first not in self.seats: raise ValueError('invalid first grid seat')
        self.seed, self.level, self.duel, self.first, self.cars = seed, level, duel, first, cars
        self.multi = cars > 2                    # the multi-car race format (points per heat position)
        self.max_heats = cars if self.multi else MAX_HEATS
        self.referee_sha = referee_sha()
        self.circuit = c = circuit(seed)
        self.length, self.point, self.project = c.length, c.point, c.project
        self.a, self.radius, _, self.track, self.tangents = track(seed)     # v2 attributes (broadcast adapter)
        self.heat_seconds = heat_seconds(seed)
        self.max_steps = self.max_heats * round(self.heat_seconds / DT)
        self.model = mujoco.MjModel.from_xml_string(scene_xml(seed, cars))
        self.data = mujoco.MjData(self.model)
        m = self.model
        # geom -> car index + 1 (0 none); wall geoms; obstacle index + 1; each car's dofs and mass.
        self.geom_car = np.zeros(m.ngeom, dtype=np.int64)
        self.geom_wall = np.zeros(m.ngeom, dtype=bool)
        self.geom_obstacle = np.zeros(m.ngeom, dtype=np.int64)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b and not (m.body(b).name or '').startswith('car_'): b = int(m.body_parentid[b])
            if b: self.geom_car[g] = self.seats.index(m.body(b).name[4:]) + 1
            name = m.geom(g).name or ''
            self.geom_wall[g] = name.startswith('wall_')
            if name.startswith('obstacle_'): self.geom_obstacle[g] = int(name[9:]) + 1
        self.body_id = {s: m.body(f'car_{s}').id for s in self.seats}
        self.qadr = {s: int(m.jnt_qposadr[m.joint(f'root_{s}').id]) for s in self.seats}
        self.vadr = {s: int(m.jnt_dofadr[m.joint(f'root_{s}').id]) for s in self.seats}
        self.dofs = {s: [] for s in self.seats}
        for j in range(m.njnt):
            name = m.joint(j).name or ''
            for s in self.seats:
                if name == f'root_{s}': self.dofs[s] += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + 6))
                elif any(name.endswith(f'{s}_{x}') for x in ('front_left', 'front_right', 'rear_left', 'rear_right')):
                    self.dofs[s].append(int(m.jnt_dofadr[j]))
        self.mass = float(m.body_subtreemass[self.body_id['A']])
        self.crash_impulse = self.mass * CRASH_DV
        self.act = {s: {side: (m.actuator(f'drive_{s}_rear_{side}').id, m.actuator(f'steer_{s}_front_{side}').id)
                        for side in ('left', 'right')} for s in self.seats}
        self.k, self.heat, self.heat_k = 0, 0, 0
        self.results, self.commands = [], {s: [0., 0.] for s in self.seats}
        self.wins = {s: 0 for s in self.seats}; self.warnings = {s: 0 for s in self.seats}
        self.points = {s: 0 for s in self.seats}
        self.obstacle_crashes = {s: 0 for s in self.seats}
        self.retired = {}                    # seat -> dict(step, heat, why): disqualified / did not start / forfeit
        self.done, self.awaiting_review, self.winner_seat, self.end_reason = False, False, None, None
        self.wall_steps, self.contact_steps = {s: 0 for s in self.seats}, 0
        self.obstacle_steps = {s: 0 for s in self.seats}
        self.crashes = []                    # every crash call of the race (car-car and obstacle)
        self.events = []                     # referee events of the last step (for the live log / broadcast)
        self.reset_heat()

    # ---- grid and state -------------------------------------------------------------------------------------------
    def inner_seat(self, heat):
        """Head-to-head. Heat 1: FIRST_PLAYER starts outside. Heat 2: sides switch. Heat 3: the faster lap of heats 1-2
        starts inside (the advantageous side; RoboRacer: the better Time Trial rank chooses); none or equal: as heat 1."""
        if heat == 0: return other(self.first)
        if heat == 1: return self.first
        laps = {r['winner']: r['finish_time'] for r in self.results if r['finish_time'] is not None}
        if len(laps) == 2 and laps['A'] != laps['B']: return min(laps, key=laps.get)
        if len(laps) == 1: return next(iter(laps))
        return other(self.first)

    def grid_order(self, heat):
        """Multi-car: the grid of heat h, pole first. Heat 1: FIRST_PLAYER on pole, then the seats in order; each heat
        rotates the grid by one slot, so over the N heats every car starts once from every slot."""
        i = self.seats.index(self.first); base = list(self.seats[i:] + self.seats[:i])
        return base[heat % self.cars:] + base[:heat % self.cars]

    def reset_heat(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.multi:
            order = self.grid_order(self.heat); slots = grid_slots(self.cars)
            place = {s: slots[order.index(s)] for s in self.seats}
            self.grid = {s: order.index(s) + 1 for s in self.seats}                 # slot 1 = pole
            self.grid_side = {s: 'inner' if place[s][1] > 0 else 'outer' for s in self.seats}
        else:
            inner = self.inner_seat(self.heat)
            place = {s: (0., LANE if s == inner else -LANE) for s in self.seats}
            self.grid = {s: ('inner' if s == inner else 'outer') for s in self.seats}
            self.grid_side = dict(self.grid)
        for s in self.seats:
            if s in self.retired: self.park(s, forward=False)
            else: self.place(s, *place[s], forward=False)
        self.commands = {s: [0., 0.] for s in self.seats}
        self.progress = {s: 0. for s in self.seats}
        self.reach_t = {s: 0. for s in self.seats}      # game time at which the current progress was first reached
        self.next_gate = {s: 1 for s in self.seats}
        self.finished = {s: None for s in self.seats}
        self.hold = {s: 0 for s in self.seats}          # obstacle crash: held at rest until this heat step
        self.heat_k = 0
        self.episodes = {}                              # contact episodes: car pairs, (car, obstacle id)
        self.heat_crashes = []
        mujoco.mj_forward(self.model, self.data)
        self.previous = {s: self.position(s)[:2].copy() for s in self.seats}

    def place(self, seat, s, lat, forward=True):
        p, t = self.point(s); xy = p + lat * np.array([-t[1], t[0]])
        q = self.qadr[seat]
        self.data.qpos[q:q+3] = [xy[0], xy[1], .155]
        self.data.qpos[q+3:q+7] = yaw_quat(math.atan2(t[1], t[0]))
        for dof in self.dofs[seat]: self.data.qvel[dof] = 0.
        if forward:
            mujoco.mj_forward(self.model, self.data)
            self.previous[seat] = self.position(seat)[:2].copy()

    def park(self, seat, forward=True):
        """A retired car leaves the track: parked at rest in a row beside the circuit, out of everyone's way."""
        q = self.qadr[seat]; i = self.seats.index(seat)
        self.data.qpos[q:q+3] = [-4. + 1.4 * i, -float(self.circuit.extent[1]) - 4., .155]
        self.data.qpos[q+3:q+7] = [1., 0., 0., 0.]
        for dof in self.dofs[seat]: self.data.qvel[dof] = 0.
        if forward:
            mujoco.mj_forward(self.model, self.data)
            self.previous[seat] = self.position(seat)[:2].copy()

    def free_spot(self, seat, s, lat, back=0.5, tries=12):
        """The first spot at or behind (s, lat) where the car touches no other car and no obstacle."""
        others = [self.position(x)[:2] for x in self.seats if x != seat and x not in self.retired]
        for i in range(tries):
            si = s - back * i
            w = self.circuit.width_at(si); li = float(np.clip(lat, -w / 2 + .4, w / 2 - .4))
            p, t = self.point(si); xy = p + li * np.array([-t[1], t[0]])
            if all(np.linalg.norm(xy - o) > .9 for o in others) and \
                    all(math.hypot(xy[0] - o['x'], xy[1] - o['y']) > o['half_len'] + .55 for o in self.circuit.obstacles):
                return si, li
        return si, li

    def position(self, seat): return self.data.xpos[self.body_id[seat]].copy()

    def heading(self, seat):
        rot = self.data.xmat[self.body_id[seat]].reshape(3, 3)
        return math.atan2(rot[1, 0], rot[0, 0])

    def planar_velocity(self, seat): return self.data.qvel[self.vadr[seat]:self.vadr[seat]+2].copy()

    def forward_speed(self, seat):
        fwd = self.data.xmat[self.body_id[seat]].reshape(3, 3)[:, 0]
        return float(self.data.qvel[self.vadr[seat]:self.vadr[seat]+3] @ fwd)

    def set_command(self, seat, command): self.commands[seat] = list(command)

    def active(self): return [s for s in self.seats if s not in self.retired]

    def track_position(self, seat):
        """Race distance this heat (gates passed * gate spacing + distance past the last gate, unwrapped) and lateral."""
        s, lat = self.project(self.position(seat)[:2])
        base = (self.next_gate[seat] - 1) * self.length / GATES
        if s - base > self.length / 2: s -= self.length
        if s - base < -self.length / 2: s += self.length
        return s, lat

    # ---- the scripted single-player opponent (trusted: it knows both cars' poses; never part of the reference) -----
    def scripted(self):
        speed = SCRIPTED_SPEED[self.level - 1]
        sB, latB = self.project(self.position('B')[:2]); sA, latA = self.project(self.position('A')[:2])
        gap = (sA - sB) % self.length
        lane = LANE if self.grid['B'] == 'inner' else -LANE
        line = self.circuit.line(sB + 1.05, lane); target = line
        edge = self.circuit.width_at(sA) / 2 - .45
        if gap < 3.5 and abs(latA - line) < 0.75:     # the other car is ahead in its line: pass on the freer side
            sides = [x for x in (latA + 0.8, latA - 0.8) if abs(x) <= edge and abs(self.circuit.line(sA, x) - x) < 1e-9]
            target = min(sides, key=lambda x: abs(x - latB)) if sides else line
        if gap < 2.0 and abs(latA - latB) < 0.55:     # close behind, still overlapping: keep a safe distance
            speed = min(speed, max(0., 2.0 * (gap - 0.85)))
        p = self.position('B')[:2]; yaw = self.heading('B')
        look = 1.05
        c, t = self.point(sB + look)
        aim = c + target * np.array([-t[1], t[0]]) - p
        err = math.atan2(math.sin(math.atan2(aim[1], aim[0]) - yaw), math.cos(math.atan2(aim[1], aim[0]) - yaw))
        return [speed, float(np.clip(math.atan2(2 * WHEELBASE * math.sin(err), look), -MAX_STEER, MAX_STEER))]

    # ---- one 10 ms control step --------------------------------------------------------------------------------------
    def step(self):
        if self.done or self.awaiting_review: raise RuntimeError('cannot step a frozen race')
        self.events = []
        if not self.duel: self.commands['B'] = self.scripted()
        for s in self.seats:
            speed, steer = (0., 0.) if (s in self.retired or self.heat_k < self.hold[s]) else self.commands[s]
            for side in ('left', 'right'):
                drive, st = self.act[s][side]
                self.data.ctrl[drive] = speed / WHEEL_RADIUS; self.data.ctrl[st] = steer
        pre_v = {s: self.planar_velocity(s) for s in self.seats}
        pre_p = {s: self.position(s)[:2].copy() for s in self.seats}
        impulse, walls = {}, set()
        f6 = np.zeros(6)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            n = self.data.ncon
            if not n: continue
            g = self.data.contact.geom[:n]
            c1, c2 = self.geom_car[g[:, 0]], self.geom_car[g[:, 1]]
            o1, o2 = self.geom_obstacle[g[:, 0]], self.geom_obstacle[g[:, 1]]
            for i in np.nonzero(((c1 > 0) & (c2 > 0) & (c1 != c2)) | ((c1 > 0) & (o2 > 0)) | ((c2 > 0) & (o1 > 0)))[0]:
                if c1[i] and c2[i]: key = tuple(sorted((self.seats[c1[i] - 1], self.seats[c2[i] - 1])))
                else: key = (self.seats[max(c1[i], c2[i]) - 1], int(max(o1[i], o2[i]) - 1))
                mujoco.mj_contactForce(self.model, self.data, int(i), f6)
                impulse[key] = impulse.get(key, 0.) + abs(float(f6[0])) * PHYSICS_DT
            w = self.geom_wall[g[:, 0]] | self.geom_wall[g[:, 1]]
            for car in np.concatenate([c1[w], c2[w]]):
                if car: walls.add(self.seats[int(car) - 1])
        self.k += 1; self.heat_k += 1
        t = self.heat_k * DT
        for s in self.seats:
            if s in self.retired: continue
            p = self.position(s)[:2]; old = self.previous[s]; gate = self.next_gate[s]
            # Strictly ordered gates. Reverse motion and teleport/corner cutting cannot earn a lap.
            center, tangent = self.point(gate * self.length / GATES)
            before, after = float((old - center) @ tangent), float((p - center) @ tangent)
            if self.finished[s] is None and before <= 0 < after:
                frac = -before / (after - before); crossed = old + frac * (p - old)
                lateral = abs(float((crossed - center) @ np.array([-tangent[1], tangent[0]])))
                if lateral <= self.circuit.width_at(gate * self.length / GATES) / 2 and np.linalg.norm(p - old) < MAX_SPEED * DT * 3:
                    self.next_gate[s] += 1
                    self.events.append(dict(ev='checkpoint', seat=s, gate=gate, t=round((self.heat_k - 1 + frac) * DT, 4)))
                    if gate == GATES: self.finished[s] = (self.heat_k - 1 + frac) * DT
            sv, _ = self.track_position(s)
            base = (self.next_gate[s] - 1) * self.length / GATES
            prog = min(self.length, max(base, min(sv, base + self.length / GATES)))
            if self.finished[s] is not None: prog = self.length
            if round(prog, 3) > round(self.progress[s], 3): self.reach_t[s] = t
            self.progress[s] = prog
            self.previous[s] = p.copy()
            if s in walls: self.wall_steps[s] += 1
        if any(isinstance(k[1], str) for k in impulse): self.contact_steps += 1
        for k in impulse:
            if isinstance(k[1], int): self.obstacle_steps[k[0]] += 1
        self.referee_contacts(impulse, pre_v, pre_p)
        if not self.done and not self.awaiting_review: self.check_heat_end()
        return dict(cars=sorted('-'.join(k) for k in impulse if isinstance(k[1], str)), walls=sorted(walls),
                    obstacles=sorted(f'{k[0]}-{k[1]}' for k in impulse if isinstance(k[1], int)))

    def referee_contacts(self, impulse, pre_v, pre_p):
        """RoboRacer 'Race penalties': touching is not penalized; a crash (the contact significantly changes a car's
        motion, or lasts 2 s) is called. Car-car: the at-fault car is placed 2 m behind the other car and receives a
        warning. Obstacle: the car is stopped and placed at the side of the track next to its position before the crash."""
        for key in [k for k in self.episodes if k not in impulse]: del self.episodes[key]
        for key in sorted(impulse, key=lambda k: (isinstance(k[1], int), str(k))):
            if self.done or self.awaiting_review: return
            if any(x in self.retired for x in key if isinstance(x, str)): continue
            ep = self.episodes.get(key)
            if ep is None:
                if isinstance(key[1], str):
                    a, b = key
                    # Fault is fixed at the first contact step from the cars' motion just before it: the car that drove
                    # into the other (its velocity towards the other car is larger) is at fault; equal: the trailing car.
                    d = pre_p[b] - pre_p[a]; d = d / max(1e-9, float(np.linalg.norm(d)))
                    close = {a: float(pre_v[a] @ d), b: float(-pre_v[b] @ d)}
                    if abs(close[a] - close[b]) > FAULT_MARGIN:
                        fault, how = max(close, key=close.get), 'drove_into'
                    else:
                        ra = {x: self.track_position(x)[0] for x in key}
                        fault, how = (min(ra, key=ra.get), 'trailing') if round(ra[a], 3) != round(ra[b], 3) else \
                            (lot(self.seed, self.heat, f'fault{self.k}', key), 'lot')
                    ep = dict(start=self.k, impulse=0., fault=fault, how=how, closing={x: round(v, 3) for x, v in close.items()})
                    self.events.append(dict(ev='contact', cars=list(key), closing=ep['closing']))
                else:
                    ep = dict(start=self.k, impulse=0.)
                    self.events.append(dict(ev='obstacle_contact', seat=key[0], obstacle=key[1]))
                self.episodes[key] = ep
            ep['impulse'] += impulse[key]
            steps = self.k - ep['start'] + 1
            if ep['impulse'] < self.crash_impulse and steps * DT < STUCK_S: continue
            del self.episodes[key]
            kind = 'stuck' if ep['impulse'] < self.crash_impulse else 'impact'
            if isinstance(key[1], int): self.obstacle_crash(key[0], key[1], ep, steps, kind)
            else: self.car_crash(key, ep, steps, kind)

    def obstacle_crash(self, seat, oid, ep, steps, kind):
        o = self.circuit.obstacles[oid]; lo, hi = self.circuit.free_gap(o)
        s, lat = self.free_spot(seat, o['s'] - o['half_len'] - .75, (lo + hi) / 2)
        self.place(seat, s, lat); self.hold[seat] = self.heat_k + round(OBSTACLE_HOLD_S / DT)
        for key in [k for k in self.episodes if seat in k]: del self.episodes[key]
        self.obstacle_crashes[seat] += 1
        call = dict(heat=self.heat + 1, t=round(self.heat_k * DT, 3), at_fault=seat, obstacle=oid, kind=kind,
                    fault_rule='obstacle', impulse=round(ep['impulse'], 4), contact_s=round(steps * DT, 3))
        self.crashes.append(call); self.heat_crashes.append(call)
        self.events.append(dict(ev='obstacle_crash', **call))

    def car_crash(self, key, ep, steps, kind):
        fault = ep['fault']; victim = key[1] if fault == key[0] else key[0]
        s_v, lat_v = self.track_position(victim)
        s, lat = self.free_spot(fault, s_v - BEHIND_M, float(np.clip(lat_v, -0.75, 0.75)))
        self.place(fault, s, lat)
        for k in [k for k in self.episodes if fault in k]: del self.episodes[k]
        self.warnings[fault] += 1
        call = dict(heat=self.heat + 1, t=round(self.heat_k * DT, 3), at_fault=fault, other=victim, fault_rule=ep['how'],
                    closing=ep['closing'], impulse=round(ep['impulse'], 4), contact_s=round(steps * DT, 3),
                    kind=kind, warning=self.warnings[fault])
        self.crashes.append(call); self.heat_crashes.append(call)
        self.events.append(dict(ev='crash', **call))
        if self.warnings[fault] >= WARNINGS_DQ:
            self.events.append(dict(ev='disqualified', seat=fault))
            if self.multi: self.retire(fault, 'disqualified')
            else: self.end_heat(victim, 'disqualification', race_winner=victim)

    # ---- retirements (multi-car race): disqualification, did not start, review forfeit -------------------------------
    def retire(self, seat, why):
        """Remove a car from the multi-car race: parked off the track, classified behind every running car. The race
        ends when fewer than two cars run. The service's retirements (start / review forfeits) are replayed from the
        action log ('retire' events)."""
        if not self.multi or seat not in self.seats or seat in self.retired or self.done: raise ValueError('cannot retire this car')
        self.retired[seat] = dict(step=self.k, heat=self.heat + 1, why=why)
        self.park(seat)
        for k in [k for k in self.episodes if seat in k]: del self.episodes[k]
        self.events.append(dict(ev='retired', seat=seat, why=why))
        if len(self.active()) <= 1:
            if not self.awaiting_review and self.heat_k > 0 and self.active(): self.end_heat(None, 'retirements')
            else: self.finish_race('retirements')

    # ---- heats ----------------------------------------------------------------------------------------------------
    def heat_summary(self):
        return {s: dict(checkpoints=self.next_gate[s] - 1, distance_m=round(self.progress[s], 3),
                        finish_time=None if self.finished[s] is None else round(self.finished[s], 4),
                        obstacle_crashes=sum(1 for c in self.heat_crashes if c['at_fault'] == s and c['fault_rule'] == 'obstacle'))
                for s in self.seats}

    def heat_order(self):
        """The finishing order of the running cars: finishers by (sub-step) finish time, then the others by progress,
        then who reached it first; a fixed draw from the seed separates exact ties. Each car's 'how' names the rule
        that placed it against its neighbour."""
        act = self.active(); fin = [s for s in act if self.finished[s] is not None]; rest = [s for s in act if s not in fin]
        fin.sort(key=lambda s: (self.finished[s], lot_key(self.seed, self.heat, 'finish', s)))
        rest.sort(key=lambda s: (-round(self.progress[s], 3), self.reach_t[s], lot_key(self.seed, self.heat, 'progress', s)))
        order = fin + rest

        def rule(x, y):
            if x is None or y is None: return None
            if self.finished[x] is not None and self.finished[y] is not None:
                return 'finish' if self.finished[x] != self.finished[y] else 'lot'
            if (self.finished[x] is None) != (self.finished[y] is None): return 'finish'
            if round(self.progress[x], 3) != round(self.progress[y], 3): return 'progress'
            return 'reached_first' if self.reach_t[x] != self.reach_t[y] else 'lot'
        how = {}
        for i, s in enumerate(order):
            nxt = order[i + 1] if i + 1 < len(order) else None; prv = order[i - 1] if i else None
            how[s] = rule(s, nxt) or rule(prv, s) or 'finish'
        return order, how

    def check_heat_end(self):
        t = self.heat_k * DT
        if not self.multi:
            done = [s for s in self.seats if self.finished[s] is not None]
            if done:
                # RoboRacer Head-to-Head evaluation: the first car to complete the laps wins the heat (sub-step time).
                if len(done) == 2 and self.finished['A'] == self.finished['B']:
                    self.end_heat(lot(self.seed, self.heat, 'finish'), 'lot')
                else:
                    self.end_heat(min(done, key=lambda s: self.finished[s]), 'finish')
            elif t >= self.heat_seconds - 1e-9:
                # Nobody finished: completed laps and progress on the track decide; then who reached it first; then a lot.
                p = {s: round(self.progress[s], 3) for s in self.seats}
                if p['A'] != p['B']: self.end_heat(max(p, key=p.get), 'progress')
                elif self.reach_t['A'] != self.reach_t['B']: self.end_heat(min(self.reach_t, key=self.reach_t.get), 'reached_first')
                else: self.end_heat(lot(self.seed, self.heat, 'progress'), 'lot')
            return
        act = self.active(); done = [s for s in act if self.finished[s] is not None]
        first = min((self.finished[s] for s in done), default=None)
        # Multi-car: the heat ends when its order is decided (all running cars but one have finished), FINISH_WINDOW_S
        # after the first finisher, or at the time limit.
        if (done and len(done) >= len(act) - 1) or (first is not None and t >= first + FINISH_WINDOW_S - 1e-9) or \
                t >= self.heat_seconds - 1e-9:
            self.end_heat(None, None)

    def end_heat(self, winner, how, race_winner=None):
        res = dict(heat=self.heat + 1, heat_time=round(self.heat_k * DT, 4), cars=self.heat_summary(),
                   grid=dict(self.grid), crashes=list(self.heat_crashes), steps=self.heat_k)
        if self.multi:
            order, rule = self.heat_order()
            for i, s in enumerate(order): self.points[s] += POINTS[i] if i < len(POINTS) else 0
            winner, how = order[0], rule[order[0]]
            res.update(order=order, decided_by_car=rule, retired=sorted(self.retired),
                       positions={s: i + 1 for i, s in enumerate(order)},
                       heat_points={s: (POINTS[i] if i < len(POINTS) else 0) for i, s in enumerate(order)})
        res.update(winner=winner, decided_by=how,
                   finish_time=None if self.finished[winner] is None else round(self.finished[winner], 4))
        if how != 'disqualification': self.wins[winner] += 1
        res['heats_won'] = dict(self.wins)
        if self.multi: res['points'] = dict(self.points)
        self.results.append(res)
        self.events.append(dict(ev='heat_end', result=res))
        if self.multi:
            if len(self.results) >= self.max_heats or len(self.active()) <= 1: self.finish_race('heats')
            else: self.awaiting_review = True
            return
        if race_winner is None and (self.wins[winner] >= HEATS_TO_WIN or len(self.results) >= MAX_HEATS):
            race_winner = winner
        if race_winner is not None:
            self.done, self.winner_seat = True, race_winner
            self.end_reason = 'disqualification' if how == 'disqualification' else 'heats'
        else:
            self.awaiting_review = True

    def classification(self):
        """Multi-car final order: points; then count-back (more first places, then more second places, ...; FIA F1
        Sporting Regulations); then the better place in the last heat; retired cars after all running cars (the later
        retirement first, then points, then the seed's draw)."""
        if not self.multi: return None
        pos = {s: [r['positions'].get(s) for r in self.results] for s in self.seats}

        def key(s):
            last = next((p for p in reversed(pos[s]) if p is not None), 99)
            return (-self.points[s], *[-pos[s].count(i) for i in range(1, self.cars + 1)], last)
        run = sorted(self.active(), key=key)
        ret = sorted(self.retired, key=lambda s: (-self.retired[s]['step'], -self.points[s], lot_key(self.seed, 0, 'retired', s)))
        return run + ret

    def finish_race(self, reason):
        self.done, self.awaiting_review = True, False
        self.end_reason = reason
        order = self.classification()
        self.winner_seat = order[0] if (order and order[0] not in self.retired) else None

    def ready(self):
        if not self.awaiting_review: raise ValueError('not in a review')
        self.heat += 1; self.awaiting_review = False; self.reset_heat()

    def winner(self): return self.winner_seat

    def scores(self):
        out = {s: dict(heats_won=self.wins[s], warnings=self.warnings[s],
                       checkpoints=sum(r['cars'][s]['checkpoints'] for r in self.results),
                       laps=sum(r['cars'][s]['finish_time'] is not None for r in self.results),
                       obstacle_crashes=self.obstacle_crashes[s]) for s in self.seats}
        if self.multi:
            order = self.classification()
            for s in self.seats:
                out[s].update(points=self.points[s], position=order.index(s) + 1, retired=int(s in self.retired))
        return out

    def summary(self):
        """What the live service records and the fresh-process replay must reproduce."""
        return dict(hash=self.digest(), results=self.results, scores=self.scores(), winner=self.winner(),
                    end_reason=self.end_reason, crashes=self.crashes, steps=self.k, done=self.done,
                    awaiting_review=self.awaiting_review, wall_steps=self.wall_steps, car_contact_steps=self.contact_steps,
                    obstacle_contact_steps=self.obstacle_steps, classification=self.classification(),
                    retired=self.retired)

    def digest(self):
        h = hashlib.sha256()
        state = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        h.update(state.tobytes())
        h.update(json.dumps(dict(k=self.k, heat=self.heat, heat_k=self.heat_k, results=self.results, commands=self.commands,
                                 gates=self.next_gate, finished=self.finished, progress=self.progress, wins=self.wins,
                                 warnings=self.warnings, crashes=self.crashes, walls=self.wall_steps,
                                 contact_steps=self.contact_steps, points=self.points, hold=self.hold,
                                 retired=self.retired, obstacle_steps=self.obstacle_steps), sort_keys=True).encode())
        return h.hexdigest()
