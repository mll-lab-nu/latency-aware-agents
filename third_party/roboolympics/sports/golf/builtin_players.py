"""Mini-golf putting's built-in players (the same action interface as a model's program; core/client.py plays them
over HTTP).

  Reference  camera-only (the PNG of the observation and /spec; no simulator import). In the overhead view (a pinhole
             straight down; pixels -> green coordinates with the camera of /spec) it finds the ball (white), the cup
             (black disc) and the obstacle blocks (red), and reads the surface heights from the white grid lines: a
             line at a known x (or y) is drawn at u = cx + f (x - L/2) / (H - z), so where it is drawn gives the
             height z of the surface under it. It fits a smooth height field to those samples (a 0.25 m grid of
             heights with a curvature penalty), then plans each putt with its own model of the published physics
             ((5/7) g slope + the constant rolling resistance of the stimpmeter, the wall rebound, the cup's capture
             speed): a coarse-to-fine search over aim and speed for the putt with the fewest expected strokes under
             the published scatter (hole it, or leave it close). Proves the sport is playable from the images alone
             (sport rule 9).
  Random     uniform random valid putts (generic tests); seeded by the seed only (mirror runs match).
No simulator import: this file runs in the player's container as the task's oracle (numpy + pillow only)."""

from __future__ import annotations

import base64
import io
import json
import math

import numpy as np

GRAV = 9.81
K_ROLL = 5.0 / 7.0  # rolling without slipping: a = -(5/7) g grad h ([spec] ball_physics)
GRID = 0.5  # the white grid lines, every 0.5 m ([spec] image_layout)
KNOT = 0.5  # the height field's spline knots (m), on the grid lines
FINE = 0.05  # the lookup grid of heights and slopes (m)
LAM = 0.3  # the spline's curvature penalty
_GREENS: dict = {}  # green readings by picture (a cache: reading takes about 0.3 s)
WALL_COST = 0.25  # a bank shot is less certain than the model says: strokes added per wall hit


def number_after(text, word, default):
    """The first number after `word` in a /spec text (the published constants), else default."""
    i = (text or "").find(word)
    if i < 0:
        return default
    for tok in text[i + len(word) :].replace(",", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return default


# ------------------------------------------------------------------------------------------------ image helpers
def blobs(mask, min_px=3):
    """Blobs (u, v, n, u0, u1, v0, v1) of the 4-connected components of a boolean mask (pixel centres at i + 0.5,
    bounding box in pixels); row runs joined by union-find."""
    runs, parent = [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prev: list[int] = []
    for y in np.nonzero(mask.any(axis=1))[0]:
        row = np.concatenate(([0], mask[y].astype(np.int8), [0]))
        d = np.diff(row)
        cur = []
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            i = len(runs)
            runs.append((int(y), int(a), int(b)))
            parent.append(i)
            for j in prev:
                py, pa, pb = runs[j]
                if py == y - 1 and pa < b and a < pb:
                    parent[find(i)] = find(j)
            cur.append(i)
        prev = cur
    acc: dict[int, list[float]] = {}
    for i, (y, a, b) in enumerate(runs):
        n = b - a
        s = acc.setdefault(find(i), [0.0, 0.0, 0.0, 1e9, -1e9, 1e9, -1e9])
        s[0] += n * ((a + b - 1) / 2 + 0.5)
        s[1] += n * (y + 0.5)
        s[2] += n
        s[3], s[4], s[5], s[6] = min(s[3], a), max(s[4], b), min(s[5], y), max(s[6], y + 1)
    return [(su / n, sv / n, int(n), u0, u1, v0, v1) for su, sv, n, u0, u1, v0, v1 in acc.values() if n >= min_px]


def tiles(view, names=("overhead",)):
    """The named camera views of the observation's PNG (int16 RGB arrays)."""
    from PIL import Image

    obs = view["observation"]
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["image_png_b64"]))).convert("RGB")).astype(np.int16)
    out = []
    for name in names:
        t = next(t for t in obs["tiles"] if t["camera"] == name)
        out.append(img[t["y"] : t["y"] + t["height"], t["x"] : t["x"] + t["width"]])
    return out


def grey_mask(img, lo=90):
    """White things in shade are grey (r = g = b); the green, its grid lines and shadows are green."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (np.abs(r - g) <= 10) & (np.abs(b - g) <= 14) & (g > lo)


def line_centres(strips, thresh=60.0, max_width=6):
    """Sub-pixel centre of the one bright line in each row of `strips` (n x m), NaN where there is none or it is not
    a thin line. The background is taken on each side separately: a line at an integer x is also the edge of a
    mowing stripe."""
    n, m = strips.shape
    i = np.argmax(strips, axis=1)
    ok = (i >= 11) & (i <= m - 12)
    i = np.clip(i, 11, m - 12)
    rows = np.arange(n)[:, None]
    left = np.median(strips[rows, i[:, None] + np.arange(-11, -4)], axis=1)
    right = np.median(strips[rows, i[:, None] + np.arange(5, 12)], axis=1)
    base = np.maximum(left, right)
    win = strips[rows, i[:, None] + np.arange(-11, 12)]
    ok &= (strips[np.arange(n), i] >= base + thresh) & ((win > (base + thresh)[:, None]).sum(1) <= max_width)
    k = np.arange(-3, 4)
    bg = np.where(k < 0, left[:, None], np.where(k > 0, right[:, None], 0.5 * (left + right)[:, None]))
    seg = strips[rows, i[:, None] + k] - bg
    seg = np.where(seg > 0, seg, 0.0)
    c = i + (seg * k).sum(1) / np.maximum(seg.sum(1), 1e-9) + 0.5
    return np.where(ok, c, np.nan)


# ------------------------------------------------------------------------------------------------ one green
class Green:
    """One hole as the reference sees it: the overhead camera's geometry (from /spec), the height field read from the
    grid lines, the cup and the blocks."""

    def __init__(self, hole, cam, r_ball, r_cup):
        self.L = float(hole["green_x_m"][1])
        self.W = float(hole["green_y_m"][1] - hole["green_y_m"][0])
        self.Hc = float(hole["overhead_camera_height_m"])
        self.f = float(cam["focal_px"])
        self.cx, self.cy = (float(c) for c in cam["principal_point_px"])
        self.nblocks = int(hole.get("obstacle_blocks", 0))
        self.rb, self.rc = r_ball, r_cup
        self.ncx = int(round(self.L / KNOT)) + 3  # cubic B-spline control points (one beyond each edge)
        self.ncy = int(round(self.W / KNOT)) + 3
        self.coef = np.zeros(self.ncx * self.ncy)
        self.fine = None  # (h, gx, gy) on a FINE grid, for the planner
        self.cup = None
        self.blocks: list[tuple[float, float, float, float]] = []

    # projection
    def to_green(self, u, v, z):
        d = self.Hc - z
        return self.L / 2 + (u - self.cx) * d / self.f, -(v - self.cy) * d / self.f

    def to_pixel(self, x, y, z):
        d = self.Hc - z
        return self.cx + self.f * (x - self.L / 2) / d, self.cy - self.f * y / d

    def locate(self, u, v, dz=0.0):
        """Pixel -> green point on the surface (+ dz), iterating the height."""
        x, y = self.to_green(u, v, dz)
        for _ in range(3):
            x, y = self.to_green(u, v, self.h(x, y) + dz)
        return x, y

    # height field: a cubic B-spline with knots every KNOT m (aligned with the grid lines)
    @staticmethod
    def _bspline(t):
        u = t - np.floor(t)
        w = [(1 - u) ** 3 / 6, (3 * u**3 - 6 * u**2 + 4) / 6, (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6, u**3 / 6]
        dw = [-((1 - u) ** 2) / 2, (3 * u**2 - 4 * u) / 2, (-3 * u**2 + 2 * u + 1) / 2, u**2 / 2]
        return np.floor(t).astype(int), w, dw

    def design(self, x, y, deriv=None):
        """Rows of the spline's basis at the points (x, y): values, or d/dx (deriv = 'x') or d/dy ('y')."""
        tx = np.clip(np.asarray(x, float), 0, self.L - 1e-9) / KNOT
        ty = (np.clip(np.asarray(y, float), -self.W / 2, self.W / 2 - 1e-9) + self.W / 2) / KNOT
        ix, wx, dwx = self._bspline(tx)
        iy, wy, dwy = self._bspline(ty)
        if deriv == "x":
            wx = [q / KNOT for q in dwx]
        elif deriv == "y":
            wy = [q / KNOT for q in dwy]
        A = np.zeros((len(tx), self.ncx * self.ncy))
        rows = np.arange(len(tx))
        for p in range(4):
            for q in range(4):
                np.add.at(A, (rows, (ix + p) * self.ncy + iy + q), wx[p] * wy[q])
        return A

    def fit(self, x, y, z, w, lam=None):
        """Weighted least squares of the spline's coefficients to height samples, a small second-difference
        penalty (fills the gaps: under the ball, the cup, a block), robust: refits without the outliers."""
        lam = LAM if lam is None else lam
        nx, ny = self.ncx, self.ncy
        D = []
        for i in range(nx):
            for j in range(ny):
                for di, dj in ((1, 0), (0, 1)):
                    if 0 <= i - di and i + di < nx and 0 <= j - dj and j + dj < ny:
                        r = np.zeros(nx * ny)
                        r[(i - di) * ny + j - dj] += 1
                        r[i * ny + j] -= 2
                        r[(i + di) * ny + j + dj] += 1
                        D.append(r)
        D = np.array(D)
        P = lam * D.T @ D
        A = self.design(x, y)
        keep = np.ones(len(x), bool)
        for tol in (0.02, 0.006, 0.004):
            Aw = A[keep] * w[keep, None]
            self.coef = np.linalg.solve(Aw.T @ A[keep] + P + 1e-9 * np.eye(nx * ny), Aw.T @ z[keep])
            keep = np.abs(A @ self.coef - z) < tol
        xs = np.arange(0.0, self.L + 1e-9, FINE)
        ys = np.arange(-self.W / 2, self.W / 2 + 1e-9, FINE)
        X, Y = (q.ravel() for q in np.meshgrid(xs, ys, indexing="ij"))
        self.fine = tuple((self.design(X, Y, d) @ self.coef).reshape(len(xs), len(ys)) for d in (None, "x", "y"))
        self.slopes = np.stack(self.fine[1:], -1)
        return self

    def _lookup(self, G, x, y):
        gx = np.clip(np.asarray(x, float) * (1.0 / FINE), 0, G.shape[0] - 1 - 1e-9)
        gy = np.clip((np.asarray(y, float) + self.W / 2) * (1.0 / FINE), 0, G.shape[1] - 1 - 1e-9)
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        if G.ndim == 3:
            fx, fy = fx[..., None], fy[..., None]
        return (
            (1 - fx) * (1 - fy) * G[ix, iy]
            + fx * (1 - fy) * G[ix + 1, iy]
            + (1 - fx) * fy * G[ix, iy + 1]
            + fx * fy * G[ix + 1, iy + 1]
        )

    def h(self, x, y):
        return self._lookup(self.fine[0], x, y) if self.fine is not None else 0.0 * np.asarray(x, float)

    def grad(self, x, y):
        g = self._lookup(self.slopes, x, y)
        return g[..., 0], g[..., 1]

    # reading the overhead image
    def read(self, img):
        """Cup, blocks and the height samples of the grid lines (the image of the hole's first decision point: no
        path drawn yet)."""
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        dark = (r < 45) & (g < 45) & (b < 45)
        area = math.pi * (self.f * self.rc / self.Hc) ** 2  # the cup's black disc, in pixels
        u_lo, v_hi = self.to_pixel(0.0, -self.W / 2, 0.0)
        u_hi, v_lo = self.to_pixel(self.L, self.W / 2, 0.0)
        cups = [
            x
            for x in blobs(dark, 20)
            if 0.4 * area < x[2] < 2.5 * area
            and u_lo < x[0] < u_hi
            and v_lo < x[1] < v_hi
            and 0.6 < (x[4] - x[3]) / max(1, x[6] - x[5]) < 1.7
        ]
        cu, cv = min(cups, key=lambda x: abs(x[2] - area))[:2] if cups else (self.cx, self.cy)
        red = (r > 120) & (g < 90) & (b < 90) & (r - g > 70)
        self.blocks = []
        for u, v, n, u0, u1, v0, v1 in blobs(red, 200):
            if n < 1500:
                continue  # the flag
            xa, ya = self.to_green(u0, v1, 0.12)
            xb, yb = self.to_green(u1, v0, 0.12)
            self.blocks.append((min(xa, xb) - 0.01, max(xa, xb) + 0.01, min(ya, yb) - 0.01, max(ya, yb) + 0.01))
        wimg = np.minimum(r, b).astype(float)
        X, Y, Z, Wt = [], [], [], []
        f, Hc = self.f, self.Hc
        for k in range(1, int(round(self.L / GRID))):  # lines at x = 0.5 k: rows across them
            off = GRID * k - self.L / 2
            if abs(off) < 0.3:
                continue
            a = int(self.cx + f * off / Hc - 30)
            vs = np.arange(int(self.cy - f * (self.W / 2 - 0.08) / Hc), int(self.cy + f * (self.W / 2 - 0.08) / Hc))
            c = line_centres(wimg[vs, a : a + 61])
            z = Hc - f * off / (a + c - self.cx)
            X.append(np.full(len(vs), GRID * k))
            Y.append(-(vs + 0.5 - self.cy) * (Hc - z) / f)
            Z.append(z)
            Wt.append(np.full(len(vs), abs(off)))
        for j in range(1, int(round(self.W / GRID))):  # lines at y = -W/2 + 0.5 j: columns across them
            yj = -self.W / 2 + GRID * j
            if abs(yj) < 0.2:
                continue
            a = int(self.cy - f * yj / Hc - 30)
            us = np.arange(int(self.cx + f * (0.08 - self.L / 2) / Hc), int(self.cx + f * (self.L / 2 - 0.08) / Hc))
            c = line_centres(wimg[a : a + 61, us].T)
            z = Hc + f * yj / (a + c - self.cy)
            X.append(self.L / 2 + (us + 0.5 - self.cx) * (Hc - z) / f)
            Y.append(np.full(len(us), yj))
            Z.append(z)
            Wt.append(np.full(len(us), abs(yj)))
        X, Y, Z, Wt = (np.concatenate(q) for q in (X, Y, Z, Wt))
        good = np.isfinite(Z)
        X, Y, Z, Wt = X[good], Y[good], Z[good], Wt[good]
        ok = (X > 0.05) & (X < self.L - 0.05) & (np.abs(Y) < self.W / 2 - 0.05) & (np.abs(Z) < 0.5)
        for x0, x1, y0, y1 in self.blocks:
            ok &= ~((X > x0 - 0.1) & (X < x1 + 0.1) & (Y > y0 - 0.1) & (Y < y1 + 0.1))
        X, Y, Z, Wt = X[ok], Y[ok], Z[ok], Wt[ok]
        # weight = 1 / variance: the height error per pixel is (H - z)^2 / (f * offset)
        sig = (Hc**2) * 0.03 / (f * np.maximum(Wt, 0.05)) + 0.0005
        self.fit(X, Y, Z, (0.001 / sig) ** 2)
        self.cup = self.locate(cu, cv)
        return self

    def ball(self, img, start=None, guess=None):
        """The ball: a white disc; after a stroke on this hole it lies under the last yellow dot of the path drawn
        over it (a dot every 0.1 s; the last ones overlap as the ball slows down), so then the end of that path."""
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        hc = float(self.h(*self.cup))
        cu, cv = self.to_pixel(*self.cup, hc)
        ring = self.f * (self.rc + 0.006) / (self.Hc - hc)
        rpx = self.f * self.rb / self.Hc  # the ball's radius in pixels
        vv, uu = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
        dr = np.hypot(uu + 0.5 - cu, vv + 0.5 - cv)
        grey = grey_mask(img) & ~((dr > ring - 2.0) & (dr < ring + 2.0))
        bl = [
            x
            for x in blobs(grey, 4)
            if 0.35 * math.pi * rpx**2 < x[2] < 4 * math.pi * rpx**2 and x[4] - x[3] <= 12 and x[6] - x[5] <= 12
        ]
        yellow = (r > 220) & (g > 180) & (g < 245) & (b < 90)
        tu, tv = self.to_pixel(*self.cup, hc + 1.5)  # the flagstick seen from above: cup -> its top
        ex, ey = tu - cu, tv - cv
        s = np.clip(((uu + 0.5 - cu) * ex + (vv + 0.5 - cv) * ey) / max(ex * ex + ey * ey, 1e-9), 0, 1)
        yellow &= np.hypot(uu + 0.5 - cu - s * ex, vv + 0.5 - cv - s * ey) > 3.0
        dots = blobs(yellow, 8)
        if not dots or start is None:
            if not bl:
                return None
            u, v = max(bl, key=lambda x: x[2])[:2]
            return self.locate(u, v, self.rb)
        # the end of the path: the ball slows down to rest, so its last dots overlap into the biggest yellow blob;
        # the ball lies under the last dot, at the far end of that blob (seen from the dots before it)
        su, sv = self.to_pixel(*start, float(self.h(*start)) + self.rb)
        dot = math.pi * 3.5**2
        big = [x for x in dots if x[2] >= 1.6 * dot]
        if guess is not None and len(big) > 1:  # several slow stretches: the one nearest my guess
            gu, gv = self.to_pixel(*guess, float(self.h(*guess)) + self.rb)
            last = min(big, key=lambda x: math.hypot(x[0] - gu, x[1] - gv))
        else:
            last = max(dots, key=lambda x: x[2])
        others = [x for x in dots if x is not last]
        if last[2] < 1.6 * dot:  # no overlap: the dot farthest from the start
            last = max(dots, key=lambda x: math.hypot(x[0] - su, x[1] - sv))
            u, v = last[0], last[1]
        else:
            if others:
                near = min(others, key=lambda x: math.hypot(x[0] - last[0], x[1] - last[1]))
                pu0, pv0 = near[0], near[1]
            else:
                pu0, pv0 = su, sv
            du, dv = last[0] - pu0, last[1] - pv0
            nrm = math.hypot(du, dv) or 1.0
            du, dv = du / nrm, dv / nrm
            ys, xs = np.nonzero(yellow[last[5] : last[6], last[3] : last[4]])
            pu, pv = xs + last[3] + 0.5, ys + last[5] + 0.5
            proj = pu * du + pv * dv
            far = proj >= proj.max() - 0.75
            u, v = float(pu[far].mean()) - 3.0 * du, float(pv[far].mean()) - 3.0 * dv
        return self.locate(u, v, self.rb)


# ------------------------------------------------------------------------------------------------ the physics model
class Model:
    """The reference's own model of the published physics, vectorised over candidate putts."""

    def __init__(self, green: Green, c_roll, e_n, e_t):
        self.g, self.C, self.en, self.et = green, c_roll, e_n, e_t

    def vmax(self, b):
        """Capture speed over the cup for a ball whose path passes at distance b from the centre: it must drop
        0.6 ball radii (the published 1.7 m/s straight at the centre) while crossing to the far wall."""
        rc, rb = self.g.rc, self.g.rb
        chord = np.sqrt(np.maximum(rc * rc - b * b, 0)) + np.sqrt(np.maximum((rc - rb) ** 2 - b * b, 0))
        return np.where(b < rc - rb, chord * math.sqrt(GRAV / (1.2 * rb)), 0.0)

    def run(self, pos, aim_deg, speed, dt=0.005, tmax=30.0, margin=0.85):
        """Simulate putts from pos -> (final x, final y, holed, closest distance to the cup, wall/block hits)."""
        gr = self.g
        n = len(aim_deg)
        a = np.radians(aim_deg)
        x, y = np.full(n, float(pos[0])), np.full(n, float(pos[1]))
        vx, vy = speed * np.cos(a), speed * np.sin(a)
        holed = np.zeros(n, bool)
        hits = np.zeros(n)
        dmin = np.full(n, 1e9)
        live = np.arange(n)
        cx, cy = gr.cup
        lo_x, hi_x, lo_y, hi_y = gr.rb, gr.L - gr.rb, -gr.W / 2 + gr.rb, gr.W / 2 - gr.rb
        fx, fy = x.copy(), y.copy()
        over = np.zeros(n, bool)
        t = 0.0
        while len(live) and t < tmax:
            X, Y, VX, VY = x[live], y[live], vx[live], vy[live]
            gx, gy = gr.grad(X, Y)
            ax, ay = -K_ROLL * GRAV * gx, -K_ROLL * GRAV * gy
            v = np.hypot(VX, VY)
            ga = np.hypot(ax, ay)
            slow = v < 1e-3
            stop = slow & (ga <= self.C)
            vs = np.maximum(v, 1e-9)
            nvx = np.where(slow, VX + ax * dt, VX + (ax - self.C * VX / vs) * dt)
            nvy = np.where(slow, VY + ay * dt, VY + (ay - self.C * VY / vs) * dt)
            stop |= (~slow) & (nvx * VX + nvy * VY <= 0) & (ga <= self.C)
            VX, VY = np.where(stop, 0.0, nvx), np.where(stop, 0.0, nvy)
            X, Y = X + VX * dt, Y + VY * dt
            # walls
            for bound, comp, sgn in ((lo_x, 0, 1), (hi_x, 0, -1), (lo_y, 1, 1), (hi_y, 1, -1)):
                P, V, T = (X, VX, VY) if comp == 0 else (Y, VY, VX)
                hit = (P - bound) * sgn < 0
                if hit.any():
                    P = np.where(hit, bound, P)
                    into = hit & (V * sgn < 0)
                    hits[live[into]] += 1
                    V = np.where(into, -self.en * V, V)
                    T = np.where(into, self.et * T, T)
                    if comp == 0:
                        X, VX, VY = P, V, T
                    else:
                        Y, VY, VX = P, V, T
            for x0, x1, y0, y1 in gr.blocks:
                ex0, ex1, ey0, ey1 = x0 - gr.rb, x1 + gr.rb, y0 - gr.rb, y1 + gr.rb
                ins = (X > ex0) & (X < ex1) & (Y > ey0) & (Y < ey1)
                if ins.any():
                    pen = np.stack([X - ex0, ex1 - X, Y - ey0, ey1 - Y])
                    side = np.argmin(pen, 0)
                    for s, (comp, val, sgn) in enumerate(((0, ex0, -1), (0, ex1, 1), (1, ey0, -1), (1, ey1, 1))):
                        m = ins & (side == s)
                        if not m.any():
                            continue
                        if comp == 0:
                            X = np.where(m, val, X)
                            into = m & (VX * sgn < 0)
                            hits[live[into]] += 1
                            VX, VY = np.where(into, -self.en * VX, VX), np.where(into, self.et * VY, VY)
                        else:
                            Y = np.where(m, val, Y)
                            into = m & (VY * sgn < 0)
                            hits[live[into]] += 1
                            VY, VX = np.where(into, -self.en * VY, VY), np.where(into, self.et * VX, VX)
            # the cup
            d = np.hypot(X - cx, Y - cy)
            dmin[live] = np.minimum(dmin[live], d)
            enter = (d < gr.rc) & ~over[live]
            if enter.any():
                vv = np.maximum(np.hypot(VX, VY), 1e-9)
                bb = np.abs((X - cx) * VY - (Y - cy) * VX) / vv
                cap = enter & (vv < margin * self.vmax(bb))
                holed[live[cap]] = True
                stop = stop | cap
            over[live] = d < gr.rc
            x[live], y[live], vx[live], vy[live] = X, Y, VX, VY
            fx[live], fy[live] = X, Y
            live = live[~stop]
            t += dt
        return fx, fy, holed, dmin, hits


def blur(a, s0, s1):
    """Gaussian blur of a 2D array with sigmas (cells) along both axes (edges: renormalised)."""
    out = a.astype(float)
    for axis, s in ((0, s0), (1, s1)):
        r = max(1, int(math.ceil(2.5 * s)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(s, 1e-6)) ** 2)
        num = np.apply_along_axis(lambda q: np.convolve(q, k, mode="same"), axis, out)
        den = np.apply_along_axis(lambda q: np.convolve(q, k, mode="same"), axis, np.ones_like(out))
        out = num / den
    return out


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec = spec
        m = spec["match"]
        self.holes = m["holes"]
        self.cam = spec["cameras"]["overhead"]
        bg = spec.get("ball_and_green", {})
        self.rb = float(bg.get("ball_radius_m", 0.02135))
        self.rc = float(bg.get("cup_radius_m", 0.054))
        txt = bg.get("green_speed", "")
        v0, d0 = number_after(txt, "starting at", 1.83), number_after(txt, "rolls", 3.05)
        self.c_roll = v0 * v0 / (2 * d0)
        wall = bg.get("walls", "")
        self.e_n, self.e_t = number_after(wall, "about", 0.7), number_after(wall, "wall,", 0.92)
        sc = spec.get("execution_scatter_1sigma") or {}
        self.sig_aim = float(sc.get("aim_deg", 0.3))
        self.sig_speed = float(sc.get("speed_rel", 0.015))
        self.greens: dict[int, Green] = {}
        self.last: dict = {}  # (hole, play-off hole) -> where my ball was
        self.guess: dict = {}  # ... and where my model expects it now
        self.seed, self.n = int(seed), 0

    def green(self, hole, img):
        if hole not in self.greens:
            key = (hole, json.dumps(self.holes[hole - 1], sort_keys=True), hash(img.tobytes()))
            if key not in _GREENS:  # the same picture gives the same reading (both seats)
                if len(_GREENS) > 16:
                    _GREENS.pop(next(iter(_GREENS)))
                _GREENS[key] = Green(self.holes[hole - 1], self.cam, self.rb, self.rc).read(img)
            self.greens[hole] = _GREENS[key]
        return self.greens[hole]

    def strokes_to_go(self, d):
        return 1.0 + np.clip((d - 0.4) / 3.0, 0.0, 1.5)

    def plan(self, gr, pos):
        model = Model(gr, self.c_roll, self.e_n, self.e_t)
        a0 = math.degrees(math.atan2(gr.cup[1] - pos[1], gr.cup[0] - pos[0]))
        dist = math.hypot(gr.cup[0] - pos[0], gr.cup[1] - pos[1])
        # coarse: a wide fan of aims and speeds (bank shots and routes around a block included)
        aims = a0 + np.arange(-75.0, 75.01, 1.5)
        speeds = np.geomspace(0.25, 5.5, 28)
        A, V = np.meshgrid(aims, speeds, indexing="ij")
        fx, fy, holed, dmin, hits = model.run(pos, A.ravel(), V.ravel(), dt=0.01)
        df = np.hypot(fx - gr.cup[0], fy - gr.cup[1])
        val = (np.where(holed, 0.0, self.strokes_to_go(df)) + WALL_COST * hits).reshape(A.shape)
        near = np.minimum(dmin, df).reshape(A.shape)
        score = blur(val, 1.0, 0.7) + 0.05 * near
        picks = []
        for i in np.argsort(score.ravel()):
            ia, iv = np.unravel_index(i, A.shape)
            if all(abs(ia - pa) > 3 or abs(iv - pv) > 3 for pa, pv in picks):
                picks.append((ia, iv))
            if len(picks) == 2:
                break
        if dist < 3.0:  # a short putt: always look at the direct line too
            iv = int(np.argmin(np.abs(speeds - math.sqrt(2 * self.c_roll * (dist + 0.4)))))
            picks.append((len(aims) // 2, iv))
        best = None
        sa, ss = 0.1, 0.006  # fine grid steps: deg, relative speed
        for ia, iv in picks:
            fa = aims[ia] + np.arange(-1.5, 1.501, sa)
            fs = speeds[iv] * np.exp(np.arange(-0.09, 0.0901, ss))
            A2, V2 = np.meshgrid(fa, fs, indexing="ij")
            fx, fy, holed, _, hits = model.run(pos, A2.ravel(), V2.ravel(), dt=0.005)
            df = np.hypot(fx - gr.cup[0], fy - gr.cup[1])
            val = (np.where(holed, 0.0, self.strokes_to_go(df)) + WALL_COST * hits).reshape(A2.shape)
            # expected strokes still to play after this one, under the scatter (inflated for the model's error)
            exp = blur(val, 1.4 * self.sig_aim / sa, 1.6 * self.sig_speed / ss)
            j = int(np.argmin(exp))
            ja, jv = np.unravel_index(j, A2.shape)
            if best is None or exp[ja, jv] < best[0]:
                best = (float(exp[ja, jv]), float(A2[ja, jv]), float(V2[ja, jv]))
        return best

    def act(self, view):
        st = view["state"]
        dec = st.get("decision") or {}
        hole = int(dec.get("hole", 1))
        (img,) = tiles(view)
        gr = self.green(hole, img)
        key = (hole, dec.get("playoff_hole"))
        pos = gr.ball(img, self.last.get(key), self.guess.get(key))
        if pos is None:
            pos = self.guess.get(key) or self.last.get(key) or tuple(gr.cup)
        self.last[key] = pos
        _, aim, speed = self.plan(gr, pos)
        self.n += 1
        aim = (aim + 180.0) % 360.0 - 180.0
        speed = round(float(np.clip(speed, 0.05, 6.0)), 4)
        fx, fy, _, _, _ = Model(gr, self.c_roll, self.e_n, self.e_t).run(pos, np.array([aim]), np.array([speed]))
        self.guess[key] = (float(fx[0]), float(fy[0]))  # where I expect the ball (to find it under its path)
        return {"type": "putt", "aim_deg": round(aim, 3), "speed": speed}

    def review(self, reply):
        return (
            "read the green's heights from the grid lines and planned the putt with my rolling model",
            "keep planning each putt for the fewest expected strokes",
        )


class Random:
    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 31)  # by seed, not by seat: mirror runs match

    def act(self, view):
        return {"type": "putt", "aim_deg": float(self.g.uniform(-40, 40)), "speed": float(self.g.uniform(0.5, 3.5))}
