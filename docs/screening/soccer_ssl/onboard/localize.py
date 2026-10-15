"""Self-localisation prototype of the onboard mode: a particle filter per robot (Monte Carlo localisation, Thrun et
al., Probabilistic Robotics ch. 8; the "visual sonar" idea of Lenser & Veloso 2003 for the floor boundary).

State: the robot's world pose (x, y, yaw). Start: the kick-off spot the robot is told (Gaussian spread). Motion: the
odometry increment between two pictures (sensors.Odometry) plus noise. Measurements per picture:
  1. floor boundary per sampled column (perceive.floor_boundary): the depression angle of the bottom of the first
     thing on the carpet, against the angle the map predicts along the same bearing (walls 0.1 m, goals, both ends
     identical). A robot in between gives a nearer boundary: a beam-model mixture (hit / short / random). The kind
     (goal foam white, goal frame tall, wall) adds a categorical term;
  2. white line points within 2.5 m (perceive.line_points): distance to the nearest field line (a precomputed grid).
The field is point symmetric: a pose and its mirror (-x, -y, yaw + pi) see the same picture. The filter only keeps
the right one because it starts from the told spot and never re-initialises globally."""
from __future__ import annotations

import math

import numpy as np

HALF_L, HALF_W = 4.5, 3.0
WALL_X, WALL_Y = 4.8, 3.3
GOAL_X0, GOAL_D, GOAL_W, GOAL_T = 4.505, 0.18, 1.0, 0.02
CENTER_R = 0.5
DA_D, DA_W = 1.0, 2.0

# kinds: 0 wall (low), 1 goal frame (tall black), 2 goal foam (tall white)
def map_segments():
    segs = []
    for s in (1, -1):
        segs.append(((s * WALL_X, -WALL_Y), (s * WALL_X, WALL_Y), 0))
        segs.append(((-WALL_X, s * WALL_Y), (WALL_X, s * WALL_Y), 0))
        yo = GOAL_W / 2 + GOAL_T
        xb = GOAL_X0 + GOAL_D
        for ys in (1, -1):
            segs.append(((s * GOAL_X0, ys * yo), (s * WALL_X, ys * yo), 1))          # side wall, outer face
            segs.append(((s * GOAL_X0, ys * GOAL_W / 2), (s * GOAL_X0, ys * yo), 1))  # its front end
            segs.append(((s * GOAL_X0, ys * GOAL_W / 2), (s * xb, ys * GOAL_W / 2), 2))  # foam, inner side
        segs.append(((s * xb, -GOAL_W / 2), (s * xb, GOAL_W / 2), 2))                # foam, back
    A = np.array([s[0] for s in segs], float)
    B = np.array([s[1] for s in segs], float)
    K = np.array([s[2] for s in segs], int)
    return A, B, K


SEG_A, SEG_B, SEG_K = map_segments()


def raycast(ox, oy, dx, dy):
    """nearest hit of rays (origin o, unit direction d; arrays of the same shape) with the map: (distance, kind)."""
    ox, oy, dx, dy = (np.asarray(v, float)[..., None] for v in (ox, oy, dx, dy))
    ax, ay = SEG_A[:, 0], SEG_A[:, 1]
    ex, ey = SEG_B[:, 0] - ax, SEG_B[:, 1] - ay
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ax - ox) * ey - (ay - oy) * ex) / den
        s = ((ax - ox) * dy - (ay - oy) * dx) / den
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (s >= 0) & (s <= 1)
    t = np.where(ok, t, np.inf)
    i = np.argmin(t, axis=-1)
    dist = np.take_along_axis(t, i[..., None], -1)[..., 0]
    return dist, SEG_K[i]


# ---------------------------------------------------------------------------------------- line distance grid
def _line_segments():
    L = []
    hl, hw = HALF_L, HALF_W
    L += [((-hl, hw), (hl, hw)), ((-hl, -hw), (hl, -hw)), ((hl, -hw), (hl, hw)), ((-hl, -hw), (-hl, hw)),
          ((0, -hw), (0, hw)), ((-hl, 0), (hl, 0))]
    n = 48
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        L.append(((CENTER_R * math.cos(a0), CENTER_R * math.sin(a0)), (CENTER_R * math.cos(a1), CENTER_R * math.sin(a1))))
    for s in (1, -1):
        xf = s * (hl - DA_D)
        L += [((xf, -DA_W / 2), (xf, DA_W / 2)), ((xf, DA_W / 2), (s * hl, DA_W / 2)), ((xf, -DA_W / 2), (s * hl, -DA_W / 2))]
    return L


GRID_RES = 0.02
GX0, GY0 = -5.2, -3.7
GNX, GNY = int(10.4 / GRID_RES) + 1, int(7.4 / GRID_RES) + 1


def _line_grid():
    xs = GX0 + GRID_RES * np.arange(GNX)
    ys = GY0 + GRID_RES * np.arange(GNY)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    D = np.full(X.shape, np.inf)
    for (x0, y0), (x1, y1) in _line_segments():
        ex, ey = x1 - x0, y1 - y0
        L2 = ex * ex + ey * ey
        t = np.clip(((X - x0) * ex + (Y - y0) * ey) / L2, 0, 1)
        D = np.minimum(D, np.hypot(X - (x0 + t * ex), Y - (y0 + t * ey)))
    return D.astype(np.float32)


_GRID = None


def line_dist(x, y):
    global _GRID
    if _GRID is None:
        _GRID = _line_grid()
    i = np.clip(np.round((np.asarray(x) - GX0) / GRID_RES).astype(int), 0, GNX - 1)
    j = np.clip(np.round((np.asarray(y) - GY0) / GRID_RES).astype(int), 0, GNY - 1)
    return _GRID[i, j]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------------- the filter
class ParticleFilter:
    def __init__(self, pose0, rng, n=300, sd0=(0.1, 0.1, math.radians(5)), cam_h=0.13, cam_fwd=0.072, f_px=320.0,
                 use_lines=True, use_boundary=True, pitch_robust=False):
        self.n, self.rng = n, rng
        self.p = np.array(pose0, float)[None, :] + rng.standard_normal((n, 3)) * np.array(sd0)
        self.w = np.full(n, 1.0 / n)
        self.cam_h, self.cam_fwd, self.f = cam_h, cam_fwd, f_px
        self.use_lines, self.use_boundary = use_lines, use_boundary
        self.pitch_robust = pitch_robust
        self.last_lik = 1.0
        self.fit = 1.0
        self.lik_slow, self.lik_fast = None, None

    def reset(self, pose, sd=(0.05, 0.05, math.radians(3))):
        self.p = np.array(pose, float)[None, :] + self.rng.standard_normal((self.n, 3)) * np.array(sd)
        self.w[:] = 1.0 / self.n

    def predict(self, d_odom):
        """d_odom = (dx, dy, dyaw) in the robot frame since the last picture (odometry)."""
        dx, dy, dth = d_odom
        dist = math.hypot(dx, dy)
        n = self.n
        r = self.rng.standard_normal((n, 3))
        sx = 0.01 * 0.1 + 0.08 * dist           # m: translation noise grows with the distance driven
        sth = math.radians(0.3) + 0.05 * abs(dth) + 0.05 * dist
        ndx = dx + r[:, 0] * sx
        ndy = dy + r[:, 1] * sx
        nth = dth + r[:, 2] * sth
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += c * ndx - s * ndy
        self.p[:, 1] += s * ndx + c * ndy
        self.p[:, 2] = wrap(self.p[:, 2] + nth)

    def update(self, boundary=None, lines=None):
        """boundary = (bx, by, kind) robot-frame ground points of the floor boundary and their kinds (0 wall, 1 tall,
        2 goal foam); lines = (m, 2) robot-frame ground points of white line pixels."""
        logw = np.zeros(self.n)
        P = self.p
        c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
        ox = P[:, 0:1] + c * self.cam_fwd
        oy = P[:, 1:2] + s * self.cam_fwd
        if self.use_boundary and boundary is not None and len(boundary[0]):
            bx, by, kind = boundary
            rx, ry = bx - self.cam_fwd, by
            rng_obs = np.hypot(rx, ry)
            az = np.arctan2(ry, rx)
            # rays per particle
            ang = P[:, 2:3] + az[None, :]
            dist, k_exp = raycast(np.broadcast_to(ox, ang.shape), np.broadcast_to(oy, ang.shape), np.cos(ang), np.sin(ang))
            dep_obs = np.arctan2(self.cam_h, rng_obs)[None, :]
            dep_exp = np.arctan2(self.cam_h, dist)
            sig = 2.0 / self.f                        # 2 px
            if self.pitch_robust:
                # the camera's pitch wobbles (chassis tilt): one unknown angle shifts every column alike. Per
                # particle, take out the common shift of the columns that roughly fit (|residual| < 3 deg), at most
                # +-2 deg, before scoring
                r_ = dep_obs - dep_exp
                m_ = np.abs(r_) < math.radians(3)
                off = np.clip((r_ * m_).sum(1) / np.maximum(m_.sum(1), 1), -math.radians(2), math.radians(2))
                dep_exp = dep_exp + off[:, None]
            z = (dep_obs - dep_exp) / sig
            p_hit = np.exp(-0.5 * z * z)
            p_short = np.where(dep_obs > dep_exp + sig, 0.25, 0.0)    # something nearer than the map (a robot)
            p_rand = 0.02
            # the kind: observed tall (1) can be a robot anywhere; goal foam (2) should be expected foam
            ko = np.asarray(kind)[None, :]
            kmatch = np.where(ko == 2, np.where(k_exp == 2, 1.0, 0.15), np.where((ko == 0) & (k_exp == 2), 0.3, 1.0))
            lik = (p_hit * kmatch + p_short + p_rand)
            # absolute fit of the best-weighted particle: share of columns that the map explains (hit within 3
            # sigma, or a nearer object); for the recovery trigger
            self._fit_cols = ((np.abs(z) < 3) | (p_short > 0))
            logw += np.log(lik).sum(1) * 0.5          # 0.5: neighbouring columns are not independent
        if self.use_lines and lines is not None and len(lines):
            lx, ly = lines[:, 0][None, :], lines[:, 1][None, :]
            wx = P[:, 0:1] + c * lx - s * ly
            wy = P[:, 1:2] + s * lx + c * ly
            dd = line_dist(wx, wy)
            rr = np.hypot(lx, ly)
            sig = 0.02 + 0.03 * rr
            lik = np.exp(-0.5 * (dd / sig) ** 2) + 0.1
            logw += np.log(lik).sum(1) * 0.5
        # the walls: a particle outside the walls is impossible
        out = (np.abs(P[:, 0]) > WALL_X - 0.08) | (np.abs(P[:, 1]) > WALL_Y - 0.08)
        logw[out] -= 50
        if hasattr(self, "_fit_cols"):
            self.fit = float(self._fit_cols[int(np.argmax(logw))].mean())
            del self._fit_cols
        m = logw.max()
        w = self.w * np.exp(logw - m)
        tot = w.sum()
        self.last_lik = float(np.mean(np.exp(logw - m)))
        self.w = w / tot if tot > 0 else np.full(self.n, 1.0 / self.n)
        ess = 1.0 / np.sum(self.w ** 2)
        if ess < self.n / 2:
            self._resample()

    def _resample(self):
        n = self.n
        pos = (self.rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(self.w), pos)
        idx = np.minimum(idx, n - 1)
        self.p = self.p[idx]
        self.w = np.full(n, 1.0 / n)
        # a little roughening against particle depletion
        self.p += self.rng.standard_normal((n, 3)) * np.array([0.005, 0.005, math.radians(0.3)])

    def inject(self, frac=0.1, radius=1.0, yaw_sd=math.radians(30)):
        """sensor-resetting light: replace a fraction of the particles by ones around the current estimate (not
        global: the symmetric field would hand a global reset its mirror pose half of the time)."""
        k = int(self.n * frac)
        if k <= 0:
            return
        e = self.estimate()
        idx = self.rng.choice(self.n, k, replace=False)
        ang = self.rng.uniform(0, 2 * math.pi, k)
        rad = radius * np.sqrt(self.rng.random(k))
        self.p[idx, 0] = e[0] + rad * np.cos(ang)
        self.p[idx, 1] = e[1] + rad * np.sin(ang)
        self.p[idx, 2] = wrap(e[2] + yaw_sd * self.rng.standard_normal(k))
        self.w[:] = 1.0 / self.n

    def estimate(self):
        w = self.w
        x, y = float(w @ self.p[:, 0]), float(w @ self.p[:, 1])
        th = math.atan2(float(w @ np.sin(self.p[:, 2])), float(w @ np.cos(self.p[:, 2])))
        return np.array([x, y, th])

    def spread(self):
        e = self.estimate()
        return float(np.sqrt(self.w @ ((self.p[:, 0] - e[0]) ** 2 + (self.p[:, 1] - e[1]) ** 2)))
