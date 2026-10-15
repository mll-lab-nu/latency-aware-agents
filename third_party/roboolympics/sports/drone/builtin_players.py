"""drone's built-in players (same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  the camera-only reference pilot of the legacy task (solution/vision_pilot.py and shutter_tracker.py),
             ported from OpenCV and SciPy to numpy (the containers have no OpenCV, the sim image no SciPy): it segments the orange gates,
             blue panels and magenta shutters in the FPV image (HSV thresholds), finds the gate frames' outer
             quadrilaterals (connected components -> convex hull -> polygon approximation), fits each gate's pose with
             the public geometry and calibration of /spec (least squares on the corners), decodes the gate numbers from
             the tag cells, keeps a map of the course, tracks the shutters' sinusoidal motion to pick a crossing window
             and plans clearances and panel detours. Its own pose is public telemetry. A new pilot every unit (it maps
             the course again). In the duel, one copy cruises a little faster than the other (by d drawn from seed and
             heat; which seat alternates with seed + heat), so two copies fly different heats and their own luck is
             balanced between the seats (the legacy oracle varied its command cadence per seat and heat).
  Random     random valid commands (generic tests: no draw, whitelist, determinism).
No simulator import: this file runs in the player's container as the task's oracle (numpy + Pillow only: the sim image that runs the checks has no SciPy)."""
from __future__ import annotations

import base64
import io
import math
import random

import numpy as np
W, H = 320, 240
CADENCE_S = 0.095                # the legacy oracle: a command every 0.06-0.08 s plus its HTTP round trips


# ------------------------------------------------------------------------------------------------ image helpers
def rgb_to_hsv(rgb):
    """OpenCV's 8-bit RGB -> HSV (H in 0..180, S and V in 0..255)."""
    x = rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=2)
    d = v - x.min(axis=2)
    s = np.where(v > 0, d * 255.0 / np.maximum(v, 1e-6), 0.0)
    dd = np.maximum(d, 1e-6)
    h = np.where(v == r, 60.0 * (g - b) / dd, np.where(v == g, 120.0 + 60.0 * (b - r) / dd, 240.0 + 60.0 * (r - g) / dd))
    h = np.where(d == 0, 0.0, h)
    h = np.where(h < 0, h + 360.0, h)
    return np.stack([np.round(h / 2), np.round(s), v], axis=2)


def in_range(hsv, lo, hi):
    return np.all((hsv >= np.asarray(lo)) & (hsv <= np.asarray(hi)), axis=2)


def _shift_all(mask, size, erode):
    """Erosion (all) or dilation (any) by a size x size square, anchor at size // 2 (OpenCV); the border neither
    erodes nor dilates."""
    a = size // 2
    pad = np.pad(mask, ((a, size - 1 - a), (a, size - 1 - a)), constant_values=erode)
    out = np.ones_like(mask) if erode else np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(size):
        for dx in range(size):
            win = pad[dy:dy + h, dx:dx + w]
            out = (out & win) if erode else (out | win)
    return out


def opening(mask, size):
    return mask if size == 1 else _shift_all(_shift_all(mask, size, True), size, False)


def closing(mask, size):
    return _shift_all(_shift_all(mask, size, False), size, True)


def components(mask):
    """The 8-connected components of a boolean mask, each as an array of its row runs (y, x_first, x_last)."""
    runs = []
    rows = np.nonzero(mask.any(axis=1))[0]
    for y in rows.tolist():
        d = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()):
            runs.append((y, a, b))
    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    prev, cur, cur_y = [], [], None
    for i, (y, a, b) in enumerate(runs):
        if y != cur_y:
            prev = cur if cur_y is not None and y == cur_y + 1 else []
            cur, cur_y = [], y
        for j in prev:
            _, pa, pb = runs[j]
            if pa <= b + 1 and pb >= a - 1:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        cur.append(i)
    groups = {}
    for i in range(len(runs)):
        groups.setdefault(find(i), []).append(runs[i])
    return [np.array(g) for g in groups.values()]


def count(runs):
    return int(np.sum(runs[:, 2] - runs[:, 1] + 1))


def hull(runs):
    """Convex hull (no collinear points) of a component's pixel centres; only each row's end pixels can be vertices."""
    p = sorted({(int(a), int(y)) for y, a, b in runs} | {(int(b), int(y)) for y, a, b in runs})
    if len(p) < 3:
        return np.array(p, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], float)


def least_squares(fun, x0, bounds, max_nfev=40, loss="linear", diff_step=None):
    """A small bounded Levenberg-Marquardt (forward-difference Jacobian, steps clipped to the box, soft_l1 by
    iteratively reweighting) for the pilot's fits of 1-4 parameters (replaces scipy.optimize.least_squares)."""
    lo, hi = (np.broadcast_to(np.asarray(b, float), np.shape(x0)) for b in bounds)
    x = np.clip(np.asarray(x0, float), lo, hi)
    r = np.asarray(fun(x), float)
    nfev = 1

    def cost(res):
        return float(np.sum(2 * (np.sqrt(1 + res ** 2) - 1))) if loss == "soft_l1" else float(res @ res)
    c, lam = cost(r), 1e-3
    while nfev + len(x) + 1 <= max_nfev:
        step = (diff_step or 1.5e-8) * np.maximum(1.0, np.abs(x))
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            xi = x.copy()
            xi[i] = xi[i] + step[i] if xi[i] + step[i] <= hi[i] else xi[i] - step[i]
            J[:, i] = (np.asarray(fun(xi), float) - r) / (xi[i] - x[i])
        nfev += len(x)
        w = 1 / np.sqrt(np.sqrt(1 + r ** 2)) if loss == "soft_l1" else np.ones_like(r)
        Jw, rw = J * w[:, None], r * w
        A, g = Jw.T @ Jw, Jw.T @ rw
        improved = False
        while nfev < max_nfev:
            try:
                dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                break
            xn = np.clip(x + dx, lo, hi)
            rn = np.asarray(fun(xn), float)
            nfev += 1
            cn = cost(rn)
            if cn < c:
                x, r, c, lam, improved = xn, rn, cn, max(lam / 3, 1e-9), True
                break
            lam *= 4
            if lam > 1e8:
                break
        if not improved or c < 1e-12:
            break
    return x


def poly_area(p):
    p = np.asarray(p, float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def perimeter(p):
    return float(np.sum(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)))


def _rdp(chain, eps):
    if len(chain) < 3:
        return list(chain)
    a, b = chain[0], chain[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    rel = chain[1:-1] - a
    d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n if n > 1e-9 else np.linalg.norm(rel, axis=1)
    i = int(np.argmax(d))
    if d[i] <= eps:
        return [a, b]
    left = _rdp(chain[: i + 2], eps)
    return left[:-1] + _rdp(chain[i + 1:], eps)


def approx_closed(poly, eps):
    """Douglas-Peucker of a closed polygon, split at two mutually far vertices (as cv2.approxPolyDP closed=True)."""
    n = len(poly)
    if n <= 4:
        return np.asarray(poly, float)
    a = int(np.argmax(np.linalg.norm(poly - poly[0], axis=1)))
    b = int(np.argmax(np.linalg.norm(poly - poly[a], axis=1)))
    i, j = sorted((a, b))
    c1 = poly[i:j + 1]
    c2 = np.concatenate([poly[j:], poly[: i + 1]])
    return np.array(_rdp(c1, eps)[:-1] + _rdp(c2, eps)[:-1], float)


def quad_of(runs, eps_frac, min_area=70):
    """The ordered (top-left, top-right, bottom-right, bottom-left) quadrilateral approximating a component's convex
    hull, or None."""
    hp = hull(runs)
    if len(hp) < 4 or poly_area(hp) < min_area:
        return None
    pts = approx_closed(hp, eps_frac * perimeter(hp))
    if len(pts) != 4 or np.min(pts) < 2 or np.max(pts[:, 0]) > W - 3 or np.max(pts[:, 1]) > H - 3:
        return None
    sums, diffs = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    q = pts[[np.argmin(sums), np.argmin(diffs), np.argmax(sums), np.argmax(diffs)]].astype(float)
    return q if len(np.unique(q, axis=0)) == 4 else None


def subpix(rgb, u, v):
    """Bilinear sample at (u, v), pixel centres at integer coordinates (cv2.getRectSubPix with a 1 x 1 patch)."""
    x0, y0 = int(math.floor(u)), int(math.floor(v))
    fx, fy = u - x0, v - y0
    x0, x1 = np.clip([x0, x0 + 1], 0, W - 1)
    y0, y1 = np.clip([y0, y0 + 1], 0, H - 1)
    p = rgb.astype(float)
    return ((1 - fx) * (1 - fy) * p[y0, x0] + fx * (1 - fy) * p[y0, x1] + (1 - fx) * fy * p[y1, x0]
            + fx * fy * p[y1, x1])


def clip_to_screen(poly):
    """Sutherland-Hodgman clip of a polygon by the image rectangle; None if nothing is visible."""
    out = [tuple(p) for p in np.asarray(poly, float)]
    for axis, bound, keep_low in ((0, 0.0, False), (0, W - 1.0, True), (1, 0.0, False), (1, H - 1.0, True)):
        inside = (lambda p: p[axis] <= bound) if keep_low else (lambda p: p[axis] >= bound)
        res = []
        for i, cur in enumerate(out):
            prev = out[i - 1]
            if inside(cur):
                if not inside(prev):
                    res.append(_cut(prev, cur, axis, bound))
                res.append(cur)
            elif inside(prev):
                res.append(_cut(prev, cur, axis, bound))
        out = res
        if not out:
            return None
    return np.array(out) if len(out) >= 3 else None


def _cut(a, b, axis, bound):
    t = (bound - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


# ------------------------------------------------------------------------------------------------ the pilot
def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ShutterTracker:
    """Estimate moving shutters from RGB and public gate geometry; no simulator state."""

    def __init__(self, spec):
        self.cfg = spec["moving_obstacles"]
        self.history, self.models, self.releases, self.wait_observations = {}, {}, [], 0
        w, h = self.cfg["width_m"], self.cfg["height_m"]
        self.corners = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])

    def observe(self, pilot, hsv, cam, C, pos, t):
        if not self.cfg["count"]:
            return
        mask = in_range(hsv, (135, 100, 55), (175, 255, 255))
        quads = [q for q in (quad_of(c, 0.025) for c in components(mask)) if q is not None]
        for number in self.cfg["gate_numbers"]:
            if number <= pilot.passed or number not in pilot.map:
                continue
            center, normal = pilot.map[number]
            distance = np.dot(center - pos, normal)
            if not 0.6 < distance < 8:
                continue
            yaw = math.atan2(normal[1], normal[0])
            lateral = np.array([-normal[1], normal[0], 0])
            front = center + (self.cfg["gate_normal_offset_m"] - self.cfg["thickness_m"] / 2) * normal
            best = None
            for q in quads:
                def residual(value):
                    world = pilot.object_world(self.corners, front + value[0] * lateral, yaw)
                    return (pilot.project(world, cam, C) - q).ravel()

                a = self.cfg["amplitude_m"] + 0.12
                x = least_squares(residual, [0.0], bounds=(-a, a), max_nfev=15)
                error = float(np.sqrt(np.mean(residual(x) ** 2)))
                if error < 3.0 and (best is None or error < best[0]):
                    best = (error, float(x[0]))
            if best is None:
                continue
            history = self.history.setdefault(number, [])
            if history and t - history[-1][0] < 0.04:
                continue
            history.append((t, best[1]))
            history[:] = [(stamp, y) for stamp, y in history if t - stamp < 10]
            self.fit(number)

    def fit(self, number):
        data = np.array(self.history[number])
        if len(data) < 12 or data[-1, 0] - data[0, 0] < 1.8:
            return
        now = data[-1, 0]
        ts = data[:, 0] - now
        amplitude = self.cfg["amplitude_m"]
        models = []
        for period in np.linspace(*self.cfg["period_range_s"], 61):
            omega = 2 * np.pi / period
            basis = np.column_stack([np.sin(omega * ts), np.cos(omega * ts), np.ones(len(ts))])
            coeff = np.linalg.lstsq(basis, data[:, 1], rcond=None)[0]
            amp = np.linalg.norm(coeff[:2])
            rms = float(np.sqrt(np.mean((basis @ coeff - data[:, 1]) ** 2)))
            if abs(amp - amplitude) > 0.15 or abs(coeff[2]) > 0.18 or rms > 0.07:
                continue
            models.append((rms ** 2 + 0.08 * (amp - amplitude) ** 2, now, omega, coeff, rms))
        if models:
            models.sort(key=lambda m: m[0])
            self.models[number] = [m for m in models if m[0] <= models[0][0] + 0.0004]

    def clear_window(self, number, t, start=0.90, end=1.55):
        models = self.models.get(number, [])
        if not models or t - models[0][1] > 0.4:
            return False
        stamps = np.linspace(t + start, t + end, 15)
        margin = self.cfg["width_m"] / 2 + 0.18 + 0.08
        for _, reference, omega, coeff, rms in models:
            phase = omega * (stamps - reference)
            y = coeff[0] * np.sin(phase) + coeff[1] * np.cos(phase) + coeff[2]
            if np.min(np.abs(y)) < margin + 2 * rms:
                return False
        return True


class VisionPilot:
    """Reference visual mapping and flight planner. Gate geometry, IDs and obstacles are inferred from RGB with the
    public calibration; the own pose is public telemetry."""

    def __init__(self, spec, speed=2.2):
        self.spec, self.speed = spec, speed
        c, g = spec["camera"], spec["gates"]
        self.f = c["height"] / (2 * np.tan(np.deg2rad(c["fovy_degrees"]) / 2))
        self.K = np.array([[self.f, 0, c["width"] / 2], [0, self.f, c["height"] / 2], [0, 0, 1.0]])
        self.cv_to_body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.width = g["opening_width_m"] + 2 * g["border_m"]
        self.height = g["opening_height_m"] + 2 * g["border_m"]
        self.corners = np.array([[-self.width / 2, -self.height / 2, 0], [self.width / 2, -self.height / 2, 0],
                                 [self.width / 2, self.height / 2, 0], [-self.width / 2, self.height / 2, 0.0]])
        iw, ih = g["opening_width_m"] / 2, g["opening_height_m"] / 2
        self.inner = np.array([[-iw, -ih, 0], [iw, -ih, 0], [iw, ih, 0], [-iw, ih, 0.0]])
        tag = g["tag"]
        self.tag_points = np.array([[(j - 3.5) * tag["cell_pitch_m"], -self.height / 2 - tag["centre_above_outer_frame_m"],
                                     tag["front_offset_m"]] for j in range(8)])
        self.map, self.panels, self.panel_quality, self.avoid = {}, [], [], []
        self.avoid_panel = self.clear_gate = self.search_origin = self.frozen_gate = self.rear_guard = None
        self.passed, self.search_start, self.detections, self.failures = -1, 0.0, 0, 0
        self.phase = "search"
        self.shutters = ShutterTracker(spec)

    def project(self, points, cam, C):
        p = (np.asarray(points) - cam) @ C
        if np.min(p[:, 2]) <= 0.08:
            return np.full((len(p), 2), 1e4)
        return p[:, :2] / p[:, 2, None] * self.f + self.K[:2, 2]

    @staticmethod
    def object_world(points, center, yaw):
        basis = np.array([[math.sin(yaw), 0, math.cos(yaw)], [-math.cos(yaw), 0, math.sin(yaw)], [0, -1, 0]])
        return np.asarray(points) @ basis.T + center

    def quads(self, mask):
        """[(quad, object corners)]: the frames' outer quadrilaterals (as the legacy pilot) and, new in the port, the
        openings' quadrilaterals (holes of the orange mask that do not touch the image border), which stay clean when
        a frame touches another gate's frame in the image."""
        candidates = []
        for size in (1, 2, 3):
            for c in components(opening(mask, size)):
                q = quad_of(c, 0.028)
                if q is not None:
                    candidates.append((poly_area(q), 0, q))
        for c in components(~mask):
            if c[:, 0].min() > 0 and c[:, 0].max() < H - 1 and c[:, 1].min() > 0 and c[:, 2].max() < W - 1:
                q = quad_of(c, 0.028)
                if q is not None:
                    candidates.append((poly_area(q), 1, q))
        best = sorted(candidates, key=lambda t: (t[1], -t[0]))
        return [(q, self.inner if kind else self.corners) for _, kind, q in best][:32]

    def initial_pose(self, q, cam, C, corners):
        """The gate's front-face centre and yaw from its outer quadrilateral (replaces cv2.solvePnP): the depth from
        the apparent height of the vertical sides, the centre on the ray through the quadrilateral's centre."""
        side = 0.5 * (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2]))
        depth = self.f * (corners[2, 1] - corners[1, 1]) / max(side, 1.0)
        u, v = q.mean(axis=0)
        ray = np.array([(u - self.K[0, 2]) / self.f, (v - self.K[1, 2]) / self.f, 1.0])
        return cam + C @ (ray * depth)

    def detect_gates(self, rgb, hsv, cam, C, pos):
        mask = in_range(hsv, (0, 105, 65), (18, 255, 255))
        accepted = []
        maxyaw = self.spec["gates"]["max_abs_yaw_rad"] + 0.05
        lower = np.array([pos[0] + 0.12, -5.5, 0.7, -maxyaw])
        upper = np.array([52.0, 5.5, 3.7, maxyaw])
        fronts = []
        for q, corners in self.quads(mask):
            if any(np.linalg.norm(q - p) < 5 for p in accepted):
                continue
            center0 = self.initial_pose(q, cam, C, corners)

            def residual(v):
                return (self.project(self.object_world(corners, v[:3], v[3]), cam, C) - q).ravel()
            best = None
            for yaw0 in (0.0, -0.45, 0.45):
                initial = np.clip(np.r_[center0, yaw0], lower + 0.001, upper - 0.001)
                x = least_squares(residual, initial, bounds=(lower, upper), max_nfev=35, loss="soft_l1")
                err = float(np.sqrt(np.mean(residual(x) ** 2)))
                if best is None or err < best[0]:
                    best = (err, x)
                if err <= 1.0:
                    break
            if best[0] > 2.0:
                continue
            front, yaw = best[1][:3], best[1][3]
            if any(np.linalg.norm(front - p) < 0.3 for p in fronts):
                continue                    # this gate's frame or opening was taken in this image already
            normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            projected = self.project(self.object_world(self.tag_points, front, yaw), cam, C)
            number = None
            if (np.all(projected > 1) and np.all(projected[:, 0] < W - 2) and np.all(projected[:, 1] < H - 2)
                    and np.min(np.linalg.norm(np.diff(projected, axis=0), axis=1)) > 2.0):
                bits = []
                for u, v in projected:
                    pixel = subpix(rgb, float(u), float(v))
                    white = pixel.min() > 55 and (pixel.max() - pixel.min()) < 0.35 * max(1, pixel.max())
                    black = pixel.max() < 45
                    bits.append(1 if white else 0 if black else -1)
                if -1 not in bits and all(bits[i] + bits[i + 4] == 1 for i in range(4)):
                    number = sum(bits[i] << (3 - i) for i in range(4))
                    if not 1 <= number <= self.spec["gates"]["count"]:
                        number = None
            center = front + normal * self.spec["gates"]["frame_depth_m"] / 2
            if number is None:
                matches = [n for n, (p, _) in self.map.items() if n > self.passed and np.linalg.norm(p - center) < 0.5]
                if len(matches) == 1:
                    number = matches[0]
            if number is None or number <= self.passed:
                continue
            spacing = self.spec["gates"]["forward_spacing_range_m"]     # reject aliased IDs (public order, spacing)
            if any(not spacing[0] * abs(number - n) - 1.5 <= (center[0] - p[0]) * (1 if number > n else -1)
                   <= spacing[1] * abs(number - n) + 1.5 for n, (p, _) in self.map.items() if n != number):
                continue
            if number in self.map:
                old, oldnormal = self.map[number]
                if np.linalg.norm(center - old) > 0.8:
                    continue
                center = 0.7 * old + 0.3 * center
                normal = 0.7 * oldnormal + 0.3 * normal
                normal /= np.linalg.norm(normal)
            self.map[number] = (center, normal)
            self.detections += 1
            accepted.append(q)
            fronts.append(front)
        if not accepted:
            self.failures += 1

    def detect_panels(self, hsv, cam, C, pos):
        obstacle = self.spec["obstacles"]
        if not obstacle["count"]:
            return
        mask = closing(in_range(hsv, (98, 100, 50), (125, 255, 255)), 7)
        w, h = obstacle["width_m"], obstacle["height_m"]
        z = obstacle["bottom_z_m"] + h / 2
        corners = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0.0]])
        comps = sorted(components(mask), key=lambda c: -count(c))[:3]
        for runs in comps:
            x, y = int(runs[:, 1].min()), int(runs[:, 0].min())
            bw, bh = int(runs[:, 2].max()) - x + 1, int(runs[:, 0].max()) - y + 1
            if count(runs) < 100 or bw < 20:
                continue
            target = np.array([x, y, x + bw - 1, y + bh - 1])
            depth = self.f * w / max(1, bw)
            guess = cam + C @ np.array([(x + bw / 2 - W / 2) / self.f * depth, 0, depth])
            lower, upper = np.array([pos[0] + 0.15, -5.0]), np.array([52.0, 5.0])
            initial = np.clip(guess[:2], lower + 0.001, upper - 0.001)

            def residual(v):
                pixels = self.project(self.object_world(corners, [v[0], v[1], z], 0.0), cam, C)
                visible = clip_to_screen(pixels)
                if visible is None:
                    return np.full(4, 1000.0)
                return np.r_[visible.min(axis=0), visible.max(axis=0)] - target

            x = least_squares(residual, initial, bounds=(lower, upper), max_nfev=40, diff_step=0.001)
            if np.sqrt(np.mean(residual(x) ** 2)) > 2.5:
                continue
            point = np.r_[x, z]
            if any(np.linalg.norm(point[:2] - c[:2]) < 1.2 for c, _ in self.map.values()):
                continue                    # an aliased view of a panel on top of a mapped gate
            matches = [i for i, p in enumerate(self.panels) if abs(p[0] - point[0]) < 4.0 and abs(p[1] - point[1]) < 1.5]
            if matches:
                i = matches[0]
                if bw >= 0.8 * self.panel_quality[i]:
                    self.panels[i] = 0.5 * self.panels[i] + 0.5 * point
                    self.panel_quality[i] = max(self.panel_quality[i], bw)
            elif len(self.panels) < obstacle["count"]:
                self.panels.append(point)
                self.panel_quality.append(bw)

    @staticmethod
    def waypoint_velocity(pos, vel, point, speed=2.2):
        v = 2.0 * (point - pos) - 0.5 * vel
        norm = np.linalg.norm(v)
        if norm > speed:
            v *= speed / norm
        v[2] = np.clip(v[2], -1.5, 1.5)
        return v

    def command(self, observation, rgb):
        robot = observation["robot"]
        pos, vel = np.asarray(robot["position"]), np.asarray(robot["velocity"])
        R = quaternion_matrix(robot["quaternion_wxyz"])
        C = R @ self.cv_to_body
        cam = pos + R @ np.asarray(self.spec["camera"]["body_position"])
        t, passed = observation["time_s"], observation["gates_passed"]
        gate_numbers = self.shutters.cfg["gate_numbers"]
        if passed != self.passed:
            if passed and passed in self.map:
                self.clear_gate = self.frozen_gate if self.frozen_gate is not None else self.map[passed]
                self.rear_guard = self.clear_gate if passed in gate_numbers else None
            self.passed, self.search_start, self.search_origin = passed, t, pos.copy()
            self.phase, self.frozen_gate = "approach", None
        hsv = rgb_to_hsv(rgb)
        self.detect_gates(rgb, hsv, cam, C, pos)
        self.detect_panels(hsv, cam, C, pos)
        self.shutters.observe(self, hsv, cam, C, pos, t)
        target = self.map.get(passed + 1)
        look_yaw, point, mode = 0.0, None, "search"
        if self.clear_gate is not None:
            center, normal = self.clear_gate
            if np.dot(pos - center, normal) < 0.45:
                point, look_yaw, mode = center + 0.7 * normal, math.atan2(normal[1], normal[0]), "clear"
            else:
                self.clear_gate = None
        if point is None:
            while self.avoid and np.linalg.norm(pos - self.avoid[0]) < 0.35:     # finish a lateral clearance first
                self.avoid.pop(0)
            if not self.avoid:
                if target is not None:
                    entry = 1.9 if passed + 1 in gate_numbers else 1.0
                    aim = target[0] - entry * target[1]
                else:
                    aim = pos + np.array([3.5, 0, 0])
                for panel in sorted(self.panels, key=lambda p: p[0]):
                    extent = np.array([0.4, self.spec["obstacles"]["width_m"] / 2 + 0.4])
                    low, high = panel[:2] - extent, panel[:2] + extent
                    lo, hi = 0.0, 1.0
                    for axis in range(2):
                        delta = aim[axis] - pos[axis]
                        if abs(delta) < 1e-8:
                            if not low[axis] <= pos[axis] <= high[axis]:
                                hi = -1.0
                        else:
                            a, b = sorted(((low[axis] - pos[axis]) / delta, (high[axis] - pos[axis]) / delta))
                            lo, hi = max(lo, a), min(hi, b)
                    if lo > hi:
                        continue
                    side = 1 if pos[1] >= panel[1] else -1
                    side_y = panel[1] + side * (self.spec["obstacles"]["width_m"] / 2 + 0.6)
                    altitude = float(np.clip(pos[2], 1.1, 3.3))
                    self.avoid = [np.array([max(pos[0], panel[0] - 0.9), side_y, altitude]),
                                  np.array([panel[0] + 0.7, side_y, altitude])]
                    self.avoid_panel = panel
                    break
            if self.avoid:
                point, mode = self.avoid[0], "avoid"
            elif target is not None:
                center, normal = self.frozen_gate if self.frozen_gate is not None else target
                signed = np.dot(pos - center, normal)
                cross = pos - center - signed * normal
                moving = passed + 1 in gate_numbers
                entry = 1.9 if moving else 1.0
                if self.phase == "approach":
                    if moving:
                        settled = np.linalg.norm(pos - (center - entry * normal)) < 0.13 and np.linalg.norm(vel) < 0.18
                        self.shutters.wait_observations += int(settled)
                        if settled and self.shutters.clear_window(passed + 1, t):
                            self.phase = "cross"
                            self.frozen_gate = (center.copy(), normal.copy())
                            self.shutters.releases.append(dict(gate=passed + 1, time=t))
                    elif np.linalg.norm(pos - (center - entry * normal)) < 0.4 or (
                            signed > -0.9 and np.linalg.norm(cross) < 0.16):
                        self.phase = "cross"
                point = center + (0.8 if self.phase == "cross" else -entry) * normal
                mode = "wait" if moving and self.phase == "approach" else self.phase
            else:
                age = t - self.search_start
                point = self.search_origin.copy()
                point[0] += min(2.0, 0.35 * age)
                point[2] = 2.1 + 0.65 * math.sin(0.8 * age)
                look_yaw = math.atan2(-pos[1], 4.0) + 0.65 * math.sin(1.1 * age)
            if target is not None:
                delta = target[0] - pos
                if np.linalg.norm(delta[:2]) > 0.35:
                    look_yaw = math.atan2(delta[1], delta[0])
            elif mode == "avoid":
                look_yaw = math.atan2(point[1] - pos[1], max(0.8, point[0] - pos[0]))
        if self.rear_guard is not None:
            center, normal = self.rear_guard
            sweep = max(self.width / 2, self.shutters.cfg["amplitude_m"] + self.shutters.cfg["width_m"] / 2)
            if pos[0] > center[0] + abs(normal[1]) * sweep + 0.7:
                self.rear_guard = None
            else:                       # a turn may cut a yawed gate / shutter again: push the target along +x
                signed = np.dot(point - center, normal)
                point = point + np.array([max(0.0, 0.65 - signed) / max(normal[0], 0.3), 0.0, 0.0])
        velocity = self.waypoint_velocity(pos, vel, point, self.speed if mode != "search" else 0.8)
        yaw = math.atan2(R[1, 0], R[0, 0])
        rate = float(np.clip(2.0 * wrap(look_yaw - yaw), -1.2, 1.2))
        self.last_mode = mode
        return velocity, rate


def fly(velocity, rate, duration=0.8):
    v = [float(x) for x in velocity]
    h = math.hypot(v[0], v[1])
    if h > 3.99:                        # the limit is 4 m/s; stay clear of it after rounding
        v[0], v[1] = v[0] * 3.99 / h, v[1] * 3.99 / h
    return {"type": "fly", "vx": round(v[0], 6), "vy": round(v[1], 6), "vz": round(float(np.clip(v[2], -2, 2)), 6),
            "yaw_rate": round(float(np.clip(rate, -1.5, 1.5)), 6), "duration": duration}


def image_of(view):
    from PIL import Image
    obs = view.get("observation") or {}
    if not obs.get("image_png_b64"):
        return None
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["image_png_b64"]))).convert("RGB"))
    tile = next((t for t in obs.get("tiles") or [] if t.get("camera") == "fpv"), None)
    if tile:
        img = img[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]]
    return img


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec, self.seat, self.seed = spec, seat, int(seed)
        self.duel = (spec.get("event_format") or {}).get("kind") == "duel"
        self.heat, self.pilot, self.next_t = None, None, 0.0

    def act(self, view):
        st = view["state"]
        if st.get("world", "running") != "running" or st.get("flight_over") or "robot" not in st:
            return None
        if st["heat"] != self.heat:
            self.heat = st["heat"]
            speed = 2.2
            if self.duel:                   # two copies must not fly identical, tied heats: one copy cruises d faster,
                d = random.Random(f"{self.seed}/{self.heat}").uniform(0.01, 0.06)    # the other d slower; the faster
                faster = "A" if (self.seed + self.heat) % 2 == 0 else "B"          # seat alternates with seed + heat
                speed *= 1 + d if self.seat == faster else 1 - d
            self.pilot, self.next_t = VisionPilot(self.spec, speed), 0.0
        if st["time_s"] < self.next_t - 1e-6:   # a command every CADENCE_S game seconds at most (it lasts 0.8 s)
            return None
        rgb = image_of(view)
        if rgb is None:
            return None
        self.next_t = st["time_s"] + CADENCE_S
        velocity, rate = self.pilot.command(st, rgb)
        return fly(velocity, rate)

    def review(self, reply):
        return ("reference pilot: unit over, gate map rebuilt each unit", "reference pilot: same policy")


class Random:
    """Random valid commands, by the seed (not the seat)."""

    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 17)

    def act(self, view):
        if view["state"].get("flight_over"):
            return None
        v = self.g.uniform([-1.0, -2.5, -1.5], [3.5, 2.5, 1.5])
        return fly(v, self.g.uniform(-1.0, 1.0), float(self.g.uniform(0.1, 1.0)))
