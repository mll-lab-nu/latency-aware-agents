"""archery's built-in players (same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  the reference archer of the legacy task (camera images and the documented physics only): it finds the
             orange stripes of the windsocks in the archer and side views, fits the horizontal wind whose sock
             directions project there (grid search, with the camera poses and focal lengths of /spec), and solves
             the documented point-mass flight (gravity + quadratic drag on the air-relative velocity) for the
             elevation and azimuth that put the arrow on the gold centre at full draw (Newton), as the legacy
             single task's reference did (the seeded personal aim error of the duel is gone with the duel, B10).
             Proves the sport is winnable (sport rule 9).
  Random     plausible random shots (generic tests: whitelist, determinism).
No simulator import: this file runs in the player's container as the task's oracle (numpy + pillow only)."""
from __future__ import annotations

import base64
import io
import math

import numpy as np

G = 9.81
DT = 0.002
SOCK_CENTROID = 0.5625        # area-weighted centre of the three orange stripes along the sock (m from the mouth)


def fly_to(D, el, az, speed, wind, K, launch):
    """(y, z) where the arrow crosses the plane x = D (extrapolated if it lands short)."""
    e, a = math.radians(el), math.radians(az)
    x, y, z = launch
    vx, vy, vz = speed * math.cos(e) * math.cos(a), speed * math.cos(e) * math.sin(a), speed * math.sin(e)
    wx, wy = wind

    def acc(vx, vy, vz):
        rx, ry = vx - wx, vy - wy
        s = math.sqrt(rx * rx + ry * ry + vz * vz)
        return -K * s * rx, -K * s * ry, -G - K * s * vz

    for _ in range(20000):
        s0 = (x, y, z, vx, vy, vz)
        k1 = (vx, vy, vz) + acc(vx, vy, vz)
        s = [s0[i] + .5 * DT * k1[i] for i in range(6)]
        k2 = (s[3], s[4], s[5]) + acc(s[3], s[4], s[5])
        s = [s0[i] + .5 * DT * k2[i] for i in range(6)]
        k3 = (s[3], s[4], s[5]) + acc(s[3], s[4], s[5])
        s = [s0[i] + DT * k3[i] for i in range(6)]
        k4 = (s[3], s[4], s[5]) + acc(s[3], s[4], s[5])
        n = [s0[i] + DT / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]
        if n[0] >= D:
            f = (D - x) / (n[0] - x)
            return y + f * (n[1] - y), z + f * (n[2] - z)
        if n[2] <= -5:
            return y + vy / vx * (D - x), z + vz / vx * (D - x)
        x, y, z, vx, vy, vz = n
    raise RuntimeError("flight did not end")


def solve(D, wind, K, v0, launch, face_z, target=(0.0, 0.0)):
    """(elevation, azimuth) in degrees that put the arrow at `target` (y, dz) of the gold centre (Newton)."""
    el = math.degrees(math.atan2(face_z - launch[2], D)) + math.degrees(0.5 * G * D / v0 ** 2)
    az = 0.0
    for _ in range(12):
        y, z = fly_to(D, el, az, v0, wind, K, launch)
        ey, ez = y - target[0], z - face_z - target[1]
        if abs(ey) < 1e-4 and abs(ez) < 1e-4:
            break
        h = 1e-3
        y1, z1 = fly_to(D, el + h, az, v0, wind, K, launch)
        y2, z2 = fly_to(D, el, az + h, v0, wind, K, launch)
        J = np.array([[(y1 - y) / h, (y2 - y) / h], [(z1 - z) / h, (z2 - z) / h]])
        d = np.linalg.solve(J, [-ey, -ez])
        el, az = el + float(d[0]), az + float(d[1])
    return el, az


def orange(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 90) & (g > 0.2 * r) & (g < 0.65 * r) & (b < 0.3 * r)


def sock_dirs(wx, wy, full):
    """Unit sock directions for arrays of winds (downwind heading; droop 90 deg x (1 - speed / full))."""
    s = np.hypot(wx, wy)
    head = np.arctan2(wy, wx)
    droop = np.radians(90.0 * (1.0 - np.minimum(s / full, 1.0)))
    return np.stack([np.cos(droop) * np.cos(head), np.cos(droop) * np.sin(head), -np.sin(droop)], -1)


def project(cam, P):
    """Pixel (u, v) of world points P (..., 3) for a /spec camera (pos, forward, right, up, focal_px, principal point)."""
    q = P - np.asarray(cam["pos"])
    depth = q @ np.asarray(cam["forward"])
    u = cam["principal_point_px"][0] + cam["focal_px"] * (q @ np.asarray(cam["right"])) / depth
    v = cam["principal_point_px"][1] - cam["focal_px"] * (q @ np.asarray(cam["up"])) / depth
    return np.stack([u, v], -1)


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec, self.seed = spec, int(seed)
        self.K, self.launch = spec["drag_K_per_m"], tuple(spec["launch_point_m"])
        self.face_z, self.v_full = spec["face_centre_z_m"], spec["full_draw_speed_m_s"]
        self.full = spec["windsock_full_speed_m_s"]
        g = np.arange(-9.5, 9.51, 0.1)
        self.WX, self.WY = (x.ravel() for x in np.meshgrid(g, g))
        self.dirs = sock_dirs(self.WX, self.WY, self.full)

    def read_wind(self, img, tiles, D):
        """Fit the wind (wx, wy) from the archer and side tiles of the observation image (end at distance D)."""
        cams = {c["name"]: c for c in self.spec["cameras_by_distance"][str(int(D))]}
        tops = self.spec["windsocks_by_distance"][str(int(D))]
        cost = np.zeros(len(self.WX))
        used = 0
        for view, which in (("archer", (0, 1)), ("side", (1,))):
            t = tiles[view]
            m = orange(img[t["y"]:t["y"] + t["height"], t["x"]:t["x"] + t["width"]])
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            pix = np.stack([xs + .5, ys + .5], -1)
            top_px = project(cams[view], np.asarray(tops))
            owner = np.argmin(((pix[:, None, :] - top_px[None]) ** 2).sum(-1), 1)
            for i in which:
                sel = pix[owner == i]
                q = np.asarray(tops[i]) - np.asarray(cams[view]["pos"])
                scale = cams[view]["focal_px"] / (q @ np.asarray(cams[view]["forward"]))   # px per metre there
                near = sel[np.hypot(*(sel - top_px[i]).T) < 1.5 * scale]
                if len(near) < 6:
                    continue
                c = near.mean(0)
                pred = project(cams[view], np.asarray(tops[i])[None] + SOCK_CENTROID * self.dirs)
                cost += (((pred - c) / scale) ** 2).sum(-1)
                used += 1
        if not used:
            return (0.0, 0.0)
        k = int(np.argmin(cost))
        return float(self.WX[k]), float(self.WY[k])

    def act(self, view):
        from PIL import Image
        st, obs = view["state"], view["observation"]
        D = float(st["distance_m"])
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["image_png_b64"]))).convert("RGB"))
        tiles = {t["camera"]: t for t in obs["tiles"]}
        wind = self.read_wind(img, tiles, D)
        el, az = solve(D, wind, self.K, self.v_full, self.launch, self.face_z)
        return {"type": "shoot", "elev_deg": round(min(30.0, max(-10.0, el)), 4),
                "azim_deg": round(min(20.0, max(-20.0, az)), 4), "draw": 1.0}

    def review(self, reply):
        return "reference archer: arrow recorded", "read the windsocks again for the next arrow"


class Random:
    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 17)

    def act(self, view):
        return {"type": "shoot", "elev_deg": round(float(self.g.uniform(1.5, 5.0)), 4),
                "azim_deg": round(float(self.g.uniform(-1.0, 1.0)), 4), "draw": round(float(self.g.uniform(0.9, 1.0)), 4)}
