"""basketball's built-in players (same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  the reference shooter (camera images and the documented physics only). Retuned for revision
             basketball-5: it projects every rim-coloured pixel of the shooter_zoom tile through the camera (its pose in
             the shooter's frame from /spec) onto the rim plane z = 3.048 and fits a circle of the rim's radius to these
             points, trimming outliers (the orange bracket behind the rim, which biased the legacy left/right-extreme
             estimate by up to +4 % in range at the wing and baseline spots); the hoop's bearing is the yaw, its distance
             the range; it solves the documented flight (gravity, drag -c_d |v| v, Magnus c_m (omega x v), backspin;
             RK4) from the documented release point for the release speed at 54 degrees and 1.5 rev/s backspin. After
             a miss at a spot, its later shots from that spot are 1 % slower if the ball touched the backboard (long),
             else 0.6 % faster (short); it reads its last result from the state. Proves the sport is winnable (sport
             rule 9).
  Random     plausible random shots (generic tests: whitelist, determinism).
No simulator import: this file runs in the player's container as the task's oracle (numpy + pillow only)."""
from __future__ import annotations

import base64
import io
import math

import numpy as np

G = 9.81
DT = 0.002
RIM_TUBE = 0.008                                 # the rim tube's radius (16 mm tube, /spec court)
ANGLE = 54.0
SPIN = 1.5


def rim_mask(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 120) & (g > 0.26 * r) & (g < 0.46 * r) & (b < 0.12 * r)


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.seed = int(seed)
        cam = next(c for c in spec["cameras"] if c["name"] == "shooter_zoom")
        self.cam_pos = np.array(cam["pos_in_shooter_frame"], float)
        p = math.radians(cam["pitch_up_deg"])
        self.fwd = np.array([math.cos(p), 0.0, math.sin(p)])
        self.right = np.array([0.0, -1.0, 0.0])
        self.up = np.array([-math.sin(p), 0.0, math.cos(p)])
        self.focal = cam["focal_px"]
        self.pp = cam["principal_point_px"]
        rel = spec["release_point_in_shooter_frame_m"]
        self.rel_fwd, self.rel_h = rel[0], rel[2]
        self.m, self.cd, self.cm = spec["ball_mass_kg"], spec["drag_coef_n_s2_m2"], spec["magnus_coef"]
        self.hoop_z = spec["hoop_centre_m"][2]
        self.rim_r = spec["rim_inner_diameter_m"] / 2 + RIM_TUBE     # the tube's centre line
        self.speed_scale = {}                            # per spot: correction factor learned from its misses
        self.learned = 0                                 # the last own shot already learned from

    def locate(self, img, tile):
        """The hoop centre (x, y) in the shooter's frame, or None. Every rim-coloured pixel's ray meets the rim plane
        z = hoop height; a circle of the rim's radius is fitted to these points (least squares, trimming points more
        than 3 cm off it: the orange bracket behind the rim)."""
        t = img[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]]
        ys, xs = np.nonzero(rim_mask(t))
        if len(xs) < 20:
            return None
        u = (xs + 0.5 - self.pp[0]) / self.focal
        v = (ys + 0.5 - self.pp[1]) / self.focal
        rays = self.fwd[None, :] + u[:, None] * self.right[None, :] - v[:, None] * self.up[None, :]
        ok = rays[:, 2] > 1e-6
        if ok.sum() < 20:
            return None
        s = (self.hoop_z - self.cam_pos[2]) / rays[ok, 2]
        pts = self.cam_pos[None, :2] + s[:, None] * rays[ok, :2]
        c = np.median(pts, axis=0)
        keep = np.ones(len(pts), bool)
        for it in range(30):
            d = pts[keep] - c
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            c = c + np.mean(d - self.rim_r * d / r[:, None], axis=0)      # Gauss-Newton step for a known radius
            if it in (9, 19):
                keep = np.abs(np.linalg.norm(pts - c, axis=1) - self.rim_r) < 0.03
                if keep.sum() < 20:
                    return None
        return float(c[0]), float(c[1])

    def flight(self, angle, speed, spin):
        """Horizontal distance from the pedestal axis where the ball's centre comes down through the rim plane."""
        a = math.radians(angle)
        x, z = self.rel_fwd, self.rel_h
        vx, vz = speed * math.cos(a), speed * math.sin(a)
        w = 2 * math.pi * spin                           # backspin about the horizontal axis (d x z)

        def acc(vx, vz):
            s = math.hypot(vx, vz)
            fx = -self.cd * s * vx + self.cm * (-w * vz)
            fz = -self.cd * s * vz + self.cm * (w * vx)
            return fx / self.m, fz / self.m - G

        for _ in range(5000):
            ax1, az1 = acc(vx, vz)
            ax2, az2 = acc(vx + .5 * DT * ax1, vz + .5 * DT * az1)
            ax3, az3 = acc(vx + .5 * DT * ax2, vz + .5 * DT * az2)
            ax4, az4 = acc(vx + DT * ax3, vz + DT * az3)
            nx = x + DT * (vx + DT / 6 * (ax1 + ax2 + ax3))
            nz = z + DT * (vz + DT / 6 * (az1 + az2 + az3))
            nvx, nvz = vx + DT / 6 * (ax1 + 2 * ax2 + 2 * ax3 + ax4), vz + DT / 6 * (az1 + 2 * az2 + 2 * az3 + az4)
            if nvz < 0 and z >= self.hoop_z > nz:
                return x + (nx - x) * (z - self.hoop_z) / (z - nz)
            if nz < 0:
                return -1.0
            x, z, vx, vz = nx, nz, nvx, nvz
        return -1.0

    def speed_for(self, dist):
        lo, hi = 3.0, 14.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.flight(ANGLE, mid, SPIN) < dist:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def learn(self, st):
        """After a miss, the later shots from that spot are a little shorter (backboard first) or longer."""
        last = st.get("last_shot") or {}
        if not last or last.get("shot", 0) <= self.learned:
            return
        self.learned = last["shot"]
        if last.get("result") == "made":
            return
        long = "backboard" in (last.get("touched") or [])
        spot = last["spot"]
        self.speed_scale[spot] = self.speed_scale.get(spot, 1.0) * (0.99 if long else 1.006)

    def act(self, view):
        from PIL import Image
        st, obs = view["state"], view["observation"]
        self.learn(st)
        spot = st.get("spot", 1)
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["image_png_b64"]))).convert("RGB"))
        tile = next(t for t in obs["tiles"] if t["camera"] == "shooter_zoom")
        h = self.locate(img, tile)
        if h is None:
            dec = {"angle_deg": ANGLE, "speed": 8.0, "spin_hz": SPIN, "yaw_deg": 0.0}
        else:
            yaw = math.degrees(math.atan2(h[1], h[0]))
            v = self.speed_for(math.hypot(*h)) * self.speed_scale.get(spot, 1.0)
            dec = {"angle_deg": ANGLE, "speed": v, "spin_hz": SPIN, "yaw_deg": yaw}
        return {"type": "shoot", "angle_deg": dec["angle_deg"], "speed": round(min(14.0, max(3.0, dec["speed"])), 4),
                "spin_hz": dec["spin_hz"], "yaw_deg": round(min(35.0, max(-35.0, dec["yaw_deg"])), 4)}

    def review(self, reply):
        return "reference shooter: shot recorded", "find the rim again"


class Random:
    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 17)

    def act(self, view):
        return {"type": "shoot", "angle_deg": round(float(self.g.uniform(45.0, 60.0)), 4),
                "speed": round(float(self.g.uniform(6.0, 9.0)), 4), "spin_hz": round(float(self.g.uniform(0.0, 3.0)), 4),
                "yaw_deg": round(float(self.g.uniform(-10.0, 10.0)), 4)}
