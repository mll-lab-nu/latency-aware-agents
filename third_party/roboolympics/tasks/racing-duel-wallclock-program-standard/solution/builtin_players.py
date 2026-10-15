"""racing's built-in players (contract ②): they play through the same interface as a model's program, from the camera
image, the seat's own car in the state and /spec only, and import no simulator.

Reference: the legacy camera-only lane follower (solution/camera_player.py of the Season-1 tasks), unchanged in its
perception and control. Perception (numpy only): the yellow centre stripe (or, without it, the middle of the green
road) is projected to the ground with the published camera pose and fitted with a parabola y(x) in the car frame; the
target is a look-ahead point offset laterally to the car's grid side (pure pursuit). Magenta pixels are obstacles: the
car detours through the obstacle's free side until it has passed it. Other pixels on the road (not road, stripe,
finish line) are cars: the player slows to keep its distance and moves to the freer side to pass. The travelled
distance is integrated over the game time between decisions (the state's t_sim; legacy: wall time / slowdown).

Random: a check policy (core/testkit.py): a fresh random target speed and steering every decision, seeded.
"""

import base64
import io
import math
import random

import numpy as np
from PIL import Image

WHEELBASE = 0.46
FAST, MID, SLOW = 3.8, 3.2, 2.6  # m/s on straights, in bends, in tight bends


class Player:
    def __init__(self, spec):
        cam = spec["camera"]
        fr = cam["front"]
        self.W, self.H = cam["width"], cam["height"]
        self.f = self.H / (2 * math.tan(math.radians(cam["fovy_deg"]) / 2))
        self.p = math.radians(fr["pitch_down_deg"])
        self.h = fr["height_m"]
        self.off = fr["offset_m"]
        v, u = np.mgrid[0 : self.H, 0 : self.W].astype(float)
        vy = (self.H / 2 - v) / self.f
        dz = -math.sin(self.p) + vy * math.cos(self.p)
        self.valid = dz < -0.02
        scale = np.where(self.valid, -self.h / np.where(self.valid, dz, -1), 0.0)
        self.gx = self.off + scale * (math.cos(self.p) + vy * math.sin(self.p))  # forward (m, from car centre)
        self.gy = -scale * (u - self.W / 2) / self.f  # left (m)
        self.valid &= self.gx < 6.0
        self.reset()

    def reset(self):
        self.travel = 0.0
        self.detour = None
        self.turn = 0.0
        self.fit = None
        self.lost = 0

    @staticmethod
    def masks(im):
        r, g, b = (im[:, :, i].astype(int) for i in range(3))
        yellow = (r > 140) & (g > 100) & (b < 100) & (r > 1.4 * b)
        green = (g > 45) & (g > 1.35 * r) & (g > 1.2 * b)
        magenta = (r > 140) & (b > 100) & (g < 90)
        white = (np.minimum(np.minimum(r, g), b) > 120) & (
            np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 60
        )
        return yellow, green, magenta, white

    @staticmethod
    def clusters(xs, ys, gap=0.6):
        if len(xs) == 0:
            return []
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        out = []
        start = 0
        for i in range(1, len(xs) + 1):
            if i == len(xs) or xs[i] - xs[i - 1] > gap:
                if i - start >= 3:
                    out.append(
                        (
                            float(xs[start]),
                            float(np.median(ys[start:i])),
                            i - start,
                            float(ys[start:i].max() - ys[start:i].min()),
                        )
                    )
                start = i
        return out

    def decide(self, image, state, dt=0.09):
        im = np.asarray(image)[:, :, :3]
        own = state.get("own", {})
        side = own.get("grid_side", own.get("grid"))
        lane = 0.38 if side == "inner" else -0.38
        self.travel += max(0.0, float(own.get("forward_speed", 0.0))) * dt
        yellow, green, magenta, white = self.masks(im)
        valid = self.valid
        # the centre line: stripe pixels, else the middle of the road in each row
        m = yellow & valid & (self.gx < 3.2)
        xs, ys = self.gx[m], self.gy[m]
        if len(xs) < 12:
            road = (green | yellow) & valid & (self.gx < 3.0)
            xs, ys = [], []
            for v in np.nonzero(road.sum(1) > 4)[0]:
                cols = np.nonzero(road[v])[0]
                if cols[0] > 0 and cols[-1] < self.W - 1:  # both road edges in view
                    c = (cols[0] + cols[-1]) // 2
                    xs.append(self.gx[v, c])
                    ys.append(self.gy[v, c])
            xs, ys = np.array(xs), np.array(ys)
        if len(xs) >= 6:
            A = np.column_stack([np.ones_like(xs), xs, xs * xs])
            coef = np.linalg.lstsq(A, ys, rcond=None)[0]
            if abs(coef[2]) > 1.2:
                coef = np.array(list(np.linalg.lstsq(A[:, :2], ys, rcond=None)[0]) + [0.0])
            self.fit = np.array(coef)
            self.lost = 0
        else:
            self.lost += 1
            if self.fit is None or self.lost > 6:
                # nothing to follow: turn the way the road last went, slowly
                return [0.7, float(np.clip(self.turn * 0.48 if self.turn else 0.3, -0.48, 0.48))]
        a, b, c = self.fit
        centre = lambda x: a + b * x + c * x * x  # noqa: E731
        # obstacles (magenta): bottom edge pixels -> ground points
        mg = magenta & valid
        edge = mg & ~np.vstack([mg[1:], np.zeros((1, self.W), bool)])
        for x0, y0, _, _ in self.clusters(self.gx[edge], self.gy[edge]):
            if not 0.2 < x0 < 4.5:
                continue
            lo = y0 - centre(x0)  # obstacle lateral relative to the centre line
            want = min(lane, lo - 0.78) if lo > 0 else max(lane, lo + 0.78)
            want = float(np.clip(want, -0.95, 0.95))
            if abs(want - lane) > 0.05 and (self.detour is None or x0 < self.detour[2]):
                self.detour = (want, self.travel + x0 + 0.9, x0)
        if self.detour and self.travel > self.detour[1]:
            self.detour = None
        target_lat = self.detour[0] if self.detour else lane
        speed = FAST
        # other cars: object pixels on the road
        road = green | yellow
        any_road = road.any(1)
        idx = np.arange(self.W)
        first = np.where(any_road, np.argmax(road, 1), self.W)
        last = np.where(any_road, self.W - 1 - np.argmax(road[:, ::-1], 1), -1)
        inside = (idx[None, :] > first[:, None]) & (idx[None, :] < last[:, None])
        obj = inside & ~road & ~magenta & ~white & valid & (self.gx < 3.5)
        # cars are wide blobs: keep pixels with object pixels 3 columns to both sides (drops stripe-edge blends)
        obj = obj & np.roll(obj, 3, 1) & np.roll(obj, -3, 1)
        oedge = obj & ~np.vstack([obj[1:], np.zeros((1, self.W), bool)])
        for x0, y0, n, spread in self.clusters(self.gx[oedge], self.gy[oedge], gap=0.4):
            # a car is at most about 0.6 m wide; a wider band across the road is paint (the start/finish line)
            if n < 5 or not 0.25 < x0 < 3.0 or spread > 1.0:
                continue
            lc = y0 - centre(x0)
            if abs(lc - target_lat) < 0.62:
                speed = min(speed, max(0.0, 2.4 * (x0 - 0.75)))
                sides = [x for x in (lc + 0.82, lc - 0.82) if abs(x) <= 0.9]
                if sides and not self.detour:
                    target_lat = min(sides, key=lambda x: abs(x - target_lat))
        L = 1.15
        ys0 = centre(L)
        slope = b + 2 * c * L
        nrm = math.hypot(1.0, slope)
        tx, ty = L - target_lat * slope / nrm, ys0 + target_lat / nrm
        steer = float(np.clip(math.atan2(2 * WHEELBASE * ty, tx * tx + ty * ty), -0.48, 0.48))
        self.turn = math.copysign(1.0, steer) if abs(steer) > 0.15 else self.turn * 0.9
        curv = abs(2 * c) + abs(steer) * 0.5
        speed = min(speed, FAST if curv < 0.12 else MID if curv < 0.25 else SLOW)
        return [float(speed), steer]


def front_image(view):
    """The front camera tile of the observation (one PNG of tiles; the default observation has the front tile only)."""
    obs = view.get("observation") or {}
    b64 = obs.get("image_png_b64")
    if not b64:
        return None
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    for t in obs.get("tiles") or []:
        if t.get("camera") == "front":
            return im.crop((t["x"], t["y"], t["x"] + t["width"], t["y"] + t["height"]))
    return im


class Reference:
    """The camera-only reference (sport.toml [players] reference): the legacy lane follower over the core's views."""

    def __init__(self, spec, seat, seed=0):
        self.seat, self.player = seat, Player(spec)
        self.heat, self.t = None, None

    def act(self, view):
        st = view.get("state") or {}
        own = st.get("own") or {}
        if own.get("retired") or st.get("race_over") or st.get("awaiting_review"):
            return None
        if st.get("heat") != self.heat:
            self.heat, self.t = st.get("heat"), None
            self.player.reset()
        image = front_image(view)
        if image is None:
            return None
        t = float(st.get("t_sim") or 0.0)
        dt = 0.05 if self.t is None else t - self.t
        self.t = t
        speed, steer = self.player.decide(image, st, dt=min(0.3, max(0.01, dt)))
        return {"type": "drive", "speed": round(speed, 6), "steering": round(steer, 6)}

    def review(self, reply):
        return ("The heat reached its finish or time limit.", "Continue visual lane following from the new grid slot.")


class Random:
    """A check policy: random targets within the limits (it rarely finishes a heat)."""

    def __init__(self, spec, seat, seed=0):
        self.rng = random.Random(f"racing-random:{seat}:{seed}")
        self.seat = seat

    def act(self, view):
        st = view.get("state") or {}
        if (st.get("own") or {}).get("retired") or st.get("race_over") or st.get("awaiting_review"):
            return None
        return {
            "type": "drive",
            "speed": round(self.rng.uniform(-1.5, 4.0), 3),
            "steering": round(self.rng.uniform(-0.48, 0.48), 3),
        }
