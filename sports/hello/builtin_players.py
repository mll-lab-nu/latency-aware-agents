"""hello's built-in players (same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  camera-only reference player: finds the target and its own balls in the overhead camera (pixels ->
             ground plane with the camera definitions of /spec), aims, and corrects each throw by the error of the
             last ones. Proves the sport is winnable from the images alone (sport rule 9).
  Random     uniform random valid actions (generic tests: no draw, mirror symmetry, whitelist, determinism).
No simulator import: this file runs in the player's container as the task's oracle (numpy + pillow only)."""
from __future__ import annotations

import base64
import io
import math

import numpy as np

PITCH = 35.0
COLOUR_MASKS = {"yellow": lambda r, g, b: (r > 180) & (g > 140) & (b < 90),
                "blue": lambda r, g, b: (b > 170) & (r < 90) & (g < 160),
                "target": lambda r, g, b: (r > 150) & (g < 120) & (b < 120) & (r - g > 80)}


def basis(cam):
    pos, look = np.array(cam["pos"], float), np.array(cam["lookat"], float)
    fwd = (look - pos) / np.linalg.norm(look - pos)
    right = np.cross(fwd, np.array(cam.get("up", [0, 0, 1]), float))
    right /= np.linalg.norm(right)
    return pos, right, np.cross(right, fwd), fwd


def ground_point(cam, u, v, z):
    """Pixel (u, v) of camera `cam` (a /spec camera definition) -> the world point at height z."""
    pos, right, up, fwd = basis(cam)
    f = cam["focal_px"]
    cx, cy = cam["principal_point_px"]
    ray = fwd + (u - cx) / f * right - (v - cy) / f * up
    t = (z - pos[2]) / ray[2]
    return pos + t * ray


def blobs(mask, min_px=3):
    """Centroids (u, v) of the 4-connected blobs of a boolean mask (pixel centres at i + 0.5)."""
    seen = np.zeros_like(mask, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, px = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            px.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if len(px) >= min_px:
            ys, xs = np.array(px).T
            out.append((xs.mean() + 0.5, ys.mean() + 0.5, len(px)))
    return out


def throw_for(x, y, release, ball_r):
    """The throw (pitch 35 deg, no roll) that lands at ground point (x, y)."""
    dx, dy, h, p = x - release[0], y - release[1], release[2] - ball_r, math.radians(PITCH)
    dist = math.hypot(dx, dy)
    v = dist / math.cos(p) * math.sqrt(9.81 / (2 * (dist * math.tan(p) + h)))
    return {"type": "throw", "yaw": math.degrees(math.atan2(dy, dx)), "pitch": PITCH, "speed": min(12.0, v)}


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec, self.high = spec, spec.get("action_level") == "high"
        self.cam = next(c for c in spec["cameras"] if c["name"] == "overhead")
        self.target, self.mine, self.aims, self.errors = None, [], [], []

    def look(self, view):
        from PIL import Image
        obs = view["observation"]
        tile = next(t for t in obs["tiles"] if t["camera"] == "overhead")
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["image_png_b64"]))).convert("RGB")).astype(int)
        img = img[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]]
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        found = {k: blobs(f(r, g, b)) for k, f in COLOUR_MASKS.items() if k != "target"}
        out = {k: [ground_point(self.cam, u, v, self.spec["ball_radius_m"])[:2]
                   for u, v, _n in sorted(bl, key=lambda q: -q[2])] for k, bl in found.items()}
        # the target: the centroid of the whole painted disc (everything not grass and not a ball), so that balls
        # lying on it hardly move the estimate (they hide < 1 % of it)
        disc = ~((g > r + 12) & (g > b + 12)) & ~COLOUR_MASKS["yellow"](r, g, b) & ~COLOUR_MASKS["blue"](r, g, b)
        ys, xs = np.nonzero(disc)
        out["target"] = [ground_point(self.cam, xs.mean() + 0.5, ys.mean() + 0.5, 0.0)[:2]] if len(xs) > 50 else []
        return out

    def act(self, view):
        seen = self.look(view)
        if seen["target"]:
            self.target = seen["target"][0]
        mine = seen[view["state"]["colour"]["you"]]
        if self.aims and len(mine) > len(self.mine):                 # the newest of my balls: farthest from the old
            new = max(mine, key=lambda p: min([np.hypot(*(p - q)) for q in self.mine] or [9]))
            self.errors.append(new - self.aims[-1])
        self.mine = mine
        corr = np.mean(self.errors[-3:], axis=0) if self.errors else np.array([0.15, 0.0])
        aim = (self.target if self.target is not None else np.array([5.0, 0.0])) - corr
        self.aims.append(aim)
        if self.high:
            return {"type": "aim", "x": float(aim[0]), "y": float(aim[1])}
        t = throw_for(aim[0], aim[1], self.spec["release_point_m"], self.spec["ball_radius_m"])
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}

    def review(self, reply):
        return "measured the landing error of my last ball", "aim the next ball corrected by that error"


class Random:
    def __init__(self, spec, seat, seed=0):
        self.high = spec.get("action_level") == "high"
        self.g = np.random.default_rng(int(seed) * 7919 + 17)        # by seed, not by seat: mirror runs match

    def act(self, view):
        if self.high:
            return {"type": "aim", "x": float(self.g.uniform(3.5, 6.5)), "y": float(self.g.uniform(-1, 1))}
        return {"type": "throw", "yaw": float(self.g.uniform(-8, 8)), "pitch": float(self.g.uniform(15, 50)),
                "speed": float(self.g.uniform(5, 10))}
