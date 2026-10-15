"""Bowling's built-in players (the same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  camera-only: reads the lane view from above (the yellow dots of its previous ball's path and the pins
             standing on the deck; pixels -> lane with the camera definitions of /spec) and the decision point.
             Strike balls: a hook line (board 29.5, 8.5 m/s, 450 rpm, axis 60 deg) whose launch angle it corrects
             after every first ball by where the ball arrived at the head pin, against the 1-3 pocket. Spare balls:
             a straight ball (no revs) at the front standing pin, offset towards the others, cross-lane for the
             outside pins. Proves the sport is playable from the images alone (sport rule 9).
  Random     uniform random valid rolls (generic tests); seeded by the seed only (mirror runs match).
No simulator import: this file runs in the player's container as the task's oracle (numpy + pillow only)."""
from __future__ import annotations

import base64
import io
import math

import numpy as np

POCKET_Y = -0.075            # the 1-3 pocket for a ball that hooks left (m, + = left)
HOOK = {"speed": 8.5, "rpm": 450.0, "axis_deg": 60.0}
STRIKE_BOARD = 29.5
SLOPE_M_PER_DEG = 0.3        # how far the arrival at the head pin moves per degree of launch angle (assumed)


def blobs(mask, min_px=3):
    """Centroids (u, v, n) of the 4-connected blobs of a boolean mask (pixel centres at i + 0.5); row runs joined
    by union-find, so a large mask stays cheap."""
    runs, parent = [], []                                # run: (row, start, end); parent: union-find over runs

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
        s = acc.setdefault(find(i), [0.0, 0.0, 0.0])
        s[0] += n * ((a + b - 1) / 2 + 0.5)
        s[1] += n * (y + 0.5)
        s[2] += n
    return [(su / n, sv / n, int(n)) for su, sv, n in acc.values() if n >= min_px]


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec = spec
        lane = spec["lane"]
        self.head_x, self.lane_w = lane["length_foul_line_to_head_pin_m"], lane["width_m"]
        self.board_w, self.n_boards = lane["board_width_m"], lane["boards"]
        cam = next(c for c in spec["cameras"] if c["name"] == "lane")
        self.panels = cam["panels"]
        self.panel_w = cam["size_px"][0]
        self.f = (self.panels[0]["rows_px"][1] - self.panels[0]["rows_px"][0]) / 2 / math.tan(
            math.radians(cam["fovy_deg_per_panel"] / 2))
        self.angle = -4.0                                    # the strike line's launch angle, corrected by feedback
        self.fixes: list[tuple[float, float]] = []           # (executed angle, arrival y at the head pin)
        self.last_kind = None

    # ---------------------------------------------------------------- images -> lane coordinates
    def lane_image(self, view):
        from PIL import Image
        obs = view["observation"]
        tile = next(t for t in obs["tiles"] if t["camera"] == "lane")
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["image_png_b64"]))).convert("RGB")).astype(int)
        return img[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]]

    def to_lane(self, u, v, z=0.0):
        """Pixel (u, v) of the lane view -> (x, y) on the plane at height z (the drawn path: the camera height)."""
        for p in self.panels:
            r0, r1 = p["rows_px"]
            if r0 <= v < r1:
                cx, _, cz = p["camera_pos_m"]
                depth = cz - z
                return cx + (u - self.panel_w / 2) * depth / self.f, ((r0 + r1) / 2 - v) * depth / self.f
        return None

    def path(self, img):
        """The previous ball's path (yellow dots) as lane points, sorted along the lane."""
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        pts = []
        for u, v, _n in blobs((r > 235) & (g > 195) & (g < 245) & (b < 60)):
            p = self.to_lane(u, v, 0.0)                      # the dots are drawn as seen at the camera's depth
            if p is not None:
                pts.append(p)
        return sorted(pts)

    def pins(self, img):
        """Standing pins on the deck (lane points), from the white pin silhouettes in the last panel."""
        p = self.panels[-1]
        r0, r1 = p["rows_px"]
        sub = img[r0:r1]
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        white = (r > 175) & (g > 175) & (b > 165) & (np.max(sub, axis=2) - np.min(sub, axis=2) < 45)
        white[:16, :] = False                                # the panel's caption and separator
        out = []
        for u, v, n in blobs(white, min_px=25):
            q = self.to_lane(u, v + r0, 0.15)                # a silhouette's centroid is about 15 cm up the pin
            if q and q[0] > self.head_x - 0.5 and abs(q[1]) < self.lane_w / 2 + 0.05:
                out.append((q[0], q[1], n))
        return out

    # ---------------------------------------------------------------- decisions
    def board_of(self, y):
        return min(self.n_boards, max(1.0, (y + self.lane_w / 2) / self.board_w + 0.5))

    def learn(self, img, last):
        """After a strike ball: the executed launch angle and where the ball arrived at the head pin."""
        pts = self.path(img)
        early = [(x, y) for x, y in pts if 0.4 < x < 6.0]
        late = [(x, y) for x, y in pts if 13.0 < x < self.head_x - 0.3]
        if len(early) < 3 or len(late) < 3 or last.get("gutter"):
            return
        xs, ys = np.array(early).T
        c = np.polyfit(xs, ys, 1 if len(early) < 5 else 2)
        a_exec = math.degrees(math.atan(np.polyval(np.polyder(c), 0.35)))
        xs, ys = np.array(late).T
        y_at_pins = float(np.polyval(np.polyfit(xs, ys, 2), self.head_x))
        self.fixes.append((a_exec, y_at_pins))
        # the angle that would have arrived in the pocket, averaged over the recent strike balls
        est = [a + (POCKET_Y - y) / SLOPE_M_PER_DEG for a, y in self.fixes[-4:]]
        w = np.arange(1, len(est) + 1, dtype=float)
        self.angle = float(np.clip(np.average(est, weights=w), -9.0, 9.0))

    def act(self, view):
        st = view["state"]
        dec = st.get("decision") or {}
        img = self.lane_image(view)
        last = dec.get("last_ball")
        if last and self.last_kind == "strike":
            self.learn(img, last)
        if dec.get("fresh_rack", True):
            self.last_kind = "strike"
            return {"type": "roll", "board": STRIKE_BOARD, "angle_deg": round(self.angle, 4), **HOOK}
        self.last_kind = "spare"
        return self.spare(self.pins(img))

    def spare(self, pins):
        if not pins:
            return {"type": "roll", "board": 20.0, "angle_deg": 0.0, "speed": 8.0, "rpm": 0.0, "axis_deg": 0.0}
        pins = sorted(pins)
        fx, fy, _ = pins[0]
        rest = [y for _x, y, _n in pins[1:]]
        ty = fy + (float(np.clip(np.mean(rest) - fy, -0.08, 0.08)) if rest else 0.0)
        lim = self.lane_w / 2 - 0.13                          # keep the ball's centre well inside the lane
        ty = float(np.clip(ty, -lim + 0.04, lim - 0.04))
        y0 = ty if abs(ty) < 0.15 else -math.copysign(0.12, ty)   # outside pins: cross the lane
        a = math.degrees(math.atan((ty - y0) / (fx - 0.35)))
        return {"type": "roll", "board": round(self.board_of(y0), 4), "angle_deg": round(a, 4), "speed": 8.0,
                "rpm": 0.0, "axis_deg": 0.0}

    def review(self, reply):
        return ("measured where my last ball arrived at the pins (lane view) and which pins stand",
                "correct the strike line's launch angle towards the pocket; shoot spares straight")


class Random:
    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 23)   # by seed

    def act(self, view):
        return {"type": "roll", "board": float(self.g.uniform(8, 32)), "angle_deg": float(self.g.uniform(-3, 3)),
                "speed": float(self.g.uniform(5, 10)), "rpm": float(self.g.uniform(0, 500)),
                "axis_deg": float(self.g.uniform(-60, 60))}
