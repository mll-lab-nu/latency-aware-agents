"""fencing's built-in players (the same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  the camera-only reference fencer (the legacy solution/fencer.py): finds the opponent (the blue robot) in
             the side and top camera images, estimates where its cart and its sword arm are and how fast it comes.
             Two styles:
               defender  keeps its distance and stop-hits (retreating) an opponent who comes into distance
               attacker  closes the distance and thrusts at the opponent's sword arm, recovers and attacks again
             Single format: defender. Duel: the style alternates with the seat and the player seed (seed 0, the live
             default: seat A attacks and seat B defends, as the legacy duel oracle), so reference against reference
             is decided by touches. High level: point poses (act / move); low level: the same decisions as joint
             goals, with its own IK on the published kinematic chain (`game spec` kinematics).
  Random     random valid actions (attacks at random points, random rail goals).
Uses only the observation image, the robot state in the replies and `spec` (cameras, kinematics); no simulator
import (numpy + pillow): this file runs in the player's container as the task's oracle."""
from __future__ import annotations

import base64
import io

import numpy as np

HAND = np.array([0.50, -0.05, 1.15])     # about where the guard is in the on-guard pose (cart frame)
REACH = 1.72                              # comfortable reach of the point from the cart (u, cart frame)
GUARD_TIP, GUARD_DIR = [1.40, -0.05, 1.10], [1.0, 0.0, -0.05]


class Cam:
    def __init__(self, c):
        self.pos = np.asarray(c["pos"], float)
        self.f = np.asarray(c["forward"], float)
        self.r = np.asarray(c["right"], float)
        self.u = np.asarray(c["up"], float)
        self.focal = float(c["focal_px"])
        self.cx, self.cy = c["principal_point_px"]
        self.w, self.h = c["tile_px"]

    def rays(self, us, vs):
        return (self.f[None] + ((us - self.cx) / self.focal)[:, None] * self.r[None]
                - ((vs - self.cy) / self.focal)[:, None] * self.u[None])

    def on_plane(self, us, vs, axis, value):
        """points where the pixels' rays meet the plane coordinate[axis] = value"""
        d = self.rays(us.astype(float) + 0.5, vs.astype(float) + 0.5)
        t = (value - self.pos[axis]) / d[:, axis]
        return self.pos[None] + t[:, None] * d


def decode(png):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def blue_mask(t):
    r, g, b = (t[..., i].astype(int) for i in range(3))
    return (b > 140) & (b - r > 55) & (b - g > 25)


# ------------------------------------------------------------------------------------------------ arm kinematics
def _quat_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _axis_mat(a, th):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Arm:
    """Forward and inverse kinematics of the arm from `spec` kinematics (cart frame)."""

    def __init__(self, kin):
        self.chain = kin["chain"]
        self.point = np.asarray(kin["point_in_epee_frame"], float)
        rng = [e["range"] for e in self.chain if "joint" in e and e["joint"].startswith("joint")]
        self.lo, self.hi = np.array([r[0] for r in rng]) + 1e-3, np.array([r[1] for r in rng]) - 1e-3

    def fk(self, q):
        R, p, i = np.eye(3), np.zeros(3), 0
        for e in self.chain:
            p = p + R @ np.asarray(e["pos"], float)
            R = R @ _quat_mat(e["quat"])
            if "joint" in e and e["joint"].startswith("joint"):
                R = R @ _axis_mat(e["axis"], q[i])
                i += 1
        return p + R @ self.point, R[:, 2]

    def ik(self, pos, direction, q0, iters=80):
        dt = np.asarray(direction, float)
        dt = dt / np.linalg.norm(dt)
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)

        def err(q):
            p, a = self.fk(q)
            return np.r_[pos - p, 0.5 * np.cross(a, dt)]
        e = err(q)
        lam = 1e-2
        for _ in range(iters):
            c = float(e @ e)
            if c < 1e-8:
                break
            J = np.zeros((6, 7))
            for j in range(7):
                dq = np.zeros(7)
                dq[j] = 1e-6
                J[:, j] = (e - err(q + dq)) / 1e-6
            step = np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ e)
            qn = np.clip(q + step, self.lo, self.hi)
            en = err(qn)
            if en @ en < c:
                q, e, lam = qn, en, max(1e-5, lam * 0.5)
            else:
                lam *= 4.0
                if lam > 1e6:
                    break
        return q, float(np.sqrt(e @ e))


# ------------------------------------------------------------------------------------------------ the reference
class Reference:
    def __init__(self, spec, seat, seed=0):
        single = "opponent_level" in spec
        self.style = "defender" if single else ("attacker", "defender")[(int(seed) + (seat == "B")) % 2]
        cams = {c["name"]: Cam(c) for c in spec["cameras"]}
        self.side, self.top = cams["side"], cams["top"]
        self.low = spec.get("action_level") == "low"
        self.arm = Arm(spec["kinematics"]) if self.low and "kinematics" in spec else None
        self.q_guard = (spec.get("robot") or {}).get("on_guard", {}).get("q")
        self.st, self.t0, self.n, self.halts = "wait", 0.0, 0, 0
        s = (self.side.h / 360.0) ** 2            # pixel counts tuned on the legacy 800 x 360 tiles
        self.min_px, self.min_pts = max(8, int(30 * s)), max(3, int(5 * s))
        self.min_top, self.min_near = max(4, int(10 * s)), max(3, int(5 * s))

    def tiles(self, img, row, col):
        w, h = self.side.w, self.side.h
        return img[row * h:(row + 1) * h, col * w:(col + 1) * w]

    def see(self, img, col):
        """the opponent in frame `col`: cart x, the front of its sword arm (x, z) (own frame), or None"""
        m = blue_mask(self.tiles(img, 0, col))
        vs, us = np.nonzero(m)
        if len(us) < self.min_px:
            return None
        p = self.side.on_plane(us, vs, 1, 0.0)
        cart = p[p[:, 2] < 0.2]
        arm = p[(p[:, 2] > 0.8) & (p[:, 2] < 1.6)]
        if len(cart) < self.min_pts or len(arm) < self.min_pts:
            return None
        xf = float(np.percentile(arm[:, 0], 3))
        front = arm[arm[:, 0] < xf + 0.08]
        return dict(cart=float(np.median(cart[:, 0])), fx=xf, fz=float(np.median(front[:, 2])))

    def side_y(self, img, fx, fz):
        """the sword arm's lateral position: the blue pixels of the top camera whose rays meet the plane z = fz just
        behind the arm's front (fx, from the side camera)"""
        t = self.tiles(img, 1, 3)
        m = blue_mask(t) & (t[..., 0] > 75)          # the light blue robot body, not the dark blue cart
        vs, us = np.nonzero(m)
        if len(us) < self.min_top:
            return 0.1
        p = self.top.on_plane(us, vs, 2, fz)
        near = p[np.abs(p[:, 0] - fx - 0.05) < 0.08]
        return float(np.median(near[:, 1])) if len(near) >= self.min_near else 0.1

    def act(self, view):
        now, obs = view["state"], view.get("observation") or {}
        if not obs.get("image_png_b64") or "robot" not in now:
            return None
        if now.get("halts", 0) != self.halts:
            self.halts, self.st = now["halts"], "wait"
        img = decode(base64.b64decode(obs["image_png_b64"]))
        times = obs.get("view_sim_times") or [0.0, 0.0, 0.0, 0.06]
        dec = self.decide(now, img, float(now.get("fencing_time_s", 0.0)), max(0.0, times[-1] - times[0]))
        return self.lower(dec, now) if (dec and self.low) else dec

    def decide(self, now, img, t, span):
        me = now["robot"]
        x = me["cart_x"]
        a, b = self.see(img, 0), self.see(img, 3)
        if b is None:
            return None
        v = (b["cart"] - a["cart"]) / span if (a is not None and span > 0) else 0.0     # < 0: coming
        gap = b["cart"] - x
        if self.st == "hit":
            if t - self.t0 > 0.8:
                self.st, self.t0 = "back", t
                return dict(type="act", t=0.0, tip=GUARD_TIP, dir=GUARD_DIR, rail=[max(-6.0, x - 0.8), 2.0])
            return None
        if self.st == "back":
            if t - self.t0 > 0.5:
                self.st = "wait"
            return None
        u_now = b["fx"] - x
        if self.style == "attacker":
            if u_now <= REACH + 0.1:
                return self._thrust(img, b, x, v, t, rail=[min(b["cart"] - 1.8, 6.0), 2.0])
            if abs(me["rail_goal"] - (b["cart"] - 2.1)) > 0.1:
                return dict(type="move", rail=[float(np.clip(b["cart"] - 2.1, -6.0, 6.0)), 2.0])
            return None
        u_pred = u_now + min(0.0, v) * 0.3
        if (u_pred <= REACH and v < -0.2) or u_now <= REACH - 0.05:
            return self._thrust(img, b, x, v, t, rail=[max(-6.0, x - 0.5), 2.0])
        if gap > 4.3 and abs(me["rail_goal"] - (b["cart"] - 4.0)) > 0.1:
            return dict(type="move", rail=[max(-6.0, b["cart"] - 4.0), 1.0])
        return None

    def _thrust(self, img, b, x, v, t, rail):
        """the point at the opponent's sword arm (led by its speed), the blade from the hand towards it"""
        y = self.side_y(img, b["fx"], b["fz"])
        tgt = np.array([b["fx"] + 0.05 + min(0.0, v) * 0.25 - x, y, b["fz"]])
        d = tgt - HAND
        d /= np.linalg.norm(d)
        tip = tgt + 0.05 * d
        if tip[0] > REACH + 0.06:
            tip = HAND + d * (REACH + 0.06 - HAND[0]) / max(d[0], 0.3)
        tip = [float(np.clip(tip[0], 0.0, 2.0)), float(np.clip(tip[1], -0.8, 0.8)), float(np.clip(tip[2], 0.1, 2.0))]
        self.st, self.t0 = "hit", t
        self.n += 1
        return dict(type="act", t=0.0, tip=tip, dir=[float(x) for x in d], rail=[float(rail[0]), float(rail[1])])

    def lower(self, dec, now):
        """the low level: the same decision as joint goals (own IK on the published chain)"""
        if dec["type"] == "move":
            return {"type": "move", "x": dec["rail"][0], "vmax": dec["rail"][1]}
        if dec["tip"] == GUARD_TIP and self.q_guard:
            q = self.q_guard
        else:
            q_now = np.asarray(now["robot"]["q"], float)
            best = None
            for q0 in (q_now, np.asarray(self.q_guard or q_now, float)):
                q, r = self.arm.ik(np.asarray(dec["tip"]), np.asarray(dec["dir"]), q0)
                if best is None or r < best[1]:
                    best = (q, r)
            q = [float(v) for v in best[0]]
        return {"type": "joints", "q": [float(v) for v in q], "t": float(dec["t"]), "rail_x": dec["rail"][0],
                "rail_vmax": dec["rail"][1]}

    def review(self, reply):
        return "reference: phrase over", f"go on as the {self.style}"


class Random:
    """Random valid actions: now and then an attack at a random point ahead, or a random rail goal."""

    def __init__(self, spec, seat, seed=0):
        self.low = spec.get("action_level") == "low"
        self.q_guard = (spec.get("robot") or {}).get("on_guard", {}).get("q")
        self.g = np.random.default_rng(int(seed) * 7919 + 17)        # by seed, not by seat: mirror runs match

    def act(self, view):
        r = self.g.random()
        if r < 0.85:
            return None
        x = float(view["state"].get("robot", {}).get("cart_x", -2.0))
        goal = float(np.clip(x + self.g.uniform(-0.8, 1.2), -6.0, 6.0))
        if r < 0.93:
            return {"type": "move", "x": goal, "vmax": 2.0} if self.low else {"type": "move", "rail": [goal, 2.0]}
        if self.low:
            q = np.asarray(self.q_guard or [0.94, 0.75, -1.5, -2.5, 1.34, 0.84, -1.28], float)
            return {"type": "joints", "q": [float(v) for v in q + self.g.uniform(-0.3, 0.3, 7) * [1, 1, 1, 0.5, 1, 1, 1]],
                    "t": float(self.g.uniform(0, 0.5)), "rail_x": goal, "rail_vmax": 2.0}
        tip = [float(self.g.uniform(1.2, 1.9)), float(self.g.uniform(-0.3, 0.3)), float(self.g.uniform(0.6, 1.5))]
        return {"type": "act", "t": float(self.g.uniform(0, 0.5)), "tip": tip, "dir": [1.0, 0.0, -0.05],
                "rail": [goal, 2.0]}
