"""Built-in baseball players (contract ②): they see what a model's program sees (the camera images, the state of
the reply, /spec) and import no simulator. factory(spec, seat, seed) -> object with act(view) -> body | None.

Reference: the Season-1 reference player (solution/bat_player.py and the duel's oracle_play.py of the legacy tasks).
  batting   camera-only: the ball is the bright white blob that differs from a background tile of the same camera
            taken during the windup; the flight from the known release point under gravity, drag and Magnus lift is
            fitted to the detections of both cameras (release velocity for each documented pitch type's spin, damped
            Gauss-Newton on the reprojection error); if the predicted crossing is a strike (or close to one with two
            strikes) the rig parameters that put the sweet spot at the ball near square, the ball 22 mm above the bat
            axis (a line drive), else a take. The swing command is relative to the step at which it takes effect:
            the player learns that delay from its own commands (the planned swing start the state reports) and
            corrects a command whose planned start is off by 2 ms or more; in-process (core/local.py) the delay is 0.
  pitching  (duel) a fixed mix near the top of each speed band, about two thirds aimed inside the strike zone, the
            rest just off an edge; its choices depend only on the pitch number and the count.
Rival: the reference with seeded aim errors of its own (the fairness policy).
Random: random swings and random pitches (the generic tests).
"""
from __future__ import annotations

import base64
import io
import math
import random

import numpy as np
from PIL import Image

G = 9.81
K_D, K_L0, R_BALL, SPIN_TAU = 0.00601, 0.01718, 0.0365, 30.0
REL = (-0.50, 16.80, 1.80)
ZONE_X, ZONE_LO, ZONE_HI = 0.216, 0.46, 1.07
PIVOT_X0, PIVOT_Y, S_SS = -0.66, 0.25, 0.69
TH0, TH1 = -math.pi, -math.pi / 6
T_SQUARE = (5 * math.pi / 3 + math.pi / 6) * S_SS          # time to square = T_SQUARE / SPEED
TILE_W, TILE_H = 480, 360
THETA_C, DELTA, PLANE, SPEED = 4.0, 0.022, 4.0, 34.0      # the swing recipe
T_FIT, T_ACT = 0.09, 0.18                                  # first fit / first swing command (s after release)
MIX = [("fastball", .45), ("slider", .20), ("changeup", .20), ("curveball", .15)]


# ------------------------------------------------------------------------------------------------- ball model
def c_lift(S):
    return 1.5 * S if S < 0.1 else 0.09 + 0.6 * S


def acc(v, w):
    vx, vy, vz = v
    sp = math.sqrt(vx * vx + vy * vy + vz * vz)
    a = [-K_D * sp * vx, -K_D * sp * vy, -G - K_D * sp * vz]
    wm = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    if wm > 0 and sp > 0:
        k = K_L0 * c_lift(R_BALL * wm / sp) * sp / wm
        a[0] += k * (w[1] * vz - w[2] * vy)
        a[1] += k * (w[2] * vx - w[0] * vz)
        a[2] += k * (w[0] * vy - w[1] * vx)
    return a


def rk4(p, v, w, h):
    a1 = acc(v, w)
    v2 = [v[i] + 0.5 * h * a1[i] for i in range(3)]
    a2 = acc(v2, w)
    v3 = [v[i] + 0.5 * h * a2[i] for i in range(3)]
    a3 = acc(v3, w)
    v4 = [v[i] + h * a3[i] for i in range(3)]
    a4 = acc(v4, w)
    p = [p[i] + h / 6 * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i]) for i in range(3)]
    v = [v[i] + h / 6 * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]) for i in range(3)]
    d = math.exp(-h / SPIN_TAU)
    return p, v, [x * d for x in w]


def positions(params, ts, h=0.01):
    """Ball centre at the times ts after release, for params = (v0, w0)."""
    v, w, p = list(params[:3]), list(params[3:]), list(REL)
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    out = [None] * len(ts)
    t = 0.0
    for i in order:
        while t + h <= ts[i]:
            p, v, w = rk4(p, v, w, h)
            t += h
        q, _, _ = rk4(p, v, w, ts[i] - t) if ts[i] > t else (p, v, w)
        out[i] = q
    return out


def path(params, t_end=0.9, h=0.001):
    """Dense flight samples (t, x, y, z) until the ball is 1.5 m behind the plate."""
    v, w, p = list(params[:3]), list(params[3:]), list(REL)
    t, out = 0.0, [(0.0, *REL)]
    while t < t_end and p[1] > -1.5 and p[2] > 0:
        p, v, w = rk4(p, v, w, h)
        t += h
        out.append((t, *p))
    return np.array(out)


# ------------------------------------------------------------------------------------------------- cameras
class Cam:
    def __init__(self, c):
        self.name = c["name"]
        self.pos = np.array(c["position"], float)
        self.f, self.up, self.r = (np.array(c[k], float) for k in ("forward", "up", "right"))
        self.focal = float(c["focal_px"])
        self.cx, self.cy = (float(x) for x in c["principal_point_px"])

    def project(self, P):
        d = np.asarray(P, float) - self.pos
        z = d @ self.f
        return np.array([self.cx + self.focal * (d @ self.r) / z, self.cy - self.focal * (d @ self.up) / z])


def components(mask):
    """Connected components (8-neighbourhood) of a boolean image, as arrays of (row, col); largest first."""
    pts = set(zip(*np.nonzero(mask)))
    out = []
    while pts:
        stack = [pts.pop()]
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.remove(q)
                        stack.append(q)
        out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def decode(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(np.int16)


def tile(img, row, col):
    return img[row * TILE_H:(row + 1) * TILE_H, col * TILE_W:(col + 1) * TILE_W]


# ------------------------------------------------------------------------------------------------- the batter
class Batter:
    def __init__(self, cameras):
        self.cams = [Cam(c) for c in cameras]          # row 0: catcher, row 1: side (spec order)
        self.bg = [None] * len(self.cams)
        self.reset()

    def reset(self):
        self.dets = {}                                   # (cam, t) -> (u, v)
        self.params = None
        self.fit_n = 0
        self.rms_px = None

    def ingest(self, img, view_times):
        """Add the frames of one observation (decoded image, each column's time since release)."""
        for col, t in enumerate(view_times):
            if t is None:
                continue
            for row, cam in enumerate(self.cams):
                tl = tile(img, row, col)
                if t < 0:
                    self.bg[row] = tl.copy()             # the ball is still in the hand: background
                    continue
                if t < 0.025 or self.bg[row] is None or (row, round(t, 3)) in self.dets:
                    continue
                uv = self.detect(tl, row, t)
                if uv is not None:
                    self.dets[(row, round(t, 3))] = uv

    def detect(self, tl, row, t):
        """The ball's centre in a tile: a near-white blob that differs from the background, refined by the centroid
        of its whole silhouette (lit and shaded side: brighter than the background) in a small window."""
        bg = self.bg[row]
        d = np.abs(tl - bg).sum(axis=2)
        m = (tl.min(axis=2) > 185) & (d > 90)
        m[:16, :200] = False                                # the time stamp
        if not m.any():
            return None
        guess = self.guess(row, t)
        best, best_score, best_n = None, None, 0
        for pts in components(m):
            c = pts.mean(axis=0)[::-1]
            dist = 0.0 if guess is None else float(np.linalg.norm(c - guess))
            if guess is not None and dist > 150:
                continue
            score = len(pts) - 0.05 * dist
            if best_score is None or score > best_score:
                best, best_score, best_n = c, score, len(pts)
        if best is None:
            return None
        r = int(math.ceil(math.sqrt(best_n / math.pi))) + 4
        x0, y0 = max(0, int(best[0]) - r), max(0, int(best[1]) - r)
        x1, y1 = int(best[0]) + r + 1, int(best[1]) + r + 1
        brighter = (tl[y0:y1, x0:x1].sum(axis=2) - bg[y0:y1, x0:x1].sum(axis=2)) > 40
        seed = (min(int(best[1]) - y0, brighter.shape[0] - 1), min(int(best[0]) - x0, brighter.shape[1] - 1))
        for pts in components(brighter):
            if (pts == np.array(seed)).all(axis=1).any() or len(pts) >= 0.8 * best_n:
                cy, cx = pts.mean(axis=0)
                return np.array([x0 + cx, y0 + cy]) + 0.5          # pixel centres are at index + 0.5
        return best + 0.5

    def guess(self, row, t):
        p = self.params if self.params is not None else self.nominal()
        try:
            return self.cams[row].project(positions(p, [t])[0])
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def nominal(speed=36.0):
        d = np.array([0.0, 0.0, 1.0]) - np.array(REL)
        d /= np.linalg.norm(d)
        return np.concatenate([d * speed + np.array([0, 0, 1.6]), np.zeros(3)])

    def fit(self, spins, iters=10):
        """The release velocity for each candidate spin vector (the documented pitch types' spins); the candidate
        with the smallest reprojection error."""
        keys = sorted(self.dets, key=lambda k: k[1])
        if len(keys) < 4:
            return None
        ts = [k[1] for k in keys]
        meas = np.array([self.dets[k] for k in keys]).ravel()
        cams = [self.cams[k[0]] for k in keys]
        best = None
        for w in spins:
            w = np.asarray(w, float)

            def resid(v):
                P = positions(np.concatenate([v, w]), ts)
                return np.concatenate([c.project(x) for c, x in zip(cams, P)]) - meas

            v = np.array(self.params[:3] if self.params is not None else self.nominal()[:3], float)
            r = resid(v)
            lam = 1e-3
            for _ in range(iters):
                J = np.stack([(resid(v + 0.01 * np.eye(3)[i]) - r) / 0.01 for i in range(3)], axis=1)
                A, g = J.T @ J, J.T @ r
                ok = False
                step = np.zeros(3)
                while lam < 1e6:
                    step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
                    rq = resid(v + step)
                    if rq @ rq < r @ r:
                        v, r, lam, ok = v + step, rq, max(lam / 3, 1e-7), True
                        break
                    lam *= 5
                if not ok or np.abs(step).max() < 1e-4:
                    break
            cost = float(r @ r)
            if best is None or cost < best[0]:
                best = (cost, np.concatenate([v, w]))
        self.params, self.fit_n = best[1], len(keys)
        self.rms_px = float(math.sqrt(best[0] / len(meas)))
        return self.params

    def crossing(self):
        """(t, x, z) where the predicted ball centre crosses y = 0."""
        P = path(self.params)
        idx = np.nonzero(P[:, 2] <= 0.0)[0]
        if not len(idx):
            return None
        j = int(idx[0])
        a, b = P[j - 1], P[j]
        f = a[2] / (a[2] - b[2])
        return tuple(a[[0, 1, 3]] + f * (b[[0, 1, 3]] - a[[0, 1, 3]]))

    def swing(self, strikes=0):
        """None (take) or dict(start, plane, height, speed, reach), the swing start in seconds since release."""
        c = self.crossing()
        if c is None:
            return None
        _, x, z = c
        m = -0.03 if strikes == 2 else 0.0                     # with two strikes, protect the edges
        if not (abs(x) <= ZONE_X + R_BALL - m and ZONE_LO - R_BALL + m <= z <= ZONE_HI + R_BALL - m):
            return None
        return self.geometry()

    def geometry(self, theta_c=THETA_C, delta=DELTA, plane=PLANE, speed=SPEED):
        P = path(self.params)
        a = math.radians(plane)
        ca, sa = math.cos(a), math.sin(a)
        th = math.radians(theta_c)
        px = PIVOT_X0
        tc = H = None
        for _ in range(6):
            ang = np.arctan2((P[:, 2] - PIVOT_Y) * ca, P[:, 1] - px)
            j = int(np.argmax(ang <= th)) if (ang <= th).any() else None
            if j is None or j == 0:
                return None
            f = (ang[j - 1] - th) / (ang[j - 1] - ang[j])
            tc, bx, by, bz = P[j - 1] + f * (P[j] - P[j - 1])
            H = bz - (delta + sa * (by - PIVOT_Y)) / ca
            e2 = ca * (by - PIVOT_Y) + sa * (bz - H)
            px_new = bx - (S_SS - math.sin(th) * e2) / math.cos(th)
            px_new = min(max(px_new, PIVOT_X0 - 0.2), PIVOT_X0 + 0.2)
            done = abs(px_new - px) < 1e-5
            px = px_new
            if done:
                break
        w = speed / S_SS
        tau1 = 2 * (TH1 - TH0) / w
        start = tc - (tau1 + (th - TH1) / w)
        if not (0.30 <= H <= 1.40):
            return None
        return {"start": float(start), "plane": plane, "height": round(float(H), 4), "speed": speed,
                "reach": round(float(px - PIVOT_X0), 4)}


def arsenal_of(spec):
    m = spec.get("match", {})
    return (m.get("pitcher") or m.get("pitching") or {}).get("arsenal", {})


# ------------------------------------------------------------------------------------------------- the players
class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec, self.seat = spec, seat
        self.spins = [[a["spin_rad_s"] * x for x in a["spin_axis"]] for a in arsenal_of(spec).values()]
        self.B = Batter(spec["cameras"])
        self.lat = None                  # game seconds from the state a command was computed on to its effect
        self.lats: list[float] = []
        self.key = None
        self.new_pitch(None)

    def new_pitch(self, key):
        self.key = key
        self.B.reset()
        self.sent = None                 # the swing command in force: {"T", "t_view", "want", "got"}
        self.finished = False            # nothing more to do this pitch
        self.pitched = 0                 # pitch commands sent this pitch (duel)

    # --- the delay between a command's state and its effect, learnt from the planned swings the state reports
    def learn(self, st, t_now):
        cmd = st.get("command")
        if self.sent is None or "got" in self.sent or not isinstance(cmd, dict) or "swing_start_in_s" not in cmd:
            return
        got = t_now + float(cmd["swing_start_in_s"])
        tsq = T_SQUARE / float(cmd.get("speed", SPEED))
        t_eff = got - self.sent["T"] + tsq
        if got > t_eff + 1e-6:                             # not clamped (a swing planned in the past starts at once)
            self.lats.append(max(0.0, t_eff - self.sent["t_view"]))
            self.lats = self.lats[-8:]
            self.lat = float(np.median(self.lats))
        self.sent["got"] = got

    def delay(self, ob):
        if self.lat is not None:
            return self.lat
        age = ob.get("obs_age_wall_s")                    # live only (in-process the observation has no wall times)
        period = (self.spec.get("timing") or {}).get("period_s")
        if age is None or not period:
            return 0.0
        return round((1.5 * float(age) + 0.15) / float(period)) * 0.01

    def act(self, view):
        st, ob = view.get("state") or {}, view.get("observation") or {}
        if st.get("role") == "pitching":
            return self.pitch(st)
        key = st.get("pitch")
        if key != self.key:
            self.new_pitch(key)
        img = ob.get("image_png_b64")
        if img and ob.get("view_times"):
            self.B.ingest(decode(img), ob["view_times"])
        t_now = st.get("t_since_release")
        if t_now is None or st.get("pitch_phase") == "pitch over":
            return None
        self.learn(st, t_now)
        cmd = st.get("command")
        if (isinstance(cmd, dict) and "swing" in cmd) or (isinstance(cmd, str) and "taken" in cmd):
            return None                                    # the swing is under way, or the pitch is being taken
        if self.finished or t_now > 0.6 or len(self.B.dets) < 6 or t_now < T_FIT:
            return None
        self.B.fit(self.spins)
        sw = self.aim(self.B.swing(int((st.get("count") or {}).get("strikes", 0))))
        lat = self.delay(ob)
        t_eff = t_now + lat
        if sw is None:                                     # a ball (or out of reach): take it
            if t_now >= T_ACT or t_eff + 0.02 > 0.25:
                self.finished = True
                return {"type": "none"} if self.sent else None
            return None
        urgent = sw["start"] - t_eff < 0.06
        if self.sent is None and not (t_now >= T_ACT or urgent):
            return None
        if sw["start"] <= t_eff + 1e-6:
            self.finished = True                           # too late to change anything
            return None
        have = self.sent.get("got", self.sent["want"]) if self.sent else None
        if have is not None and abs(sw["start"] - have) < 0.002:
            return None
        T = sw["start"] + T_SQUARE / sw["speed"] - t_eff
        if not 0.0 < T <= 2.5:
            return None
        self.sent = {"T": round(T, 5), "t_view": t_now, "want": sw["start"]}
        return {"type": "swing", "t": round(T, 5), "plane": sw["plane"], "height": sw["height"],
                "speed": sw["speed"], "reach": sw["reach"]}

    def aim(self, sw):
        return sw

    def pitch(self, st):
        key = st.get("pitch")
        if key != self.key:
            self.new_pitch(key)
        if st.get("t_since_release") is not None or not str(st.get("pitch_phase", "")).startswith("windup") \
                or isinstance(st.get("command"), dict) or self.pitched >= 3:
            return None
        self.pitched += 1
        return choose_pitch(self.spec, st, self.pitched - 1)


def choose_pitch(spec, now, attempt=0):
    """The legacy reference pitcher (a deterministic function of the pitch number and the count)."""
    c = now["count"]
    rng = random.Random(f"{now['pitch']}-{c['balls']}-{c['strikes']}" + (f"-{attempt}" if attempt else ""))
    u, name = rng.random(), MIX[-1][0]
    for n, p in MIX:
        if u < p:
            name = n
            break
        u -= p
    lo, hi = arsenal_of(spec)[name]["speed_mps"]
    speed = hi - 1.5 * rng.random()
    if rng.random() < (0.85 if c["balls"] == 3 else 0.65):
        x, z = -0.17 + 0.34 * rng.random(), 0.52 + 0.48 * rng.random()
    else:
        side = rng.randrange(4)
        x, z = [(-0.33, 0.45 + 0.6 * rng.random()), (0.33, 0.45 + 0.6 * rng.random()),
                (-0.2 + 0.4 * rng.random(), 0.36), (-0.2 + 0.4 * rng.random(), 1.18)][side]
    return {"type": "pitch", "pitch": name, "speed": round(speed, 2), "x": round(x, 3), "z": round(z, 3)}


class Rival(Reference):
    """The reference with aim errors of its own, drawn per pitch from its seed and seat (swing start 1 sigma = ERR_T,
    pivot height ERR_H): two copies of the deterministic reference hit a home run in nearly every plate appearance
    in-process (no delay), so every duel goes to the last-resort rule; rivals measure the format between two equal,
    imperfect players (the fairness policy)."""
    ERR_T, ERR_H = 0.006, 0.012

    def __init__(self, spec, seat, seed=0):
        super().__init__(spec, seat, seed)
        self.seed = seed
        self.err = None

    def new_pitch(self, key):
        super().new_pitch(key)
        r = random.Random(f"rival-{getattr(self, 'seed', 0)}-{self.seat}-{key}")
        self.err = (r.gauss(0.0, self.ERR_T), r.gauss(0.0, self.ERR_H))

    def aim(self, sw):
        if sw is None:
            return None
        h = min(1.40, max(0.30, sw["height"] + self.err[1]))
        return dict(sw, start=sw["start"] + self.err[0], height=round(h, 4))


class Random:
    """Random swings (one per pitch, at a random time after the release) and random pitches."""

    def __init__(self, spec, seat, seed=0):
        self.spec, self.rng, self.key, self.done = spec, random.Random(f"{seed}-{seat}-random"), None, False

    def act(self, view):
        st = view.get("state") or {}
        if st.get("pitch") != self.key:
            self.key, self.done = st.get("pitch"), False
            self.when = self.rng.uniform(0.05, 0.35)
        if self.done:
            return None
        if st.get("role") == "pitching":
            if not str(st.get("pitch_phase", "")).startswith("windup"):
                return None
            self.done = True
            ars = arsenal_of(self.spec)
            name = self.rng.choice(sorted(ars))
            lo, hi = ars[name]["speed_mps"]
            return {"type": "pitch", "pitch": name, "speed": round(self.rng.uniform(lo, hi), 2),
                    "x": round(self.rng.uniform(-0.4, 0.4), 3), "z": round(self.rng.uniform(0.3, 1.3), 3)}
        t = st.get("t_since_release")
        if t is None or t < self.when or st.get("pitch_phase") == "pitch over":
            return None
        self.done = True
        if self.rng.random() < 0.3:
            return None                                    # take this one
        return {"type": "swing", "t": round(self.rng.uniform(0.05, 0.3), 4), "plane": round(self.rng.uniform(-5, 15), 2),
                "height": round(self.rng.uniform(0.5, 1.1), 3), "speed": round(self.rng.uniform(25, 34), 2),
                "reach": round(self.rng.uniform(-0.2, 0.2), 3)}
