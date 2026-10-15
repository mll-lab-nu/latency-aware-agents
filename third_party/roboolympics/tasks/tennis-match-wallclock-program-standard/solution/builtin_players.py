"""tennis's built-in players (the same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  the camera-only reference player of the legacy tasks (scripts/olympiad/tennis/*/solution/tn_player.py,
             its perception, flight fit and stroke choice below): it detects the ball in the 8 tiles, fits its flight
             (drag, Magnus lift, the bounce law) to the rays, chooses the contact after the bounce on its half and the
             racket face from the documented contact law, brings the base with `move` and sends `act` with T counted
             from the act's effect step; it re-acts when the prediction moves or the contact step it gets back
             (state.command) differs. Only the images, the state, /spec and the documented physics.
             Revision 7 (the players serve): it serves (Serve below: the base to the serve spot, the toss, then the
             act timed from the toss's step, corrected once from the delivered head velocity in state.command; the
             placement depends only on the half and on how many points it has served from that half, so both players
             of a duel get the same serve sequence); it receives from the side of its box (RECEIVE_SPOT), keeps a
             delivered stroke in its last COMMIT_S before the contact, and starts every point afresh.
  Random     random base moves (a check policy).
No simulator import: this file runs in the player's container as the task's oracle (numpy + Pillow only).
"""
from __future__ import annotations

import base64
import io
import math
import time

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------------------------- documented physics
G = 9.81
R = 0.0335                                   # ball radius
K_DRAG = 0.5 * 1.21 * 0.55 * math.pi * R * R / 0.0577
K_LIFT = 0.5 * 1.21 * math.pi * R * R / 0.0577
ALPHA = 0.55
E_COURT, MU_COURT = 0.76, 0.6
E_NET, MU_NET = 0.3, 0.3                  # net cord (the net's top edge)
E_RACKET, MU_RACKET, GAIN = 0.82, 0.4, 5.0
COURT_X, COURT_Y = 4.115, 11.885
NET_C, NET_P, NET_STICK, NET_POST = 0.914, 1.07, 5.029, 6.40
STANCE_OFF = np.array([0.62, 0.30])
BASE_LO, BASE_HI = np.array([-5.9, -15.4]), np.array([5.9, -0.4])
HOME = np.array([0.0, -11.4])
BASE_V, BASE_A = 4.0, 8.0
DT = 0.01
TILE_W, TILE_H = 640, 480


def net_top(x):
    return NET_C + (NET_P - NET_C) * min(abs(x), NET_STICK) / NET_STICK


def impact(v, w, n, e, mu):
    """The documented impulse law on a rigid surface with unit normal n (towards the ball); v relative to it."""
    v = np.asarray(v, float); w = np.asarray(w, float); n = np.asarray(n, float)
    vn = float(v @ n)
    if vn >= 0:
        return v.copy(), w.copy()
    r = -R * n
    u = v + np.cross(w, r)
    t = u - (u @ n) * n
    tn = float(np.linalg.norm(t))
    jn = (1.0 + e) * (-vn)
    J = jn * n
    if tn > 1e-12:
        jt = min(mu * jn, tn * ALPHA / (1.0 + ALPHA))
        J = J - jt * t / tn
    return v + J, w + np.cross(r, J) / (ALPHA * R * R)


def _cross(a, b):
    """Row-wise cross product of (N, 3) arrays (np.cross without its axis handling: the player's hot loop)."""
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=1)


def _norm(a):
    return np.sqrt(np.einsum("ij,ij->i", a, a))[:, None]


def accel(V, W):
    """Batch acceleration: V, W (N, 3)."""
    sp = _norm(V)
    a = -K_DRAG * sp * V
    a[:, 2] -= G
    c = _cross(W, V)
    cn = _norm(c)
    wn = _norm(W)
    ok = (cn > 1e-9) & (sp > 0.1)
    S = R * wn / np.maximum(sp, 1e-9)
    cl = S / (2.0 * S + 1.0)
    f = np.where(ok, K_LIFT * cl * sp * sp / np.maximum(cn, 1e-12), 0.0)
    return a + f * c


def accel1(v, w):
    """accel() for one state, in plain floats (the single-flight loops)."""
    vx, vy, vz = v
    wx, wy, wz = w
    sp = math.sqrt(vx * vx + vy * vy + vz * vz)
    ax, ay, az = -K_DRAG * sp * vx, -K_DRAG * sp * vy, -K_DRAG * sp * vz - G
    cx, cy, cz = wy * vz - wz * vy, wz * vx - wx * vz, wx * vy - wy * vx
    cn = math.sqrt(cx * cx + cy * cy + cz * cz)
    if cn > 1e-9 and sp > 0.1:
        S = R * math.sqrt(wx * wx + wy * wy + wz * wz) / max(sp, 1e-9)
        f = K_LIFT * (S / (2.0 * S + 1.0)) * sp * sp / max(cn, 1e-12)
        ax, ay, az = ax + f * cx, ay + f * cy, az + f * cz
    return np.array((ax, ay, az))


def fly(P, V, W, times, dt=0.002, bounce_court=True):
    """Batch flight from t = 0 of N states; positions at the requested times (sorted, >= 0): (N, len(times), 3), and
    the first court bounce time / position of each state (nan if none)."""
    P, V, W = P.copy(), V.copy(), W.copy()
    N = len(P)
    out = np.zeros((N, len(times), 3))
    bt = np.full(N, np.nan); bp = np.full((N, 2), np.nan)
    t, j = 0.0, 0
    while j < len(times) and times[j] <= 1e-12:
        out[:, j] = P; j += 1
    while j < len(times):
        h = min(dt, times[j] - t)
        A = accel(V, W)
        V2 = V + A * h
        P2 = P + V2 * h
        if bounce_court:
            hit = (P2[:, 2] < R) & (V2[:, 2] < 0)
            for i in np.nonzero(hit)[0]:
                f = (P[i, 2] - R) / (P[i, 2] - P2[i, 2]) if P[i, 2] != P2[i, 2] else 1.0
                f = min(max(f, 0.0), 1.0)
                c = P[i] + f * (P2[i] - P[i])
                v2, w2 = impact(V2[i], W[i], (0.0, 0.0, 1.0), E_COURT, MU_COURT)
                rem = (1.0 - f) * h
                P2[i] = [c[0] + v2[0] * rem, c[1] + v2[1] * rem, R + v2[2] * rem]
                V2[i] = v2; W[i] = w2
                if np.isnan(bt[i]):
                    bt[i] = t + f * h; bp[i] = c[:2]
        P, V = P2, V2
        t += h
        while j < len(times) and times[j] - t <= 1e-9:
            out[:, j] = P; j += 1
    return out, bt, bp


def _cord(p, q):
    """The documented net cord: where the step p -> q first brings the ball's centre within R of the net's top edge
    (y = 0, z = net top): (fraction, contact point, unit normal from the edge to the ball) or None."""
    xm = 0.5 * (p[0] + q[0])
    if abs(xm) > NET_POST:
        return None
    top = net_top(xm)
    y0, dy, z0, dz = p[1], q[1] - p[1], p[2] - top, q[2] - p[2]
    c = y0 * y0 + z0 * z0 - R * R
    a = dy * dy + dz * dz
    if c <= 0 or a <= 0:
        return None
    b = 2 * (y0 * dy + z0 * dz)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = (-b - math.sqrt(disc)) / (2 * a)
    if not 0 <= s <= 1:
        return None
    cp = p + s * (q - p)
    n = np.array([0.0, cp[1], cp[2] - top]); n /= np.linalg.norm(n)
    return s, cp, n


def trajectory(p, v, w, t_end, dt=0.001):
    """Single flight sampled every dt: times, P, V, W and court bounces [(t, x, y)] (stops at a net crossing below the
    net top, or the second bounce)."""
    p = np.asarray(p, float).copy(); v = np.asarray(v, float).copy(); w = np.asarray(w, float).copy()
    ts, Ps, Vs, Ws, bounces = [0.0], [p.copy()], [v.copy()], [w.copy()], []
    t = 0.0
    while t < t_end - 1e-9:
        a = accel1(v, w)
        v2 = v + a * dt
        p2 = p + v2 * dt
        cord = _cord(p, p2)
        if cord is not None and v2[1] * cord[2][1] + v2[2] * cord[2][2] < 0:
            s, cp, n = cord                                  # the ball touches the net's top edge and goes on
            v2, w = impact(v2, w, n, E_NET, MU_NET)
            p2 = cp + v2 * (1.0 - s) * dt
            bounces.append((t + s * dt, cp[0], cp[2], "cord"))
        elif (p[1] > 0) != (p2[1] > 0) and p[1] != p2[1]:
            f = p[1] / (p[1] - p2[1])
            xn, zn = p[0] + f * (p2[0] - p[0]), p[2] + f * (p2[2] - p[2])
            if abs(xn) <= NET_POST and zn < net_top(xn) - R:
                bounces.append((t + f * dt, xn, 0.0, "net"))
                break
        if p2[2] < R and v2[2] < 0:
            f = (p[2] - R) / (p[2] - p2[2]) if p[2] != p2[2] else 1.0
            f = min(max(f, 0.0), 1.0)
            c = p + f * (p2 - p)
            v2, w = impact(v2, w, (0.0, 0.0, 1.0), E_COURT, MU_COURT)
            rem = (1.0 - f) * dt
            p2 = np.array([c[0] + v2[0] * rem, c[1] + v2[1] * rem, R + v2[2] * rem])
            bounces.append((t + f * dt, c[0], c[1], "bounce"))
        p, v = p2, v2
        t += dt
        ts.append(t); Ps.append(p.copy()); Vs.append(v.copy()); Ws.append(w.copy())
        if sum(1 for b in bounces if b[3] == "bounce") >= 2:
            break
    return np.array(ts), np.array(Ps), np.array(Vs), np.array(Ws), bounces


def landings(P0, V0, W0, dt=0.003, t_max=2.5):
    """Batch first landing of outgoing shots (towards +y): landing xy (nan if none), net clearance at y = 0."""
    P, V, W = P0.copy(), V0.copy(), W0.copy()
    N = len(P)
    land = np.full((N, 2), np.nan); clear = np.full(N, np.nan)
    alive = np.ones(N, bool)
    t = 0.0
    while t < t_max and alive.any():
        A = accel(V, W)
        V2 = V + A * dt
        P2 = P + V2 * dt
        cross = alive & (P[:, 1] <= 0) & (P2[:, 1] > 0)
        for i in np.nonzero(cross)[0]:
            f = -P[i, 1] / (P2[i, 1] - P[i, 1])
            xn, zn = P[i, 0] + f * (P2[i, 0] - P[i, 0]), P[i, 2] + f * (P2[i, 2] - P[i, 2])
            clear[i] = zn - R - net_top(xn)
            if abs(xn) <= NET_POST and clear[i] < 0:
                alive[i] = False
        down = alive & (P2[:, 2] < R) & (V2[:, 2] < 0)
        for i in np.nonzero(down)[0]:
            f = (P[i, 2] - R) / (P[i, 2] - P2[i, 2])
            land[i] = (P[i, :2] + f * (P2[i, :2] - P[i, :2]))
            alive[i] = False
        P, V = P2, V2
        t += dt
    return land, clear


def contact_out(v_in, w_in, n, s):
    """The documented racket contact law at the head centre: commanded head velocity s * n."""
    vr = s * np.asarray(n, float) * GAIN
    u2, w2 = impact(np.asarray(v_in, float) - vr, w_in, n, E_RACKET, MU_RACKET)
    return u2 + vr, w2


def nvec(az, el):
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def base_time(dist):
    a, vm = BASE_A * 0.85, BASE_V * 0.98
    return 2 * math.sqrt(dist / a) if dist < vm * vm / a else dist / vm + vm / a


def stance(pc):
    return np.clip(np.asarray(pc[:2], float) - STANCE_OFF, BASE_LO, BASE_HI)


# ---------------------------------------------------------------------------------------------- the serve
COURT_X_S, SERVICE_Y = 4.115, 6.40
SERVE_SPOT = {"right": np.array([0.5, -12.2]), "left": np.array([-1.3, -12.2])}   # the base behind the base line
RECEIVE_SPOT = (1.6, -12.4)                    # receiving: the base at the side of the box (x mirrored for the left)
COMMIT_S = 0.6                                 # a delivered stroke is kept in its last 0.6 s before the contact
TARGET_X = {"wide": 3.6, "body": 2.0, "T": 0.45}
TARGET_Y = {"deep": 5.6, "mid": 4.2}
FIRST_SERVE = dict(zc=1.4, toss_z=2.0, height=0.8, depth="deep", sp=2.0, brush=0.0)      # placement drawn
# sp 2.0 (about 20 m/s): from the left half this forehand serve reaches only 16-21 m/s whatever the spot (the arm's
# joint speeds along that swing), from the right half sp 3.0 gives 24-31 m/s; the same speed class from both halves
# keeps the reference's points from depending on which half it serves from (a +18-20 pp first-server edge otherwise)
SECOND_SERVE = dict(zc=1.4, toss_z=2.0, height=0.8, place="body", depth="mid", sp=2.0, brush=1.0)


def serve_plan(serve_number, count, court="right", seed=0):
    """The serve: the first one flat and deep, wide / body / T drawn from the match seed, the half and how many points
    this player has served from that half (role-symmetric: both players of a duel draw the same sequence; varied
    across seeds, so the order of the point types does not repeat in every match); the second one slower with
    topspin, deep-middle, at the body."""
    if serve_number == 1:
        k = int(np.random.default_rng([int(seed), 0 if court == "right" else 1, int(count)]).integers(3))
        return dict(FIRST_SERVE, place=("wide", "body", "T")[k])
    return dict(SECOND_SERVE)


def toss_contact(P, height, zc):
    """The toss from P projected up by `height` (no spin): the control steps from the release until the falling ball
    is at zc, and its position and velocity then (the documented physics, 1 ms steps)."""
    vz = math.sqrt(2.0 * G * height)
    for _ in range(30):                          # the speed that rises `height` with the drag
        ts, Ps, Vs, Ws, _b = trajectory(P, np.array([0.0, 0.0, vz]), np.zeros(3), 3.0)
        top = float(Ps[:, 2].max()) - P[2]
        if abs(top - height) < 1e-4 or height <= 0:
            break
        vz *= math.sqrt(max(height, 1e-6) / max(top, 1e-6))
    for j in range(1, len(Ps) // 10):
        i = 10 * j
        if Vs[i, 2] < 0 and Ps[i, 2] <= zc:
            return j, Ps[i], Vs[i]
    return None


def flight_out(p, v, w):
    """First landing (x, y) and net clearance of an outgoing ball (towards +y)."""
    ts, P, V, W, bounces = trajectory(p, v, w, 2.5)
    clear = None
    for i in range(1, len(P)):
        if P[i - 1, 1] <= 0 < P[i, 1]:
            f = -P[i - 1, 1] / (P[i, 1] - P[i - 1, 1])
            xn, zn = P[i - 1, 0] + f * (P[i, 0] - P[i - 1, 0]), P[i - 1, 2] + f * (P[i, 2] - P[i - 1, 2])
            clear = zn - R - net_top(xn)
            break
    for b in bounces:
        if b[3] == "net":
            return None, clear
        if b[3] == "bounce":
            return (b[1], b[2]), clear
    return None, clear


def racket_vel(n, sp, brush):
    """Head velocity sp along the face normal plus `brush` up along the face (topspin)."""
    n = np.asarray(n, float)
    up = np.array([0.0, 0.0, 1.0]) - n[2] * n
    up /= max(1e-9, float(np.linalg.norm(up)))
    return sp * n + brush * up


def contact_vel(v_in, n, vel):
    vr = np.asarray(vel, float) * GAIN
    u2, w2 = impact(np.asarray(v_in, float) - vr, np.zeros(3), n, E_RACKET, MU_RACKET)
    return u2 + vr, w2


def solve_serve(p_b, v_in, target, sp, brush=0.0, clear_min=0.12):
    """The face normal for a head speed sp (+ brush) that sends the ball from p_b to first land at target with >=
    clear_min over the net: dict(n, vel, land, clear) or None."""
    p_b, tgt = np.asarray(p_b, float), np.asarray(target, float)
    az0 = math.atan2(tgt[1] - p_b[1], tgt[0] - p_b[0])
    for el0 in (0.0, 0.08, 0.16, -0.06, 0.25, 0.35, 0.45, 0.6):
        a, e = az0, el0
        for _ in range(14):
            n = nvec(a, e)
            vo, wo = contact_vel(v_in, n, racket_vel(n, sp, brush))
            L, clr = flight_out(p_b, vo, wo)
            if L is None:
                e += 0.04
                continue
            err = np.array(L) - tgt
            if np.linalg.norm(err) < 0.05:
                if clr is not None and clr >= clear_min:
                    return {"n": n, "vel": racket_vel(n, sp, brush), "land": L, "clear": clr,
                            "speed": float(np.linalg.norm(vo))}
                break
            J = np.zeros((2, 2))
            ok = True
            for c, (da, de) in enumerate(((2e-3, 0.0), (0.0, 2e-3))):
                n2 = nvec(a + da, e + de)
                vo2, wo2 = contact_vel(v_in, n2, racket_vel(n2, sp, brush))
                L2, _ = flight_out(p_b, vo2, wo2)
                if L2 is None:
                    ok = False
                    break
                J[:, c] = (np.array(L2) - np.array(L)) / 2e-3
            if not ok:
                e += 0.02
                continue
            try:
                dx = np.clip(np.linalg.solve(J, err), -0.06, 0.06)
            except np.linalg.LinAlgError:
                break
            a -= dx[0]
            e -= dx[1]
    return None


class Serve:
    """One serve of the reference: move -> settle -> toss -> act (timed from the toss's step) -> one correction."""

    def __init__(self, key, court, plan, lead):
        self.key, self.court, self.plan, self.lead = key, court, plan, lead
        self.stage, self.settle, self.aim_adj, self.tries = "move", 0, np.zeros(2), 0
        self.base = SERVE_SPOT[court]
        cxy = self.base + STANCE_OFF
        self.P = np.array([cxy[0], cxy[1], plan["toss_z"]])
        self.geo = toss_contact(self.P, plan["height"], plan["zc"])
        sx = -1.0 if court == "right" else 1.0      # the box diagonally opposite (own frame: x < 0 from the right)
        tx, ty = (TARGET_X.get(plan["place"], plan["place"]), TARGET_Y.get(plan["depth"], plan["depth"]))
        self.target = np.array([sx * float(tx), float(ty)])
        self.contact_k = None
        self.sol = None

    def stroke(self, k_eff):
        n_c, p_b, v_in = self.geo
        sol = None
        for f in (1.0, 0.9, 0.8, 1.15, 0.7, 0.6):
            sp = self.plan["sp"] * f
            sol = solve_serve(p_b, v_in, self.target + self.aim_adj, sp, self.plan["brush"] * f,
                              self.plan.get("clear", 0.12))
            if sol is not None:
                break
        if sol is None:
            return None
        self.sol = sol
        T = (self.contact_k - k_eff) * DT
        if not 0.0 < T <= 4.9:
            return None
        n = sol["n"]
        return {"type": "act", "t": round(T, 4), "pos": [float(x) for x in p_b - R * n],
                "normal": [float(x) for x in n], "vel": [float(x) for x in sol["vel"]]}

    def check(self, cmd):
        """After the act took effect: the landing that the delivered head velocity gives; shift the aim if it misses."""
        vd = cmd.get("vel_delivered") if isinstance(cmd, dict) else None
        if vd is None or self.sol is None:
            return False
        n_c, p_b, v_in = self.geo
        vo, wo = contact_vel(v_in, self.sol["n"], np.asarray(vd, float))
        L, clr = flight_out(p_b, vo, wo)
        err = (np.array(L) - self.target) if L is not None else np.array([0.0, -1.5])
        if np.linalg.norm(err) > 0.2 or clr is None or clr < 0.08:
            self.aim_adj = self.aim_adj - err
            return True
        return False


# ---------------------------------------------------------------------------------------------- cameras
class Cam:
    def __init__(self, c: dict):
        self.pos = np.array(c["position"], float)
        self.fw = np.array(c["forward"], float); self.fw /= np.linalg.norm(self.fw)
        self.up = np.array(c["up"], float); self.up /= np.linalg.norm(self.up)
        self.rt = np.cross(self.fw, self.up)
        self.f = float(c.get("focal_px", 579.4))
        pp = c.get("principal_point_px", [320, 240])
        self.cx, self.cy = float(pp[0]), float(pp[1])

    def ray(self, u, v):
        d = self.fw + ((u - self.cx) / self.f) * self.rt - ((v - self.cy) / self.f) * self.up
        return self.pos, d / np.linalg.norm(d)


def decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def detect(img: np.ndarray):
    """[(row, col, u, v)] of the ball's centroid in each tile where it is seen (yellow-green: G high, G > R, B low)."""
    a = img.astype(np.int16)
    Rr, Gg, Bb = a[..., 0], a[..., 1], a[..., 2]
    mask = (Gg > 110) & (Gg > Rr + 8) & (Gg > 2 * Bb + 20)
    out = []
    rows, cols = img.shape[0] // TILE_H, img.shape[1] // TILE_W
    for r in range(rows):
        for c in range(cols):
            m = mask[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W]
            ys, xs = np.nonzero(m)
            if len(xs) < 2:
                continue
            g = Gg[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W][ys, xs]
            i = int(np.argmax(g))
            near = (np.abs(xs - xs[i]) < 25) & (np.abs(ys - ys[i]) < 25)
            if near.sum() < 2:
                continue
            wgt = g[near].astype(float)
            out.append((r, c, float((xs[near] * wgt).sum() / wgt.sum()) + 0.5, float((ys[near] * wgt).sum() / wgt.sum()) + 0.5))
    return out


# ---------------------------------------------------------------------------------------------- flight fit
class Flight:
    """The rays of one ball (one leg) and the fitted flight."""

    def __init__(self, leg_id, launch_k):
        self.leg_id, self.launch_k = leg_id, launch_k
        self.rays = {}                     # (k, cam) -> (origin, direction)
        self.fit = None                    # (t_ref k, p, v, w, rms)
        self.n_fit = 0

    def add(self, k, cam, o, d):
        self.rays[(k, cam)] = (o, d)

    def span(self):
        ks = sorted({k for k, _ in self.rays})
        return (ks[-1] - ks[0]) if ks else 0, len(ks)

    def solve(self):
        keys = sorted(self.rays)
        if len({k for k, _ in keys}) < 3:
            return None
        k0 = keys[0][0]
        ts = np.array([(k - k0) * DT for k, _ in keys])
        O = np.array([self.rays[x][0] for x in keys]); D = np.array([self.rays[x][1] for x in keys])
        Pm = np.eye(3)[None] - D[:, :, None] * D[:, None, :]              # projectors orthogonal to each ray
        wgt = 1.0 / (0.002 * np.maximum(5.0, np.linalg.norm(O - np.array([0, -8.0, 1.0]), axis=1)))
        if self.fit is not None and self.fit[0] == k0:
            x = np.r_[self.fit[1], self.fit[2], self.fit[3]]
        else:
            # init: gravity-only fit to the rays before any bounce (linear in p, v)
            pre = ts <= ts.min() + 0.25
            A, b = [], []
            for i in np.nonzero(pre)[0]:
                t = ts[i]
                M = np.hstack([Pm[i], Pm[i] * t])
                A.append(M); b.append(Pm[i] @ (O[i] - np.array([0, 0, -0.5 * G * t * t])))
            A = np.vstack(A); b = np.concatenate(b)
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            x = np.r_[sol, 0.0, 0.0, 0.0]
        uniq = np.unique(ts)
        idx = np.searchsorted(uniq, ts)
        lam = 1e-2
        prior = np.array([0, 0, 0, 0, 0, 0, 1 / 400., 1 / 400., 1 / 400.])

        def resid(X):
            out, _, _ = fly(X[:, :3], X[:, 3:6], X[:, 6:9], uniq)
            Q = out[:, idx]                                               # (N, n_rays, 3)
            r = np.einsum("nij,mnj->mni", Pm, Q - O[None]) * wgt[None, :, None]
            r = r.reshape(len(X), -1)
            return np.hstack([r, X * prior[None]])

        cost = None
        for _ in range(8):
            h = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1.0, 1.0, 1.0])
            X = np.vstack([x] + [x + np.eye(9)[i] * h[i] for i in range(9)])
            Rr = resid(X)
            r0 = Rr[0]; c0 = float(r0 @ r0)
            J = ((Rr[1:] - r0) / h[:, None]).T
            H = J.T @ J; g = J.T @ r0
            for _try in range(4):
                dx = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), g)
                xn = x + dx
                rn = resid(xn[None])[0]; cn = float(rn @ rn)
                if cn < c0:
                    x, lam = xn, max(lam * 0.3, 1e-6)
                    break
                lam *= 10
            else:
                break
            cost = cn
            if abs(c0 - cn) < 1e-4 * c0:
                break
        r = resid(x[None])[0][:-9].reshape(-1, 3) / wgt[:, None]
        rms = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
        self.fit = (k0, x[:3], x[3:6], x[6:9], rms)
        self.n_fit = len(keys)
        return self.fit


# ---------------------------------------------------------------------------------------------- the player
class Player:
    def __init__(self, cams, lead_steps):
        self.cams = cams
        self.lead = int(lead_steps)
        self.flight = None
        self.cmd = None             # the act in effect for this ball: dict(t_c abs, pos, normal, k_e, ...)
        self.moved = None           # base goal sent for this ball
        self.n_shots = 0
        self.opp_x = 0.0
        self.log = []
        self.hit_done = None

    # ------------------------------------------------ observations
    def update(self, ob, img):
        legs = ob.get("view_leg_ids") or []
        steps = ob.get("view_steps") or []
        times = ob.get("view_times") or []
        if not legs:
            return
        cur = legs[-1]
        if self.flight is None or self.flight.leg_id != cur:
            launch = int(steps[-1] - round(times[-1] / DT))
            self.flight = Flight(cur, launch)
            self.cmd = None; self.moved = None; self.hit_done = None
        if img is None:
            return
        for (r, c, u, v) in detect(img):
            if c >= len(steps) or legs[c] != cur or times[c] <= 0.0:
                continue
            o, d = self.cams[r].ray(u, v)
            self.flight.add(int(steps[c]), r, o, d)

    # ------------------------------------------------ prediction and stroke
    def predict(self):
        f = self.flight
        if f is None:
            return None
        fit = f.solve()
        if fit is None:
            return None
        k0, p, v, w, rms = fit
        t_off = k0 * DT
        ts, P, V, W, bounces = trajectory(p, v, w, 3.0)
        return dict(k0=k0, t0=t_off, ts=ts + t_off, P=P, V=V, W=W, bounces=bounces, rms=rms, p=p, v=v, w=w)

    def choose(self, pr, k_now, base_now, latency_s):
        """The contact on the predicted flight after its bounce on our half: index into pr arrays, or None."""
        bs = [b for b in pr["bounces"] if b[3] == "bounce"]
        if not bs or any(b[3] == "net" for b in pr["bounces"]):
            return None, "no bounce"
        b0 = bs[0]
        if not (abs(b0[1]) <= COURT_X and -COURT_Y <= b0[2] < 0):
            return None, "lands out"
        tb0 = b0[0] + pr["t0"]
        tb1 = bs[1][0] + pr["t0"] if len(bs) > 1 else pr["ts"][-1]
        ts, P = pr["ts"], pr["P"]
        t_now = k_now * DT
        cands, late = [], []
        for i in range(len(ts)):
            t = ts[i]
            if t <= tb0 + 0.02 or t >= tb1 - 0.02:
                continue
            k = int(round(t / DT))
            if abs(t - k * DT) > 1e-6:
                continue
            q = P[i]
            if not (0.55 <= q[2] <= 1.4) or q[1] > -0.5 or q[1] < -15.0 or abs(q[0]) > 6.4:
                continue
            st = stance(q)
            avail = t - t_now - latency_s - self.lead * DT
            if avail < 0.12:
                continue
            need = base_time(float(np.hypot(*(st - base_now))))
            if need > avail + (0.0 if self.moved is None else 0.25):
                late.append((need - avail, i))
                continue
            cands.append(i)
        if not cands and late:
            # the base cannot get to any stance in time: try the contact it misses by the least (the controller
            # brings the base as far as it gets; a weak or missed shot is still better than none)
            return min(late)[1], "ok (base late)"
        if not cands:
            return None, "no reachable contact"
        seg = [i for i in range(len(ts)) if tb0 < ts[i] < tb1]
        apex = max(seg, key=lambda i: P[i, 2]) if seg else cands[0]
        for i in cands:
            if i >= apex and P[i, 2] <= 1.15:
                return i, "ok"
        return min(cands, key=lambda i: abs(P[i, 2] - 1.0)), "ok"

    def aim(self):
        ox = self.opp_x
        sgn = -1.0 if ox > 0.4 else (1.0 if ox < -0.4 else (1.0 if self.n_shots % 2 else -1.0))
        return np.array([sgn * 3.2, 9.8])

    def stroke(self, p_b, v_in, w_in, aim, sp=3.0):
        """Face normal for head speed sp along it so that the ball lands at aim: (n, sp, v_out) or None."""
        for _ in range(4):
            d_out = np.r_[aim - p_b[:2], 0.0]; d_out /= np.linalg.norm(d_out); d_out[2] = 0.12
            d_out /= np.linalg.norm(d_out)
            n0 = d_out - v_in / np.linalg.norm(v_in); n0 /= np.linalg.norm(n0)
            az = math.atan2(n0[1], n0[0])
            els = np.linspace(-0.25, 0.75, 21)
            outs = [contact_out(v_in, w_in, nvec(az, e), sp) for e in els]
            L, C = landings(np.tile(p_b, (len(els), 1)), np.array([o[0] for o in outs]), np.array([o[1] for o in outs]))
            ok = ~np.isnan(L[:, 0]) & (C > 0.1)
            if not ok.any():
                sp *= 0.85; continue
            e_ = np.where(ok, np.abs(L[:, 1] - aim[1]) + 0.3 * np.abs(L[:, 0] - aim[0]), np.inf)
            el = float(els[int(np.argmin(e_))])
            sol = None
            for _it in range(14):
                N3 = [nvec(az, el), nvec(az + 2e-3, el), nvec(az, el + 2e-3)]
                outs = [contact_out(v_in, w_in, n, sp) for n in N3]
                L, C = landings(np.tile(p_b, (3, 1)), np.array([o[0] for o in outs]), np.array([o[1] for o in outs]))
                if np.isnan(L).any():
                    break
                err = L[0] - aim
                if np.linalg.norm(err) < 0.08:
                    sol = (N3[0], outs[0][0], C[0], L[0])
                    break
                J = np.column_stack([(L[1] - L[0]) / 2e-3, (L[2] - L[0]) / 2e-3])
                try:
                    dx = np.clip(np.linalg.solve(J, err), -0.08, 0.08)
                except np.linalg.LinAlgError:
                    break
                az -= dx[0]; el -= dx[1]
            if sol is None or np.isnan(sol[2]) or sol[2] < 0.4:
                sp *= 0.85
                continue
            return sol[0], sp, sol[1], sol[3]
        return None

    def plan(self, pr, k_now, base_now, latency_s):
        i, why = self.choose(pr, k_now, base_now, latency_s)
        if i is None:
            return None, why
        p_b, v_in, w_in = pr["P"][i], pr["V"][i], pr["W"][i]
        r = self.stroke(p_b, v_in, w_in, self.aim())
        if r is None:
            return None, "no stroke"
        n, sp, vo, land = r
        return dict(t_c=float(pr["ts"][i]), k_c=int(round(pr["ts"][i] / DT)), p_b=p_b, pos=p_b - R * n, normal=n,
                    vel=sp * n, v_out=vo, land=land), "ok"


# ---------------------------------------------------------------------------------------------- the act(view) players
class Reference:
    """The legacy play loop (tn_player.play) as act(view): one command (or None) per call. Over HTTP (the observation
    has an obs_id) the current step is estimated from the wall clock as the legacy client did; in-process (core/local.py)
    the view's step is exact."""

    def __init__(self, spec, seat, seed=0):
        self.high = spec.get("action_level", "high") == "high"
        self.seed = int(seed)                  # the serve placement draws (serve_plan)
        self.cams = [Cam(c) for c in spec["cameras"]]
        self.lead = int(spec["act"]["lead_steps"])
        self.period = float((spec.get("timing") or {}).get("period_s") or 0.2)
        self.P = Player(self.cams, self.lead)
        self.lat = [0.3]                       # wall seconds from an observation to an act's arrival
        self.home_sent = None
        self.awaiting = False                  # an act was sent: read its true contact step back from the state
        self.fit_k = -99
        self.k_bias = 0.0                      # learned offset of the estimated effect step (over HTTP)
        self.point = None                      # every point starts afresh (the flight, the stroke, the moves)
        self.srv = None                        # the serve under way (Serve)
        self.served = []                       # (point, court) of my serves: the role-symmetric placement counter
        self.rx_sent = None                    # receiving: the spot was sent for (point, serve number)

    def review(self, reply):
        return "reference player: scripted", "reference player: same policy"

    def act(self, view):
        st, ob = view.get("state") or {}, view.get("observation") or {}
        if st.get("world") != "running" or any(c.get("type") != "toss" for c in st.get("pending_commands") or []):
            return None                        # (a pending toss does not block the act: a long act lead)
        if not self.high:
            return None                        # the low level: no stroke planner here (the reference is passive)
        live = "obs_id" in ob
        t_entry = time.monotonic()
        k_obs = float(st.get("step", 0))
        if live:
            k_obs += (float(ob.get("render_wall_s") or 0.0) + 0.03) / self.period

        def now_k():
            return k_obs + ((time.monotonic() - t_entry) / self.period if live else 0.0)
        if st.get("point") != self.point:      # a new point: start afresh
            self.point = st.get("point")
            self.P = Player(self.cams, self.lead)
            self.awaiting, self.fit_k, self.home_sent = False, -99, None
        sv = st.get("serve") or {}
        if sv.get("phase") in ("waiting", "open", "tossed"):
            if st.get("you_serve"):
                return self.serve(st, sv, now_k)
            key = (st.get("point"), sv.get("serve_number"))
            if st.get("you_receive_serve") and self.rx_sent != key:
                self.rx_sent = key             # receive from the side of the box
                x, y = RECEIVE_SPOT
                return {"type": "move", "base": [x if sv.get("court") == "right" else -x, y]}
            return None
        P, lead, period = self.P, self.lead, self.period
        if self.awaiting and P.cmd is not None:
            c = st.get("command")
            P.cmd["k_c_true"] = (int(st["step"] + round(c["contact_in_s"] / DT))
                                 if isinstance(c, dict) and c.get("contact_in_s") is not None else -1)
            if P.cmd["k_c_true"] > 0 and abs(P.cmd["k_c_true"] - P.cmd["k_c"]) <= 15:
                # over HTTP the effect step is estimated from the wall clock; a lagging world shifts it: learn the
                # offset from the contact steps the service reports back (in-process it is always 0)
                self.k_bias = max(-10.0, min(10.0, self.k_bias + 0.7 * (P.cmd["k_c_true"] - P.cmd["k_c"])))
            self.awaiting = False
        state = st.get("ball_state")
        incoming = state == "coming to you" and st.get("ball_in_play")
        base_now = np.array(st.get("robot_base_xy") or HOME, float)
        legs = ob.get("view_leg_ids") or [None]
        if not incoming:
            if state == "going to the opponent" and self.home_sent != legs[-1]:
                self.home_sent = legs[-1]
                if P.cmd is not None:
                    P.n_shots += 1
                P.cmd = None
                P.moved = None
                return {"type": "move", "base": [0.0, -11.4]}
            return None
        img = decode(base64.b64decode(ob["image_png_b64"])) if ob.get("image_png_b64") else None
        P.update(ob, img)
        f = P.flight
        if f is None:
            return None
        span, nk = f.span()
        if nk < 3:
            return None
        if P.cmd is not None and st.get("step", 0) - self.fit_k < 10:
            return None                        # a stroke is set: re-fit the flight at most every 10 steps
        self.fit_k = st.get("step", 0)
        pr = P.predict()
        if pr is None:
            return None
        if pr["k0"] - f.launch_k < 30:          # where the opponent played the ball from (the flight's start)
            P.opp_x = float(pr["p"][0] - pr["v"][0] * (pr["k0"] - f.launch_k) * DT) * 0.4
        k_now = now_k()
        plan, why = P.plan(pr, k_now, base_now, max(self.lat[-5:]) * 1.2 / period * DT)
        if plan is None:
            return None
        cmd = P.cmd
        lat_g = max(self.lat[-5:]) * DT / period
        left = (plan["k_c"] - k_now) * DT - lead * DT - lat_g
        if cmd is not None:
            same = (abs(cmd["k_c"] - plan["k_c"]) <= 1 and np.linalg.norm(cmd["pos"] - plan["pos"]) < 0.06
                    and cmd.get("k_c_true") in (None, cmd["k_c"]))
            c = st.get("command")
            ok = isinstance(c, dict) and str(c.get("stroke", "")).startswith("ok")
            if same or left < (COMMIT_S if ok else 0.3):   # keep a delivered stroke in its last COMMIT_S
                return None
        elif span < 25 and left > 0.45:
            stn = stance(plan["pos"])               # not yet sure where the ball goes: only bring the base
            if (P.moved is None or np.linalg.norm(P.moved - stn) > 0.5) and np.linalg.norm(stn - base_now) > 0.5:
                P.moved = stn
                return {"type": "move", "base": [float(stn[0]), float(stn[1])]}
            return None
        P.moved = stance(plan["pos"])               # the act's own base plan brings the base to the stance
        k_send = now_k()
        if live:
            self.lat.append((k_send - float(st.get("step", 0))) * period + 0.05)
        k_e = int(math.floor(k_send + 0.002 / period + self.k_bias)) + lead
        T = round(plan["t_c"] - k_e * DT, 4)
        if not 0.0 < T <= 4.9:
            return None
        P.cmd = dict(plan, k_e=k_e, T=T, k_c_true=None)
        self.awaiting = True
        return {"type": "act", "t": T, "pos": [float(x) for x in plan["pos"]],
                "normal": [float(x) for x in plan["normal"]], "vel": [float(x) for x in plan["vel"]]}


    def serve(self, st, sv, now_k):
        """Our serve (Serve): the base to the spot, the toss once the window is open, then the act timed from the
        toss's step; one correction from the delivered head velocity, and a re-act if the contact step is off."""
        key, court = (st.get("point"), sv.get("serve_number")), sv.get("court", "right")
        if self.srv is None or self.srv.key != key:
            if (st.get("point"), court) not in self.served:
                self.served.append((st.get("point"), court))
            count = sum(1 for (_p, c) in self.served if c == court) - 1
            self.srv = Serve(key, court, serve_plan(int(sv.get("serve_number") or 1), count, court, self.seed),
                             self.lead)
        S = self.srv
        if S.geo is None:
            return None
        base = np.array(st.get("robot_base_xy") or HOME, float)
        if S.stage == "move":
            S.stage = "settle"
            return {"type": "move", "base": [float(S.base[0]), float(S.base[1])]}
        if S.stage == "settle":
            S.settle = S.settle + 1 if np.linalg.norm(base - S.base) < 0.004 else 0
            if S.settle >= 2 and sv.get("phase") == "open":
                S.stage = "tossed_wait"
                return {"type": "toss", "pos": [float(x) for x in S.P], "height": float(S.plan["height"])}
            return None
        if S.stage == "tossed_wait":
            if sv.get("tossed_at_step") is None:
                if sv.get("toss_takes_effect_at_step") is not None:   # a long act lead: the act before the release
                    S.contact_k = int(sv["toss_takes_effect_at_step"]) + S.geo[0]
                    S.stage = "act"
                else:
                    if sv.get("phase") == "open":
                        S.stage, S.settle = "settle", 0      # the toss was refused: settle and toss again
                    return None
            else:
                S.contact_k = int(sv["tossed_at_step"]) + S.geo[0]
                S.stage = "act"
        k = now_k()
        if S.stage == "act":
            k_e = int(math.floor(k + 0.002 / self.period + self.k_bias)) + self.lead
            a = S.stroke(k_e)
            S.stage = "acted" if a is not None else "done"
            return a
        if S.stage == "acted":
            c = st.get("command")
            if not (isinstance(c, dict) and c.get("contact_in_s") is not None):
                return None
            k_true = int(st["step"] + round(c["contact_in_s"] / DT))
            left = S.contact_k - k
            if k_true != S.contact_k and S.tries < 3 and left > 15:
                self.k_bias = max(-10.0, min(10.0, self.k_bias + (k_true - S.contact_k)))
                S.tries += 1
                S.stage = "act"
            elif S.check(c) and S.tries < 3 and left > 40:
                S.tries += 1
                S.stage = "act"
            else:
                S.stage = "done"
        return None


class Random:
    """Random base moves and cancels (a check policy; seeded by the seed, not the seat)."""

    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 17)

    def act(self, view):
        st = view.get("state") or {}
        if st.get("world") != "running" or st.get("pending_commands") or self.g.uniform() > 0.2:
            return None
        if self.g.uniform() < 0.2:
            return {"type": "none"}
        return {"type": "move", "base": [float(self.g.uniform(-3, 3)), float(self.g.uniform(-13, -9))]}
