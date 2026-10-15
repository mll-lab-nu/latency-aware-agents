"""Disc golf's built-in players (the same action interface as a model's program; core/client.py plays them over HTTP).

  Reference  the Season-1 reference disc golfer (scripts/olympiad/discgolf/*/solution/discgolf_ref.py, moved here
             unchanged below the line): camera images, the referee's decision point and the documented physics of
             /spec only (numpy + Pillow). Per throw it
               1. reads the wind from the windsock tile (the sock's direction and projected length, the documented
                  hang law);
               2. maps the trees from the overview tile (canopy-green pixels inside the course limits, an occupancy
                  grid; the trees are taken as tall as the documented maximum, so the planner flies around them);
               3. plans: it simulates candidate throws with its own vectorized copy of the documented flight model
                  (lift, drag, pitching moment, gyroscopic precession, spin-down, in the wind; ground skid; basket
                  catch) over a grid of discs, speeds, aims, banks, spins and launch angles, keeps the best, evaluates
                  them under the documented execution scatter and refines locally. The objective is the expected
                  number of throws.
             Its random refinements are seeded by the throw number (as in Season 1, so it plays the Season-1
             rounds).
  Random     uniform random valid throws roughly towards the basket (generic tests); seeded by the seed only.
No simulator import: this file runs in the player's container as the task's oracle (numpy + pillow only)."""
from __future__ import annotations

import base64
import io
import math

import numpy as np

DISC_NAMES = ["putter", "midrange", "driver"]


class Reference:
    def __init__(self, spec, seat, seed=0):
        self.spec, self.seed = spec, int(seed)
        self.model = Model(spec)
        self.holes: dict = {}
        self.n = 0

    def act(self, view):
        st = view["state"]
        dec = st["decision"]
        t = tiles(base64.b64decode(view["observation"]["image_png_b64"]), self.spec)
        key = (dec["hole"], dec.get("playoff_hole", 0))
        if key not in self.holes:
            self.holes[key] = Hole(self.spec["match"]["holes"][dec["hole"] - 1], self.model)
            self.holes[key].set_overview(t["overview"], None)
        wind = read_wind(t["wind"], self.spec)
        act = plan(self.holes[key], dec["lie"], wind, mando_cleared=dec.get("mandatory_cleared", True),
                   rng_seed=self.n)
        self.n += 1
        return {"type": "throw", **{k: v for k, v in act.items() if k != "expected_throws"}}

    def review(self, reply):
        return ("read the wind from the windsock and the trees from the overview; planned the throw with the flight "
                "model under the published scatter", "plan the next throw the same way from the new lie")


class Random:
    def __init__(self, spec, seat, seed=0):
        self.g = np.random.default_rng(int(seed) * 7919 + 23)   # by seed, not by seat: mirror runs match

    def act(self, view):
        dec = view["state"].get("decision") or {}
        g = self.g
        head = float(dec.get("heading_to_basket_deg", 0.0)) + float(g.uniform(-30, 30))
        return {"type": "throw", "disc": DISC_NAMES[int(g.integers(0, 3))], "speed": float(g.uniform(8, 28)),
                "launch_deg": float(g.uniform(0, 20)), "nose_deg": float(g.uniform(-3, 5)),
                "bank_deg": float(g.uniform(-30, 30)), "spin_rps": float(g.choice([-1, 1]) * g.uniform(6, 18)),
                "aim_deg": round(((head + 180.0) % 360.0) - 180.0, 4)}


# ---------------------------------------------------------------------------------- the Season-1 reference planner

class Model:
    """The documented physics (game spec: discs, air, contacts, throw)."""

    def __init__(self, spec):
        self.spec = spec
        d = spec["discs"]
        self.coef = {k: np.array([d[n][k] for n in DISC_NAMES]) for k in
                     ("CL0", "CLa", "CD0", "CDa", "ALPHA0", "CM0", "CMa", "CS")}
        p = d["putter"]
        self.m, self.diam, self.izz = p["mass_kg"], p["diameter_m"], p["Izz_kg_m2"]
        self.area = math.pi * (self.diam / 2) ** 2
        self.rho, self.g = spec["air"]["rho_kg_m3"], spec["air"]["g_m_s2"]
        c = spec["contacts"]
        self.mu_i, self.mu_k = c["ground"]["mu_impact"], c["ground"]["mu_kinetic"]
        b = c["basket"]
        self.catch_r, self.chains, self.catch_v = b["catch_radius_m"], b["chains_m"], b["max_catch_speed_m_s"]
        self.tree_top = c["tree_heights_m"][1]
        self.release_h = float(spec["throw"]["release"].split()[1])
        self.limits = spec["throw"]["limits"]
        self.sd = spec["throw"]["scatter_sd"]


class Hole:
    def __init__(self, sign, model):
        self.sign = sign
        self.M = model
        self.basket = tuple(sign["basket"])
        lim = sign["course_limits"]
        self.bounds = (lim["x"][0], lim["x"][1], lim["y"][0], lim["y"][1])
        self.ob = [np.array(p) for p in sign["ob_areas"]]
        self.tilt = sign["ground"]["tilt"]
        self.hills = sign["ground"]["hills"]
        self.mando = sign.get("mandatory")
        self.mask = None
        self.zb = float(self.ground(np.array([self.basket[0]]), np.array([self.basket[1]]))[0])

    def ground(self, x, y):
        z = self.tilt[0] * x + self.tilt[1] * y
        for h in self.hills:
            z = z + h["height"] * np.exp(-((x - h["x"]) ** 2 + (y - h["y"]) ** 2) / (2 * h["sigma"] ** 2))
        return z

    def in_bounds(self, x, y):
        x0, x1, y0, y1 = self.bounds
        ok = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        for p in self.ob:          # OB areas are axis-aligned rectangles on this course; general polygons: ray cast
            ok &= ~_in_poly(p, x, y)
        return ok

    # ------------------------------------------------------------------ the tree map from the overview tile
    def set_overview(self, img, scale_ppm_px):
        cx, cy = self.sign["overview"]["centre"]
        ext = self.sign["overview"]["extent_m"]
        h, w = img.shape[:2]
        r, g, b = (img[..., i].astype(int) for i in range(3))
        canopy = (r < 75) & (g > 85) & (g - r > 55) & (b < 80)
        s = ext / h                                     # m per pixel
        self.ov = (cx, cy, s, w, h)
        # dilate by the disc radius + a margin (2 px ~ 0.3 m)
        m = canopy.copy()
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                if dx * dx + dy * dy <= 5:
                    m |= np.roll(np.roll(canopy, dy, 0), dx, 1)
        self.mask = m

    def blocked(self, x, y):
        """Occupancy lookup (arrays of world x, y) -> bool array."""
        cx, cy, s, w, h = self.ov
        u = np.floor(w / 2 + (x - cx) / s).astype(int)
        v = np.floor(h / 2 - (y - cy) / s).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        out = np.zeros(x.shape, bool)
        out[ok] = self.mask[v[ok], u[ok]]
        return out

    def line_blocked(self, px, py):
        """Is the straight line from (px, py) to the basket through trees (beyond 3 m from both ends)?"""
        bx, by = self.basket
        d = np.hypot(bx - px, by - py)
        out = np.zeros(px.shape, bool)
        for f in np.linspace(0.0, 1.0, 40):
            x, y = px + f * (bx - px), py + f * (by - py)
            far = (f * d > 3.0) & ((1 - f) * d > 3.0)
            out |= far & self.blocked(x, y)
        return out


def _in_poly(p, x, y):
    inside = np.zeros(np.shape(x), bool)
    n = len(p)
    for i in range(n):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % n]
        if y0 == y1:
            continue
        cond = (y0 > y) != (y1 > y)
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (xc > x)
    return inside


# ---------------------------------------------------------------------------------- the flight (vectorized)
def simulate(hole, lie, P, wind, dt=0.01, max_t=15.0, mando_cleared=True):
    """P: dict of arrays aim_deg, speed, launch_deg, nose_deg, bank_deg, spin_rps, disc (int). Returns dict of arrays:
    holed, rest x, y, hit (trees), mando crossing y (nan if none), last in-bounds x, y."""
    M = hole.M
    N = len(P["speed"])
    di = P["disc"].astype(int)
    C = {k: v[di] for k, v in M.coef.items()}
    psi, gam = np.radians(P["aim_deg"]), np.radians(P["launch_deg"])
    th, be = np.radians(P["launch_deg"] + P["nose_deg"]), np.radians(P["bank_deg"])
    fx, fy = np.cos(psi), np.sin(psi)
    lx, ly = -np.sin(psi), np.cos(psi)
    x = np.full(N, float(lie[0]))
    y = np.full(N, float(lie[1]))
    z = hole.ground(x, y) + M.release_h
    vx, vy, vz = P["speed"] * np.cos(gam) * fx, P["speed"] * np.cos(gam) * fy, P["speed"] * np.sin(gam)
    nx = -np.sin(th) * fx + np.cos(th) * np.sin(be) * lx
    ny = -np.sin(th) * fy + np.cos(th) * np.sin(be) * ly
    nz = np.cos(th) * np.cos(be)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    ws = -2 * np.pi * P["spin_rps"]
    S = np.stack([x, y, z, vx, vy, vz, nx, ny, nz, ws])
    wx, wy = wind
    q0 = 0.5 * M.rho * M.area

    def f(S, C):
        x, y, z, vx, vy, vz, nx, ny, nz, ws = S
        ux, uy, uz = vx - wx, vy - wy, vz
        U = np.sqrt(ux * ux + uy * uy + uz * uz) + 1e-9
        un = ux * nx + uy * ny + uz * nz
        px, py, pz = ux - un * nx, uy - un * ny, uz - un * nz
        UP = np.sqrt(px * px + py * py + pz * pz) + 1e-12
        alpha = np.arctan2(-un, UP)
        q = q0 * U * U
        hx, hy, hz = ux / U, uy / U, uz / U
        nu = nx * hx + ny * hy + nz * hz
        lx_, ly_, lz_ = nx - nu * hx, ny - nu * hy, nz - nu * hz
        LD = np.sqrt(lx_ * lx_ + ly_ * ly_ + lz_ * lz_) + 1e-12
        CL = C["CL0"] + C["CLa"] * alpha
        CD = C["CD0"] + C["CDa"] * (alpha - C["ALPHA0"]) ** 2
        fl = q * CL / M.m / LD
        fd = q * CD / M.m
        ax = fl * lx_ - fd * hx
        ay = fl * ly_ - fd * hy
        az = fl * lz_ - fd * hz - M.g
        xax, xay, xaz = px / UP, py / UP, pz / UP
        yax, yay, yaz = ny * xaz - nz * xay, nz * xax - nx * xaz, nx * xay - ny * xax
        Mm = q * M.diam * (C["CM0"] + C["CMa"] * alpha)
        k = -Mm / (M.izz * ws)
        dws = q * M.diam * C["CS"] * (ws * M.diam / (2 * U)) / M.izz
        return np.stack([vx, vy, vz, ax, ay, az, k * yax, k * yay, k * yaz, dws])

    alive = np.ones(N, bool)
    holed = np.zeros(N, bool)
    hit = np.zeros(N, bool)
    rest = np.zeros((2, N))
    landed_v = np.zeros((3, N))
    landed = np.zeros(N, bool)
    lastin = S[:2].copy()
    mcross = np.full(N, np.nan)
    bx, by = hole.basket
    zb = hole.zb
    t = 0.0
    x0s, y0s = S[0].copy(), S[1].copy()
    while t < max_t and alive.any():
        idx = np.nonzero(alive)[0]
        s = S[:, idx]
        Ci = {k: v[idx] for k, v in C.items()}
        k1 = f(s, Ci)
        k2 = f(s + 0.5 * dt * k1, Ci)
        k3 = f(s + 0.5 * dt * k2, Ci)
        k4 = f(s + dt * k3, Ci)
        s1 = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        nn = np.sqrt(s1[6] ** 2 + s1[7] ** 2 + s1[8] ** 2)
        s1[6:9] /= nn
        t += dt
        X, Y, Z = s1[0], s1[1], s1[2]
        gz = hole.ground(X, Y)
        hz = Z - gz
        # basket
        rb = np.hypot(X - bx, Y - by)
        spd = np.sqrt(s1[3] ** 2 + s1[4] ** 2 + s1[5] ** 2)
        inb = (rb <= M.catch_r) & (Z - zb >= M.chains[0] - 0.05) & (Z - zb <= M.chains[1])
        catch = inb & (spd <= M.catch_v)
        spit = inb & ~catch
        # trees: the occupancy grid, up to the tallest tree; not within 3 m of the start
        dstart = np.hypot(X - x0s[idx], Y - y0s[idx])
        tree = hole.blocked(X, Y) & (hz < M.tree_top) & (dstart > 3.0) & ~inb
        ground = hz <= 0
        # mando crossing
        if hole.mando is not None and not mando_cleared:
            mx = hole.mando["point"][0]
            xp = s[0]
            cr = ((xp - mx) * (X - mx) < 0) & np.isnan(mcross[idx])
            if cr.any():
                fr = (mx - xp[cr]) / (X[cr] - xp[cr])
                mcross[idx[cr]] = s[1][cr] + fr * (Y[cr] - s[1][cr])
        ib = hole.in_bounds(X, Y)
        lastin[0, idx[ib]] = X[ib]
        lastin[1, idx[ib]] = Y[ib]
        S[:, idx] = s1
        done = catch | spit | tree | ground
        if done.any():
            j = idx[done]
            holed[idx[catch]] = True
            rest[0, idx[catch]] = bx
            rest[1, idx[catch]] = by
            # spat out: drops at the basket; a tree: drops below the contact (the canopy keeps a quarter of the speed)
            for m_, back in ((spit, 0.8), (tree, 1.5)):
                jj = idx[m_]
                if len(jj):
                    vh = np.hypot(S[3, jj], S[4, jj]) + 1e-9
                    rest[0, jj] = S[0, jj] - back * S[3, jj] / vh
                    rest[1, jj] = S[1, jj] - back * S[4, jj] / vh
            hit[idx[tree]] = True
            gg = idx[ground & ~catch & ~spit & ~tree]
            landed[gg] = True
            landed_v[:, gg] = S[3:6, gg]
            rest[0, gg] = S[0, gg]
            rest[1, gg] = S[1, gg]
            alive[j] = False
    # still flying at max_t: where it is
    rest[0, alive] = S[0, alive]
    rest[1, alive] = S[1, alive]
    # skid
    vh = np.hypot(landed_v[0], landed_v[1]) + 1e-12
    sk = np.maximum(0.0, vh - M.mu_i * np.abs(landed_v[2])) ** 2 / (2 * M.mu_k * M.g)
    sk = np.where(landed, sk, 0.0)
    rx = rest[0] + sk * landed_v[0] / vh
    ry = rest[1] + sk * landed_v[1] / vh
    # a skid into the trees stops there (approximately: at the first blocked point)
    for fr in np.linspace(0.1, 1.0, 10):
        px, py = rest[0] + fr * sk * landed_v[0] / vh, rest[1] + fr * sk * landed_v[1] / vh
        stop = landed & hole.blocked(px, py) & (sk > 0.3)
        if stop.any():
            rx = np.where(stop, px - 0.5 * landed_v[0] / vh, rx)
            ry = np.where(stop, py - 0.5 * landed_v[1] / vh, ry)
            landed = landed & ~stop
    return dict(holed=holed, x=rx, y=ry, hit=hit, mando_y=mcross, lastin_x=lastin[0], lastin_y=lastin[1])


def remaining(d):
    """Estimated throws left from distance d (m) in the open."""
    d = np.asarray(d, float)
    return np.where(d <= 4.0, 1.0, np.where(d <= 20.0, 1.0 + (d - 4.0) / 16.0 * 0.8,
                    np.where(d <= 90.0, 1.8 + (d - 20.0) / 70.0 * 0.9, 2.7 + (d - 90.0) / 110.0)))


def cost(hole, res, mando_cleared=True):
    bx, by = hole.basket
    x, y = res["x"], res["y"]
    ob = ~hole.in_bounds(x, y)
    nx = np.where(ob, res["lastin_x"], x)
    ny = np.where(ob, res["lastin_y"], y)
    pen = ob.astype(float)
    if hole.mando is not None and not mando_cleared:
        my = hole.mando["point"][1]
        side = hole.mando["pass_on"]
        crossed = ~np.isnan(res["mando_y"])
        wrong = crossed & ((res["mando_y"] >= my) if side == "right" else (res["mando_y"] <= my))
        dz = hole.mando["drop_zone"]
        nx = np.where(wrong, dz[0], nx)
        ny = np.where(wrong, dz[1], ny)
        pen = np.where(wrong, 1.0, pen)
    d = np.hypot(nx - bx, ny - by)
    c = pen + remaining(d) + 0.6 * hole.line_blocked(nx, ny)
    return np.where(res["holed"], 0.0, c)


# ---------------------------------------------------------------------------------- the planner
KEYS = ("aim_deg", "speed", "launch_deg", "nose_deg", "bank_deg", "spin_rps", "disc")


def _grid(**axes):
    names = list(axes)
    mesh = np.meshgrid(*[np.asarray(axes[n], float) for n in names], indexing="ij")
    return {n: m.ravel() for n, m in zip(names, mesh)}


def _clip(P, lim):
    for k, (lo, hi) in lim.items():
        P[k] = np.clip(P[k], lo, hi)
    return P


def _scattered(P, sd, k, rng):
    n = len(P["speed"])
    Q = {key: np.repeat(v, k) for key, v in P.items()}
    m = n * k
    Q["aim_deg"] = Q["aim_deg"] + rng.standard_normal(m) * sd["aim_deg"]
    Q["launch_deg"] = Q["launch_deg"] + rng.standard_normal(m) * sd["launch_deg"]
    Q["nose_deg"] = Q["nose_deg"] + rng.standard_normal(m) * sd["nose_deg"]
    Q["bank_deg"] = Q["bank_deg"] + rng.standard_normal(m) * sd["bank_deg"]
    Q["speed"] = Q["speed"] * (1 + rng.standard_normal(m) * sd["speed_frac"])
    Q["spin_rps"] = Q["spin_rps"] * (1 + rng.standard_normal(m) * sd["spin_frac"])
    return Q


def plan(hole, lie, wind, mando_cleared=True, rng_seed=0, straight=False):
    """The throw with the lowest expected number of throws (dict for the `throw` command)."""
    M = hole.M
    rng = np.random.default_rng(rng_seed)
    bx, by = hole.basket
    d = math.hypot(bx - lie[0], by - lie[1])
    head = math.degrees(math.atan2(by - lie[1], bx - lie[0]))
    if straight:                           # the baseline: aim at the basket, flat, only disc and speed chosen
        if d < 22:
            G = _grid(aim_deg=[head], speed=np.arange(5.0, 16.0, 0.25), launch_deg=np.arange(-2.0, 14.0, 1.0),
                      nose_deg=[0.0], bank_deg=[0.0], spin_rps=[12.0], disc=[0])
        else:
            G = _grid(aim_deg=[head], speed=np.arange(8.0, 30.01, 1.0), launch_deg=[10.0], nose_deg=[0.0],
                      bank_deg=[0.0], spin_rps=[14.0], disc=[0, 1, 2])
        # the straight thrower knows neither the wind nor the trees: it judges in still air on an open course
        open_hole = _OpenHole(hole)
        res = simulate(open_hole, lie, G, (0.0, 0.0), dt=0.02, mando_cleared=True)
        c = np.where(res["holed"], 0.0, np.hypot(res["x"] - bx, res["y"] - by) / 100.0 + 0.5)
        i = int(np.argmin(c))
        return {k: (DISC_NAMES[int(G[k][i])] if k == "disc" else round(float(G[k][i]), 3)) for k in KEYS}
    if d < 22:
        G = _grid(aim_deg=head + np.arange(-3.0, 3.01, 0.5), speed=np.arange(5.0, 16.0, 0.5),
                  launch_deg=np.arange(-2.0, 14.0, 1.5), nose_deg=[0.0], bank_deg=[0.0], spin_rps=[12.0], disc=[0])
        if hole.line_blocked(np.array([lie[0]]), np.array([lie[1]]))[0]:
            G2 = _grid(aim_deg=head + np.arange(-40.0, 40.01, 4.0), speed=np.arange(6.0, 16.0, 1.5),
                       launch_deg=[4.0, 10.0], nose_deg=[0.0], bank_deg=[-40.0, -20.0, 20.0, 40.0],
                       spin_rps=[12.0, -12.0], disc=[0])
            G = {k: np.concatenate([G[k], G2[k]]) for k in G}
    else:
        discs = [0, 1] if d < 60 else [0, 1, 2]
        G = _grid(aim_deg=head + np.arange(-44.0, 44.01, 4.0), speed=np.arange(9.0, 30.01, 1.75),
                  launch_deg=[6.0, 13.0], nose_deg=[0.0], bank_deg=[-40.0, -20.0, 0.0, 20.0, 40.0],
                  spin_rps=[14.0, -14.0], disc=discs)
    G = _clip(G, M.limits)
    res = simulate(hole, lie, G, wind, dt=0.02, mando_cleared=mando_cleared)
    c0 = cost(hole, res, mando_cleared)
    order = np.argsort(c0, kind="stable")[:48]
    cand = {k: v[order] for k, v in G.items()}
    K = 12

    def expected(P):
        Q = _clip(_scattered(P, M.sd, K, rng), {k: v for k, v in M.limits.items() if k != "aim_deg"})
        r = simulate(hole, lie, Q, wind, dt=0.01, mando_cleared=mando_cleared)
        return cost(hole, r, mando_cleared).reshape(-1, K).mean(axis=1)

    e = expected(cand)
    for it in range(2):
        best = np.argsort(e, kind="stable")[:4]
        P = {k: np.repeat(cand[k][best], 30) for k in cand}
        n = len(P["speed"])
        sc = 1.0 / (it + 1)
        P["aim_deg"] = P["aim_deg"] + rng.uniform(-2.0, 2.0, n) * sc
        P["speed"] = P["speed"] + rng.uniform(-1.2, 1.2, n) * sc
        P["launch_deg"] = P["launch_deg"] + rng.uniform(-2.0, 2.0, n) * sc
        P["bank_deg"] = P["bank_deg"] + np.where(np.abs(P["bank_deg"]) > 0 if d >= 22 else False,
                                                 rng.uniform(-6.0, 6.0, n) * sc, 0.0)
        P = _clip(P, M.limits)
        e2 = expected(P)
        cand = {k: np.concatenate([cand[k], P[k]]) for k in cand}
        e = np.concatenate([e, e2])
    i = int(np.argmin(e))
    out = {k: (DISC_NAMES[int(cand[k][i])] if k == "disc" else round(float(cand[k][i]), 3)) for k in KEYS}
    out["spin_rps"] = float(np.sign(out["spin_rps"]) * max(abs(out["spin_rps"]), 4.0))
    out["aim_deg"] = round(((out["aim_deg"] + 180.0) % 360.0) - 180.0, 3)
    out["expected_throws"] = round(float(e[i]), 3)
    return out


class _OpenHole(Hole):
    """The hole without trees (for the straight baseline's own judgement)."""

    def __init__(self, h):
        self.__dict__.update(h.__dict__)

    def blocked(self, x, y):
        return np.zeros(np.shape(x), bool)


# ---------------------------------------------------------------------------------- the images
def tiles(png_bytes, spec):
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    out = {}
    for k, t in spec["image"]["tiles"].items():
        out[k] = im[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]]
    return out


def read_wind(img, spec):
    """The windsock tile (orthographic, straight down, the pole top at the centre): direction = from the centre to
    the sock, projected length L cos(a) -> the hang angle a -> speed = 10 (1 - a / 90)."""
    cam = spec["cameras"]["wind"]
    ext = float(cam["scale"].split()[0])
    h, w = img.shape[:2]
    r, g, b = (img[..., i].astype(int) for i in range(3))
    orange = (r > 170) & (r - b > 110) & (g < 190)
    white = (r > 205) & (g > 205) & (b > 205)
    sock = orange | white
    vv, uu = np.nonzero(sock)
    if len(uu) < 20:
        return (0.0, 0.0)
    s = ext / h
    X = (uu + 0.5 - w / 2) * s
    Y = -(vv + 0.5 - h / 2) * s
    rr = np.hypot(X, Y)
    far = rr > 0.12
    if far.sum() < 5:
        return (0.0, 0.0)
    ang = math.atan2(float(np.mean(Y[far])), float(np.mean(X[far])))
    proj = X * math.cos(ang) + Y * math.sin(ang)
    L = float(np.percentile(proj, 99.5)) - 0.05          # the sock's end is about 0.12 m across
    c = min(1.0, max(0.0, L / 1.2))
    a = math.degrees(math.acos(c))
    sp = 10.0 * (1.0 - a / 90.0)
    return (sp * math.cos(ang), sp * math.sin(ang))
