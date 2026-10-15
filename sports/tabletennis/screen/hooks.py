"""Table tennis: the screening hooks (tools/screen/README.md; docs/construction_zh.md step 4). Not part of the
simulator (tools/sportlib.py leaves sports/<sport>/screen/ out of a task).

Every player here is the camera-only reference (builtin_players.Reference) with exactly one part changed:

  reference   the reference itself; it compensates the effect delay D (its act takes effect D control steps after the
              observation it was computed from: it assumes D)
  ctrl        a replica that changes nothing that matters (every number of an act rounded to 3 decimals instead of 4:
              < 0.5 mm, < 0.5 ms, far below the 4 mm / 0.6 deg / 2 % execution scatter): the noise floor of a chaotic
              simulation (the validity pilot: 0.1 mm makes the match diverge within a few strokes)

simple baselines (they serve with the reference's own rehearsed serves, so they are stronger than "simple": the gate
is conservative)
  fixed       one fixed act per incoming ball, the same every time (contact time, point, face, blade velocity), sent at
              the first observation of the ball; never looks at the images
  random      one random act per incoming ball (seeded): contact time, point within reach, a forward-facing blade, a
              blade speed up to 2 m/s; never looks at the images
  reflex      perception and prediction as the reference (the contact point and time), but a fixed blade face and
              velocity (no contact law, no aim): it only puts the paddle where the ball will be
  center      the reference aiming at the middle (aim_x = 0: no placement away from the opponent)
  short_serve only the reference's two short serves (short_centre_top, short_bh_top), and the fixed act for every
              incoming ball (no returns): the shortcut the step-5 model runs found (serve short, never rally)

skill necessity (the high level's actions other than the core skill `act`/hit: move, toss, none)
  no_hit      serves its tosses and moves its base towards the incoming ball's side (the reference's perception), but
              never sends an act

capability sensitivity (docs/construction_zh.md "能力敏感性"; the validity pilot's impairments,
the review notes measurement/pilot_tt_validity_zh.md, ported to revision 11/9)
  a1_s1, a1_s2   A1 3-D perception: Gaussian noise of 1 / 2 px on every detected ball centroid before triangulation
  a2_drag        A2 physics (weak): the flight model without air drag
  a2_bounce      A2 physics (strong): a wrong table bounce (vertical restitution 0.80 instead of 0.96, no tangential
                 loss)
  a4_m2, a4_m3   A4 time: the latency under-compensated by 2 / 3 control steps (20 / 30 ms)
  b2_rand        B2 opponent (weak): never looks at the opponent; the side is a seeded coin flip per ball
  b2_fixed       B2 opponent (strong): never looks at the opponent; always the same corner
"""
from __future__ import annotations

import math

import numpy as np

import builtin_players as bp

FORMAT = "match"
ACTION_LEVEL = "high"
DELAY = 5                         # the effect delay (control steps): about 1 s of thinking at the x20 slowdown
LEVELS = [1, 2, 3, 4]             # sport.toml [official] levels
REFERENCE = "reference"
REPLICAS = ["ctrl"]
BASELINES = ["fixed", "random", "reflex", "center", "short_serve"]
NECESSITY = ["no_hit"]
SENSITIVITY = {"A1": ["a1_s1", "a1_s2"], "A2": ["a2_drag", "a2_bounce"], "A4": ["a4_m2", "a4_m3"],
               "B2": ["b2_rand", "b2_fixed"]}
TRUTH = ["reference_truth", "reflex_truth", "center_truth"]
REFERENCE_TRUTH = "reference_truth"
TRUTH_OF = {"reflex": "reflex_truth", "center": "center_truth"}   # fixed, random, short_serve do not perceive   # the same players with the ball's true position
VARIANTS = ([REFERENCE] + REPLICAS + BASELINES + NECESSITY + [v for vs in SENSITIVITY.values() for v in vs] + TRUTH)
NO_IMAGES = {"fixed", "random", "short_serve", "reference_truth", "reflex_truth", "center_truth"}

FIXED_ACT = {"t": 0.62, "pos": [0.0, -1.42, 0.22], "normal": [0.0, 0.95, 0.31], "vel": [0.0, 1.2, 0.5]}
REFLEX_N = np.array([0.0, 0.95, 0.31]) / np.linalg.norm([0.0, 0.95, 0.31])
REFLEX_V = np.array([0.0, 1.2, 0.5])


def prepare(variant, opts):
    """Process-wide changes of the player's own model (each match runs in its own process)."""
    if variant == "short_serve":
        bp.SERVE_ORDER = ("short_centre_top", "short_bh_top")
    if variant == "a2_drag":
        bp.KD = 1e-6                    # no drag (not 0: toss_speed divides by it); the serve planning uses the same model
    if variant == "a2_bounce":
        bp.E_BOUNCE_Z, bp.E_BOUNCE_XY = 0.74, 1.0   # the true bounce (0.90, 0.87) off by the same amounts as before revision 14 (0.96 -> 0.80)
    if variant.startswith("a1_s"):
        _install_noisy_detector(float(variant[4:]))


_NOISE = {"sigma": 0.0, "seed": 0, "draws": {}}


def _install_noisy_detector(sigma):
    """builtin_players.detect_views with N(0, sigma) px on each detected centroid; one draw per (ball, camera, frame)
    so that repeated views of a frame agree, as a real detector would."""
    _NOISE["sigma"] = sigma
    orig_tile = bp.detect_tile

    def detect_views(img, cams, times, max_res=0.03):
        out = []
        for j, tt in enumerate(times):
            rays = []
            for i, c in enumerate(cams):
                tile = img[i * bp.TILE_H:(i + 1) * bp.TILE_H, j * bp.TILE_W:(j + 1) * bp.TILE_W]
                uv = orig_tile(tile)
                if uv is not None:
                    key = (_NOISE.get("launch"), i, round(float(tt), 3))
                    if key not in _NOISE["draws"]:
                        r = np.random.default_rng([_NOISE["seed"], 7, abs(hash(key)) % (2 ** 31)])
                        _NOISE["draws"][key] = r.normal(0.0, 1.0, 2)
                    du, dv = _NOISE["draws"][key] * sigma
                    rays.append(c.ray(uv[0] + du, uv[1] + dv))
            if len(rays) == 2:
                X, res = bp.triangulate(rays)
                if res < max_res and -0.2 < X[2] < 2.0 and abs(X[1]) < 3.0:
                    out.append((float(tt), X))
        return out
    bp.detect_views = detect_views


class VPlayer(bp.Player):
    """The reference's Player with the side choice as a hook (B2) and the noisy detector's ball key (A1)."""

    variant = "reference"
    seed = 0

    truth = False

    def update(self, obs, img, k_req):
        _NOISE["launch"] = int(round(k_req - float(obs["t_since_played"]) / bp.DT))   # the A1 noise's ball key
        if not self.truth:
            return super().update(obs, img, k_req)
        L = super().update(obs, None, k_req)        # truth perception: the ball's true position in the frames the
        for k, pos in (obs.get("ball_true") or []):  # cameras would show (the same steps), on this ball's clock
            t = round((int(k) - L["launch"]) * bp.DT, 3)
            if not L["done"] and t > 0.005:
                L["frames"][t] = np.asarray(pos, float)
        return L

    def _side(self, p_hit):
        v = self.variant
        if v == "b2_fixed":
            return 1.0
        if v == "b2_rand":
            r = np.random.default_rng([self.seed, 13, int(self.leg["launch"])])
            return 1.0 if r.uniform() < 0.5 else -1.0
        ox = self.opp_x()
        return -1.0 if ox > 0.05 else (1.0 if ox < -0.05 else (-1.0 if p_hit[0] > 0 else 1.0))

    def stroke(self, p_hit, v_in, vp_max=None):
        """= builtin_players.Player.stroke with the side from _side() (reflex: a fixed face and blade velocity)."""
        if self.variant == "reflex":
            self._chosen = (None, None)
            return np.asarray(p_hit) - bp.OFFSET * REFLEX_N, REFLEX_N.copy(), REFLEX_V.copy(), np.array([0.0, 0.85])
        side = self._side(p_hit)
        aim = np.array([side * self.aim_x, self.aim_y])
        best = None
        cap = self.vp_max if vp_max is None else vp_max
        L = self.leg
        last = L["acts"][-1] if (L and L["acts"]) else None
        vys = self.vys
        if last is not None and last.get("vy") is not None and last.get("side") == side:
            vys = (last["vy"],) + tuple(v for v in self.vys if v != last["vy"])
            cap = cap * 1.1
        for vy in vys:
            v_out, ok = bp.solve_out(p_hit, aim, vy)
            if not ok:
                continue
            p_c, n, v_p = bp.paddle_for(p_hit, v_in, v_out, self.tangential)
            sp = float(np.linalg.norm(v_p))
            if sp <= cap:
                self._chosen = (vy, side)
                return p_c, n, v_p, aim
            if best is None or sp < best[0]:
                best = (sp, p_c, n, v_p)
        if best is None:
            v_out, _ = bp.solve_out(p_hit, aim, self.vys[-1])
            p_c, n, v_p = bp.paddle_for(p_hit, v_in, v_out, self.tangential)
            return p_c, n, v_p, aim
        return best[1], best[2], best[3], aim


class Variant(bp.Reference):
    def __init__(self, spec, seat, seed, variant, delay):
        super().__init__(spec, seat, seed)
        self.variant, self.seed, self.delay = variant, int(seed), int(delay)
        _NOISE["seed"] = int(seed)
        P = VPlayer([bp.Cam(c) for c in spec["cameras"]], vp_max=bp.VP_MAX)
        P.variant, P.seed = variant.replace("_truth", ""), int(seed)
        P.truth = variant.endswith("_truth")
        if variant.startswith("center"):
            P.aim_x = 0.0
        self.P = P
        self.assumed = delay - (int(variant[4:]) if variant.startswith("a4_m") else 0)
        self._ball_sent = None
        self.rng = np.random.default_rng([int(seed), 101])

    def _clock(self, st, ob):
        step = float(st["step"]) + self.assumed            # where the player believes its act will take effect
        return lambda: step

    def act(self, view):
        st = view["state"]
        incoming = st.get("ball_state") == "coming to you" and bool(st.get("ball_in_play"))
        v = self.variant
        if v in ("fixed", "random", "short_serve"):
            if incoming:
                key = (st.get("point"), st.get("ball"))
                launch = int(round(float(st["step"]) - float(st.get("t_since_played") or 0) / bp.DT))
                if self._ball_sent == (key, launch):
                    return None
                self._ball_sent = (key, launch)
                return {"type": "act", **(self._random_act() if v == "random" else FIXED_ACT)}
            return super().act(view)
        a = super().act(view)
        if v == "no_hit":
            if a and a.get("type") == "act":
                if not incoming:
                    return None                              # (the service's strike: never)
                # receiving: follow the ball's predicted contact with the base only (a move, not a stroke)
                bx = float(np.clip(a["pos"][0], -1.0, 1.0))
                return {"type": "move", "base": [round(bx, 3), -1.6]}
            return a
        if a and v == "ctrl" and a.get("type") == "act":
            a = {**a, "t": round(a["t"], 3), "pos": [round(x, 3) for x in a["pos"]],
                 "normal": [round(x, 3) for x in a["normal"]], "vel": [round(x, 3) for x in a["vel"]]}
        return a

    def _random_act(self):
        r = self.rng
        n = np.array([r.uniform(-0.4, 0.4), 1.0, r.uniform(-0.2, 0.8)])
        n /= np.linalg.norm(n)
        d = r.normal(size=3)
        vel = d / np.linalg.norm(d) * r.uniform(0.0, 2.0)
        vel[1] = abs(vel[1])
        return {"t": round(float(r.uniform(0.35, 0.95)), 3),
                "pos": [round(float(r.uniform(-0.7, 0.7)), 3), round(float(r.uniform(-1.75, -1.3)), 3),
                        round(float(r.uniform(0.12, 0.45)), 3)],
                "normal": [round(float(x), 4) for x in n], "vel": [round(float(x), 3) for x in vel]}


def patch_match(m, variant, opts):
    """B2 experiment (an opponent change, not a variant of the player): `--opt recover_scale=F` multiplies the
    scripted opponent's recovery speed back to the centre after its hit (match.LEVELS recover) by F, so that it is
    still off-centre when the player hits: placing the ball away from it can pay."""
    sy = opts.get("short_y1")                     # "1:0.62,2:0.58,...": the opponent's short-serve receive limit per level
    if sy and hasattr(m, "R"):
        table = {int(k): float(v) for k, v in (x.split(":") for x in sy.split(","))}
        if int(m.level) in table:
            m.R.lv["short_y1"] = table[int(m.level)]
    lo = opts.get("lv")                           # "1.reach=0.20,2.chase=0.40,...": any level parameter (match.LEVELS)
    if lo and hasattr(m, "R"):
        for item in lo.split(","):
            key, val = item.split("=")
            lvl, name = key.split(".")
            if int(lvl) == int(m.level):
                m.R.lv[name] = float(val)
    f = opts.get("recover_scale")
    if f is not None and hasattr(m, "R"):
        m.R.lv["recover"] = float(m.R.lv["recover"]) * float(f)
        m.R.opp["speed"] = m.R.lv["recover"]


def make_player(variant, spec, seat, seed, delay, opts):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    return Variant(spec, seat, seed, variant, delay)


def augment_view(view, m, seat, player):
    """Truth-perception variants (the user's rule of 2026-10-11: simple baselines with true perception first): the
    ball's true position on the ball's own clock, only for those variants and only while a ball comes to the player."""
    if not str(getattr(player, "variant", "")).endswith("_truth") or not hasattr(m, "R"):
        return
    st = view["state"]
    if st.get("ball_state") == "coming to you" and st.get("ball_in_play"):
        import match as _match                      # the frames the cameras would draw (plugin camera_job's selection)
        R = m.R
        hist = list(R.hist) or [{"k": R.k, "qpos": R.s.d.qpos.copy()}]
        sel = [hist[max(0, len(hist) - 1 - b)] for b in reversed(_match.Match.VIEW_BACK)]
        st["ball_true"] = [(int(h["k"]), [float(x) for x in h["qpos"][0:3]]) for h in sel]


def wants_images(player, m, seat):
    """Speed only: the images are drawn only while a ball is coming to a player that looks at them (the reference
    decodes them only then: Reference.act `incoming`), so skipping them otherwise changes nothing it sees."""
    if getattr(player, "variant", None) in NO_IMAGES:
        return False
    if not hasattr(m, "R"):                       # format duel: always drawn
        return True
    R = m.R
    return bool(R.in_play() and R.leg["state"] == "incoming")


def summary(res, players):
    """The numbers the gates use: won, points for / against, and how the points were won (the serve's share)."""
    ff = res["final_fields"]
    if res.get("format") == "duel":               # the fairness check: who won, who served first
        sc = ff.get("score") or res.get("scores") or {}
        return {"won": res.get("winner") == "A", "first_server": ff.get("first_server"), "score": sc,
                "self_collision": ff.get("self_collision")}
    pts = ff.get("points") or []
    you = [p for p in pts if p.get("winner") == "you"]
    serve_pts = sum(1 for p in you if p.get("server") == "you" and ("too short" in (p.get("how") or "")
                                                                     or "service" in (p.get("how") or "")))
    shots = ff.get("results") or []
    return {"won": bool(ff.get("won")), "points_for": ff.get("points_for"), "points_against": ff.get("points_against"),
            "point_share": (ff.get("points_for") or 0) / max(1, (ff.get("points_for") or 0) + (ff.get("points_against") or 0)),
            "points_won_on_own_serve_unreturned": serve_pts,
            "returns": sum(1 for s in shots if s.get("returned")), "strokes": len(shots),
            "self_collision": ff.get("self_collision")}
