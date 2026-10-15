"""Disc golf for the turn service (server.py): a round of 6 holes per seat (discgolf.py: the course, the rules and the
rendering; flight.py: the disc's flight).

In a duel both players play the same course with their own discs, in the same group, and the discs never meet. The
order of play is PDGA 802.02: on the first tee FIRST_PLAYER throws first; on later tees the lower score on the
previous hole throws first (a tie keeps the previous order); once both have thrown from the tee, the player whose lie
is farther from the basket (the away player) throws. The n-th throw of either player on a hole meets the same wind and
the same execution scatter (seeded by seed, hole and n). A player sees only its own disc; it hears the opponent's
results and score. Fewest throws wins (PDGA 808); a tie goes to a sudden-death play-off (PDGA Competition Manual
1.09: holes 1, 2, ... again), then the chess clock.
"""
import math
import os

SEED_DEFAULT = 20261005                      # the default course (discgolf.BASE_SEED)
SEED = int(os.environ.get("TURN_SEED") or SEED_DEFAULT)
os.environ["DISCGOLF_SEED"] = str(SEED)
import discgolf as D  # noqa: E402
import flight as FL  # noqa: E402

NAME = "disc golf"
LOWER_WINS = True
PAR_TOTAL = D.PAR_TOTAL
SUCCESS_MAX_TOTAL = 21                       # single player: the success line, par + 1 (spec.py)
ACTION_KEYS = ("disc_id", "speed", "launch_deg", "nose_deg", "bank_deg", "spin_rps", "aim_deg")
TILES = {"overview": (0, 0, D.OV_W, D.OV_H), "tee": (0, D.OV_H, D.TEE_W, D.TEE_H),
         "wind": (D.TEE_W, D.OV_H, D.WIND_W, D.WIND_H)}
IMAGE_SIZE = (D.OV_W, D.OV_H + D.TEE_H)
IMAGE_LAYOUT = (f"one PNG, {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}: top, overview ({D.OV_W} x {D.OV_H}, orthographic, "
                f"straight down on the whole hole; image right = +x, up = +y; the flight of your previous throw on this "
                f"hole as yellow dots); bottom left at y = {D.OV_H}, tee ({D.TEE_W} x {D.TEE_H}, from behind your "
                f"launcher towards the basket); bottom right at x = {D.TEE_W}, y = {D.OV_H}, wind ({D.WIND_W} x "
                f"{D.WIND_H}, orthographic, straight down on the windsock; the rest of that corner is black)")


def validate(body: dict):
    if body.get("type") != "throw":
        return ("the action must be {'type': 'throw', 'disc': 'putter'|'midrange'|'driver', 'speed': v, "
                "'launch_deg': a, 'nose_deg': n, 'bank_deg': b, 'spin_rps': s, 'aim_deg': h}")
    disc = body.get("disc")
    if disc not in FL.DISC_NAMES:
        return f"disc must be one of {FL.DISC_NAMES}"
    out = {"disc": disc, "disc_id": FL.DISC_NAMES.index(disc)}
    for k, (lo, hi) in D.LIMITS.items():
        try:
            v = float(body[k])
        except (KeyError, TypeError, ValueError):
            return f"throw needs {', '.join(D.LIMITS)} (numbers) and disc"
        if not math.isfinite(v) or not lo <= v <= hi:
            return f"{k} must be a number in [{lo:g}, {hi:g}]"
        out[k] = round(v, 4)
    if abs(out["spin_rps"]) < D.MIN_SPIN:
        return f"|spin_rps| must be at least {D.MIN_SPIN:g} (the launcher always spins the disc)"
    return out


def hole_scores(c, clock_out=False) -> list:
    """Per hole: throws + penalty throws (par + 4 for a hole ended after par + 3); when the clock ran out, the
    unfinished hole and the rest score par + 4."""
    out = []
    for i, h in enumerate(c.order):
        if c.finished(i):
            out.append(c.hole_score(i))
        elif clock_out:
            out.append(D.HOLES[h]["par"] + D.MAX_OVER_PAR + 1)
        else:
            out.append(None)
    return out


TIEBREAK_MAX = 3                 # duel: sudden-death play-off holes (holes 1-3) before the chess clock decides
TIEBREAK_NAME = "play-off hole"


def playoff_course(n):
    """The n-th sudden-death play-off hole (0-based): hole n % 6 played again, with its own wind draws."""
    return D.Course(holes=[n % D.N_HOLES], key_base=100 + n)


def score_of(c, clock_out=False, playoffs=()) -> dict:
    hs = hole_scores(c, clock_out)
    return {"total": sum(x for x in hs if x is not None), "holes": hs, "throws_per_hole": list(c.throws),
            "penalties_per_hole": list(c.penalties), "ended_unfinished": list(c.ended), "complete": c.done,
            "par": PAR_TOTAL,
            "par_of_holes_scored": sum(D.HOLES[h]["par"] for h, x in zip(c.order, hs) if x is not None),
            "playoff": [{"hole": p.order[0] + 1, "score": p.hole_score(0) if p.done else None, "complete": p.done}
                        for p in playoffs]}


def decide(scores: dict):
    """The seat ahead (fewer throws) after the round and any complete play-off holes, or None while level."""
    a, b = scores
    if scores[a]["total"] != scores[b]["total"]:
        return a if scores[a]["total"] < scores[b]["total"] else b
    for pa, pb in zip(scores[a]["playoff"], scores[b]["playoff"]):
        if pa["complete"] and pb["complete"] and pa["score"] != pb["score"]:
            return a if pa["score"] < pb["score"] else b
    return None


def success(score: dict) -> bool:
    return score["complete"] and score["total"] <= SUCCESS_MAX_TOTAL


class Match:
    """Per seat: the round, then (duel, tied) the sudden-death play-off holes; the PDGA order of play."""

    def __init__(self, seats, order):
        self.c = {s: D.Course() for s in seats}
        self.p = {s: [] for s in seats}
        self.v = None

    @property
    def n_tiebreaks(self):
        return len(next(iter(self.p.values())))

    def done(self, s):
        return self.p[s][-1].done if self.p[s] else self.c[s].done

    def _cur(self, s):
        return self.p[s][-1] if self.p[s] else self.c[s]

    def _units(self, s):
        c = self.c[s]
        return (D.N_HOLES if c.done else c.i) + sum(1 for p in self.p[s] if p.done)

    def _unit_score(self, s, u):
        if u < D.N_HOLES:
            return self.c[s].hole_score(u)
        return self.p[s][u - D.N_HOLES].hole_score(0)

    def _thrown(self, s):
        c = self._cur(s)
        return c.count() > 0

    def _dist(self, s):
        c = self._cur(s)
        bx, by = D.HOLES[c.hole]["basket"]
        return math.hypot(c.lie[0] - bx, c.lie[1] - by)

    def tee_order(self, u, order):
        """PDGA 802.02: the first tee in the given order; later tees by the previous hole's score (lower first), a
        tie keeping the previous tee order."""
        o = list(order)
        for k in range(1, u + 1):
            prev = o
            o = sorted(prev, key=lambda s: (self._unit_score(s, k - 1), prev.index(s)))
        return o

    def turn(self, order):
        live = [s for s in order if not self.done(s)]
        if not live:
            return None
        u = min(self._units(s) for s in live)
        cands = [s for s in live if self._units(s) == u]
        tee = [s for s in self.tee_order(u, order) if s in cands]
        fresh = [s for s in tee if not self._thrown(s)]
        if fresh:                                  # everyone tees off first, in the tee order
            return fresh[0]
        return max(tee, key=lambda s: (round(self._dist(s), 6), -tee.index(s)))   # then the away player

    def winner(self):
        return decide({s: self.score(s) for s in self.c})

    def start_tiebreak(self):
        n = self.n_tiebreaks
        for s in self.p:
            self.p[s].append(playoff_course(n))

    def shoot(self, s, dec):
        c = self._cur(s)
        rec, _ = c.play(dec)
        out = dict(rec)
        if self.p[s]:
            out["playoff_hole"] = len(self.p[s])
        else:
            out["total_so_far"] = sum(x for x in hole_scores(c) if x is not None)
        return out

    def decision(self, s):
        c = self._cur(s)
        H = D.HOLES[c.hole]
        bx, by = H["basket"]
        d = {"hole": c.hole + 1, "par": H["par"], "throw": c.count() + 1,
             "throws_left_before_cap": H["par"] + D.MAX_OVER_PAR - c.count(),
             "lie": [round(c.lie[0], 3), round(c.lie[1], 3)], "lie_ground_z_m": round(D.ground(H, *c.lie), 3),
             "basket": [bx, by], "distance_to_basket_m": round(math.hypot(c.lie[0] - bx, c.lie[1] - by), 2),
             "heading_to_basket_deg": round(D.heading_to_basket(H, c.lie), 3)}
        if H.get("mando"):
            d["mandatory_cleared"] = c.mando_cleared
        if self.p[s]:
            d["playoff_hole"] = len(self.p[s])
        if c.last:
            L = {k: v for k, v in c.last.items() if k not in ("trail", "wind")}
            d["last_throw"] = L
        return d

    def views(self, s):
        if self.v is None:
            self.v = D.Views()
        c = self._cur(s)
        wind = D.wind_of(c.key(), c.throws[c.i])
        return self.v.render(c.hole, c.lie, wind, trail=c.last["trail"] if c.last else None)

    def summary(self, s):
        c = self.c[s]
        hs = hole_scores(c)
        done = [x for x in hs if x is not None]
        out = {"total": sum(done), "holes_done": D.N_HOLES if c.done else c.i,
               "par_of_holes_done": sum(D.HOLES[h]["par"] for h in c.order[:len(done)]), "hole_scores": hs}
        if self.p[s]:
            out["playoff"] = [{"hole": p.order[0] + 1, "score": p.hole_score(0) if p.done else None} for p in self.p[s]]
        return out

    def score(self, s, clock_out=False):
        return score_of(self.c[s], clock_out, self.p[s])

    def state_sha(self, s):
        return "/".join([self.c[s].state_sha()] + [p.state_sha() for p in self.p[s]])


def _act(a):
    return {k: a[k] for k in ("disc", "speed", "launch_deg", "nose_deg", "bank_deg", "spin_rps", "aim_deg")}


def replay(actions, clock_out=False, n_tiebreaks=0) -> dict:
    c = D.Course()
    ps = [playoff_course(k) for k in range(n_tiebreaks)]
    i = -1
    for a in actions:
        if i < 0 and not c.done:
            c.play(_act(a))
            continue
        i = max(i, 0)
        while ps[i].done:
            i += 1
        ps[i].play(_act(a))
    return {"state_sha": "/".join([c.state_sha()] + [p.state_sha() for p in ps]), "score": score_of(c, clock_out, ps)}


def render_video(actions, out):
    n = 0
    c = D.Course()
    for a in actions:                                   # the round only (play-off throws are not shown)
        if c.done:
            break
        c.play(_act(a))
        n += 1
    D.render_video([_act(a) for a in actions[:n]], out)


def spec(duel: bool) -> dict:
    rnd = {"holes": D.spec_holes(), "par_total": PAR_TOTAL,
           "rulebook": "PDGA Official Rules of Disc Golf (rev. 2026-01-01); stroke play (808)",
           "cap": f"a hole not holed out after par + {D.MAX_OVER_PAR} throws (penalty throws included) ends and scores "
                  f"par + {D.MAX_OVER_PAR + 1}",
           "clock_out": f"the unfinished hole and the holes not played score par + {D.MAX_OVER_PAR + 1} each"}
    if duel:
        rnd["duel"] = ("both players play the same course with their own discs (the discs never meet), in one group: "
                       "on the first tee the first player throws first, on later tees the lower score on the previous "
                       "hole first (a tie keeps the order); after both tee shots the player farther from the basket "
                       "throws (PDGA 802.02); the n-th throw of either player on a hole meets the same wind and the "
                       "same execution scatter; fewest throws wins")
        rnd["tie"] = (f"sudden-death play-off (PDGA Competition Manual 1.09): holes 1, 2, 3 are played again, each "
                      f"scored alone; the lower score on a play-off hole wins; at most {TIEBREAK_MAX} play-off holes; "
                      "if still level, the player with more chess-clock time left wins")
    else:
        rnd["success"] = f"round complete and total <= {SUCCESS_MAX_TOTAL} (par {PAR_TOTAL} + 1); fewer throws is better"
    return {
        "simulator": "deterministic disc-flight model (Hummel 2003 lift / drag / pitching moment, gyroscopic "
                     "precession, spin-down; RK4, dt 0.01 s) with trees, basket and ground contact; MuJoCo 3.3.7 scene "
                     "for rendering (osmesa)",
        "match": rnd,
        "throw": {"command": "{'type':'throw','disc':D,'speed':v,'launch_deg':a,'nose_deg':n,'bank_deg':b,'spin_rps':s,"
                             "'aim_deg':h}",
                  "limits": {k: list(v) for k, v in D.LIMITS.items()}, "min_abs_spin_rps": D.MIN_SPIN,
                  "scatter_sd": dict(D.SCATTER),
                  "release": f"from {D.RELEASE_H} m above the ground at the lie"},
        "discs": {k: dict(v, mass_kg=FL.MASS, diameter_m=FL.DIAM, Izz_kg_m2=FL.IZZ) for k, v in FL.DISCS.items()},
        "air": {"rho_kg_m3": FL.RHO, "g_m_s2": FL.G,
                "wind": "uniform and horizontal over the hole, constant during a flight; each hole has its own wind "
                        "and it gusts from throw to throw; the windsock shows the wind of the throw you are about to "
                        "make"},
        "contacts": {"trunk_radius_m": D.TRUNK_R, "canopy_base_m": D.CANOPY_BASE, "trunk_restitution": [D.TRUNK_E, D.TRUNK_T],
                     "canopy_keep": D.CANOPY_KEEP, "tree_heights_m": [12.0, 22.0],
                     "basket": {"catch_radius_m": D.CATCH_R, "chains_m": [D.CHAIN_LO, D.CHAIN_HI],
                                "max_catch_speed_m_s": D.CATCH_V, "cage_radius_m": D.CAGE_R,
                                "lid": [D.LID_R, D.CHAIN_HI + 0.12]},
                     "ground": {"mu_impact": D.MU_I, "mu_kinetic": D.MU_K}},
        "cameras": D.spec_cameras(),
        "world": "each hole has its own frame: the tee at (0, 0), x down the fairway, y to the left, z up (metres)",
    }
