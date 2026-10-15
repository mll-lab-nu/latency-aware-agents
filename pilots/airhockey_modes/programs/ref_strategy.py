"""Our reference STRATEGY for the air-hockey two-mode pilot (seat A, true state): the built-in camera reference's rules
(sports/airhockey/builtin_players.py, revision v5: predictive guard on the defence line, the near-post block, deep
pucks cleared from the goal side, strikes at the part of the goal the opponent's mallet needs longest to reach,
straight or off a rail) expressed in the high-level skill API:

  guard, clearing, repositioning  ->  {"type": "move", "x", "y"}
  a strike                        ->  {"type": "strike", "t", "x", "y", "dx", "dy", "speed"}: the puck's predicted
                                      centre at contact time t, the drive direction, SPEED (the implementation caps it
                                      to what the arm can do on that line)

    s = RefStrategy(); cmds = s.tick(obs)        # obs: the pilot's true-state observation; [] = keep the last command

Used by: the high-level reference player (ref_high_player.py), the low-level competition test (our strategy calls the
implementation under test) and the API test's case generation (its strike requests). numpy only."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ahref as R  # noqa: E402  (a copy of sports/airhockey/builtin_players.py: the reference's geometry and rules)

SPEED = 1.6                 # requested mallet speed at contact (m/s); capped by the implementation to the arm's limit
DAMP = 0.1                  # 1/s: the air cushion's drag on the puck
WIND_EST = R.CONTACT + .09  # where the mallet roughly winds up behind the contact point (for the clear-path check)
STRIKE_TAIL = .35           # s after the contact before the strategy takes over again
LEAD_MIN, LEAD_0, LEAD_K = .5, .45, 1.5      # the strike's lead time: LEAD_0 + LEAD_K x (distance to the wind-up point)


def predict(p, w, T):
    """the puck (local) after T s: drag, side-rail bounces (restitution E_RAIL) folded in; the end rails ignored"""
    s = (1 - math.exp(-DAMP * T)) / DAMP
    u, v = p[0] + w[0] * s, p[1] + w[1] * s
    r = R.RAIL
    for _ in range(4):
        if v > r:
            v = r - (v - r) * R.E_RAIL
        elif v < -r:
            v = -r + (-r - v) * R.E_RAIL
        else:
            break
    return np.array([u, v])


class RefStrategy(R.Reference):
    def __init__(self, goal_width=0.35, seat="A"):
        cam = {"name": "front", "pos": [0, 0, 1], "lookat": [0, 0, 0], "focal_px": 1, "principal_point_px": [0, 0]}
        super().__init__({"cameras": [cam], "geometry": {"goal_width_m": goal_width}}, seat)
        self.last = None
        self.last_t = -1.
        self.strikes = []

    def reset(self):
        super().reset()
        self.strike_until = -1.
        self.strike_req = None

    def cmd_move(self, xy, t):
        x, y = float(xy[0]), float(xy[1])
        if self.last is not None and self.last[0] == "move" and math.hypot(x - self.last[1], y - self.last[2]) < .004 \
                and t - self.last_t < .2:
            return []
        self.last, self.last_t = ("move", x, y), t
        return [{"type": "move", "x": round(x, 4), "y": round(y, 4)}]

    def tick(self, obs):
        t = float(obs["t"])
        if obs.get("rally") != self.rally:
            self.rally = obs.get("rally")
            self.reset()
            self.last = None
        p = np.array([self.own * obs["puck"][0], obs["puck"][1]])
        w = np.array([self.own * obs["puck_v"][0], obs["puck_v"][1]])
        m = np.array([self.own * obs["mallet"][0], obs["mallet"][1]])
        o = np.array([self.own * obs["opp_mallet"][0], obs["opp_mallet"][1]])
        self.w, self.p, self.hist = w, p, [(t, p)]
        self.m_speed = float(np.linalg.norm(obs.get("mallet_v") or [0, 0]))
        self.m_prev = (t, m.copy())
        self.m_now = m
        if p[0] > R.PUCK_R:
            self.t_side = t if self.t_side is None else self.t_side
        else:
            self.t_side = None
        if t < self.strike_until:                    # a strike under way: until the puck has gone (after the contact)
            p0, w0, t0 = self.strike_from
            off = float(np.linalg.norm(p - predict(p0, w0, t - t0)))
            if t < self.strike_req["t"] - .02 and off > .04:     # the puck went elsewhere (touched, bounced): re-plan
                self.strike_until = -1.
            elif t > self.strike_req["t"] + .08 and (w[0] < -.3 or p[0] < 0):
                self.strike_until = -1.
            else:
                return []
        deep = p[0] > .86 and abs(p[1]) < .26 and w[0] > -.6
        if t < self.until:                           # a clearing push under way
            if w[0] < -.4 or p[0] < 0 or (deep and not self.deep_stroke):
                self.until = -1.
            else:
                return []
        self.deep_stroke = deep
        if deep:
            return self.cmd_move(self.clear_deep(p + w * .05, w, m, t), t)
        if p[0] < -.02 or w[0] > .35 or (w[0] < -.3 and p[0] < .3) or (w[0] > .12 and m[0] < p[0] + .05):
            return self.cmd_move(self.guard(p, w), t)
        # attack: strike the puck in the own half
        T = 0.0
        pp = p
        for _ in range(3):                           # the lead time: the mallet's way to the wind-up point
            d = None
            for wind in (R.WIND, .2, .14):
                d = R.choose(pp, o, wind=wind, mouth=self.mouth)
                if d is not None:
                    break
            if d is None:
                break
            W = pp - d * WIND_EST
            T = min(1.5, max(LEAD_MIN, LEAD_0 + LEAD_K * float(np.linalg.norm(m - W))))
            pp = predict(p, w, T)
        if d is None:                                # a corner: no line-up possible
            q = pp - m
            q = q / max(1e-9, float(np.linalg.norm(q)))
            if q[0] > -.2:
                return self.cmd_move(self.guard(p, w) if p[0] < .86 else
                                     self.world(R.X_FAR, float(np.sign(p[1]) or 1.) * self.post()), t)
            self.stroke, self.until = self.world(*(pp + q * .25)), t + .3
            return self.cmd_move(self.stroke, t)
        W = pp - d * WIND_EST
        if R.seg_dist(pp, m, W) < R.CONTACT + .03 and float((m - pp) @ d) > -WIND_EST + .04:
            n = np.array([-d[1], d[0]])                  # the way to the wind-up point runs through the puck:
            s = 1. if float(n @ (m - pp)) >= 0 else -1.  # first beside it, on the mallet's side
            q = pp + n * s * .17
            if not R.inside(*q):
                q = pp - d * .12 + n * s * .14
            if not R.inside(*q):
                q = pp - d * .12 - n * s * .14
            return self.cmd_move(self.world(*q), t)
        if not R.inside(*W):
            return self.cmd_move(self.guard(p, w), t)
        req = {"type": "strike", "t": round(t + T, 4), "x": round(self.own * float(pp[0]), 5), "y": round(float(pp[1]), 5),
               "dx": round(self.own * float(d[0]), 5), "dy": round(float(d[1]), 5), "speed": SPEED}
        self.strike_req, self.strike_until, self.strike_from = req, t + T + STRIKE_TAIL, (p.copy(), w.copy(), t)
        self.last, self.last_t = ("strike",), t
        self.strikes.append(req)
        return [req]
