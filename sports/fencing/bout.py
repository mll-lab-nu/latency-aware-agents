"""The referee of an epee pool bout (FIE Technical Rules, August 2026: t.22, t.25, t.31, t.34, t.37, t.38, t.44,
t.92-t.94).

A Bout owns the World. Between 'Play!' and 'Halt!' the caller steps it with both fencers' joint targets (Bout.step);
the Phrase referee in fencing.py registers touches (point force >= 7.35 N along the blade for >= 2 ms on the
opponent's body; the 45 ms lockout makes doubles), the rear limit (t.34) and corps-a-corps (t.25). At a halt the bout
judges it and decides where the fencers are put on guard again; Bout.resume() places them (frozen time: the review)
and fencing goes on.

Format (t.38, pools): 5 touches or 3 minutes of fencing time (t.37). A double touch scores for both (t.92), except at
4-4, where it is not counted and the fencers stay where they are (t.38.1.b); in the extra minute a double scores for
both like any double, which leaves the scores equal (the minute is for a deciding hit, t.38.2.b). At time
the fencer ahead wins; at equal scores one more minute is fenced for a deciding touch, the priority drawn by lot
before it (seeded); if the scores are still equal after the minute, the fencer with priority wins (t.38.2.b). There
are no draws. Placement: after a valid touch, on the on-guard lines (t.22.6); after a halt without a touch, where the
fencers were, each retreating the same distance to the on-guard distance of 4 m between the carts (t.22.7, t.31), not
behind the rear limit if in front of it (t.22.9); the extra minute starts on the on-guard lines (t.22.8).

Unwillingness to fight (t.124, deviation D9: the FIE applies it to direct-elimination and team bouts, we apply it to
this pool bout): one minute of fencing time (PASSIVE_S = 60 s of the bout clock, 6000 control steps) without a hit ->
'Halt!'. The first time both fencers get a P-red card, a penalty hit each (t.124.1.a); the second time the fencer with
the lower score gets the P-black card and loses the bout, at equal scores the one who loses a seeded lot (t.124.1.b:
lots for fencers not in the FIE ranking). No P-card at 4-4 (our 14-14, t.124.3.a): the halt is called and the minute
restarts. The minute restarts after each hit, double, annulled double and penalty hit (t.124.3.d; a rear-limit
crossing is a hit, t.34), not after a corps-a-corps; t.124 does not apply in the extra minute (t.124.3.f).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

import fencing as F

BOUT_S, EXTRA_S, BOUT_TOUCHES = 180.0, 60.0, 5
BOUT_STEPS = int(round(BOUT_S / F.CONTROL_DT))
EXTRA_STEPS = int(round(EXTRA_S / F.CONTROL_DT))
PASSIVE_S = 60.0                  # t.124: one minute of fencing (time, t.37) without a hit (D9)
PASSIVE_STEPS = int(round(PASSIVE_S / F.CONTROL_DT))
OTHER = {"A": "B", "B": "A"}
ON_GUARD_XS = {"A": -F.ON_GUARD, "B": -F.ON_GUARD}


def replace_on_guard(end):
    """t.22.7 / t.31 / t.22.9: no touch awarded -> where they were, each retreating the same distance to the on-guard
    distance (4 m between the carts); a fencer in front of the rear limit is not put behind it (the other retreats
    more instead)."""
    g = -(end["A"] + end["B"])                       # distance between the carts (own-frame x of each is negative)
    need = max(0.0, 2 * F.ON_GUARD - g)
    x = {"A": end["A"] - need / 2, "B": end["B"] - need / 2}
    lim = -F.PISTE_HALF + F.CART_HALF + 0.01         # the cart still entirely in front of the rear limit
    for me in ("A", "B"):
        ot = OTHER[me]
        if x[me] < lim and end[me] >= lim:
            short = lim - x[me]
            x[me], x[ot] = lim, x[ot] - short
    return {s: round(float(v), 6) for s, v in x.items()}


class Bout:
    passive_steps = PASSIVE_STEPS       # None: no t.124 (the rules up to fencing-4; the exploit test compares)

    def __init__(self, seed: int, world: F.World | None = None):
        self.seed = int(seed)
        self.w = world or F.World()
        self.score = {"A": 0, "B": 0}
        self.k = 0                      # control steps fenced in this bout (the fencing time is k x 10 ms)
        self.extra = False
        self.priority = None
        self.phase = "on_guard"         # on_guard (frozen, before 'Play!') | fencing | halt | over
        self.halts: list[dict] = []
        self.winner = None
        self.how = None
        self.actions = {"A": [], "B": []}
        self.place = dict(ON_GUARD_XS)
        self.w.place(self.place)
        self.phrase = F.Phrase(self.w)
        self.phrase_start_k = 0
        self.minute_k0 = 0              # t.124.3.d: where the current minute without a hit started (fencing steps)
        self.p_calls = 0                # P-cards given for unwillingness to fight (1: P-red to both; 2: P-black)

    # ---- time
    def limit_steps(self):
        return BOUT_STEPS + (EXTRA_STEPS if self.extra else 0)

    def time_left_s(self):
        return round(max(0, self.limit_steps() - self.k) * F.CONTROL_DT, 2)

    def no_hit_s(self):
        """t.124: the fencing time since the minute restarted (None where t.124 does not apply: the extra minute)"""
        if self.passive_steps is None or self.extra:
            return None
        return round((self.k - self.minute_k0) * F.CONTROL_DT, 2)

    # ---- fencing
    def fence(self):
        """'Play!' (t.22.11, t.23.1): from on_guard (or after a halt's placement) the bout runs"""
        assert self.phase in ("on_guard",), self.phase
        self.phase = "fencing"
        self.phrase_start_k = self.k

    def step(self, targets: dict) -> dict | None:
        """one control step of fencing with both fencers' 8 joint targets; returns the halt record or None"""
        assert self.phase == "fencing", self.phase
        for s in F.SEATS:
            t = targets[s]
            self.actions[s].append(np.r_[t["x"], t["q"][-1]] if isinstance(t, dict) else np.asarray(t, float).copy())
        rec = self.w.step(targets)
        self.k += 1
        h = self.phrase.update(rec, self.w.k - 1)
        if h is None and self.k >= self.limit_steps():
            # t.44.1: the clock cuts off the apparatus; a touch already registered counts (its lockout window ends)
            if self.phrase.first is not None:
                h = self.phrase._halt("touch", self.w.k - 1)
            else:
                h = self.phrase._halt("time", self.w.k - 1)
        elif (h is None and self.passive_steps is not None and not self.extra and self.phrase.first is None
              and self.k - self.minute_k0 >= self.passive_steps):
            h = self.phrase._halt("passivity", self.w.k - 1)  # t.124: 'Halt!' as a decision of fact (t.136.2)
        if h is not None:
            self._judge(h)
        return self.halts[-1] if h is not None else None

    def _judge(self, h):
        end = {s: self.w.cart_x(s) for s in F.SEATS}
        tch = h.get("touches", {})
        awarded, annulled = [], False
        if h["why"] == "touch":
            if len(tch) == 2:
                if self.score["A"] == self.score["B"] == BOUT_TOUCHES - 1:
                    annulled = True                      # t.38.1.b: at 4-4 a double hit is not counted
                else:
                    awarded = ["A", "B"]                 # t.92: a double touch counts for both
            else:
                awarded = [next(iter(tch))]
        elif h["why"] == "rear_limit":
            awarded = [OTHER[h["seat"]]]                 # t.34
        cards, black = {}, None
        if h["why"] == "passivity":                      # t.124 (D9)
            a, b = self.score["A"], self.score["B"]
            if a == b == BOUT_TOUCHES - 1:
                pass                                     # t.124.3.a: no P-card at 4-4 (14-14); the minute restarts
            elif self.p_calls == 0:
                self.p_calls = 1                         # t.124.1.a: P-red to both, a penalty hit each
                cards = {"A": "P-red", "B": "P-red"}
                awarded = ["A", "B"]
            else:
                self.p_calls = 2                         # t.124.1.b: P-black to the lower score, by lot if equal
                if a != b:
                    black = "A" if a < b else "B"
                else:
                    black = "A" if np.random.default_rng([self.seed, 124]).random() < 0.5 else "B"
                cards = {black: "P-black"}
        for s in awarded:
            self.score[s] += 1
        rec = dict(n=len(self.halts) + 1, why=h["why"], k=self.k, phrase_s=round((self.k - self.phrase_start_k) *
                   F.CONTROL_DT, 2), touches=tch, awarded=awarded, double=len(tch) == 2, annulled=annulled,
                   score=dict(self.score), carts={s: round(v, 4) for s, v in end.items()}, extra=self.extra)
        if h.get("seat"):
            rec["seat"] = h["seat"]
        if h["why"] == "passivity":
            rec["no_hit_s"] = round((self.k - self.minute_k0) * F.CONTROL_DT, 2)
            rec["cards"] = cards
        if awarded or annulled or h["why"] == "passivity":
            self.minute_k0 = self.k                      # t.124.3.d: the minute starts again
        # the end of the bout
        a, b = self.score["A"], self.score["B"]
        time_up = self.k >= self.limit_steps()
        if black is not None:
            self.winner, self.how = OTHER[black], "passivity"
        elif self.extra and a != b:
            self.winner, self.how = ("A" if a > b else "B"), "deciding_touch"
        elif not self.extra and max(a, b) >= BOUT_TOUCHES and a != b:
            self.winner, self.how = ("A" if a > b else "B"), "touches"
        elif time_up and not self.extra:
            if a != b:
                self.winner, self.how = ("A" if a > b else "B"), "time"
            else:                                        # t.38.2.b: one more minute, priority drawn by lot first
                self.extra = True
                self.priority = "A" if np.random.default_rng([self.seed, 38]).random() < 0.5 else "B"
                rec["priority"] = self.priority
        elif time_up and self.extra:
            self.winner, self.how = self.priority, "priority"
        if self.winner is not None:
            self.phase = "over"
            rec["winner"], rec["how"] = self.winner, self.how
        else:
            self.phase = "halt"
            if awarded or rec.get("priority"):
                self.place = dict(ON_GUARD_XS)            # t.22.6 / t.22.8
            else:
                self.place = replace_on_guard(end)        # t.22.7 / t.31 (also an annulled double, t.38.1.b)
            rec["next_on_guard"] = self.place
        self.halts.append(rec)

    def resume(self):
        """the referee puts the fencers on guard (frozen time); 'Play!' follows with fence()"""
        assert self.phase == "halt", self.phase
        self.w.place(self.place)
        self.phrase = F.Phrase(self.w)
        self.phase = "on_guard"

    def state_sha(self):
        h = hashlib.sha256(self.w.state_sha().encode())
        h.update(json.dumps([self.score, self.k, self.extra, self.priority, self.halts], sort_keys=True,
                            default=str).encode())
        return h.hexdigest()
