"""Scripted fencing strategies (privileged state: they see the opponent's cart, point and arm with a reaction
latency). Used by the feasibility study (scripts/olympiad/fencing/feasibility.py) and by the single-player task's
scripted opponent (opponent.py). Every strategy works in its own frame (fencing.py): it stands at x < 0 facing +x.
"""
from __future__ import annotations

import numpy as np

import fencing as F

HAND = np.array([0.50, -0.05, 1.15])          # roughly where the guard is in the on-guard pose (cart frame)
REACH = 1.72                                  # comfortable reach of the point from the cart (u, cart frame)


def snapshot(w):
    """world-frame snapshot of what the scripted players may see of each fencer"""
    d, m = w.d, w.m
    out = {}
    for s in F.SEATS:
        g = lambda n: d.geom_xpos[m.geom(f"{s}_{n}").id].copy()
        out[s] = dict(x=float(d.qpos[w.qa[s][0]]), v=float(d.qvel[w.va[s][0]]),
                      tip=d.site_xpos[w.tip_site[s]].copy(),
                      guard=d.site_xpos[m.site(f"{s}_guard_s").id].copy(),
                      fore=0.5 * (g("link6_c") + g("link7_c")), elbow=g("link4_c"), upper=g("link3_c"),
                      trunk=np.array([float(d.qpos[w.qa[s][0]]), 0.0, 0.75]),
                      foot=np.array([float(d.qpos[w.qa[s][0]]), 0.0, 0.10]))
    return out


def view(snap, me):
    """a snapshot in `me`'s own frame"""
    sg = F.SIGN[me]
    res = {}
    for s, v in snap.items():
        r = {}
        for k, a in v.items():
            if isinstance(a, np.ndarray):
                r[k] = np.array([sg * a[0], sg * a[1], a[2]])
            else:
                r[k] = sg * a
        res["me" if s == me else "op"] = r
    return res


class Strategy:
    name = "base"

    def __init__(self, ctl, rng):
        self.c, self.rng, self.st, self.mem = ctl, rng, "start", {}
        self.thrusts = 0

    # helpers (own frame)
    def thrust(self, me, tgt, T=0.0, push=0.05, lead=None):
        """point at the target point `tgt` (own frame), the blade from the hand towards it, pushing `push` beyond"""
        cart = np.array([me["x"], 0.0, 0.0])
        t = tgt - cart
        if lead is not None:
            t = t + lead
        dirn = t - HAND
        dirn = dirn / np.linalg.norm(dirn)
        tip = t + push * dirn
        if tip[0] > REACH + 0.06:      # beyond reach: as far as it goes along the line
            tip = HAND + dirn * (REACH + 0.06 - HAND[0]) / max(dirn[0], 0.3)
        self.thrusts += 1
        return self.c.arm(T, tip, dirn)

    def guard(self, T=0.25):
        return self.c.arm(T, F.GUARD_TIP, F.GUARD_DIR)

    def act(self, t, o):
        raise NotImplementedError


def gap(o):
    return o["op"]["x"] - o["me"]["x"]


def u_of(o, key, dt=0.0):
    """the opponent's point `key` relative to my cart, predicted dt ahead from both carts' velocities"""
    return (o["op"][key][0] + o["op"]["v"] * dt) - (o["me"]["x"] + o["me"]["v"] * dt)


class Lunge(Strategy):
    """direct attack: close the distance at full speed and thrust at the forearm as soon as it comes within reach;
    recover and repeat"""
    name = "lunge"
    target = "fore"

    def act(self, t, o):
        c = self.c
        if self.st in ("start", "close"):
            self.st = "close"
            c.rail(o["op"]["x"] - 1.9)
            if u_of(o, self.target, 0.35) <= REACH:
                self.thrust(o["me"], o["op"][self.target])
                c.rail(o["op"]["x"] - 1.75)
                self.st, self.mem["t0"] = "thrust", t
        elif self.st == "thrust":
            if t - self.mem["t0"] > 0.9:
                self.guard(); c.rail(o["me"]["x"] - 1.0)
                self.st, self.mem["t0"] = "recover", t
            elif t - self.mem["t0"] > 0.1 and int((t - self.mem["t0"]) / 0.1) != self.mem.get("upd"):
                self.mem["upd"] = int((t - self.mem["t0"]) / 0.1)       # remise: re-aim every 0.1 s
                self.thrust(o["me"], o["op"][self.target])
        elif self.st == "recover":
            if t - self.mem["t0"] > 0.6:
                self.st = "close"


class LungeLow(Lunge):
    """direct attack at the foot (the cart)"""
    name = "lunge_low"
    target = "foot"


class Prep(Lunge):
    """attack on preparation: keep a long distance; when the opponent steps forward, attack at once"""
    name = "prep"

    def act(self, t, o):
        c = self.c
        if self.st in ("start", "wait"):
            self.st = "wait"
            g = gap(o)
            if g < 3.3:
                c.rail(o["op"]["x"] - 3.6)
            elif g > 4.2:
                c.rail(o["op"]["x"] - 3.8)
            else:
                c.rail(o["me"]["x"])
            if o["op"]["v"] < -0.35 and g < 4.0:
                self.st = "close"
                return Lunge.act(self, t, o)
            if t - self.mem.get("idle0", 0.0) > 5.0 and self.mem.get("idle0") is not None:
                self.st = "close"
            self.mem.setdefault("idle0", t)
            return
        if self.st == "recover" and t - self.mem["t0"] > 0.6:
            self.st, self.mem["idle0"] = "wait", t
            return
        Lunge.act(self, t, o)


class StopHit(Strategy):
    """counter-attack: point in line; when the opponent comes into distance, stop-hit its advancing forearm"""
    name = "stop"

    def act(self, t, o):
        c = self.c
        if self.st == "start":
            self.st = "wait"
        if self.st == "wait":
            if gap(o) > 4.3:
                c.rail(o["op"]["x"] - 4.0)
            else:
                c.rail(o["me"]["x"])
            if u_of(o, "fore", 0.3) <= REACH and o["op"]["v"] < -0.2:
                self.thrust(o["me"], o["op"]["fore"], lead=np.array([o["op"]["v"] * 0.25, 0, 0]))
                c.rail(o["me"]["x"] - 0.15)
                self.st, self.mem["t0"] = "hit", t
            elif u_of(o, "fore") <= REACH - 0.05:
                self.thrust(o["me"], o["op"]["fore"])
                self.st, self.mem["t0"] = "hit", t
        elif self.st == "hit":
            if t - self.mem["t0"] > 0.8:
                self.guard(); c.rail(o["me"]["x"] - 0.8)
                self.st, self.mem["t0"] = "back", t
        elif self.st == "back" and t - self.mem["t0"] > 0.5:
            self.st = "wait"


class Parry(Strategy):
    """parry-riposte: wait at distance; when the opponent's point comes at the guard, beat it aside with a lateral
    sweep of the blade (parry), then riposte to the forearm"""
    name = "parry"
    side = 1.0

    def act(self, t, o):
        c = self.c
        me, op = o["me"], o["op"]
        if self.st == "start":
            self.st = "wait"
        if self.st == "wait":
            if gap(o) > 4.3:
                c.rail(op["x"] - 4.0)
            else:
                c.rail(me["x"])
            closing = -(op["v"] - me["v"])
            d_tip = op["tip"][0] - me["guard"][0]
            if d_tip < 0.75 and (closing > 0.3 or d_tip < 0.45):
                # sweep: the point moves across to the parry side, the blade turned outwards
                c.arm(0.0, np.array([1.20, 0.35 * self.side, 1.12]), np.array([1.0, 0.75 * self.side, 0.0]))
                c.rail(me["x"] - 0.2)
                self.st, self.mem["t0"] = "parry", t
        elif self.st == "parry":
            if t - self.mem["t0"] > 0.22:
                self.thrust(me, op["fore"])
                c.rail(op["x"] - 1.85)
                self.st, self.mem["t0"] = "riposte", t
        elif self.st == "riposte":
            if t - self.mem["t0"] > 0.8:
                self.guard(); c.rail(me["x"] - 1.0)
                self.st, self.mem["t0"] = "back", t
        elif self.st == "back" and t - self.mem["t0"] > 0.5:
            self.st = "wait"


class Distance(Lunge):
    """distance management: retreat to keep out of distance while the opponent advances; attack when it stops or
    turns back, or when the rear limit is near"""
    name = "distance"

    def act(self, t, o):
        c = self.c
        me, op = o["me"], o["op"]
        if self.st in ("start", "keep"):
            self.st = "keep"
            g = gap(o)
            if g < 3.6:
                c.rail(op["x"] - 3.9)
            else:
                c.rail(me["x"])
            slow = op["v"] > -0.15
            if (slow and self.mem.get("adv") and g < 3.9) or me["x"] < -5.6:
                self.mem["adv"] = False
                self.st = "close"
                return Lunge.act(self, t, o)
            if op["v"] < -0.4:
                self.mem["adv"] = True
            if t - self.mem.setdefault("k0", t) > 4.0:
                self.mem["k0"] = t
                self.st = "close"
            return
        if self.st == "recover" and t - self.mem["t0"] > 0.6:
            self.st, self.mem["k0"] = "keep", t
            return
        Lunge.act(self, t, o)


class Feint(Strategy):
    """feint-disengage: close in; feint high at the forearm; if the opponent's blade moves (to parry), go under it
    and thrust low at the forearm from below"""
    name = "feint"

    def act(self, t, o):
        c = self.c
        me, op = o["me"], o["op"]
        if self.st in ("start", "close"):
            self.st = "close"
            c.rail(op["x"] - 2.2)
            if gap(o) < 2.45:
                c.arm(0.0, np.array([1.55, 0.05, 1.28]), np.array([1.0, 0.05, 0.12]))
                self.mem["tip0"] = op["tip"].copy()
                self.st, self.mem["t0"] = "feint", t
        elif self.st == "feint":
            if t - self.mem["t0"] > 0.18:
                moved = float(np.linalg.norm(op["tip"][1:] - self.mem["tip0"][1:]))
                low = op["fore"] + np.array([0.0, 0.0, -0.06])
                self.thrust(me, low, lead=np.array([0, 0, -0.04]) if moved > 0.05 else None)
                c.rail(op["x"] - 1.75)
                self.st, self.mem["t0"] = "thrust", t
        elif self.st == "thrust":
            if t - self.mem["t0"] > 0.9:
                self.guard(); c.rail(me["x"] - 1.0)
                self.st, self.mem["t0"] = "recover", t
        elif self.st == "recover" and t - self.mem["t0"] > 0.6:
            self.st = "close"


class Beat(Strategy):
    """attack on the blade: close in, beat the opponent's blade aside with a fast lateral sweep, thrust at once"""
    name = "beat"

    def act(self, t, o):
        c = self.c
        me, op = o["me"], o["op"]
        if self.st in ("start", "close"):
            self.st = "close"
            c.rail(op["x"] - 2.35)
            if gap(o) < 2.65:
                y = op["tip"][1]
                c.arm(0.0, np.array([1.40, y + 0.30, 1.10]), np.array([1.0, 0.45, -0.05]))
                self.st, self.mem["t0"] = "beat", t
        elif self.st == "beat":
            if t - self.mem["t0"] > 0.16:
                self.thrust(me, op["fore"])
                c.rail(op["x"] - 1.75)
                self.st, self.mem["t0"] = "thrust", t
        elif self.st == "thrust":
            if t - self.mem["t0"] > 0.9:
                self.guard(); c.rail(me["x"] - 1.0)
                self.st, self.mem["t0"] = "recover", t
        elif self.st == "recover" and t - self.mem["t0"] > 0.6:
            self.st = "close"


class Derobe(Strategy):
    """point in line, waiting; when the opponent sweeps its blade across to find mine (a beat or a parry), drop the
    point under its blade (derobement) and stop-hit its forearm; else stop-hit an advance into distance"""
    name = "derobe"

    def act(self, t, o):
        c = self.c
        me, op = o["me"], o["op"]
        prev = self.mem.get("prev_tip")
        self.mem["prev_tip"] = op["tip"].copy()
        if self.st == "start":
            self.st = "wait"
        if self.st == "wait":
            if gap(o) > 4.3:
                c.rail(op["x"] - 4.0)
            else:
                c.rail(me["x"])
            lat = 0.0 if prev is None else abs(op["tip"][1] - prev[1]) / F.CONTROL_DT
            if prev is not None and lat > 0.6 and gap(o) < 3.0:
                low = op["fore"] + np.array([0.0, 0.0, -0.08])
                self.thrust(me, low, lead=np.array([op["v"] * 0.2, 0, -0.03]))
                self.st, self.mem["t0"] = "hit", t
            elif u_of(o, "fore", 0.3) <= REACH and op["v"] < -0.2:
                self.thrust(me, op["fore"], lead=np.array([op["v"] * 0.25, 0, 0]))
                self.st, self.mem["t0"] = "hit", t
        elif self.st == "hit":
            if t - self.mem["t0"] > 0.8:
                self.guard(); c.rail(me["x"] - 0.8)
                self.st, self.mem["t0"] = "back", t
        elif self.st == "back" and t - self.mem["t0"] > 0.5:
            self.st = "wait"


class Wall(Strategy):
    """no attack: point in line at the opponent's forearm, never moves the cart (stalling)"""
    name = "wall"

    def act(self, t, o):
        if self.st == "start":
            self.st = "wall"


class CounterBeat(Strategy):
    """counter to the beat: wait at distance; when the opponent comes into beat distance, beat its blade first and
    thrust"""
    name="counterbeat"
    def act(self,t,o):
        c=self.c; me,op=o["me"],o["op"]
        if self.st in ("start","wait"):
            self.st="wait"
            g=gap(o)
            c.rail(op["x"]-3.2 if g<3.0 else me["x"])
            if g<2.95 and op["v"]<-0.2:
                y=op["tip"][1]
                c.arm(0.0,np.array([1.40,y+0.30,1.10]),np.array([1.0,0.45,-0.05]))
                self.st,self.mem["t0"]="beat",t
        elif self.st=="beat":
            if t-self.mem["t0"]>0.16:
                self.thrust(me,op["fore"]); c.rail(op["x"]-1.75)
                self.st,self.mem["t0"]="thrust",t
        elif self.st=="thrust":
            if t-self.mem["t0"]>0.9:
                self.guard(); c.rail(me["x"]-1.0); self.st,self.mem["t0"]="back",t
        elif self.st=="back" and t-self.mem["t0"]>0.6: self.st="wait"
class LowStop(Strategy):
    """when the opponent closes into distance, drop the point and stop-hit under its blade at the arm"""
    name="lowstop"
    def act(self,t,o):
        c=self.c; me,op=o["me"],o["op"]
        if self.st in ("start","wait"):
            self.st="wait"
            g=gap(o)
            c.rail(op["x"]-4.0 if g>4.3 else me["x"])
            if g<float(self.mem.get("trig",2.9)) and op["v"]<-0.2:
                low=op["fore"]+np.array([0,0,-0.10])
                self.thrust(me,low,lead=np.array([op["v"]*0.3,0,-0.04]))
                self.st,self.mem["t0"]="hit",t
        elif self.st=="hit":
            if t-self.mem["t0"]>0.8:
                self.guard(); c.rail(me["x"]-0.8); self.st,self.mem["t0"]="back",t
        elif self.st=="back" and t-self.mem["t0"]>0.5: self.st="wait"
class Retreat(Strategy):
    """break distance when the opponent closes, counter-attack when it stops (the beat lands on air)"""
    name="retreat"
    def act(self,t,o):
        c=self.c; me,op=o["me"],o["op"]
        g=gap(o)
        if self.st in ("start","keep"):
            self.st="keep"
            if g<3.1: c.rail(op["x"]-3.6)
            else: c.rail(me["x"])
            if (op["v"]>-0.1 and g<3.3 and self.mem.get("adv")) or me["x"]<-5.8:
                self.thrust(me,op["fore"]); c.rail(op["x"]-1.75); self.st,self.mem["t0"]="hit",t; self.mem["adv"]=False
            if op["v"]<-0.3: self.mem["adv"]=True
        elif self.st=="hit":
            if t-self.mem["t0"]>0.9:
                self.guard(); c.rail(me["x"]-1.0); self.st,self.mem["t0"]="back",t
        elif self.st=="back" and t-self.mem["t0"]>0.5: self.st="keep"


class Fleche(Strategy):
    """running attack with the arm extended from the start: the point straight ahead at full reach, the cart at full
    speed past the centre line (found by a model in a smoke run)"""
    name = "fleche"

    def act(self, t, o):
        if self.st == "start":
            self.c.arm(0.5, np.array([2.0, 0.0, 1.1]), np.array([1.0, 0.0, 0.0]))
            self.c.rail(0.4, F.RAIL_VMAX)
            self.st = "go"


STRATS = [Lunge, LungeLow, Prep, StopHit, Parry, Distance, Feint, Beat, Derobe, CounterBeat, LowStop, Retreat, Wall,
          Fleche]
BY_NAME = {s.name: s for s in STRATS}
