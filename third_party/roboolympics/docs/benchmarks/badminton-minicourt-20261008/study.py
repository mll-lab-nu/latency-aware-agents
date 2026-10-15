"""Badminton minicourt B9 study (2026-10-08; ported from the serve-receivability study): which legal serves can the receiver return, under the official
physics (variant a) and under real-robot limits (variants b-*; knobs.py)? Nothing here is imported by the sport.

Harness: the sport's own plugin (Badminton duel world, RacketLayer: an act is parsed at its arrival step, planned by the
official stroke controller on an exact copy of the robot advanced to its effect step = arrival + lead, lowered and
installed then; core/local.py's loop: every 5 control steps each seat may act).
  * Server A: ForcedServer = the built-in Reference's serve (same toss, contact at its forehand point at 0.95 m, the
    documented contact law to the racket state, the stroke sent 1.0 s ahead, the toss timed to meet it) with the serve
    FORCED to one target in the diagonal service court and one launch-angle family (no fallback to other kinds).
  * Receiver B, three kinds:
      ref     the built-in Reference unchanged, perception replaced by the true shuttle position at the 4 frames'
              times + 3 mm noise (the README's "perception-free" runs);
      cam     the built-in Reference with the real 8-tile images and its real detection (slow; a subset);
      oracle  upper bound (oracle.py): knows the TRUE shuttle state REACT steps after the serve's hit and searches over
              feasible strokes played through the official planner (with its execution scatter) on deep copies.
Each serve is run until the receiver's first hit (or the point's end); a return is good if the shuttle, from the
receiver's hit, lands in the server's court (shuttle-only prediction of the documented flight, net included).
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import knobs as KN  # noqa: F401  (installs the variant before the sport is used)

BM, BD, BP, PL = KN.BM, KN.BD, KN.BP, KN.PL
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused  # noqa: E402

DT = BM.CONTROL_DT
OUT = Path(os.environ.get("BM_OUT", "results"))
SLOWDOWN = float(os.environ.get("LEAD_SLOWDOWN", "20"))

# ------------------------------------------------------------------------------------------------ perception
_CUR = {"obs": None}
_REAL_DECODE, _REAL_DETECT = BP.decode, BP.detect_views


def _decode(b):
    return b if _CUR["obs"] is not None else _REAL_DECODE(b)


def _detect_views(img, cams, times, max_res=0.12):
    if _CUR["obs"] is None:
        return _REAL_DETECT(img, cams, times, max_res)
    ts = {round(float(t), 4) for t in times}
    return [(t, X) for t, X in _CUR["obs"] if round(float(t), 4) in ts]


BP.decode, BP.detect_views = _decode, _detect_views


def make(seed=1, first="A"):
    cfg = MatchConfig(sport="badminton", format="duel-wallclock", seats=("A", "B"), seed=int(seed), first=first,
                      slowdown=SLOWDOWN)
    m = PL.SPORT.build(cfg)
    m._ctx()
    return cfg, m, PL.SPORT.action_layer("high")


def seat_spec(cfg, m, seat):
    return {"seat": seat, "action_level": "high", **PL.SPORT.spec(cfg, seat, m)}


class PerfectRef(BP.Reference):
    camera = False

    def __init__(self, spec, seat, seed=0, noise=0.003):
        super().__init__(spec, seat, seed)
        self.noise, self.seed_, self.seat = noise, int(seed), seat

    def perceive(self, m, seat):
        job, meta = m.camera_job(seat)
        if self.camera:
            import base64
            png = PL.SPORT.worker_call(None, "render", job)["png"]
            return {**meta, "image_png_b64": base64.b64encode(png).decode(), "_obs": None}
        obs = []
        for sn, t in zip(job["snaps"], meta["view_sim_times"]):
            X = BD.to_side(seat, sn["sh"])
            if X[2] < -0.1:
                continue
            g = np.random.default_rng([self.seed_, ord(seat), int(sn["k"])])
            obs.append((float(t), X + g.normal(0.0, self.noise, 3)))
        return {**meta, "image_png_b64": "eA==", "_obs": obs}

    def act(self, view):
        _CUR["obs"] = (view.get("observation") or {}).get("_obs")
        try:
            return super().act(view)
        finally:
            _CUR["obs"] = None


class CameraRef(PerfectRef):
    camera = True


class Idle:
    def act(self, view):
        return None


# ------------------------------------------------------------------------------------------------ the forced serve
# launch-angle families of the forced serves. The receivability study's (low 2-16, mid 20-36, high 44-60 deg) do not
# clear the net on the small court: the serve's contact is only about 1.07 m behind the net, and a serve must rise
# over 1.524 m from below 1.15 m (net clearance 20 cm needs >= 40 deg at the back of the service court, >= 64 deg at its
# front). The small court's families: flat 38-54, mid 54-66, high 66-78 deg.
FAMILIES = {"low": (38, 42, 46, 50, 54), "mid": (54, 58, 62, 66), "high": (66, 70, 74, 78)}
SERVE_NET_MARGIN = 0.3          # the forced server's net clearance (the final library runs b9f-*; the reference: 0.4)


def serve_grid(court="right"):
    """The serve targets of the README's grid (receiver frame = the server's frame mirrored; here in the SERVER's
    frame, as Reference.serve_choice gives them): 5 x 7 over the receiver's diagonal service court x 3 families."""
    sgn = -1.0 if court == "right" else 1.0
    S = BM.COURT_SCALE                  # the BWF-court grid of the receivability study, scaled to the small court
    xs = tuple(round(x * S, 3) for x in (0.2, 0.75, 1.3, 1.9, 2.45))
    ys = tuple(round(float(y), 3) for y in np.linspace(2.05 * S, 6.6 * S, 7))
    return [dict(fam=f, x=sgn * x, y=y) for f in FAMILIES for y in ys for x in xs]


class ForcedServer(PerfectRef):
    """Reference.serve with the kind (launch angles) and target forced; no fallback to another kind or target."""

    def __init__(self, spec, seat, seed, plan):
        super().__init__(spec, seat, seed)
        self.plan_ = plan
        self.note = None

    def serve(self, st, ob):
        sv = st.get("serve") or {}
        phase, point = sv.get("phase"), st.get("point")
        if self.srv is None or self.srv["point"] != point:
            if phase not in ("waiting", "open"):
                return None
            base = np.asarray(st.get("robot_base_xy") or [0.0, float(BM.SPOT_Y)], float)
            target = np.array([self.plan_["x"], self.plan_["y"]])
            old, old_m = BP.SERVE_THETAS, BP.NET_MARGIN
            BP.SERVE_THETAS = {"forced": FAMILIES[self.plan_["fam"]]}
            BP.NET_MARGIN = SERVE_NET_MARGIN
            try:
                pc, rel, v_in, opts = BP.serve_plan(base, "forced", target, self.plan_.get("cap"))
            finally:
                BP.SERVE_THETAS, BP.NET_MARGIN = old, old_m
            if not opts:
                self.note = "no stroke within the head-speed cap reaches the target"
                self.srv = {"point": point, "dead": True}
                return None
            self.srv = {"point": point, "kind": "forced", "target": target, "pc": pc, "rel": rel, "opts": opts,
                        "k_c": None, "released": False, "fixed": False}
        if self.srv.get("dead"):
            return None
        return super().serve(st, ob)


# ------------------------------------------------------------------------------------------------ the loop
def play(cfg, m, layer, players, every=5, max_steps=20000, stop=None):
    """core/local.py's realtime loop (acts planned in-process from the snapshot of their arrival step)."""
    refused = {s: 0 for s in cfg.seats}
    while m.phase().state != FINISHED and m.k < max_steps:
        if m.phase().state == UNIT_OVER:
            m.resume()
        if stop is not None and stop(m):
            break
        if m.k % every == 0:
            for seat in cfg.seats:
                pl = players.get(seat)
                if pl is None:
                    continue
                st = {"seat": seat, "phase": "running", "world": "running", "step": m.k, "t_sim": m.k * DT,
                      "done": False, **m.observe(seat)}
                body = pl.act({"state": st, "observation": pl.perceive(m, seat) if hasattr(pl, "perceive") else {}})
                if body is None:
                    continue
                try:
                    act_apply(layer, m, seat, body)
                except Refused:
                    refused[seat] += 1
        m.step()
    return refused


def act_apply(layer, m, seat, body):
    cmd = layer.parse(seat, body, m, {})
    job = layer.snapshot(seat, cmd, m, m.k)
    prepared = layer.prepare(job) if job is not None else None
    low, info = layer.lower(seat, cmd, prepared, m)
    return m.apply(seat, low), info


class Tracker:
    """Collects the point's events (World.events is cleared when the next point starts)."""

    def __init__(self):
        self.hits, self.seen, self.toss_k, self.end = [], set(), None, None
        self.apex = None

    def __call__(self, m):
        g = m.g
        for e in g.events:
            key = (e["kind"], e["k"], e.get("side"))
            if key in self.seen:
                continue
            self.seen.add(key)
            if e["kind"] == "hit":
                self.hits.append(dict(e))
            elif e["kind"] == "toss":
                self.toss_k = e["k"]
        if len(self.hits) == 1 and g.sh_live:
            self.apex = max(self.apex or 0.0, float(g.sh_p[2]))
        if g.points and self.end is None:
            self.end = dict(g.points[0])


def landing(p, v):
    """Shuttle-only flight (world frame) from p, v: (x, y, 'floor'|'net', t)."""
    P, V = BM.predict(p, v, n_max=8000)
    if not len(P):
        return None
    last = P[-1]
    kind = "floor" if last[2] <= 1e-9 else ("net" if abs(last[1]) < 1e-6 else "air")
    return float(last[0]), float(last[1]), kind, len(P) * BM.PHYS_DT


def serve_flight_info(hit):
    """The serve's flight from its hit (world = server A's frame): landing, flight time, apex."""
    p, v = np.array(hit["pos"], float), np.array(hit["v_out"], float)
    P, V = BM.predict(p, v, n_max=8000)
    last = P[-1]
    return {"land": [round(float(last[0]), 3), round(float(last[1]), 3)], "t_flight": round(len(P) * BM.PHYS_DT, 3),
            "apex": round(float(P[:, 2].max()), 3), "net_z": _net_z(P),
            "lands_floor": bool(last[2] <= 1e-9), "speed": round(float(np.linalg.norm(v)), 2),
            "elev_deg": round(float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))), 1)}


def _net_z(P):
    s = np.nonzero(np.sign(P[1:, 1]) != np.sign(P[:-1, 1]))[0]
    return None if not len(s) else round(float(P[s[0] + 1, 2]), 3)


def classify(fi):
    """Serve type by its actual flight (receiver's depth from the net d = landing y in A's frame), the BWF-court bounds
    scaled: short d < 3.5 S; long d >= 5.0 S: high (apex >= NET_TOP + 1.0 m) or flat (flick / drive serve); else mid."""
    d = fi["land"][1]
    S = BM.COURT_SCALE
    if d < 3.5 * S:
        return "short"
    if d >= 5.0 * S:
        return "long-high" if fi["apex"] >= BM.NET_TOP + 1.0 else "long-flat"
    return "mid"


def run_serve(plan, seed=1, receiver="ref", stop_extra=40):
    """One forced serve from A's right court (point 1, score 0-0); B receives with `receiver`. Returns a record."""
    cfg, m, layer = make(seed, "A")
    srv = ForcedServer(seat_spec(cfg, m, "A"), "A", seed, plan)
    if receiver == "ref":
        rx = PerfectRef(seat_spec(cfg, m, "B"), "B", seed)
    elif receiver == "cam":
        rx = CameraRef(seat_spec(cfg, m, "B"), "B", seed)
    else:
        rx = Idle()
    tr = Tracker()
    st = {}

    def stop(mm):
        tr(mm)
        if tr.end is not None:
            return True
        if len(tr.hits) >= 2 and "k2" not in st:
            st["k2"] = mm.k
        return "k2" in st and mm.k >= st["k2"] + 2
    t0 = time.time()
    play(cfg, m, layer, {"A": srv, "B": rx}, stop=stop, max_steps=3000)
    rec = {"plan": plan, "seed": seed, "receiver": receiver, "wall_s": round(time.time() - t0, 1)}
    sv = tr.hits[0] if tr.hits else None
    rec["serve_note"] = srv.note
    if sv is None or sv["side"] != "A":
        rec["legal"] = False
        rec["fault"] = ("not producible: " + srv.note) if srv.note else (tr.end or {}).get("how", "no serve")
        return rec
    rec["serve_hit"] = {k: sv[k] for k in ("k", "pos", "speed", "racket_speed", "v_out", "offcentre")}
    fi = serve_flight_info(sv)
    rec["flight"] = fi
    rec["type"] = classify(fi)
    how = (tr.end or {}).get("how", "")
    rec["legal"] = not how.startswith("serve_fault") and bm_in_service(fi)
    if not rec["legal"]:
        rec["fault"] = how or "serve lands outside the service court"
        return rec
    rb = next((h for h in tr.hits[1:] if h["side"] == "B"), None)
    rec["rx_hit"] = None if rb is None else {k: rb[k] for k in ("k", "pos", "speed", "racket_speed", "v_out")}
    if rb is None:
        rec["returned"] = False
        rec["rx_outcome"] = how or "no hit"
    else:
        L = landing(np.array(rb["pos"]), np.array(rb["v_out"]))
        good = L is not None and L[2] == "floor" and BM.in_court(L[0], L[1], -1)
        rec["returned"] = bool(good)
        rec["rx_land"] = None if L is None else [round(L[0], 3), round(L[1], 3), L[2]]
        rec["rx_outcome"] = "in" if good else ("net" if L and L[2] == "net" else "out")
        rec["rx_speed"] = round(float(np.linalg.norm(rb["v_out"])), 2)
    return rec


def bm_in_service(fi):
    """The serve lands in B's diagonal service court (A served from its right court: B's right court)."""
    x, y = fi["land"]
    xr, yr = BD.to_side("B", [x, y, 0.0])[:2]
    return fi["lands_floor"] and BM.in_service_court(xr, yr, True)


def shard(items):
    k, n = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARD", "1"))
    return [x for i, x in enumerate(items) if i % n == k]


def main():
    tag = os.environ.get("TAG") or KN.VARIANT
    recv = sys.argv[1] if len(sys.argv) > 1 else "ref"
    plans = serve_grid(os.environ.get("COURT", "right"))
    fams = os.environ.get("FAMS")
    if fams:
        plans = [p for p in plans if p["fam"] in fams.split(",")]
    sel = os.environ.get("SEL")
    if sel:
        idx = {int(i) for i in sel.split(",")}
        plans = [p for i, p in enumerate(plans) if i in idx]
    for i, p in enumerate(plans):
        p["i"] = i
    plans = shard(plans)
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{recv}_{os.environ.get('SHARD', '0')}.jsonl"
    done = set()
    if f.exists():
        done = {json.loads(l)["plan"]["i"] for l in f.read_text().splitlines() if l.strip()}
    with f.open("a") as fh:
        for p in plans:
            if p["i"] in done:
                continue
            tried = []
            for cap in KN.SERVE_CAPS:
                p["cap"] = cap
                if recv == "oracle":
                    import oracle
                    rec = oracle.run(dict(p))
                else:
                    rec = run_serve(dict(p), receiver=recv)
                tried.append({"cap": cap, "legal": rec.get("legal"), "fault": rec.get("fault")})
                if rec.get("legal"):
                    break
            rec["caps_tried"] = tried
            rec["knobs"] = KN.describe()
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(p["i"], p["fam"], p["x"], p["y"], rec.get("legal"), rec.get("type"), rec.get("returned"),
                  rec.get("fault", ""), rec.get("wall_s"), flush=True)


if __name__ == "__main__":
    main()
