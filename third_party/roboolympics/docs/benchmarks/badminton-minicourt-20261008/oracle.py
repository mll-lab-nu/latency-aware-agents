"""Upper-bound receiver: knows the served shuttle's TRUE state REACT control steps after the serve's hit and searches
over feasible strokes, each played through the official planner (RacketLayer: parse at the decision step, planned
on the robot's copy advanced by the act lead, with the execution scatter) on a deep copy of the live world.

Candidates (receiver B's frame):
  * contact steps j after the decision (every 2 steps = 20 ms) on the true flight, j >= lead + 3, the shuttle
    0.3-2.7 m high, -7.4 <= y <= -0.45, |x| <= 3.3, inside the act's ranges;
  * the base: the stroke controller's forehand stance for the contact (bm.stance); kept if the base generator's
    rest-to-rest time (bm.base_time, the variant's speed / acceleration / jerk) fits in the time before the contact
    (slack >= -0.10 s: the arm can make up a little);
  * strokes: the documented contact law (the variant's gain) for launches that land on one of AIMS in the server's
    court and clear the net by 20 cm (builtin_players.solve_launch), head speed <= HEAD_ORACLE (7.9 m/s, the act's
    range), slowest head first.
Up to MAXTRIES candidates, round robin over up to 14 contact steps (best slack and contact height first). A serve
is RETURNABLE if any candidate's return lands in the server's court (shuttle-only prediction from the hit).
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np

import study as S

BM, BD, BP, KN = S.BM, S.BD, S.BP, S.KN
DT = S.DT
REACT = int(os.environ.get("REACT", "0"))
MAXTRIES = int(os.environ.get("MAXTRIES", "40"))
HEAD_ORACLE = float(os.environ.get("HEAD_ORACLE", "7.9"))
MIN_DEPTH = float(os.environ.get("MIN_DEPTH", "0"))          # a good return lands at least this far beyond the net
_S = BM.COURT_SCALE                          # the BWF-court aims of the receivability study, scaled
AIMS = [(-1.9, 5.9), (1.9, 5.9), (0.0, 4.5), (-1.9, 2.3), (1.9, 2.3), (0.0, 2.4), (0.0, 6.2)]
if os.environ.get("AIMSET") == "deep":      # rally check: only returns to the server's back court (>= 5.5 S deep)
    AIMS = [(-1.9, 5.9), (1.9, 5.9), (0.0, 6.2), (-1.0, 5.6), (1.0, 5.6)]
AIMS = [(x * _S, y * _S) for x, y in AIMS]
THETAS = (45, 35, 55, 25, 60, 15, 5, -5, -15)


def decision_world(plan, seed=1):
    cfg, m, layer = S.make(seed, "A")
    srv = S.ForcedServer(S.seat_spec(cfg, m, "A"), "A", seed, plan)
    tr = S.Tracker()
    st = {}

    def stop(mm):
        tr(mm)
        if tr.end is not None:
            return True
        if tr.hits and "kd" not in st:
            st["kd"] = tr.hits[0]["k"] + 1 + REACT
        return "kd" in st and mm.k >= st["kd"]
    S.play(cfg, m, layer, {"A": srv}, stop=stop, max_steps=3000)
    return cfg, m, layer, srv, tr


def candidates(m, lead):
    g = m.g
    p_b = BD.to_side("B", g.sh_p); v_b = BD.to_side("B", g.sh_v)
    P, V = BM.predict(p_b, v_b, n_max=8000)
    side = g.sides["B"]
    b0 = side.s.d.qpos[side.s.qa[:2]].copy()
    contacts = []
    for j in range(lead + 3, len(P) // 10 + 1, 2):
        i = 10 * j - 1
        if i >= len(P):
            break
        p, v = P[i], V[i]
        if not (0.3 <= p[2] <= 2.7 and -(BM.COURT_Y + 0.7) <= p[1] <= -0.45 and abs(p[0]) <= BM.COURT_X + 0.9):
            continue
        st = BM.stance(p)
        slack = (j - lead) * DT - BM.base_time(float(np.linalg.norm(st - b0)))
        if slack < -0.10:
            continue
        contacts.append((j, p.copy(), v.copy(), slack))
    info = {"n_contacts": len(contacts), "flight_steps": len(P) // 10}
    if not contacts:
        return [], info
    # prefer enough slack and a comfortable height; keep up to 14 spread over the window
    def pref(c):
        j, p, v, slack = c
        return (min(slack, 0.25) + (0.1 if 0.9 <= p[2] <= 2.3 else 0.0))
    contacts = sorted(contacts, key=pref, reverse=True)
    if len(contacts) > 14:
        contacts = contacts[:14]
    vmax = min(60.0, KN.GAIN * HEAD_ORACLE + 2.0)
    per = []
    n_strokes = 0
    for j, p, v, slack in contacts:
        opts = []
        for aim in AIMS:
            for th, sp, v_out in BP.solve_launch(p, np.array(aim), THETAS, vmax=vmax):
                n, vel = BP.racket_state(v, v_out)
                hs = float(np.linalg.norm(vel))
                if hs <= HEAD_ORACLE:
                    opts.append((hs, aim, th, sp, n, vel))
        opts.sort(key=lambda o: o[0])
        # diversity: alternate aims (the slowest head per aim first)
        seen, first, rest = set(), [], []
        for o in opts:
            (rest if o[1] in seen else first).append(o)
            seen.add(o[1])
        opts = first + rest
        n_strokes += len(opts)
        per.append([(j, p, v, slack, o) for o in opts])
    info["n_strokes"] = n_strokes
    out = []
    r = 0
    while len(out) < MAXTRIES and any(r < len(x) for x in per):
        for x in per:
            if r < len(x) and len(out) < MAXTRIES:
                out.append(x[r])
        r += 1
    return out, info


def trial(m, layer, cand):
    j, p, v, slack, (hs, aim, th, sp, n, vel) = cand
    m2 = copy.deepcopy(m)
    k0 = m2.k
    body = {"type": "act", "t": round((j - layer.lead) * DT, 4), "pos": [float(x) for x in p],
            "normal": [float(x) for x in n], "vel": [float(x) for x in vel]}
    try:
        _, info = S.act_apply(layer, m2, "B", body)
    except S.Refused as e:
        return {"ok": False, "why": f"refused: {e}"}
    n_ev = len(m2.g.events)
    hit = None
    while m2.k < k0 + j + 25 and not m2.g.points:
        m2.step()
        hit = next((e for e in m2.g.events if e["kind"] == "hit" and e.get("side") == "B"), None)
        if hit:
            break
    st = (info or {}).get("stroke") or {}
    rec = {"j": j, "z": round(float(p[2]), 2), "slack": round(float(slack), 3), "head": round(hs, 2), "aim": aim,
           "theta": th, "shot_speed": round(sp, 1), "stroke": st.get("stroke"), "base": st.get("base")}
    if hit is None:
        rec.update(ok=False, why="miss")
        return rec
    L = S.landing(np.array(hit["pos"]), np.array(hit["v_out"]))
    good = L is not None and L[2] == "floor" and BM.in_court(L[0], L[1], -1) and -L[1] >= MIN_DEPTH
    shallow = L is not None and L[2] == "floor" and BM.in_court(L[0], L[1], -1) and not good
    why = "in" if good else "too_short" if shallow else ("net" if L and L[2] == "net" else ("own_side" if L and L[1] > 0 else "out"))
    rec.update(ok=bool(good), why=why, out_speed=round(float(hit["speed"]), 2),
               land=None if L is None else [round(L[0], 2), round(L[1], 2)])
    return rec


def run(plan, seed=1):
    t0 = time.time()
    cfg, m, layer, srv, tr = decision_world(plan, seed)
    rec = {"plan": plan, "seed": seed, "receiver": "oracle", "react": REACT, "lead": layer.lead,
           "serve_note": srv.note}
    sv = tr.hits[0] if tr.hits else None
    if sv is None or tr.end is not None and str(tr.end.get("how", "")).startswith("serve_fault"):
        rec["legal"] = False
        rec["fault"] = ("not producible: " + srv.note) if srv.note else (tr.end or {}).get("how", "no serve")
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    fi = S.serve_flight_info(sv)
    rec["serve_hit"] = {k: sv[k] for k in ("k", "pos", "speed", "racket_speed", "v_out", "offcentre")}
    rec["flight"] = fi
    rec["type"] = S.classify(fi)
    rec["legal"] = S.bm_in_service(fi)
    if not rec["legal"]:
        rec["fault"] = "serve lands outside the service court"
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    cands, info = candidates(m, layer.lead)
    rec.update(info)
    tries = []
    ok = None
    for c in cands:
        t = trial(m, layer, c)
        tries.append(t)
        if t.get("ok"):
            ok = t
            break
    rec["n_tries"] = len(tries)
    rec["returned"] = ok is not None
    rec["success"] = ok
    if ok is None:
        if not info["n_contacts"]:
            rec["cause"] = "base: no contact point reachable in time"
        elif not info.get("n_strokes"):
            rec["cause"] = "racket: no stroke within the head-speed cap reaches the server's court over the net"
        else:
            whys = {}
            for t in tries:
                whys[t["why"]] = whys.get(t["why"], 0) + 1
            rec["cause"] = "strokes tried, none good: " + ", ".join(f"{k} {v}" for k, v in sorted(whys.items()))
        rec["tries_sample"] = tries[:6]
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec
