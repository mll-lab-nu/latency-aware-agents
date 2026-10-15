"""Do rallies exist under the real physics? Upper-bound players on both seats: the built-in reference serves
(perception-free); every shot is then returned by an informed player that knows the shuttle's TRUE state REACT steps
after the opponent's hit and searches strokes (oracle.py's candidates: contacts on the true flight that the base
reaches in time, the documented contact law to aims in the opponent's court) played on a deep copy of the world through
the official high level (planner, real limits, scatter) until one lands in the opponent's court; that stroke is then
played in the real world. Points end when no candidate works (or a rally reaches RALLY_CAP hits).
   run.sh rally_oracle.py   (SEEDS, REACT, MAXTRIES; results/<TAG>/rally_<shard>.jsonl)"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

import study as S

BM, BD, BP = S.BM, S.BD, S.BP
DT = S.DT
REACT = int(os.environ.get("REACT", "0"))
MAXTRIES = int(os.environ.get("MAXTRIES", "25"))
RALLY_CAP = int(os.environ.get("RALLY_CAP", "30"))
_S = BM.COURT_SCALE
# rally-keeping shots: lofted, to the back and the middle of the opponent's court (the question is whether rallies
# CAN go on, not who wins them); a short aim only when nothing else works
AIMS = [(x * _S, y * _S) for x, y in [(-1.9, 5.9), (1.9, 5.9), (0.0, 6.2), (0.0, 4.5), (0.0, 3.0)]]
THETAS = (60, 50, 70, 40, 30)


def candidates(m, side, lead):
    g = m.g
    p, v = BD.to_side(side, g.sh_p), BD.to_side(side, g.sh_v)
    P, V = BM.predict(p, v, n_max=8000)
    me = g.sides[side]
    b0 = me.s.d.qpos[me.s.qa[:2]].copy()
    contacts = []
    for j in range(lead + 3, len(P) // 10 + 1, 2):
        i = 10 * j - 1
        if i >= len(P):
            break
        q, w = P[i], V[i]
        if not (0.3 <= q[2] <= 2.7 and -(BM.COURT_Y + 0.7) <= q[1] <= -0.45 and abs(q[0]) <= BM.COURT_X + 0.9):
            continue
        slack = (j - lead) * DT - BM.base_time(float(np.linalg.norm(BM.stance(q) - b0)))
        if slack < -0.10:
            continue
        contacts.append((j, q.copy(), w.copy(), slack))
    contacts = sorted(contacts, key=lambda c: min(c[3], 0.25) + (0.1 if 0.9 <= c[1][2] <= 2.3 else 0.0) +
                      0.3 * min((c[0] - lead) * DT, 1.0), reverse=True)[:14]
    per = []
    for j, q, w, slack in contacts:
        opts = []
        for aim in AIMS:
            for th, sp, v_out in BP.solve_launch(q, np.array(aim), THETAS, vmax=BM.RACKET_GAIN * 7.9 + 2):
                n, vel = BP.racket_state(w, v_out)
                if np.linalg.norm(vel) <= 7.9:
                    opts.append((AIMS.index(aim) // 3, float(np.linalg.norm(vel)), aim, n, vel))
        opts.sort(key=lambda o: (o[0], o[1]))
        opts = [o[1:] for o in opts]
        per.append([(j, q, n, vel) for _, aim, n, vel in opts])
    out, r = [], 0
    while len(out) < MAXTRIES and any(r < len(x) for x in per):
        for x in per:
            if r < len(x) and len(out) < MAXTRIES:
                out.append(x[r])
        r += 1
    return out


def body_of(c, lead):
    j, q, n, vel = c
    return {"type": "act", "t": round((j - lead) * DT, 4), "pos": [float(x) for x in q], "normal": [float(x) for x in n],
            "vel": [float(x) for x in vel]}


def good_return(m2, side, k_until):
    """Advance the copy until `side` hits (or the point ends); the return lands in the opponent's court?"""
    n0 = len(m2.g.points)
    while m2.k < k_until and len(m2.g.points) == n0:
        m2.step()
        hit = next((e for e in m2.g.events if e["kind"] == "hit" and e.get("side") == side), None)
        if hit:
            L = S.landing(np.array(hit["pos"]), np.array(hit["v_out"]))
            opp = BD.OTHER[side]
            sign = -1 if opp == "A" else 1
            return L is not None and L[2] == "floor" and BM.in_court(L[0], L[1], sign)
    return False


def play_point(m, layer, server_pl, seat_srv):
    """One point: the reference serves; then oracle returns. Returns (hits, how)."""
    g = m.g
    n0 = len(g.points)
    hits_done = 0
    while len(g.points) == n0 and not g.done:
        if m.phase().state == "unit_over":
            m.resume()
        rs = g.rs
        if rs["state"] in ("ready", "to_serve", "tossed") and m.k % 5 == 0:
            st = {"seat": seat_srv, "phase": "running", "world": "running", "step": m.k, "t_sim": m.k * DT,
                  "done": False, **m.observe(seat_srv)}
            b = server_pl.act({"state": st, "observation": server_pl.perceive(m, seat_srv)})
            if b is not None:
                try:
                    S.act_apply(layer, m, seat_srv, b)
                except S.Refused:
                    pass
        if rs["state"] == "live" and rs["hits_in_point"] > hits_done and m.k - rs["played_k"] >= 1 + REACT:
            hits_done = rs["hits_in_point"]
            if hits_done >= RALLY_CAP:
                return hits_done, "rally cap"
            side = rs["expect"]
            ok = None
            for c in candidates(m, side, layer.lead):
                m2 = copy.deepcopy(m)
                body = body_of(c, layer.lead)
                try:
                    S.act_apply(layer, m2, side, body)
                except S.Refused:
                    continue
                if good_return(m2, side, m.k + c[0] + 30):
                    ok = body
                    break
            if ok is not None:
                S.act_apply(layer, m, side, ok)
        m.step()
    p = g.points[-1]
    return p["hits"], p["how"]


def main():
    out = S.OUT / os.environ.get("TAG", "rally")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "1").split(",")]
    npts = int(os.environ.get("NPTS", "10"))
    with (out / f"rally_{os.environ.get('SHARD', '0')}.jsonl").open("a") as fh:
        for seed in seeds:
            cfg, m, layer = S.make(seed, "A")
            refs = {s: S.PerfectRef(S.seat_spec(cfg, m, s), s, seed) for s in ("A", "B")}
            for i in range(npts):
                if m.g.done:
                    break
                srv = m.g.rs["server"]
                t0 = time.time()
                h, how = play_point(m, layer, refs[srv], srv)
                r = {"seed": seed, "point": i, "server": srv, "hits": h, "how": how[:100], "wall_s": round(time.time() - t0, 1)}
                fh.write(json.dumps(r) + "\n"); fh.flush()
                print(r, flush=True)
                if how == "rally cap":
                    break


if __name__ == "__main__":
    main()
