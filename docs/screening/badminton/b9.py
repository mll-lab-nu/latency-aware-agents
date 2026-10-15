"""B9 for badminton (docs/benchmarks/badminton-minicourt-20261008: study.py oracle), with the receiver's
self-collision counted: the forced server's grid serves x the upper-bound receiver (oracle.py, unchanged candidates
and planner), REACT steps; every tried stroke is played on through its follow-through (to the end of B's plan + 30
steps) and B's self-collision events / warnings (Monitor: 10 ms mean force > 20 N) of that try are recorded.
returned = the first good return (as oracle.run); returned_clean = the first good return whose try has no
self-collision warning (the search goes on, up to MAXTRIES). Env: TAG, REACT, MAXTRIES, SHARD, NSHARD."""
from __future__ import annotations

import copy
import json
import os
import sys
import time

sys.path.insert(0, "/repo/docs/benchmarks/badminton-minicourt-20261008")
import numpy as np  # noqa: E402

import study as S  # noqa: E402
import oracle as O  # noqa: E402

BM, BD, KN = S.BM, S.BD, S.KN
DT = S.DT


def trial(m, layer, cand):
    j, p, v, slack, (hs, aim, th, sp, n, vel) = cand
    m2 = copy.deepcopy(m)
    k0 = m2.k
    monB = m2.g.sides["B"].s.selfcol["A"]
    e0, w0 = monB.events, monB.warnings + sum(1 for ev in monB.open.values() if ev["avg"] > (monB.warn or 1e9))
    body = {"type": "act", "t": round((j - layer.lead) * DT, 4), "pos": [float(x) for x in p],
            "normal": [float(x) for x in n], "vel": [float(x) for x in vel]}
    try:
        _, info = S.act_apply(layer, m2, "B", body)
    except S.Refused as e:
        return {"ok": False, "why": f"refused: {e}", "sc_events": 0, "sc_warn": 0}
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
    else:
        L = S.landing(np.array(hit["pos"]), np.array(hit["v_out"]))
        good = L is not None and L[2] == "floor" and BM.in_court(L[0], L[1], -1) and -L[1] >= O.MIN_DEPTH
        shallow = L is not None and L[2] == "floor" and BM.in_court(L[0], L[1], -1) and not good
        why = "in" if good else "too_short" if shallow else ("net" if L and L[2] == "net" else ("own_side" if L and L[1] > 0 else "out"))
        rec.update(ok=bool(good), why=why, out_speed=round(float(hit["speed"]), 2),
                   land=None if L is None else [round(L[0], 2), round(L[1], 2)])
    # play on through the follow-through: to the end of B's stroke plan + 30 steps (or the point's end)
    side = m2.g.sides["B"]
    k_end = m2.k + max(0, len(side.plan) - side.plan_j) + 30
    while m2.k < k_end and not m2.g.done and not m2.g.points:
        m2.step()
    w1 = monB.warnings + sum(1 for ev in monB.open.values() if ev["avg"] > (monB.warn or 1e9))
    rec["sc_events"] = monB.events - e0
    rec["sc_warn"] = w1 - w0
    rec["sc_depth"] = round(monB.max_depth, 4)
    return rec


def run(plan, seed=1):
    t0 = time.time()
    cfg, m, layer, srv, tr = O.decision_world(plan, seed)
    monA = m.g.sides["A"].s.selfcol["A"]
    rec = {"plan": plan, "seed": seed, "receiver": "oracle", "react": O.REACT, "lead": layer.lead,
           "serve_note": srv.note, "server_sc_events": monA.events, "server_sc_warn": monA.warnings}
    sv = tr.hits[0] if tr.hits else None
    if sv is None or tr.end is not None and str(tr.end.get("how", "")).startswith("serve_fault"):
        rec["legal"] = False
        rec["fault"] = ("not producible: " + srv.note) if srv.note else (tr.end or {}).get("how", "no serve")
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    fi = S.serve_flight_info(sv)
    rec["flight"] = fi
    rec["type"] = S.classify(fi)
    rec["legal"] = S.bm_in_service(fi)
    if not rec["legal"]:
        rec["fault"] = "serve lands outside the service court"
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    cands, info = O.candidates(m, layer.lead)
    rec.update(info)
    tries, ok, clean = [], None, None
    for i, c in enumerate(cands):
        t = trial(m, layer, c)
        t["i"] = i
        tries.append(t)
        if t.get("ok") and ok is None:
            ok = t
        if t.get("ok") and t["sc_warn"] == 0:
            clean = t
            break
    rec["n_tries"] = len(tries)
    rec["returned"] = ok is not None
    rec["returned_clean"] = clean is not None
    rec["success"] = ok
    rec["success_clean"] = clean
    rec["tries_sc_warn"] = sum(t["sc_warn"] > 0 for t in tries)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main():
    tag = os.environ.get("TAG", "b9")
    plans = S.serve_grid(os.environ.get("COURT", "right"))
    for i, p in enumerate(plans):
        p["i"] = i
    plans = S.shard(plans)
    f = f"/work/out/b9_{tag}_{os.environ.get('SHARD', '0')}.jsonl"
    done = set()
    if os.path.exists(f):
        done = {json.loads(l)["plan"]["i"] for l in open(f) if l.strip()}
    with open(f, "a") as fh:
        for p in plans:
            if p["i"] in done:
                continue
            tried = []
            for cap in KN.SERVE_CAPS:
                p["cap"] = cap
                rec = run(dict(p))
                tried.append({"cap": cap, "legal": rec.get("legal"), "fault": rec.get("fault")})
                if rec.get("legal"):
                    break
            rec["caps_tried"] = tried
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(p["i"], p["fam"], rec.get("legal"), rec.get("returned"), rec.get("returned_clean"), rec.get("wall_s"),
                  flush=True)


if __name__ == "__main__":
    main()
