"""Do the basic shots exist (B9 a)? Upper bound with the official planner: for incoming flights sent to player B
(true state known, B at rest at its home in the ready pose), search contacts on the true flight (at least T_MIN s
ahead, the base reachable in time) and strokes from the documented contact law to each shot type's targets; every
candidate is played on a deep copy of the world through the official high level (planner, real limits, scatter).
A shot type EXISTS for an incoming flight if a candidate lands where the type says. Output results/<TAG>/shots.jsonl.

Types (B's frame, the opponent's court y > 0; L = half length, S = short service line):
  clear   lands >= 0.75 L deep (in)                launch 40-65 deg
  drop    lands <= S + 0.3 (in, beyond the net)    launch 0-45 deg
  drive   launch <= 15 deg and lands in            launch -5..15 deg
  smash   launch < 0 (downward) and lands in       launch -25..-3 deg
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

import study as S

BM, BD, BP = S.BM, S.BD, S.BP
DT = S.DT
T_MIN = float(os.environ.get("T_MIN", "0.5"))
MAXTRIES = int(os.environ.get("MAXTRIES", "30"))
L, SS, X = BM.COURT_Y, BM.SHORT_SERVICE_Y, BM.COURT_X
TYPES = {
    "clear": dict(aims=[(-0.6 * X, 0.85 * L), (0.6 * X, 0.85 * L), (0.0, 0.9 * L)], th=(45, 50, 40, 55, 60, 65)),
    "drop": dict(aims=[(-0.6 * X, SS + 0.1), (0.6 * X, SS + 0.1), (0.0, 0.5 * SS + 0.2)], th=(30, 20, 40, 10, 45, 0)),
    "drive": dict(aims=[(-0.6 * X, 0.7 * L), (0.6 * X, 0.7 * L), (0.0, 0.8 * L)], th=(10, 5, 15, 0, -5)),
    "smash": dict(aims=[(-0.6 * X, 0.5 * L), (0.6 * X, 0.5 * L), (0.0, 0.7 * L)], th=(-5, -10, -3, -15, -20, -25)),
}


def ok_type(kind, v_out, land):
    if land is None or land[2] != "floor":
        return False
    x, y = land[0], -land[1]                      # world (A's frame) -> B's frame: y beyond the net is A's half
    inside = abs(x) <= X and 0 < y <= L
    elev = float(np.degrees(np.arctan2(v_out[2], np.hypot(v_out[0], v_out[1]))))
    return inside and {"clear": y >= 0.75 * L, "drop": y <= SS + 0.3, "drive": elev <= 15.0,
                       "smash": elev < 0.0}[kind]


def world_with_incoming(p_a, v_a, seed=1):
    """A duel world whose shuttle flies to B (state live, B to hit), B at its home in the ready pose, at rest."""
    cfg, m, layer = S.make(seed, "A")
    w = m.g
    for _ in range(5):
        m.step()
    w.rs.update(state="live", expect="B", last_hitter="A", played_k=w.k, serve_flight=False)
    w.sh_p, w.sh_v = tuple(map(float, p_a)), tuple(map(float, v_a))
    w.sh_live, w.hit_enabled = True, True
    return cfg, m, layer


def run_case(p_a, v_a, kind):
    cfg, m, layer = world_with_incoming(p_a, v_a)
    g = m.g
    pb, vb = BD.to_side("B", g.sh_p), BD.to_side("B", g.sh_v)
    P, V = BM.predict(pb, vb, n_max=8000)
    b0 = g.sides["B"].s.d.qpos[g.sides["B"].s.qa[:2]].copy()
    cands = []
    lead = layer.lead
    for j in range(lead + 3, len(P) // 10, 3):
        i = 10 * j - 1
        p, v = P[i], V[i]
        if (j - lead) * DT < T_MIN or not (0.3 <= p[2] <= 2.6 and p[1] <= -0.4):
            continue
        slack = (j - lead) * DT - BM.base_time(float(np.linalg.norm(BM.stance(p) - b0)))
        if slack < -0.05:
            continue
        for aim in TYPES[kind]["aims"]:
            for th, sp, v_out in BP.solve_launch(p, np.array(aim), TYPES[kind]["th"], vmax=BM.RACKET_GAIN * 7.9 + 2):
                n, vel = BP.racket_state(v, v_out)
                if np.linalg.norm(vel) <= 7.9:
                    cands.append((float(np.linalg.norm(vel)) - 0.3 * min((j - lead) * DT, 1.2), j, p, n, vel, th))
    cands.sort(key=lambda c: c[0])
    tries = []
    for c in cands[:MAXTRIES]:
        _, j, p, n, vel, th = c
        m2 = copy.deepcopy(m)
        k0 = m2.k
        body = {"type": "act", "t": round((j - lead) * DT, 4), "pos": [float(x) for x in p],
                "normal": [float(x) for x in n], "vel": [float(x) for x in vel]}
        try:
            S.act_apply(layer, m2, "B", body)
        except S.Refused:
            continue
        hit = None
        while m2.k < k0 + j + 25 and not m2.g.points:
            m2.step()
            hit = next((e for e in m2.g.events if e["kind"] == "hit" and e.get("side") == "B"), None)
            if hit:
                break
        if hit is None:
            tries.append("miss")
            continue
        land = S.landing(np.array(hit["pos"]), np.array(hit["v_out"]))
        if ok_type(kind, BD.to_side("B", hit["v_out"]), land):
            return {"exists": True, "tries": len(tries) + 1, "T": body["t"], "z": round(float(p[2]), 2),
                    "speed": hit["speed"], "launch_deg": round(float(np.degrees(np.arctan2(hit["v_out"][2], np.hypot(*hit["v_out"][:2])))), 1),
                    "land_B": [round(land[0] * -1, 2), round(-land[1], 2)]}
        tries.append("wrong" if land else "none")
    return {"exists": False, "tries": len(tries), "n_cands": len(cands)}


def incoming():
    """A's shots towards B (world frame = A's): lofted shots landing at several depths in B's court, from A's
    mid court, and short high lifts (for overhead contacts)."""
    out = []
    for depth, x in ((0.6 * L, 0.0), (0.85 * L, 0.4), (0.85 * L, -0.4), (0.45 * L, 0.0)):
        for th in (50, 65):
            p0 = np.array([0.0, -1.2, 1.1])
            c = BP.solve_launch(p0, np.array([x, depth]), (th,), vmax=20.0)
            if c:
                out.append((f"lofted th{th} to y={depth:.2f} x={x}", p0, c[0][2]))
    p0 = np.array([0.0, -0.6, 0.6])                # a lift from near the net, high, landing mid court
    c = BP.solve_launch(p0, np.array([0.0, 1.3]), (75,), vmax=20.0)
    if c:
        out.append(("high lift th75 to y=1.3", p0, c[0][2]))
    return out


def main():
    out = S.OUT / os.environ.get("TAG", "shots")
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"shots_{os.environ.get('SHARD', '0')}.jsonl").open("a") as fh:
        for i, (name, p0, v0) in enumerate(incoming()):
            if i % int(os.environ.get("NSHARD", "1")) != int(os.environ.get("SHARD", "0")):
                continue
            for kind in TYPES:
                t0 = time.time()
                r = run_case(p0, v0, kind)
                r.update(incoming=name, kind=kind, wall_s=round(time.time() - t0, 1))
                fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
                print(name, kind, r.get("exists"), r.get("T"), r.get("z"), r.get("launch_deg"), r.get("land_B"), r["wall_s"], flush=True)


if __name__ == "__main__":
    main()
