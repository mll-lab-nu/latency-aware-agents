"""Showcase step 1 (runs in the sport's own sim image, CPU): re-run a finished match from its deterministic replay
record and keep the joint state of every control step in a window, for an offline re-render.

    python3 capture.py <sport> <simdir> <datadir> <out.npz> [--k0 K] [--k1 K] [--no-verify]

<simdir>  a copy of the match's sim directory (the task's environment/sim, same fingerprint as the match)
<datadir> the trial's verifier/ (or artifacts/data/) directory: trajectory.json / episode.jsonl / replay_check.json

The replay is the verifier's own (duel.Duel.step for table tennis / tennis, soccer.World via replay.run for soccer), so
the states are the match's states. Unless --no-verify, the run continues to the end and the final state hash is
checked against the verifier's replay_check.json: a clip is only made from a replay that reproduces the match.

Output npz: q (n, nq) qpos per control step k0..k1-1, k (n,), dt (control step s), sport, mode, events (json),
state_sha (the reproduced final hash, or "" with --no-verify).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def tt_like(simdir, data, k0, k1, verify, module="duel"):
    raw = json.load(open(os.path.join(data, "trajectory.json")))
    os.environ["DUEL_SEED"] = str(raw["seed"])
    os.environ["FIRST_RECEIVER"] = raw["first_receiver"]
    sys.path.insert(0, simdir)
    duel = __import__(module)
    if os.path.exists(os.path.join(simdir, "server_fingerprint.py")):      # tennis
        from server_fingerprint import fingerprint
    else:
        from server_duel import fingerprint
    if raw["source_sha"] != fingerprint() or raw["revision"] != duel.REVISION:
        raise SystemExit("simulator fingerprint mismatch: use the sim directory the match ran with")
    dt = duel.CONTROL_DT
    g = duel.Duel()
    A, B = raw["A"], raw["B"]
    k1 = len(A) if k1 is None else min(k1, len(A))
    qs, ks, ev, extra = [], [], [], []
    for i, (a, b) in enumerate(zip(A, B)):
        if i >= k1 and not verify:
            break
        npts = len(g.points)
        g.step(np.asarray(a, float).ravel(), np.asarray(b, float).ravel())
        if len(g.points) > npts:
            p = g.points[-1]
            ev.append(dict(k=g.k, ev="point", score=dict(g.score), winner=p.get("winner"), how=p.get("how")))
        if k0 <= g.k < k1:
            qs.append(g.d.qpos.copy()); ks.append(g.k)
            if getattr(g, "b_p", None) is not None:          # tennis: the ball is drawn as a mocap body
                extra.append(np.asarray(g.b_p, float).ravel().copy())
    sha = g.state_sha() if verify else ""
    return np.array(qs), np.array(ks), dt, ev, sha, (np.array(extra) if extra else None)


def soccer(simdir, data, k0, k1, verify):
    sys.path.insert(0, simdir)
    import replay as R
    _, meta, k_end, acts = R.load(data)
    import soccer as S
    from server import fingerprint
    if meta["source_sha"] != fingerprint() or meta["revision"] != S.REVISION:
        raise SystemExit("simulator fingerprint mismatch: use the sim directory the match ran with")
    k1 = k_end if k1 is None else min(k1, k_end)
    qs, ks, ev = [], [], []
    nstops = [0]

    class Stop(Exception):
        pass

    def cb(w):
        if len(w.stops) > nstops[0]:
            for s in w.stops[nstops[0]:]:
                ev.append(dict(k=w.k, ev=s.get("kind"), score=dict(w.score), team=s.get("team"), robot=s.get("robot")))
            nstops[0] = len(w.stops)
        if k0 <= w.k < k1:
            qs.append(w.d.qpos.copy()); ks.append(w.k)
        elif w.k >= k1 and not verify:
            raise Stop

    try:
        w = R.run(meta, k_end, acts, cb)
        sha = w.state_sha() if verify else ""
    except Stop:
        sha = ""
    return np.array(qs), np.array(ks), S.CONTROL_DT, ev, sha, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sport"); ap.add_argument("simdir"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--k0", type=int, default=0); ap.add_argument("--k1", type=int, default=None)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    verify = not a.no_verify
    if a.sport == "tabletennis":
        q, k, dt, ev, sha, extra = tt_like(a.simdir, a.data, a.k0, a.k1, verify)
        mode = "duel"
    elif a.sport == "tennis":
        q, k, dt, ev, sha, extra = tt_like(a.simdir, a.data, a.k0, a.k1, verify)
        mode = "duel"
    elif a.sport == "soccer":
        q, k, dt, ev, sha, extra = soccer(a.simdir, a.data, a.k0, a.k1, verify)
        mode = "team"
    else:
        raise SystemExit(f"no showcase capture for {a.sport}")
    if verify:
        want = json.load(open(os.path.join(a.data, "replay_check.json")))["state_sha"]
        if sha != want:
            raise SystemExit(f"replay does not reproduce the match: state_sha {sha} != {want}")
    out = dict(q=q, k=k, dt=dt, sport=a.sport, mode=mode, events=json.dumps(ev), state_sha=sha)
    if extra is not None:
        out["mocap"] = extra
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(frames=len(k), k0=int(k[0]) if len(k) else None, k1=int(k[-1]) if len(k) else None,
                          verified=bool(verify), state_sha=sha, seconds=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
