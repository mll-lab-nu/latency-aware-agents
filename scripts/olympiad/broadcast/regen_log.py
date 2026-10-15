"""Test tooling (program_harness.py): rebuild the broadcast log of a FINISHED table-tennis duel from what its trial
kept, for a "replay as live" test of the program feed when the sim's own /data/broadcast.jsonl is gone with its
container. Runs inside the eai-olympiad image (MuJoCo), offline, on a copy of the match's sim directory:

    python3 regen_log.py <simdir> <data dir with trajectory.json + episode.jsonl> <out.jsonl>

The recorded joint commands are replayed through the sim's own duel.Duel (the deterministic replay the verifier runs,
replay.py) and logged as the live server would (server_duel.bc_frame / bc_event: a frame every 2 control steps, the
point's last state forced, point / pause / run / end events). Every line gets "_ts", the wall time it would have been
written in the real match: the run segments are anchored at the episode log's "run" events (S x dt per step), the
events at the episode log's own times.
"""
from __future__ import annotations

import json
import os
import sys


def main(simdir, data, out):
    raw = json.load(open(os.path.join(data, "trajectory.json")))
    os.environ["DUEL_SEED"] = str(raw["seed"])
    os.environ["FIRST_RECEIVER"] = raw["first_receiver"]
    sys.path.insert(0, simdir)
    import duel  # noqa: E402
    from rcb_tt import CONTROL_DT  # noqa: E402

    ev = [json.loads(ln) for ln in open(os.path.join(data, "episode.jsonl"))]
    ev = [e for e in ev if e.get("ev") in ("start", "run", "pause_for_review", "point", "end")]
    start = next(e for e in ev if e["ev"] == "start")
    S = float(start.get("slowdown") or 1.0)
    runs = [(e["step"], e["ts"]) for e in ev if e["ev"] == "run"]
    f = open(out, "w")

    def w(obj, ts):
        f.write(json.dumps(dict(_ts=round(ts, 3), **obj)) + "\n")

    def ts_of(k):
        k0, t0 = max((r for r in runs if r[0] < k), default=(0, start["ts"]))   # the segment that played step k
        return t0 + (k - k0) * CONTROL_DT * S

    w(dict(ev="hdr", every=2, sport="tabletennis", mode="duel", slowdown=S, dt=CONTROL_DT, seed=raw["seed"],
           first_receiver=raw["first_receiver"], players={"A": "A", "B": "B"}), start["ts"])
    g = duel.Duel()
    fr = lambda k, q: dict(k=int(k), q=[round(float(x), 5) for x in q])
    w(fr(g.k, g.d.qpos), start["ts"])
    A, B = raw["A"], raw["B"]
    point_ev = [e for e in ev if e["ev"] == "point"]
    pause_ev = {e["step"]: e for e in ev if e["ev"] == "pause_for_review"}
    run_at = {k: t for k, t in runs}
    if runs:
        w(dict(ev="run", k=runs[0][0], score={"A": 0, "B": 0}), runs[0][1])
    for a, b in zip(A, B):
        before = len(g.points)
        g.step(a, b)
        ended = len(g.points) > before
        if ended:
            last = g.review_hist[-1]
            w(fr(g.k, last["qpos"]), ts_of(g.k))
            p = g.points[-1]
            pe = point_ev[len(g.points) - 1] if len(point_ev) >= len(g.points) else None
            t = pe["ts"] if pe else ts_of(g.k)
            w(dict(ev="point", k=g.k, score=dict(g.score), winner=p.get("winner"), how=p.get("how"), n=len(g.points)), t)
            if g.k in pause_ev:
                w(dict(ev="pause", k=g.k, score=dict(g.score)), pause_ev[g.k]["ts"])
            if g.k in run_at:
                w(dict(ev="run", k=g.k, score=dict(g.score)), run_at[g.k])
        elif g.k % 2 == 0:
            w(fr(g.k, g.d.qpos), ts_of(g.k))
    end = next((e for e in ev if e["ev"] == "end"), None)
    if end:
        w(dict(ev="end", k=g.k, score=dict(g.score), reason=end.get("reason"), winner=g.winner), end["ts"])
    f.close()
    print(json.dumps(dict(steps=g.k, points=len(g.points), score=g.score)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
