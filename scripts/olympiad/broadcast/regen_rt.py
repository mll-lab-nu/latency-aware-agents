"""Test tooling (program_harness.py): rebuild the broadcast log of a FINISHED racing or air hockey match from what its
trial kept, for a "replay as live" test of the program feed (the sim's own /data/broadcast.jsonl is gone with its
container). Runs inside the eai-olympiad image (MuJoCo), offline, on a copy of the match's sim directory:

    python3 regen_rt.py <racing|airhockey> <simdir> <data dir with trajectory.json + episode.jsonl + final.json> <out.jsonl>

The recorded commands and review transitions are replayed through the sim's own world / referee (race.Race,
hockey.Hockey: the deterministic replay the verifier runs, replay.py) and logged as the live server does
(server.py with broadcast_log.BroadcastLog: racing a frame every 4 control steps plus every step with referee events,
start / run / checkpoint / contact / crash / disqualified / heat_end / pause / review / end; air hockey a frame every 2
steps, run / point / stoppage / pause / end). Every line gets "_ts", the wall time it would have been written in the
real match: the run segments are anchored at the episode log's "run" (racing) or last start / review (air hockey)
times (S x dt per step), reviews and the end at the episode log's own times. The replayed result must equal the
recorded one (else exit 1).
"""
from __future__ import annotations

import json
import os
import sys


class Log:
    def __init__(self, path, every):
        self.f, self.every, self.last_k = open(path, "w"), every, None

    def w(self, obj, ts):
        self.f.write(json.dumps(dict(_ts=round(ts, 3), **obj)) + "\n")

    def frame(self, k, q, ts, force=False, **extra):
        if k == self.last_k or (not force and k % self.every):
            return
        self.last_k = k
        self.w(dict(k=int(k), q=[round(float(x), 5) for x in q], **extra), ts)

    def event(self, ev, k, ts, **kw):
        self.w(dict(ev=ev, k=int(k), ts=round(ts, 3), **kw), ts)


def load(data):
    raw = json.load(open(os.path.join(data, "trajectory.json")))
    ep = [json.loads(ln) for ln in open(os.path.join(data, "episode.jsonl")) if ln.strip()]
    final = json.load(open(os.path.join(data, "final.json")))
    return raw, [e for e in ep if e.get("event") != "act_applied"], final


def schedule(raw):
    acts, ready = {}, set()
    for e in raw["events"]:
        if e["event"] == "act":
            acts.setdefault(e["step"], []).append(e)
        elif e["event"] == "ready":
            ready.add(e["step"])
    return acts, ready


def racing(simdir, data, out):
    sys.path.insert(0, simdir)
    import race  # noqa: E402
    raw, ep, final = load(data)
    S = float(final.get("slowdown") or 1.0)
    cfg = raw["config"]
    r = race.Race(**cfg)
    duel = bool(cfg.get("duel"))
    starts = [e for e in ep if e["event"] == "start"]
    runs = [(e["step"], e["wall"]) for e in ep if e["event"] == "run"]
    reviews = [e for e in ep if e["event"] == "review"]
    end = next((e for e in ep if e["event"] == "end"), None)
    t_start = min(e["wall"] for e in starts)

    def ts_of(k):
        k0, t0 = max((x for x in runs if x[0] <= k - 1), default=(0, runs[0][1] if runs else t_start))
        return t0 + (k - k0) * race.DT * S

    log = Log(out, 4)
    log.w(dict(ev="hdr", ts=round(t_start, 3), every=4, sport="racing", mode="duel" if duel else "single", slowdown=S,
               dt=race.DT, seed=cfg["seed"], level=cfg["level"], first_player=cfg.get("first", "A"), nq=int(r.model.nq),
               players={"A": "A", "B": "B" if duel else "opponent"},
               track=dict(straight_half_m=r.a, radius_m=r.radius, width_m=2.4, gates=race.GATES),
               qpos_layout={s: int(r.qadr[s]) for s in ("A", "B")}), t_start)
    log.frame(0, r.data.qpos, t_start, force=True, heat=1)

    def bc(ev, ts, **kw):
        log.event(ev, r.k, ts, heat=r.heat + 1, heats_won=dict(r.wins), **kw)

    def public(res):
        return dict(heat=res["heat"], winner=res["winner"], decided_by=res["decided_by"], heat_time=res["heat_time"],
                    winner_finish_time=res["finish_time"], grid=res["grid"],
                    cars={s: dict(checkpoints=v["checkpoints"], finish_time=v["finish_time"]) for s, v in res["cars"].items()},
                    crashes=[dict(t=c["t"], at_fault=c["at_fault"], kind=c["kind"]) for c in res["crashes"]])

    for e in starts:
        bc("start", e["wall"], seat=e["seat"])
    acts, ready = schedule(raw)
    run_walls = dict(runs)
    if 0 in run_walls:
        bc("run", run_walls[0], grid=dict(r.grid))
    for k in range(raw["steps"]):
        if k in ready:
            for rv in [x for x in reviews if x["step"] == k]:
                bc("review", rv["wall"], seat=rv["seat"], why=rv["why"][:300], plan=rv["plan"][:300])
            r.ready()
            log.frame(r.k, r.data.qpos, run_walls.get(k, ts_of(k)), force=True, heat=r.heat + 1)
            bc("run", run_walls.get(k, ts_of(k)), grid=dict(r.grid))
        for a in acts.get(k, []):
            r.set_command(a["seat"], a["command"])
        r.step()
        ts = ts_of(r.k)
        for e in r.events:
            kind, fields = e["ev"], {x: v for x, v in e.items() if x != "ev"}
            if kind == "heat_end":
                bc("heat_end", ts, result=public(e["result"]))
            else:
                bc(kind, ts, **fields)
        log.frame(r.k, r.data.qpos, ts, force=bool(r.events), heat=r.heat + 1)
        if r.awaiting_review:
            bc("pause", ts, review=True)
        if r.done:
            break
    if r.done:
        bc("end", end["wall"] if end else ts_of(r.k), reason="race_over", winner=r.winner())
    log.f.close()
    s = r.summary()
    ok = s["hash"] == raw["live"].get("hash")
    print(json.dumps(dict(steps=r.k, heats=len(r.results), winner=r.winner(), crashes=len(r.crashes), matches_live=ok)))
    return ok


def airhockey(simdir, data, out):
    sys.path.insert(0, simdir)
    import hockey  # noqa: E402
    raw, ep, final = load(data)
    S = float(final.get("slowdown") or 1.0)
    cfg = raw["config"]
    r = hockey.Hockey(**cfg)
    duel = bool(cfg.get("duel"))
    starts = [e for e in ep if e["event"] == "start"]
    reviews = [e for e in ep if e["event"] == "review"]
    end = next((e for e in ep if e["event"] == "end"), None)
    t_start, t_run = min(e["wall"] for e in starts), max(e["wall"] for e in starts)
    acts, ready = schedule(raw)
    # the world runs again when the last review of a pause is in
    runs = [(0, t_run)] + [(k, max(x["wall"] for x in reviews if x["step"] == k)) for k in sorted(ready)
                           if any(x["step"] == k for x in reviews)]
    run_walls = dict(runs)

    def ts_of(k):
        k0, t0 = max((x for x in runs if x[0] <= k - 1), default=runs[0])
        return t0 + (k - k0) * hockey.DT * S

    log = Log(out, 2)
    log.w(dict(ev="hdr", ts=round(t_start, 3), every=2, sport="airhockey", mode="duel" if duel else "single",
               slowdown=S, dt=hockey.DT, level=cfg["level"], seed=cfg["seed"], nq=int(r.model.nq), first=cfg.get("first", "A"),
               players={"A": "seat A", "B": "seat B"} if duel else {"A": "you", "B": "opponent"}), t_start)
    log.frame(0, r.data.qpos, t_start, force=True)

    def bc(ev, ts, **kw):
        log.event(ev, r.k, ts, score={s: v["points"] for s, v in r.scores().items()}, **kw)

    bc("run", t_run, rally=1)
    for k in range(raw["steps"]):
        if k in ready:
            r.ready()
            bc("run", run_walls.get(k, ts_of(k)), rally=len(r.results) + 1)
        for a in acts.get(k, []):
            r.set_command(a["seat"], a["command"])
        n = len(r.results)
        r.step()
        ts = ts_of(r.k)
        log.frame(r.k, r.data.qpos, ts)
        if len(r.results) > n:
            res = r.results[-1]
            log.frame(r.k, r.data.qpos, ts, force=True)
            bc("point" if res["how"] == "goal" else "stoppage", ts, how=res["how"], winner=res["winner"],
               guilty=res["guilty"], next=res["next"], rally=res["rally"])
        if r.awaiting_review:
            bc("pause", ts, rally=len(r.results))
        if r.done:
            break
    if r.done:
        bc("end", end["wall"] if end else ts_of(r.k), reason="match_over", forfeiting=None, winner=r.winner())
    log.f.close()
    ok = r.digest() == raw["live"].get("hash")
    print(json.dumps(dict(steps=r.k, rallies=len(r.results), winner=r.winner(), decided_by=r.decided_by,
                          scores=r.scores(), matches_live=ok)))
    return ok


if __name__ == "__main__":
    sport, simdir, data, out = sys.argv[1:5]
    sys.exit(0 if {"racing": racing, "airhockey": airhockey}[sport](simdir, data, out) else 1)
