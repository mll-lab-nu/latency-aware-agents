"""The legacy soccer records of the golden-data store -> core episodes, for tools/golden.py replay soccer --external
STORE (or RO_GOLDEN_DATA). One directory per case: commands.jsonl.gz (header + every applied act + the end row),
manifest.json, raw/ (the legacy service's episode.jsonl.gz and final.json.gz).

All three legacy tasks: soccer-match-wallclock-program-standard (format `match`: seat A against the scripted team B,
whose commands are not acts: the world plays them), soccer-duel-wallclock-program-standard and
soccer-team-wallclock-program-standard. The conversion:
  acts      1:1: {"seat", "robot": "A2", "cmd": {"type": "walk", "v": [...]}} at step k -> a cmd row at step k with
            {"type": "walk", "v": [...], "entity": "2"} (the robot's number), in the commands file's order
  reviews   every review that ended (a legacy `run` row after the first) -> a resume row at that step, before the
            step's acts (the legacy replay resumed the world first, then applied the acts)
  forfeits  a review timeout (the end row's `forfeiting` [team, seats]) -> a forfeit row (seat, "ready_timeout")
  no show   a seat that never started (raw final.json `joined`) -> a forfeit row (seat, "no_show") [INTENDED below]
Expected = the legacy result: the state hash (manifest final_sha = soccer.World.state_sha()), the scores (match:
{"A", "B"} -> {"A", "opponent"}), the winner and the number of control steps.

INTENDED rule changes (the core's, not this sport's; the replayed world and its hash are the legacy ones):
  no_show    core 0.2.2: a side that never starts loses (valid, reason no_show). Legacy duel/team: "incomplete" (no
             winner, not valid) -> v0.2 winner = the other side. Legacy match (single): "collected" with the scripted
             team as the live winner -> core 0.2.7 (tag rule-change:core-0.2.7-prestart-noshow): the single player's
             no-show at the collect, a valid loss without a winner (cores 0.2.2-0.2.6 left a session collected inside
             the start window unfinished).
  abandoned  core 0.2.6: a single player's session that ends mid-match: legacy F6 made it a valid loss (live winner
             B); cores 0.2.2-0.2.5 left the match unfinished (no winner, not a valid trial); since 0.2.6 the core
             writes a forfeit row (seat A, "abandoned"): a valid loss without a winner side (the replayed world, its
             hash and the winner, none, are the same).
"""
import gzip
import json
from pathlib import Path

TASKS = {"soccer-match-wallclock-program-standard": "match", "soccer-duel-wallclock-program-standard": "duel",
         "soccer-team-wallclock-program-standard": "team"}
SEATS = {"match": ["A"], "duel": ["A", "B"], "team": ["A1", "A2", "B1", "B2"]}
SIDES = {"team": {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}}
INTENDED = {}                                           # case -> the intended change it shows (filled by convert())


RETIRED_BY = "soccer-2v2-4"      # the world these records were recorded on (soccer-2v2-3) changed: real T1 limits and
                                # the swing kick (README "Robot limits and the kick"); the same commands no longer
                                # reproduce the legacy state hashes, so none of the store's soccer records is replayed


def cases(store: Path):
    return []                   # all retired (RETIRED_BY); the conversion below documents how they were replayed


def _rows(p: Path):
    return [json.loads(x) for x in gzip.open(p, "rt") if x.strip()] if p.exists() else []


def _cmd(robot: str, c: dict) -> dict:
    t = c["type"]
    out = {"type": t}
    if t == "walk":
        out["v"] = c["v"]
    elif t == "kick":
        out["power"] = c["power"]
    out["entity"] = robot[1]
    return out


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    rows = _rows(old / "commands.jsonl.gz")
    head = rows[0]
    end = next(r for r in reversed(rows) if r["ev"] == "end")
    raw = _rows(old / "raw" / "episode.jsonl.gz")
    fin = json.loads(gzip.open(old / "raw" / "final.json.gz", "rt").read())
    cfg = {"sport": "soccer", "format": fmt, "seats": SEATS[fmt], "seed": head["seed"], "first": head["first_kickoff"],
           "action_level": "low", "play": "program", "params": {}, "sides": SIDES.get(fmt),
           "level": head["level"] if fmt == "match" else None}
    k_end = int(end["k"])
    body = [{"ev": "resume", "k": r["step"]} for r in [r for r in raw if r.get("ev") == "run"][1:]]
    stops = [s["k"] for s in fin["summary"]["stops"] if not s.get("final") and s["k"] < k_end]
    if [r["k"] for r in body] != stops:
        raise ValueError(f"{old.name}: the resumed reviews {[r['k'] for r in body]} differ from the stoppages {stops}")
    body += [{"ev": "cmd", "k": r["k"], "seat": r["seat"], "cmd": _cmd(r["robot"], r["cmd"])}
             for r in rows if r["ev"] == "cmd"]
    body.sort(key=lambda r: (r["k"], r["ev"] != "resume"))          # stable: the acts keep the file's order
    m = json.loads((old / "manifest.json").read_text())
    side = {"A": "A", "B": "opponent" if fmt == "match" else "B"}
    winner = side[m["winner"]] if m.get("winner") else None
    reason = end["reason"]
    joined = fin.get("joined") or {}
    absent = [s for s in SEATS[fmt] if not joined.get(s)]
    if reason == "ready_timeout":
        team, seats = end["forfeiting"]
        body.append({"ev": "forfeit", "k": k_end, "seat": seats[0], "why": "ready_timeout"})
        winner = side["B" if team == "A" else "A"]
    elif k_end == 0 and absent and len(absent) < len(SEATS[fmt]) + (fmt == "match"):
        body.append({"ev": "forfeit", "k": k_end, "seat": absent[0], "why": "no_show"})
        reason = "no_show"
        winner = None if fmt == "match" else ("B" if absent[0][0] == "A" else "A")
        INTENDED[old.name] = f"no_show: {absent[0]} never started (legacy: {end['reason']}, winner {m.get('winner')})"
        if fmt == "match":
            INTENDED[old.name] = "rule-change:core-0.2.7-prestart-noshow: the session was collected inside the start " \
                                 "window (cores 0.2.2-0.2.6: unfinished; core 0.2.7: a valid loss at the collect); " \
                                 + INTENDED[old.name]
    elif reason == "collected" and fmt == "match" and k_end > 0:
        body.append({"ev": "forfeit", "k": k_end, "seat": "A", "why": "abandoned"})
        reason, winner = "abandoned", None
        INTENDED[old.name] = f"rule-change:core-0.2.6-abandoned: a single session that ended at step {k_end} " \
                             f"(legacy F6: a valid loss, winner {m.get('winner')}; cores 0.2.2-0.2.5: unfinished; " \
                             f"core 0.2.6: a valid loss by forfeit, abandoned, no winner side)"
    elif reason == "collected":
        winner = None
        INTENDED[old.name] = f"collected: a session that ended at step {k_end} (legacy: winner {m.get('winner')}; " \
                             f"v0.2: unfinished)"
    body.append({"ev": "end", "k": k_end, "reason": reason})
    ep = [{"ev": "header", "sport": "soccer", "revision": "converted:" + head["revision"], "protocol": "realtime",
           "source_sha": None, "cfg": cfg}] + body
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    scores = {side[t]: int(v) for t, v in m["scores"].items()}
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": int(m["steps"])}


def trace_check(old: Path, out: Path) -> dict:
    """Replay the converted case through the core (python path: the task's sim directory) and compare the world's
    state hash every `hash_every` control steps with the legacy trace (trace.json of the golden-data store). The plugin:
    RO_PLUGIN, else ../plugin.py."""
    from core.episode import load_plugin, read_rows, replay
    tr = json.loads((old / "trace.json").read_text())
    every, want = int(tr["hash_every"]), {int(k): h for k, h in tr["hashes"]}
    got = {}

    def seen(m):
        if m.k % every == 0 and m.k not in got:
            got[m.k] = m.state_digest()

    import os
    sport = load_plugin(os.environ.get("RO_PLUGIN") or str(Path(__file__).resolve().parents[1] / "plugin.py"))
    rep = replay(sport, read_rows(out / "episode.jsonl"), observer=seen)
    got[rep["k"]] = rep["state_sha"]
    bad = sorted(k for k in want if got.get(k) != want[k])
    return {"hashes": len(want), "first_diff": bad[0] if bad else None, "ok": not bad}


if __name__ == "__main__":     # python3 golden/convert.py STORE OUT [--trace]: convert every case (and check its trace)
    import sys
    store, dst = Path(sys.argv[1]), Path(sys.argv[2])
    for c in cases(store):
        d = dst / f"{c.parent.name}--{c.name}"
        d.mkdir(parents=True, exist_ok=True)
        want = convert(c, d)
        (d / "expected.json").write_text(json.dumps({"expect": want, "intended": INTENDED.get(c.name)}, indent=1) + "\n")
        tr = trace_check(c, d) if "--trace" in sys.argv else ""
        print(c.parent.name, c.name, json.dumps(want)[:160], INTENDED.get(c.name, ""), tr, flush=True)
