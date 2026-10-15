"""The legacy fencing records of the golden-data store -> core episodes (tools/golden.py replay fencing --external
STORE). The legacy service recorded both fencers' joint targets per control step (raw/trajectory.json, world frame);
each phrase becomes low-level `traj` commands (rows in the seat's own frame, at most 600 per command) at the phrase's
first step, every halt but the last a `resume` row, a review forfeit a `forfeit` row. In the single-player bout only
seat A's targets are commands: the scripted opponent (side B) runs inside the world, as it did in the legacy loop,
and the replay re-runs it (its targets must come out identical, or the state hash differs).

Expected = the legacy result (final.json state_sha, score, winner, steps): a record replays the decisions it holds.
One intended rule change is tagged in the converted header (`rule_change`): under core 0.2.2 a duel side that never
starts LOSES (valid, "no_show"); the legacy server voided the bout (reason incomplete, valid_trial 0). The no-show
record (`nop-B`) is converted as recorded (no forfeit row is invented, no play either), so it replays to the legacy
result (no winner); a v0.2.2 live match in the same situation records a `no_show` forfeit and the present side wins.
A second one (core 0.2.6, "abandoned"): the bout record `abandon` (the fencer started, the session ended mid-bout) is
converted as recorded too and tagged; a 0.2.6 live bout records an `abandoned` forfeit, a valid loss (no winner side,
the same replayed result), where cores 0.2.2-0.2.5 left it incomplete.
A third one (core 0.2.7, "no_show" at the collect): the bout record `nop` (the fencer never started, the session ended
inside the start window; legacy reason timeout, a valid loss) is converted as recorded too and tagged; a 0.2.7 live
bout records a `no_show` forfeit at the collect, a valid loss (no winner side, the same replayed result), where cores
0.2.2-0.2.6 left it incomplete.
"""
import gzip
import json
from pathlib import Path

TASKS = {"fencing-bout-wallclock-program-standard": "bout",
         "fencing-duel-wallclock-program-standard": "duel"}
CHUNK = 600
SIGN = {"A": 1.0, "B": -1.0}


def _read(p: Path):
    return gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()


# Every legacy record is superseded by fencing-3 (2026-10-07, review item C3): the arms are held to the real Franka
# Panda's limits (velocity, acceleration, jerk, torque, torque rate; bodies/franka_panda), so the recorded joint
# targets no longer produce the recorded motion and none replays to its legacy result (they would test the old arm).
# They stay in the store as they are; the replacement records are golden/*/ (re-recorded under fencing-3).
RETIRED_ALL = "fencing-3: the arms are held to the real Franka Panda's joint limits (bodies/franka_panda)"


def cases(store: Path):
    if RETIRED_ALL:
        return []
    out = []
    for task in TASKS:
        for p in sorted((store / task).glob("*/")):
            if (p / "raw" / "final.json.gz").exists() and (p / "raw" / "trajectory.json.gz").exists():
                out.append(p)
    return out


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    raw = old / "raw"
    fin = json.loads(_read(raw / "final.json.gz"))
    tr = json.loads(_read(raw / "trajectory.json.gz"))
    single = fmt == "bout"
    seats = ["A"] if single else ["A", "B"]
    cfg = {"sport": "fencing", "format": fmt, "seats": seats, "seed": int(tr["seed"]),
           "first": tr.get("first_player") or "A", "level": int(tr["level"]) if single else None,
           "action_level": "high", "play": "program", "params": {}}
    rows = [{"ev": "header", "sport": "fencing", "revision": "converted", "protocol": "realtime", "source_sha": None,
             "cfg": cfg, "legacy": {"task": old.parent.name, "case": old.name, "revision": tr.get("revision")}}]
    steps = int(fin["steps"])
    K = len(tr["A"]) // 8 if tr["A"] and not isinstance(tr["A"][0], list) else len(tr["A"])
    acts = {s: (tr[s] if not tr[s] or isinstance(tr[s][0], list) else
                [tr[s][i:i + 8] for i in range(0, len(tr[s]), 8)]) for s in ("A", "B")}
    assert K == steps, (K, steps)
    cuts = [0] + [h["k"] for h in fin.get("halts") or [] if h["k"] < K] + [K]
    for a, b in zip(cuts, cuts[1:]):
        if a > 0:
            rows.append({"ev": "resume", "k": a})
        for s in seats:
            q = [[SIGN[s] * r[0], *r[1:]] for r in acts[s][a:b]]          # rail: world x -> own frame (exact)
            for i in range(0, len(q), CHUNK):
                rows.append({"ev": "cmd", "k": a + i, "seat": s, "cmd": {"type": "traj", "q": q[i:i + CHUNK]}})
    tag = None
    if single:
        winner = "A" if fin.get("won") else ("B" if fin.get("winner") == "opponent" else None)
        if fin.get("reason") == "timeout" and steps > 0:
            tag = ("rule-change:core-0.2.6-abandoned: the fencer started and the session ended mid-bout (legacy reason "
                   "timeout, a valid loss); this record replays that decision as recorded (no forfeit row is "
                   "invented); cores 0.2.2-0.2.5 ended such a live bout collected (incomplete, not a valid trial), "
                   "core 0.2.6 records an abandoned forfeit of seat A: a valid loss, success 0, no winner side (the "
                   "world, the score and the state hash are the same)")
        elif fin.get("reason") == "timeout" and steps == 0:
            tag = ("rule-change:core-0.2.7-prestart-noshow: the fencer never started and the session ended inside the "
                   "start window (legacy reason timeout, a valid loss); this record replays that decision as recorded "
                   "(no forfeit row is invented); cores 0.2.2-0.2.6 ended such a live bout incomplete (not a valid "
                   "trial), core 0.2.7 records a no_show forfeit of seat A at the collect: a valid loss, success 0, no "
                   "winner side (the world, the score and the state hash are the same)")
    else:
        winner = fin.get("winner")
        if fin.get("reason") == "ready_timeout" and fin.get("forfeiting") in seats:
            rows.append({"ev": "forfeit", "k": steps, "seat": fin["forfeiting"], "why": "ready_timeout"})
        elif fin.get("reason") == "incomplete" and sum(bool(v) for v in fin.get("joined", {}).values()) == 1:
            absent = next(s for s, v in fin["joined"].items() if not v)
            present = next(s for s in seats if s != absent)
            tag = (f"core 0.2.2 no-show rule: seat {absent} never started; the legacy server voided the bout (reason "
                   f"incomplete, valid_trial 0, no winner) and this record replays that decision as recorded; under "
                   f"core 0.2.2 a live match records a no_show forfeit of seat {absent}: a valid match won by seat "
                   f"{present} (the world, the score and the state hash are the same: a forfeit is not in the hash)")
    if tag:
        rows[0]["rule_change"] = tag                      # the tag travels with the converted episode
    rows.append({"ev": "end", "k": steps, "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    exp = {"state_sha": fin["state_sha"], "scores": fin["score"] if not single else
           {"A": fin["score"]["you"], "B": fin["score"]["opponent"]}, "winner": winner, "k": steps}
    return exp
