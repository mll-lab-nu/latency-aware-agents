"""The Season-1 foosball records of the golden-data store (RO_GOLDEN_DATA / --external; one directory per case with
raw/trajectory.json.gz, raw/final.json.gz and manifest.json, recorded by the legacy tasks
foosball-match-wallclock-program-standard and foosball-duel-wallclock-program-standard) -> core episodes, for
tools/golden.py replay foosball.

The legacy record (trajectory.json) is what the legacy replay.py read: the config and the events, an `act` (seat,
rods) at the control step that took it and a `ready` (both reviews in) at the step where play resumed. The conversion
is 1:1: an act at step K becomes the low-level command {"type": "rods", "rods": ...} at step K (plugin.FoosMatch.apply
schedules it for K + 2, as the legacy service did; the values are not touched) and a ready at step K a resume row at
K, before the commands of that step (the legacy replay also handed in the review first). A legacy forfeit becomes a
forfeit row at the last step: start_forfeit -> "no_show", ready_forfeit -> "ready_timeout"; the single player who
never started (no_start, the session collected inside the start window) -> "no_show" (core 0.2.7).

Expected = the legacy result: the state hash (manifest final_sha = foos.Foosball.digest() of the live match: the
MuJoCo integration state, the motor paths, the effect queue, the referee, the scripted team and the seeded restarts),
the scores (goals; format match: {"A", "opponent"}), the winner of the live match and the steps.

Tagged cases (TAGS): the core's rules differ from the legacy service's there; the state hash, the scores and the
steps still replay bit-exactly, only the decision differs as stated."""

import gzip
import json
from pathlib import Path

TASKS = {"foosball-match-wallclock-program-standard": "match", "foosball-duel-wallclock-program-standard": "duel"}
WHY = {"start_forfeit": "no_show", "ready_forfeit": "ready_timeout", "no_start": "no_show"}
TAGS = {
    "foosball-duel-wallclock-program-standard/nop-B": "no-show rule (core 0.2.2): B never started, A wins by no_show; "
    "the same outcome as the legacy service (start_forfeit, A won); a core 0.2.1 live match would be incomplete",
    "foosball-match-wallclock-program-standard/nop": "rule-change:core-0.2.7-prestart-noshow: the single player never "
    "started and its session was collected inside the start window: a valid loss (no_show at the collect, success 0) "
    "without a winner side (cores 0.2.2-0.2.6: an incomplete match); legacy: reason no_start, a valid loss with winner "
    "B (the scripted team)",
    "foosball-match-wallclock-program-standard/abandon-60s": "rule-change:core-0.2.6-abandoned: the player started and "
    "the session ended in the middle of the game. Core 0.2.6: a valid loss by forfeit (abandoned, success 0) without a "
    "winner side, as the legacy service scored it (session_ended, winner B, the scripted team); cores 0.2.2-0.2.5: an "
    "incomplete match (no winner, not a valid trial). The episode carries the forfeit row the core now writes; the "
    "replayed result is the same",
}


def _read(p: Path):
    return json.loads(gzip.open(p, "rt").read())


def cases(store: Path):
    return [p for t in TASKS for p in sorted((store / t).glob("*/")) if (p / "raw" / "trajectory.json.gz").exists()]


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    tr = _read(old / "raw" / "trajectory.json.gz")
    fin = _read(old / "raw" / "final.json.gz")
    m = json.loads((old / "manifest.json").read_text())
    c = tr["config"]
    if bool(c["duel"]) != (fmt == "duel") or tr["live"]["hash"] != m["final_sha"]:
        raise ValueError("the record does not match its task or its manifest")
    seats = ["A", "B"] if fmt == "duel" else ["A"]
    cfg = {
        "sport": "foosball",
        "format": fmt,
        "seats": seats,
        "seed": c["seed"],
        "first": c["first"],
        "action_level": "low",
        "play": "program",
        "params": {},
        "level": None if fmt == "duel" else c["level"],
    }
    ep = [
        {
            "ev": "header",
            "sport": "foosball",
            "revision": "converted:" + tr["revision"],
            "protocol": "realtime",
            "source_sha": None,
            "cfg": cfg,
        }
    ]
    body = []
    for i, e in enumerate(tr["events"]):
        if e["event"] == "ready":
            body.append(((e["step"], 0, i), {"ev": "resume", "k": e["step"]}))
        elif e["event"] == "act":
            body.append(
                (
                    (e["step"], 1, i),
                    {"ev": "cmd", "k": e["step"], "seat": e["seat"], "cmd": {"type": "rods", "rods": e["rods"]}},
                )
            )
        else:
            raise ValueError(f"unknown legacy event {e['event']!r}")
    ep += [r for _, r in sorted(body, key=lambda x: x[0])]
    k = tr["steps"]
    reason = fin.get("reason")
    winner = fin.get("winner")
    if reason in ("start_forfeit", "ready_forfeit"):
        ep.append({"ev": "forfeit", "k": k, "seat": fin["forfeit"], "why": WHY[reason]})
    elif reason == "no_start" and fmt == "match":
        ep.append({"ev": "forfeit", "k": k, "seat": "A", "why": WHY[reason]})
        winner = None  # a single side that forfeits: a loss without a winner side
    elif reason == "session_ended" and fmt == "match":
        ep.append({"ev": "forfeit", "k": k, "seat": "A", "why": "abandoned"})
        winner = None  # core 0.2.6: a valid loss without a winner side (TAGS)
    elif reason != "match_over":
        raise ValueError(f"legacy reason {reason!r} is not convertible")
    ep.append({"ev": "end", "k": k, "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    sc = m["scores"]
    scores = {"A": sc["A"]["points"], ("B" if fmt == "duel" else "opponent"): sc["B"]["points"]}
    if fmt == "match" and winner == "B":
        winner = "opponent"
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": k}
