"""The Season-1 air hockey records of the golden-data store (RO_GOLDEN_DATA / --external; one directory per case with
raw/trajectory.json.gz, raw/final.json.gz and manifest.json, recorded by the legacy tasks
airhockey-match-wallclock-program-standard and airhockey-duel-wallclock-program-standard) -> core episodes, for
tools/golden.py replay airhockey.

The legacy record (trajectory.json) holds every applied command with the control step at which it took effect and
every end of a review (`ready`, the step at which play resumed); the legacy replay (replay.py) applied, before step k,
the review end and then the commands of step k in their order. The conversion is 1:1: a command [x, y] -> the
low-level {"type": "target", x, y} (not rounded), a command null (`act none`) -> {"type": "none"}, a review end -> a
resume row, all at their step (the resume first). A legacy forfeit (final.json reason / forfeit) becomes a forfeit
row at the last step: ready_forfeit -> "ready_timeout", start_forfeit and the single player's no_start -> "no_show"
(core 0.2.2's name for a side that never started), the single player's session_ended -> "abandoned" (core 0.2.6).

Expected = the legacy result: the state hash (manifest final_sha = the legacy hockey.Hockey.digest(), the full MuJoCo
integration state, controllers, RNG, scripted opponent and referee), the goals by side (format match: the scripted
opponent's side is "opponent"), the winner and the steps. The winner is the legacy service's (final.json winner: also
the forfeits), except where the rules of the v0.2 core differ (TAGS):
  duel nop-B        B never starts. Legacy: a start forfeit, A won (manifest winner null: the world never ran). Core
                    0.2.2: B loses by no_show, A wins: the same outcome; a core 0.2.1 match would be incomplete.
  match nop         the player never starts; its session (a 30 s nop agent) is collected inside the start window.
                    Legacy: no_start, a valid loss (final winner "B"). Core 0.2.7: no_show at the collect, a valid loss
                    (success 0; cores 0.2.2-0.2.6: collected, not a valid trial); a single player who forfeits has no
                    winner side, so expected winner null.
  match abandon-60s the player starts, idles and the session ends mid-game. Legacy: session_ended, a valid loss
                    (final winner "B"). Core 0.2.6: a valid loss by forfeit ("abandoned", success 0; cores 0.2.2-0.2.5
                    left it an incomplete match); a single player's forfeit has no winner side, so expected winner
                    null. The replay is the same either way (308 control steps of the scripted opponent alone)."""
import gzip
import json
from pathlib import Path

TASKS = {"airhockey-match-wallclock-program-standard": "match", "airhockey-duel-wallclock-program-standard": "duel"}
WHY = {"ready_forfeit": "ready_timeout", "start_forfeit": "no_show", "no_start": "no_show",
       "session_ended": "abandoned"}
TAGS = {
    "airhockey-duel-wallclock-program-standard/nop-B":
        "no-show rule (core 0.2.2): B never started and loses by no_show, A wins; same outcome as the legacy service "
        "(start_forfeit, A won; the legacy manifest winner is null); a core 0.2.1 match would be incomplete",
    "airhockey-match-wallclock-program-standard/nop":
        "rule-change:core-0.2.7-prestart-noshow: the single player never started and its session was collected inside "
        "the start window: a valid loss by no_show (success 0) at the collect, as in the legacy no_start (cores "
        "0.2.2-0.2.6: collected, not a valid trial); a single player's forfeit has no winner side (legacy final "
        "winner: B); the replay itself is identical",
    "airhockey-match-wallclock-program-standard/abandon-60s":
        "rule-change:core-0.2.6-abandoned: a started single game collected mid-game is a valid loss by forfeit "
        "(abandoned, success 0) as the legacy session_ended (final winner: B; a single player's forfeit has no winner "
        "side); cores 0.2.2-0.2.5 left it incomplete (technical failure); the replay itself is identical",
}


def _read(p: Path):
    return json.loads(gzip.open(p, "rt").read())


# Revision v3 (2026-10-07, review item B13) widened the goal mouth from 0.25 m to 0.40 m (sport.toml D10). The records
# in which the game was played (o-L1, oo-s1-A, oo-s1-B, ready-forfeit-B) belong to revision v2 and no longer replay
# (their commands now meet a different table); golden/ holds native v3 recordings instead. The three records without
# a played rally that the referee scores (nop, nop-B: the world never ran; abandon-60s: 308 steps of the scripted
# opponent alone, no goal) replay identically under v3 and keep covering the no-show and abandoned rules. Revision v4
# (2026-10-08) narrowed the mouth to 0.35 m: the three kept records still replay identically, and the native cases of
# golden/ were re-recorded under v4. Revision v5 (2026-10-08, the real iiwa 14 limits) adds the arms' setpoint
# generators to the state hash and moves the scripted opponent under the real limits, so no legacy record keeps its
# legacy hash: all seven are retired; golden/ keeps the three converted no-play records (duel-nop-B, match-nop,
# match-abandon-60s) with their v5 hashes (scores, winner and steps unchanged) and the native v5 recordings. Revision
# v6 (2026-10-10, the mallet within +-2 cm of the table) adds the mallets' path generators to the state hash: the three
# kept records carry their v6 hashes (scores, winner and steps unchanged), the native cases were re-recorded under v6.
RETIRED = {"o-L1", "oo-s1-A", "oo-s1-B", "ready-forfeit-B", "nop", "nop-B", "abandon-60s"}


def cases(store: Path):
    return [p for t in TASKS for p in sorted((store / t).glob("*/"))
            if (p / "raw" / "trajectory.json.gz").exists() and p.name not in RETIRED]


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    tr = _read(old / "raw" / "trajectory.json.gz")
    fin = _read(old / "raw" / "final.json.gz")
    m = json.loads((old / "manifest.json").read_text())
    c = tr["config"]
    if tr["live"]["hash"] != m["final_sha"] or bool(c["duel"]) != (fmt == "duel"):
        raise ValueError("the record's live hash is not the manifest's final_sha, or the format differs")
    seats = ["A", "B"] if c["duel"] else ["A"]
    cfg = {"sport": "airhockey", "format": fmt, "seats": seats, "seed": c["seed"], "first": c["first"],
           "action_level": "low", "play": "program", "params": {}, "level": None if c["duel"] else c["level"]}
    ep = [{"ev": "header", "sport": "airhockey", "revision": "converted:" + tr["revision"], "protocol": "realtime",
           "source_sha": None, "cfg": cfg}]
    body = []
    for e in tr["events"]:
        if e["event"] == "ready":
            body.append({"ev": "resume", "k": e["step"]})
        elif e["event"] == "act":
            x = e["command"]
            cmd = {"type": "none"} if x is None else {"type": "target", "x": x[0], "y": x[1]}
            body.append({"ev": "cmd", "k": e["step"], "seat": e["seat"], "cmd": cmd})
        else:
            raise ValueError(f"unknown legacy event {e['event']!r}")
    body.sort(key=lambda r: (r["k"], r["ev"] != "resume"))          # stable: commands keep their order
    ep += body
    k = tr["steps"]
    reason = fin.get("reason")
    seat = fin.get("forfeit") or ("A" if reason in ("no_start", "session_ended") and fmt == "match" else None)
    if reason in WHY and seat:
        ep.append({"ev": "forfeit", "k": k, "seat": seat, "why": WHY[reason]})
    ep.append({"ev": "end", "k": k, "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    side = {"A": "A", "B": "B" if c["duel"] else "opponent"}
    sc = tr["live"]["scores"]
    winner = fin.get("winner") if reason in ("match_over", "ready_forfeit", "start_forfeit") else None
    return {"state_sha": m["final_sha"], "scores": {side[s]: sc[s]["points"] for s in ("A", "B")},
            "winner": side.get(winner) if winner else None, "k": k}
