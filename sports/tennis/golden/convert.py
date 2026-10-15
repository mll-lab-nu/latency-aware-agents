"""RETIRED (revision wallclock-7 + duel-5, 2026-10-07): the players serve, so no legacy record replays; cases() returns
nothing (RETIRED_ALL, as fencing-3). Kept for the history of the migration:

The legacy tennis records of the golden-data store (RO_GOLDEN_DATA; one directory per case: raw/trajectory.json.gz,
raw/final.json.gz, manifest.json) -> core episodes, for tools/golden.py replay tennis --external STORE.

Both legacy tasks keep the joint target of every control step (trajectory.json: the duel's A and B, the single
player's `actions`), which determine everything else. Each point's targets become one low-level `traj` command per
seat at the point's first step (legacy: no command carries over to the next point), the review ends become `resume`
rows, a review forfeit (final.json reason ready_timeout) a `forfeit` row. Expected = the legacy result: the state hash
(manifest final_sha = the legacy simulator's state_sha()), the scores (single: {"you", "opponent"} -> {"A",
"opponent"}), the winner and the steps.

Intended rule change (core 0.2.2, "no_show"): a duel side that never starts LOSES (a valid match, the present side
wins); the legacy service marked that match invalid without a winner (start_timeout). An old record replays its old
decision (core: golden replays of old records stay as recorded), so such a case (nop-B) converts without a forfeit row
and expects no winner, as recorded; its header carries the tag below. A live v0.2 match of the same situation is decided by the core (the present side wins).

Intended rule change (core 0.2.6, "abandoned"): a single player who started and whose session ends mid-match LOSES (a
valid loss, success 0, no winner side); the legacy service also scored it a valid loss (reason timeout), cores
0.2.2-0.2.5 an incomplete match. The record (quit) converts as recorded (no forfeit row, no winner) and its header
carries ABANDON_TAG; a live 0.2.6 match writes an `abandoned` forfeit row, with the same replayed result.

Intended rule change (core 0.2.7, "no_show" at the collect): a single player who never started and whose session ended
inside the start window LOSES (a valid loss, success 0, no winner side), as the legacy service scored it (reason
timeout); cores 0.2.2-0.2.6 an incomplete match. The record (nop) converts as recorded (no forfeit row, no winner) and
its header carries PRESTART_TAG; a live 0.2.7 match writes a `no_show` forfeit row, with the same replayed result."""
import gzip
import json
from pathlib import Path

TASKS = {"tennis-duel-wallclock-program-standard": "duel",
         "tennis-match-wallclock-program-standard": "match"}
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the single player never started and the session ended inside the "
                "start window (legacy: reason timeout, a valid loss, no winner); recorded without a forfeit row; "
                "core 0.2.7 decides such a live match at the collect: a no_show forfeit, a valid loss (success 0, "
                "no winner side; cores 0.2.2-0.2.6: incomplete, not a valid trial); the replayed result (state "
                "hash, scores, no winner, k = 0) is the same")
NO_SHOW_TAG = ("rule-change:core-0.2.2-no_show: recorded under the legacy rule (start_timeout, invalid, no winner); "
               "under core 0.2.2 the side that never started loses and the present side wins")
ABANDON_TAG = ("rule-change:core-0.2.6-abandoned: the single player started and the session ended mid-match (legacy: "
               "reason timeout, a valid loss); recorded without a forfeit row; a core 0.2.6 live match records an "
               "abandoned forfeit (a valid loss, success 0, no winner side; cores 0.2.2-0.2.5: incomplete); the "
               "replayed result is the same")


def _read(p: Path):
    return json.loads(gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text())


RETIRED_ALL = ("tennis-wallclock-7 + duel-5: the players serve (ITF Rules 16-22) instead of the scripted opponent / the "
               "serving machine and the rails cover the whole half; the legacy records no longer replay")


def cases(store: Path):
    if RETIRED_ALL:
        return []
    return sorted(p for t in TASKS for p in (store / t).glob("*/") if (p / "raw" / "trajectory.json.gz").exists())


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    raw = old / "raw"
    tr, fin = _read(raw / "trajectory.json.gz"), _read(raw / "final.json.gz")
    m = json.loads((old / "manifest.json").read_text())
    single = fmt == "match"
    if single:
        cfg = {"sport": "tennis", "format": fmt, "seats": ["A"], "seed": tr["seed"], "level": tr["level"],
               "first": None, "action_level": "high", "play": "program", "params": {}}
        targets = {"A": tr["actions"]}
    else:
        cfg = {"sport": "tennis", "format": fmt, "seats": ["A", "B"], "seed": tr["seed"],
               "first": tr["first_receiver"], "action_level": "high", "play": "program", "params": {}}
        targets = {"A": tr["A"], "B": tr["B"]}
    K = len(targets["A"])
    head = {"ev": "header", "sport": "tennis", "revision": "converted:" + tr["revision"], "protocol": "realtime",
            "source_sha": None, "cfg": cfg}
    rows = []
    cuts = [0] + [p["step"] for p in fin.get("points", []) if 0 < p["step"] < K] + [K]
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        if a > 0:
            rows.append({"ev": "resume", "k": a})
        for s, q in targets.items():
            rows.append({"ev": "cmd", "k": a, "seat": s, "cmd": {"type": "traj", "k0": None, "q": q[a:b]}})
    reason, winner = fin.get("reason"), m.get("winner")
    if single:
        scores = {"A": m["scores"][0], "opponent": m["scores"][1]}
        winner = {"you": "A", "opponent": "opponent"}.get(winner)
        end = "match_over" if reason == "match_over" else "converted"
        if reason == "timeout" and K > 0 and fin.get("started", True):   # started, then the session ended mid-match
            head["tag"] = ABANDON_TAG
        elif reason == "timeout" and K == 0 and fin.get("started") is False:   # never started: a no-show (core 0.2.7)
            head["tag"] = PRESTART_TAG
    else:
        scores = dict(m["scores"])
        end = reason
        if reason == "ready_timeout" and fin.get("forfeiting") in ("A", "B"):
            rows.append({"ev": "forfeit", "k": K, "seat": fin["forfeiting"], "why": "ready_timeout"})
        elif reason == "start_timeout" and K == 0:
            joined = {s: any(r.get("ev") == "start" and r.get("seat") == s for r in _episode(raw)) for s in ("A", "B")}
            if sum(joined.values()) == 1:                            # one side never started: tag the rule change
                head["tag"] = NO_SHOW_TAG
    rows.append({"ev": "end", "k": K, "reason": end})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [head] + rows))
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": K}


def _episode(raw: Path):
    return [json.loads(x) for x in gzip.open(raw / "episode.jsonl.gz", "rt") if x.strip()]
