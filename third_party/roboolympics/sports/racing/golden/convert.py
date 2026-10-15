"""The Season-1 racing records of the golden-data store (RO_GOLDEN_DATA / --external; one directory per case with
raw/trajectory.json.gz, raw/final.json.gz and manifest.json, recorded by the legacy tasks
racing-{match,duel,race}-wallclock-program-standard) -> core episodes, for tools/golden.py replay racing.

The legacy record (trajectory.json) is the legacy replay's input: the race configuration, the number of control steps
and the events, each at the control step before which it took effect: `act` (a seat's command [speed, steering]),
`ready` (a review ended: the next heat starts) and, in the multi-car race, `retire` (a car whose seat never started,
`did_not_start`, or missed a review, `review_forfeit`). The conversion is 1:1 and keeps the legacy order within a step
(retirements, then the review's end, then the commands): act -> {"ev": "cmd", "cmd": {"type": "drive", "speed",
"steering"}} (the numbers unrounded; a legacy `none` was logged as [0, 0], the same command), ready -> {"ev": "resume"},
retire -> {"ev": "retire"} with the core's reason (no_show, ready_timeout; plugin.RETIRE_WHY maps it back, since
race.py keeps the reason in its state hash). A legacy forfeit, which the record does not hold, comes from final.json
(reason, winner): start_forfeit -> a forfeit row "no_show" of the seat that did not start, ready_forfeit -> a forfeit
row "ready_timeout" of the seat that missed the review.

Expected = the legacy result: the state hash (manifest final_sha = race.Race.digest() of the live race), the scores
(race.Race.scores(); the match format's scripted car B is the side "opponent"), the winner the v0.2 core decides from
the converted episode, and the number of control steps.

Tagged cases (TAGS, recorded in expected.json): the outcome differs from the legacy service's, by an intended rule of
the v0.2 core, while the world replays bit-exactly."""

import gzip
import hashlib
import json
from pathlib import Path

TASKS = {
    "racing-match-wallclock-program-standard": "match",
    "racing-duel-wallclock-program-standard": "duel",
    "racing-race-wallclock-program-standard": "race",
}
WHY = {"did_not_start": "no_show", "review_forfeit": "ready_timeout"}
ORDER = {"retire": 0, "ready": 1, "act": 2}
TAGS = {
    "racing-duel-wallclock-program-standard/nop-B": "no-show rule (core 0.2.2): B never started, A wins by no_show (valid); the legacy racing server also gave A "
    "the win (start_forfeit); core 0.2.1 would have ended the match incomplete",
    "racing-match-wallclock-program-standard/nop": "rule-change:core-0.2.7-prestart-noshow: the single player never started and the session was collected "
    "inside the start window; legacy: a valid loss (no_start) whenever the session was collected; core 0.2.7: the same, a "
    "valid loss (no_show) at the collect or when the start window closes, whichever comes first (cores 0.2.2-0.2.6: a "
    "session collected before the window closed was an incomplete match)",
    "racing-race-wallclock-program-standard/nop-D": "no-show rule (core 0.2.2) with more than two sides: D never started and is retired (Match.retire), the "
    "others race on; the same retirement as the legacy server's (did_not_start)",
    "racing-match-wallclock-program-standard/abandon-60s": "rule-change:core-0.2.6-abandoned: session collected mid-race. Core 0.2.6: a started single "
    "player whose session ends before the end loses (forfeit row `abandoned`, a valid loss, success 0), as the legacy "
    "racing server scored it (session_ended); cores 0.2.2-0.2.5 ended it `collected` (incomplete, no valid result). "
    "The world replays bit-exactly; the replayed result (no winner) is the same",
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
    if tr["live"]["hash"] != m["final_sha"]:
        raise ValueError("the record's live hash is not the manifest's final_sha")
    c = tr["config"]
    seats = ["A"] if fmt == "match" else list("ABCDEF"[: c["cars"]])
    cfg = {
        "sport": "racing",
        "format": fmt,
        "seats": seats,
        "seed": c["seed"],
        "first": c["first"],
        "level": c["level"] if fmt == "match" else None,
        "action_level": "low",
        "play": "program",
        "params": {},
    }
    ep = [
        {
            "ev": "header",
            "sport": "racing",
            "revision": "converted:race-v3",
            "protocol": "realtime",
            "source_sha": None,
            "cfg": cfg,
        }
    ]
    events = sorted(enumerate(tr["events"]), key=lambda x: (x[1]["step"], ORDER[x[1]["event"]], x[0]))
    for _, e in events:
        if e["event"] == "act":
            v, s = e["command"]
            ep.append(
                {"ev": "cmd", "k": e["step"], "seat": e["seat"], "cmd": {"type": "drive", "speed": v, "steering": s}}
            )
        elif e["event"] == "ready":
            ep.append({"ev": "resume", "k": e["step"]})
        elif e["event"] == "retire":
            ep.append({"ev": "retire", "k": e["step"], "seat": e["seat"], "side": e["seat"], "why": WHY[e["why"]]})
        else:
            raise ValueError(f"unknown legacy event {e['event']!r}")
    k, reason, winner = tr["steps"], fin["reason"], fin.get("winner")
    if reason == "race_over":
        end = "match_over"
        winner = "opponent" if fmt == "match" and winner == "B" else winner
    elif reason == "start_forfeit" and fmt != "match":
        lost = next(s for s in seats if s not in fin["started"])
        ep.append({"ev": "forfeit", "k": k, "seat": lost, "why": "no_show"})
        end = "no_show"
    elif reason == "ready_forfeit" and fmt == "duel":
        lost = "B" if winner == "A" else "A"
        ep.append({"ev": "forfeit", "k": k, "seat": lost, "why": "ready_timeout"})
        end = "ready_timeout"
    elif reason == "no_start" and fmt == "match":
        ep.append({"ev": "forfeit", "k": k, "seat": "A", "why": "no_show"})
        end, winner = "no_show", None
    elif reason == "session_ended" and fmt == "match":
        # INTENDED RULE CHANGE (core 0.2.6): a started single player whose session ends mid-race loses (abandoned)
        ep.append({"ev": "forfeit", "k": k, "seat": "A", "why": "abandoned"})
        end, winner = "abandoned", None
    elif reason == "session_ended":
        end, winner = "collected", None
    else:
        end, winner = "incomplete", None
    ep.append({"ev": "end", "k": k, "reason": end})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    scores = {("opponent" if fmt == "match" and s == "B" else s): v for s, v in m["scores"].items()}
    if fmt == "race" and fin["results"] and reason == "race_over":
        order = classification(tr["live"], c["seed"], seats)
        scores = {s: dict(v, position=order.index(s) + 1) for s, v in scores.items()}
        winner = order[0] if order[0] not in tr["live"]["retired"] else None
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": k}


def lot_key(seed, heat, what, seat):  # = race.lot_key
    return hashlib.sha256(f"{seed}:{heat}:{what}:{seat}".encode()).hexdigest()


def classification(live, seed, seats):
    """The v0.2 classification (plugin.Race.classification, deviation D10) of a legacy race summary: points, count-back,
    then the seed's draw (legacy: the better place in the last heat); retired cars after the running ones as before."""
    pos = {s: [r["positions"].get(s) for r in live["results"]] for s in seats}
    points = {s: live["scores"][s]["points"] for s in seats}
    retired = live["retired"]

    def key(s):
        return (
            -points[s],
            *[-pos[s].count(i) for i in range(1, len(seats) + 1)],
            lot_key(seed, 0, "classification", s),
        )

    run = sorted([s for s in seats if s not in retired], key=key)
    ret = sorted(retired, key=lambda s: (-retired[s]["step"], -points[s], lot_key(seed, 0, "retired", s)))
    return run + ret


def tag(old: Path):
    return TAGS.get(f"{old.parent.name}/{old.name}")
