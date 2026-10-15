"""The Season-1 drone records of the golden-data store (RO_GOLDEN_DATA / --external; one directory per case with
raw/trajectory.json.gz, raw/final.json.gz and manifest.json, recorded by the legacy tasks
drone-racing-wallclock-program-standard and drone-duel-wallclock-program-standard) -> core episodes, for
tools/golden.py replay drone.

The legacy record (trajectory.json) holds per heat every seat's actions with the heat tick at which each took effect;
the conversion is 1:1: an action {velocity, yaw_rate, duration} at tick t of a heat that starts at global step K
becomes the low-level command {"type": "fly", vx, vy, vz, yaw_rate, duration} at step K + t (the actions are not
rounded), and every heat after the first starts with a resume row at its first step. A legacy forfeit (final.json
`forfeit`, `stop_reason`) becomes a forfeit row at the last step: ready_timeout -> "ready_timeout", start_timeout ->
"no_show" (core 0.2.2's name for a side that never started).

Expected = the legacy result: the state hash is sha_json({"races": final_hash, "heats": the closed heats}) of the
legacy record (plugin.Drone.state_digest(); final_hash = every seat's race.Race.state_hash() of the last heat, whose
sha_json is the manifest's final_sha, checked below), so an identical hash means every seat's MuJoCo world and every
heat result replays bit-exactly; the scores (duel: heat wins; racing: rounds finished), the winner of the live match
(final.json winner: also the forfeits) and the steps.

Tagged cases (TAGS): the duel's nop-B (B never starts). The legacy drone server already gave the starter the win
(start_timeout forfeit, valid). Core 0.2.1 would end such a match "incomplete"; core 0.2.2's no-show rule makes the
present side win (valid), i.e. the legacy outcome again. The converted episode carries the forfeit row "no_show" that
a 0.2.2 service writes, so the replay gives A the win under either core (the replay applies forfeit rows as logged).
The racing record idle (the pilot started, then the session ended mid-event, legacy stop_reason session_ended): core
0.2.6 makes it a valid loss (forfeit row "abandoned"; cores 0.2.2-0.2.5: collected, incomplete); the same replayed
result. The racing record nop (the pilot never started, the session ended inside the start window, legacy stop_reason
not_started): core 0.2.7 makes it a valid loss at the collect (forfeit row "no_show"; cores 0.2.2-0.2.6: collected,
incomplete); the same replayed result."""
import gzip
import hashlib
import json
from pathlib import Path

TASKS = {"drone-racing-wallclock-program-standard": "racing",
         "drone-duel-wallclock-program-standard": "duel"}
WHY = {"ready_timeout": "ready_timeout", "start_timeout": "no_show"}
TAGS = {"drone-duel-wallclock-program-standard/nop-B": "no-show rule (core 0.2.2): A wins by no_show; same outcome as "
        "the legacy drone server (start_timeout forfeit); a core 0.2.1 live match would be incomplete",
        "drone-racing-wallclock-program-standard/idle": "rule-change:core-0.2.6-abandoned: the single pilot started, "
        "never flew and the session ended at the first review. Core 0.2.6: a valid loss by forfeit (abandoned, success "
        "0); cores 0.2.2-0.2.5 ended it collected (incomplete, not a valid trial). The episode carries the forfeit row "
        "the core now writes; the replayed result (no winner) is the same",
        "drone-racing-wallclock-program-standard/nop": "rule-change:core-0.2.7-prestart-noshow: the single pilot never "
        "started and the session ended inside the start window (legacy: not_started, a valid loss). Core 0.2.7: a valid "
        "loss by forfeit (no_show, success 0, no winner side) decided at the collect; cores 0.2.2-0.2.6 ended it "
        "collected (incomplete, not a valid trial). The episode carries the forfeit row the core now writes; the "
        "replayed result (no winner, k = 0) is the same"}


def sha_json(value) -> str:                        # = core.sport.sha_json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def _read(p: Path):
    return json.loads(gzip.open(p, "rt").read())


def cases(store: Path):
    return [p for t in TASKS for p in sorted((store / t).glob("*/")) if (p / "raw" / "trajectory.json.gz").exists()]


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    tr = _read(old / "raw" / "trajectory.json.gz")
    fin = _read(old / "raw" / "final.json.gz")
    m = json.loads((old / "manifest.json").read_text())
    if sha_json(tr["final_hash"]) != m["final_sha"]:
        raise ValueError("the record's final_hash is not the manifest's final_sha")
    seats = tr["seats"]
    cfg = {"sport": "drone", "format": fmt, "seats": seats, "seed": tr["seed"], "first": "A", "action_level": "low",
           "play": "program", "params": {}, "level": None if len(seats) == 2 else tr["level"]}
    ep = [{"ev": "header", "sport": "drone", "revision": "converted:" + tr["version"], "protocol": "realtime",
           "source_sha": None, "cfg": cfg}]
    base = 0
    for i, h in enumerate(tr["heats"]):
        if i:
            ep.append({"ev": "resume", "k": base})
        acts = sorted(((a["tick"], s, a["action"]) for s, lst in h["actions"].items() for a in lst),
                      key=lambda x: (x[0], seats.index(x[1])))
        for t, s, a in acts:
            v = a["velocity"]
            ep.append({"ev": "cmd", "k": base + t, "seat": s,
                       "cmd": {"type": "fly", "vx": v[0], "vy": v[1], "vz": v[2], "yaw_rate": a["yaw_rate"],
                               "duration": a["duration"]}})
        base += h["ticks"]
    if fin.get("forfeit"):
        ep.append({"ev": "forfeit", "k": base, "seat": fin["forfeit"], "why": WHY[fin["stop_reason"]]})
    elif len(seats) == 1 and fin.get("stop_reason") == "session_ended" and fin.get("started"):
        # INTENDED RULE CHANGE (core 0.2.6): a started single pilot whose session ends mid-event loses (abandoned)
        ep.append({"ev": "forfeit", "k": base, "seat": seats[0], "why": "abandoned"})
    elif len(seats) == 1 and fin.get("stop_reason") == "not_started" and not fin.get("started"):
        # INTENDED RULE CHANGE (core 0.2.7): a session that ends before the start is a no-show
        ep.append({"ev": "forfeit", "k": base, "seat": seats[0], "why": "no_show"})
    ep.append({"ev": "end", "k": base, "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    ev = tr["event"]
    if len(seats) == 2:
        scores = ev["wins"]
    else:
        scores = {"A": ev["standings"]["A"]["finished"]}
    return {"state_sha": sha_json({"races": tr["final_hash"], "heats": ev["heats"]}), "scores": scores,
            "winner": fin.get("winner"), "k": base}
