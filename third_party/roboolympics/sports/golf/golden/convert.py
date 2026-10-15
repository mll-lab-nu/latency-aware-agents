"""The Season-1 mini-golf putting records of the golden-data store (RO_GOLDEN_DATA / --external; one directory per case
with commands.jsonl.gz and manifest.json, recorded by the legacy task golf-putting-program-standard) -> core episodes,
for tools/golden.py replay golf. The records of the legacy duel (golf-putting-duel-program-standard) are no longer
converted: the duel was retired on 2026-10-08 (no interaction between the players, as review item B10).

The legacy service (the shared turn service, byte-identical to bowling's) logged every stroke with both chess clocks
and its shot number, and each forfeit; the conversion is 1:1 (strokes in shot order, clock_out -> forfeit "clock",
start_timeout -> forfeit "no_show"). The expected result is the legacy one: scores,
winner, strokes (k) and the state hash, which is sha_json of the legacy per-seat round hashes (manifest final_sha,
sim/sport.py Match.state_sha: the round's and the play-off holes' Course.state_sha) - the plugin's state_digest() -
so an identical hash means every seat's ball after every stroke replays bit-exactly.

Tagged (PRESTART_TAG, in the converted header): the single record nop (the recorded player never started, its session
ended inside the start window): core 0.2.7 decides it at the collect, a valid loss by forfeit (no_show); the converted
episode carries that forfeit row."""

import gzip
import hashlib
import json
from pathlib import Path

TASKS = ("golf-putting-program-standard",)  # the duel's records retired with the duel (2026-10-08)
KEYS = ("aim_deg", "speed")  # sim/sport.py ACTION_KEYS
WHY = {"clock_out": "clock", "start_timeout": "no_show"}  # the core's names (no_show: core 0.2.2)
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the single player never started (recorded player nop) and the "
               "session ended inside the start window (legacy: a valid loss, no winner). Core 0.2.7: a valid loss by "
               "forfeit (no_show, success 0, no winner side) decided at the collect; cores 0.2.2-0.2.6 ended it "
               "unfinished (not a valid trial). The episode carries the forfeit row the core now writes; the "
               "replayed result (state hash, scores, no winner, k = 0) is the same")


def sha_json(value) -> str:  # = core.sport.sha_json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def cases(store: Path):
    return [p for t in TASKS for p in sorted((store / t).glob("*/")) if (p / "commands.jsonl.gz").exists()]


def convert(old: Path, out: Path) -> dict:
    rows = [json.loads(x) for x in gzip.open(old / "commands.jsonl.gz", "rt") if x.strip()]
    head = rows[0]
    seats = head["seats"]
    cfg = {
        "sport": "golf",
        "format": "duel" if len(seats) > 1 else "single",
        "seats": seats,
        "seed": head["seed"],
        "first": head["first_player"],
        "action_level": "low",
        "play": "program",
        "params": {},
    }
    ep = [
        {
            "ev": "header",
            "sport": "golf",
            "revision": "converted:" + head["revision"],
            "protocol": "turn",
            "source_sha": None,
            "cfg": cfg,
        }
    ]
    shots = sorted((r for r in rows if r["ev"] == "cmd"), key=lambda r: r["cmd"]["shot_n"])
    for k, r in enumerate(shots):
        c = r["cmd"]
        ep.append(
            {
                "ev": "cmd",
                "k": k,
                "seat": r["seat"],
                "cmd": {"type": "putt", **{x: c[x] for x in KEYS}, "clock_left_s": c["clock_left_s"]},
            }
        )
    fo = next((r for r in rows if r["ev"] == "forfeit"), None)
    if fo:
        # start_timeout (case nop-B): INTENDED RULE CHANGE (core 0.2.2, `no_show`): a duel side that never starts
        # loses, the present side wins (valid). The legacy service already gave the starter the win (manifest winner
        # A), so the expected result stays the legacy one.
        ep.append({"ev": "forfeit", "k": len(shots), "seat": fo["seat"], "why": WHY.get(fo["why"], fo["why"])})
    m = json.loads((old / "manifest.json").read_text())
    if not fo and not shots and len(seats) == 1 and (m.get("players") or {}).get(seats[0]) == "nop":
        # INTENDED RULE CHANGE (core 0.2.7): a session that ends before the start is a no-show
        ep[0]["tag"] = PRESTART_TAG
        ep.append({"ev": "forfeit", "k": 0, "seat": seats[0], "why": "no_show"})
    ep.append({"ev": "end", "k": len(shots), "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    return {"state_sha": sha_json(m["final_sha"]), "scores": m["scores"], "winner": m["winner"], "k": m["steps"]}
