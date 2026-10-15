"""The legacy archery records of the golden-data store (one directory per case: commands.jsonl.gz, manifest.json)
-> core episodes, for tools/golden.py replay archery --external STORE. The legacy services log each arrow's decision
(elev_deg, azim_deg, draw; already rounded to 4 decimals) and each forfeit, so the conversion is 1:1: a decision
becomes the low-level command {"type": "shoot", ...}. The legacy single scored {total, n_x}; the core scores by side,
so the expected scores are {"A": total}. The duel records (archery-duel-program-standard) are no longer converted:
the duel format was retired (review item B10, 2026-10-07).

Tagged (PRESTART_TAG, in the converted header): the single record nop (the recorded player never started, its session
ended inside the start window): core 0.2.7 decides it at the collect, a valid loss by forfeit (no_show); the converted
episode carries that forfeit row."""
import gzip
import json
from pathlib import Path

TASKS = {"archery-program-standard": "single"}
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the single player never started (recorded player nop) and the "
               "session ended inside the start window (legacy: a valid loss, no winner). Core 0.2.7: a valid loss by "
               "forfeit (no_show, success 0, no winner side) decided at the collect; cores 0.2.2-0.2.6 ended it "
               "unfinished (not a valid trial). The episode carries the forfeit row the core now writes; the "
               "replayed result (state hash, scores, no winner, k = 0) is the same")


def cases(store: Path):
    return [p for task in TASKS for p in sorted((store / task).glob("*/")) if (p / "commands.jsonl.gz").exists()]


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    rows = [json.loads(x) for x in gzip.open(old / "commands.jsonl.gz", "rt") if x.strip()]
    head = rows[0]
    cfg = {"sport": "archery", "format": fmt, "seats": ["A"], "seed": head["seed"],
           "first": head.get("first_player", "A"), "action_level": "low", "play": "program", "params": {}}
    ep = [{"ev": "header", "sport": "archery", "revision": "converted:" + head["revision"], "protocol": "turn",
           "source_sha": None, "cfg": cfg}]
    k = 0
    for r in rows[1:]:
        if r["ev"] == "cmd":
            ep.append({"ev": "cmd", "k": k, "seat": r["seat"], "cmd": {"type": "shoot", **r["cmd"]}})
            k += 1
        elif r["ev"] == "forfeit":
            ep.append({"ev": "forfeit", "k": k, "seat": r["seat"], "why": r["why"]})
    m = json.loads((old / "manifest.json").read_text())
    if fmt == "single" and k == 0 and len(ep) == 1 and (m.get("players") or {}).get("A") == "nop":
        # INTENDED RULE CHANGE (core 0.2.7): a session that ends before the start is a no-show
        ep[0]["tag"] = PRESTART_TAG
        ep.append({"ev": "forfeit", "k": 0, "seat": "A", "why": "no_show"})
    ep.append({"ev": "end", "k": k, "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    return {"state_sha": m["final_sha"], "scores": {"A": m["scores"]["total"]}, "winner": m["winner"], "k": k}
