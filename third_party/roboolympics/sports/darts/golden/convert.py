"""The legacy darts records of the golden-data store (RO_GOLDEN_DATA; one directory per case: commands.jsonl.gz,
manifest.json) -> core episodes, for tools/golden.py replay darts. Both legacy tasks: darts-501-program-standard
(darts.Match: the player against the scripted opponent; header level + seed, the player's throws, a clock forfeit)
and darts-duel-program-standard (darts.Duel: each throw with both chess clocks, forfeits). The conversion is 1:1;
the expected result is the legacy one in the core's terms: format 501 scores {A, opponent}, its winner "A" or
"opponent", and no winner after a forfeit (core/outcome.py: a single player who forfeits loses to the line).

Tagged (PRESTART_TAG, in the converted header): the 501 record nop (the recorded player never started, its session
ended inside the start window): core 0.2.7 decides it at the collect, a valid loss by forfeit (no_show). The record
replays as recorded (no forfeit row): the plugin's forfeit hook records a single player's forfeit in the world (done,
winner, reason), so the state hash of a live 0.2.7 match differs from the legacy one, which had no forfeit."""
import gzip
import json
from pathlib import Path

TASKS = {"darts-501-program-standard": "501", "darts-duel-program-standard": "duel"}
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the single player never started (recorded player nop) and the "
               "session ended inside the start window (legacy: a valid loss, no winner). Core 0.2.7: a valid loss by "
               "forfeit (no_show, success 0, no winner side) decided at the collect; cores 0.2.2-0.2.6 ended it "
               "unfinished (not a valid trial). Recorded without a forfeit row: darts records a single player's "
               "forfeit in the world, so a live 0.2.7 match ends with another state hash")


# Retired (revision darts-5, 2026-10-08): the duel records with throws were played with common random numbers (the
# n-th dart of either player scattered alike); since darts-5 each player draws its own stream, so their darts land
# elsewhere and they cannot replay. The duel nop-B (no throw) and every 501 record still replay bit-exactly.
RETIRED = {("darts-duel-program-standard", "clock-6s"), ("darts-duel-program-standard", "oo-s1-A"),
           ("darts-duel-program-standard", "oo-s1-B")}


def cases(store: Path):
    return sorted(p for t in TASKS for p in (store / t).glob("*/")
                  if (p / "commands.jsonl.gz").exists() and (t, p.name) not in RETIRED)


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    rows = [json.loads(x) for x in gzip.open(old / "commands.jsonl.gz", "rt") if x.strip()]
    head = rows[0]
    if fmt == "501":
        cfg = {"sport": "darts", "format": "501", "seats": ["A"], "seed": head["seed"], "first": "A",
               "level": head["level"], "action_level": "low", "play": "program", "params": {}}
    else:
        cfg = {"sport": "darts", "format": "duel", "seats": ["A", "B"], "seed": head["seed"],
               "first": head["first_player"], "action_level": "low", "play": "program", "params": {}}
    ep = [{"ev": "header", "sport": "darts", "revision": "converted:" + head["revision"], "protocol": "turn",
           "source_sha": None, "cfg": cfg}]
    k, forfeit = 0, False
    for r in rows[1:]:
        if r["ev"] == "cmd":
            ep.append({"ev": "cmd", "k": k, "seat": r["seat"], "cmd": r["cmd"]})
            k += 1
        elif r["ev"] == "forfeit":
            ep.append({"ev": "forfeit", "k": k, "seat": r["seat"], "why": r["why"]})
            forfeit = True
    m = json.loads((old / "manifest.json").read_text())
    if fmt == "501" and k == 0 and not forfeit and (m.get("players") or {}).get("A") == "nop":
        # INTENDED RULE CHANGE (core 0.2.7): a session that ends before the start is a no-show; replayed as recorded
        ep[0]["tag"] = PRESTART_TAG
    ep.append({"ev": "end", "k": k, "reason": "converted"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    if fmt == "501":
        scores = {"A": m["scores"]["you"], "opponent": m["scores"]["opponent"]}
        winner = None if forfeit else {"player": "A", "opponent": "opponent"}.get(m["winner"])
        return {"state_sha": m["final_sha"], "scores": scores, "winner": winner}
    return {"state_sha": m["final_sha"], "scores": m["scores"], "winner": m["winner"]}
