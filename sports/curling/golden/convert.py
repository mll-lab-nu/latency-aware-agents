"""The legacy curling records of the golden-data store -> core episodes, for tools/golden.py replay curling --external
STORE (or RO_GOLDEN_DATA). One directory per case: commands.jsonl.gz (header + the throws), manifest.json, raw/.

Both legacy tasks: curling-match-program-standard (one seat against the scripted team; the opponent's throws are not
commands, the world plays them) and curling-duel-program-standard. The conversion is 1:1 per throw ({v, angle, spin}
-> {speed, angle, spin}); a clock forfeit, which the legacy commands file does not hold, comes from the end row of the
raw episode (raw/episode.jsonl.gz). A duel seat that never started loses (core 0.2.2, `no_show`: an intended rule
change, the legacy manifest has no winner there); a single player who never started and whose session ended inside the
start window (the record nop): a valid loss by forfeit (no_show) at the collect since core 0.2.7 (cores 0.2.2-0.2.6:
an incomplete match), tagged PRESTART_TAG in the converted header.

Expected = the legacy result: the state hash (manifest final_sha = the legacy simulator's state_sha()), the scores
(match: {"you", "opponent"} -> {"A", "opponent"}), the winner of the live match (the duel's clock forfeit or no-show: the other
seat; match: "player" -> "A", a forfeit or an unfinished match -> none) and the number of throws."""
import gzip
import json
from pathlib import Path

TASKS = ("curling-match-program-standard", "curling-duel-program-standard")
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the single player never started (recorded player nop) and "
                "the session ended inside the start window (legacy: a valid loss, no winner). Core 0.2.7: a valid loss "
                "by forfeit (no_show, success 0, no winner side) decided at the collect; cores 0.2.2-0.2.6 ended it "
                "incomplete (not a valid trial). The episode carries the forfeit row the core now writes; the replayed "
                "result (state hash, scores, no winner, k = 0) is the same")


# Retired (revision curling-2, 2026-10-08): the records with throws were played when both teams' n-th stones drew the
# same execution noise; since curling-2 each team draws its own stream, so their stones end elsewhere. The records
# without a throw (nop, nop-B) still replay bit-exactly.
RETIRED = {("curling-duel-program-standard", "clock-short"), ("curling-duel-program-standard", "oo-s1-A"),
           ("curling-duel-program-standard", "oo-s1-B"), ("curling-match-program-standard", "clock-short"),
           ("curling-match-program-standard", "o-L1")}


def cases(store: Path):
    return sorted(p for t in TASKS for p in (store / t).glob("*/")
                  if (p / "commands.jsonl.gz").exists() and (t, p.name) not in RETIRED)


def _end_row(old: Path) -> dict:
    raw = old / "raw" / "episode.jsonl.gz"
    if not raw.exists():
        return {}
    rows = [json.loads(x) for x in gzip.open(raw, "rt") if x.strip()]
    return next((r for r in reversed(rows) if r.get("ev") == "end"), {})


def convert(old: Path, out: Path) -> dict:
    rows = [json.loads(x) for x in gzip.open(old / "commands.jsonl.gz", "rt") if x.strip()]
    head, cmds = rows[0], [r for r in rows[1:] if r.get("ev") == "cmd"]
    duel = "first_player" in head
    if duel:
        cfg = {"sport": "curling", "format": "duel", "seats": ["A", "B"], "seed": head["seed"],
               "first": head["first_player"], "action_level": "low", "play": "program", "params": {}}
    else:
        if not head.get("followed", True):
            raise ValueError("a match that ended during a review (opponent's reply not played) is not convertible")
        cfg = {"sport": "curling", "format": "match", "seats": ["A"], "seed": head["seed"], "level": head["level"],
               "first": "A", "action_level": "low", "play": "program", "params": {}}
    if head.get("tiebreak"):
        raise ValueError("tie after the last extra end: the legacy clocks are not in the commands file")
    ep = [{"ev": "header", "sport": "curling", "revision": "converted:" + head["revision"], "protocol": "turn",
           "source_sha": None, "cfg": cfg}]
    for k, r in enumerate(cmds):
        c = r["cmd"]
        ep.append({"ev": "cmd", "k": k, "seat": r.get("seat", "A"),
                   "cmd": {"type": "throw", "speed": c["v"], "angle": c["angle"], "spin": c["spin"]}})
    k = len(cmds)
    end = _end_row(old)
    m = json.loads((old / "manifest.json").read_text())
    winner = None
    if end.get("reason") == "clock_expired":
        seat = end.get("forfeiting") or "A"
        ep.append({"ev": "forfeit", "k": k, "seat": seat, "why": "clock"})
        ep.append({"ev": "end", "k": k, "reason": "clock"})
        winner = {"A": "B", "B": "A"}[seat] if duel else None
    elif end.get("reason") == "start_timeout" and duel and end.get("forfeiting") in ("A", "B"):
        # INTENDED RULE CHANGE (core 0.2.2): a duel side that never starts loses (valid, reason no_show). The legacy
        # service also gave the present seat the win (reward A_won = 1), but its manifest/replay winner is null.
        seat = end["forfeiting"]
        ep.append({"ev": "forfeit", "k": k, "seat": seat, "why": "no_show"})
        ep.append({"ev": "end", "k": k, "reason": "no_show"})
        winner = {"A": "B", "B": "A"}[seat]
    elif end.get("reason") == "match_over" or (not end and k):
        ep.append({"ev": "end", "k": k, "reason": "match_over"})
    elif not duel and not k and (m.get("players") or {}).get("A") == "nop":
        # INTENDED RULE CHANGE (core 0.2.7): a session that ends before the start is a no-show
        ep[0]["tag"] = PRESTART_TAG
        ep.append({"ev": "forfeit", "k": k, "seat": "A", "why": "no_show"})
        ep.append({"ev": "end", "k": k, "reason": "no_show"})
    else:
        ep.append({"ev": "end", "k": k, "reason": "incomplete"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    if winner is None and m.get("winner"):
        winner = "A" if m["winner"] == "player" else m["winner"]
    sc = m["scores"]
    scores = dict(sc) if duel else {"A": sc["you"], "opponent": sc["opponent"]}
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": k}
