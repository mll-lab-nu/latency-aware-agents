"""The legacy billiards records of the golden-data store -> core episodes, for tools/golden.py replay billiards
--external STORE (or RO_GOLDEN_DATA). One directory per case: commands.jsonl.gz (header + the shots), manifest.json,
raw/.

Both legacy tasks: billiards-eightball-program-standard (one seat against the scripted opponent; the opponent's shots
are not commands, the world plays them) and billiards-duel-program-standard. The conversion is 1:1 per shot
({aim_deg, speed, spin_x, spin_y, [cue_xy], [call]} -> {aim_deg, speed, spin_x, spin_y, [cue_x, cue_y], [call]}, the
numbers unchanged); a clock forfeit or a start forfeit, which the legacy commands file does not hold, comes from the
end row of the raw episode (raw/episode.jsonl.gz).

INTENDED RULE CHANGE (core 0.2.2, tagged below): a duel seat that never starts loses (valid, reason no_show, the
present seat wins). The legacy duel service also gave the present seat the win (reward A_won = 1), but its manifest /
replay winner is null; the converted record expects the present seat as winner. INTENDED RULE CHANGE (core 0.2.7,
PRESTART_TAG in the converted header): a single player who never started and whose session ended inside the start
window (the record nop) loses at the collect, a valid loss by forfeit (no_show, no winner, as in the legacy record);
cores 0.2.2-0.2.6 left it an incomplete match.

Expected = the legacy result: the state hash (manifest final_sha = the legacy simulator's state_sha()), the scores
(balls left in each group, 7 while open: duel = the manifest scores; eightball = raw/final.json your_balls_left /
opponent_balls_left as {"A", "opponent"}, since the store's manifest lists the shots taken there), the winner of the
live match (duel: a clock forfeit or no-show -> the other seat; eightball: 0 -> "A", 1 -> "opponent", a forfeit or an
unfinished match -> none) and the number of shots the players sent."""
import gzip
import json
from pathlib import Path

TASKS = ("billiards-eightball-program-standard", "billiards-duel-program-standard")
OTHER = {"A": "B", "B": "A"}
INTENDED = "core-0.2.2-no-show"          # the tag of the intended rule change (a duel side that never starts loses)
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the single player never started (recorded player nop) and "
                "the session ended inside the start window (legacy: a valid loss, no winner). Core 0.2.7: a valid loss "
                "by forfeit (no_show, success 0, no winner side) decided at the collect; cores 0.2.2-0.2.6 ended it "
                "incomplete (not a valid trial). The episode carries the forfeit row the core now writes; the replayed "
                "result (state hash, scores, no winner, k = 0) is the same")
# Legacy records superseded by a later rule change: no longer replayed (they would test the old rule), kept in the
# store as they are. billiards-2 (2026-10-07, review item B7): a coin toss drawn from the seed decides who breaks in
# the duel (deviation D9); the legacy duel let FIRST_PLAYER break. All four legacy duel records are seed 1, first A,
# whose toss falls 1 (B breaks), so their first shot is now out of turn (nop-B: no shot, but the state hash holds the
# breaker). The rule's replacement record is golden/duel-s1-A (seed 1, first A, re-recorded under billiards-2).
RETIRED = {f"billiards-duel-program-standard/{c}": "billiards-2: the seed's coin toss decides who breaks (seed 1: B)"
           for c in ("clock-1s", "nop-B", "oo-s1-A", "oo-s1-B")}


def cases(store: Path):
    return sorted(p for t in TASKS for p in (store / t).glob("*/") if (p / "commands.jsonl.gz").exists()
                  and f"{t}/{p.name}" not in RETIRED)


def _raw(old: Path, name: str):
    p = old / "raw" / name
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        return f.read()


def _end_row(old: Path) -> dict:
    raw = _raw(old, "episode.jsonl.gz")
    rows = [json.loads(x) for x in (raw or "").splitlines() if x.strip()]
    return next((r for r in reversed(rows) if r.get("ev") == "end"), {})


def command(c: dict) -> dict:
    out = {"type": "shoot", "aim_deg": c["aim_deg"], "speed": c["speed"], "spin_x": c.get("spin_x", 0.0),
           "spin_y": c.get("spin_y", 0.0)}
    if c.get("cue_xy") is not None:
        out["cue_x"], out["cue_y"] = c["cue_xy"][0], c["cue_xy"][1]
    if c.get("call") is not None:
        out["call"] = c["call"]
    return out


def convert(old: Path, out: Path) -> dict:
    rows = [json.loads(x) for x in gzip.open(old / "commands.jsonl.gz", "rt") if x.strip()]
    head, cmds = rows[0], [r for r in rows[1:] if r.get("ev") == "cmd"]
    duel = "first_player" in head
    if duel:
        cfg = {"sport": "billiards", "format": "duel", "seats": ["A", "B"], "seed": head["seed"],
               "first": head["first_player"], "action_level": "low", "play": "program", "params": {}}
    else:
        fin = json.loads(_raw(old, "final.json.gz") or "{}")
        if not head.get("followed", True) and not fin.get("game_over"):
            # the last shot was not followed by the opponent's turn: fine when it ended the rack (nothing to follow),
            # not convertible when the rack ended during a review (core: the opponent replies inside the action)
            raise ValueError("a rack that ended during a review (the opponent's reply not played) is not convertible")
        cfg = {"sport": "billiards", "format": "eightball", "seats": ["A"], "seed": head["seed"],
               "level": head["level"], "first": "A", "action_level": "low", "play": "program", "params": {}}
    if head.get("tiebreak"):
        raise ValueError("tie at the shot limit: the legacy clocks are not in the commands file")
    ep = [{"ev": "header", "sport": "billiards", "revision": "converted:" + head["revision"], "protocol": "turn",
           "source_sha": None, "cfg": cfg}]
    for k, r in enumerate(cmds):
        ep.append({"ev": "cmd", "k": k, "seat": r.get("seat", "A"), "cmd": command(r["cmd"])})
    k = len(cmds)
    end = _end_row(old)
    m = json.loads((old / "manifest.json").read_text())
    winner, tags = None, []
    if end.get("reason") == "clock_expired":
        seat = end.get("forfeiting") or "A"
        ep.append({"ev": "forfeit", "k": k, "seat": seat, "why": "clock"})
        ep.append({"ev": "end", "k": k, "reason": "clock"})
        winner = OTHER[seat] if duel else None
    elif end.get("reason") == "start_timeout" and duel and end.get("forfeiting") in ("A", "B"):
        # INTENDED RULE CHANGE (core 0.2.2): a duel side that never starts loses (valid, reason no_show).
        seat = end["forfeiting"]
        ep.append({"ev": "forfeit", "k": k, "seat": seat, "why": "no_show"})
        ep.append({"ev": "end", "k": k, "reason": "no_show"})
        winner, tags = OTHER[seat], [INTENDED]
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
    if duel:
        scores = dict(m["scores"])
        if winner is None and m.get("winner") in ("A", "B"):
            winner = m["winner"]
    else:
        scores = {"A": int(fin.get("your_balls_left", 7)), "opponent": int(fin.get("opponent_balls_left", 7))}
        if winner is None and m.get("winner") in (0, 1):
            winner = "A" if m["winner"] == 0 else "opponent"
    if tags:
        (out / "intended_change.txt").write_text(f"{INTENDED}: legacy manifest winner {m.get('winner')!r}, "
                                                 f"expected {winner!r} (core 0.2.2: a duel side that never starts "
                                                 "loses)\n")
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": k}
