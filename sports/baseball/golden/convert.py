"""The legacy baseball records of the golden-data store (RO_GOLDEN_DATA; one directory per case: commands.jsonl.gz,
raw/episode.jsonl.gz, manifest.json, trace.json) -> core episodes, for tools/golden.py replay baseball --external STORE.

Both legacy tasks: baseball-batting-wallclock-program-standard (bbrt.RT: seat A against the scripted pitcher; header
level + seed, every swing command with its pitch and effect step) and baseball-duel-wallclock-program-standard
(bbduel.Duel: every pitch and swing command with its pitch, step and seat). The commands are the legacy ones, 1:1, at
their effect steps; the reviews become `resume` rows at the step of each pitch's end (raw episode `pitch_over` rows),
a review forfeit (end reason ready_timeout) a `forfeit` row, a collection before the end an `end` row "collected" (a
started batting game: a forfeit row "abandoned", below).

Intended rule change (core 0.2.2, "no_show", tagged in the episode header): a side that never starts LOSES. Duel: the
legacy service voided that match (reason incomplete, no winner); converted, it is a forfeit of the absent seat and the
present seat wins (expected winner A for the record nop-B). Batting (core 0.2.7, PRESTART_TAG): the legacy service had
no start window (the record nop was collected at step 0, a valid game with 0 bases); cores 0.2.2-0.2.6 would end a
session collected inside the start window incomplete; core 0.2.7 makes it a no_show at the collect (a valid loss), as
converted: the replayed result (state hash, score, no winner) is the same.

Intended rule change (core 0.2.6, "abandoned", with baseball's B2 decision of 2026-10-07; tagged in the header): a
batter who started and whose session ended before the end of the game LOSES (a valid loss, success 0). The legacy
service counted such a game as valid and successful at 4 bases or more (record quit-60s: 4 bases, legacy success 1);
cores 0.2.2-0.2.5 ended it "collected" (incomplete). Converted, it carries the forfeit row "abandoned" the core now
writes; the replayed result (state hash, bases, no winner, steps) is the same.

Expected = the legacy result: the state hash (manifest final_sha = the legacy state_sha()), the scores (batting:
bases -> {"A": bases}), the winner and the control steps.

    python3 golden/convert.py STORE     (in the sim image, from the sport directory) replays every converted case
                                        in-process and also checks the state hash after every pitch against the
                                        legacy trace (trace.json hashes)
"""
import gzip
import json
from pathlib import Path

TASKS = {"baseball-batting-wallclock-program-standard": "batting",
         "baseball-duel-wallclock-program-standard": "duel"}
NO_SHOW_TAG = ("rule-change:core-0.2.2-no_show: the legacy service recorded this start timeout under its own rule "
               "(duel: incomplete, no winner; batting: no start window, collected); under core 0.2.2 the side that "
               "never started loses (duel: the present side wins; batting: a valid loss)")
PRESTART_TAG = ("rule-change:core-0.2.7-prestart-noshow: the batter never started and the session was collected at "
                "step 0 (legacy: no start window, a valid game with 0 bases); cores 0.2.2-0.2.6: incomplete (the "
                "collect came inside the start window); core 0.2.7: a valid loss by forfeit at the collect (no_show, "
                "success 0); the replayed result (state hash, score, no winner) is the same")
ABANDON_TAG = ("rule-change:core-0.2.6-abandoned: the batter started and the session ended the game before its end "
               "(legacy reason timeout: a valid game, success at 4 bases or more, the legacy rule kept by grade.py "
               "until 2026-10-07); under core 0.2.6 a valid loss by forfeit (abandoned) and, with the B2 decision "
               "(success needs a complete game), success 0 whatever the bases; cores 0.2.2-0.2.5: collected, an "
               "incomplete game. The replayed result (state hash, bases, no winner, steps) is the same")


# Retired (revision duel-2, 2026-10-08): the duel records with pitches were played when the n-th pitch to either
# batter had the same windup and scatter; since duel-2 each batter's plate appearances draw their own stream, so the
# windups, pitches and swings differ. The duel nop-B (no pitch) and every batting record still replay bit-exactly.
RETIRED = {("baseball-duel-wallclock-program-standard", "oo-s1-A"),
           ("baseball-duel-wallclock-program-standard", "oo-s1-B"),
           ("baseball-duel-wallclock-program-standard", "quit-B")}


def cases(store: Path):
    return sorted(p for t in TASKS for p in (store / t).glob("*/")
                  if (p / "commands.jsonl.gz").exists() and (t, p.name) not in RETIRED)


def _rows(p: Path):
    return [json.loads(x) for x in gzip.open(p, "rt") if x.strip()]


def convert(old: Path, out: Path) -> dict:
    fmt = TASKS[old.parent.name]
    rows = _rows(old / "commands.jsonl.gz")
    head = rows[0]
    raw = _rows(old / "raw" / "episode.jsonl.gz")
    m = json.loads((old / "manifest.json").read_text())
    end = next(r for r in reversed(raw) if r.get("ev") == "end")
    k_end = int(end["k_end"])
    if fmt == "batting":
        cfg = {"sport": "baseball", "format": "batting", "seats": ["A"], "seed": head["seed"], "level": head["level"],
               "first": "A", "action_level": "low", "play": "program", "params": {}}
        cmds = [(int(r["k"]), {"ev": "cmd", "k": int(r["k"]), "seat": r.get("seat", "A"), "cmd": r["cmd"]})
                for r in rows[1:] if r.get("ev") == "cmd"]
        started = any(r.get("ev") == "game_start" for r in raw)
        joined = {"A": started}
    else:
        cfg = {"sport": "baseball", "format": "duel", "seats": ["A", "B"], "seed": head["seed"],
               "first": head["first_batter"], "action_level": "low", "play": "program", "params": {}}
        cmds = [(int(r["k"]), {"ev": "cmd", "k": int(r["k"]), "seat": r["seat"], "cmd": r["cmd"]})
                for r in rows[1:] if r.get("ev") == "cmd"]
        joined = {s: any(r.get("ev") == "ready" and r.get("seat") == s for r in raw) for s in ("A", "B")}
    hdr = {"ev": "header", "sport": "baseball", "revision": "converted:" + str(head["revision"]),
           "protocol": "realtime", "source_sha": None, "cfg": cfg}
    timed = [(int(r["step"]), 0, {"ev": "resume", "k": int(r["step"])}) for r in raw
             if r.get("ev") == "pitch_over" and int(r["step"]) < k_end]
    timed += [(k, 1, row) for k, row in cmds]
    timed.sort(key=lambda x: (x[0], x[1]))
    ep = [hdr] + [row for _, _, row in timed]
    reason = end.get("reason")
    winner = m.get("winner") if fmt == "duel" else None
    absent = [s for s, j in joined.items() if not j]
    if reason in ("game_over",):
        ep.append({"ev": "end", "k": k_end, "reason": "match_over"})
    elif reason == "ready_timeout" and end.get("forfeiting") in ("A", "B"):
        ep.append({"ev": "forfeit", "k": k_end, "seat": end["forfeiting"], "why": "ready_timeout"})
        ep.append({"ev": "end", "k": k_end, "reason": "ready_timeout"})
    elif k_end == 0 and len(absent) == 1:
        # INTENDED RULE CHANGE (core 0.2.2; batting: core 0.2.7): the side that never started loses (module docstring)
        hdr["tag"] = NO_SHOW_TAG if fmt == "duel" else PRESTART_TAG
        ep.append({"ev": "forfeit", "k": 0, "seat": absent[0], "why": "no_show"})
        ep.append({"ev": "end", "k": 0, "reason": "no_show"})
        if fmt == "duel":
            winner = {"A": "B", "B": "A"}[absent[0]]
    elif fmt == "batting" and started:
        # INTENDED RULE CHANGE (core 0.2.6, and baseball's B2 decision): the batter started and the session ended the
        # game before its end: a valid loss by forfeit (abandoned), success 0 whatever the bases (see the docstring)
        hdr["tag"] = ABANDON_TAG
        ep.append({"ev": "forfeit", "k": k_end, "seat": "A", "why": "abandoned"})
        ep.append({"ev": "end", "k": k_end, "reason": "abandoned"})
    else:                                              # the legacy collect hook ended the game before its end
        ep.append({"ev": "end", "k": k_end, "reason": "collected"})
    (out / "episode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
    sc = m["scores"]
    scores = {"A": int(sc)} if fmt == "batting" else {s: int(v) for s, v in sc.items()}
    return {"state_sha": m["final_sha"], "scores": scores, "winner": winner, "k": int(m["steps"])}


def trace_hashes(old: Path) -> list:
    """The legacy state hash after every judged pitch: [[pitch number, sha], ...] (trace.json)."""
    return json.loads((old / "trace.json").read_text()).get("hashes", [])


def main(argv):
    """Replay every converted case in this process (cwd: a directory holding core/ and sport/ as in the sim) and
    compare the final result and the state hash after every pitch with the legacy trace."""
    import sys
    import tempfile
    sys.path.insert(0, ".")
    from core.episode import load_plugin, read_rows, replay
    from core.sport import UNIT_OVER, FINISHED
    sport = load_plugin("sport/plugin.py")
    bad = 0
    for old in cases(Path(argv[0])):
        with tempfile.TemporaryDirectory() as tmp:
            want = convert(old, Path(tmp))
            rows = read_rows(Path(tmp) / "episode.jsonl")
        seen = []

        def obs(match):
            ph = match.phase().state
            if ph in (UNIT_OVER, FINISHED) and (not seen or seen[-1][0] != match.unit()):
                seen.append((match.unit(), match.state_digest()))
        got = replay(sport, rows, observer=obs)
        hashes = [[n, h] for n, h in trace_hashes(old) if n > 0]
        per_pitch = [[n, h] for n, h in seen]
        ok_final = all(got[k] == want[k] for k in want)
        ok_trace = per_pitch == hashes
        bad += not (ok_final and ok_trace)
        print(json.dumps({"case": f"{old.parent.name}/{old.name}", "final": ok_final, "per_pitch": ok_trace,
                          "pitches": len(hashes), "scores": got["scores"], "winner": got["winner"], "k": got["k"],
                          "tag": rows[0].get("tag", "")[:40]}))
    print(f"baseball legacy records: {'all identical' if not bad else f'{bad} differ'}")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
