"""final.json compatibility of the v0.2 core with the Season-1 readers (core/service.py COMPAT): a darts duel played on
the v0.2 core (sports/darts, driven with the throws of a golden match through the turn service's
own endpoints) writes a final.json that live_board summarizes like a legacy darts duel, and that carries every common
field regrade_transient.py and run_round.py read. Needs the darts sim assembled by tools/sportlib.py:

    python3 scripts/olympiad/test_final_compat.py SIM_DIR      (in the sim image; SIM_DIR holds core/ and sport/)

e.g. SIM_DIR = tasks/darts-duel-program-standard/environment/sim. It renders, so it needs a GL backend (osmesa in the
sim image, or MUJOCO_GL=egl on a GPU host) and Pillow; not run by CI.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(SIM), str(HERE)]
os.environ.update(PLAYER_MODE_A="oracle", PLAYER_MODE_B="oracle", RCB_BROADCAST="0")

from core.episode import load_plugin  # noqa: E402
from core.sport import MatchConfig  # noqa: E402
from core.turn import TurnService  # noqa: E402

PLUGIN = str(SIM / "sport" / "plugin.py")
GOLDEN = HERE.parents[1] / "sports" / "darts" / "golden" / "duel-s1-A" / "episode.jsonl"   # stored uncompressed


def play_v02_darts(data: Path) -> dict:
    rows = [json.loads(x) for x in open(GOLDEN) if x.strip()]
    cfg = MatchConfig.from_json(rows[0]["cfg"])
    svc = TurnService(load_plugin(PLUGIN), PLUGIN, cfg, data)
    post = svc.routes(True)
    post["start"]("A", {})
    post["start"]("B", {})
    for r in rows:
        if r["ev"] != "cmd":
            continue
        c = r["cmd"]
        code, out = post["act"](r["seat"], {"type": "throw", "yaw": c["yaw"], "pitch": c["pitch"], "speed": c["speed"]})
        assert code == 200, out
        if svc.phase != "over":
            assert post["ready"](r["seat"], {"why": "w", "plan": "p"})[0] == 200
    t = time.time()
    while not svc.finalized and time.time() - t < 300:
        time.sleep(0.2)
    return json.loads((data / "final.json").read_text())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        F = play_v02_darts(Path(tmp))
    bad = []
    if F.get("compat", {}).get("schema") != "v0.1-toplevel-aliases":
        bad.append("no compat block")
    for k in ("legs", "remaining", "score", "winner", "match_winner", "valid_trial", "technical_failure", "reason",
              "forfeiting", "proxy_error", "transient_api_errors", "unrecovered_api_errors", "clock_used_s", "clock_s"):
        if k not in F:
            bad.append(f"missing {k}")
    import live_board as LB
    m = type("M", (), {})()
    m.final, m.last, m.clock_left = F, {}, {}
    m.info = {"sport": "darts", "duel": True, "players": {"A": "oracle", "B": "oracle"}}
    s = LB.summarize(m)
    if s["main"] != [F["legs"]["A"], F["legs"]["B"]] or s["unit"] != "局" or "剩余" not in s["clocks"]:
        bad.append(f"live_board summary {s}")
    print(json.dumps({"legs": F.get("legs"), "winner": F.get("winner"), "valid": F.get("valid_trial"),
                      "lifted": F.get("compat", {}).get("lifted"), "live_board": s}, ensure_ascii=False)[:700])
    print("FAIL " + "; ".join(bad) if bad else "PASS: v0.2 final.json read by live_board and the regrade fields")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
