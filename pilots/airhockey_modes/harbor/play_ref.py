#!/usr/bin/env python3
"""The tasks' oracle (solution/): plays the game with our references through the match service and leaves the
deliverable in /app/output/skill/. High: the reference Player (player.py: the reference strategy with the platform's
move / strike). Low: the same strategy's move / strike requests turned into joint plans by our low-level reference
implementation (skill.py, low-level tools only). Every 2 control steps it observes and acts; after each rally it
sends its review."""
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = Path("/app/output/skill")
SKILL.mkdir(parents=True, exist_ok=True)
for f in (HERE / "skill").iterdir():
    shutil.copy(f, SKILL / f.name)
sys.path.insert(0, str(SKILL))
sys.path.insert(0, "/opt/robot")
sys.path.insert(0, "/opt/robot/lib")
from client import Game  # noqa: E402

g = Game()
for _ in range(120):
    sp = g.spec()
    if "level" in sp:
        break
    time.sleep(2)
level = sp["level"]
from ref_strategy import RefStrategy  # noqa: E402

strat = RefStrategy()
impl = None
if level == "low":
    from skill import Impl
    impl = Impl(sp)
g.start()
replies, last_k, rally = [], -1, None
log = open("/app/output/oracle.log", "a")
while True:
    o = g.observe()
    st = o.get("state")
    if st == "over":
        break
    if st == "review":
        g.ready("reference player", "same policy")
        continue
    if st != "running" or o.get("k") == last_k:
        time.sleep(0.01)
        continue
    last_k = o["k"]
    view = {"state": o, "replies": replies}
    cmds = strat.tick(o)
    if level == "low":
        out = []
        for c in cmds:
            if c["type"] == "move":
                out = impl.move(view, c["x"], c["y"])
            elif c["type"] == "strike":
                out = impl.strike(view, {k: c[k] for k in ("t", "x", "y", "dx", "dy", "speed")})
        cmds = out if cmds else (impl.step(view) or [])
    if cmds:
        r = g.act(cmds)
        replies = r.get("results", [])
        bad = [x for x in replies if isinstance(x, str)]
        if bad:
            log.write(json.dumps({"k": o["k"], "took_effect": r.get("took_effect"), "refused": bad}) + "\n")
    time.sleep(0.005)
print(json.dumps(g.status()))
