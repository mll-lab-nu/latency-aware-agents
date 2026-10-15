"""Air hockey on the program feed (broadcast.py HockeyMatch / HockeyProgram), offline: fake clock, no render helper.

A recorded log (testdata/broadcast_airhockey_single_oracle_L1_s101.jsonl.gz: the broadcast log of the finished oracle
single ahfix-airhockey-oracle-L1-x20-s101-20261005T141851Z, rebuilt with broadcast/regen_rt.py from its trajectory,
replay hash equal to the live one; every event, the frames of the first 0.6 s and of the 3 s before each rally end):
rallies as replay units (three seven-second fouls, three goals, the time cap), goals and fouls in the bug, the play
clock, the texts ("Goal for ...", foul -> the puck to the other player, full time with the tie-break), the packages
(main 1x -> goal-line camera / puck close-up 0.5x -> overhead 0.5x), the cut at a new rally. Synthetic frames for the
seven-second count and a dead puck's face-off banner; the director (stinger -> package -> back to live on resume);
job names and discovery; captions from the sim's /data."""
import gzip
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("broadcast_airhockey_test", HERE / "broadcast.py")
b = importlib.util.module_from_spec(sp)
sp.loader.exec_module(b)
lb = b.lb
LOG = HERE / "testdata" / "broadcast_airhockey_single_oracle_L1_s101.jsonl.gz"

# ---- job names, discovery, the match class
d = lb.parse_job("ahfix-airhockey-duel-x10-s1-lunaA-lunaB-20261005T141932Z", "airhockey-duel-wallclock-program__p4je2tr")
assert d["sport"] == "airhockey" and d["duel"] and d["players"] == {"A": "luna", "B": "luna"}, d
JOB, PROJ = "ahfix-airhockey-oracle-L1-x20-s101-20261005T141851Z", "airhockey-match-wallclock-progra__tdwufvx"
d = lb.parse_job(JOB, PROJ)
assert d["sport"] == "airhockey" and not d["duel"] and d["level"] == "1" and d["slow"] == "20", d
assert "/data/guichi/rcb-oly-airhockey-fix" in lb.TREES["local"] and "airhockey" in lb.REALTIME
assert ("airhockey", "duel") in b.SUPPORTED and ("airhockey", "single") in b.SUPPORTED
duel = b.make_match(dict(job="ahfix-airhockey-duel-x10-s1-lunaA-lunaB-20261005T141932Z",
                         proj="airhockey-duel-wallclock-program__p4je2tr", host="local", sim="-"))
assert type(duel) is b.HockeyMatch and duel.mode == "duel" and duel.labels() == ("Luna · Blue", "Luna · Orange")

LINES = [json.loads(ln) for ln in gzip.open(LOG, "rt")]


def match(lines):
    bm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
    bm.simdir_ok = True
    for x in lines:
        bm.ingest(json.dumps(x).encode())
    return bm


# ---- the recorded log
bm = match(LINES)
assert type(bm) is b.HockeyMatch and bm.program_cls is b.HockeyProgram and bm.ready() and bm.unit == "rally"
assert bm.labels() == ("Oracle", "Program L1"), bm.labels()
assert [p["kind"] for p in bm.points] == ["seven_second_foul"] * 3 + ["goal"] * 3 + ["time_cap"], bm.points
assert [p["k"] for p in bm.points] == [1750, 5364, 7542, 7715, 9526, 10821, 12000]
assert bm.points[0]["how"] == "Foul · seven-second rule · Oracle kept the puck in its half · puck to Program L1", bm.points[0]
assert bm.points[3]["how"] == "Goal for Oracle · 1–0" and bm.points[5]["how"] == "Goal for Oracle · 3–0", bm.points[3:6]
assert bm.points[-1]["final"] and bm.points[-1]["how"] == "Time cap · 2:00 of play · Full time 3–0 · Oracle wins", bm.points[-1]
assert bm.score_at(7714) == (0, 0) and bm.score_at(7715) == (1, 0) and bm.score_at(bm.ks[-1]) == (3, 0)
assert bm.fouls_at(12000, "A") == 1 and bm.fouls_at(12000, "B") == 2
assert bm.phase_at(bm.ks[-1]) == "end" and all(len(r["q"]) == 21 for r in bm.recs)
assert all(r.get("cut") for r in bm.recs if r["k"] in (1752, 5366, 7544, 7717)), "a new rally is a cut"
pr = b.HockeyProgram(bm)
sport, unit, rows, phase = pr.bug(0, 7715)
assert sport == "Air hockey · Single" and unit == "Rally 4 · 1:17 / 2:00", unit
assert rows == [["A", "Oracle", 1, "1 foul"], ["B", "Program L1", 0, "2 fouls"]], rows
assert pr.bug(0, 7800)[1] == "Rally 5 · 1:18 / 2:00"
assert pr.live_banner(7720) == "Rally 5 · Program L1 serves (scored upon)", pr.live_banner(7720)
assert pr.live_banner(1760) == "Rally 2 · possession change · puck to Program L1", pr.live_banner(1760)
assert pr.live_banner(1750 + 400) is None
goal = b.hockey_build_package(bm, bm.points[3])
assert [s["speed"] for s in goal] == [1.0, 0.5, 0.5] and goal[0]["cam"] == "main" and goal[2]["cam"] == "high", goal
assert goal[1]["label"] == "Goal-line camera · 0.5×" and goal[1]["cam"][0][0] > 0.5, goal   # A scores at +x
foul = b.hockey_build_package(bm, bm.points[0])
assert foul[1]["label"] == "Puck close-up · 0.5×" and foul[1]["cam"][0][0] < 0, foul           # A held it at -x
assert [s["label"] for s in b.hockey_build_package(bm, bm.points[-1])] == ["Main camera · real speed"]
assert all(p["k0"] <= s["k0"] < s["k1"] <= p["k"] for p in bm.points for s in b.hockey_build_package(bm, p))
info = bm.info_json()
assert info["unit"] == "rally" and info["points"][3]["label"] == "Rally 4" and info["points"][3]["winner"] == "Oracle"

# ---- synthetic: the seven-second count; a dead puck, the face-off; the end tie-break texts
HDR = dict(ev="hdr", every=2, sport="airhockey", mode="duel", slowdown=10.0, dt=0.01, level=1, seed=1, nq=21, first="A")


def q(x, y=0.0):
    return [x, y, 0.0] + [0.0] * 18


syn = [HDR, dict(ev="run", k=0, score={"A": 0, "B": 0}, rally=1)]
syn += [dict(k=k, q=q(-0.5 + k * 0.0004)) for k in range(0, 601, 2)]             # in A's half for 6 s
syn += [dict(k=k, q=q(0.0)) for k in range(602, 1302, 2)]                        # then resting on the centre line
syn += [dict(ev="stoppage", k=1300, score={"A": 0, "B": 0}, how="dead_puck", winner=None, guilty=None, next="faceoff",
             rally=1), dict(ev="pause", k=1300, score={"A": 0, "B": 0}, rally=1),
        dict(ev="run", k=1300, score={"A": 0, "B": 0}, rally=2)]
syn += [dict(k=k, q=q(0.0)) for k in range(1302, 1500, 2)]
sm = b.make_match(dict(job="ahfix-airhockey-duel-x10-s1-lunaA-lunaB-20261005T141932Z",
                       proj="airhockey-duel-wallclock-program__p4je2tr", host="local", sim="-"))
sm.simdir_ok = True
for x in syn:
    sm.ingest(json.dumps(x).encode())
sp_ = b.HockeyProgram(sm)
assert sp_.bug(0, 500)[3] == "Seven-second count · Luna · Blue · 5.0 s", sp_.bug(0, 500)
assert sp_.bug(0, 150)[3] == ""
assert sm.points[0]["how"] == "Dead puck · face-off at the centre" and sm.rec_at(1302).get("cut")
assert sp_.live_banner(1400) == "Rally 2 · face-off at the centre", sp_.live_banner(1400)
tm = match([x for x in LINES if x.get("k", 0) <= 7542 and x.get("ev") != "end"] +
           [dict(ev="stoppage", k=7600, score={"A": 0, "B": 0}, how="time_cap", winner=None, guilty=None,
                 next="faceoff", rally=4), dict(ev="end", k=7600, score={"A": 0, "B": 0}, reason="match_over",
                                                  forfeiting=None, winner="B")])
assert tm.points[-1]["how"].endswith("Full time 0–0 · Program L1 wins on territory (time-cap tie-break)"), tm.points[-1]

# ---- the director on a fake clock: a goal freezes the world -> stinger -> package -> back to live on resume
fm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
fm.simdir_ok = True
upto = [x for x in LINES if x.get("k", 0) <= 7715 and not (x.get("ev") == "run" and x["k"] == 7715)]
for x in upto:
    fm.ingest(json.dumps(x).encode())
T = [100.0]
fp = b.HockeyProgram(fm)
fp.clock = lambda: T[0]
fp.last_t = T[0]
seen = []


def run(seconds, dt=0.05):
    end = T[0] + seconds
    while T[0] < end:
        fp.live_step(T[0])
        fp.direct(T[0])
        ra, rb, alpha, ov, key = fp.compose(T[0])
        seen.append((round(T[0], 2), fp.mode, ov.get("stinger"), ov.get("shot"), ov.get("banner"), ov["bug"]))
        T[0] += dt


run(6.0)
modes = [s[1] for s in seen]
assert "replay" in modes and any(s[2] is not None for s in seen[:modes.index("replay")]), seen[-3:]
run(25.0)
shots = {s[3] for s in seen if s[1] == "replay" and s[2] is None}
assert {x["label"] for x in goal} <= shots, shots
assert any(s[4] == "Goal for Oracle · 1–0" for s in seen if s[1] == "replay")
assert seen[-1][5]["rows"][0][2] == 1 and seen[-1][5]["sport"] == "Air hockey · Single", seen[-1][5]
for x in [x for x in LINES if 7715 <= x.get("k", 0) <= 7800 and x not in upto]:
    fm.ingest(json.dumps(x).encode())
run(1.5)
assert seen[-1][1] == "live" and any("back to live" in x for _, x in fp.log), seen[-1][:3]

# ---- captions from the sim's /data
out = ('@@EP 0 999\n' + json.dumps(dict(event="review", step=3000, wall=time.time() - 4, seat="A",
                                        why="B kept the puck in its half for seven seconds", plan="attack the serve")) +
       '\n@@TC think_calls_B.jsonl\n' + json.dumps(dict(seq=3, t1=time.time() - 1, excerpt="I will guard the goal line.")) +
       '\n').encode()
b.lb.dexec = lambda *a, **k: out
caps = b.RTCaptions(duel)
caps.update(time.time())
low = caps.lower(time.time(), b.HockeyProgram(duel).seats())
assert [(c["seat"], c["kind"]) for c in low] == [("A", "Review"), ("B", "Says")], low
assert low[0]["text"] == "B kept the puck in its half for seven seconds → attack the serve", low
print("PASS: air hockey job names / discovery; recorded oracle single: fouls, goals, time cap; goals / fouls / play "
      "clock in the bug; goal / foul / full-time texts; package main 1x -> goal-line camera or puck close-up 0.5x -> "
      "overhead 0.5x; cut at a new rally; seven-second count; face-off and possession banners; director stinger -> "
      "package -> live on resume; captions from /data")
