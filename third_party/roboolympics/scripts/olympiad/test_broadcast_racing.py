"""3D racing on the program feed (broadcast.py RacingMatch / RacingProgram), offline: fake clock, no render helper.

A recorded log (testdata/broadcast_racing_duel_oracle_s1.jsonl.gz: the broadcast log of the finished oracle duel
racefix-racing-duel-x10-s1-oracleA-oracleB-20261005T135617Z, rebuilt with broadcast/regen_rt.py from its trajectory,
replay hash equal to the live one) for the parser: heats as replay units, heats won, checkpoints, the race order and the
chase camera, the cut between heats, the heat package (main 1x -> finish line 0.5x -> overhead 0.5x). No finished run
had a crash, so a crash and a disqualification are added to a copy of the log in the sim's own event format (race.py
Race.referee_contacts): the crash banner and warning in the bug, the instant replay while the race goes on (main 1x ->
trackside 0.5x -> back to live, the world never frozen), the disqualification banner. Job names and discovery, the
director's heat-end package and the cut back to live, captions from the sim's /data (reviews, model output)."""
import gzip
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("broadcast_racing_test", HERE / "broadcast.py")
b = importlib.util.module_from_spec(sp)
sp.loader.exec_module(b)
lb = b.lb
LOG = HERE / "testdata" / "broadcast_racing_duel_oracle_s1.jsonl.gz"

# ---- job names, discovery, the match class
JOB = "racefix-racing-duel-x10-s1-oracleA-oracleB-20261005T135617Z"
PROJ = "racing-duel-wallclock-program-st__3uxqads"
d = lb.parse_job(JOB, PROJ)
assert d["sport"] == "racing" and d["duel"] and d["slow"] == "10" and d["players"] == {"A": "oracle", "B": "oracle"}, d
d = lb.parse_job("racefix-racing-luna-L1-x10-s101-20261005T135706Z", "racing-match-wallclock-program-s__syduopl")
assert d["sport"] == "racing" and not d["duel"] and d["level"] == "1" and d["players"] == {"A": "luna"}, d
assert "/data/guichi/rcb-oly-racing-fix" in lb.TREES["local"] and "racing" in lb.REALTIME and "racing" in lb.NAMES
assert ("racing", "duel") in b.SUPPORTED and ("racing", "single") in b.SUPPORTED
single = b.make_match(dict(job="racefix-racing-luna-L1-x10-s101-20261005T135706Z",
                           proj="racing-match-wallclock-program-s__syduopl", host="local", sim="-"))
assert type(single) is b.RacingMatch and single.labels() == ("Luna", "Program L1"), single.labels()

LINES = [json.loads(ln) for ln in gzip.open(LOG, "rt")]


def match(lines):
    bm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
    bm.simdir_ok = True
    for x in lines:
        bm.ingest(json.dumps(x).encode())
    return bm


# ---- the recorded log
bm = match(LINES)
assert type(bm) is b.RacingMatch and bm.mode == "duel" and bm.program_cls is b.RacingProgram
assert bm.scene_opts() == {"seed": 1} and bm.ready()
assert [p["label"] for p in bm.points] == ["Heat 1", "Heat 2", "Heat 3"], bm.points
assert [p["winner"] for p in bm.points] == ["B", "A", "B"]
assert bm.points[0]["how"].startswith("Heat 1 · Oracle · Orange wins — first across the line · 13.09 s · "
                                      "Oracle · Blue 14/16 checkpoints"), bm.points[0]["how"]
assert bm.points[-1]["final"] and bm.points[-1]["how"].endswith("Oracle · Orange wins the race 2–1"), bm.points[-1]
assert bm.score_at(bm.ks[-1]) == (1, 2) and bm.score_at(1309) == (0, 0) and bm.score_at(1310) == (0, 1)
assert bm.phase_at(bm.ks[-1]) == "end" and bm.unit == "heat"
assert sum(1 for r in bm.recs if r.get("cut")) == 2                           # the grids of heats 2 and 3
assert all(len(r["q"]) == 26 and "look" in r and r["lead"] in "AB" for r in bm.recs)
r = bm.rec_at(1200)
assert r["lead"] == "B" and 0 < r["gap"] < 8 and r["prog"]["B"] > r["prog"]["A"], r
assert bm.cp_at(1200, 1, "B") == 14 and bm.cp_at(1310, 1, "B") == 16 and bm.cp_at(1400, 2, "B") <= 2
assert abs(bm.heat_clock(1310) - 13.1) < 1e-6 and bm.heat_at(1400) == 2
for p in bm.points:
    pkg = b.race_build_package(bm, p)
    assert [s["speed"] for s in pkg] == [1.0, 0.5, 0.5] and pkg[0]["cam"] == "main" and pkg[2]["cam"] == "high", pkg
    assert pkg[1]["label"] == "Finish line · 0.5×" and isinstance(pkg[1]["cam"], tuple), pkg
    assert all(p["k0"] < s["k0"] < s["k1"] <= p["k"] for s in pkg), pkg
pr = b.RacingProgram(bm)
sport, unit, rows, phase = pr.bug(0, 1200)
assert sport == "Racing · Duel" and unit == "Heat 1 · 12.0 s", unit
assert rows == [["A", "Oracle · Blue", 0, "CP 12/16"], ["B", "Oracle · Orange", 0, "CP 14/16"]], rows
assert pr.bug(0, 1310)[2][1][3] == "Finished"
assert pr.live_banner(20) == "Heat 1 · go! · Oracle · Orange starts on the inside", pr.live_banner(20)
info = bm.info_json()
assert info["unit"] == "heat" and [p["label"] for p in info["points"]] == ["Heat 1", "Heat 2", "Heat 3"], info["points"]

# ---- a crash and a disqualification (the sim's event format) in a copy of the log
CK = 600


def crash(k, fault, warning, heat=1, won=None):
    return dict(ev="crash", k=k, heat=heat, heats_won=won or {"A": 0, "B": 0}, t=round((k % 1310) * 0.01, 3),
                at_fault=fault, fault_rule="drove_into", closing={"A": 1.2, "B": 0.1}, impulse=0.71, contact_s=0.02,
                kind="impact", warning=warning)


lines = []
for x in LINES:
    if x.get("k") == CK and "q" in x and "ev" not in x:
        lines.append(crash(CK, "A", 1))
    lines.append(x)
cm = match(lines)
cp = next(p for p in cm.points if p["kind"] == "crash")
assert cp["label"] == "Crash 1" and cp["seat"] == "A" and cp["k"] == CK, cp
assert cp["how"] == "Crash · Oracle · Blue at fault (drove into Oracle · Orange) · warning 1 of 3 · placed 2 m behind", cp
assert cm.rec_at(CK).get("cut") and cm.warnings_at(CK, "A") == 1 and cm.warnings_at(CK - 1, "A") == 0
cpr = b.RacingProgram(cm)
assert cpr.bug(0, CK + 50)[2][0][3].endswith(" · 1 warning"), cpr.bug(0, CK + 50)
assert cpr.live_banner(CK + 100) == cp["how"] and cpr.live_banner(CK + 400) is None
i = cm.ks.index(CK)
assert cpr.record_at(CK - 2)[2] is None                                       # no interpolation into the placement
live_pkg = b.race_build_package(cm, cp, live=True)
assert [(s["cam"] == "main", s["speed"]) for s in live_pkg] == [(True, 1.0), (False, 0.5)], live_pkg
assert live_pkg[1]["label"] == "Trackside · 0.5×" and live_pkg[1]["k0"] < CK < live_pkg[1]["k1"], live_pkg
assert [s["label"] for s in b.race_build_package(cm, cp)][-1] == "Overhead · 0.5×"

dq = [dict(ev="crash", **{k: v for k, v in crash(1200, "A", 3).items() if k != "ev"}),
      dict(ev="disqualified", k=1200, heat=1, heats_won={"A": 0, "B": 0}, seat="A"),
      dict(ev="heat_end", k=1200, heat=1, heats_won={"A": 0, "B": 0},
           result=dict(heat=1, winner="B", decided_by="disqualification", heat_time=12.0, winner_finish_time=None,
                       grid={"A": "outer", "B": "inner"}, cars={"A": dict(checkpoints=14, finish_time=None),
                                                                  "B": dict(checkpoints=15, finish_time=None)},
                       crashes=[])),
      dict(ev="end", k=1200, heat=1, heats_won={"A": 0, "B": 0}, reason="race_over", winner="B")]
dm = match([x for x in LINES if x.get("k", 0) < 1200 or x.get("ev") == "hdr"] + dq)
assert dm.points[-1]["kind"] == "dq" and dm.points[-1]["final"], dm.points
assert dm.points[-1]["how"] == "Heat 1 · Oracle · Blue disqualified (third warning) · Oracle · Orange wins the race", dm.points
assert [s["label"] for s in b.race_build_package(dm, dm.points[-1])] == \
    ["Main camera · real speed", "Trackside · 0.5×", "Overhead · 0.5×"]

# ---- the director on a fake clock, the log fed as the sim writes it (x10: 10 control steps per wall second)
fm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
fm.simdir_ok = True
feed = {"i": 0, "hold": True}


def feed_to(k):
    """The log up to game step k; held at the review (a "run" after a pause) while feed["hold"]."""
    while feed["i"] < len(lines):
        x = lines[feed["i"]]
        if x.get("k", 0) > k and x.get("ev") != "hdr" or feed["hold"] and x.get("ev") == "run" and x["k"] > 0:
            break
        fm.ingest(json.dumps(x).encode())
        feed["i"] += 1


T = [100.0]
fp = b.RacingProgram(fm)
fp.clock = lambda: T[0]
fp.last_t = T[0]
seen = []


def run(seconds, k_from=None, dt=0.05):
    end = T[0] + seconds
    while T[0] < end:
        if k_from is not None:
            feed_to(k_from[0] + (T[0] - k_from[1]) * 10.0)
        fp.live_step(T[0])
        fp.direct(T[0])
        ra, rb, alpha, ov, key = fp.compose(T[0])
        seen.append((round(T[0], 2), fp.mode, ov.get("stinger"), ov.get("shot"), ov.get("banner"), fp.pos,
                     fm.phase_at(fm.ks[-1]), ov["bug"]))
        T[0] += dt


feed_to(560)
run(20.0, k_from=(560, T[0]))
modes = [s[1] for s in seen]
i_rep = modes.index("replay")
assert all(s[6] == "run" for s in seen[:i_rep + 5]), "the crash replay runs while the race goes on"
assert any(s[2] is not None for s in seen[:i_rep]) and seen[i_rep][3] == "Main camera · real speed"
shots = [s[3] for s in seen if s[1] == "replay" and s[2] is None]
assert "Trackside · 0.5×" in shots, set(shots)
assert seen[-1][1] == "live" and seen[-1][5] > CK + 50, seen[-1][:6]
assert any("instant replay while the race goes on" in x for _, x in fp.log), list(fp.log)
assert any(s[4] == cp["how"] for s in seen if s[1] == "live"), "crash banner on the live picture"
# the heat ends: frozen, the heat package; heat 2 starts: back to live
run(100.0, k_from=(seen[-1][5] + 20, T[0]))                  # to the end of heat 1, then the review lasts
assert seen[-1][6] == "pause", seen[-1][:7]
feed["hold"] = False
run(15.0, k_from=(1310, T[0]))                                # heat 2 starts
assert any(s[1] == "replay" and s[3] == "Finish line · 0.5×" for s in seen), "heat package"
assert any(s[6] == "pause" for s in seen)
heat_banner = [s[4] for s in seen if s[1] == "replay" and s[3] == "Overhead · 0.5×"]
assert heat_banner and heat_banner[-1].startswith("Heat 1 · Oracle · Orange wins"), heat_banner[-3:]
assert seen[-1][1] == "live" and fm.heat_at(int(seen[-1][5])) == 2, seen[-1][:7]
assert seen[-1][7]["sport"] == "Racing · Duel" and seen[-1][7]["unit"].startswith("Heat 2"), seen[-1][7]

# ---- captions from the sim's /data (racing's episode log: "event" / "wall"; think_calls_<seat>.jsonl)
out = ('@@EP 0 999\n' + json.dumps(dict(event="review", step=1310, wall=time.time() - 5, seat="B",
                                        why="B won heat 1 on the inside line", plan="keep the line, brake later")) +
       '\n@@TC think_calls_A.jsonl\n' + json.dumps(dict(seq=7, t1=time.time() - 2, excerpt='[tool exec] const r = await '
                                                        'tools.exec_command({cmd:"python3 /app/output/driver.py"});')) +
       '\n').encode()
b.lb.dexec = lambda *a, **k: out
caps = b.RTCaptions(bm)
caps.update(time.time())
low = caps.lower(time.time(), pr.seats())
assert [(c["seat"], c["kind"]) for c in low] == [("A", "Action"), ("B", "Review")], low
assert low[0]["text"] == "Running a control script" and low[1]["text"].startswith("B won heat 1"), low
assert "sk-" not in b.redact("token sk-ABCDEFGHIJKLMNOPQRSTUV0123")
print("PASS: racing job names / discovery; recorded oracle duel: 3 heats, heats won, checkpoints, race order, chase "
      "camera, cut between heats, heat package main 1x -> finish line 0.5x -> overhead 0.5x; crash banner, warning, "
      "instant replay while the race goes on (main 1x -> trackside 0.5x -> live); disqualification banner; heat-end "
      "package and back to live for heat 2; captions from /data")
