"""Soccer (RoboOlympics 2 v 2) on the program feed (broadcast.py SoccerMatch / SoccerProgram), offline: fake clock,
synthetic broadcast log in the soccer sim's format, no render helper. Job-name parsing and discovery of the three
formats, stoppages as the replay unit (goal, penalty banner, shoot-out kick, full time), the score bug (teams, score,
half and clock, shoot-out), the package (main 1x -> behind the scorer 0.5x -> overhead 0.5x), the director's cut to
the replay and back on resume, captions (team channel, model output; two seats shown, reasoning never, secrets
redacted).

With SOCCER_LOG=<a soccer broadcast.jsonl> it also checks the parser on a recorded log."""
import importlib.util
import json
import os
import time
from pathlib import Path

p = Path(__file__).resolve().with_name("broadcast.py")
sp = importlib.util.spec_from_file_location("broadcast_soccer_test", p)
b = importlib.util.module_from_spec(sp)
sp.loader.exec_module(b)
lb = b.lb

JOB = "show-soccer-team-x10-s1-lunaA-lunaB-20261005T121533Z"
PROJ = "soccer-team-wallclock-program-st__testsoc__env"
d = lb.parse_job(JOB, PROJ)
assert d["sport"] == "soccer" and d["duel"] and d["team"] and d["players"] == {"A": "luna", "B": "luna"}, d
d = lb.parse_job("soccerval2-soccer-duel-x10-s1-oracleA-oracleB-20261005T044840Z", "soccer-duel-x__a")
assert d["duel"] and not d["team"], d
d = lb.parse_job("soccerval2-soccer-oracle-L1-x10-s101-20261005T044710Z", "soccer-match-x__b")
assert not d["duel"] and d["level"] == "1", d
assert lb.parse_job("r1-tabletennis-duel-x10-s1-opusA-lunaB-20261004T000000Z", "tabletennis-duel__x")["team"] is False
assert "/data/guichi/rcb-oly-soccer" in lb.TREES["local"]
bm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
assert type(bm) is b.SoccerMatch and bm.mode == "team" and (bm.sport, bm.mode) in b.SUPPORTED
assert type(b.make_match(dict(job="r1-tabletennis-duel-x10-s1-opusA-lunaB-20261004T000000Z",
                              proj="tabletennis-duel-wallclock-progr__x", host="local", sim="-"))) is b.BMatch
single = b.make_match(dict(job="soccerval2-soccer-oracle-L1-x10-s101-20261005T044710Z", proj="soccer-match-x__b",
                           host="local", sim="-"))
assert single.mode == "single" and single.labels()[1] == "B  Program L1", single.labels()
bm.simdir_ok = True

DT, EVERY = 0.02, 5


def q(k, ball):
    out = list(ball) + [1, 0, 0, 0]
    for i, x in enumerate((-1.3, -3.0, 1.8, 3.4)):                     # four robots: root pos + quat + 12 joints
        out += [x, 0.1 * i, 0.67, 1, 0, 0, 0] + [0.0] * 12
    return out


lines = [dict(ev="hdr", every=EVERY, sport="soccer", mode="team", slowdown=10.0, dt=DT, seed=1, level=0,
              players={s: s for s in ("A1", "A2", "B1", "B2")}, robots=["A1", "A2", "B1", "B2"], half_s=150.0),
         dict(ev="run", k=0, score={"A": 0, "B": 0})]
for k in range(0, 600, EVERY):                                           # B attacks -x: the ball rolls into A's goal
    lines.append(dict(k=k, q=q(k, [-k / 600 * 5.2, 0.2, 0.11]), clock=k * DT, period=1, score={"A": 0, "B": 0}))
lines.append(dict(k=599, q=q(599, [-5.2, 0.2, 0.11]), clock=599 * DT, period=1, score={"A": 0, "B": 0}))
lines += [dict(ev="ref_penalty", k=300, robot="A1", why="local game stuck", seconds=15.0),
          dict(ev="ref_goal", k=600, team="B", robot="B1", own_goal=False, clock=12.0, period=1, score={"A": 0, "B": 1}),
          dict(ev="ref_stoppage", k=600, kind="goal", team="B", robot="B1", period=1, score={"A": 0, "B": 1}),
          dict(ev="pause", k=600, kind="goal", score={"A": 0, "B": 1})]
for x in lines:
    bm.ingest(json.dumps(x).encode())
assert len(bm.points) == 1, bm.points
pt = bm.points[0]
assert pt["k"] == 599 and pt["winner"] == "B" and pt["how"] == "Goal · B1 scores for B", pt
assert bm.score_at(599) == (0, 1) and bm.score_at(400) == (0, 0)
assert bm.phase_at(bm.ks[-1]) == "pause"                                 # the pause is logged one step after the frame
assert bm.clock_text(400) == "1st half 0:08", bm.clock_text(400)
assert "Penalty · A1 removed for 15 s" in bm.flag_at(320) and bm.flag_at(520) is None
assert -b.SOC_LOOK_X <= bm.recs[-1]["look"][0] < -1.5, bm.recs[-1]["look"]      # the main camera follows the ball

pkg = b.soc_build_package(bm, pt)
assert [s["speed"] for s in pkg] == [1.0, 0.5, 0.5], pkg
assert pkg[0]["cam"] == "main" and pkg[2]["cam"] == "high" and "Behind the scorer" in pkg[1]["label"], pkg
assert isinstance(pkg[1]["cam"], tuple) and pkg[1]["cam"][0][0] < 0 and pkg[1]["cam"][2] == 180.0, pkg  # B's goal: -x
assert all(bm.ks[0] <= s["k0"] < s["k1"] <= 599 for s in pkg), pkg

# the director and the graphics, on a fake clock
T = [100.0]
pr = b.get_program.__globals__["SoccerProgram"](bm)
assert type(pr) is b.SoccerProgram and bm.program_cls is b.SoccerProgram
pr.clock = lambda: T[0]
pr.last_t = T[0]
now = time.time()
pr.caps.seat.update({
    "A1": dict(kind="Action", text="run game observe", t=now - 30),
    "A2": dict(kind="Team", text="I cover our half", t=now - 20, to="A1"),
    "B1": dict(kind="Says", text="pressing high", t=now - 2),
    "B2": dict(kind="Action", text="run python3 ctl.py", t=now - 1)})
low = pr.caps.lower(now, pr.seats())
assert [c["seat"] for c in low] == ["A2", "B1"], low                     # two seats: team message and speech first
assert low[0]["kind"] == "Team → A1" and low[0]["name"] == "Luna", low
assert b.soc_caption('[tool exec] const r = await tools.exec_command({cmd:"cd /app && python3 /app/output/p.py",'
                     'workdir:"/app"});') == ("Action", "Running a control script")
assert b.soc_caption('[tool exec] const r = await tools.exec_command({cmd:"game say \\"hi\\""});') is None
assert b.soc_caption('I will pass to A2 [tool Bash] {"command": "game status"}') == ("Action", "Checking the score and match state")
assert b.soc_caption("Plan: hold the left wing") == ("Says", "Plan: hold the left wing")
assert "sk-" not in b.redact("key sk-ABCDEFGHIJKLMNOPQRSTUV0123 here")

seen = []


def run(seconds, dt=0.05):
    end = T[0] + seconds
    while T[0] < end:
        pr.live_step(T[0])
        pr.direct(T[0])
        ra, rb, alpha, ov, key = pr.compose(T[0])
        seen.append((round(T[0], 2), pr.mode, ov.get("stinger"), ov.get("tag"), ov.get("shot"), ov["bug"], ov.get("banner")))
        T[0] += dt


run(4.0)
modes = [s[1] for s in seen]
assert "replay" in modes, seen[-3:]
assert any(s[2] is not None for s in seen[:modes.index("replay")]), "no stinger before the replay"
bug = seen[-1][5]
assert bug["sport"] == "Soccer · Team (4 agents)" and bug["rows"] == [["A", "A · Luna ×2", 0], ["B", "B · Luna ×2", 1]], bug
assert bug["unit"].startswith("1st half 0:11"), bug
assert seen[-1][6] == "Goal · B1 scores for B", seen[-1]
ov = pr.base_overlay(time.time(), bm.ks[-1])
assert ov["seat_rgb"]["A2"] and ov["phase"].startswith("Stoppage"), ov
run(25.0)
shots = {s[4] for s in seen if s[1] == "replay" and s[2] is None}
assert {x["label"] for x in pkg} <= shots, shots
assert seen[-1][1] == "live", seen[-1]
# play resumes, then a shoot-out kick decides the match
more = [dict(ev="run", k=600, score={"A": 0, "B": 1})]
more += [dict(k=k, q=q(k, [0.0, 0.0, 0.11]), clock=0.0, period="pso", score={"A": 0, "B": 1}) for k in list(range(605, 800, 5)) + [799]]
more += [dict(ev="ref_pso_kick", k=800, n=1, team="A", striker="A1", keeper="B2", result="saved"),
         dict(ev="ref_end", k=800, winner="B", how="regulation", score={"A": 0, "B": 1}),
         dict(ev="end", k=800, score={"A": 0, "B": 1}, winner="B", reason="match_over")]
for x in more:
    bm.ingest(json.dumps(x).encode())
run(1.0)
assert seen[-1][1] == "live", seen[-1]
assert len(bm.points) == 2 and bm.points[1]["kind"] == "pso_kick" and bm.points[1]["final"], bm.points
assert bm.points[1]["how"].startswith("Shoot-out kick 1 · A1 saved by B2 · Full time 0–1 · B wins"), bm.points[1]
assert bm.clock_text(bm.ks[-1]) == "Shoot-out · pens 0–0", bm.clock_text(bm.ks[-1])
assert bm.phase_at(bm.ks[-1]) == "end"
info = bm.info_json()
assert info["mode"] == "team" and info["unit"] == "stoppage" and info["supported"] and len(info["points"]) == 2, info

if os.environ.get("SOCCER_LOG"):                                         # a recorded / regenerated log
    rm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
    for ln in open(os.environ["SOCCER_LOG"], "rb"):
        rm.ingest(ln)
    goals = [x for x in rm.points if x.get("goal")]
    assert rm.points and rm.ks and all(len(r["q"]) == 83 for r in rm.recs[:50]), len(rm.points)
    assert rm.score_at(rm.ks[-1]) == tuple((rm.points[-1]["score"] or {}).get(t, 0) for t in "AB")
    assert all(b.soc_build_package(rm, x) for x in rm.points)
    print(f"recorded log: {len(rm.recs)} frames, {len(rm.points)} stoppages, {len(goals)} goals, "
          f"final {rm.score_at(rm.ks[-1])}, {rm.clock_text(rm.ks[-1])}")
print("PASS: soccer job names / discovery; stoppages as replay units (goal, shoot-out kick, full time); score bug "
      "with teams, half clock, shoot-out; penalty banner; package main 1x -> behind the scorer 0.5x -> overhead 0.5x; "
      "live -> stinger -> replay -> live on resume; two seat captions with team messages; no reasoning, secrets redacted")
