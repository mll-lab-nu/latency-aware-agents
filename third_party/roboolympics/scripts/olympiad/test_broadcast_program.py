"""Program feed (broadcast.py Program): the director's cuts, the highlight package and the agent captions, offline
(fake clock, synthetic log, no render helper)."""
import importlib.util
import json
import time
from pathlib import Path

p = Path(__file__).resolve().with_name("broadcast.py")
sp = importlib.util.spec_from_file_location("broadcast_program_test", p)
b = importlib.util.module_from_spec(sp)
sp.loader.exec_module(b)

# a table-tennis duel log: hdr, run, one point (the ball flies to +y, is hit back at k=60, bounces at k=90), pause
bm = b.BMatch(dict(job="r1-tabletennis-duel-x10-s1-opusA-lunaB-20261004T000000Z",
                   proj="tabletennis-duel-wallclock-progr__testabc", host="local", sim="-"))
bm.simdir_ok = True
lines = [dict(ev="hdr", every=2, sport="tabletennis", mode="duel", slowdown=10.0, dt=0.01), dict(ev="run", k=0)]


def ball(k):
    t = k * 0.01
    if k <= 60:
        return [0.0, -1.2 + 4.0 * t, 0.96]
    if k <= 90:
        return [0.0, 1.2 - 4.0 * (t - 0.6), 0.96 - 3.0 * (t - 0.6)]      # hit back, falling
    return [0.0, 0.0 - 4.0 * (t - 0.9), 0.06 + 3.0 * (t - 0.9)]          # the bounce


for k in range(0, 121, 2):
    lines.append(dict(k=k, q=ball(k) + [1, 0, 0, 0]))
lines += [dict(ev="point", k=120, score={"A": 0, "B": 1}, winner="B", how="A missed the ball", n=1),
          dict(ev="pause", k=120)]
for x in lines:
    bm.ingest(json.dumps(x).encode())

moments = b.key_moments(bm, 0, len(bm.ks))
assert [m[1] for m in moments] == ["contact", "bounce"], moments
assert abs(moments[0][0] - 60) <= 2 and abs(moments[1][0] - 90) <= 2, moments
pkg = b.build_package(bm, bm.points[-1])
assert [s["speed"] for s in pkg] == [1.0, 0.5, 0.5, 0.25], pkg
assert pkg[1]["cam"] == "end_b" and "bounce" in pkg[-1]["label"], pkg       # behind B (last hit at y > 0); the bounce
assert isinstance(pkg[-1]["cam"], tuple) and all(bm.ks[0] <= s["k0"] < s["k1"] <= 120 for s in pkg), pkg

# the director, on a fake clock
T = [100.0]
pr = b.Program(bm)
pr.clock = lambda: T[0]
pr.last_t = T[0]
pr.caps.card = lambda now: dict(id=bm.id, seats=[
    dict(seat="A", h_age=3, steps=[dict(k="思考", t="hidden"), dict(k="运行命令", t="cd /app && python3 /app/output/player.py")]),
    dict(seat="B", h_age=1, steps=[dict(k="说", t="Using key sk-ABCDEFGHIJKLMNOPQRSTUV0123 now")])],
    events=[dict(t="10:00:00", c="rev", x="复盘 B（luna） WHY：it was short PLAN：add topspin", age=2)])
pr.caps.update(time.time())
low = pr.caps.lower(time.time(), pr.seats())
assert low[0]["text"] == "Running a control script" and low[0]["kind"] == "Action", low       # never the reasoning
assert low[1]["kind"] == "Review" and low[1]["text"] == "it was short → add topspin", low     # the review first
pr.caps.review.clear()
assert "sk-" not in pr.caps.lower(time.time(), pr.seats())[1]["text"]                    # secrets redacted

seen = []


def run(seconds, dt=0.04):
    end = T[0] + seconds
    while T[0] < end:
        pr.live_step(T[0])
        pr.direct(T[0])
        ra, rb, alpha, ov, key = pr.compose(T[0])
        seen.append((round(T[0], 2), pr.mode, ov.get("stinger"), ov.get("tag"), ov.get("shot"), ov["bug"]["rows"]))
        T[0] += dt


run(3.0)                                    # the playhead reaches the point's end; the world is paused
modes = [s[1] for s in seen]
assert "replay" in modes, seen[-5:]
first_replay = modes.index("replay")
assert any(s[2] is not None for s in seen[:first_replay]), "no stinger before the replay"
assert seen[-1][3] == "REPLAY" and seen[-1][5] == [["A", "Opus 5.5", 0], ["B", "Luna", 1]], seen[-1]
run(30.0)                                   # the whole package, then back to live (still paused: live hold)
shots = [s[4] for s in seen if s[1] == "replay" and s[2] is None]
assert all(x in shots for x in (pkg[0]["label"], "Behind Luna · 0.5×", pkg[2]["label"], pkg[3]["label"])), set(shots)
assert seen[-1][1] == "live", seen[-1]
run(b.LIVE_HOLD_S + 2)                      # a long pause: the calm loop cuts to the replay again
assert seen[-1][1] == "replay", seen[-1]
# play resumes: back to live at once (after the stinger)
for x in [dict(ev="run", k=120)] + [dict(k=k, q=[0, 0, -10, 1, 0, 0, 0]) for k in range(122, 200, 2)]:
    bm.ingest(json.dumps(x).encode())
run(1.0)
assert seen[-1][1] == "live" and seen[-1][2] is None, seen[-1]
assert any("back to live" in x for _, x in pr.log), list(pr.log)
print("PASS: contact/bounce found; package 1x/0.5x/0.5x/0.25x with end_b + close-up; live -> stinger -> replay -> "
      "live; calm loop in a long pause; cut back on resume; captions skip reasoning, review first, secrets redacted")
