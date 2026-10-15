"""Billiards (8-ball) on the program feed (broadcast.py BilliardsMatch / BilliardsProgram): the episode log read as the
live input, the table state (groups, balls left, turn, ball in hand), the director (live shot -> stinger -> replay along
the cue line and at the pocket -> stinger -> live table; shots in a row back to back; a new shot cuts a replay; replay
loops while a player thinks), English captions with the episode's reviews. Offline: fake clock, no render helper.

With the task's image and sim code on this host (and BILLIARDS_HELPER_TEST != 0) it also renders one frame of a real
recorded break through the billiards render helper container (the task's own pooltool replay and renderer)."""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="bc_billiards_test_")
os.environ["BROADCAST_CACHE"] = TMP
p = Path(__file__).resolve().with_name("broadcast.py")
sp = importlib.util.spec_from_file_location("broadcast_billiards_test", p)
b = importlib.util.module_from_spec(sp)
sp.loader.exec_module(b)

JOB = "r1-billiards-duel-s1-opusA-lunaB-20261004T061844Z"
PROJ = "billiards-duel-program-standard__testabc"
assert type(b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))) is b.BilliardsMatch
assert type(b.make_match(dict(job="r1-tabletennis-duel-x10-s1-opusA-lunaB-20261004T000000Z",
                              proj="tabletennis-duel-wallclock-progr__x", host="local", sim="-"))) is b.BMatch
bm = b.make_match(dict(job=JOB, proj=PROJ, host="local", sim="-"))
bm.simdir_ok = True
T0 = time.time()


def shot(k, seat, **kw):
    r = dict(ev="shot", k=k, player=seat, ts=T0 + k, potted=[], pockets={}, foul=None, call=None, call_made=False,
             group_assigned=None, keeps_turn=False, game_over=False, result=None, eight_respotted=False, rerack=False,
             off_table=[], cue_ball_pocketed=False, first_hit="1")
    r["break"] = k == 0
    r.update(kw)
    return r


def dec(seat, call=None):
    d = dict(type="shoot", aim_deg=90.0, speed=2.0, spin_x=0.0, spin_y=0.0)
    if call:
        d["call"] = call
    return dict(ev="decision", seat=seat, decision=d, clock_left_s=3500.0, ts=T0)


LOG = [dict(ev="service", sport="8-ball", seed=1, first_player="A", ts=T0), dict(ev="match_begins", ts=T0),
       dec("A"), shot(0, "A", potted=["9"], pockets={"9": "lc"}, keeps_turn=True),
       dict(ev="review", seat="A", after_shot=1, score={"A": 7, "B": 7}, why="Hard break potted the 9", plan="pick the easiest pot",
            clock_left_s=3590.0, ts=T0 + 1),
       dec("A", "5:rt"), shot(1, "A", potted=["5"], pockets={"5": "rt"}, call="5:rt", call_made=True, group_assigned="solids",
                              keeps_turn=True),
       dec("A", "6:lt"), shot(2, "A", call="6:lt", foul="cue ball pocketed", cue_ball_pocketed=True, potted=[])]


def info(r, i):
    """What the render helper returns per shot (here: made up, 2 s of motion; pots at 1.5 s)."""
    return dict(n=i, k=i, player=0 if r["player"] == "A" else 1, T=2.0, phi=90.0, V=2.0,
                pos0={"cue": (0.5, 0.4)}, pos1={}, t_first=0.3,
                pots=[(x, r["pockets"][x], 1.5) for x in r["potted"]] + ([("cue", "lt", 1.6)] if r["cue_ball_pocketed"] else []),
                rec={x: v for x, v in r.items() if x not in ("ev", "ts")} | {"player": 0 if r["player"] == "A" else 1})


def feed(lines):
    for x in lines:
        bm.ingest(json.dumps(x).encode())
    bm.binfo = [info(r, i) for i, r in enumerate(bm.shots)]
    bm._states = None
    bm.refresh_points()


bm.sync = lambda helper=None: None
feed(LOG[:2])
assert bm.ready() and bm.phase_at() == "run" and bm.info_json()["frames"] == 1         # the rack can be shown
feed(LOG[2:])

# ---- the table state, from the shot records
st = bm.state(1)
assert st["group"] == {"A": "solids", "B": "stripes"} and bm.left(st) == (6, 6), st
assert st["turn"] == "A" and st["bih"] is None
st = bm.state(2)
assert st["turn"] == "B" and st["bih"] == "anywhere", st                               # the scratch: B, ball in hand
assert bm.group_balls(st, "B") == ["10", "11", "12", "13", "14", "15"]
assert "Potted 5 in the bottom-right corner as called" in bm.summary(1) and "stays at the table" in bm.summary(1)
assert bm.summary(2).startswith("Foul: cue ball pocketed") and "ball in hand" in bm.summary(2)
js = bm.info_json()
assert js["sport"] == "billiards" and js["unit"] == "shot" and js["turn_based"] and js["score"] == [6, 6], js
assert [x["n"] for x in js["points"]] == [1, 2, 3] and js["points"][1]["winner"] == "Opus 5.5", js["points"]

# ---- the director, on a fake clock (the feed joins at present: 3 shots already shown)
T = [100.0]
pr = b.BilliardsProgram(bm)
pr.clock = lambda: T[0]
pr.caps.card = lambda now: dict(id=bm.id, seats=[
    dict(seat="B", h_age=1, steps=[dict(k="思考", t="hidden"), dict(k="运行命令", t="cd /app && python3 /app/output/shooter.py")])],
    events=[])
pr.caps.update(time.time())
low = pr.caps.lower(time.time(), pr.seats())
assert low[0]["kind"] == "Review" and low[0]["text"] == "Hard break potted the 9 → pick the easiest pot", low   # episode
assert low[1]["kind"] == "Action" and low[1]["text"] == "Running a control script", low                         # board
seen = []


def run(seconds, dt=0.04):
    end = T[0] + seconds
    while T[0] < end:
        pr.direct(T[0])
        fs, ov, key = pr.compose(T[0])
        seen.append(dict(t=round(T[0], 2), mode=pr.mode, st=ov.get("stinger"), tag=ov.get("tag"), shot=ov.get("shot"),
                         fs=fs, ov=ov))
        T[0] += dt


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from strings(v)


run(2.0)
assert pr.shown == 3 and seen[-1]["mode"] == "table" and seen[-1]["tag"] == "LIVE", seen[-1]
ov = seen[-1]["ov"]
assert ov["call"][0] == "BALL IN HAND" and ov["bug"]["turn"] == "B", ov                 # B to play, ball in hand
assert ov["bug"]["rows"] == [["A", "Opus 5.5", 6], ["B", "Luna", 6]], ov["bug"]
assert [r["label"] for r in ov["rack"]] == ["Solids", "Stripes"] and "9" not in ov["rack"][1]["balls"], ov["rack"]
assert seen[-1]["fs"] == dict(after=2, cam={"kind": "top"})

# a new shot: live (cue lined up with the call marked, the stroke at real speed, the result), then the replay
feed([dec("B", "12:lb"), shot(3, "B", potted=["12"], pockets={"12": "lb"}, call="12:lb", call_made=True, keeps_turn=True)])
run(0.5)
s = seen[-1]
assert s["mode"] == "shot" and s["fs"]["t"] == -1 and s["fs"]["marks"] == [{"ball": "12"}, {"pocket": "lb"}], s["fs"]
assert s["ov"]["call"] == ["CALL", "12 in the top-left corner"] and s["ov"]["speed"] == "Real speed 1×", s["ov"]
run(b.B_AIM_S - 0.5 + 1.0)
assert seen[-1]["mode"] == "shot" and 0.8 <= seen[-1]["fs"]["t"] <= 1.1, seen[-1]["fs"]         # the stroke, 1x
run(1.2 + b.B_HOLD_S + 0.8)
modes = [x["mode"] for x in seen]
assert "replay" in modes, modes[-10:]
i = modes.index("replay")
assert any(x["st"] is not None for x in seen[:i]), "no stinger before the replay"
assert any(x["fs"].get("after") == 3 and "banner" in x["ov"] for x in seen if x["mode"] == "shot"), "no result hold"
run(12.0)
labels = [x["shot"] for x in seen if x["mode"] == "replay" and x["st"] is None]
assert "Cue line · 0.5×" in labels and "Pocket cam · top-left corner · 0.5×" in labels, set(labels)
cue = next(x for x in seen if x["shot"] == "Cue line · 0.5×" and x["st"] is None)
assert cue["fs"]["cam"] == {"kind": "cue"} and cue["tag"] == "REPLAY" and cue["ov"]["speed"] == "Slow motion 0.5×"
assert seen[-1]["mode"] == "table" and any("back to the live table" in x for _, x in pr.log), list(pr.log)

# a long think: the last shot's replay again
run(b.B_IDLE_S + 2.0)
assert seen[-1]["mode"] == "replay" and any("replay loop" in x for _, x in pr.log), list(pr.log)
# a new shot during the replay: back to live at once (after the stinger)
feed([dec("B", "13:rc"), shot(4, "B", call="13:rc")])
run(1.0)
assert seen[-1]["mode"] == "shot" and any("new shot: back to live" in x for _, x in pr.log), list(pr.log)
# two more in a row (program play shoots in bursts): back to back, live, no replay between
feed([dec("A", "1:rb"), shot(5, "A", potted=["1"], pockets={"1": "rb"}, call="1:rb", call_made=True, keeps_turn=True),
      dec("A", "2:rb"), shot(6, "A", call="2:rb")])
n0 = len(seen)
run(3 * (b.B_AIM_S + 2.0 + b.B_HOLD_S) + 0.5)
burst = seen[n0:]
lives = [x for _, x in pr.log if x.startswith("live: shot")]
assert lives[-3:] == ["live: shot 5 by Luna", "live: shot 6 by Opus 5.5", "live: shot 7 by Opus 5.5"], lives
k6 = next(j for j, x in enumerate(burst) if x["mode"] == "shot" and x["fs"].get("shot") == 6)
assert all(x["mode"] == "shot" for x in burst[:k6]), "a replay between shots in a row"

# the end of the match: the result in the bug, loops over the shots that potted
feed([dec("A", "8:lt"), shot(7, "A", potted=["8"], pockets={"8": "lt"}, call="8:lt", call_made=True, game_over=True,
                             result="8 potted legally in the called pocket"),
      dict(ev="end", reason="match_over", winner="A", ts=T0 + 9)])
run(30.0)
assert bm.phase_at() == "end" and bm.info_json()["winner"] == "A"
end_ov = next(x["ov"] for x in reversed(seen) if x["mode"] == "table")
assert end_ov["phase"] == "Match over · Opus 5.5 wins" and "turn" not in end_ov["bug"] or end_ov["bug"]["turn"] is None, end_ov
assert any(x.startswith("replay loop") for _, x in pr.log)

# every on-screen text is English (no CJK; the live board's Chinese step kinds never reach the picture)
bad = {t for x in seen for t in strings(x["ov"]) if any(ord(c) >= 0x2E80 for c in t)}
assert not bad, bad

# ---- the real helper (optional): one frame of the recorded break, rendered from the task's own code
SIM = Path("/data/guichi/rcb-olympiad/tasks/billiards-duel-program-standard/environment/sim")
img = subprocess.run(["docker", "image", "inspect", b.BILLIARDS_IMAGE], capture_output=True).returncode == 0
if os.environ.get("BILLIARDS_HELPER_TEST", "1") != "0" and img and SIM.is_dir():
    real = b.make_match(dict(job=JOB, proj="billiards-duel-program-standard__helpertest", host="local", sim="-"))
    shutil.copytree(SIM, Path(TMP) / "sims" / real.id, ignore=shutil.ignore_patterns("__pycache__", "think_proxy.py"))
    real.simdir_ok = True
    for x in [dict(ev="service", seed=1, first_player="A"), dict(ev="match_begins"),
              dict(ev="decision", seat="A", decision=dict(type="shoot", aim_deg=90.0, speed=6.5, spin_x=0.0, spin_y=-0.1,
                                                          cue_xy=[0.47, 0.42]))]:
        real.ingest(json.dumps(x).encode())
    try:
        b.bill_call(b.BILL, real, ("sync", real.id, real.cfg, None))
        assert len(real.binfo) == 1 and real.binfo[0]["rec"]["potted"] == ["9"], real.binfo   # = the recorded log
        rp = b.BilliardsProgram(real)
        jpg, rt = b.bill_call(b.BILL, real, ("frame", real.id, dict(shot=0, t=0.8, cam={"kind": "cue"}),
                                             rp.overlay_for(0, "REPLAY", "Slow motion 0.5×"), b.SIZE))
        assert jpg[:2] == b"\xff\xd8" and len(jpg) > 20000, len(jpg)
        print(f"helper: break replayed (potted 9 as in the log), cue-line frame {len(jpg)} bytes in {rt:.3f} s")
    finally:
        b.BILL.stop()
else:
    print("helper part skipped (no billiards image or task sim dir)")
shutil.rmtree(TMP, ignore_errors=True)
print("PASS: billiards match from the episode log; groups / balls left / turn / ball in hand; live shot (aim with the "
      "call marked, 1x stroke, result) -> stinger -> cue-line + pocket-cam replays at 0.5x -> live table; replay loop "
      "while thinking; new shot cuts the replay; bursts back to back; match end; captions: episode reviews + board "
      "actions; English only")
