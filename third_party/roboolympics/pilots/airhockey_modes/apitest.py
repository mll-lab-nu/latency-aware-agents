"""The air-hockey pilot's post-match tests.

  gen      API-test cases from our reference games: the reference strategy with the platform's skills against the
           scripted opponent L1-L3; every strike request is a candidate (the full state of seat A's arm, its servo and
           the puck at the request). Feasibility filter: the speed is clamped to what our planner can do on that
           line, our planner must plan it on time (no lateness), the ideal (perfect-mallet) outcome must meet the puck
           at the requested time (within 15 ms: the request's puck prediction holds) and our platform implementation
           must touch the puck. Split by game: public (examples for the agents) / hidden (only after the match).
  api      the API test of an implementation on the cases (contract.py: errors, success, goals)
  comp     the competition test: our fixed reference strategy calls the implementation in games against L1-L3
  high     the HIGH level's post-match test: a deliverable player (with the platform's skills) in the same games

    python3 apitest.py gen OUT.json [--seeds 1 6]
    python3 apitest.py api platform|ref_low|baseline|PATH_TO_skill.py [--cases hidden|public] [--out f.jsonl]
    python3 apitest.py comp platform|ref_low|baseline|PATH [--out f.jsonl]
    python3 apitest.py high ref|PATH_TO_player.py [--out f.jsonl]

evaluate(level, deliverable_dir) is the verifier's entry point (the deliverable runs in sandbox.py)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "programs"))
import contract as CT  # noqa: E402
import harness as HS  # noqa: E402
import planner as PL  # noqa: E402
import world as W  # noqa: E402

CASES = Path(os.environ.get("AHM_CASES") or HERE / "apitest_cases.json")
COMP_GAMES = [(1, 101), (2, 202), (3, 303)]          # (opponent level, seed): the competition and high tests
GEN_LEVELS = (1, 2, 3)
PUBLIC_SEEDS = {1, 2}                                 # cases from these game seeds are public examples
HORIZON_S = 2.6                                       # after the requested contact time


# ------------------------------------------------------------------------------------------------ state snapshots
def snapshot(h, s="A"):
    j = h.jl[s]
    return {"puck": h.puck().tolist(), "puck_v": h.puck_velocity().tolist(),
            "q": h.data.qpos[h.qadr[s]].tolist(), "qd": h.data.qvel[h.dadr[s]].tolist(),
            "q_target": h.qtarget[s].tolist(), "q_target_v": h.tv[s].tolist(),
            "servo": [j.q.tolist(), j.qd.tolist(), j.qdd.tolist()]}


def restore(state):
    """a fresh world (no scripted opponent; seat B parked away from its mouth) at time 0 with seat A's arm, its servo
    and the puck as in the snapshot; seat A holds its current target until commanded"""
    h = W.ModesHockey(seed=0, duel=True, first="A")
    h.park("B", W.PARK_B)
    s = "A"
    h.data.qpos[h.qadr[s]] = state["q"]
    h.data.qvel[h.dadr[s]] = state["qd"]
    h.qtarget[s] = np.array(state["q_target"], float)
    h.tv[s] = np.array(state["q_target_v"], float)
    h.jl[s].reset(*[np.array(x, float) for x in state["servo"]])
    h.ctl[s] = "rows"
    h.rows[s].clear()
    h.data.qpos[h.px], h.data.qpos[h.py] = state["puck"]
    h.data.qvel[h.pvx], h.data.qvel[h.pvx + 1] = state["puck_v"]
    h.data.qvel[h.pvx + 2] = 0.0
    mujoco.mj_forward(h.model, h.data)
    h.prev_pos = {x: h.position(x)[:2].copy() for x in "AB"}
    h.side = "A" if state["puck"][0] < -H_PUCK_R() else ("B" if state["puck"][0] > H_PUCK_R() else None)
    h.side_steps = 0
    return h


def seg_dist(q, a, b):
    ab = b - a
    s = float(np.clip((q - a) @ ab / max(1e-9, float(ab @ ab)), 0, 1))
    return float(np.linalg.norm(q - (a + s * ab)))


def H_PUCK_R():
    return W.H.PUCK_R


# ------------------------------------------------------------------------------------------------ one strike
def run_strike(state, req, impl_call, level):
    """execute one strike request from a snapshot: impl_call(name, *args) -> commands (low: plan / hold) or, for the
    platform, level = 'high' and the request goes to the platform's skill. Returns the measurement."""
    h = restore(state)
    seat = W.Seat(h, level)
    replies = []
    meas = {"contact_t": None, "goal": 0, "errors": []}
    view = {"state": h.match_view("A"), "replies": []}
    try:
        if level == "high":
            cmds = [{"type": "strike", **req}]
        else:
            cmds = impl_call("strike", view, dict(req)) or []
    except Exception as e:  # noqa: BLE001
        meas["errors"].append(f"{type(e).__name__}: {e}"[:300])
        cmds = []
    replies = [seat.command(c) for c in cmds]
    if level == "high" and replies and isinstance(replies[0], dict):
        meas["platform_reply"] = replies[0]
    t_end = float(req["t"]) + HORIZON_S
    k_out = None
    while h.k * W.DT < t_end and not (h.done or h.awaiting_review):
        if level != "high" and h.k > 0 and h.k % HS.TICK == 0:
            view = {"state": h.match_view("A"), "replies": replies}
            try:
                cmds = impl_call("step", view) or []
            except Exception as e:  # noqa: BLE001
                meas["errors"].append(f"{type(e).__name__}: {e}"[:300])
                cmds = []
                if len(meas["errors"]) > 20:
                    break
            if cmds:
                replies = [seat.command(c) for c in cmds]
        m0, p0, w0 = h.position("A")[:2].copy(), h.puck().copy(), h.puck_velocity().copy()
        k0 = h.k
        c = h.step()
        if meas["contact_t"] is None and "A" in c:
            m1 = h.position("A")[:2].copy()
            tau, vm = CT.contact_instant(m0, m1, p0, w0)
            meas["contact_t"] = round(k0 * W.DT + tau, 4)
            meas["mallet_pos"] = (m0 + vm * tau).tolist()
            meas["mallet_v"] = vm.tolist()
            k_out = h.k + int(round(CT.OUT_DT / W.DT))
        if k_out is not None and h.k == k_out:
            meas["puck_v_out"] = h.puck_velocity().tolist()
    if h.awaiting_review or h.done:
        r = h.results[-1]
        if r["how"] == "goal":
            meas["goal"] = 1 if r["winner"] == "A" else -1
    meas["refusals"] = seat.refusals
    meas["refusal_samples"] = [x for x in replies if isinstance(x, str)][:2]
    return meas


# ------------------------------------------------------------------------------------------------ case generation
def gen(out, seeds=(1, 6)):
    from ref_strategy import RefStrategy
    pl = PL.Planner()
    ideal = CT.Ideal()
    raw = []
    for lv in GEN_LEVELS:
        for sd in range(seeds[0], seeds[1] + 1):
            h = W.ModesHockey(seed=sd, level=lv)
            seat = W.Seat(h, "high", planner=pl)
            strat = RefStrategy()
            while not h.done:
                if h.awaiting_review:
                    h.ready()
                    strat = RefStrategy()
                    continue
                if h.k % HS.TICK == 0:
                    for c in strat.tick(h.observe("A")):
                        if c["type"] == "strike":
                            t0 = h.k * W.DT
                            req = {k: c[k] for k in ("x", "y", "dx", "dy", "speed")}
                            req["t"] = round(c["t"] - t0, 4)
                            raw.append({"source": {"opp_level": lv, "seed": sd, "t0": round(t0, 3)},
                                        "state": snapshot(h), "req": req})
                        seat.command(c)
                h.step()
            print(f"gen L{lv} s{sd}: {len(raw)} raw", flush=True)
    cases = []
    for i, cs in enumerate(raw):
        st, req = cs["state"], dict(cs["req"])
        h0 = restore(st)
        if float(np.linalg.norm(h0.position("A")[:2] - h0.puck())) < CT.CONTACT + .01:
            continue                                     # the mallet already touches the puck
        plan = pl.strike(st["q_target"], st["q_target_v"], 0.0, req)
        if "rows" not in plan or plan["late_s"] > 0:
            continue
        if plan["speed"] < req["speed"] - 1e-6:
            req["speed"] = math.floor(plan["speed"] * 0.98 * 100) / 100
            plan = pl.strike(st["q_target"], st["q_target_v"], 0.0, req)
            if "rows" not in plan or not plan["feasible"]:
                continue
        if req["speed"] < PL.SPEED_MIN:
            continue
        idl = ideal.run(st["puck"], st["puck_v"], req)
        if idl["contact_t"] is None or abs(idl["contact_t"] - req["t"]) > .008 or idl["puck_v_out"] is None:
            continue
        # the puck's own path until shortly before the contact must stay clear of the mallet's straight way to its
        # wind-up point (else the request needs the caller to clear the way first)
        u = np.array([req["dx"], req["dy"]]) / max(1e-9, math.hypot(req["dx"], req["dy"]))
        Wp = np.array([req["x"], req["y"]]) - (CT.CONTACT + .1) * u
        m0 = h0.position("A")[:2]
        path = ideal.path(st["puck"], st["puck_v"], max(0.0, req["t"] - .05))
        if len(path) and min(seg_dist(q, m0, Wp) for q in path) < CT.CONTACT + .02:
            continue
        meas = run_strike(st, req, None, "high")
        if meas["contact_t"] is None:
            continue
        cases.append({"id": f"c{len(cases):03d}", **{k: cs[k] for k in ("source", "state")}, "req": req,
                      "ideal": idl, "public": cs["source"]["seed"] in PUBLIC_SEEDS})
    Path(out).write_text(json.dumps({"n_raw": len(raw), "cases": cases}))
    print(f"{len(raw)} raw strike requests -> {len(cases)} feasible cases "
          f"({sum(c['public'] for c in cases)} public, {sum(not c['public'] for c in cases)} hidden)")


def load_cases(which="hidden", path=CASES):
    d = json.loads(Path(path).read_text())
    return [c for c in d["cases"] if which == "all" or c["public"] == (which == "public")]


# ------------------------------------------------------------------------------------------------ implementations
def impl_of(kind, spec=None):
    """an in-process implementation (ours) or a sandboxed deliverable (a path): (call, level)"""
    if kind == "platform":
        return None, "high"
    if kind in ("ref_low", "baseline"):
        mod = __import__("ref_low_skill" if kind == "ref_low" else "baseline_ik")
        obj = mod.Impl(spec or {})
        return (lambda name, *a: getattr(obj, name)(*a)), "low"
    raise ValueError(kind)


def api_test(call, level, cases):
    rows = []
    for cs in cases:
        t0 = time.time()
        meas = run_strike(cs["state"], cs["req"], call, level)
        e = CT.errors(cs["req"], meas, cs["ideal"])
        rows.append({"id": cs["id"], **e, "errors": meas["errors"][:2], "refusal_samples": meas["refusal_samples"],
                     "cpu_s": round(time.time() - t0, 2)})
    return rows


def summarize_api(rows):
    n = len(rows)
    ok = lambda k: sum(1 for r in rows if (r.get("ok") or {}).get(k))
    med = lambda k: (round(float(np.median([abs(r[k]) for r in rows if k in r])), 4) if any(k in r for r in rows) else None)
    return {"n": n, "success": sum(r["success"] for r in rows), "success_rate": round(sum(r["success"] for r in rows) / max(1, n), 3),
            "contact": sum(r["contact"] for r in rows), "ok_time": ok("time"), "ok_pos": ok("pos"),
            "ok_puck_dir": ok("puck_dir"), "ok_puck_speed": ok("puck_speed"),
            "goals": sum(r["goal"] == 1 for r in rows), "own_goals": sum(r["goal"] == -1 for r in rows),
            "ideal_goals": sum(r["ideal_goal"] == 1 for r in rows),
            "refusals": sum(r["refusals"] for r in rows), "errors": sum(bool(r["errors"]) for r in rows),
            "med_time_err_s": med("time_err_s"), "med_pos_err_m": med("pos_err_m"),
            "med_mallet_speed_err": med("mallet_speed_err"), "med_mallet_dir_err_deg": med("mallet_dir_err_deg"),
            "med_puck_speed_err": med("puck_speed_err"), "med_puck_dir_err_deg": med("puck_dir_err_deg")}


def comp_test(call, level, games=COMP_GAMES):
    from ref_strategy import RefStrategy
    rows = []
    for lv, sd in games:
        drv = HS.PlatformDriver(RefStrategy) if level == "high" else HS.ImplDriver(RefStrategy, call)
        rows.append(HS.play(drv, lv, sd))
    return rows


def high_test(call, games=COMP_GAMES):
    rows = []
    for lv, sd in games:
        rows.append(HS.play(HS.PlayerDriver(call), lv, sd))
    return rows


def summarize_games(rows):
    return {"games": len(rows), "won": sum(r["won"] for r in rows),
            "goals": f'{sum(r["goals_for"] for r in rows)}:{sum(r["goals_against"] for r in rows)}',
            "goals_for": sum(r["goals_for"] for r in rows), "goals_against": sum(r["goals_against"] for r in rows),
            "own_goals_against": sum(r["own_goals_against"] for r in rows),
            "strikes": sum(r["strikes"] for r in rows), "refusals": sum(r["refusals"] for r in rows),
            "errors": sum(r["errors"] for r in rows), "complete": sum(r["complete"] for r in rows),
            "by_level": {f'L{r["opp_level"]}': f'{r["score"][0]}:{r["score"][1]}' for r in rows}}


# ------------------------------------------------------------------------------------------------ the verifier
def evaluate(level, deliverable_dir, out_json=None, toolkit_dir=None):
    """the post-match evaluation of a FINAL deliverable (sandboxed): low -> API test (hidden) + competition test; high
    -> the player's games against L1-L3 with the platform's skills"""
    from sandbox import Sandbox, SandboxError
    spec = json.loads((HERE / f"spec_{level}.json").read_text()) if (HERE / f"spec_{level}.json").exists() else {}
    spec["toolkit_dir"] = toolkit_dir or os.environ.get("AHM_TOOLKIT_DIR", "/opt/robot")
    libs = [Path(spec["toolkit_dir"]) / "lib", Path(spec["toolkit_dir"])]
    res = {}
    if level == "low":
        path = Path(deliverable_dir) / "skill.py"

        def sandboxed(fn):
            try:
                with Sandbox(path, "Impl", spec, lib_dirs=libs) as sb:
                    return fn(lambda name, *a: sb.call(name, *a))
            except SandboxError as e:
                return {"sandbox_error": str(e)[:500]}
        rows = []
        cases = load_cases("hidden")
        for cs in cases:                    # one sandbox per case: a crash or a stuck call costs only that case
            r = sandboxed(lambda call: api_test(call, "low", [cs])[0])
            if "sandbox_error" in r:
                r = {"id": cs["id"], "success": False, "contact": False, "refusals": 0, "goal": 0,
                     "ideal_goal": cs["ideal"]["goal"], "errors": [r["sandbox_error"]], "refusal_samples": []}
            rows.append(r)
        res["api_test"] = summarize_api(rows)
        res["api_rows"] = rows
        games = []
        for g in COMP_GAMES:
            r = sandboxed(lambda call: comp_test(call, "low", [g])[0])
            if "sandbox_error" in r:
                r = {"opp_level": g[0], "seed": g[1], "score": [0, 0], "won": False, "complete": False, "goals_for": 0,
                     "goals_against": 0, "own_goals_against": 0, "strikes": 0, "refusals": 0, "errors": 1,
                     "error_samples": [r["sandbox_error"]]}
            games.append(r)
        res["competition"] = summarize_games(games)
        res["competition_rows"] = games
    else:
        path = Path(deliverable_dir) / "player.py"
        games = []
        for g in COMP_GAMES:
            try:
                with Sandbox(path, "Player", spec, lib_dirs=libs) as sb:
                    games.append(high_test(lambda name, *a: sb.call(name, *a), [g])[0])
            except SandboxError as e:
                games.append({"opp_level": g[0], "seed": g[1], "score": [0, 0], "won": False, "complete": False,
                              "goals_for": 0, "goals_against": 0, "own_goals_against": 0, "strikes": 0, "refusals": 0,
                              "errors": 1, "error_samples": [str(e)[:500]]})
        res["high_test"] = summarize_games(games)
        res["high_rows"] = games
    if out_json:
        Path(out_json).write_text(json.dumps(res, default=float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "api", "comp", "high"])
    ap.add_argument("what", nargs="?")
    ap.add_argument("--seeds", nargs=2, type=int, default=[1, 6])
    ap.add_argument("--cases", default="hidden")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "gen":
        return gen(a.what or str(CASES), tuple(a.seeds))
    if a.cmd == "high":
        if a.what == "ref":
            from ref_high_player import Player
            p = Player({})
            rows = high_test(lambda name, *x: getattr(p, name)(*x))
        else:
            from sandbox import Sandbox
            with Sandbox(a.what, "Player", {}) as sb:
                rows = high_test(lambda name, *x: sb.call(name, *x))
        summ = summarize_games(rows)
    else:
        if a.what in ("platform", "ref_low", "baseline"):
            call, level = impl_of(a.what)
            rows = api_test(call, level, load_cases(a.cases)) if a.cmd == "api" else comp_test(call, level)
        else:
            from sandbox import Sandbox
            spec = {"toolkit_dir": str(HERE / "toolkit")}
            with Sandbox(a.what, "Impl", spec, lib_dirs=[HERE / "toolkit" / "lib"]) as sb:
                call = lambda name, *x: sb.call(name, *x)
                rows = api_test(call, "low", load_cases(a.cases)) if a.cmd == "api" else comp_test(call, "low")
        summ = summarize_api(rows) if a.cmd == "api" else summarize_games(rows)
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
