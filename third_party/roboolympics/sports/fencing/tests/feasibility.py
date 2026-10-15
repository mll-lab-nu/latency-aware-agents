"""Feasibility gate for robot epee fencing (run before building the tasks; evidence for S4 and R6).

Scripted strategies with privileged state (the opponent's cart, point and arm, seen with a reaction latency) fence
phrases from the on-guard lines in the shared fencing world. A round robin over seeded small start variations and both
seats shows whether one trivial strategy dominates, how many phrases end in double touches, and whether the outcome is
decided by tiny timing noise (a 2 mm start perturbation flipping the result). Not part of any task (entry gate
evidence, sports/fencing/README.md; moved here from scripts/olympiad/fencing/ in the v0.2 migration).

usage: python feasibility.py [--seeds N] [--set RAIL_VMAX=2.0,...] [--latency 0.06] [--perturb 0.002] [--out f.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # sports/fencing (the simulator)
import fencing as F  # noqa: E402
import strategies  # noqa: E402

PHRASE_MAX_S = 20.0
SENSE = dict(latency=0.06)
from strategies import STRATS, snapshot, view  # noqa: E402  (the scripted strategies, shared with the opponent)


def play_phrase(SA, SB, seed, jitter, perturb=0.0, start=None, max_s=PHRASE_MAX_S, w=None):
    w = w or F.World()
    rng = np.random.default_rng(10_000 + seed)
    x0 = start or {"A": -F.ON_GUARD, "B": -F.ON_GUARD}
    xa = x0["A"] + rng.uniform(-0.05, 0.05) * jitter + perturb
    xb = x0["B"] + rng.uniform(-0.05, 0.05) * jitter
    w.place({"A": xa, "B": xb})
    ctl = {s: F.Controller(w, s) for s in F.SEATS}
    strat = {"A": SA(ctl["A"], np.random.default_rng([seed, 1])), "B": SB(ctl["B"], np.random.default_rng([seed, 2]))}
    lag = {s: int(round(SENSE["latency"] / F.CONTROL_DT)) for s in F.SEATS}
    hist = []
    ph = F.Phrase(w)
    halt = None
    max_tf = {s: 0.0 for s in F.SEATS}
    while halt is None:
        hist.append(snapshot(w))
        t = w.k * F.CONTROL_DT
        if t >= max_s - 1e-9:
            halt = dict(why="phrase_cap", k=w.k, touches={})
            break
        for s in F.SEATS:
            snap = hist[max(0, len(hist) - 1 - lag[s])]
            o = view(snap, s)
            o["me"] = view(hist[-1], s)["me"]          # own state without delay
            strat[s].act(t, o)
        rec = w.step({s: ctl[s].next_target() for s in F.SEATS})
        for s in F.SEATS:
            max_tf[s] = max(max_tf[s], rec["tip_force"][s])
        halt = ph.update(rec, w.k - 1)
    tch = halt.get("touches", {})
    if halt["why"] == "touch":
        res = "double" if len(tch) == 2 else next(iter(tch))
    elif halt["why"] == "rear_limit":
        res = "B" if halt["seat"] == "A" else "A"
    else:
        res = halt["why"]
    return dict(result=res, t=round(w.k * F.CONTROL_DT, 2), why=halt["why"], end={s: w.cart_x(s) for s in F.SEATS},
                thrusts={s: strat[s].thrusts for s in F.SEATS},
                carts={s: round(w.cart_x(s), 2) for s in F.SEATS})


BOUT_S, EXTRA_S, BOUT_TOUCHES = 180.0, 60.0, 5


def replace_on_guard(end):
    """t.22.7 / t.31: no touch awarded -> where they were, each retreating equally to the on-guard distance (4 m
    between the carts), nobody pushed behind the rear limit who was in front of it (t.22.8)"""
    g = -(end["A"] + end["B"])                 # distance between the carts (own-frame x of each is negative)
    need = max(0.0, 2 * F.ON_GUARD - g)
    xa, xb = end["A"] - need / 2, end["B"] - need / 2
    for me, ot in (("A", "B"), ("B", "A")):
        x = {"A": xa, "B": xb}
        lim = -F.PISTE_HALF + F.CART_HALF + 0.01
        if x[me] < lim and end[me] >= lim:
            short = lim - x[me]
            if me == "A":
                xa, xb = lim, xb - short
            else:
                xb, xa = lim, xa - short
    return {"A": xa, "B": xb}


def play_bout(SA, SB, seed, jitter, perturb=0.0):
    """a pool bout (t.38): 5 touches or 3 min; at 4-4 doubles annulled; tie at time -> one minute for a deciding
    touch with priority drawn by lot (doubles annulled)"""
    w = F.World()
    sc = {"A": 0, "B": 0}
    clock = 0.0
    start = {"A": -F.ON_GUARD, "B": -F.ON_GUARD}
    extra, prio = False, None
    phrases = []
    dbl_aw = 0
    i = 0
    how = None
    while True:
        limit = (BOUT_S if not extra else BOUT_S + EXTRA_S) - clock
        r = play_phrase(SA, SB, seed * 1000 + i, jitter, perturb if i == 0 else 0.0, start=start, max_s=limit, w=w)
        i += 1
        clock += r["t"]
        res = r["result"]
        phrases.append(res)
        annul = res == "double" and (extra or (sc["A"] == sc["B"] == BOUT_TOUCHES - 1))
        if res in ("A", "B"):
            sc[res] += 1
        elif res == "double" and not annul:
            sc["A"] += 1; sc["B"] += 1
            dbl_aw += 1
        if res in ("A", "B") or (res == "double" and not annul):
            start = {"A": -F.ON_GUARD, "B": -F.ON_GUARD}
        else:
            start = replace_on_guard(r["end"])
        if extra and sc["A"] != sc["B"]:
            how = "priority_minute_touch"; break
        if max(sc.values()) >= BOUT_TOUCHES and sc["A"] != sc["B"]:
            how = "touches"; break
        if clock >= BOUT_S - 1e-9 and not extra:
            if sc["A"] != sc["B"]:
                how = "time"; break
            extra = True
            prio = "A" if np.random.default_rng([seed, 77, *map(ord, SA.name + "|" + SB.name)]).random() < 0.5 else "B"
            start = {"A": -F.ON_GUARD, "B": -F.ON_GUARD}
            continue
        if extra and clock >= BOUT_S + EXTRA_S - 1e-9:
            how = "priority_lot"; break
    win = prio if how == "priority_lot" else ("A" if sc["A"] > sc["B"] else "B")
    return dict(winner=win, how=how, score=sc, clock=round(clock, 2), phrases=phrases, doubles_awarded=dbl_aw)


_BASE = {k: getattr(F, k) for k in ("ARM", "ARM_VEL", "ARM_ACC", "ARM_JERK", "RAIL_VMAX", "RAIL_AMAX", "RAIL_JMAX", "F_TOUCH",
                                     "DOUBLE_MS", "T_TOUCH_SAMPLES")}


def apply_sets(sets):
    """fencing constants for a variant; ARM_SCALE multiplies the arm's joint speed and acceleration limits"""
    for k, v in _BASE.items():           # every job starts from the module's own values (pool workers are reused)
        setattr(F, k, v)
    for k, v in (sets or {}).items():
        if k == "ARM_SCALE":
            F.ARM = _BASE["ARM"].scaled(v)
            F.ARM_VEL, F.ARM_ACC, F.ARM_JERK = F.ARM.vel, F.ARM.acc, F.ARM.jerk
        else:
            setattr(F, k, v)


def _bjob(job):
    ia, ib, seed, jitter, sets, perturb, latency = job
    apply_sets(sets)
    SENSE["latency"] = latency
    r = play_bout(STRATS[ia], STRATS[ib], seed, jitter, perturb)
    return dict(A=STRATS[ia].name, B=STRATS[ib].name, seed=seed, **r)


def run_bouts(names, seeds, jitter=1.0, sets=None, perturb=0.0, latency=0.06, workers=8):
    from multiprocessing import Pool
    idx = [i for i, s in enumerate(STRATS) if s.name in names]
    jobs = [(ia, ib, seed, jitter, sets, perturb, latency) for ia in idx for ib in idx for seed in range(seeds)]
    with Pool(workers) as p:
        return p.map(_bjob, jobs, chunksize=1)


def summarize_bouts(rows, names):
    tab = {a: {b: [0, 0] for b in names} for a in names}
    how, side, dbl, tch = {}, [0, 0], 0, 0
    for r in rows:
        how[r["how"]] = how.get(r["how"], 0) + 1
        side[0 if r["winner"] == "A" else 1] += 1
        wn, ls = (r["A"], r["B"]) if r["winner"] == "A" else (r["B"], r["A"])
        tab[wn][ls][0] += 1
        tab[ls][wn][1] += 1
        dbl += r["phrases"].count("double")
        tch += sum(1 for p in r["phrases"] if p in ("A", "B", "double"))
    share = {}
    for a in names:
        ws = sum(tab[a][b][0] for b in names if b != a)
        ls = sum(tab[a][b][1] for b in names if b != a)
        share[a] = round(ws / max(1, ws + ls), 3)
    cross = [r for r in rows if r["A"] != r["B"]]
    active = [r for r in cross if "wall" not in (r["A"], r["B"])]
    lot = lambda rs: round(sum(r["how"] == "priority_lot" for r in rs) / max(1, len(rs)), 3)
    # touches awarded (a double awards one to each; annulled doubles award nothing)
    aw_d = sum(2 * r.get("doubles_awarded", 0) for r in cross)
    aw = sum(r["score"]["A"] + r["score"]["B"] for r in cross)
    # bouts whose margin needed the doubles: without the awarded doubles the winner would not lead
    return dict(table={a: {b: "%d-%d" % tuple(tab[a][b]) for b in names} for a in names}, bout_win_share=share,
                how=how, seat_A_B_wins=side, double_share_of_touch_phrases=round(dbl / max(1, tch), 3), n=len(rows),
                lot_share_cross=lot(cross), lot_share_active=lot(active),
                awarded_touches_from_doubles=round(aw_d / max(1, aw), 3),
                lot_by_pair={f"{a}|{b}": sum(r["how"] == "priority_lot" for r in cross if {r["A"], r["B"]} == {a, b})
                             for i, a in enumerate(names) for b in names[i + 1:]})


def show_bouts(sm, names, title=""):
    print("== bouts", title)
    print("%10s " % "" + " ".join("%9s" % n for n in names) + "   bout win share vs others")
    for a in names:
        print("%10s " % a + " ".join("%9s" % sm["table"][a][b] for b in names), "  ", sm["bout_win_share"][a])
    print("how decided:", sm["how"], " seat A/B wins:", sm["seat_A_B_wins"], " doubles/touch phrases:",
          sm["double_share_of_touch_phrases"])
    print("decided by the priority lot: cross pairs", sm["lot_share_cross"], " without the stalling 'wall':",
          sm["lot_share_active"], "  awarded touches from doubles:", sm["awarded_touches_from_doubles"])


def _job(job):
    ia, ib, seed, jitter, sets, perturb, latency = job
    apply_sets(sets)
    SENSE["latency"] = latency
    r = play_phrase(STRATS[ia], STRATS[ib], seed, jitter, perturb)
    return dict(A=STRATS[ia].name, B=STRATS[ib].name, seed=seed, **r)


def run_all(names, seeds, jitter=1.0, sets=None, perturb=0.0, latency=0.06, workers=8):
    from multiprocessing import Pool
    idx = [i for i, s in enumerate(STRATS) if s.name in names]
    jobs = [(ia, ib, seed, jitter, sets, perturb, latency) for ia in idx for ib in idx for seed in range(seeds)]
    with Pool(workers) as p:
        return p.map(_job, jobs, chunksize=2)


def summarize(rows, names):
    """per (row strategy, column strategy): row touches only - column only - doubles - no touch"""
    tab = {a: {b: [0, 0, 0, 0] for b in names} for a in names}
    side = [0, 0]
    why = {}
    for r in rows:
        why[r["result"] if r["result"] not in "AB" else "single"] = why.get(r["result"] if r["result"] not in "AB" else "single", 0) + 1
        a, b = r["A"], r["B"]
        res = r["result"]
        if res in ("A", "B"):
            side[0 if res == "A" else 1] += 1
            wn, ls = (a, b) if res == "A" else (b, a)
            tab[wn][ls][0] += 1
            tab[ls][wn][1] += 1
        elif res == "double":
            tab[a][b][2] += 1
            if a != b:
                tab[b][a][2] += 1
        else:
            tab[a][b][3] += 1
            if a != b:
                tab[b][a][3] += 1
    share = {}
    for a in names:
        wsum = sum(tab[a][b][0] for b in names if b != a)
        lsum = sum(tab[a][b][1] for b in names if b != a)
        share[a] = round(wsum / max(1, wsum + lsum), 3)
    n = len(rows)
    dbl = sum(1 for r in rows if r["result"] == "double")
    single = sum(1 for r in rows if r["result"] in ("A", "B"))
    return dict(table={a: {b: "-".join(map(str, tab[a][b])) for b in names} for a in names},
                single_touch_share_vs_others=share, seat_A_B_single=side, outcomes=why,
                double_share_of_touch_phrases=round(dbl / max(1, dbl + single), 3),
                median_phrase_s=float(np.median([r["t"] for r in rows])), n=n)


def show(sm, names, title=""):
    print("==", title)
    print("row vs column: row touches - column touches - doubles - no touch (both seats)")
    print("%10s " % "" + " ".join("%11s" % n for n in names) + "   single-touch share vs others")
    for a in names:
        print("%10s " % a + " ".join("%11s" % sm["table"][a][b] for b in names), "  ", sm["single_touch_share_vs_others"][a])
    print("seat A/B single touches:", sm["seat_A_B_single"], " outcomes:", sm["outcomes"],
          " doubles/(touch phrases):", sm["double_share_of_touch_phrases"], " median phrase s:", sm["median_phrase_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--set", default="", help="fencing constants, e.g. RAIL_VMAX=2.5,F_TOUCH=7.35")
    ap.add_argument("--latency", type=float, default=0.06)
    ap.add_argument("--perturb", type=float, default=0.0, help="also replay every phrase with A's cart moved by this (m)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bouts", action="store_true", help="play full pool bouts instead of single phrases")
    args = ap.parse_args()
    names = [s.name for s in STRATS if not args.only or s.name in args.only.split(",")]
    sets = {k: float(v) for k, v in (kv.split("=") for kv in args.set.split(",") if kv)}
    if args.bouts:
        rows = run_bouts(names, args.seeds, args.jitter, sets, 0.0, args.latency, args.workers)
        sm = summarize_bouts(rows, names)
        show_bouts(sm, names, f"seeds={args.seeds} latency={args.latency} set={sets}")
        out = dict(args=vars(args), names=names, summary=sm,
                   rows=[{k: r[k] for k in ("A", "B", "seed", "winner", "how", "score", "clock", "doubles_awarded")}
                         | {"n_phrases": len(r["phrases"]), "n_doubles": r["phrases"].count("double")} for r in rows])
        if args.perturb:
            rows2 = run_bouts(names, args.seeds, args.jitter, sets, args.perturb, args.latency, args.workers)
            flips = sum(1 for r1, r2 in zip(rows, rows2) if r1["winner"] != r2["winner"])
            print(f"perturbation {args.perturb} m: bout winner changed in {flips}/{len(rows)} bouts")
            out["perturb"] = dict(flips=flips, n=len(rows))
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=1))
        return
    rows = run_all(names, args.seeds, args.jitter, sets, 0.0, args.latency, args.workers)
    sm = summarize(rows, names)
    show(sm, names, f"seeds={args.seeds} latency={args.latency} set={sets}")
    out = dict(args=vars(args), names=names, summary=sm)
    if args.perturb:
        rows2 = run_all(names, args.seeds, args.jitter, sets, args.perturb, args.latency, args.workers)
        flips = sum(1 for r1, r2 in zip(rows, rows2) if r1["result"] != r2["result"])
        print(f"perturbation {args.perturb} m: outcome changed in {flips}/{len(rows)} phrases")
        out["perturb"] = dict(flips=flips, n=len(rows))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
