#!/usr/bin/env python3
"""The Olympiad run tools' view of the sports, read from the one source (python >= 3.11, standard library only):

  sports/<sport>/sport.toml   [sport] protocol / publish, [[tasks]] (id, format, action_level, play, slowdown),
                              [formats] (seats, sides, opponent, levels), [official] (seeds, levels, hidden_seeds),
                              [timing] slowdown
  tasks/<id>/                 the generated task (tools/generate.py): it must exist, and its own verifier
                              (tests/grade.py + tests/sport_grade.py + tests/contract.json) decides which seeds are
                              official for a format, level and slowdown

Nothing here keeps a copy of a sport's configuration. run_round.py, monitor.py, live_board.py and run_single.sh read
it through this module; tools/generate.py names the tasks (task_id).

The plan of round r (rounds 1-3): for every format kind of a sport (single = one seat, duel = two, team = two sides of
several seats, race = more than two sides), the sport's first [[tasks]] entry of that kind (or of --action-level) and,
for every level of the format, the r-th seed of [official] seeds (in sport.toml order; with a registered hidden seed set, core 0.2.14, the reference
"hidden:<r-1>" of the format x level's own cell, which the run scripts resolve on the run host from the private set:
tools/seeds.py) that the task's verifier marks
official_config = 1 at that level, slowdown and first player A (seeds()). Levels: the format's own `levels`, else [official]
levels when the format has a scripted opponent, else none (a solo scoring format); only those in [official] levels (a
format level outside them is a warm-up level: playable, never scheduled). Slowdown (real-time sports only):
the task's default ([[tasks]] slowdown, else [timing] slowdown, else 20, as tools/generate.py writes it into the task).

    python3 scripts/olympiad/plan.py show [SPORT ...]               the round 1-3 seeds of every task and level
    python3 scripts/olympiad/plan.py seed TASK [--level L] [--slowdown S] [--round R]   one seed (run_single.sh)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))         # core/seeds.py (last: never shadows a caller's modules)
try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:          # python 3.10 (the run hosts' system python): tomli reads the same TOML
    try:
        import tomli
    except ModuleNotFoundError:
        sys.exit("scripts/olympiad/plan.py needs python >= 3.11 or the tomli package")
    sys.modules["tomllib"] = tomli
import generate  # noqa: E402  (task ids; stdlib only)
from core import seeds as seedcore  # noqa: E402  (core/seeds.py: stdlib only)
import sportlib  # noqa: E402

ROUNDS = (1, 2, 3)
KINDS = ("single", "duel", "team", "race")


_DECL: dict = {}


def decl(sport: str) -> dict:
    """sport.toml, parsed again only when the file changed (the monitor asks on every page)."""
    mtime = (sportlib.SPORTS / sport / "sport.toml").stat().st_mtime_ns
    if _DECL.get(sport, (None,))[0] != mtime:
        _DECL[sport] = (mtime, sportlib.decl(sport))
    return _DECL[sport][1]


def sports() -> list[str]:
    """The sports whose tasks are published in tasks/: status "sport" and publish = true (tools/generate.py
    generated_sports and out_root, read through the cache)."""
    return [s for s in sportlib.names()
            if decl(s)["sport"].get("status", "sport") == "sport" and decl(s)["sport"].get("publish")]


def realtime(sport: str) -> bool:
    return decl(sport)["sport"].get("protocol") == "realtime"


def kind_of(fmt: dict) -> str:
    """single (one seat: alone or against a scripted opponent), duel (two seats), team (two sides of several seats
    each), race (more than two sides)."""
    seats = fmt["seats"]
    if len(seats) == 1:
        return "single"
    sides = set((fmt.get("sides") or {s: s for s in seats}).values())
    if len(sides) == 2:
        return "duel" if len(seats) == 2 else "team"
    return "race"


def levels_of(d: dict, fmt: dict) -> list:
    """The official opponent / difficulty levels of a single-seat format ([None] = no levels): the format's own
    `levels`, else [official] levels when it has a scripted opponent, in both cases only those in [official] levels
    when the sport declares them. A format level outside [official] levels is a warm-up level (soccer L1): playable
    (RCB_LEVEL), but it has no official seeds and is never scheduled (tools/seeds.py cells and tools/check.py
    opponent_levels agree)."""
    if len(fmt["seats"]) != 1:
        return [None]
    off = (d.get("official") or {}).get("levels")
    lv = fmt.get("levels") or (off if fmt.get("opponent") else None)
    if lv and off:
        lv = [x for x in lv if x in off]
    return list(lv) if lv else [None]


def tasks(sport: str) -> list[dict]:
    """The sport's [[tasks]] in sport.toml order, with what the run tools need."""
    d = decl(sport)
    out = []
    for t in d.get("tasks", []):
        tid = generate.task_id(d, t)
        fmt = d["formats"][t["format"]]
        out.append(dict(sport=sport, id=tid, format=t["format"], kind=kind_of(fmt), seats=list(fmt["seats"]),
                        sides=fmt.get("sides"), opponent=bool(fmt.get("opponent")),
                        action_level=t.get("action_level", "high"), play=t.get("play", "program"),
                        levels=levels_of(d, fmt),
                        slowdown=(t.get("slowdown", (d.get("timing") or {}).get("slowdown", 20))
                                  if realtime(sport) else None),
                        dir=REPO / ("tasks" if d["sport"].get("publish") else "build/tasks") / tid))
    return out


def task(sport: str, kind: str, action_level: str | None = None) -> dict | None:
    """The sport's first task of this kind (and action level), None if it has none."""
    return next((t for t in tasks(sport) if t["kind"] == kind and action_level in (None, t["action_level"])), None)


def task_by_id(tid: str) -> dict:
    for s in sports():
        for t in tasks(s):
            if t["id"] == tid:
                return t
    raise KeyError(f"no task {tid} in any sport.toml [[tasks]]")


# ---- official configurations: the task's own verifier ----------------------------------------------------------
@lru_cache(maxsize=None)
def _verifier(task_dir: str):
    """(grade module, sport evaluate, contract) of a generated task's tests/, loaded with an empty RCB_DATA."""
    tests = Path(task_dir) / "tests"
    if not (tests / "grade.py").is_file():
        raise FileNotFoundError(f"{tests}/grade.py: the task is not generated (tools/generate.py)")
    old = os.environ.get("RCB_DATA")
    os.environ["RCB_DATA"] = tempfile.mkdtemp(prefix="ro-plan-")      # no match data: the graders read nothing
    try:
        mods = []
        for name in ("grade", "sport_grade"):
            spec = importlib.util.spec_from_file_location(f"ro_plan_{name}_{abs(hash(task_dir))}", tests / f"{name}.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            mods.append(m)
    finally:
        if old is None:
            os.environ.pop("RCB_DATA", None)
        else:
            os.environ["RCB_DATA"] = old
    return mods[0], mods[1].evaluate, json.loads((tests / "contract.json").read_text())


def official(t: dict, seed: int, level=None, slowdown=None, first: str = "A", salt: str | None = None,
             commit: str | None = None) -> bool:
    """Whether the task's verifier marks a valid match at this seed, level, slowdown and first player as an official
    configuration (reward official_config = 1). The final.json is a minimal valid one with no play in it; with `salt`
    (a hidden official seed) it records what the sim would: seed -1 and the seed's commitment; with `commit` the same
    from a public commitment alone (no private set needed: the verifier only ever sees the commitment)."""
    if commit is None and salt:
        commit = seedcore.commitment(t["sport"], seed, salt)
    grade, evaluate, contract = _verifier(str(t["dir"]))
    seats = t["seats"]
    side = t["sides"] or {s: s for s in seats}
    d = dict(cfg=dict(seats=seats, level=level), seed=seed, slowdown=slowdown, first=first, play=t["play"],
             format=t["format"], sides=t["sides"], action_level=t["action_level"],
             sport_fields=dict(level=level, cars=len(seats)), level=level,
             solo=len(seats) == 1 and not t["opponent"], reason="match_over", errors=[], valid_trial=True,
             finalized=True, deterministic=True, state_sha="-", commands_sha="-", source_sha="-", steps=0,
             scores={x: 0 for x in set(side.values())}, match_winner=side[seats[0]], ranking=sorted(set(side.values())),
             meter_error={}, proxy_error={}, transient_api_errors={}, unrecovered_api_errors={},
             player_modes={s: "model" for s in seats}, joined={s: True for s in seats}, started={s: True for s in seats},
             seats={}, clock_s=0, clock_used_s={}, elapsed_s=0.0, sim_lag={},
             seed_proof={"commitment": commit} if commit else None)
    if commit:
        d["seed"] = d["cfg"]["seed"] = -1
    r = dict(state_sha="-", scores=d["scores"], winner=d["match_winner"], commands_sha="-", source_sha="-", k=0,
             seed_commitment=(d["seed_proof"] or {}).get("commitment"))
    with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
        return grade.grade(d, r, contract, evaluate).get("official_config") == 1


def hidden(sport: str) -> dict | None:
    """The sport's registered hidden official seed set (sport.toml [official.hidden_seeds]: sha256, counts per cell)."""
    return (decl(sport).get("official") or {}).get("hidden_seeds")


def hidden_cell(t: dict, level=None) -> str | None:
    """The cell (format x level) of the task's matches at a level in the sport's hidden seed set (core/seeds.py)."""
    return seedcore.cell_of(t["format"], level, hidden(t["sport"])["counts"])


def hidden_commitment(sport: str, cell: str, i: int) -> str:
    """The public commitment of entry i of a cell (sports/<sport>/seed_commitments.json, checked against sport.toml):
    what planning, listing and the tests use, on any host."""
    import seeds as seedtool                     # tools/seeds.py
    return seedtool.registration(sport)["cells"][cell][i]


def hidden_entry(sport: str, cell: str, i: int) -> tuple[int, str]:
    """(seed, salt) of entry i of a cell of the sport's private hidden seed set (tools/seeds.py; FileNotFoundError
    when the set is not on this host). Only launching an official match needs it (the run scripts resolve
    "hidden:I" with tools/seeds.py resolve); planning does not."""
    import seeds as seedtool                     # tools/seeds.py
    reg = seedtool.registration(sport)
    return seedcore.resolve(seedtool.load_private(sport), cell, i, reg)


def preview(t: dict) -> bool:
    """A format of a frozen sport (registered hidden seed set) without a cell at any of its levels: playable, never
    official (its verifier never marks a match official), never scheduled (e.g. soccer team, 2026-10-13)."""
    if not hidden(t["sport"]):
        return False
    return all(hidden_cell(t, lv) is None for lv in (t.get("levels") or [None]))


def seeds(t: dict, level=None, slowdown=None) -> list:
    """The official seeds of a task at a level and slowdown, in sport.toml [official] seeds order. A format without
    levels may still be played at a fixed level the sim reports (drone: the duel always flies level 2): when no seed
    is official without a level, a seed counts if it is official at one of the [official] levels.
    A sport with a registered hidden seed set (core 0.2.14) plays only those, from the cell of the format and level:
    the references "hidden:0" .. "hidden:<n-1>" (the run scripts resolve them on the run host, tools/seeds.py
    resolve), provided the task's verifier accepts the cell's first entry at this level and slowdown (checked with its
    public commitment: any host, no private set)."""
    off = decl(t["sport"]).get("official") or {}
    hs = hidden(t["sport"])
    if hs:
        cell = hidden_cell(t, level)
        if cell is None:
            return []
        commit = hidden_commitment(t["sport"], cell, 0)
        ok = official(t, -1, level, slowdown, commit=commit) or (level is None and any(
            official(t, -1, lv, slowdown, commit=commit) for lv in off.get("levels", [])))
        return [f"hidden:{i}" for i in range(int(hs["counts"][cell]))] if ok else []
    cand = [int(s) for s in off.get("seeds", [1, 2, 3])]
    out = [s for s in cand if official(t, s, level, slowdown)]
    if not out and level is None:
        out = [s for s in cand if any(official(t, s, lv, slowdown) for lv in off.get("levels", []))]
    return out


def round_seed(t: dict, r: int, level=None, slowdown=None):
    ss = seeds(t, level, slowdown)
    if len(ss) < r:
        raise ValueError(f"{t['id']}: {len(ss)} official seeds at level {level}, slowdown {slowdown} "
                         f"({ss}); round {r} needs {r}")
    return ss[r - 1]


# ---- command line ---------------------------------------------------------------------------------------------
def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("sport", nargs="*")
    sd = sub.add_parser("seed")
    sd.add_argument("task")
    sd.add_argument("--level", type=int)
    sd.add_argument("--slowdown", type=float)
    sd.add_argument("--round", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "seed":
        t = task_by_id(Path(a.task).name)
        slow = a.slowdown if a.slowdown is not None else t["slowdown"]
        print(round_seed(t, a.round, a.level, slow))
        return 0
    for s in a.sport or sports():
        for t in tasks(s):
            for lv in t["levels"]:
                ss = seeds(t, lv, t["slowdown"])
                print(f"{t['id']:55s} {t['kind']:6s} L{lv if lv is not None else '-'} "
                      f"x{t['slowdown'] if t['slowdown'] is not None else '-'}  rounds 1-3: {ss[:3]}  official: {ss}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
