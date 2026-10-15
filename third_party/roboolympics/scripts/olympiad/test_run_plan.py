"""The run tools' match plan is the sports' declarations (python >= 3.11; no simulator, no Harbor, no model).

For every published sport (sports/*/sport.toml, tasks/) and round 1-3, run_round.matches() and monitor.planned() give a
non-empty plan, and every planned match agrees with the two sources:
  * sport.toml: the task is a [[tasks]] entry of the format's kind, the seed is in [official] seeds, the level is one
    of the format's levels (or [official] levels for a format with a scripted opponent), the slowdown is the task's
    default ([[tasks]] slowdown, else [timing] slowdown) for a real-time sport and absent for a turn-based one;
  * tasks/<id>/: the task directory exists, its docker-compose.yaml defaults RO_SLOWDOWN to the same slowdown and takes
    the seed from the variable the run script sets (RCB_SEED for one seat, DUEL_SEED for several), its task.toml has the
    seats the run script fills, and its own verifier marks the configuration official (official_config = 1);
  * the three rounds of a task and level use three different seeds;
  * the run script command carries the same task, seed, level and slowdown.
A sport with a registered hidden official seed set (core 0.2.14) plans "hidden:I" references of its format x level cell
(I < the cell's count), the verifier accepting the cell's public commitment; planning needs no private set. The plan
is checked twice: without the private sets on this host (RO_SEEDS_DIR = an empty directory: CI, collaborators; only
resolving a "hidden:I" fails) and with a private set present (a synthetic one in a temporary RO_SEEDS_DIR): the plans
must be the same, a set that is not the registered one is refused, and an entry resolves to its own commitment. Once
frozen (core 0.2.19), the verifier of every planned task and level accepts its cell's commitment only: no public
[official] seed and no commitment of another cell is official (frozen_only_hidden).
Also: the default players are players.toml ids with an account lane, and a built-in player plans on lane `builtin`;
a warm-up level (a format level outside [official] levels) is never planned (warmup_levels).

    python3 scripts/olympiad/test_run_plan.py
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import monitor  # noqa: E402
import plan  # noqa: E402
import run_round  # noqa: E402

ROOT = HERE.parents[1]
SCRIPT = {"single": "run_single.sh", "duel": "run_duel.sh", "team": "run_team.sh", "race": "run_race.sh"}


def arg(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def compose_default(task_dir, var):
    m = re.search(rf'{var}: "\$\{{(\w+):-([^}}"]*)\}}"', (task_dir / "environment/docker-compose.yaml").read_text())
    return (m.group(1), m.group(2)) if m else (None, None)


def task_seats(task_dir):
    toml = (task_dir / "task.toml").read_text()
    return re.findall(r'^\[\[seats\]\]\s*\nname = "([^"]+)"', toml, re.M)


def check_match(sport, r, m, d, errors):
    where = f"{sport} round {r} {m['kind']} {m.get('player') or m.get('a') or m.get('players')}"
    t = next((x for x in plan.tasks(sport) if x["id"] == m["task"]), None)
    if t is None:
        errors.append(f"{where}: task {m['task']} is not a [[tasks]] entry of sports/{sport}/sport.toml")
        return
    tdir = ROOT / "tasks" / t["id"]
    if not (tdir / "task.toml").is_file():
        errors.append(f"{where}: no tasks/{t['id']}/task.toml")
        return
    if t["kind"] != m["kind"]:
        errors.append(f"{where}: task {t['id']} is a {t['kind']} format")
    off = d.get("official") or {}
    hs = plan.hidden(sport)
    commit = None
    if hs:                                      # a registered hidden seed set: "hidden:I" of the match's cell
        cell = plan.hidden_cell(t, m.get("level"))
        mm = re.fullmatch(r"hidden:(\d+)", str(m["seed"]))
        if cell is None or not mm or int(mm.group(1)) >= int(hs["counts"][cell]):
            errors.append(f"{where}: seed {m['seed']} is not a hidden reference of cell {cell} ({hs['counts']})")
            return
        commit = plan.hidden_commitment(sport, cell, int(mm.group(1)))
    elif m["seed"] not in off.get("seeds", [1, 2, 3]):
        errors.append(f"{where}: seed {m['seed']} not in [official] seeds {off.get('seeds')}")
    fmt = d["formats"][t["format"]]
    levels = fmt.get("levels") or (off.get("levels") if len(fmt["seats"]) == 1 and fmt.get("opponent") else None)
    if m["kind"] == "single" and (m["level"] not in levels if levels else m["level"] is not None):
        errors.append(f"{where}: level {m['level']}, the format's levels are {levels}")
    rt = d["sport"]["protocol"] == "realtime"
    want = None
    if rt:
        tt = next(x for x in d["tasks"] if plan.generate.task_id(d, x) == t["id"])
        want = tt.get("slowdown", (d.get("timing") or {}).get("slowdown", 20))
    if m["slowdown"] != want:
        errors.append(f"{where}: slowdown {m['slowdown']}, sport.toml says {want}")
    var, default = compose_default(tdir, "RO_SLOWDOWN")
    if rt and (var != "RCB_SLOWDOWN" or float(default) != float(want)):
        errors.append(f"{where}: tasks/{t['id']} defaults RO_SLOWDOWN to {var}:-{default}, the plan to {want}")
    if not rt and var:
        errors.append(f"{where}: a turn-based task with RO_SLOWDOWN")
    seed_var = compose_default(tdir, "RO_SEED")[0]
    if seed_var != ("RCB_SEED" if m["kind"] == "single" else "DUEL_SEED"):
        errors.append(f"{where}: tasks/{t['id']} takes the seed from {seed_var}, {SCRIPT[m['kind']]} sets another")
    if task_seats(tdir) != list(fmt["seats"]):
        errors.append(f"{where}: tasks/{t['id']} seats {task_seats(tdir)} != the format's {fmt['seats']}")
    seed = -1 if commit else m["seed"]
    if not plan.official(t, seed, m.get("level"), m["slowdown"], commit=commit) and not any(
            plan.official(t, seed, lv, m["slowdown"], commit=commit) for lv in off.get("levels", [])):
        errors.append(f"{where}: the task's verifier does not mark seed {m['seed']} level {m.get('level')} official")
    cmd = m["cmd"]
    if Path(cmd[0]).name != SCRIPT[m["kind"]] or not (ROOT / cmd[0]).is_file():
        errors.append(f"{where}: command {cmd[0]}")
    if arg(cmd, "--task") != t["id"] or arg(cmd, "--seed") != str(m["seed"]) or arg(cmd, "--tag") != f"r{r}":
        errors.append(f"{where}: command {' '.join(cmd)}")
    if (arg(cmd, "--level") or None) != (str(m["level"]) if m.get("level") is not None else None):
        errors.append(f"{where}: command level {arg(cmd, '--level')} != {m.get('level')}")
    if (float(arg(cmd, "--slowdown")) if arg(cmd, "--slowdown") else None) != m["slowdown"]:
        errors.append(f"{where}: command slowdown {arg(cmd, '--slowdown')} != {m['slowdown']}")


def plan_all():
    """{sport: {round: matches}} of every published sport (what run_round plans)."""
    players = run_round.PLAYERS
    out = {}
    for sport in plan.sports():
        out[sport] = {r: run_round.matches(r, sport, None, players, players[:len(plan.task(sport, "race")["seats"])]
                                           if plan.task(sport, "race") else None) for r in plan.ROUNDS}
    return out


def with_private_set(sport, before, errors):
    """The two hosts: without the private set (CI, a collaborator) planning works and only resolving a "hidden:I"
    fails (FileNotFoundError); with a private set present (a synthetic one in a temporary RO_SEEDS_DIR) the plan is
    the same (planning never reads it), a set that is not the registered one is refused, and a set's entry resolves to
    its own commitment (core/seeds.py resolve, which tools/seeds.py resolve and the run scripts use)."""
    import seeds as seedtool                    # tools/seeds.py (plan put tools/ on sys.path)
    from core import seeds as seedcore
    reg = seedtool.registration(sport)
    cell = sorted(reg["cells"])[0]
    try:
        plan.hidden_entry(sport, cell, 0)
        errors.append(f"{sport}: hidden_entry resolved with no private set on this host")
    except FileNotFoundError:
        pass
    priv = seedcore.new(sport, {c: len(v) for c, v in reg["cells"].items()})
    pub = seedcore.public(priv)
    old_dir = os.environ.get("RO_SEEDS_DIR")
    with tempfile.TemporaryDirectory(prefix="ro-seeds-") as tmp:
        os.chmod(tmp, 0o700)
        (Path(tmp) / f"{sport}.hidden_seeds.json").write_text(json.dumps(priv))
        os.environ["RO_SEEDS_DIR"] = tmp
        try:
            if plan_all().get(sport) != before.get(sport):
                errors.append(f"{sport}: the plan differs when a private seed set is on this host")
            try:
                plan.hidden_entry(sport, cell, 0)
                errors.append(f"{sport}: hidden_entry accepted a private set that is not the registered one")
            except ValueError:
                pass
            seed, salt = seedcore.resolve(seedtool.load_private(sport), cell, 0, {"sha256": pub["sha256"]})
            if seedcore.commitment(sport, seed, salt) != pub["cells"][cell][0]:
                errors.append(f"{sport}: a resolved entry does not match its commitment")
        finally:
            if old_dir is None:
                os.environ.pop("RO_SEEDS_DIR", None)
            else:
                os.environ["RO_SEEDS_DIR"] = old_dir


def frozen_only_hidden(sport, errors):
    """Core 0.2.19: once a sport registered its hidden seed set, the task's own verifier marks as official only a
    commitment of the match's cell: never a public [official] seed (the sport's legacy rule cannot make it official)
    and never a commitment of another cell; the cell's own commitment is official (also checked by check_match)."""
    hs = plan.hidden(sport)
    off = plan.decl(sport).get("official") or {}
    import seeds as seedtool                                        # tools/seeds.py (as plan.hidden_commitment)
    cells = seedtool.registration(sport)["cells"]
    for tid, lv, slow in sorted({(m["task"], m.get("level"), m["slowdown"])
                                 for m in run_round.matches(1, sport, None, run_round.PLAYERS)}, key=str):
        t = next(x for x in plan.tasks(sport) if x["id"] == tid)
        where = f"{sport} {t['id']} level {lv} slowdown {slow}"
        cell = plan.hidden_cell(t, lv)
        if not plan.official(t, -1, lv, slow, commit=cells[cell][0]):
            errors.append(f"{where}: a hidden seed of its cell {cell} is not official")
        for s in off.get("seeds", []):
            for first in ("A", "B"):
                if plan.official(t, int(s), lv, slow, first=first):
                    errors.append(f"{where}: public seed {s} (first {first}) is official although {sport} is frozen "
                                  f"({sorted(hs['counts'])})")
        for other in sorted(set(cells) - {cell}):
            if plan.official(t, -1, lv, slow, commit=cells[other][0]):
                errors.append(f"{where}: a hidden seed of cell {other} is official in cell {cell}")


def warmup_levels(errors):
    """A warm-up level (a format level outside [official] levels, soccer L1) stays out of the official plan: plan.py
    levels_of, run_round.matches and tools/seeds.py cells skip it, with the same task, its other levels and seeds.
    Played on a published sport whose sport.toml is made to declare its lowest level a warm-up one (the task's own
    verifier still marks that level official, so only the declaration keeps it out)."""
    import copy
    import seeds as seedtool                    # tools/seeds.py
    fmt = {"seats": ["A"], "opponent": True, "levels": [1, 2, 3]}
    if plan.levels_of({"official": {"levels": [2, 3]}}, fmt) != [2, 3]:
        errors.append(f"warm-up: levels_of {plan.levels_of({'official': {'levels': [2, 3]}}, fmt)} != [2, 3]")
    if plan.levels_of({}, fmt) != [1, 2, 3] or plan.levels_of({"official": {"levels": [2]}},
                                                             {"seats": ["A"], "opponent": True}) != [2]:
        errors.append("warm-up: levels_of without [official] levels / without format levels")
    want = ["match/L2", "match/L3", "duel"]
    got = seedtool.cells({"formats": {"match": fmt, "duel": {"seats": ["A", "B"]}}, "official": {"levels": [2, 3]}})
    if got != want:
        errors.append(f"warm-up: tools/seeds.py cells {got} != {want}")
    sport = next((s for s in plan.sports() if (plan.task(s, "single") or {}).get("levels", [None])[0] is not None
                  and len(plan.task(s, "single")["levels"]) >= 2 and not plan.hidden(s)), None)
    if sport is None:
        return
    real = plan.decl(sport)
    t = plan.task(sport, "single")
    full = list(t["levels"])
    d = copy.deepcopy(real)
    d["formats"][t["format"]]["levels"] = full
    d.setdefault("official", {})["levels"] = full[1:]          # the lowest level becomes a warm-up level
    before = run_round.matches(1, sport, None, ["nop"])
    old = plan.decl
    plan.decl = lambda s: d if s == sport else old(s)
    try:
        after = run_round.matches(1, sport, None, ["nop"])
    finally:
        plan.decl = old
    lv = sorted({m["level"] for m in after if m["kind"] == "single"})
    if lv != full[1:]:
        errors.append(f"warm-up: {sport} with [official] levels {full[1:]} plans levels {lv}")
    if [m for m in before if m.get("level") != full[0]] != after:
        errors.append(f"warm-up: {sport}: the plan of the official levels changed when level {full[0]} became a warm-up")


def main():
    errors, rows = [], []
    empty = tempfile.mkdtemp(prefix="ro-seeds-none-")
    os.environ["RO_SEEDS_DIR"] = empty            # no private set on this host (CI, a collaborator's machine)
    sports = plan.sports()
    assert sports, "no published sport"
    declared = sorted(p.parent.name for p in (ROOT / "sports").glob("*/sport.toml")
                      if plan.decl(p.parent.name)["sport"].get("publish"))
    assert sports == declared, (sports, declared)
    players = run_round.PLAYERS
    assert players and all(p in {x["id"] for x in run_round.ROSTER} for p in players), players
    for sport in sports:
        d = plan.decl(sport)
        seeds = {}
        for r in plan.ROUNDS:
            ms = run_round.matches(r, sport, None, players, players[:len(plan.task(sport, "race")["seats"])]
                                   if plan.task(sport, "race") else None)
            mon = monitor.planned(sport, r)
            if not ms or not mon:
                errors.append(f"{sport} round {r}: empty plan (run_round {len(ms)}, monitor {len(mon)})")
            if any(x not in ms for x in mon):
                errors.append(f"{sport} round {r}: monitor plans a match run_round does not")
            for m in ms:
                check_match(sport, r, m, d, errors)
                seeds.setdefault((m["task"], m.get("level")), {})[r] = m["seed"]
        for (tid, lv), by_round in seeds.items():
            if len(set(by_round.values())) != len(plan.ROUNDS):
                errors.append(f"{tid} level {lv}: rounds 1-3 seeds {by_round} are not three different seeds")
        rows.append(f"{sport:12s} " + ", ".join(f"{tid.split('-', 1)[1]} L{lv if lv is not None else '-'} "
                                                f"{[by_round[r] for r in plan.ROUNDS]}"
                                                for (tid, lv), by_round in sorted(seeds.items(), key=str)))
    hidden_sports = [s for s in sports if plan.hidden(s)]
    if hidden_sports:
        before = plan_all()
        for sport in hidden_sports:
            with_private_set(sport, before, errors)
            frozen_only_hidden(sport, errors)
    warmup_levels(errors)
    # built-in players need no account lane
    oracle = run_round.matches(1, "darts", None, ["oracle"])
    assert oracle and all(m["lane"] == "builtin" for m in oracle), oracle
    assert run_round.matches(1, "darts", None, ["oracle", "oracle"])[-1]["kind"] == "duel"
    for e in errors:
        print("FAIL", e)
    if errors:
        sys.exit(1)
    print("\n".join(rows))
    print(f"PASS: the round 1-3 plans of {len(sports)} sports agree with sport.toml and tasks/")


if __name__ == "__main__":
    main()
