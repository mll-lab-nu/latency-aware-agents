#!/usr/bin/env python3
"""tools/check SPORT: every check a sport must pass before its pull request (docs/framework_zh.md §8 step ④, §9).

    tools/check SPORT [--stages 1,2,3,4,5,6] [--quick] [--all-tasks] [--harbor] [--external DIR]

  1 contract   sport.toml: schema, rule-book references and deviations, formats, modes, observation whitelist,
               tiebreaks, exploit list (each names a test that exists), players, fairness (only with a format of two or more sides), tasks; the sport's files
               (plugin, grader, tests; no module that would shadow the framework; a camera-only reference player);
               the two modes when declared ([modes.high] / [modes.low], skilleval/declare.py: skill API and contract,
               toolkit manifest and its sha256, seat files, deliverable, evaluator hooks, references, hidden cases);
               core 0.2.14: [score] (the continuous score; WARN when missing, FAIL when missing in a sport with hidden
               official seeds) and [official.hidden_seeds] = sports/<sport>/seed_commitments.json
  2 tests      the generic test suite in the sport's image (core/testkit.py: determinism + fresh-process replay,
               no draw, mirror symmetry, whitelist enforcement, observe is pure, malformed actions refused, frame
               budget of the task renderer) and the sport's exploit tests
  3 golden     the sport's golden matches replay identically in a fresh process (tools/golden.py); for a migrated
               sport also the converted legacy records of the golden-data store (--external or RO_GOLDEN_DATA)
  4 fairness   built-in players on every seat, both first orders, N seeds: the first-mover and the seat advantage
               with their 95% confidence intervals; FAIL if an interval lies entirely beyond +-5 pp, WARN if it is
               wider than +-5 pp (too few matches); N is reported
  5 live       the generated tasks (tools/generate.py, into a temporary build dir) run offline (tools/live.py):
               nop on every seat scores 0; the reference player on every seat scores success = 1 in a duel (the
               match is playable) and a solo format (the success line), and plays a valid match to its end against a
               scripted opponent or with graded levels (end to end, no win required; --harbor: Harbor). Such a format
               also gets the reference check of 2026-10-11 (docs/screening/README.md): the sport's [reference] match
               hook with truth perception, >= 20 seeds per opponent level: the majority won at the lowest level, wins
               not rising with the level (opponent calibration); camera-only can win: the camera reference (the same
               hook, perception "camera", [reference] camera_seeds) wins the lowest level's majority, or a model's
               camera-only win is registered in [[reference.camera_wins]]; the camera reference's playability (the
               hook's report counters, e.g. legal returns) is reported with no numeric gate. Without the hook: WARN,
               FAIL once the sport is frozen ([official.hidden_seeds]). --jobs N: containers at once
  6 modes      only with [modes.low] (docs/framework_zh.md §6.3.7, §9 robot and mode rules 4, 5, 10): the post-match
               evaluator (tools/skill_eval.py, isolated container) on our implementation (every hidden request
               feasible, the contract's request schema = the high level's act schema), on the reference skill (API
               test >= reference_min, deterministic) and on the boundary baseline (API test <= boundary_max), and
               the isolation probe (the skill reads nothing of the simulator, writes nothing, has no network)

Needs python >= 3.11 and docker. Exit 0 iff every stage run passed.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capabilities  # noqa: E402
import generate  # noqa: E402
import golden  # noqa: E402
import live  # noqa: E402
import seeds as seedtool  # noqa: E402  (tools/seeds.py: hidden official seeds; puts the repository on sys.path)
import skill_eval  # noqa: E402
import sportlib  # noqa: E402
from core import score as score_mod  # noqa: E402  (core/score.py: standard library only)

sys.path.insert(0, str(sportlib.REPO / "skilleval"))
import declare as skilleval_declare  # noqa: E402  (skilleval/declare.py: standard library only)

FRAMEWORK_NAMES = {"core", "renderers", "players", "tools", "sports", "metering", "seasons", "skilleval", "bodies",
                   "mujoco", "numpy", "PIL", "scipy", "builtins"}
CAMERA_ONLY_IMPORTS = {"numpy", "PIL", "scipy", "math", "base64", "io", "json", "os", "sys", "time", "collections",
                       "itertools", "functools", "dataclasses", "typing", "random", "statistics", "__future__"}


def registry() -> dict:
    """renderers.REGISTRY, read without importing the renderers (no numpy on the host)."""
    tree = ast.parse((sportlib.REPO / "renderers" / "__init__.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "REGISTRY" for t in n.targets):
            return ast.literal_eval(n.value)
    return {}


def presentation_problems(pdir: Path) -> tuple[bool, str]:
    """sports/<s>/presentation/ (renderers/presentation): presentation.toml with version >= 1 and robots whose models
    the asset manifest lists; data files only (no code). Its kinematics are checked by renderers/test_renderers.py."""
    import tomllib
    try:
        p = tomllib.loads((pdir / "presentation.toml").read_text()).get("presentation") or {}
    except Exception as e:  # noqa: BLE001
        return False, f"presentation.toml: {e}"
    models = {}
    for m in sportlib.REPO.glob("renderers/presentation/*.json"):
        models.update(json.loads(m.read_text()).get("models", {}))
    bad = [] if int(p.get("version", 0)) >= 1 and p.get("robot") else ["version >= 1 and [[presentation.robot]]"]
    bad += [f"model {r.get('model')!r} not in the manifest nor a file in presentation/" for r in p.get("robot", [])
            if r.get("model") not in models and not (str(r.get("model", "")).startswith("./")
                                                     and (pdir / r["model"]).is_file())]
    bad += [f"scene file {f} missing" for f in p.get("scenes", []) if not (pdir.parent / f).is_file()]
    bad += [f"{r.get('model')}: bodies and body" for r in p.get("robot", []) if not (r.get("bodies") and r.get("body"))]
    bad += [f"code file {f.name}" for f in pdir.rglob("*") if f.suffix in (".py", ".sh", ".so")]
    bad += [] if "presentation" in sportlib.NOT_IN_SIM else ["presentation is not in tools/sportlib.NOT_IN_SIM"]
    players = p.get("players")
    if players is not None:                      # the players' images use it: the meshes must exist and be pinned
        lods = json.loads((sportlib.REPO / "renderers" / "presentation" / "lod.json").read_text()).get("models", {})
        bad += [f"players = {players!r}: no level {players} of {r['model']} in renderers/presentation/lod.json"
                for r in p.get("robot", []) if players != "full" and not str(r.get("model", "")).startswith("./")
                and players not in lods.get(r.get("model"), {})]
        bad += ["presentation/assets/ is written by the generator only"] if (pdir / "assets").exists() else []
    what = ", ".join(f"{r['model']} on {r['bodies']}" for r in p.get("robot", []))
    if players is not None:
        what += f"; the players' images too ({players} meshes, bundled into the simulator)"
    return not bad, "; ".join(bad) or what


class Report:
    def __init__(self):
        self.rows, self.warnings = [], []

    def add(self, stage, name, ok, detail="", warn=False):
        self.rows.append((stage, name, bool(ok), str(detail)))
        if warn and ok:
            self.warnings.append(name)
        tag = ("WARN" if warn else "ok  ") if ok else "FAIL"
        print(f"  {tag} [{stage}] {name}" + (f": {detail}" if detail else ""), flush=True)
        return ok

    def ok(self):
        return all(r[2] for r in self.rows)


# ------------------------------------------------------------------------------------------------ stage 1
def contract(sport: str, R: Report):
    S = "1 contract"
    src = sportlib.SPORTS / sport
    try:
        d = sportlib.decl(sport)
    except Exception as e:  # noqa: BLE001
        return R.add(S, "sport.toml parses", False, e)
    sp = d.get("sport", {})
    R.add(S, "[sport] name = directory", sp.get("name") == sport, sp.get("name"))
    for k in ("protocol", "revision", "image", "status"):
        R.add(S, f"[sport] {k}", bool(sp.get(k)), sp.get(k))
    R.add(S, "[sport] protocol", sp.get("protocol") in ("realtime", "turn"), sp.get("protocol"))
    R.add(S, "[sport] status", sp.get("status") == "sport", sp.get("status"))   # "wrapper" (stage-0 legacy shells) retired 2026-10-10
    for f in ("plugin.py", "grade.py"):
        R.add(S, f"file {f}", (src / f).exists())
    shadow = sorted(p.stem for p in src.glob("*.py") if p.stem in FRAMEWORK_NAMES) + \
        sorted(p.name for p in src.iterdir() if p.is_dir() and p.name in FRAMEWORK_NAMES)
    R.add(S, "no module shadows the framework", not shadow, shadow or "")
    cap = capabilities.problems(d)
    R.add(S, "[capabilities] a1..b3 in {2,1,0}, physics from the vocabulary", not cap,
          "; ".join(cap) or {k: d["capabilities"].get(k) for k in [*capabilities.CAPABILITIES, "physics"]})
    wl = (d.get("observation") or {}).get("fields")
    R.add(S, "[observation] fields (whitelist)", isinstance(wl, list) and wl and all(isinstance(x, str) for x in wl),
          f"{len(wl or [])} fields")
    for k in ("title", "summary", "physics"):
        R.add(S, f"[sport] {k}", bool(sp.get(k)))
    rb = d.get("rulebook", {})
    R.add(S, "[rulebook] name, version, url", all(rb.get(k) for k in ("name", "version", "url")), rb.get("name"))
    devs = d.get("deviations", [])
    ids = [x.get("id") for x in devs]
    R.add(S, "[[deviations]] id, rule, text; ids unique",
          all(x.get("id") and x.get("rule") and x.get("text") for x in devs) and len(ids) == len(set(ids)),
          f"{len(devs)} deviation(s)")
    fm = d.get("formats", {})
    R.add(S, "[formats] declared", bool(fm), sorted(fm))
    multi = False
    for name, f in fm.items():
        seats = f.get("seats") or []
        R.add(S, f"[formats.{name}] seats", bool(seats) and len(set(seats)) == len(seats), seats)
        sides = set((f.get("sides") or {s: s for s in seats}).values())
        multi |= len(sides) > 1
        if len(sides) > 1:
            R.add(S, f"[formats.{name}] first orders", bool(f.get("first")), f.get("first"))
        elif not f.get("opponent", False):
            line = f.get("success", {})
            R.add(S, f"[formats.{name}] success line (solo)", "score_at_least" in line or "score_at_most" in line, line)
    md = d.get("modes", {})
    lv, pl = md.get("action_levels", []), md.get("plays", [])
    R.add(S, "[modes] action_levels include low", "low" in lv and set(lv) <= {"high", "low"}, lv)
    R.add(S, "[modes] plays", bool(pl) and set(pl) <= {"program"}, pl)   # direct play: archive/direct-play
    ob = d.get("observation", {})
    reg = registry()
    rn = ob.get("renderer", "mujoco")
    R.add(S, "[observation] renderer is registered and real time", rn in reg and reg[rn][1] is True, rn)
    R.add(S, "[observation] frame_budget_ms", isinstance(ob.get("frame_budget_ms"), (int, float)),
          ob.get("frame_budget_ms"))
    plug = src / "plugin.py"
    hook = plug.exists() and re.search(r"^\s+def reply_views\(", plug.read_text(), re.M) is not None
    R.add(S, "[observation] reply_images declared iff the plugin has reply_views (core 0.2.8)",
          bool(hook) == bool(isinstance(ob.get("reply_images"), str) and ob.get("reply_images").strip()),
          ob.get("reply_images") or ("plugin defines reply_views" if hook else "no reply images"))
    hs = (d.get("official") or {}).get("hidden_seeds")
    if hs:                                  # core 0.2.14: a frozen sport's hidden official seed set (tools/seeds.py)
        try:
            reg = seedtool.registration(sport)
            R.add(S, "[official.hidden_seeds] = the public commitments (core/seeds.py)", True,
                  f"{reg['counts']}, sha256 {reg['sha256'][:16]}...")
        except Exception as e:  # noqa: BLE001
            R.add(S, "[official.hidden_seeds] = the public commitments (core/seeds.py)", False, e)
    has_cont = plug.exists() and re.search(r"^\s+def continuous_score\(", plug.read_text(), re.M) is not None
    if d.get("score"):                      # core 0.2.14: the continuous score (core/score.py)
        bad = score_mod.problems(d, has_cont)
        R.add(S, "[score] continuous score: name, text, from, forfeit (core 0.2.14)", not bad,
              "; ".join(bad) or f"{d['score']['name']} (from {d['score']['from']}, forfeit {d['score']['forfeit']})")
    else:
        R.add(S, "[score] continuous score declared (core 0.2.14)", not hs,
              "missing: required once hidden official seeds are registered (frozen)" if hs else
              "not declared: the leaderboards get no continuous score for this sport", warn=not hs)
    for name, ok, det, warn in reaction_delay_checks(d, src):
        R.add(S, name, ok, det, warn=warn)
    tb = d.get("tiebreaks", {})
    if multi:
        R.add(S, "[tiebreaks] order and text (no draws)", bool(tb.get("order")) and bool(tb.get("text")), tb.get("order"))
        R.add(S, "[tiebreaks] last resort in the prompt", tb.get("last_resort_in_prompt") is True)
    tests = "\n".join(p.read_text() for p in sorted((src / "tests").glob("test_*.py"))) if (src / "tests").is_dir() else ""
    ex = d.get("exploits", [])
    R.add(S, "[[exploits]] declared", bool(ex), f"{len(ex)} exploit(s)")
    for e in ex:
        R.add(S, f"exploit {e.get('id')} has its test", bool(e.get("text")) and
              re.search(rf"^def {re.escape(e.get('test', '?'))}\(", tests, re.M) is not None, e.get("test"))
    pls = d.get("players", {})
    ref = pls.get("reference_class", "")
    mod_file = src / (ref.split(":")[0] + ".py")
    R.add(S, "[players] reference + reference_class", bool(pls.get("reference")) and mod_file.exists(), ref)
    if mod_file.exists():
        tree = ast.parse(mod_file.read_text())
        imps = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imps |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        bad = sorted(imps - CAMERA_ONLY_IMPORTS)
        R.add(S, "reference player imports no simulator or framework (camera-only)", not bad, bad or sorted(imps))
    fa = d.get("fairness", {})
    if multi or fa:
        R.add(S, "[fairness] policy, matches, max <= 5 pp, mirror",
              fa.get("policy") and fa.get("matches_per_order", 0) >= 20 and 0 < fa.get("max_advantage_pp", 99) <= 5
              and fa.get("mirror") in ("exact", "statistical"), fa)
    else:                               # single-side formats only: no seat or first-order advantage to measure
        R.add(S, "[fairness] not needed (no format with two or more sides)", True)
    if (src / "presentation").is_dir():
        R.add(S, "presentation/ declared (spectators only unless `players`), data only, models in the manifest",
              *presentation_problems(src / "presentation"))
    arm = d.get("arm")
    if arm is not None:
        bt = sportlib.REPO / "bodies" / str(arm.get("body")) / "body.toml"
        try:
            lim = tomllib.loads(bt.read_text())["limits"]["arm"]
            re.compile(arm.get("joints", ""))
            ok = all(len(lim.get(k) or []) == 7 for k in ("speed_rad_s", "torque_nm")) and "(?P<j>" in arm["joints"]
            ok = ok and all((sportlib.REPO / "bodies" / str(f)).is_file() for f in arm.get("files", []))
        except Exception as e:  # noqa: BLE001
            ok, lim = False, e
        R.add(S, "[arm] body has bodies/<body>/body.toml [limits.arm]; joints pattern with group j", ok,
              f"{arm.get('body')}: {arm.get('joints')}")
    legs = d.get("legs")
    if legs is not None:                # a limits-only legged robot (bodies/__init__.py LEGS)
        name = str(legs.get("body"))
        tree = ast.parse((sportlib.REPO / "bodies" / "__init__.py").read_text())
        reg = next((ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "LEGS" for t in n.targets)), ())
        try:
            lim = tomllib.loads((sportlib.REPO / "bodies" / name / "body.toml").read_text())
            L, W = lim["limits"]["leg"], lim["limits"]["walk"]
            ok = name in reg and len(L["speed_rad_s"]) == len(L["torque_nm"]) == 6 and len(W["command_max"]) == 3 \
                and len(W["command_accel"]) == 3 and set(lim["motors"]["assign"]) >= {
                    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"}
        except Exception as e:  # noqa: BLE001
            ok, lim = False, e
        R.add(S, "[legs] body in bodies LEGS with body.toml [limits.leg], [limits.walk] and [motors]", ok, name)
    bd = d.get("body")
    if bd is not None:                  # a robot of the robot library (bodies/README.md)
        R.add(S, "[body] a library robot, its attachments and control libraries exist; a test-use robot is not "
                 "published", *body_problems(bd, d))
    for name, ok, det in skilleval_declare.check(sport, d, src):
        R.add(S, name, ok, det)
    tk = d.get("tasks", [])
    R.add(S, "[[tasks]] declared and consistent", bool(tk) and all(
        t.get("format") in fm and t.get("action_level", "high") in lv and t.get("play", "program") in pl for t in tk),
        [generate.task_id(d, t) for t in tk])
    return R.ok()


def body_registry(init_py: Path, names=("BODIES", "PLANAR")) -> tuple:
    """The robots a sport may name in [body]: bodies/__init__.py BODIES (robots with a model, rig and servo) plus PLANAR
    (planar-base robots with their own interface, e.g. a wheeled soccer robot; absent = none). Read without importing."""
    tree = ast.parse(init_py.read_text())
    reg = ()
    for want in names:
        reg += next((tuple(ast.literal_eval(n.value)) for n in tree.body if isinstance(n, ast.Assign)
                     and any(getattr(t, "id", "") == want for t in n.targets)), ())
    return reg


def reaction_delay_checks(d: dict, src: Path) -> list[tuple]:
    """[timing] reaction_delay_steps (core 0.2.18, core/spec.reaction_delay): an integer >= 0, real-time sports only;
    the screening hooks (screen/hooks.py) set no different DELAY (tools/screen/play.py plays the declared D). A
    real-time sport that declares none gets a warning: its acts take effect at the next control-step boundary, so
    screening (hooks DELAY) and the official matches may disagree (review LATENCY, 2026-10-13)."""
    tm = d.get("timing") or {}
    realtime = (d.get("sport") or {}).get("protocol") == "realtime"
    name = "[timing] reaction_delay_steps (declared reaction delay, core 0.2.18)"
    if "reaction_delay_steps" not in tm:
        return [(name, True, "not declared: an act takes effect at the next control-step boundary; screening uses "
                 "the hooks' DELAY", realtime)] if realtime else []
    v = tm["reaction_delay_steps"]
    ok = realtime and not isinstance(v, bool) and isinstance(v, int) and v >= 0
    out = [(name, ok, f"D = {v!r}" + ("" if realtime else " (a turn-based sport declares none)"), False)]
    hooks = src / "screen" / "hooks.py"
    m = re.search(r"^DELAY\s*=\s*(\d+)\s*(#.*)?$", hooks.read_text(), re.M) if hooks.exists() else None
    if m:
        out.append(("screen/hooks.py DELAY = the declared reaction delay", ok and int(m.group(1)) == v,
                    f"hooks DELAY = {m.group(1)}, declared D = {v!r} (remove DELAY: tools/screen/play.py reads D)",
                    False))
    return out


def body_problems(bd: dict, d: dict) -> tuple[bool, str]:
    """sport.toml [body] against bodies/: name in the library (bodies/__init__.py BODIES or PLANAR), hand / other_hand
    in bodies/hands/hands.toml, controllers in bodies/control/, and a robot whose body.toml says status = "test-use"
    only in sports with publish = false."""
    root = sportlib.REPO / "bodies"
    reg = body_registry(root / "__init__.py")
    bad = []
    name = bd.get("name")
    if name not in reg:
        bad.append(f"{name!r} not in the library {list(reg)}")
    hands = tomllib.loads((root / "hands" / "hands.toml").read_text())
    bad += [f"{k} {bd[k]!r} not in bodies/hands/hands.toml" for k in ("hand", "other_hand") if bd.get(k)
            and bd[k] not in hands]
    bad += [f"controller {c!r}: no bodies/control/{c}.py" for c in bd.get("controllers", [])
            if not (root / "control" / f"{c}.py").exists()]
    if name in reg:
        status = tomllib.loads((root / name / "body.toml").read_text())["body"].get("status", "library")
        if status == "test-use" and d["sport"].get("publish", False):
            bad.append(f"{name} is for test use only (body.toml status; its licence is not cleared): publish = false")
    return not bad, "; ".join(bad) or f"{name} + {bd.get('hand')} + {bd.get('controllers', [])}"


# ------------------------------------------------------------------------------------------------ docker helpers
_RUNS = itertools.count()               # unique container names when stage 5 runs several at once (--jobs)


def in_image(sport: str, args: list[str], timeout=None, module: str = "core.testkit",
             script: str | None = None, files: tuple = (), screen: bool = False) -> subprocess.CompletedProcess:
    """Run `python3 -m module ARGS` (or the given `script`, as /out/run.py) in the sport's image, the assembled sim at
    /srv (read-only, the working directory), /out writable; r.rows = /out/res.json. files: copied to /out (next to
    run.py, importable by it); screen: also the sport's screening directory, at /srv/sport/screen (never a task's)."""
    timeout = timeout or sportlib.stage_timeout(sport)
    with tempfile.TemporaryDirectory(prefix=f"check-{sport}-") as tmp:
        sim = sportlib.assemble_sim(sport, Path(tmp) / "sim", with_tests=True)
        if screen and (sportlib.SPORTS / sport / "screen").is_dir():
            shutil.copytree(sportlib.SPORTS / sport / "screen", sim / "sport" / "screen",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        out = Path(tmp) / "out"
        out.mkdir()
        for f in files:
            shutil.copy2(f, out / Path(f).name)
        if script is not None:
            (out / "run.py").write_text(script)
        cmd = ["docker", "run", "--rm", "--name", f"check-{sport}-{os.getpid()}-{int(time.time())}-{next(_RUNS)}",
               "--network", "none",
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "MUJOCO_GL=osmesa",
               "-e", "PYOPENGL_PLATFORM=osmesa", "-v", f"{sim}:/srv:ro", "-v", f"{out}:/out", "-w", "/srv",
               sportlib.image(sport), "python3", *(["/out/run.py"] if script is not None else ["-m", module]), *args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        r.rows = json.loads((out / "res.json").read_text()) if (out / "res.json").exists() else []
        return r


def stage_testkit(sport, R: Report, cmd: str, stage: str, extra=()):
    r = in_image(sport, [cmd, "/srv/sport/plugin.py", "--json", "/out/res.json", *extra])
    if not r.rows:
        return R.add(stage, f"core.testkit {cmd} ran", False, (r.stderr or r.stdout)[-800:])
    for x in r.rows:
        R.add(stage, x["name"], x["ok"], x["detail"], warn=x.get("warn", False))
    return R.ok()


# ------------------------------------------------------------------------------------------------ stages 3-5
def stage_golden(sport, R: Report, external=None):
    S = "3 golden"
    res = golden.replay(sport, external)
    R.add(S, "golden cases present", bool(res), f"{len(res)} case(s)")
    for x in res:
        R.add(S, ("external " if x.get("external") else "") + x["case"], x["ok"],
              x.get("diff") or x.get("error") or f"{x.get('scores')} {x.get('winner')} k={x.get('k')}")


# ------------------------------------------------------------------------------------------------ stage 5 reference
# The reference check (user rule BASEPERC, 2026-10-11; docs/construction_zh.md §2 problem 6 and step 4 "参考解";
# docs/screening/README.md "Reference check"), in two parts:
#   opponent calibration (one-side formats against a scripted opponent or with graded levels): the reference with
#     TRUTH perception wins the majority of the lowest level and its wins do not rise with the level (>= 20 seeds per
#     level); without the [reference] match hook: WARN, FAIL once frozen ([official.hidden_seeds]); truth matches
#     played by the camera reference (a hook without a truth variant): NOTE, FAIL once frozen; plus evidence that
#     camera-only can win: the camera-only reference wins the majority of the lowest level, or a model won a step-5
#     match with cameras only (registered in sport.toml [[reference.camera_wins]]);
#   end to end (every format): the camera-only reference's matches run normally (the live task run, and for a scripted
#     opponent its in-process matches); their playability (e.g. legal returns) is reported, with no numeric gate.
# A duel (two or more sides) only has to be playable: the live run of the reference on every seat is a valid match.
# A one-side format with neither a scripted opponent nor levels keeps its success line (reference -> success 1).
MIN_SEEDS = 20                          # per level (the user's 20-seed rule of 2026-10-11)
CAMERA_SEEDS = 5                        # camera matches only confirm (docs/construction_zh.md step 4: 5-10 per level)

REF_RUNNER = r'''
"""tools/check stage 5: one sport.toml [reference] match hook, several seeds, in the sport's image (/srv). /out (this
file's directory, on sys.path) also holds tools/screen/play.py and reference_hook.py, the shared hook."""
import importlib, json, sys, time, traceback
sys.path.insert(0, "/srv")
from core.episode import load_plugin
load_plugin("/srv/sport/plugin.py")         # the sport's directory goes first on sys.path
hook, fmt, level, perception, seeds = sys.argv[1:6]
mod, fn = hook.split(":")
f = getattr(importlib.import_module(mod), fn)
rows = []
for s in seeds.split(","):
    t0 = time.time()
    try:
        r = dict(f(format=fmt, level=None if level == "none" else int(level), seed=int(s), perception=perception))
        rows.append({"seed": int(s), "won": bool(r.get("won")), "scores": [float(x) for x in r.get("scores") or []],
                     "report": {k: [float(v[0]), float(v[1])] for k, v in (r.get("report") or {}).items()},
                     "wall_s": round(time.time() - t0, 1),
                     **({"played": r["perception"]} if r.get("perception") not in (None, perception) else {})})
    except Exception:                       # noqa: BLE001  (a match that cannot be played is a finding, not a crash)
        rows.append({"seed": int(s), "error": traceback.format_exc()[-800:], "wall_s": round(time.time() - t0, 1)})
    with open("/out/res.json", "w") as fh:  # after every match: a timeout keeps what was played
        json.dump(rows, fh)
    print(perception, fmt, level, s, rows[-1].get("won"), rows[-1].get("scores"), rows[-1].get("error", "")[-200:],
          flush=True)
'''


def scripted(d: dict, fmt: str) -> bool:
    """A format that gets the opponent calibration: one side, and either against the sport's scripted opponent
    (`opponent = true`) or with graded levels of its own ([formats.<f>] levels, e.g. baseball batting, drone racing;
    user decision 2026-10-11)."""
    f = d["formats"][fmt]
    return (bool(f.get("opponent")) or bool(f.get("levels"))) and \
        len(set((f.get("sides") or {s: s for s in f["seats"]}).values())) == 1


def opponent_levels(d: dict, fmt: str) -> list:
    """The format's official opponent levels ([formats.<f>] levels, else [official] levels; only those in [official]
    levels when the sport declares them), lowest first; [None] = the sport's default opponent only. A format level
    outside [official] levels is a warm-up level (soccer L1): listed and playable, but outside every official check
    (not calibrated, no camera evidence required; as scripts/olympiad/plan.py levels_of)."""
    off = (d.get("official") or {}).get("levels")
    lv = d["formats"][fmt].get("levels") or off or [None]
    if off and None not in lv:
        lv = [x for x in lv if x in off] or [None]
    return sorted(lv) if None not in lv else [None]


def seed_list(v, default: list[int]) -> list[int]:
    """[reference] seeds / camera_seeds: a list of seeds, or a count n (seeds 1..n)."""
    if v is None:
        return list(default)
    return list(range(1, int(v) + 1)) if isinstance(v, int) else [int(x) for x in v]


def tally(rows: list[dict]) -> dict:
    """Wins, matches, point share and the summed report counters of one batch of hook results."""
    ok = [r for r in rows if "error" not in r]
    ours = sum(r["scores"][0] for r in ok if len(r["scores"]) >= 2)
    tot = sum(r["scores"][0] + r["scores"][1] for r in ok if len(r["scores"]) >= 2)
    rep: dict = {}
    for r in ok:
        for k, (a, b) in r.get("report", {}).items():
            x = rep.setdefault(k, [0.0, 0.0])
            x[0] += a
            x[1] += b
    return {"wins": sum(r["won"] for r in ok), "n": len(ok), "errors": [r for r in rows if "error" in r],
            "share": round(ours / tot, 2) if tot else None, "report": rep}


def fmt_report(rep: dict) -> str:
    return ", ".join(f"{k} {a:g}/{b:g} ({100 * a / b:.0f} %)" if b else f"{k} {a:g}/0" for k, (a, b) in rep.items())


def calibration_verdict(by_level: dict, levels: list, n_min: int = MIN_SEEDS) -> tuple[bool, str]:
    """Truth reference over the levels (lowest first): >= n_min matches per level, no match that could not be played,
    the majority won at the lowest level, wins not rising from one level to the next."""
    t = {lv: tally(by_level.get(lv, [])) for lv in levels}
    bad = [f"L{lv}: {len(x['errors'])} match(es) failed (seed {x['errors'][0]['seed']}: "
           f"{x['errors'][0]['error'].strip().splitlines()[-1][:160]})" for lv, x in t.items() if x["errors"]]
    bad += [f"L{lv}: {x['n']} matches < {n_min}" for lv, x in t.items() if x["n"] < n_min]
    low = t[levels[0]]
    if not low["n"] or 2 * low["wins"] <= low["n"]:
        bad.append(f"the lowest level is not won by the majority ({low['wins']}/{low['n']})")
    rates = [t[lv]["wins"] / t[lv]["n"] if t[lv]["n"] else 0.0 for lv in levels]
    bad += [f"wins rise from L{a} to L{b}" for (a, b), (x, y) in zip(zip(levels, levels[1:]), zip(rates, rates[1:]))
            if y > x]
    detail = "; ".join(f"L{lv} {x['wins']}/{x['n']} (point share {x['share']})" for lv, x in t.items())
    return not bad, detail + ("" if not bad else " -- " + "; ".join(bad))


def registered_wins(d: dict, fmt: str, levels: list, repo: Path = sportlib.REPO) -> tuple[list[str], list[str]]:
    """sport.toml [[reference.camera_wins]] for `fmt`: (valid entries as text, problems). An entry: format, level (one
    of the format's levels), who (the model / agent), score, evidence (a file in the repository, `#anchor` allowed)."""
    good, bad = [], []
    for e in (d.get("reference") or {}).get("camera_wins", []):
        if e.get("format") != fmt:
            continue
        miss = [k for k in ("who", "score", "evidence") if not str(e.get(k, "")).strip()]
        if miss:
            bad.append(f"camera_wins entry without {', '.join(miss)}")
        elif e.get("level") not in levels and levels != [None]:
            bad.append(f"camera_wins level {e.get('level')!r} is not one of {levels}")
        elif not (repo / str(e["evidence"]).split("#")[0]).is_file():
            bad.append(f"camera_wins evidence {e['evidence']!r}: no such file")
        else:
            good.append(f"{e['who']} won L{e.get('level')} {e['score']} with cameras only ({e['evidence']})")
    return good, bad


def camera_verdict(cam: dict, good: list[str], bad: list[str], level) -> tuple[bool, str]:
    """"Camera-only can win": the camera reference won the majority of the lowest level, or a registered model win."""
    won = cam["n"] > 0 and 2 * cam["wins"] > cam["n"] and not cam["errors"]
    parts = [f"camera reference L{level} {cam['wins']}/{cam['n']}" + (" (majority)" if won else "")] + good + bad
    return won or bool(good), "; ".join(parts)


SHARED_HOOK = (Path(__file__).resolve().parent / "screen" / "play.py",          # the step-4 loop and the shared hook
               Path(__file__).resolve().parent / "screen" / "reference_hook.py")  # (docs/screening/README.md)


def run_hook(sport: str, hook: str, fmt: str, level, seeds: list[int], perception: str) -> list[dict]:
    r = in_image(sport, [hook, fmt, "none" if level is None else str(level), perception, ",".join(map(str, seeds))],
                 script=REF_RUNNER, files=SHARED_HOOK, screen=True)
    rows = r.rows or []
    done = {x["seed"] for x in rows}
    tail = (r.stderr or r.stdout or "")[-600:]
    return rows + [{"seed": s, "error": f"not played (container exit {r.returncode}): {tail}"} for s in seeds
                   if s not in done]


def stand_in(by_level: dict, frozen: bool = False) -> tuple[bool, str]:
    """The truth matches a hook played with another perception (the shared hook without a truth variant plays the
    camera reference and says so): (ok, text). Before the freeze a NOTE, so that the reader sees the calibration is
    the camera reference's; once frozen ([official.hidden_seeds] registered) a FAIL: a frozen sport needs a truth
    variant (user decision 2026-10-13)."""
    n = {}
    for rows in by_level.values():
        for r in rows:
            if r.get("played"):
                n[r["played"]] = n.get(r["played"], 0) + 1
    tag = "FAIL (frozen: [official.hidden_seeds] registered; add a truth variant)" if frozen else "NOTE"
    return not (frozen and n), "".join(f"; {tag}: {k} of the truth matches played with perception {p!r} (the sport "
                                       f"has no truth variant: a stand-in)" for p, k in n.items())


def stage_reference(sport, R: Report, fmt: str, jobs: int = 1):
    """Opponent calibration (truth) and the camera-only evidence for one scripted-opponent format, plus the camera
    reference's playability report (end to end, no gate)."""
    S = "5 live"
    d = sportlib.decl(sport)
    rd = d.get("reference") or {}
    hook = rd.get("match")
    levels = opponent_levels(d, fmt)
    if not (isinstance(hook, str) and re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", hook)
            and (sportlib.SPORTS / sport / (hook.split(":")[0].replace(".", "/") + ".py")).is_file()):
        frozen = bool((d.get("official") or {}).get("hidden_seeds"))   # transition (user, 2026-10-11): WARN until frozen
        return R.add(S, f"{fmt}: opponent calibration (truth reference)", not frozen,
                     ("FAIL (frozen: [official.hidden_seeds] registered): " if frozen else
                      "WARN (transition: FAIL once the sport registers [official.hidden_seeds]): ") +
                     "declare sport.toml [reference] match = \"module:function\": a function of the sport that plays "
                     "one match of the reference against the scripted opponent, function(format, level, seed, "
                     "perception) with perception \"truth\" (true state) or \"camera\" (the camera-only reference) "
                     "-> {won, scores: [ours, theirs], report: {name: [count, of]}}; the shared one is two lines "
                     "(tools/screen/reference_hook.py: sports/<sport>/screen/reference.py `from reference_hook import "
                     "hook; match = hook()`, match = \"screen.reference:match\"; docs/screening/README.md, "
                     "\"Reference check\")" + (f"; got {hook!r}" if hook else ""), warn=not frozen)
    seeds = seed_list(rd.get("seeds"), list(range(1, MIN_SEEDS + 1)))
    cseeds = seed_list(rd.get("camera_seeds"), seeds[:CAMERA_SEEDS])
    work = [("truth", lv, seeds) for lv in levels] + [("camera", levels[0], cseeds)]
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        res = list(ex.map(lambda w: run_hook(sport, hook, fmt, w[1], w[2], w[0]), work))
    by_level = {lv: rows for (p, lv, _), rows in zip(work, res) if p == "truth"}
    ok, detail = calibration_verdict(by_level, levels)
    real, note = stand_in(by_level, bool((d.get("official") or {}).get("hidden_seeds")))
    rep = fmt_report(tally([r for rows in by_level.values() for r in rows])["report"])
    R.add(S, f"{fmt}: opponent calibration: the truth reference wins the lowest level, wins fall with the level",
          ok and real, detail + (f"; truth reference: {rep}" if rep else "") + note + f" ({time.time() - t0:.0f} s)")
    cam = tally(res[-1])
    good, bad = registered_wins(d, fmt, levels)
    cok, cdetail = camera_verdict(cam, good, bad, levels[0])
    preview = bool(d["formats"][fmt].get("preview"))       # a preview format: not official yet, so a WARN
    R.add(S, f"{fmt}: camera-only can win (camera reference majority at L{levels[0]}, or a registered model win)",
          cok or preview, ("" if cok or not preview else "PREVIEW format, not official (WARN; required once official): ")
          + cdetail, warn=preview and not cok)
    err = cam["errors"]
    R.add(S, f"{fmt}: end to end: the camera reference's matches run (report only, no numeric gate)", not err,
          f"L{levels[0]} {cam['wins']}/{cam['n']} won, point share {cam['share']}"
          + (f"; {fmt_report(cam['report'])}" if cam["report"] else "")
          + (f"; {len(err)} failed (seed {err[0]['seed']}: {err[0]['error'].strip().splitlines()[-1][:200]})"
             if err else ""))


def stage_live(sport, R: Report, all_tasks=False, harbor=False, jobs=1):
    S = "5 live"
    d = sportlib.decl(sport)
    tasks = generate.tasks_of(sport)
    picked, seen = [], set()
    for tid, t in tasks:
        if all_tasks or t["format"] not in seen:
            picked.append((tid, t))
            seen.add(t["format"])
    for fmt in dict.fromkeys(t["format"] for _, t in picked):
        if scripted(d, fmt):
            stage_reference(sport, R, fmt, jobs)
    with tempfile.TemporaryDirectory(prefix=f"live-{sport}-") as tmp:
        root = Path(tmp) / "tasks"
        for tid, t in picked:
            task = generate.write_task(sport, tid, t, root)
            instr = (task / "instruction.md").read_text()
            tb = (d.get("tiebreaks") or {}).get("text")
            if tb and len(d["formats"][t["format"]]["seats"]) > 1:
                R.add(S, f"{tid}: the prompt states the tiebreaks", tb in instr)
            seats = d["formats"][t["format"]]["seats"]
            opp = scripted(d, t["format"])
            for who, want in (("oracle", 1), ("nop", 0)):
                over = {"START_TIMEOUT_S": "20"} if who == "nop" else {}
                run = live.run_harbor if harbor else live.run_offline
                t0 = time.time()
                rw = run(task, Path(tmp) / f"{tid}-{who}", [who] * len(seats), over)
                got = rw.get("success")
                detail = {k: rw.get(k) for k in ("success", "valid_trial", "deterministic", "match_over") if k in rw}
                detail.update({k: v for k, v in rw.items() if k.endswith("_won") or k.endswith("_points")})
                if opp and who == "oracle":     # end to end: the camera reference plays the task through (no win gate)
                    R.add(S, f"{tid}: end to end: the reference (camera only) plays a valid match to its end",
                          rw.get("valid_trial") == 1 and rw.get("match_over") == 1,
                          f"{detail} ({time.time() - t0:.0f} s)")
                    continue
                R.add(S, f"{tid}: {who} on every seat -> success {want}", got == want,
                      f"{detail} ({time.time() - t0:.0f} s)")


def stage_modes(sport, R: Report):
    """Stage 6: the post-match evaluator on our implementation, the reference skill, the boundary baseline and the
    isolation probe, each in its own isolated container (tools/skill_eval.py)."""
    S = "6 modes"
    lo = (sportlib.decl(sport).get("modes") or {}).get("low")
    if not isinstance(lo, dict):
        return R.add(S, "no [modes.low]: no skill to evaluate", True)
    th = lo.get("thresholds") or {}
    rmin, bmax = th.get("reference_min", 1.0), th.get("boundary_max", 0.0)
    b9 = (sportlib.decl(sport).get("modes") or {}).get("b9_test")
    r = in_image(sport, ["/srv/sport/plugin.py", str(b9), "--json", "/out/res.json"], module="skilleval.runtest")
    R.add(S, f"B9 feasibility test {b9} (robot and mode rule 2)", bool(r.rows) and all(x["ok"] for x in r.rows),
          (r.rows[0]["detail"] if r.rows else (r.stderr or r.stdout)[-600:]))
    t = sportlib.stage_timeout(sport)                   # each evaluation container, like every in-image stage
    pr = skill_eval.run(sport, "probe", timeout=t)
    R.add(S, "isolation probe: the skill reads no simulator file or hidden request, writes nothing, has no network",
          bool(pr.get("valid")), skill_eval.summary(pr))
    for who in ("platform", "reference", "boundary"):
        r = skill_eval.run(sport, who, checks=who == "platform", timeout=t)
        rate = (r.get("api_test") or {}).get("success_rate", 0.0)
        R.add(S, f"{who}: the evaluator ran (API test + competition test)", bool(r.get("valid")), skill_eval.summary(r))
        if who == "platform":
            R.add(S, f"platform: every hidden request feasible (API test >= reference_min {rmin})", rate >= rmin, rate)
            for c in r.get("checks", []):
                R.add(S, f"platform: {c['name']}", c["ok"], "" if c["ok"] else c["detail"])
        elif who == "reference":
            R.add(S, f"reference skill (toolkit only): API test >= {rmin} (rule 4)", rate >= rmin, rate)
            R.add(S, "reference skill: deterministic (rule 10)",
                  (r.get("determinism") or {}).get("deterministic") is True, r.get("determinism"))
        else:
            R.add(S, f"boundary baseline: API test <= {bmax} (core-skill boundary, rule 5)", rate <= bmax, rate)


def main(argv):
    ap = argparse.ArgumentParser(prog="tools/check")
    ap.add_argument("sport")
    ap.add_argument("--stages", default="1,2,3,4,5,6")
    ap.add_argument("--quick", action="store_true", help="fairness with 20 matches per order (expect a CI WARN)")
    ap.add_argument("--all-tasks", action="store_true")
    ap.add_argument("--harbor", action="store_true")
    ap.add_argument("--external")
    ap.add_argument("--jobs", type=int, default=1, help="stage 5: reference-check containers at once (one per level)")
    a = ap.parse_args(argv)
    if a.sport not in sportlib.names():
        print(f"no sports/{a.sport}/sport.toml")
        return 2
    stages = {int(x) for x in a.stages.split(",")}
    R = Report()
    t0 = time.time()
    for st in sorted(stages):
        print(f"== stage {st}", flush=True)
        if st == 1:
            contract(a.sport, R)
        elif st == 2:
            stage_testkit(a.sport, R, "tests", "2 tests")
        elif st == 3:
            stage_golden(a.sport, R, a.external)
        elif st == 4:
            stage_testkit(a.sport, R, "fairness", "4 fairness", ["--matches", "20"] if a.quick else [])
        elif st == 5:
            stage_live(a.sport, R, a.all_tasks, a.harbor, a.jobs)
        elif st == 6:
            stage_modes(a.sport, R)
    bad = [r for r in R.rows if not r[2]]
    print(f"tools/check {a.sport}: {len(R.rows) - len(bad)}/{len(R.rows)} passed in {time.time() - t0:.0f} s"
          + ("" if not bad else f"; FAILED: {', '.join(r[1] for r in bad[:8])}")
          + (f"; warnings: {', '.join(R.warnings)}" if R.warnings else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
