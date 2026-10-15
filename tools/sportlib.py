"""What the tools share about sports (stdlib only, python >= 3.11): the declarations (sports/<sport>/sport.toml) and
the assembly of a task's simulator directory, which is what the generator writes into tasks/<id>/environment/sim and
what the golden replay and the checks run:

  <sim>/core/          the match core (one copy)
  <sim>/renderers/     the renderer plugins (the players' images), without renderers/presentation (spectators only)
  <sim>/metering/      think_proxy.py of players/metering (byte-identical; the core's launcher runs one per seat)
  <sim>/spectator/     broadcast_log.py of scripts/olympiad/broadcast (byte-identical; the spectator broadcast log)
  <sim>/sport/         the sport: sports/<sport>/ without tests/, golden/, presentation/ and README.md
  <sim>/skilleval/     only for a sport that declares [modes.high] / [modes.low] (skilleval/README.md): the two-mode library
                       (contract, cases, sandbox, evaluator); the API test's hidden requests ([modes.low.cases] hidden)
                       stay out of a task's simulator directory (only the checks and the evaluator get them)
  <sim>/bodies/        only for a sport that declares [arm] body: bodies/<body>/body.toml (the real robot's joint
                       limits) and their enforcement, bodies/limits.py (and servo.py, sources.py if the sport's code
                       imports bodies/servo.py), and [arm] files (shared bodies/ modules the sport imports, e.g.
                       trajopt.py)
  <sim>/bodies/        only for a sport that declares [legs] body (a limits-only legged robot): bodies/<body>/body.toml
                       and the limit / servo path (limits.py, servo.py, sources.py)
  <sim>/bodies/        only for a sport that declares [body] (a robot of the robot library, bodies/README.md): the
                       robot without its fine meshes and provenance tools, bodies/hands/, the limit / servo path
                       (limits.py, servo.py, sources.py), assembly.py, the control libraries of [body] controllers, and
                       configuration.json (the assembled configuration's id and hash, bodies/assembly.py)

A sport whose presentation.toml declares `players` (its players' images use the presentation scene, renderer rule 6)
also gets <sim>/sport/presentation/ (the declaration and, under assets/, its models' files with the declared meshes,
checked against their sha256 in renderers/presentation/*.json, with their licences) and <sim>/renderers/presentation/
(the code and manifests, without the lod/ store): offline, and part of source_sha with the rest of the sport.
"""
from __future__ import annotations

import os
import re
import shutil
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPORTS = REPO / "sports"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")
NOT_IN_SIM = {"tests", "golden", "README.md", "presentation",   # presentation/: spectator renders only
              "seed_commitments.json",                     # the verifier's (core/seeds.py), not the sim's
              "screen",                                    # the screening hooks (tools/screen), not the sim's
              "diagnose.py", "diag"}                       # post-match analysis (core/diagnose.py), never in a match
BODY_NOT_IN_SIM = {"meshes_fine", "tools", "README.md", "__pycache__", "motion.pt", ".gitignore"}  # motion.pt: as .npz


def names() -> list[str]:
    return sorted(p.parent.name for p in SPORTS.glob("*/sport.toml"))


def decl(sport: str) -> dict:
    return tomllib.loads((SPORTS / sport / "sport.toml").read_text())


def image(sport: str) -> str:
    return decl(sport)["sport"]["image"]


def stage_timeout(sport: str) -> float:
    """Seconds one in-image stage may take (tools/check; tools/golden.py per recording or replay): sport.toml [check]
    stage_timeout_s (a slow sport's fairness run), else $RO_CHECK_TIMEOUT_S, else 3600."""
    t = (decl(sport).get("check") or {}).get("stage_timeout_s")
    return float(t or os.environ.get("RO_CHECK_TIMEOUT_S") or 3600)


def post_match_timeouts(d: dict) -> tuple[int, int]:
    """A task's limits after the match, sport.toml [timing] replay_timeout_s and video_timeout_s: the seconds the sim
    may take for the fresh-process replay (core/service.py finalize; a replay that does not finish voids the trial),
    then for the spectator video (defaults 600, 300); the task's collect and tools/live.py wait for both."""
    tm = d.get("timing", {})
    return int(tm.get("replay_timeout_s", 600)), int(tm.get("video_timeout_s", 300))


def players_meshes(sport: str) -> str | None:
    """The meshes of the sport's players' presentation scene ("full" or a level of detail), None = spectators only."""
    p = SPORTS / sport / "presentation" / "presentation.toml"
    if not p.is_file():
        return None
    return (tomllib.loads(p.read_text()).get("presentation") or {}).get("players")


def _assets_module():
    """renderers/presentation/assets.py loaded by path (standard library only; the package needs numpy)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ro_presentation_assets",
                                                  REPO / "renderers" / "presentation" / "assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bundle_presentation(sport: str, sim: Path) -> list[str]:
    """Copy the players' presentation of `sport` into the simulator directory `sim` (see the module docstring);
    returns the bundled model files (relative to <sim>/sport/presentation/assets)."""
    meshes = players_meshes(sport)
    src = SPORTS / sport / "presentation"
    dst = Path(sim) / "sport" / "presentation"
    shutil.copytree(src, dst, ignore=IGNORE)
    shutil.copytree(REPO / "renderers" / "presentation", Path(sim) / "renderers" / "presentation",
                    ignore=lambda dirpath, items: [x for x in items if x in ("__pycache__", "lod", "lod.py")
                                                   or x.endswith(".pyc")])
    decl = tomllib.loads((src / "presentation.toml").read_text())["presentation"]
    assets, out = _assets_module(), []
    for model in sorted({r["model"] for r in decl.get("robot", []) if not r["model"].startswith("./")}):
        out += assets.bundle(model, meshes, dst / "assets")
    return out


def two_modes(d: dict) -> bool:
    m = d.get("modes") or {}
    return isinstance(m.get("high"), dict) or isinstance(m.get("low"), dict)


def hidden_cases(d: dict) -> str | None:
    """The API test's hidden requests ([modes.low.cases] hidden, a path in the sport), None without a low mode."""
    return (((d.get("modes") or {}).get("low") or {}).get("cases") or {}).get("hidden")


def assemble_sim(sport: str, out: Path, with_tests: bool = False, hidden: bool | None = None) -> Path:
    """Write the simulator directory of `sport` to `out` (replaced); with_tests: also the sport's tests/ (checks);
    hidden: also the API test's hidden requests (default: with the tests; never in a generated task)."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(REPO / "core", out / "core", ignore=IGNORE)
    shutil.copytree(REPO / "renderers", out / "renderers", ignore=lambda dirpath, items: [
        x for x in items if x in ("__pycache__", "README.md") or x.endswith(".pyc") or x.startswith("test_")
        or (Path(dirpath) == REPO / "renderers" and x == "presentation")])   # spectators only, like NOT_IN_SIM
    (out / "metering").mkdir()
    shutil.copy2(REPO / "players" / "metering" / "think_proxy.py", out / "metering" / "think_proxy.py")
    (out / "spectator").mkdir()
    shutil.copy2(REPO / "scripts" / "olympiad" / "broadcast" / "broadcast_log.py", out / "spectator" / "broadcast_log.py")
    d, src = decl(sport), SPORTS / sport
    dst = out / "sport"
    shutil.copytree(src, dst, ignore=lambda dirpath, items: [
        x for x in items if x in ("__pycache__",) or x.endswith(".pyc")
        or (Path(dirpath) == src and x in NOT_IN_SIM)])
    if two_modes(d):
        shutil.copytree(REPO / "skilleval", out / "skilleval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "README.md"))
        hc = hidden_cases(d)
        if hc and not (with_tests if hidden is None else hidden):
            (dst / hc).unlink(missing_ok=True)
    if players_meshes(sport) is not None:
        bundle_presentation(sport, out)
    if (d.get("arm") or {}).get("body"):
        copy_arm_limits(d["arm"]["body"], out / "bodies", d["arm"].get("files", ()), servo=uses_servo(src))
    if (d.get("legs") or {}).get("body"):
        copy_leg_limits(d["legs"]["body"], out / "bodies")
    if d.get("body"):
        copy_body(d["body"], out / "bodies")
    if (d.get("self_collision") or {}).get("body"):
        copy_self_collision(d["self_collision"]["body"], out / "bodies")
    if with_tests and (src / "tests").is_dir():
        shutil.copytree(src / "tests", dst / "tests", ignore=IGNORE)
    return out


def uses_servo(src: Path) -> bool:
    """Whether a sport's code imports the library's servo module (bodies/servo.py) besides bodies/limits.py."""
    pat = re.compile(r"from bodies import [^\n]*\bservo\b|import bodies\.servo|from bodies\.servo import")
    return any(pat.search(f.read_text()) for f in src.rglob("*.py") if "tests" not in f.parts)


def copy_arm_limits(body: str, dst: Path, files=(), servo: bool = False) -> Path:
    """The arm limits of `body` as a sport's simulator needs them (sport.toml [arm] body): its body.toml and
    bodies/limits.py; with servo (the sport's code imports bodies/servo.py), also servo.py and the stdlib-only
    sources.py it imports; plus the shared bodies/ modules the sport imports ([arm] files, e.g. trajopt.py)."""
    src = REPO / "bodies"
    (dst / body).mkdir(parents=True, exist_ok=True)
    for f in ("limits.py",) + (("servo.py", "sources.py") if servo else ()):
        shutil.copy2(src / f, dst / f)
    shutil.copy2(src / body / "body.toml", dst / body / "body.toml")
    for f in files:
        shutil.copy2(src / f, dst / f)
    return dst


def copy_self_collision(body: str, dst: Path) -> Path:
    """What a sport whose own scene carries a library arm needs to record its self-collisions (sport.toml
    [self_collision]: bodies/selfcollision.py Monitor, and the arm's body.toml for its warn_force_n)."""
    (dst / body).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "bodies" / "selfcollision.py", dst / "selfcollision.py")
    shutil.copy2(REPO / "bodies" / body / "body.toml", dst / body / "body.toml")
    return dst


def copy_leg_limits(body: str, dst: Path) -> Path:
    """The leg limits of a limits-only legged robot (sport.toml [legs] body; bodies/__init__.py LEGS) as a sport's
    simulator needs them: bodies/<body>/body.toml and the limit / servo path (servo.py Actuators + WalkCommand, which
    imports limits.py; sources.py)."""
    src = REPO / "bodies"
    (dst / body).mkdir(parents=True, exist_ok=True)
    for f in ("limits.py", "servo.py", "sources.py"):
        shutil.copy2(src / f, dst / f)
    shutil.copy2(src / body / "body.toml", dst / body / "body.toml")
    return dst


def copy_body(spec: dict, dst: Path) -> Path:
    """A robot of the robot library as a sport's simulator needs it (sport.toml [body]: name, hand, other_hand, arm,
    controllers; bodies/README.md), and configuration.json: the assembled configuration (bodies/assembly.py)."""
    src = REPO / "bodies"
    name = spec["name"]
    dst.mkdir(parents=True, exist_ok=True)              # another part may share bodies/ (e.g. arm limits)
    for f in ("__init__.py", "limits.py", "servo.py", "selfcollision.py", "sources.py", "assembly.py"):
        shutil.copy2(src / f, dst / f)
    for f in ("contacts.py",):                          # contacts between robots (bodies/README.md §4.2), where present
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    shutil.copytree(src / "hands", dst / "hands", ignore=IGNORE, dirs_exist_ok=True)
    shutil.copytree(src / name, dst / name, dirs_exist_ok=True, ignore=lambda dirpath, items: [
        x for x in items if x in BODY_NOT_IN_SIM or x.endswith(".pyc")])
    ctl = list(spec.get("controllers", []))
    if ctl:
        (dst / "control").mkdir(exist_ok=True)
        shutil.copy2(src / "control" / "__init__.py", dst / "control" / "__init__.py")
        for c in ctl:
            shutil.copy2(src / "control" / f"{c}.py", dst / "control" / f"{c}.py")
    asm = _assembly()
    cfg = asm.assemble(name, spec.get("hand"), spec.get("other_hand"), ctl, spec.get("arm", "right"), root=dst)
    cfg.write(dst / asm.FILE_NAME)
    return dst


def configuration_id(d: dict) -> str | None:
    """The assembled configuration's id of a sport declaring [body] (task.toml metadata), else None."""
    b = d.get("body")
    if not b:
        return None
    return _assembly().assemble(b["name"], b.get("hand"), b.get("other_hand"), b.get("controllers", []),
                        b.get("arm", "right")).id


def _assembly():
    """bodies/assembly.py (stdlib only), loaded by path: the tools do not import bodies/ as a package."""
    import importlib.util
    import sys
    if "_ro_assembly" not in sys.modules:
        spec = importlib.util.spec_from_file_location("_ro_assembly", REPO / "bodies" / "assembly.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_ro_assembly"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["_ro_assembly"]
