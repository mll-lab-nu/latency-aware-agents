#!/usr/bin/env python3
"""The robot library's real limits are declared and go through the one trusted limit / servo path (bodies/README.md
"限位与伺服"; audit 2026-10-08 S4). Static, stdlib only (no numpy, no simulator).

Library (every robot of bodies/):
  L1  body.toml declares its limits with sources (bodies/sources.py source_problems: every quantity S<k> /
      ESTIMATE: / DERIVED: / NOT PUBLISHED:, the required quantities of each kind of group present)
  L2  a robot with a model (bodies/__init__.py BODIES) builds its servo in bodies/servo.py (its body.py imports it and
      calls wheeled_layout / legged_layout / Servo) and never writes its actuators itself (no `.ctrl[` outside
      bodies/servo.py); the optional control libraries and skills (bodies/control/, bodies/<robot>/*.py) neither
  L3  a robot whose body.toml says status = "test-use" has a README saying so
  L4  self-collision declared (B0, 2026-10-09; bodies/selfcollision.py): every robot with a model and every arm a
      sport uses has body.toml [self_collision]: parts (regular expressions that compile; for a robot with a model
      each matches a body of model.xml), rules, warn_force_n with a source, every exclusion two body patterns that
      match bodies and a written reason
  L5  self-collision fits the model (needs mujoco + numpy: CI installs them; locally skipped with a warning): per
      robot with a model, in every hand configuration, no enabled pair penetrates at the rest pose, none overlaps in
      (nearly) every random legal pose (a mount overlap must be an exclusion with a reason), no exclusion is unused
      (python3 -m bodies.selfcollision check)

Sports (every sports/<sport>/sport.toml that references a library robot: [arm] body, [legs] body or [body] name):
  S1  the robot is in the library (bodies/<name>/body.toml)
  S2  the sport's code imports the library's limit path (bodies.limits / bodies.servo, or load_body for [body]) and
      uses it: [arm] JerkLimiter and TorqueRateLimiter (or servo.Servo); [legs] servo.Actuators (the legs' motors) and
      servo.WalkCommand (the walking command); [body] the robot's rig (`.rig(`) or the servo
  S3  no bypass: the sport defines no class of the same name as a trusted limiter (bodies/sources.py TRUSTED), does
      not assign to attributes of the limits / servo modules or of a rig's servo, and does not loosen the limits
      (`.scaled(k)` with k > 1)
  S4  a sport whose scenes carry a library arm with [self_collision] (sport.toml [self_collision]: body and
      [[self_collision.scenes]]: scene file, robots = [{prefix, suffix, world_geoms}], geoms) has in each scene exactly
      the pairs bodies/selfcollision.py plans from the declaration (the generated block: `python3 -m
      bodies.selfcollision scene sports/<s>` writes it), and no scene switches contacts off (<flag contact="disable"/>);
      a sport whose arm declares it but lists no scenes only warns
Warnings (not failures): a [body] sport that writes `.ctrl[` itself (allowed for its own non-robot actuators: make
sure none is the robot's), and sports whose scene carries a library robot without referencing the library (the
audit's G1: their arms are not yet on the shared path).

    python3 ci_checks/check-robot-limits.py            exit 1 on a failure
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODIES = ROOT / "bodies"
sys.path.insert(0, str(BODIES))
import sources  # noqa: E402  (bodies/sources.py: stdlib only)

HINTS = {"franka_panda": r"\bPanda\b|Franka", "kuka_iiwa14": r"iiwa|KUKA", "wheeled_humanoid": r"RB-Y1",
         "unitree_g1": r"Unitree G1", "unitree_h1": r"Unitree H1", "galaxea_r1pro": r"R1 Pro",
         "tianji_marvin_pro": r"Marvin", "abb_irb1100": r"IRB ?1100", "booster_t1": r"Booster T1"}


def registry(name: str):
    tree = ast.parse((BODIES / "__init__.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == name for t in n.targets):
            return _names(n.value)
    return ()


def _names(v):
    """a registry tuple, or a sum of registry names (ROBOTS = BODIES + ARMS + LEGS)"""
    if isinstance(v, ast.BinOp):
        return _names(v.left) + _names(v.right)
    if isinstance(v, ast.Name):
        return registry(v.id)
    return tuple(ast.literal_eval(v))


def code_of(d: Path) -> dict[Path, str]:
    return {p: p.read_text() for p in sorted(d.rglob("*.py")) if "__pycache__" not in p.parts and "tests" not in p.parts}


def main() -> int:
    fails, warns = [], []
    F = lambda m: fails.append(m)
    W = lambda m: warns.append(m)
    bodies, robots = registry("BODIES"), registry("ROBOTS")
    # ------------------------------------------------------------------ the library
    for r in robots:
        bt = BODIES / r / "body.toml"
        if not bt.exists():
            F(f"L1 bodies/{r}: no body.toml")
            continue
        decl = tomllib.loads(bt.read_text())
        for p in sources.source_problems(decl):
            F(f"L1 bodies/{r}/body.toml: {p}")
        if decl["body"].get("status") == "test-use":
            rd = BODIES / r / "README.md"
            if not rd.exists() or not re.search(r"test use|TEST USE|测试", rd.read_text()):
                F(f"L3 bodies/{r}: status test-use but its README does not say so")
    for r in bodies:
        bp = BODIES / r / "body.py"
        s = bp.read_text() if bp.exists() else ""
        if "from .. import servo" not in s or not re.search(r"SV\.(wheeled_layout|legged_layout|Servo)\(", s):
            F(f"L2 bodies/{r}/body.py: does not build its servo in bodies/servo.py")
    for p, s in {**code_of(BODIES / "control"), **{q: q.read_text() for r in bodies for q in (BODIES / r).glob("*.py")}}.items():
        for i, line in enumerate(s.splitlines(), 1):
            if re.search(r"\.ctrl\[[^\]]*\]\s*=", line):
                F(f"L2 {p.relative_to(ROOT)}:{i}: writes actuators itself (only bodies/servo.py Actuators.write may)")
    # ------------------------------------------------------------------ self-collision (L4, L5)
    import xml.etree.ElementTree as ET
    arms_used = set()
    for st in sorted((ROOT / "sports").glob("*/sport.toml")):
        b = (tomllib.loads(st.read_text()).get("arm") or {}).get("body")
        if b:
            arms_used.add(b)
    for r in list(bodies) + sorted(a for a in arms_used if a in robots and a not in bodies):
        decl = tomllib.loads((BODIES / r / "body.toml").read_text())
        sc = decl.get("self_collision")
        if not sc:
            msg = f"L4 bodies/{r}/body.toml: no [self_collision] (real limits B0: the arms cannot pass through the body)"
            (F if r in bodies or r == "franka_panda" else W)(msg)
            continue
        names = None
        if r in bodies:
            root = ET.parse(BODIES / r / decl["body"]["model"]).getroot()
            names = [b.get("name") for b in root.iter("body") if b.get("name")]
        pats = {}
        for k, v in (sc.get("parts") or {}).items():
            try:
                pats[k] = re.compile(v)
            except re.error as e:
                F(f"L4 bodies/{r}: [self_collision] parts.{k} does not compile ({e})")
                continue
            if names is not None and not any(pats[k].fullmatch(n) for n in names):
                F(f"L4 bodies/{r}: [self_collision] parts.{k} matches no body of {decl['body']['model']}")
        if "body" not in pats or not any(k.endswith("arm") for k in pats):
            F(f"L4 bodies/{r}: [self_collision] parts needs a body part and at least one arm part")
        if not isinstance(sc.get("rules"), dict):
            F(f"L4 bodies/{r}: [self_collision] rules missing")
        if "warn_force_n" not in sc or not sources.SOURCE_RE.match(str((sc.get("source") or {}).get("warn_force_n", ""))):
            F(f"L4 bodies/{r}: [self_collision] warn_force_n needs a value and a source in [self_collision.source]")
        for e in sc.get("exclude", []):
            bs, why = e.get("bodies"), str(e.get("reason", ""))
            if not (isinstance(bs, list) and len(bs) == 2) or len(why.strip()) < 20:
                F(f"L4 bodies/{r}: a [[self_collision.exclude]] needs bodies = [two patterns] and a written reason")
                continue
            for pat in bs:
                try:
                    rx = re.compile(pat)
                except re.error as e2:
                    F(f"L4 bodies/{r}: exclusion pattern {pat!r} does not compile ({e2})")
                    continue
                if names is not None and not any(rx.fullmatch(n) for n in names):
                    F(f"L4 bodies/{r}: exclusion pattern {pat!r} matches no body of the model")
    try:
        import mujoco  # noqa: F401
        import numpy  # noqa: F401
        have_mj = True
    except ImportError:
        have_mj = False
    if have_mj:
        import subprocess
        res = subprocess.run([sys.executable, "-m", "bodies.selfcollision", "check", *bodies], cwd=ROOT,
                             capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.startswith("FAIL [self-collision] "):
                F("L5 " + line[len("FAIL [self-collision] "):])
            elif line.startswith("self-collision "):
                print("    " + line[:200])
        if res.returncode and not any(x.startswith("L5 ") for x in fails):
            F("L5 python3 -m bodies.selfcollision check failed: " + (res.stderr or res.stdout)[-600:])
    elif os.environ.get("CI"):
        F("L5 the self-collision check needs mujoco and numpy (pip install mujoco==3.3.7 numpy)")
    else:
        W("L5 skipped: no mujoco here (python3 -m bodies.selfcollision check runs it; CI installs mujoco)")
    # ------------------------------------------------------------------ the sports
    users = []
    for st in sorted((ROOT / "sports").glob("*/sport.toml")):
        sport = st.parent.name
        d = tomllib.loads(st.read_text())
        arm, body = (d.get("arm") or {}).get("body"), (d.get("body") or {}).get("name")
        legs = (d.get("legs") or {}).get("body")
        if not arm and not body and not legs:
            text = st.read_text()
            hit = [r for r, pat in HINTS.items() if re.search(pat, text)]
            if hit:
                W(f"sports/{sport}: its robot looks like {hit} but sport.toml references no library robot "
                  "([arm] body / [body]): its limits are not on the shared path yet")
            continue
        users.append(sport)
        name = arm or legs or body
        if name not in robots:
            F(f"S1 sports/{sport}: {name!r} is not a library robot {list(robots)}")
            continue
        code = code_of(st.parent)
        allc = "\n".join(code.values())
        imports = re.search(r"from bodies import [^\n]*\b(limits|servo|load_body)\b|import bodies\.(limits|servo)\b|"
                            r"from bodies\.(limits|servo) import", allc)
        if not imports:
            F(f"S2 sports/{sport}: references {name} but imports no bodies.limits / bodies.servo / load_body")
        if arm and not (("JerkLimiter" in allc and "TorqueRateLimiter" in allc) or "Servo(" in allc):
            F(f"S2 sports/{sport}: [arm] {name}: the arm's setpoints and torques do not go through "
              "JerkLimiter + TorqueRateLimiter (or servo.Servo)")
        if legs and not ("Actuators(" in allc and "WalkCommand(" in allc):
            F(f"S2 sports/{sport}: [legs] {name}: the legs' motors and walking command do not go through "
              "servo.Actuators and servo.WalkCommand")
        if body and not (".rig(" in allc or "Servo(" in allc):
            F(f"S2 sports/{sport}: [body] {name}: the robot is not driven through its rig (rig.apply) or servo.Servo")
        for p, s in code.items():
            rel = p.relative_to(ROOT)
            try:
                tree = ast.parse(s)
            except SyntaxError as e:
                F(f"S3 {rel}: does not parse ({e})")
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef) and n.name in sources.TRUSTED:
                    F(f"S3 {rel}:{n.lineno}: defines its own {n.name} (the library's is the only one)")
                if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        txt = ast.unparse(t)
                        if re.match(r"^(AL|SV|limits|servo|bodies\.limits|bodies\.servo)\.", txt) or ".servo." in txt:
                            F(f"S3 {rel}:{n.lineno}: assigns {txt} (patches the limit path)")
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "scaled":
                    k = n.args[0] if n.args else None
                    if isinstance(k, ast.Constant) and isinstance(k.value, (int, float)):
                        if k.value > 1:
                            F(f"S3 {rel}:{n.lineno}: loosens the limits (scaled({k.value}))")
                    else:
                        W(f"{rel}:{n.lineno}: scaled() with a non-literal factor: make sure it is <= 1")
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setattr" and n.args and \
                        ast.unparse(n.args[0]) in ("AL", "SV", "limits", "servo"):
                    F(f"S3 {rel}:{n.lineno}: setattr on the limit path")
            if body and re.search(r"\.ctrl\[[^\]]*\]\s*=", s):
                W(f"{rel}: a [body] sport writes .ctrl itself: only its own non-robot actuators may be written so")
    # ------------------------------------------------------------------ S4 self-collision in the sports' own scenes
    import selfcollision as SCN  # noqa: E402  (bodies/selfcollision.py: the planning part is stdlib only)
    for st in sorted((ROOT / "sports").glob("*/sport.toml")):
        sport = st.parent.name
        d = tomllib.loads(st.read_text())
        sc = d.get("self_collision")
        arm = (d.get("arm") or {}).get("body")
        if not sc:
            hit = [r for r, pat in HINTS.items() if r in robots and r not in bodies and re.search(pat, st.read_text())]
            scenes = [x.read_text() for x in st.parent.rglob("*.xml")]
            carries = any(re.search(r'<body name="[A-Za-z_]*link7', x) for x in scenes)   # a Panda in its own scene
            body = arm or (hit[0] if hit and carries else None)
            if body and tomllib.loads((BODIES / body / "body.toml").read_text()).get("self_collision"):
                W(f"sports/{sport}: its arm ({body}) declares [self_collision] but sport.toml has no [self_collision] "
                  "scenes: its scenes carry no generated self-collision pairs yet")
            continue
        if sc.get("body") not in robots or (arm and sc.get("body") != arm):
            F(f"S4 sports/{sport}: [self_collision] body {sc.get('body')!r} is not its arm / a library robot")
            continue
        try:
            for f, plans in SCN.sport_scenes(st.parent):
                want = [(f"{p.tag}_{i}", g1, g2) for p in plans for i, (g1, g2, *_r) in enumerate(p.pairs)]
                have = SCN.scene_pairs(f.read_text())
                opt = ET.fromstring(f.read_text()).find("option")
                flag = opt.find("flag") if opt is not None else None
                if flag is not None and flag.get("contact") == "disable":
                    F(f"S4 {f.relative_to(ROOT)}: <flag contact=\"disable\"/> switches every contact off, the "
                      "self-collision pairs included")
                if have != want:
                    F(f"S4 {f.relative_to(ROOT)}: its self-collision pairs ({len(have)}) are not what "
                      f"bodies/selfcollision.py plans ({len(want)}): python3 -m bodies.selfcollision scene "
                      f"sports/{sport}")
                for p in plans:
                    for u in p.unused_exclusions:
                        F(f"S4 {f.relative_to(ROOT)}: exclusion {u} of {sc['body']} matches no pair")
        except (KeyError, ValueError, OSError, ET.ParseError) as e:
            F(f"S4 sports/{sport}: [self_collision] scenes cannot be planned ({type(e).__name__}: {e})")
    for m in warns:
        print("WARN [robot limits]", m)
    for m in fails:
        print("FAIL [robot limits]", m)
    print(f"robot limits: {len(robots)} library robots, sports on the shared path: {users or 'none'}"
          + ("" if fails else "; ok"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
