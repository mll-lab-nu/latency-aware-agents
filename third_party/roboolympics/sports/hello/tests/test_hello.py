"""hello's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite). Run alone: python3 -m core.testkit tests sports/hello/plugin.py (in the sport's image)."""
import os
import tempfile
from pathlib import Path

from core.episode import load_plugin
from core.sport import MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DUEL = MatchConfig(sport="hello", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low")


def test_exploit_bad_actions_are_refused():
    m, layer = SPORT.build(DUEL), SPORT.action_layer("low")
    d0 = m.state_digest()
    for body in ({"type": "throw", "yaw": 90, "pitch": 30, "speed": 6}, {"type": "throw", "yaw": 0, "pitch": 30},
                 {"type": "throw", "yaw": 0, "pitch": 30, "speed": float("inf")}, {"type": "aim", "x": 5, "y": 0},
                 {"type": "throw", "yaw": 0, "pitch": 30, "speed": 6, "spin": 3}):
        try:
            layer.parse("A", body, m, {})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400
    assert m.state_digest() == d0 and m.k == 0


def test_exploit_out_of_turn():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    with tempfile.TemporaryDirectory() as tmp:
        svc = TurnService(SPORT, PLUGIN, DUEL, Path(tmp))
        post = svc.routes(True)
        body = {"type": "throw", "yaw": 0.0, "pitch": 35.0, "speed": 7.0}
        assert post["act"]("A", body)[0] == 409                      # before the start: refused
        post["start"]("A", {})
        post["start"]("B", {})
        assert post["act"]("B", body)[0] == 409                      # A throws first
        code, out = post["act"]("A", body)
        assert code == 200 and svc.m.k == 1
        assert post["act"]("A", body)[0] == 409                      # the review comes first
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", body)[0] == 409                      # now it is B's turn
        assert svc.m.k == 1


def test_exploit_no_positions_in_observation():
    m = SPORT.build(DUEL)
    for s in "ABAB":
        m.apply(s, {"type": "throw", "yaw": 0.0, "pitch": 35.0, "speed": 7.0})
    text = repr(m.observe("A")) + repr(m.observe("B"))
    for x in (*m.target, *(t["rest"][0] for t in m.throws), *(t["r"] for t in m.throws)):
        assert repr(x) not in text and f"{x:.3f}" not in text, x


# ------------------------------------------------------------------ the two modes (sport.toml [modes.high] / [modes.low])
HERE = Path(__file__).resolve().parents[1]


def _toml():
    import tomllib
    return tomllib.loads((HERE / "sport.toml").read_text())


def _module(name, rel):
    import importlib.util
    import sys
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / rel)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def _flight():
    return _module("hello_flight", "seat/lib/flight.py")


def _evaluator():
    return _module("hello_evaluator", "lowlevel/evaluator.py")


def test_b9_feasible_under_the_launcher_ranges():
    """Robot and mode rule 2 (B9) for the launcher: every API-test request (hidden and public) and the corners of the
    area the seeds place targets in (4-6 m ahead, +-0.5 m, with 0.5 m to spare) are delivered by our implementation
    within the contract's tolerance, with a throw inside the launcher's ranges."""
    import math
    from skilleval import cases as C
    from skill_api import AIM
    import plugin as P
    from core.sport import check_command
    ev = _evaluator()
    lo = _toml()["modes"]["low"]["cases"]
    reqs = [c["request"] for c in C.load(HERE / lo["hidden"]) + C.load(HERE / lo["public"])]
    reqs += [{"x": x, "y": y} for x in (3.5, 6.5) for y in (-1.0, 1.0)]
    plat = ev.platform()
    for r in reqs:
        cmd = check_command(P.SCHEMA, plat.aim(r["x"], r["y"]))             # refused if outside the ranges
        x, y = ev.landing(cmd)
        assert AIM.judge({"landing_err_m": math.hypot(x - r["x"], y - r["y"]), "refused": 0})["success"], (r, x, y)


def test_exploit_toolkit_is_the_manifest():
    import ast
    import math
    from skilleval import declare, toolkit
    assert toolkit.problems(HERE / "seat" / "toolkit.toml", "hello", "low") == []
    rows = declare.check("hello", _toml(), HERE)
    assert all(ok for _, ok, _ in rows), [r for r in rows if not r[1]]
    src = (HERE / "seat" / "lib" / "flight.py").read_text()
    imps = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    imps |= {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert imps <= {"math"}, imps                                          # nothing of the simulator
    fl = _flight()
    ev = _evaluator()
    for yaw, pitch, speed in ((0.0, 35.0, 7.0), (-12.0, 20.0, 9.5), (8.0, 55.0, 4.0), (19.0, 5.0, 11.0)):
        p = fl.land(yaw, pitch, speed)
        x, y = ev.landing({"type": "throw", "yaw": yaw, "pitch": pitch, "speed": speed})
        assert math.hypot(p["x"] - x, p["y"] - y) < 1e-6, (yaw, pitch, speed, p, x, y)


def test_exploit_hidden_cases_stay_in_the_simulator():
    from skilleval import cases as C
    d = _toml()
    cs = d["modes"]["low"]["cases"]
    hidden, public = C.load(HERE / cs["hidden"]), C.load(HERE / cs["public"])
    assert C.problems(hidden, public) == []
    seat = d["seat"]["low"]["files"]
    assert seat[cs["practice_path"]] == cs["public"]
    assert all(not cs["hidden"].startswith(v.rstrip("/")) for v in seat.values())
    assert all(set(c) <= set(C.PRACTICE_KEYS) for c in public)


def test_exploit_skill_runs_isolated():
    import json
    from skilleval.sandbox import SkillError, SkillProcess
    bodies = {
        "ok": "def aim(x, y, *, state=None, spec=None):\n    print('noise on stdout')\n    import os\n"
              "    os.write(1, b'raw noise\\n')\n    return {'type': 'throw', 'yaw': 0.0, 'pitch': 35.0, 'speed': 7.0,"
              " 'env': sorted(os.environ), 'spec': spec}\n",
        "raises": "def aim(x, y, *, state=None, spec=None):\n    raise ValueError('bad')\n",
        "slow": "import time\ndef aim(x, y, *, state=None, spec=None):\n    time.sleep(5)\n",
        "imports": "def aim(x, y, *, state=None, spec=None):\n    import plugin\n",
        "nan": "def aim(x, y, *, state=None, spec=None):\n    return {'speed': float('nan')}\n",
    }
    os.environ["RO_TEST_SECRET"] = "s3cr3t"
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in bodies.items():
            (Path(tmp) / name).mkdir()
            (Path(tmp) / name / "aim.py").write_text(body)
        with SkillProcess(Path(tmp) / "ok", "aim", context={"spec": {"k": 1}}, timeouts={"aim": 5}) as s:
            r = s.call("aim", 5.0, 0.0, state={})
            assert r["yaw"] == 0.0 and r["spec"] == {"k": 1} and "RO_TEST_SECRET" not in r["env"], r
            assert s.call("aim", 5.0, 0.0, state={})["speed"] == 7.0      # the protocol survives the noise
        for name, kind in (("raises", "error"), ("imports", "error"), ("nan", "error"), ("slow", "timeout")):
            with SkillProcess(Path(tmp) / name, "aim", timeouts={"aim": 1}) as s:
                try:
                    s.call("aim", 5.0, 0.0, state={})
                    raise AssertionError(f"{name}: no SkillError")
                except SkillError as e:
                    assert e.kind == kind, (name, e)
                assert s.counters()["skill_errors" if kind == "error" else "skill_timeouts"] == 1
                if kind == "timeout":
                    assert not s.alive
                if name == "imports":
                    assert "No module named 'plugin'" in json.dumps(s.errors), s.errors
    os.environ.pop("RO_TEST_SECRET")
