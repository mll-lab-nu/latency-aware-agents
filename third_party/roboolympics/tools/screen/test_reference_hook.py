#!/usr/bin/env python3
"""Unit tests of the shared [reference] match hook (tools/screen/reference_hook.py) and of how tools/check stage 5
runs it (no docker, no simulator): python3.11+ tools/screen/test_reference_hook.py"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import reference_hook as rh  # noqa: E402
import check  # noqa: E402


def result(winner="A", scores=None, summary=None, finished=True):
    """play.py's result.json, the fields the hook reads."""
    r = {"winner": winner, "scores": {"A": 5, "B": 3} if scores is None else scores, "finished": finished, "k": 900}
    if summary is not None:
        r["summary"] = summary
    return r


class ReadResult(unittest.TestCase):
    def test_play_py_scores(self):
        self.assertEqual(rh.read_result(result()), {"won": True, "scores": [5.0, 3.0], "report": {}})
        self.assertEqual(rh.read_result(result("B", {"A": 1, "opponent": 5}))["scores"], [1.0, 5.0])
        self.assertFalse(rh.read_result(result("B"))["won"])

    def test_summary_first(self):          # the hooks' summary(): won, scores, report win over play.py's fields
        r = rh.read_result(result("B", summary={"won": True, "scores": [7, 6], "report": {"returns": [3, 4]}}))
        self.assertEqual(r, {"won": True, "scores": [7.0, 6.0], "report": {"returns": [3, 4]}})

    def test_scores_not_numbers(self):     # e.g. racing's per-side dicts: no point share, the win still counts
        r = rh.read_result(result(scores={"A": {"heats_won": 2}, "B": {"heats_won": 1}}))
        self.assertEqual(r["scores"], [])
        self.assertTrue(r["won"])


class Hook(unittest.TestCase):
    def play(self, res=None):
        calls = []

        def fake(path, variant, fmt, level, seed, **kw):
            calls.append((path, variant, fmt, level, seed, kw))
            return res or result()
        return calls, mock.patch.object(rh, "play_one", side_effect=fake)

    def test_no_truth_variant_is_a_reported_stand_in(self):
        calls, p = self.play()
        with p:
            out = rh.hook()(format="match", level=2, seed=7, perception="truth")
        self.assertEqual(out["perception"], "camera")
        path, variant, fmt, level, seed, _ = calls[0]
        self.assertEqual((Path(path).name, variant, fmt, level, seed), ("reference_hook.py", "reference", "match", 2, 7))

    def test_truth_and_camera_variants(self):
        calls, p = self.play()
        with p:
            m = rh.hook("hooks.py", camera="ref", truth="ref_truth", delay=5)
            self.assertEqual(m(level=1, seed=1, perception="truth")["perception"], "truth")
            self.assertEqual(m(level=1, seed=1, perception="camera")["perception"], "camera")
        self.assertEqual([c[1] for c in calls], ["ref_truth", "ref"])
        self.assertEqual(calls[0][0], str(HERE / "hooks.py"))          # relative to the caller's file
        self.assertEqual(calls[0][5]["delay"], 5)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            rh.hook(truth="x")                                           # the built-in hooks have no truth variant
        calls, p = self.play(result(finished=False))
        with p:
            with self.assertRaises(ValueError):
                rh.hook()(perception="x-ray")
            with self.assertRaisesRegex(RuntimeError, "did not finish"):
                rh.hook()(perception="camera")

    def test_custom_read(self):
        calls, p = self.play(result(scores={"A": {"heats_won": 2}, "B": {"heats_won": 0}}))
        with p:
            out = rh.hook(read=lambda r: {"won": True, "scores": [r["scores"]["A"]["heats_won"], 0]})(seed=3)
        self.assertEqual(out, {"won": True, "scores": [2, 0], "perception": "camera"})


class PlayOne(unittest.TestCase):
    def srv(self, t, levels='["low"]'):
        (Path(t) / "sport").mkdir()
        (Path(t) / "sport" / "sport.toml").write_text(
            f'[modes]\naction_levels = {levels}\n[players]\nreference_class = "mine:Ref"\n')
        (Path(t) / "sport" / "mine.py").write_text(
            "class Ref:\n    def __init__(self, spec, seat, seed):\n        self.args = (spec, seat, seed)\n")
        return t

    def test_argv_and_result(self):
        seen = {}

        def main(argv):
            seen["argv"] = argv
            out = argv[argv.index("--out") + 1]
            Path(out, "result.json").write_text(json.dumps(result()))
        fake = types.SimpleNamespace(SRV=Path("/srv"), main=main, load_hooks=lambda p: types.SimpleNamespace())
        with tempfile.TemporaryDirectory() as t, mock.patch.object(rh, "_play_module", return_value=fake):
            self.srv(t)
            res = rh.play_one("/h.py", "reference", "match", 3, 9, every=2, opts={"x": 1}, srv=t)
            self.assertEqual(fake.SRV, Path(t))
        a = seen["argv"]
        self.assertEqual(a[:10], ["--hooks", "/h.py", "--variant", "reference", "--format", "match", "--seed", "9",
                                  "--action-level", "low"])                  # sport.toml [modes]: no high level
        self.assertIn("--level", a)
        self.assertEqual(a[a.index("--every") + 1], "2")
        self.assertEqual(a[a.index("--opt") + 1], "x=1")
        self.assertNotIn("--delay", a)                                         # the hooks' DELAY (play.py)
        self.assertNotIn("--sport-delay", a)
        self.assertEqual(res["winner"], "A")

    def test_action_level(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(rh.default_action_level(self.srv(t, '["low", "high"]')), "high")

    def test_builtin_hooks_build_the_reference(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(rh, "SRV", t):
            self.srv(t)
            sys.path.insert(0, str(Path(t) / "sport"))
            try:
                p = rh.make_player("reference", {"x": 1}, "A", 4, 0, {})
            finally:
                sys.path.remove(str(Path(t) / "sport"))
                sys.modules.pop("mine", None)
            self.assertEqual(p.args, ({"x": 1}, "A", 4))
            with self.assertRaises(ValueError):
                rh.make_player("reference_truth", {}, "A", 4, 0, {})

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "play.py needs numpy (CI installs it)")
    def test_the_real_play_py_is_loaded_by_path(self):
        play = rh._play_module()
        self.assertTrue(callable(play.main) and callable(play.load_hooks))
        self.assertEqual(Path(play.__file__).resolve(), HERE / "play.py")


class Stage5(unittest.TestCase):
    def test_run_hook_ships_the_shared_hook_and_the_screen_directory(self):
        seen = {}

        def run(cmd, **kw):
            srv = Path(cmd[cmd.index("-v") + 1].split(":")[0])
            out = Path([c for c in cmd if c.endswith(":/out")][0].split(":")[0])
            seen["out"] = sorted(p.name for p in out.iterdir())
            seen["screen"] = sorted(p.name for p in (srv / "sport" / "screen").iterdir())
            (out / "res.json").write_text(json.dumps([{"seed": 1, "won": True, "scores": [5, 3], "report": {}}]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with tempfile.TemporaryDirectory() as t:
            sports = Path(t) / "sports"
            (sports / "x" / "screen").mkdir(parents=True)
            (sports / "x" / "screen" / "reference.py").write_text("")

            def assemble(sport, out, with_tests=False):
                (Path(out) / "sport").mkdir(parents=True)
                return Path(out)
            with mock.patch.object(check.sportlib, "SPORTS", sports), \
                    mock.patch.object(check.sportlib, "assemble_sim", side_effect=assemble), \
                    mock.patch.object(check.sportlib, "image", return_value="img"), \
                    mock.patch.object(check.sportlib, "stage_timeout", return_value=10), \
                    mock.patch.object(check.subprocess, "run", side_effect=run):
                rows = check.run_hook("x", "screen.reference:match", "match", 1, [1, 2], "truth")
        self.assertEqual(seen["out"], ["play.py", "reference_hook.py", "run.py"])
        self.assertEqual(seen["screen"], ["reference.py"])
        self.assertEqual(rows[0]["won"], True)
        self.assertIn("not played", rows[1]["error"])                      # seed 2 never came back

    def test_stand_in_is_reported(self):
        rows = [{"seed": 1, "won": True, "scores": [5, 3], "report": {}, "played": "camera"}]
        ok, text = check.stand_in({1: rows, 2: rows})
        self.assertTrue(ok)                                                  # before the freeze: a NOTE
        self.assertIn("NOTE: 2 of the truth matches played with perception 'camera'", text)
        ok, text = check.stand_in({1: rows, 2: rows}, frozen=True)
        self.assertFalse(ok)                                                 # frozen: a FAIL (user, 2026-10-13)
        self.assertIn("FAIL (frozen", text)
        self.assertEqual(check.stand_in({1: [{"seed": 1, "won": True}]}, frozen=True), (True, ""))

    def test_the_runner_keeps_the_perception_played(self):
        """REF_RUNNER in this process, with a fake core and a hook returning perception camera for a truth match."""
        with tempfile.TemporaryDirectory() as t:
            core = Path(t) / "core"
            core.mkdir()
            (core / "__init__.py").write_text("")
            (core / "episode.py").write_text("def load_plugin(p):\n    pass\n")
            (Path(t) / "fakehook.py").write_text(
                "def match(format, level, seed, perception):\n"
                "    return {'won': seed == 1, 'scores': [5, 3], 'perception': 'camera'}\n")
            src = check.REF_RUNNER.replace('"/srv"', repr(t)).replace('"/out/res.json"', repr(t + "/res.json"))
            run = Path(t) / "run.py"
            run.write_text(src)
            subprocess.run([sys.executable, str(run), "fakehook:match", "match", "1", "truth", "1,2"], check=True,
                           capture_output=True, cwd=t, env={**os.environ, "PYTHONPATH": t})
            rows = json.loads((Path(t) / "res.json").read_text())
        self.assertEqual([(r["won"], r.get("played")) for r in rows], [(True, "camera"), (False, "camera")])


if __name__ == "__main__":
    unittest.main()
