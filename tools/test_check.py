#!/usr/bin/env python3
"""Unit tests of tools/check (stage 5's reference check, the [body] registry; no docker, no simulator):
python3.11+ tools/test_check.py"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check  # noqa: E402

DECL = {
    "formats": {
        "match": {"seats": ["A"], "opponent": True, "levels": [3, 1, 2]},
        "duel": {"seats": ["A", "B"], "first": ["A", "B"]},
        "single": {"seats": ["A"], "success": {"score_at_least": 1}},
        "batting": {"seats": ["A"], "levels": [2, 1], "success": {"score_at_least": 1}},   # graded, no opponent
        "team": {"seats": ["A1", "A2"], "sides": {"A1": "A", "A2": "A"}, "opponent": True},
    },
    "official": {"levels": [1, 2]},
}


def rows(wins, n, ours=11, theirs=5, report=None):
    """n matches, the first `wins` won 11:5, the others lost 5:11 (scores ours first)."""
    return [{"seed": i + 1, "won": i < wins, "scores": [ours, theirs] if i < wins else [theirs, ours],
             "report": report or {}} for i in range(n)]


class Rep:
    def __init__(self):
        self.rows = []

    def add(self, stage, name, ok, detail="", warn=False):
        self.rows.append((name, bool(ok), str(detail)))
        return ok


class Formats(unittest.TestCase):
    def test_scripted(self):
        self.assertTrue(check.scripted(DECL, "match"))
        self.assertTrue(check.scripted(DECL, "team"))         # two seats, one side, against the scripted opponent
        self.assertFalse(check.scripted(DECL, "duel"))
        self.assertFalse(check.scripted(DECL, "single"))      # solo: the success line, unchanged
        self.assertTrue(check.scripted(DECL, "batting"))      # graded levels without opponent = true (2026-10-11)
        self.assertEqual(check.opponent_levels(DECL, "batting"), [1, 2])

    def test_levels(self):
        self.assertEqual(check.opponent_levels(DECL, "match"), [1, 2])        # L3 not in [official] levels: warm-up
        self.assertEqual(check.opponent_levels({**DECL, "official": {}}, "match"), [1, 2, 3])   # no [official] levels
        self.assertEqual(check.opponent_levels(DECL, "team"), [1, 2])          # [official] levels
        self.assertEqual(check.opponent_levels({"formats": {"m": {"seats": ["A"]}}}, "m"), [None])

    def test_warmup_level_not_calibrated(self):
        """A warm-up level (soccer L1: format levels 1-3, [official] levels 2, 3) is outside the calibration: the
        lowest level the reference must win is L2, a loss at L1 or more wins at L1 than at L2 do not matter, and a
        camera win registered at L1 is not evidence."""
        d = {"formats": {"match": {"seats": ["A"], "opponent": True, "levels": [1, 2, 3]}},
             "official": {"levels": [2, 3]}}
        lv = check.opponent_levels(d, "match")
        self.assertEqual(lv, [2, 3])
        ok, detail = check.calibration_verdict({1: rows(0, 20), 2: rows(12, 20), 3: rows(8, 20)}, lv)
        self.assertTrue(ok, detail)
        self.assertNotIn("L1", detail)
        ok, detail = check.calibration_verdict({1: rows(20, 20), 2: rows(8, 20), 3: rows(5, 20)}, lv)
        self.assertFalse(ok)                                  # L2, the lowest official level, not won by the majority
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ev.md").write_text("x")
            good, bad = check.registered_wins({**d, "reference": {"camera_wins": [
                {"format": "match", "level": 1, "who": "m", "score": "3:0", "evidence": "ev.md"}]}}, "match", lv,
                repo=Path(tmp))
        self.assertEqual(good, [])
        self.assertTrue(bad and "level 1" in bad[0], bad)

    def test_seeds(self):
        self.assertEqual(check.seed_list(None, [1, 2]), [1, 2])
        self.assertEqual(check.seed_list(3, []), [1, 2, 3])
        self.assertEqual(check.seed_list([101, 111], []), [101, 111])


class Calibration(unittest.TestCase):
    def test_badminton_like_passes(self):          # 18/20, 9/20, 0/20 (docs/screening/badminton.md)
        ok, det = check.calibration_verdict({1: rows(18, 20), 2: rows(9, 20), 3: rows(0, 20)}, [1, 2, 3])
        self.assertTrue(ok, det)
        self.assertIn("L1 18/20", det)

    def test_flat_top_levels_pass(self):           # 0/20 at two ceiling levels: not rising
        ok, det = check.calibration_verdict({1: rows(15, 20), 2: rows(0, 20), 3: rows(0, 20)}, [1, 2, 3])
        self.assertTrue(ok, det)

    def test_no_majority_fails(self):
        ok, det = check.calibration_verdict({1: rows(10, 20), 2: rows(2, 20)}, [1, 2])
        self.assertFalse(ok)
        self.assertIn("not won by the majority", det)

    def test_rising_fails(self):
        ok, det = check.calibration_verdict({1: rows(12, 20), 2: rows(14, 20)}, [1, 2])
        self.assertFalse(ok)
        self.assertIn("rise from L1 to L2", det)

    def test_too_few_seeds_fails(self):
        ok, det = check.calibration_verdict({1: rows(10, 10)}, [1])
        self.assertFalse(ok)
        self.assertIn("10 matches < 20", det)

    def test_failed_match_fails(self):
        r = rows(20, 20) + [{"seed": 99, "error": "Traceback\nValueError: boom"}]
        ok, det = check.calibration_verdict({1: r}, [1])
        self.assertFalse(ok)
        self.assertIn("ValueError: boom", det)

    def test_report_counters_summed(self):
        t = check.tally(rows(2, 3, report={"legal returns": [5, 10]}))
        self.assertEqual(t["report"], {"legal returns": [15.0, 30.0]})
        self.assertEqual(check.fmt_report(t["report"]), "legal returns 15/30 (50 %)")
        self.assertEqual(t["share"], round((11 * 2 + 5) / (16 * 3), 2))


class CameraEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "docs").mkdir()
        (self.repo / "docs" / "r.md").write_text("x")

    def tearDown(self):
        self.tmp.cleanup()

    def decl(self, **e):
        entry = {"format": "match", "level": 1, "who": "Opus 5.5", "score": "14:12", "evidence": "docs/r.md#ref", **e}
        return {**DECL, "reference": {"camera_wins": [entry]}}

    def test_registered_model_win(self):
        good, bad = check.registered_wins(self.decl(), "match", [1, 2, 3], self.repo)
        self.assertEqual(bad, [])
        ok, det = check.camera_verdict(check.tally(rows(1, 10)), good, bad, 1)
        self.assertTrue(ok, det)                       # the camera reference lost (1/10), the model's win counts
        self.assertIn("Opus 5.5 won L1 14:12", det)

    def test_registered_entry_problems(self):
        for e, msg in (({"evidence": "docs/none.md"}, "no such file"), ({"level": 7}, "not one of"),
                       ({"who": ""}, "without who")):
            good, bad = check.registered_wins(self.decl(**e), "match", [1, 2, 3], self.repo)
            self.assertEqual(good, [])
            self.assertIn(msg, bad[0])
        self.assertEqual(check.registered_wins(self.decl(), "team", [1, 2], self.repo), ([], []))   # other format

    def test_camera_majority(self):
        self.assertTrue(check.camera_verdict(check.tally(rows(3, 5)), [], [], 1)[0])
        self.assertFalse(check.camera_verdict(check.tally(rows(2, 5)), [], [], 1)[0])
        self.assertFalse(check.camera_verdict(check.tally([]), [], [], 1)[0])


class StageReference(unittest.TestCase):
    def run_stage(self, decl, results):
        R = Rep()
        calls = []

        def fake(sport, hook, fmt, level, seeds, perception):
            calls.append((perception, level, len(seeds)))
            return results[(perception, level)]
        with mock.patch.object(check.sportlib, "decl", return_value=decl), \
                mock.patch.object(check, "run_hook", side_effect=fake), \
                mock.patch.object(check, "registered_wins", return_value=(["Opus won L1 14:12"], [])):
            with tempfile.TemporaryDirectory() as t:
                (Path(t) / "x").mkdir()
                (Path(t) / "x" / "builtin_players.py").write_text("")
                with mock.patch.object(check.sportlib, "SPORTS", Path(t)):
                    check.stage_reference("x", R, "match")
        return R.rows, calls

    def missing(self, decl, fmt="match"):
        R = check.Report()
        with mock.patch.object(check.sportlib, "decl", return_value=decl), mock.patch("builtins.print"):
            check.stage_reference("x", R, fmt)
        self.assertEqual(len(R.rows), 1)
        return R

    def test_missing_hook_warns_before_freeze(self):        # transition (user, 2026-10-11)
        for fmt in ("match", "batting"):
            R = self.missing(DECL, fmt)
            _, name, ok, det = R.rows[0]
            self.assertTrue(ok)
            self.assertEqual(R.warnings, [name])
            self.assertIn('[reference] match = "module:function"', det)
            self.assertIn("WARN (transition", det)

    def test_missing_hook_fails_once_frozen(self):          # [official.hidden_seeds] registered (core 0.2.14)
        decl = {**DECL, "official": {"levels": [1, 2], "hidden_seeds": "seed_commitments.json"}}
        R = self.missing(decl)
        _, name, ok, det = R.rows[0]
        self.assertFalse(ok)
        self.assertEqual(R.warnings, [])
        self.assertIn("FAIL (frozen", det)
        self.assertIn('[reference] match = "module:function"', det)

    def test_badminton_like(self):
        decl = {**DECL, "official": {"levels": [1, 2, 3]},
                "reference": {"match": "builtin_players:reference_match", "camera_seeds": 10}}
        res = {("truth", 1): rows(18, 20), ("truth", 2): rows(9, 20), ("truth", 3): rows(0, 20),
               ("camera", 1): rows(1, 10, report={"legal returns": [11, 22]})}
        out, calls = self.run_stage(decl, res)
        self.assertEqual(sorted(calls), [("camera", 1, 10), ("truth", 1, 20), ("truth", 2, 20), ("truth", 3, 20)])
        self.assertEqual([ok for _, ok, _ in out], [True, True, True], out)
        self.assertIn("legal returns 110/220 (50 %)", out[2][2])

    def test_warmup_level_not_run(self):                 # DECL: match levels 1-3, [official] levels 1, 2
        decl = {**DECL, "reference": {"match": "builtin_players:reference_match", "camera_seeds": 10}}
        res = {("truth", 1): rows(18, 20), ("truth", 2): rows(9, 20), ("truth", 3): rows(20, 20),   # L3 never asked
               ("camera", 1): rows(6, 10)}
        out, calls = self.run_stage(decl, res)
        self.assertEqual(sorted(calls), [("camera", 1, 10), ("truth", 1, 20), ("truth", 2, 20)])
        self.assertEqual([ok for _, ok, _ in out], [True, True, True], out)
        self.assertNotIn("L3", out[0][2])

    def test_stand_in_is_a_note_before_the_freeze_and_fails_once_frozen(self):   # user decision 2026-10-13
        cam = lambda w, n: [{**r, "played": "camera"} for r in rows(w, n)]           # noqa: E731
        res = {("truth", 1): cam(18, 20), ("truth", 2): cam(9, 20), ("camera", 1): rows(4, 5)}
        decl = {**DECL, "reference": {"match": "builtin_players:reference_match"}}
        out, _ = self.run_stage(decl, res)
        self.assertTrue(out[0][1], out[0])
        self.assertIn("NOTE: 40 of the truth matches played with perception 'camera'", out[0][2])
        frozen = {**decl, "official": {"levels": [1, 2], "hidden_seeds": "seed_commitments.json"}}
        out, _ = self.run_stage(frozen, res)
        self.assertFalse(out[0][1], out[0])
        self.assertIn("FAIL (frozen", out[0][2])
        res[("truth", 1)], res[("truth", 2)] = rows(18, 20), rows(9, 20)            # a real truth variant: passes
        out, _ = self.run_stage(frozen, res)
        self.assertTrue(out[0][1], out[0])
        self.assertNotIn("stand-in", out[0][2])

    def test_camera_crash_fails_end_to_end(self):
        decl = {**DECL, "reference": {"match": "builtin_players:reference_match"}}
        res = {("truth", 1): rows(18, 20), ("truth", 2): rows(9, 20), ("truth", 3): rows(0, 20),
               ("camera", 1): rows(1, 4) + [{"seed": 5, "error": "x\nRuntimeError: no image"}]}
        out, _ = self.run_stage(decl, res)
        self.assertEqual([ok for _, ok, _ in out], [True, True, False])
        self.assertIn("RuntimeError: no image", out[2][2])


class BodyRegistry(unittest.TestCase):
    INIT = ('BODIES = ("arm_a", "humanoid_b")\nARMS = ("limits_only",)\nPLANAR = ("wheeled_c",)\n'
            'ROBOTS = BODIES + ARMS + PLANAR\n')

    def test_planar_robots_may_be_named(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "__init__.py"
            p.write_text(self.INIT)
            self.assertEqual(check.body_registry(p), ("arm_a", "humanoid_b", "wheeled_c"))

    def test_no_planar_registry(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "__init__.py"
            p.write_text('BODIES = ("arm_a",)\nARMS = ("limits_only",)\n')
            self.assertEqual(check.body_registry(p), ("arm_a",))

    def test_library_registry(self):
        """the real bodies/__init__.py: every robot with a model is accepted, limits-only entries are not"""
        reg = check.body_registry(check.sportlib.REPO / "bodies" / "__init__.py")
        self.assertIn("unitree_g1", reg)
        self.assertNotIn("franka_panda", reg)


class TestReactionDelay(unittest.TestCase):
    """The declared reaction delay (core 0.2.18): tools/check stage 1, tools/screen/play.py's D and the prompt."""

    def checks(self, decl, hooks=None):
        with tempfile.TemporaryDirectory() as tmp:
            if hooks is not None:
                (Path(tmp) / "screen").mkdir()
                (Path(tmp) / "screen" / "hooks.py").write_text(hooks)
            return [(ok, warn) for _, ok, _, warn in check.reaction_delay_checks(decl, Path(tmp))]

    def test_contract(self):
        rt, turn = {"sport": {"protocol": "realtime"}}, {"sport": {"protocol": "turn"}}
        self.assertEqual(self.checks(rt), [(True, True)])                       # not declared: a warning
        self.assertEqual(self.checks(turn), [])
        self.assertEqual(self.checks({**rt, "timing": {"reaction_delay_steps": 2}}), [(True, False)])
        self.assertEqual(self.checks({**rt, "timing": {"reaction_delay_steps": 0}}), [(True, False)])
        for bad in (-1, 1.5, True, "2"):
            self.assertEqual(self.checks({**rt, "timing": {"reaction_delay_steps": bad}}), [(False, False)], bad)
        self.assertEqual(self.checks({**turn, "timing": {"reaction_delay_steps": 2}}), [(False, False)])
        d2 = {**rt, "timing": {"reaction_delay_steps": 2}}
        self.assertEqual(self.checks(d2, "DELAY = 2   # steps\n"), [(True, False), (True, False)])
        self.assertEqual(self.checks(d2, "DELAY = 4\n"), [(True, False), (False, False)])
        self.assertEqual(self.checks(d2, "FORMAT = 'match'\n"), [(True, False)])

    def test_screen_delay(self):
        import importlib.util
        sys.path.insert(0, str(check.sportlib.REPO))
        spec = importlib.util.spec_from_file_location("screen_play", check.sportlib.REPO / "tools/screen/play.py")
        play = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(play)
        d2 = {"timing": {"reaction_delay_steps": 2}}
        self.assertEqual(play.screen_delay(d2, None, None)[0], 2)
        self.assertEqual(play.screen_delay(d2, 2, None)[0], 2)
        self.assertEqual(play.screen_delay(d2, None, 7), (7, "--delay"))        # an experiment
        with self.assertRaises(SystemExit):
            play.screen_delay(d2, 4, None)                                     # hooks DELAY disagrees
        self.assertEqual(play.screen_delay({}, 5, None)[0], 5)                  # not declared: the hooks, as before
        self.assertEqual(play.screen_delay({}, None, None)[0], 0)

    def test_reference_hook_delay_follows_the_declaration(self):
        """reference_hook.py's delay= reaches play.py as --sport-delay: the declared D wins, a different one stops."""
        import importlib.util
        import json as js
        import types
        sys.path.insert(0, str(check.sportlib.REPO / "tools" / "screen"))
        import reference_hook as rh
        seen = {}

        def main(argv):
            seen["argv"] = argv
            Path(argv[argv.index("--out") + 1], "result.json").write_text(js.dumps({"finished": True}))
        fake = types.SimpleNamespace(SRV=Path("/srv"), main=main, load_hooks=lambda p: types.SimpleNamespace())
        with mock.patch.object(rh, "_play_module", return_value=fake):
            rh.play_one("/h.py", "reference", "match", 2, 1, delay=5, action_level="high", srv="/srv")
        a = seen["argv"]
        self.assertEqual(a[a.index("--sport-delay") + 1], "5")
        self.assertNotIn("--delay", a)
        spec = importlib.util.spec_from_file_location("screen_play2", check.sportlib.REPO / "tools/screen/play.py")
        play = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(play)
        with self.assertRaises(SystemExit):
            play.screen_delay({"timing": {"reaction_delay_steps": 2}}, 5, None)

    def test_prompt(self):
        self.assertEqual(check.generate.reaction_delay_text({}), "")
        self.assertEqual(check.generate.reaction_delay_text({"timing": {"reaction_delay_steps": 0}}), "")
        self.assertIn("no earlier than 2 control steps", check.generate.reaction_delay_text(
            {"timing": {"reaction_delay_steps": 2}}))


# the shared [reference] match hook's tests (tools/screen/test_reference_hook.py) run with these (CI runs this file)
sys.path.insert(0, str(Path(__file__).resolve().parent / "screen"))
from test_reference_hook import Hook, PlayOne, ReadResult, Stage5  # noqa: E402,F401

# tools/frozen.py (frozen = registered hidden seeds; tools/test_frozen.py) runs with these too
from test_frozen import Frozen  # noqa: E402,F401

if __name__ == "__main__":
    unittest.main()
