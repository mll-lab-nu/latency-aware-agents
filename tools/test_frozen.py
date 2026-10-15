#!/usr/bin/env python3
"""Tests of tools/frozen.py (frozen = a registered hidden seed set whose public commitments verify). Standard library
only; CI runs them through tools/test_check.py."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen  # noqa: E402
from core import seeds  # noqa: E402

TOML = """[sport]
name = "{name}"
title = "T {name}"
revision = "{name}-3+core-0.2"
status = "sport"
publish = {publish}

[formats.match]
seats = ["A"]
opponent = true
levels = [1, 2, 3]
[formats.duel]
seats = ["A", "B"]

[official]
seeds = [1, 2]
levels = [2, 3]
{hidden}
[[tasks]]
id = "{name}-match-wallclock-program-standard"
format = "match"
"""


def sport(root: Path, name: str, publish=True, counts=None, tamper=None) -> None:
    """A sport in a scratch repository; counts = a registered set ({cell: n}); tamper(pub) edits its commitments."""
    hidden = ""
    if counts:
        pub = seeds.public(seeds.new(name, counts))
        hidden = (f'\n[official.hidden_seeds]\nsha256 = "{pub["sha256"]}"\n'
                  f'counts = {{ {", ".join(f"{json.dumps(c)} = {n}" for c, n in pub["counts"].items())} }}\n')
        if tamper:
            tamper(pub)
        (root / "sports" / name).mkdir(parents=True, exist_ok=True)
        (root / "sports" / name / seeds.COMMITMENTS).write_text(json.dumps(pub))
    (root / "sports" / name).mkdir(parents=True, exist_ok=True)
    (root / "sports" / name / "sport.toml").write_text(
        TOML.format(name=name, publish="true" if publish else "false", hidden=hidden))
    t = root / "tasks" / f"{name}-match-wallclock-program-standard"
    t.mkdir(parents=True, exist_ok=True)
    (t / "task.toml").write_text(f'[metadata]\nrevision = "{name}-3+core-0.2"\n')


def run(root: Path, *argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = frozen.main(list(argv), root)
    return rc, out.getvalue()


class Frozen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_registered_set_is_frozen_with_its_cells(self):
        sport(self.root, "aa", counts={"match/L2": 20, "match/L3": 20, "duel": 20})
        sport(self.root, "bb")
        sport(self.root, "ex", publish=False)
        rc, out = run(self.root, "--json")
        self.assertEqual(rc, 0)
        rep = json.loads(out)
        self.assertEqual(rep["schema"], frozen.SCHEMA)
        self.assertEqual(rep["frozen"], ["aa"])
        self.assertEqual(rep["screening"], ["bb"])           # the unpublished example is neither
        aa = {r["sport"]: r for r in rep["sports"]}["aa"]
        self.assertEqual(aa["hidden_seeds"]["counts"], {"duel": 20, "match/L2": 20, "match/L3": 20})
        self.assertEqual(aa["hidden_seeds"]["formats"], {"duel": None, "match": [2, 3]})
        self.assertEqual(aa["revision"], "aa-3+core-0.2")
        self.assertEqual(aa["tasks"], [{"id": "aa-match-wallclock-program-standard", "format": "match",
                                        "revision": "aa-3+core-0.2"}])
        self.assertEqual(aa["official_levels"], [2, 3])
        self.assertIsNone(aa["problem"])
        rc, out = run(self.root, "--names")
        self.assertEqual(out.split(), ["aa"])
        rc, out = run(self.root)
        self.assertIn("frozen (1): aa", out)
        self.assertIn("match/L2=20", out)

    def test_unverified_registration_is_not_frozen(self):
        def other_sha(pub):
            pub["sha256"] = "0" * 64
        def dropped(pub):
            pub["cells"]["duel"] = pub["cells"]["duel"][:-1]
            pub["counts"]["duel"] -= 1
        sport(self.root, "sha", counts={"duel": 3}, tamper=other_sha)      # file's sha256 differs from sport.toml
        sport(self.root, "cut", counts={"duel": 3}, tamper=dropped)        # fewer commitments than registered
        sport(self.root, "gone", counts={"duel": 3})
        (self.root / "sports" / "gone" / seeds.COMMITMENTS).unlink()        # registration without commitments
        rc, out = run(self.root, "--json")
        self.assertEqual(rc, 1)
        rep = json.loads(out)
        self.assertEqual(rep["frozen"], [])
        rows = {r["sport"]: r for r in rep["sports"]}
        self.assertIn("sha256 differs", rows["sha"]["problem"])
        self.assertIn("cell duel has 2 commitments, count 3", rows["cut"]["problem"])
        self.assertIn("missing", rows["gone"]["problem"])
        rc, out = run(self.root)
        self.assertIn("NOT FROZEN, registration invalid", out)

    def test_unknown_sport(self):
        sport(self.root, "aa")
        self.assertEqual(run(self.root, "zz")[0], 2)

    def test_repository_registrations_verify(self):
        """Every registration in this repository verifies (a broken one would silently drop a sport from the
        official part)."""
        rc, out = run(frozen.REPO, "--json")
        rep = json.loads(out)
        self.assertEqual([r["sport"] for r in rep["sports"] if r["problem"]], [])
        self.assertEqual(rc, 0)
        for r in rep["sports"]:
            self.assertEqual(r["frozen"], bool(r["hidden_seeds"]))


if __name__ == "__main__":
    unittest.main()
