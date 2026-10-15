"""hello: the reference post-match analysis (sports/hello/diagnose.py, core/diagnose.py) on an in-process match of
two reference players. Not an exploit test (tools/check does not run it; the analysis is not in the simulator). Run
in the sport's image from the repository root:
    docker run --rm -v $PWD:/w -w /w -e MUJOCO_GL=osmesa ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 \\
        python3 sports/hello/tests/test_diagnose.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
HELLO = str(HERE / "plugin.py")


def test_hello_reference_diagnosis():
    from core import diagnose, local
    from core.episode import load_plugin
    from core.sport import MatchConfig
    sport = load_plugin(HELLO)
    cfg = MatchConfig(sport="hello", format="duel", seats=("A", "B"), seed=3, first="A", action_level="high")
    players = local.make_players(sport, cfg, {"A": "reference", "B": "reference"}, plugin=HELLO)
    d = Path(tempfile.mkdtemp(prefix="diag-hello-"))
    res = local.play(sport, cfg, players, data=d, render=True, plugin=HELLO)
    r = subprocess.run([sys.executable, "-m", "core.diagnose", HELLO, str(d)], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads((d / "diag" / "summary.json").read_text())
    throws = res["final_fields"]["throws"]
    assert s["units"] == len(throws) and s["claims"] == len(throws) and not s["errors"], s
    claims = [json.loads(x) for x in (d / "diag" / "claims.jsonl").read_text().splitlines()]
    tgt = res["final_fields"]["target"]
    for c in claims:
        assert c["truth"] == tgt
    rests = sorted(tuple(t["rest"]) for t in throws)
    assert sorted(tuple(c["rest"]) for c in claims) == rests
    for seat in ("A", "B"):
        m = s["metrics"][seat]
        assert m["aim_error_m"]["capability"] == "A1" and m["aim_error_m"]["status"] == "unvalidated"
        assert s["linkage"][seat]["with_observation"] == s["linkage"][seat]["acts"] > 0
    # the turn protocol's observe rows carry the step and the frame of the images
    rep = diagnose.load(d, HELLO)
    o = next(iter(rep["obs"]["A"].values()))
    assert o["view_steps"] == [o["k"]]


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    test_hello_reference_diagnosis()
    print("ok   test_hello_reference_diagnosis")
