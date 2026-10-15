"""hello's hooks for the post-match evaluator of the low level (skilleval/evaluate.py; sport.toml [modes.low] evaluator).

skill_spec()                  what every call of the skill receives as `spec`: the low-level seat's game spec
platform()                    our implementation of aim (the high level's AimLayer), in process: the reference row
api_case(case, skill)         one hidden request: the skill's throw, checked against the throw schema, thrown in
                              the simulator without execution scatter, measured at the first ground contact
competition_match(skill, s)   our fixed reference strategy (builtin_players.Reference at the high level, unchanged;
                              its aims clipped to the request ranges) with its aims lowered by the skill
                              (skilleval/layer.py), single format, seed s
fallback(seat, match)         what a failed skill call throws in the competition test (the weakest throw: 0 points)

  python3 sports/hello/lowlevel/evaluator.py make-cases     (sport image, repo root) rebuild lowlevel/cases/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPORT_DIR = HERE.parent
for p in (SPORT_DIR, *(q for q in (SPORT_DIR.parent, SPORT_DIR.parents[1]) if (q / "core").is_dir())):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mujoco  # noqa: E402

import plugin as P  # noqa: E402
from core import local  # noqa: E402
from core.spec import load  # noqa: E402
from core.sport import MatchConfig, Refused, check_command  # noqa: E402
from skilleval import cases as C  # noqa: E402
from skilleval.layer import SkillLayer, with_layer  # noqa: E402
from skill_api import AIM, CONTRACTS  # noqa: E402

PLUGIN = str(SPORT_DIR / "plugin.py")
DECL = load(PLUGIN)
COMPETITION_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
CASE_SEEDS = range(1001, 1041)  # the reference player's matches the API-test requests come from
WEAKEST = {"type": "throw", "yaw": 0.0, "pitch": 0.0, "speed": 1.0}


def cfg(seed: int, level: str) -> MatchConfig:
    return MatchConfig(sport="hello", format="single", seats=("A",), seed=seed, first="A", action_level=level)


def skill_spec() -> dict:
    return local.seat_spec(P.SPORT, cfg(1, "low"), "A", plugin=PLUGIN)


class Platform:
    """Our implementation: the high level's AimLayer (the throw of pitch 35 deg that lands at the point)."""

    def aim(self, x, y, *, state=None, spec=None):
        return P.AimLayer().lower("A", {"type": "aim", "x": x, "y": y}, None, None)[0]


def platform():
    return Platform()


def landing(cmd: dict) -> tuple[float, float]:
    """The first ground contact of a throw without execution scatter (the simulator's flight model, plugin.FLIGHT):
    the ball centre where it reaches the ball's radius, interpolated inside the 2 ms step."""
    d = mujoco.MjData(P.FLIGHT)
    y, p = math.radians(cmd["yaw"]), math.radians(cmd["pitch"])
    s = cmd["speed"]
    d.qvel[:3] = [s * math.cos(p) * math.cos(y), s * math.cos(p) * math.sin(y), s * math.sin(p)]
    prev = d.qpos[:3].copy()
    for _ in range(5000):
        mujoco.mj_step(P.FLIGHT, d)
        cur = d.qpos[:3]
        if cur[2] <= P.R_BALL:
            f = (prev[2] - P.R_BALL) / (prev[2] - cur[2])
            return float(prev[0] + f * (cur[0] - prev[0])), float(prev[1] + f * (cur[1] - prev[1]))
        prev = cur.copy()
    return float(prev[0]), float(prev[1])


def api_case(case: dict, skill) -> dict:
    r = case["request"]
    cmd = skill.call("aim", r["x"], r["y"], state=case.get("state", {}))
    try:
        cmd = check_command(P.SCHEMA, cmd)
    except Refused as e:
        return {"measured": False, "refused": 1, "why": str(e)[:200]}
    x, y = landing(cmd)
    return {
        "measured": True,
        "refused": 0,
        "landing_err_m": round(math.hypot(x - r["x"], y - r["y"]), 5),
        "landing": [round(x, 4), round(y, 4)],
    }


def fallback(seat, match):
    return dict(WEAKEST)


class Clipped:
    """The reference player with its aims clipped to the skill's request ranges: a bad implementation can drive its
    error correction out of range, and an out-of-range act would only be refused again and again."""

    def __init__(self, player):
        self.p = player

    def act(self, view):
        a = dict(self.p.act(view))
        for k, f in AIM.args.items():
            a[k] = min(max(a[k], f["min"]), f["max"])
        return a


def competition_match(skill, seed: int) -> dict:
    layer = SkillLayer(P.SPORT.action_layer("high"), CONTRACTS, skill, DECL, fallback=fallback)
    c = cfg(seed, "high")
    players = {s: Clipped(p) for s, p in local.make_players(P.SPORT, c, {"A": "reference"}, plugin=PLUGIN).items()}
    res = local.play(with_layer(P.SPORT, "high", layer), c, players, plugin=PLUGIN)
    score = int(res["scores"]["A"])
    line = DECL["formats"]["single"]["success"]["score_at_least"]
    return {
        "seed": seed,
        "success": int(res["finished"] and score >= line),
        "score": score,
        "skill_failures": len(layer.failures),
    }


# ------------------------------------------------------------------ the API-test cases
class _Recorder:
    def __init__(self):
        self.high, self.seen = P.AimLayer(), []

    def __getattr__(self, k):
        return getattr(self.high, k)

    def lower(self, seat, cmd, prepared, match):
        self.seen.append({"x": cmd["x"], "y": cmd["y"]})
        return self.high.lower(seat, cmd, prepared, match)


def make_cases(share: float = 0.3) -> tuple[int, int]:
    """Requests from our reference player's own matches (CASE_SEEDS, single, high level): every aim it sent. Kept if
    feasible under the launcher's ranges: our implementation delivers it within the tolerance with a valid command."""
    rows, plat = [], Platform()
    for seed in CASE_SEEDS:
        rec = _Recorder()
        c = cfg(seed, "high")
        players = local.make_players(P.SPORT, c, {"A": "reference"}, plugin=PLUGIN)
        local.play(with_layer(P.SPORT, "high", rec), c, players, plugin=PLUGIN)
        for i, req in enumerate(rec.seen):
            req = {k: round(float(v), 4) for k, v in req.items()}
            if AIM.valid_request(req):
                continue
            try:
                cmd = check_command(P.SCHEMA, plat.aim(**req))
            except Refused:
                continue
            x, y = landing(cmd)
            err = math.hypot(x - req["x"], y - req["y"])
            if AIM.judge({"landing_err_m": err, "refused": 0})["success"]:
                rows.append({"id": f"s{seed}-b{i + 1}", "request": req, "reference": {"landing_err_m": round(err, 5)}})
    public, hidden = C.split(rows, share)
    C.write(HERE / "cases" / "hidden.jsonl", hidden)
    C.write(HERE / "cases" / "public.jsonl", [C.practice_view(r) for r in public])
    return len(hidden), len(public)


if __name__ == "__main__":
    if sys.argv[1:2] == ["make-cases"]:
        print("hidden, public:", make_cases())
    else:
        print(__doc__)
