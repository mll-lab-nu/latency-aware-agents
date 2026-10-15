#!/usr/bin/env python3
"""How close a sport's robot arms come to the real robot's limits (bodies/<body>/body.toml, bodies/limits.py).

    python3 tools/arm_limits.py measure SPORT [--cases a,b] [--episodes DIR ...] [--json OUT]

Replays the sport's golden matches (recorded with its built-in reference players; or the episode directories given)
in the sport's simulator image, without rendering and without network, and records every physics step of every arm
joint: the joint positions (velocity, acceleration and jerk by finite differences, cut where the world is reset: the
time does not advance by one step, or a joint moves faster than 20 rad/s over one step, i.e. it was placed) and the
torque of the joint's motor (torque rate by differences). The replay itself is unchanged (the recorder only reads the
MjData after each mujoco.mj_step), so the result must still match expected.json; the tool checks that. Prints per
joint p50 / p95 / p99 / max of |velocity|, |acceleration|, |jerk|, |torque|, |torque rate| against the limits, the
share of samples above them, and a one-line summary per quantity (worst joint's p99 / limit). What it measures is the
joints' actual motion: external forces (a hit, a bind) can push a joint beyond what its servo commands. For a sport
whose servo uses bodies/limits.JerkLimiter it also reports the commanded setpoints.

Which joints are a sport's arms and which body they are: sport.toml [arm] (body, joints = a regular expression with
the groups `arm` and `j`, the joint number 1..7), else the table ARMS below (the sports measured before they declare
[arm]). A sport whose arm is kinematic (positions set, no motor) gets no torque rows. Needs python >= 3.11 and docker.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sportlib  # noqa: E402

# sports measured before they declare [arm] (phase 1 of the arm-limits work): body + the arm-joint pattern (basketball
# left the table with basketball-5: its shooter is a declared shooting machine, the kinematic Panda arm is gone)
ARMS = {
    "tabletennis": ("franka_panda", r"^joint(?P<j>[1-7])(?P<arm>_b)?$"),
    "tennis": ("franka_panda", r"^joint(?P<j>[1-7])(?P<arm>_b)?$"),
    "badminton": ("franka_panda", r"^joint(?P<j>[1-7])(?P<arm>_b)?$"),
    "airhockey": ("kuka_iiwa14", r"^iiwa_(?P<arm>[12])/joint_(?P<j>[1-7])$"),
}

RECORDER = r"""
import json, re, sys
import numpy as np
import mujoco
sys.path.insert(0, "/srv")
from core.episode import load_plugin, read_rows, replay

PAT = re.compile(sys.argv[1])
traces = {}                                       # id(MjData) -> recorder
_step = mujoco.mj_step


class Rec:
    def __init__(self, m):
        arms = {}
        for i in range(m.njnt):
            mt = PAT.match(m.joint(i).name)
            if mt:
                arms.setdefault(mt.group("arm") or "-", {})[int(mt.group("j"))] = i
        self.arms = {a: [js[k] for k in sorted(js)] for a, js in sorted(arms.items()) if len(js) == 7}
        self.qadr = {a: np.array([m.jnt_qposadr[j] for j in js]) for a, js in self.arms.items()}
        act = {}
        for k in range(m.nu):
            if m.actuator_trntype[k] == mujoco.mjtTrn.mjTRN_JOINT:
                act[int(m.actuator_trnid[k, 0])] = k
        self.aadr = {a: (np.array([act[j] for j in js]) if all(j in act for j in js) else None)
                     for a, js in self.arms.items()}
        self.dt = float(m.opt.timestep)
        self.t, self.q, self.tau = [], {a: [] for a in self.arms}, {a: [] for a in self.arms}

    def add(self, d):
        self.t.append(float(d.time))
        for a in self.arms:
            self.q[a].append(d.qpos[self.qadr[a]].copy())
            if self.aadr[a] is not None:
                self.tau[a].append(d.actuator_force[self.aadr[a]].copy())


def mj_step(m, d, *args, **kw):
    _step(m, d, *args, **kw)
    r = traces.get(id(d))
    if r is None:
        r = traces[id(d)] = Rec(m)
        r.keep = d                                # keep the MjData alive: its id stays unique
    if r.arms:
        r.add(d)


mujoco.mj_step = mj_step
# the commanded setpoints, for a sport whose servo uses the shared setpoint generator (bodies/limits.JerkLimiter): every
# instance's output per step; an outside change of its state (a reset, a re-sync to the arm) cuts the series
setpoints = {}
try:
    from bodies import limits as _BL
    _jl_step = _BL.JerkLimiter.step

    def _step_rec(self, *a, **kw):
        rec = setpoints.setdefault(id(self), {"keep": self, "q": [], "cut": [], "last": None})
        if rec["last"] is not None and not np.array_equal(rec["last"], self.q):
            rec["cut"].append(len(rec["q"]))
        out = _jl_step(self, *a, **kw)
        rec["q"].append(out[0].copy())
        rec["last"] = out[0].copy()
        return out
    _BL.JerkLimiter.step = _step_rec
except ImportError:
    pass
sport = load_plugin("/srv/sport/plugin.py")
out = replay(sport, read_rows("/ep/episode.jsonl"))
best = max((r for r in traces.values() if r.arms), key=lambda r: len(r.t), default=None)
if best is None:                                  # e.g. a match decided before the first physics step
    json.dump({"replay": {k: out[k] for k in ("state_sha", "scores", "winner", "k")}, "arms": [], "steps": 0},
              open("/ep/trace.json", "w"), default=float)
    sys.exit(0)
sps = sorted((r for r in setpoints.values() if len(r["q"]) > 1), key=lambda r: -len(r["q"]))
sp_arrays = {}
for i, r in enumerate(sps):
    sp_arrays[f"sp_{i}"] = np.array(r["q"])
    sp_arrays[f"spcut_{i}"] = np.array([0] + r["cut"], dtype=int)
np.savez_compressed("/ep/trace.npz", t=np.array(best.t), dt=best.dt, n_sp=len(sps), **sp_arrays,
                    **{f"q_{a}": np.array(best.q[a]) for a in best.arms},
                    **{f"tau_{a}": np.array(best.tau[a]) for a in best.arms if best.tau[a]})
json.dump({"replay": {k: out[k] for k in ("state_sha", "scores", "winner", "k")}, "arms": list(best.arms),
           "steps": len(best.t), "other_datas": sorted(len(r.t) for r in traces.values() if r is not best and r.arms)},
          open("/ep/trace.json", "w"), default=float)
"""


def arm_decl(sport: str):
    d = sportlib.decl(sport).get("arm") or {}
    if d.get("body"):
        return d["body"], d["joints"], int(d.get("measure_decimate", 1))
    if sport in ARMS:
        return (*ARMS[sport], 1)[:3]
    raise SystemExit(f"{sport}: no [arm] in sport.toml and not in tools/arm_limits.py ARMS")


def run_case(sport, sim, src: Path, work: Path, pattern: str) -> dict:
    case = work / src.name
    case.mkdir(parents=True, exist_ok=True)
    if (src / "episode.jsonl").exists():
        shutil.copy2(src / "episode.jsonl", case / "episode.jsonl")
    else:
        with gzip.open(src / "episode.jsonl.gz", "rb") as f:
            (case / "episode.jsonl").write_bytes(f.read())
    (case / "rec.py").write_text(RECORDER)
    r = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"armlim-{sport}-{src.name}-{os.getpid()}",
            "--network",
            "none",
            "-v",
            f"{sim}:/srv:ro",
            "-v",
            f"{case}:/ep",
            "-w",
            "/srv",
            "-e",
            "MUJOCO_GL=osmesa",
            "-e",
            "PYOPENGL_PLATFORM=osmesa",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            sportlib.image(sport),
            "python3",
            "/ep/rec.py",
            pattern,
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if r.returncode != 0:
        raise SystemExit(f"{src.name}: {(r.stderr or r.stdout)[-1500:]}")
    info = json.loads((case / "trace.json").read_text())
    exp = src / "expected.json"
    if exp.exists():
        want = json.loads(exp.read_text())["expect"]
        info["replay_ok"] = all(info["replay"].get(k) == v for k, v in want.items())
    return {"case": src.name, "dir": case, **info}


ANALYSE = r"""
import json, sys
import numpy as np
sys.path.insert(0, "/limits")
from bodies import limits as L
body, dec, cases = sys.argv[1], int(sys.argv[2]), json.loads(open("/work/cases.json").read())
lim = L.load(body, "/limits/bodies")
pooled = {k: [] for k in ("vel", "acc", "jerk", "tau", "dtau")}
sp_pooled = {k: [] for k in ("vel", "acc", "jerk")}
n = 0
for c in cases:
    if not c["arms"]:
        continue
    z = dict(np.load(f"/work/cases/{c['case']}/trace.npz"))
    if dec > 1:                                   # a kinematic arm posed once per control step: sample at that step
        z = {k: (v[dec - 1::dec] if np.ndim(v) else v) for k, v in z.items()}
        z["dt"] = float(z["dt"]) * dec
    t, dt = z["t"], float(z["dt"])
    n += len(t)
    for i in range(int(z.get("n_sp", 0))):        # commanded setpoints (the shared JerkLimiter's output)
        q = z[f"sp_{i}"]
        cuts = sorted(set(int(c) for c in z[f"spcut_{i}"]))
        der = L.derivatives(q, float(np.load(f"/work/cases/{c['case']}/trace.npz")["dt"]), cuts)
        for b in cuts:
            der["acc"][b:b + 2] = np.nan
            der["jerk"][b:b + 3] = np.nan
        for k in sp_pooled:
            sp_pooled[k].append(der[k])
    brk = [0] + [int(i) for i in np.nonzero(np.abs(np.diff(t) - dt) > 1e-7)[0] + 1]
    for a in c["arms"]:
        q = z[f"q_{a}"]
        jump = [int(i) for i in np.nonzero(np.max(np.abs(np.diff(q, axis=0)), axis=1) > 20.0 * dt)[0] + 1]
        cuts = sorted(set(brk + jump))
        der = L.derivatives(q, dt, cuts)
        for b in cuts:                            # differences must not reach back across a cut
            der["acc"][b:b + 2] = np.nan
            der["jerk"][b:b + 3] = np.nan
        for k in ("vel", "acc", "jerk"):
            pooled[k].append(der[k])
        if f"tau_{a}" in z:
            tau = z[f"tau_{a}"]
            dtau = np.full_like(tau, np.nan)
            dtau[1:] = np.diff(tau, axis=0) / dt
            dtau[cuts] = np.nan
            pooled["tau"].append(tau)
            pooled["dtau"].append(dtau)
series = {k: np.vstack(v) for k, v in pooled.items() if v}
sp_series = {k: np.vstack(v) for k, v in sp_pooled.items() if v}
json.dump({"body": body, "limits": lim.table(), "stats": L.stats(series, lim), "samples": int(n),
           "setpoint_stats": L.stats(sp_series, lim) if sp_series else None},
          open("/work/analysis.json", "w"), default=float)
"""


def analyse(sport, work: Path, cases: list[dict], body: str, decimate: int = 1) -> dict:
    lim_dir = work / "limits"
    sportlib.copy_arm_limits(body, lim_dir / "bodies")
    (work / "cases.json").write_text(json.dumps([{"case": c["case"], "arms": c["arms"]} for c in cases]))
    (work / "analyse.py").write_text(ANALYSE)
    r = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"armlim-an-{sport}-{os.getpid()}",
            "--network",
            "none",
            "-v",
            f"{work}:/work",
            "-v",
            f"{lim_dir}:/limits:ro",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            sportlib.image(sport),
            "python3",
            "/work/analyse.py",
            body,
            str(decimate),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if r.returncode != 0:
        raise SystemExit(f"analysis: {(r.stderr or r.stdout)[-1500:]}")
    return json.loads((work / "analysis.json").read_text())


def table(res: dict) -> str:
    lines = [f"body {res['body']} ({res['limits']['model']}), {res['samples']} physics steps"]
    for key, s in res["stats"].items():
        lines.append(f"\n{s['title']} [{s['unit']}]: joint  p50  p95  p99  max  limit  p99/limit  max/limit  over")
        for r in s["joints"]:
            if r is None:
                continue
            lim = r.get("limit")
            lines.append(
                f"  {r['joint']}  {r['p50']:.4g}  {r['p95']:.4g}  {r['p99']:.4g}  {r['max']:.4g}  "
                + (
                    f"{lim:g}  {r['p99_ratio']:.2f}  {r['max_ratio']:.2f}  {100 * r['over_share']:.1f}%"
                    if lim is not None
                    else "-  -  -  -"
                )
            )
    sp = res.get("setpoint_stats")
    if sp:
        lines.append(
            "\ncommanded setpoints (bodies/limits.JerkLimiter output), worst joint: "
            + "; ".join(
                f"{s['title']} p99/limit {max(r['p99_ratio'] for r in s['joints'] if r):.3f}, "
                f"max/limit {max(r['max_ratio'] for r in s['joints'] if r):.3f}"
                for s in sp.values()
            )
        )
    lines.append(
        "\nsummary (worst joint): "
        + "; ".join(
            f"{s['title']} p99/limit {max(r['p99_ratio'] for r in s['joints'] if r and 'p99_ratio' in r):.2f}, "
            f"max/limit {max(r['max_ratio'] for r in s['joints'] if r and 'max_ratio' in r):.2f}"
            for s in res["stats"].values()
            if any(r and "p99_ratio" in r for r in s["joints"])
        )
    )
    return "\n".join(lines)


def measure(sport, cases=None, episodes=None, out_json=None):
    body, pattern, decimate = arm_decl(sport)
    gdir = sportlib.SPORTS / sport / sportlib.decl(sport).get("golden", {}).get("episodes", "golden")
    srcs = (
        [Path(e) for e in episodes]
        if episodes
        else sorted(p for p in gdir.glob("*/") if (p / "expected.json").exists() and (not cases or p.name in cases))
    )
    with tempfile.TemporaryDirectory(prefix=f"armlim-{sport}-") as tmp:
        sim = sportlib.assemble_sim(sport, Path(tmp) / "sim")
        done = [run_case(sport, sim, s, Path(tmp) / "cases", pattern) for s in srcs]
        res = analyse(sport, Path(tmp), done, body, decimate)
        res["sport"] = sport
        res["cases"] = [{k: v for k, v in c.items() if k != "dir"} for c in done]
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"{sport}: cases {[(c['case'], c['steps'], c.get('replay_ok')) for c in res['cases']]}")
    print(table(res))
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure",))
    ap.add_argument("sport")
    ap.add_argument("--cases", default="")
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    measure(a.sport, [c for c in a.cases.split(",") if c], a.episodes, a.json)


if __name__ == "__main__":
    main(sys.argv[1:])
