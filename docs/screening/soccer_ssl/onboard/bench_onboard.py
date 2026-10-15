"""Render cost of the onboard front cameras (10 per observation: both teams' 5 robots), measured like line C
(docs/screening/soccer_ssl/render/bench.py): inside the competition image, CPU-limited, one fresh process per
configuration, a new random state before every observation (robots and ball moved), warm-up 3, then n observations.

    docker run --rm --cpus 2 -e LP_NUM_THREADS=2 -v $REPO:/srv:ro -v $OUT:/out -w /srv \\
        ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 python docs/screening/soccer_ssl/onboard/bench_onboard.py --tag cpus2

Per configuration: render time of the 10 pictures (and per camera), the core's tiling + PNG of one team's 5 pictures
(core/observe.compose + png: what one seat would receive), PNG size, the process's resident memory (start, after the
model, peak). One JSON line per configuration in /out/bench_onboard_<tag>.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

SIZES = [(160, 120), (320, 240), (480, 360), (640, 480), (800, 600), (1280, 960)]


def rss_kb(field="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return -1


def random_state(w, rng):
    """robots anywhere on the field (0.25 m apart), random headings; the ball on the field."""
    import numpy as np
    d = w.data
    pts = []
    for rid in w.ids:
        while True:
            p = (float(rng.uniform(-4.4, 4.4)), float(rng.uniform(-2.9, 2.9)))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.25 for q in pts):
                break
        pts.append(p)
        w.base[rid].set_pose(d, (p[0], p[1], float(rng.uniform(-math.pi, math.pi))))
    w.ball.place(float(rng.uniform(-4.4, 4.4)), float(rng.uniform(-2.9, 2.9)))
    return np.array(d.qpos)


def one(cfg: dict) -> dict:
    import numpy as np
    import mujoco as mj
    from core.observe import compose, png
    from sports.soccer_ssl.sim import onboard as OB
    from sports.soccer_ssl.sim.world import World
    rss0 = rss_kb()
    with OB.world_builder(size=tuple(cfg["size"]), msaa=cfg.get("msaa", False),
                          team_band=cfg.get("band", True), hfov_deg=cfg.get("hfov", OB.HFOV_DEG)) as cams:
        w = World(seed=1)
    m, d = w.model, w.data
    names = [cams[f"{t}{n}_robot"] for t, n in w.ids]
    rng = np.random.default_rng(7)
    states = [random_state(w, rng) for _ in range(cfg["n"] + cfg["warmup"])]
    rss_model = rss_kb()
    t_render, t_png, nbytes = [], [], []
    for i, q in enumerate(states):
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        t0 = time.perf_counter()
        imgs = OB.render(m, d, names, tuple(cfg["size"]))
        t1 = time.perf_counter()
        sheet, _ = compose(imgs[:5], names[:5], 5)       # one team's five pictures (one seat)
        b = png(sheet)
        t2 = time.perf_counter()
        if i >= cfg["warmup"]:
            t_render.append((t1 - t0) * 1e3)
            t_png.append((t2 - t1) * 1e3)
            nbytes.append(len(b))
    hwm = rss_kb("VmHWM")

    def stats(v):
        v = sorted(v)
        return {"median": round(v[len(v) // 2], 2), "p90": round(v[int(0.9 * (len(v) - 1))], 2),
                "min": round(v[0], 2)}

    return {**cfg, "cams": len(names), "render10_ms": stats(t_render),
            "per_cam_ms": round(sorted(t_render)[len(t_render) // 2] / len(names), 2),
            "png5_ms": stats(t_png), "png5_kb": round(sum(nbytes) / len(nbytes) / 1024, 1),
            "rss_mb": {"start": round(rss0 / 1024, 1), "model": round(rss_model / 1024, 1), "peak": round(hwm / 1024, 1)},
            "env": {k: os.environ.get(k) for k in ("MUJOCO_GL", "LP_NUM_THREADS")}, "loadavg": os.getloadavg()[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="only 320x240 and 640x480")
    ap.add_argument("--extras", action="store_true", help="also MSAA on, the team band, a 120 deg lens at 640x480")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(one(json.loads(a.one))), flush=True)
        return
    sizes = [(320, 240), (640, 480)] if a.quick else SIZES
    cfgs = [{"size": list(s)} for s in sizes]
    if a.extras:
        cfgs += [{"size": [640, 480], "msaa": True}, {"size": [640, 480], "band": True},
                 {"size": [640, 480], "hfov": 120.0}]
    out = Path("/out") / f"bench_onboard_{a.tag}.jsonl"
    with out.open("a") as f:
        for c in cfgs:
            c.update(n=a.n, warmup=3, tag=a.tag)
            p = subprocess.run([sys.executable, __file__, "--tag", a.tag, "--one", json.dumps(c)],
                               capture_output=True, text=True)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print("FAILED", c, p.stderr[-2000:], flush=True)
                continue
            r = json.loads(line[0][7:])
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps({k: r[k] for k in ("size", "render10_ms", "per_cam_ms", "png5_ms", "png5_kb", "rss_mb")}
                             | {"msaa": c.get("msaa", False), "band": c.get("band", False)}), flush=True)


if __name__ == "__main__":
    main()
