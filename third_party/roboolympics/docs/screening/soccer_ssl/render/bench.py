"""Render-cost benchmark of the SSL overhead cameras (C line). Run inside the competition image, CPU-limited like a
match (the sim container's render worker gets render_threads llvmpipe threads):

    docker run --rm --cpus 2 -e LP_NUM_THREADS=2 -v $REPO:/srv:ro -v $OUT:/out \\
        ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 python /srv/docs/screening/soccer_ssl/render/bench.py --tag cpus2 \\
        [--scene stub|real] [--quick]

Every configuration runs in a fresh process (clean memory numbers). Per configuration: the wall time to draw one
observation (all cameras of the layout, a new random state each time: robots and ball moved), the core's tiling + PNG
encoding of that observation (core/observe.compose + png, what the seat receives), the PNG size, the process's peak
resident memory before / after the off-screen contexts, and for reference the cost of one 10 ms control step of
physics (10 x mj_step) of the same scene. Output: one JSON line per configuration in /out/bench_<tag>.jsonl."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parents[4] / "sports" / "soccer_ssl" / "sim"))

SIZES = {
    "ssl4": [(400, 300), (640, 480), (800, 600), (1024, 768), (1280, 960)],
    "ssl2": [(300, 400), (480, 640), (600, 800), (768, 1024), (960, 1280)],
    "single": [(600, 420), (960, 672), (1200, 840), (1600, 1120), (2000, 1400)],
}


def rss_kb(field="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return -1


def one(cfg: dict) -> dict:
    import numpy as np
    import mujoco as mj
    import camera
    from core.observe import compose, png
    if cfg.get("scene", "stub") == "real":
        import real_scene as scene_mod
    else:
        import stub_scene as scene_mod
    rss0 = rss_kb()
    spec, names, _ = scene_mod.build(cfg["layout"], size=tuple(cfg["size"]), offsamples=cfg.get("offsamples"),
                                     shadows=cfg.get("shadows", False))
    m = spec.compile()
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    rng = np.random.default_rng(7)
    states = [scene_mod.random_state(m, rng)[0] for _ in range(cfg["n"] + cfg["warmup"])]
    rss_model = rss_kb()
    path = cfg.get("path", "camera")
    if path == "core":                                   # the core's renderer from CameraDefs (what CameraWorker uses)
        from renderers.base import Scene
        from renderers.mujoco import MujocoRenderer
        r = MujocoRenderer(Scene(xml=spec.to_xml()), n_cams=len(names), size=tuple(cfg["size"]))
        defs = camera.camera_defs(cfg["layout"], cfg["size"])

        def draw(q):
            return r.render({"qpos": q}, defs)
    else:
        def draw(q):
            d.qpos[:] = q
            mj.mj_kinematics(m, d)
            mj.mj_camlight(m, d)
            return camera.render(m, d, names, tuple(cfg["size"]))
    t_render, t_png, nbytes = [], [], []
    for i, q in enumerate(states):
        t0 = time.perf_counter()
        imgs = draw(q)
        t1 = time.perf_counter()
        sheet, _ = compose(imgs, names, 2)
        b = png(sheet)
        t2 = time.perf_counter()
        if i >= cfg["warmup"]:
            t_render.append((t1 - t0) * 1e3)
            t_png.append((t2 - t1) * 1e3)
            nbytes.append(len(b))
    rss_after = rss_kb()
    hwm = rss_kb("VmHWM")
    # physics of one 10 ms control step (10 x 1 ms), for scale
    d.qpos[:] = states[0]
    mj.mj_forward(m, d)
    t0 = time.perf_counter()
    for _ in range(100):
        mj.mj_step(m, d)
    t_phys = (time.perf_counter() - t0) / 10 * 1e3

    def stats(v):
        v = sorted(v)
        return {"median": round(v[len(v) // 2], 2), "p90": round(v[int(0.9 * (len(v) - 1))], 2),
                "min": round(v[0], 2), "mean": round(sum(v) / len(v), 2)}

    return {**cfg, "cams": len(names), "pixels_per_obs": len(names) * cfg["size"][0] * cfg["size"][1],
            "render_ms": stats(t_render), "png_ms": stats(t_png), "png_kb": round(sum(nbytes) / len(nbytes) / 1024, 1),
            "rss_mb": {"start": round(rss0 / 1024, 1), "model": round(rss_model / 1024, 1),
                       "after": round(rss_after / 1024, 1), "peak": round(hwm / 1024, 1)},
            "physics_ms_per_control_step": round(t_phys, 3),
            "env": {k: os.environ.get(k) for k in ("MUJOCO_GL", "LP_NUM_THREADS")}, "loadavg": os.getloadavg()[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scene", default="stub")
    ap.add_argument("--quick", action="store_true", help="only the default tile sizes")
    ap.add_argument("--extras", action="store_true", help="also MSAA off, shadows on, the core renderer path")
    ap.add_argument("--layouts", default="single,ssl2,ssl4")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--offsamples", type=int, default=None, help="MSAA samples of the sweep (MuJoCo default 4)")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(one(json.loads(a.one))), flush=True)
        return
    import camera
    cfgs = []
    for layout in a.layouts.split(","):
        sizes = [camera.DEFAULT_SIZE[layout]] if a.quick else SIZES[layout]
        cfgs += [{"layout": layout, "size": list(s), "offsamples": a.offsamples} for s in sizes]
    if a.extras:
        base = {"layout": "ssl4", "size": list(camera.DEFAULT_SIZE["ssl4"])}
        cfgs += [{**base, "offsamples": 0}, {**base, "shadows": True}, {**base, "path": "core"}]
    out = Path("/out") / f"bench_{a.tag}.jsonl"
    with out.open("a") as f:
        for c in cfgs:
            c.update(n=a.n, warmup=3, scene=a.scene, tag=a.tag)
            p = subprocess.run([sys.executable, __file__, "--tag", a.tag, "--one", json.dumps(c)],
                               capture_output=True, text=True)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print("FAILED", c, p.stderr[-2000:], flush=True)
                continue
            res = json.loads(line[0][7:])
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['layout']:6s} {res['size'][0]:4d}x{res['size'][1]:<4d} x{res['cams']} "
                  f"{'msaa0 ' if c.get('offsamples') == 0 else ''}{'shadow ' if c.get('shadows') else ''}"
                  f"{'core ' if c.get('path') == 'core' else ''}"
                  f"render {res['render_ms']['median']:7.1f} ms (p90 {res['render_ms']['p90']:7.1f})  "
                  f"png {res['png_ms']['median']:6.1f} ms {res['png_kb']:6.0f} KB  peak {res['rss_mb']['peak']:6.0f} MB  "
                  f"phys {res['physics_ms_per_control_step']:.2f} ms  load {res['loadavg']:.0f}", flush=True)


if __name__ == "__main__":
    main()
