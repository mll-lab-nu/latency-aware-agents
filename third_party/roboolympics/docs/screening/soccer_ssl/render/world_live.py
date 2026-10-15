"""The cameras inside B's running world (sim/world.World(cameras=True), A's robots): robots driven to random targets,
the ball shot across the field every 0.5 s at 2-6.5 m/s; every `every` control steps the four tiles are drawn from the
live MjData (camera.render) and read by detect.py's detector, scored against World.snapshot(). Also times
World.step() against the render, i.e. the render share of the sim's work per observation.

    docker run --rm --cpus 2 -e LP_NUM_THREADS=2 -v $REPO:/srv:ro -v $OUT:/out -w /srv \\
        ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 python docs/screening/soccer_ssl/render/world_live.py [--seconds 12]
-> /out/world_live.json"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parents[4] / "sports" / "soccer_ssl" / "sim"))
import camera  # noqa: E402
import detect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--every", type=int, default=2, help="control steps between observations (2 = 50 Hz game time)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    from sports.soccer_ssl.sim.world import World
    w = World(seed=a.seed, cameras=True)
    m, d = w.model, w.data
    defs = camera.camera_defs()
    top_z = float(w.body.load_decl()["geometry"]["height"]) + 0.001
    from sports.soccer_ssl.sim import ball as BALL
    ball_r = float(BALL.RADIUS)
    rng = np.random.default_rng(a.seed)
    t_step, t_render, t_detect, tot, n_obs, ball_speeds, missed = [], [], [], None, 0, [], []
    steps = int(round(a.seconds / 0.01))
    for k in range(steps):
        if k % 100 == 0:                                  # new random targets every second (team frames)
            for team, num in w.ids:
                w.apply(team, num, "move", dict(x=float(rng.uniform(-4.2, 4.2)), y=float(rng.uniform(-2.8, 2.8)),
                                                theta=float(rng.uniform(-math.pi, math.pi))))
        if k % 50 == 0:                                   # shoot the ball from where it is (every 0.5 s)
            p = w.ball.pos
            x, y = float(np.clip(p[0], -4.0, 4.0)), float(np.clip(p[1], -2.6, 2.6))
            ang, sp = rng.uniform(-math.pi, math.pi), rng.uniform(2.0, 6.5)
            w.ball.place(x, y, vel=(sp * math.cos(ang), sp * math.sin(ang), 0.0))
        t0 = time.perf_counter()
        w.step()
        t_step.append(time.perf_counter() - t0)
        if k % a.every:
            continue
        t0 = time.perf_counter()
        imgs = camera.render(m, d, w.camera_names)
        t1 = time.perf_counter()
        per = [detect.detect_tile(img, cam, top_z, ball_r) for img, cam in zip(imgs, defs)]
        robots, balls = detect.merge(per)
        t_detect.append(time.perf_counter() - t1)
        t_render.append(t1 - t0)
        snap = w.snapshot()
        truth = {"robots": [{"team": r["team"], "number": r["number"], "x": r["pose"][0], "y": r["pose"][1],
                             "yaw": r["pose"][2]} for r in snap["robots"] if r["on_field"]],
                 "ball": {"x": snap["ball"]["pos"][0], "y": snap["ball"]["pos"][1]}}
        s = detect.score(robots, balls, truth, camera.metres_per_pixel(defs[0], top_z))
        bv = float(np.hypot(*snap["ball"]["vel"][:2]))
        ball_speeds.append(bv)
        if not s["ball_found"]:
            near = min(math.hypot(r["x"] - truth["ball"]["x"], r["y"] - truth["ball"]["y"]) for r in truth["robots"])
            missed.append({"t": snap["t"], "ball": snap["ball"]["pos"], "speed": round(bv, 2),
                           "nearest_robot_m": round(near, 3)})
        n_obs += 1
        s["n_robots"] = len(truth["robots"])
        tot = s if tot is None else {key: tot[key] + v for key, v in s.items()}

    def q(v, p):
        return round(float(np.percentile(v, p)), 4) if len(v) else None

    res = {"seconds": a.seconds, "observations": n_obs, "every_control_steps": a.every,
           "world_step_ms": {"median": q(np.array(t_step) * 1e3, 50), "p90": q(np.array(t_step) * 1e3, 90)},
           "render_ms": {"median": q(np.array(t_render) * 1e3, 50), "p90": q(np.array(t_render) * 1e3, 90)},
           "detect_ms_numpy": {"median": q(np.array(t_detect) * 1e3, 50)},
           "robots_seen": tot["n_robots"], "robot_found": tot["robot_found"], "robot_id_ok": tot["robot_id_ok"],
           "robot_false": tot["robot_false"],
           "robot_pos_m": {"median": q(tot["robot_pos_m"], 50), "p95": q(tot["robot_pos_m"], 95)},
           "robot_yaw_deg": {"median": q(tot["robot_yaw_deg"], 50), "p95": q(tot["robot_yaw_deg"], 95)},
           "ball_found": tot["ball_found"], "ball_false": tot["ball_false"],
           "ball_pos_m": {"median": q(tot["ball_pos_m"], 50), "p95": q(tot["ball_pos_m"], 95),
                          "max": q(tot["ball_pos_m"], 100)},
           "ball_speed_m_s": {"median": q(ball_speeds, 50), "max": q(ball_speeds, 100)}, "ball_missed": missed[:20]}
    print(json.dumps(res, indent=1))
    Path("/out/world_live.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
