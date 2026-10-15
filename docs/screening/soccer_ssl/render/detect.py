"""Is the overhead picture enough? A plain colour-threshold detector (numpy only, no tracking, no filtering) that finds
the ball and every robot's team, id, position and heading in the SSL camera tiles, scored against the true state.
It is not a reference player: it only shows that the picture carries the information (C line, task 4).

Per camera tile: HSV thresholds -> blobs (4-connected) -> blob centroids back-projected onto the plane of the robots'
tops (team dots, id dots) or of the ball's centre (camera.ground_point, the published pinhole calibration). A robot =
a blue / yellow centre blob with its four pink / green dots within 4-9 cm (on the top plane): the dots' angular order
gives the heading (camera.PATTERN: front dots at +-57.4 deg, back dots at +-147.4 deg from the heading), their
colours the id. Robots and the ball seen by several cameras (the seams) are taken from the camera whose axis is
nearest (least parallax).

    docker run --rm --cpus 2 -e LP_NUM_THREADS=2 -v $REPO:/srv:ro -v $OUT:/out ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 \\
        python /srv/docs/screening/soccer_ssl/render/detect.py --tag stub [--scene stub|real] [--n 40]
-> /out/detect_<tag>.json (per configuration: detection rates, position errors in m and px, heading errors, id and
team mistakes) and /out/detect_<tag>_<layout>_<w>x<h>.png (one tile with the detections drawn)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parents[4] / "sports" / "soccer_ssl" / "sim"))
import camera  # noqa: E402

FRONT = math.atan2(0.054772, 0.035)          # 57.42 deg
BACK = math.atan2(0.035, -0.054772)          # 147.42 deg
TEMPLATE = {"FL": FRONT, "BL": BACK, "BR": -BACK, "FR": -FRONT}
CODE_TO_ID = {c: i for i, c in enumerate(camera.PATTERN["ids"])}


def hsv(img):
    x = img[..., :3].astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(-1)
    mn = x.min(-1)
    c = v - mn
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0)
    cc = np.maximum(c, 1e-6)
    h = np.where(v == r, ((g - b) / cc) % 6, np.where(v == g, (b - r) / cc + 2, (r - g) / cc + 4)) * 60.0
    return h, s, v


def masks(img):
    h, s, v = hsv(img)
    return {
        "ball": (h > 12) & (h < 42) & (s > 0.6) & (v > 0.5),
        "yellow": (h > 48) & (h < 72) & (s > 0.5) & (v > 0.55),
        "G": (h > 95) & (h < 145) & (s > 0.6) & (v > 0.7),          # the carpet green is far darker (v ~ 0.45)
        "blue": (h > 215) & (h < 265) & (s > 0.6) & (v > 0.45),
        "P": (h > 280) & (h < 320) & (s > 0.5) & (v > 0.5),
    }


def blobs(mask, min_px=3):
    """4-connected components (min-label propagation with pointer jumping over the mask's pixels only)
    -> [(cy, cx, area)] (pixel-centre coordinates)."""
    ys, xs = np.nonzero(mask)
    n = ys.size
    if n == 0:
        return []
    H, W = mask.shape
    idx = np.full((H + 2, W + 2), -1, np.int64)
    idx[ys + 1, xs + 1] = np.arange(n)
    nbs = [idx[ys + 1 + dy, xs + 1 + dx] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    lab = np.arange(n)
    while True:
        new = lab.copy()
        for nb in nbs:
            ok = nb >= 0
            new[ok] = np.minimum(new[ok], lab[nb[ok]])
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    ids, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    sy = np.bincount(inv, ys + 0.5)
    sx = np.bincount(inv, xs + 0.5)
    return [(sy[k] / cnt[k], sx[k] / cnt[k], int(cnt[k])) for k in range(len(ids)) if cnt[k] >= min_px]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def detect_tile(img, cam, top_z, ball_z):
    """-> {"robots": [...], "balls": [...]} in world coordinates, with the camera's axis distance for merging."""
    mk = masks(img)
    bl = {k: blobs(v) for k, v in mk.items()}
    ax = np.array(cam["pos"][:2])

    def world(b, z):
        p = camera.ground_point(cam, b[1], b[0], z)
        return p[:2]

    dots = [(world(b, top_z), col) for col in ("P", "G") for b in bl[col]]
    robots = []
    for team in ("blue", "yellow"):
        for b in bl[team]:
            if b[2] < 6:
                continue
            c = world(b, top_z)
            near = [(p, col) for p, col in dots if 0.035 < np.linalg.norm(p - c) < 0.095]
            rob = {"team": team, "x": float(c[0]), "y": float(c[1]), "axis_d": float(np.linalg.norm(c - ax)),
                   "px": (b[1], b[0]), "ndots": len(near)}
            if len(near) == 4:
                ang = sorted((math.atan2(p[1] - c[1], p[0] - c[0]), col) for p, col in near)
                order = ["FL", "BL", "BR", "FR"]          # counter-clockwise from the front-left dot
                best = None
                for k in range(4):
                    labels = order[k:] + order[:k]
                    diffs = [wrap(a - TEMPLATE[lab]) for (a, _), lab in zip(ang, labels)]
                    th = math.atan2(sum(math.sin(d) for d in diffs), sum(math.cos(d) for d in diffs))
                    res = sum(wrap(d - th) ** 2 for d in diffs)
                    if best is None or res < best[0]:
                        best = (res, th, {lab: col for (_, col), lab in zip(ang, labels)})
                _, th, cols = best
                rob["yaw"] = th
                rob["id"] = CODE_TO_ID.get("".join(cols[k] for k in ("FL", "FR", "BL", "BR")))
            robots.append(rob)
    balls = []
    for b in bl["ball"]:
        if b[2] < 3:
            continue
        p = world(b, ball_z)
        balls.append({"x": float(p[0]), "y": float(p[1]), "axis_d": float(np.linalg.norm(p - ax)), "px": (b[1], b[0]),
                      "area": b[2]})
    return {"robots": robots, "balls": balls}


def merge(per_cam):
    robots, balls = [], []
    for det in per_cam:
        for r in det["robots"]:
            same = [o for o in robots if o["team"] == r["team"] and math.hypot(o["x"] - r["x"], o["y"] - r["y"]) < 0.1]
            if not same:
                robots.append(r)
            else:
                o = same[0]
                better = ((r.get("id") is not None, -r["axis_d"]) > (o.get("id") is not None, -o["axis_d"]))
                if better:
                    robots[robots.index(o)] = r
        for b in det["balls"]:
            same = [o for o in balls if math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.1]
            if not same:
                balls.append(b)
            elif b["axis_d"] < same[0]["axis_d"]:
                balls[balls.index(same[0])] = b
    return robots, balls


def score(robots, balls, truth, mpp):
    out = {"robot_found": 0, "robot_id_ok": 0, "robot_team_ok": 0, "robot_pos_m": [], "robot_pos_px": [],
           "robot_yaw_deg": [], "robot_false": 0, "ball_found": 0, "ball_pos_m": [], "ball_pos_px": [], "ball_false": 0}
    used = set()
    for t in truth["robots"]:
        cands = [(math.hypot(r["x"] - t["x"], r["y"] - t["y"]), i) for i, r in enumerate(robots) if i not in used]
        cands = [c for c in cands if c[0] < 0.09]
        if not cands:
            continue
        e, i = min(cands)
        used.add(i)
        r = robots[i]
        out["robot_found"] += 1
        out["robot_team_ok"] += r["team"] == t["team"]
        out["robot_id_ok"] += (r["team"] == t["team"] and r.get("id") == t["number"])
        out["robot_pos_m"].append(e)
        out["robot_pos_px"].append(e / mpp)
        if "yaw" in r:
            out["robot_yaw_deg"].append(abs(math.degrees(wrap(r["yaw"] - t["yaw"]))))
    out["robot_false"] = len(robots) - len(used)
    tb = truth["ball"]
    if balls:
        e = min(math.hypot(b["x"] - tb["x"], b["y"] - tb["y"]) for b in balls)
        if e < 0.1:
            out["ball_found"] = 1
            out["ball_pos_m"].append(e)
            out["ball_pos_px"].append(e / mpp)
        out["ball_false"] = len(balls) - (1 if e < 0.1 else 0)
    return out


def draw(img, robots_tile, balls_tile):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for r in robots_tile:
        u, v = r["px"]
        d.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(255, 255, 255))
        if "yaw" in r:
            d.line([u, v, u + 22 * math.cos(r["yaw"]), v - 22 * math.sin(r["yaw"])], fill=(255, 255, 255), width=2)
        d.text((u + 13, v - 14), f"{r['team'][0].upper()}{r.get('id')}", fill=(255, 255, 255))
    for b in balls_tile:
        u, v = b["px"]
        d.rectangle([u - 7, v - 7, u + 7, v + 7], outline=(255, 255, 255))
    return im


def run(cfg, n, seed, outdir, tag, scene_mod, top_z):
    import mujoco as mj
    spec, names, _ = scene_mod.build(cfg["layout"], size=tuple(cfg["size"]), offsamples=cfg.get("offsamples"))
    m = spec.compile()
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    defs = camera.camera_defs(cfg["layout"], cfg["size"])
    rng = np.random.default_rng(seed)
    tot = None
    worst_ball = []
    for k in range(n):
        q, truth = scene_mod.random_state(m, rng)
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        imgs = camera.render(m, d, names, tuple(cfg["size"]))
        per = [detect_tile(img, cam, top_z, scene_mod.BALL_R) for img, cam in zip(imgs, defs)]
        robots, balls = merge(per)
        s = score(robots, balls, truth, camera.metres_per_pixel(defs[0], top_z))
        tb = truth["ball"]
        near = min(math.hypot(r["x"] - tb["x"], r["y"] - tb["y"]) for r in truth["robots"])
        at_robot = near < 0.1                       # touching a robot's front (the dribbler spot)
        s["ball_at_robot"] = int(at_robot)
        s["ball_found_at_robot"] = s["ball_found"] if at_robot else 0
        if not s["ball_found"]:
            worst_ball.append({"scene": k, "ball": tb, "nearest_robot_m": round(near, 4),
                               "dets": [(b["x"], b["y"]) for b in balls]})
        if tot is None:
            tot = s
        else:
            for key, v in s.items():
                tot[key] = tot[key] + v
        if k == 0:
            draw(imgs[0], per[0]["robots"], per[0]["balls"]).save(
                Path(outdir) / f"detect_{tag}_{cfg['layout']}_{cfg['size'][0]}x{cfg['size'][1]}"
                               f"{'_msaa0' if cfg.get('offsamples') == 0 else ''}.png")
    nr = n * 10

    def q(v, p):
        return round(float(np.percentile(v, p)), 4) if v else None

    return {**cfg, "scenes": n, "mm_per_px_top": round(1e3 * camera.metres_per_pixel(defs[0], top_z), 2),
            "robot_found": tot["robot_found"] / nr, "robot_team_ok": tot["robot_team_ok"] / nr,
            "robot_id_ok": tot["robot_id_ok"] / nr, "robot_false": tot["robot_false"],
            "robot_pos_m": {"median": q(tot["robot_pos_m"], 50), "p95": q(tot["robot_pos_m"], 95),
                            "max": q(tot["robot_pos_m"], 100)},
            "robot_pos_px": {"median": q(tot["robot_pos_px"], 50), "p95": q(tot["robot_pos_px"], 95)},
            "robot_yaw_deg": {"median": q(tot["robot_yaw_deg"], 50), "p95": q(tot["robot_yaw_deg"], 95),
                              "max": q(tot["robot_yaw_deg"], 100)},
            "ball_found": tot["ball_found"] / n, "ball_false": tot["ball_false"],
            "ball_found_free": (tot["ball_found"] - tot["ball_found_at_robot"]) / max(1, n - tot["ball_at_robot"]),
            "ball_found_at_robot": tot["ball_found_at_robot"] / max(1, tot["ball_at_robot"]),
            "ball_at_robot_scenes": tot["ball_at_robot"],
            "ball_pos_m": {"median": q(tot["ball_pos_m"], 50), "p95": q(tot["ball_pos_m"], 95),
                           "max": q(tot["ball_pos_m"], 100)},
            "ball_pos_px": {"median": q(tot["ball_pos_px"], 50), "p95": q(tot["ball_pos_px"], 95)},
            "ball_missed": worst_ball[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scene", default="stub")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.scene == "real":
        import real_scene as scene_mod
    else:
        import stub_scene as scene_mod
    top_z = getattr(scene_mod, "TOP_Z", scene_mod.ROBOT_H + 0.001)
    cfgs = [{"layout": "ssl4", "size": [640, 480]}, {"layout": "ssl4", "size": [800, 600]},
            {"layout": "ssl4", "size": [1024, 768]}, {"layout": "ssl4", "size": [640, 480], "offsamples": 0},
            {"layout": "ssl4", "size": [800, 600], "offsamples": 0}, {"layout": "ssl2", "size": [600, 800]},
            {"layout": "single", "size": [1200, 840]}, {"layout": "single", "size": [1200, 840], "offsamples": 0},
            {"layout": "single", "size": [960, 672]}, {"layout": "ssl4", "size": [400, 300]}]
    res = []
    for c in cfgs:
        r = run(c, a.n, 11, "/out", a.tag, scene_mod, top_z)
        res.append(r)
        print(f"{c['layout']:6s} {c['size'][0]}x{c['size'][1]} msaa{0 if c.get('offsamples') == 0 else 4} "
              f"{r['mm_per_px_top']} mm/px | robots found {r['robot_found']:.3f} id ok {r['robot_id_ok']:.3f} "
              f"false {r['robot_false']} pos {r['robot_pos_m']['median'] * 1e3:.1f}/{r['robot_pos_m']['p95'] * 1e3:.1f} mm "
              f"({r['robot_pos_px']['median']:.2f}/{r['robot_pos_px']['p95']:.2f} px) yaw {r['robot_yaw_deg']['median']:.2f}/"
              f"{r['robot_yaw_deg']['p95']:.2f} deg | ball found {r['ball_found']:.3f} (free {r['ball_found_free']:.3f}, at robot {r['ball_found_at_robot']:.3f}) false {r['ball_false']} "
              f"pos {1e3 * (r['ball_pos_m']['median'] or 0):.1f}/{1e3 * (r['ball_pos_m']['p95'] or 0):.1f} mm", flush=True)
    Path(f"/out/detect_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
