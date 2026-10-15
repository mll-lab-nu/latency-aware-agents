"""The onboard prototype on a recorded match (record.py): every robot re-draws its front camera at `fps` (game time)
from the recorded qpos, its odometry and gyro are simulated (sensors.py), a particle filter tracks its pose from
the told kick-off spot (localize.py), and the detector (perceive.py) reports the ball and robots. Everything is
scored against the recorded truth.

    docker run --rm --cpus 2 -e LP_NUM_THREADS=2 -v $REPO:/srv -w /srv ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 \\
        python docs/screening/soccer_ssl/onboard/run_onboard.py --rec results/rec_L2L3_s1.npz --tag L2L3_s1

Output: results/run_<tag>.json (summary) and results/run_<tag>_series.npz (per frame and robot)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import localize as LZ  # noqa: E402
import perceive as PC  # noqa: E402
import sensors as SN  # noqa: E402
from sports.soccer_ssl.sim import onboard as OB  # noqa: E402


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_world(pose, x, y):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * x - s * y, pose[1] + s * x + c * y


def to_robot(pose, x, y):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = x - pose[0], y - pose[1]
    return c * dx + s * dy, -s * dx + c * dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--hfov", type=float, default=OB.HFOV_DEG)
    ap.add_argument("--pitch", type=float, default=OB.PITCH_DEG)
    ap.add_argument("--no-band", action="store_true", help="robots without the side team colour (rules §3.2.3)")
    ap.add_argument("--particles", type=int, default=300)
    ap.add_argument("--drift", type=float, default=1.0, help="multiplies every odometry / gyro error")
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--no-boundary", action="store_true")
    ap.add_argument("--robots", default="all", help="comma list like blue0,yellow3 or 'all'")
    ap.add_argument("--save-frames", type=int, default=0, help="save every n-th picture of the first robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pitch-jitter", type=float, default=0.0,
                    help="deg: per picture the camera's true pitch differs from the calibrated one (chassis tilt, vibration)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian pixel noise (0-255 scale) added to every picture")
    ap.add_argument("--kidnap", type=float, default=None,
                    help="at this game time every filter is displaced by 0.8 m and 20 deg (a lost robot)")
    ap.add_argument("--pitch-robust", action="store_true",
                    help="the filter takes out a common pitch offset per particle (see localize.py)")
    ap.add_argument("--global-init", action="store_true",
                    help="particles uniform over the field at the start (no told kick-off spot)")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    import mujoco as mj
    from sports.soccer_ssl.sim.world import World
    R = np.load(a.rec, allow_pickle=False)
    meta = json.loads(str(R["meta"]))
    with OB.world_builder(size=size, hfov_deg=a.hfov, pitch_deg=a.pitch, team_band=not a.no_band) as cams:
        w = World(seed=meta["seed"], first_kickoff="blue" if meta["seed"] % 2 else "yellow")
    m, d = w.model, w.data
    assert m.nq == R["qpos"].shape[1], (m.nq, R["qpos"].shape)
    ids = [f"{t}{n}" for t, n in w.ids]
    assert ids == meta["ids"]
    sel = list(range(len(ids))) if a.robots == "all" else [ids.index(r) for r in a.robots.split(",")]
    cam_names = [cams[f"{ids[i]}_robot"] for i in range(len(ids))]
    body_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{ids[i]}_robot") for i in range(len(ids))]
    ball_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "ball")
    cam = PC.Camera(size, a.hfov, a.pitch)
    T = R["t"].shape[0] if a.seconds is None else min(R["t"].shape[0], int(round(a.seconds / 0.01)))
    every = max(1, int(round(1.0 / a.fps / 0.01)))
    rng = np.random.default_rng(a.seed)
    pose, vel, vsp, on, tele = R["pose"], R["vel"], R["vsp"], R["on"], R["teleport"]
    odo = {i: SN.Odometry(np.random.default_rng(1000 + i + a.seed), pose0=pose[0, i], scale=a.drift) for i in range(len(ids))}
    pf = {i: LZ.ParticleFilter(pose[0, i], np.random.default_rng(2000 + i + a.seed), n=a.particles, f_px=cam.f,
                               cam_h=cam.h, cam_fwd=cam.fwd, use_lines=not a.no_lines,
                               use_boundary=not a.no_boundary, pitch_robust=a.pitch_robust) for i in sel}
    if a.global_init:
        for i in sel:
            P_ = pf[i].p
            P_[:, 0] = rng.uniform(-4.5, 4.5, len(P_))
            P_[:, 1] = rng.uniform(-3.0, 3.0, len(P_))
            P_[:, 2] = rng.uniform(-math.pi, math.pi, len(P_))
    lowfit = {i: 0 for i in sel}
    injections = 0
    last_odo = {i: odo[i].pose.copy() for i in sel}
    deadrec = {i: np.array(pose[0, i], float) for i in sel}     # odometry alone (same sensors), for comparison
    geomgroup = np.array([1, 1, 1, 0, 0, 0], np.uint8)
    gid = np.zeros(1, np.int32)
    rows = []           # per frame per robot
    ball_seen_team = []
    det_robot_rows = []
    timing = dict(render=0.0, perceive=0.0, pf=0.0, n=0)
    frame_dir = HERE / "results" / f"frames_{a.tag}"
    if a.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
    teleports = 0
    est_last = {i: np.array(pose[0, i], float) for i in sel}
    for k in range(T):
        # odometry every control step (all robots: the team-mate association uses every robot's estimate)
        for i in range(len(ids)):
            if tele[k, i]:
                odo[i].pose = np.array(pose[k, i], float)
                odo[i].gyro_yaw = float(pose[k, i, 2])
                continue
            vb = SN.body_velocity(vel[k, i], pose[k, i, 2])
            vs = SN.body_velocity(vsp[k, i], pose[k, i, 2])
            odo[i].step(vb, vs, 0.01)
        for i in sel:
            if tele[k, i]:
                # the referee moved the robot: the platform tells the agent the new pose (decision 5)
                pf[i].reset(pose[k, i])
                deadrec[i] = np.array(pose[k, i], float)
                last_odo[i] = odo[i].pose.copy()
                teleports += 1
        if k % every:
            continue
        d.qpos[:] = R["qpos"][k]
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        t0 = time.perf_counter()
        imgs = OB.render(m, d, [cam_names[i] for i in sel], size)
        timing["render"] += time.perf_counter() - t0
        ball_true = R["ball"][k, :3]
        seen_by = {"blue": False, "yellow": False}
        est_now = {}
        for j, i in enumerate(sel):
            # motion
            o_now, o_prev = odo[i].pose, last_odo[i]
            dxw, dyw = o_now[0] - o_prev[0], o_now[1] - o_prev[1]
            c, s = math.cos(o_prev[2]), math.sin(o_prev[2])
            dlt = (c * dxw + s * dyw, -s * dxw + c * dyw, wrap(o_now[2] - o_prev[2]))
            last_odo[i] = o_now.copy()
            cdr, sdr = math.cos(deadrec[i][2]), math.sin(deadrec[i][2])
            deadrec[i] += (cdr * dlt[0] - sdr * dlt[1], sdr * dlt[0] + cdr * dlt[1], dlt[2])
            t1 = time.perf_counter()
            img = imgs[j]
            if a.noise > 0:
                img = np.clip(img.astype(np.int16) + rng.normal(0, a.noise, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
            cam_i = cam
            if a.pitch_jitter > 0:      # the picture was taken at pitch + delta; the robot believes the calibrated pitch
                cam_i = PC.Camera(size, a.hfov, a.pitch - float(rng.normal(0, a.pitch_jitter)))
            C = PC.classes(img)
            us, vs_, kinds = PC.floor_boundary(C, cam)
            bx, by = cam_i.ground(us, vs_, 0.0)
            kk = np.array([{"wall": 0, "tall": 1, "goal": 2}[x] for x in kinds], int)
            ok = np.isfinite(bx)
            lines = PC.line_points(C, cam_i, rng=rng)
            bdet = PC.ball(C, cam_i)
            rdet = PC.robots(C, cam_i)
            t2 = time.perf_counter()
            if a.kidnap is not None and abs(R["t"][k] - a.kidnap) < 0.6 * every * 0.01:
                ang_ = float(rng.uniform(-math.pi, math.pi))
                pf[i].p[:, 0] += 0.8 * math.cos(ang_)
                pf[i].p[:, 1] += 0.8 * math.sin(ang_)
                pf[i].p[:, 2] = LZ.wrap(pf[i].p[:, 2] + math.radians(20) * (1 if rng.random() < 0.5 else -1))
            pf[i].predict(dlt)
            pf[i].update((bx[ok], by[ok], kk[ok]), lines)
            # recovery: when the map explains < 60 % of the floor-boundary columns for 5 pictures in a row, re-seed 10 % of the
            # particles within 1 m / 30 deg of the estimate (never globally: the field is point symmetric)
            lowfit[i] = lowfit[i] + 1 if pf[i].fit < 0.6 else 0
            if lowfit[i] >= 5:
                pf[i].inject(0.1, 1.0)
                lowfit[i] = 0
                injections += 1
            est = pf[i].estimate()
            t3 = time.perf_counter()
            timing["perceive"] += t2 - t1
            timing["pf"] += t3 - t2
            timing["n"] += 1
            est_now[i] = est
            est_last[i] = est
            tp = pose[k, i]
            err = math.hypot(est[0] - tp[0], est[1] - tp[1])
            yerr = abs(wrap(est[2] - tp[2]))
            dr_err = math.hypot(deadrec[i][0] - tp[0], deadrec[i][1] - tp[1])
            team = ids[i][:-1]
            # the ball: visible in truth?
            camw = np.array([*to_world(tp, cam.fwd, 0.0), cam.h])
            bxr, byr = to_robot(tp, ball_true[0], ball_true[1])
            u, v = cam.project(bxr, byr, ball_true[2])
            in_fov = bool(np.isfinite(u) and 0 <= u < size[0] and 0 <= v < size[1])
            vis = False
            if in_fov:
                vec = ball_true - camw
                x = mj.mj_ray(m, d, camw, vec, geomgroup, 1, body_ids[i], gid)
                vis = bool(0 <= x <= 1.0 + 1e-6 and m.geom_bodyid[gid[0]] == ball_body)
            bdet_err_r = bdet_err_w = None
            if bdet is not None:
                wx, wy = to_world(tp, bdet["x"], bdet["y"])
                bdet_err_r = math.hypot(wx - ball_true[0], wy - ball_true[1])
                ex, ey = to_world(est, bdet["x"], bdet["y"])
                bdet_err_w = math.hypot(ex - ball_true[0], ey - ball_true[1])
            if bdet is not None and bdet_err_r is not None and bdet_err_r < 0.5:
                seen_by[team] = True
            ball_rng = math.hypot(bxr - cam.fwd, byr)
            # robots: visibility of each other robot (centre ray at z = 0.1 m, and two side rays)
            vis_r = {}
            for jj in range(len(ids)):
                if jj == i or not on[k, jj]:
                    continue
                q = pose[k, jj]
                xr, yr = to_robot(tp, q[0], q[1])
                rr = math.hypot(xr - cam.fwd, yr)
                hits = 0
                for lat in (-0.07, 0.0, 0.07):
                    px_, py_ = xr - lat * yr / max(rr, 1e-6), yr + lat * (xr - cam.fwd) / max(rr, 1e-6)
                    uu, _ = cam.project(px_, py_, 0.1)
                    if not (np.isfinite(uu) and 0 <= uu < size[0]):
                        continue
                    tw = np.array([*to_world(tp, px_, py_), 0.1])
                    x = mj.mj_ray(m, d, camw, tw - camw, geomgroup, 1, body_ids[i], gid)
                    if 0 <= x and m.body_rootid[m.geom_bodyid[gid[0]]] == body_ids[jj]:
                        hits += 1
                if hits:
                    vis_r[jj] = (hits / 3.0, rr)
            # match detections (true pose -> world) to robots
            dets = []
            for r in rdet:
                wx, wy = to_world(tp, r["x"], r["y"])
                if abs(wx) > 4.4 and abs(wy) < 0.75:
                    continue     # a goal's black frame (the agent knows the map); not a robot
                dets.append((wx, wy, r))
            used = set()
            for jj, (frac, rr) in vis_r.items():
                q = pose[k, jj]
                best, bd = None, 0.5
                for n_, (wx, wy, r) in enumerate(dets):
                    dd = math.hypot(wx - q[0], wy - q[1])
                    if n_ not in used and dd < bd:
                        best, bd = n_, dd
                if best is not None:
                    used.add(best)
                    r = dets[best][2]
                    ex, ey = to_world(est, r["x"], r["y"])
                    # without a team colour: a team-mate if a team-mate's own estimate is within 0.35 m
                    mates = [est_last[q_] for q_ in sel if q_ != i and ids[q_][:-1] == team and on[k, q_]]
                    near = min((math.hypot(ex - e_[0], ey - e_[1]) for e_ in mates), default=9.0)
                    assoc = (0 if team == "blue" else 1) if near < 0.35 else (1 if team == "blue" else 0)
                    det_robot_rows.append((k, i, jj, frac, rr, 1, bd, math.hypot(ex - q[0], ey - q[1]),
                                           {"blue": 0, "yellow": 1, None: -1}[r["team"]],
                                           0 if ids[jj].startswith("blue") else 1, assoc))
                else:
                    det_robot_rows.append((k, i, jj, frac, rr, 0, -1, -1, -1, 0 if ids[jj].startswith("blue") else 1, -1))
            fp = len(dets) - len(used)
            rows.append((k, i, err, yerr, dr_err, pf[i].spread(), int(in_fov), int(vis), int(bdet is not None),
                         -1 if bdet_err_r is None else bdet_err_r, -1 if bdet_err_w is None else bdet_err_w,
                         ball_rng, fp, len(vis_r), int(odo[i].slipping), int(R["touch_robot"][k, i]),
                         abs(float(vel[k, i, 2]))))
            if a.save_frames and j == 0 and (k // every) % a.save_frames == 0:
                Image.fromarray(imgs[j]).save(frame_dir / f"{ids[i]}_{k:05d}.png")
        ball_seen_team.append((k, int(seen_by["blue"]), int(seen_by["yellow"])))
        if k % 1000 == 0:
            e = np.array([r[2] for r in rows[-len(sel):]])
            print(f"t={R['t'][k]:.1f} loc err median {np.median(e):.3f} max {e.max():.3f} "
                  f"render {timing['render']:.0f}s perceive {timing['perceive']:.0f}s pf {timing['pf']:.0f}s", flush=True)
    cols = "k i err yerr dr_err spread in_fov ball_vis ball_det ball_err_r ball_err_w ball_rng fp n_vis_robots slip touch wabs".split()
    S = np.array(rows, float)
    D = np.array(det_robot_rows, float) if det_robot_rows else np.zeros((0, 11))
    B = np.array(ball_seen_team, float)
    np.savez_compressed(HERE / "results" / f"run_{a.tag}_series.npz", rows=S, cols=np.array(cols), det=D, ball_team=B,
                        ids=np.array(ids), t=R["t"][:T])
    summary = summarize(S, cols, D, B, R["t"][:T], ids, sel)
    summary.update(tag=a.tag, rec=a.rec, fps=a.fps, size=size, hfov=a.hfov, pitch=a.pitch, band=not a.no_band,
                   particles=a.particles, drift=a.drift, lines=not a.no_lines, boundary=not a.no_boundary,
                   seconds=float(R["t"][T - 1]), teleports_reset=teleports, injections=injections,
                   pitch_jitter=a.pitch_jitter, pitch_robust=a.pitch_robust, noise=a.noise, kidnap=a.kidnap, global_init=a.global_init, meta={k: meta[k] for k in ("blue", "yellow", "seed", "score")},
                   timing_ms_per_robot_frame={k: round(1e3 * timing[k] / max(timing["n"], 1), 2) for k in ("render", "perceive", "pf")})
    (HERE / "results" / f"run_{a.tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)[:3000])


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def summarize(S, cols, D, B, t, ids, sel):
    c = {n: j for j, n in enumerate(cols)}
    out = {}
    tt = t[S[:, c["k"]].astype(int)]
    err = S[:, c["err"]]
    out["loc_err_m"] = dict(median=pct(err, 50), p90=pct(err, 90), p99=pct(err, 99), max=float(err.max()),
                            frac_over_0_3=float((err > 0.3).mean()), frac_over_1=float((err > 1.0).mean()))
    out["yaw_err_deg"] = dict(median=math.degrees(pct(S[:, c["yerr"]], 50)), p90=math.degrees(pct(S[:, c["yerr"]], 90)))
    out["deadreckon_err_m"] = dict(median=pct(S[:, c["dr_err"]], 50), p90=pct(S[:, c["dr_err"]], 90),
                                   final_median=None)
    # error over time: per 10 s bins
    bins = {}
    for lo in range(0, int(tt.max()) + 1, 10):
        msk = (tt >= lo) & (tt < lo + 10)
        if msk.any():
            bins[f"{lo}-{lo + 10}s"] = dict(loc_med=round(pct(err[msk], 50), 3), loc_p90=round(pct(err[msk], 90), 3),
                                           dr_med=round(pct(S[msk, c["dr_err"]], 50), 3))
    out["loc_err_by_time"] = bins
    # lost: a robot whose error exceeds 1 m at any time, and whether it came back under 0.3 m
    lost = {}
    for i in sel:
        msk = S[:, c["i"]] == i
        e = err[msk]
        lost[ids[i]] = dict(max=round(float(e.max()), 3), final=round(float(e[-1]), 3),
                            frac_over_0_5=round(float((e > 0.5).mean()), 3))
    out["per_robot"] = lost
    # the ball
    vis = S[:, c["ball_vis"]] > 0
    det = S[:, c["ball_det"]] > 0
    br = S[:, c["ball_rng"]]
    out["ball"] = dict(frames=int(len(S)), in_fov=float((S[:, c["in_fov"]] > 0).mean()), visible=float(vis.mean()),
                       detected_when_visible=float(det[vis].mean()) if vis.any() else None,
                       false_detect_when_not_visible=float(det[~vis].mean()) if (~vis).any() else None)
    er = S[:, c["ball_err_r"]]
    ew = S[:, c["ball_err_w"]]
    ok = vis & det & (er >= 0)
    by_rng = {}
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 10)):
        msk = ok & (br >= lo) & (br < hi)
        if msk.any():
            by_rng[f"{lo}-{hi}m"] = dict(n=int(msk.sum()), err_robotframe_med=round(pct(er[msk], 50), 3),
                                         err_robotframe_p90=round(pct(er[msk], 90), 3),
                                         err_world_med=round(pct(ew[msk], 50), 3), err_world_p90=round(pct(ew[msk], 90), 3),
                                         detected_when_visible=round(float(det[vis & (br >= lo) & (br < hi)].mean()), 3))
    out["ball"]["by_range"] = by_rng
    if len(B):
        out["ball"]["seen_by_team_any"] = dict(blue=float(B[:, 1].mean()), yellow=float(B[:, 2].mean()))
        gaps = {}
        dt = float(t[int(B[1, 0])] - t[int(B[0, 0])]) if len(B) > 1 else 0.033
        for j, team in ((1, "blue"), (2, "yellow")):
            g, cur = [], 0
            for v in B[:, j]:
                if v:
                    if cur:
                        g.append(cur * dt)
                    cur = 0
                else:
                    cur += 1
            if cur:
                g.append(cur * dt)
            g = np.array(g) if g else np.zeros(1)
            gaps[team] = dict(n=int(len(g)), median_s=round(float(np.median(g)), 2), p90_s=round(pct(g, 90), 2),
                              max_s=round(float(g.max()), 2))
        out["ball"]["unseen_gaps"] = gaps
    # robots
    if len(D):
        k_, i_, jj, frac, rr, hit, e_r, e_w, team_seen, team_true, assoc = D.T
        out["robots"] = dict(visible_pairs=int(len(D)), detected=float(hit.mean()),
                             detected_full=float(hit[frac > 0.99].mean()) if (frac > 0.99).any() else None,
                             detected_partial=float(hit[frac < 0.99].mean()) if (frac < 0.99).any() else None,
                             partial_share=float((frac < 0.99).mean()))
        by = {}
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 10)):
            msk = (rr >= lo) & (rr < hi)
            if msk.any():
                h = msk & (hit > 0)
                by[f"{lo}-{hi}m"] = dict(n=int(msk.sum()), detected=round(float(hit[msk].mean()), 3),
                                         err_robotframe_med=round(pct(e_r[h], 50), 3) if h.any() else None,
                                         err_world_med=round(pct(e_w[h], 50), 3) if h.any() else None,
                                         err_world_p90=round(pct(e_w[h], 90), 3) if h.any() else None)
        out["robots"]["by_range"] = by
        h = hit > 0
        tid = h & (team_seen >= 0)
        out["robots"]["team_colour_read"] = float(tid.sum() / max(h.sum(), 1))
        out["robots"]["team_colour_correct"] = float((team_seen[tid] == team_true[tid]).mean()) if tid.any() else None
        out["robots"]["team_by_association_correct"] = float((assoc[h] == team_true[h]).mean()) if h.any() else None
        mate = h & (team_true == np.array([0 if ids[int(x)].startswith("blue") else 1 for x in i_]))
        opp = h & ~mate
        out["robots"]["assoc_correct_mates"] = float((assoc[mate] == team_true[mate]).mean()) if mate.any() else None
        out["robots"]["assoc_correct_opponents"] = float((assoc[opp] == team_true[opp]).mean()) if opp.any() else None
        # per frame and team: how many of the 5 opponents does at least one team-mate see
        seen = {}
        for team_true_v, team in ((1, "blue"), (0, "yellow")):   # the opponents of `team`
            msk = h & (team_true == team_true_v) & np.isin(i_, [n for n, s in enumerate(ids) if s.startswith(team)])
            per_k = {}
            for kk_, j_ in zip(k_[msk], jj[msk]):
                per_k.setdefault(int(kk_), set()).add(int(j_))
            nk = len(set(k_.astype(int)))
            vals = [len(per_k.get(kk_, ())) for kk_ in sorted(set(k_.astype(int)))]
            seen[team] = dict(opponents_seen_mean=round(float(np.mean(vals)), 2) if vals else 0.0, frames=nk)
        out["robots"]["opponents_seen_by_team"] = seen
    return out


if __name__ == "__main__":
    main()
