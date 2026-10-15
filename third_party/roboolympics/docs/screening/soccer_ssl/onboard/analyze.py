"""Post-analysis of the onboard runs (run_onboard.py): recovery after collisions and sharp turns, the localisation
error over time (figure), and one table of all runs.

    python3 analyze.py            -> results/analysis.json, results/loc_err_time.png, prints the tables"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def episodes(flag, t, min_gap=1.0):
    """start indices of runs of True in flag, merged when closer than min_gap seconds."""
    out, last_end = [], -1e9
    on = False
    for k, f in enumerate(flag):
        if f and not on:
            if t[k] - last_end > min_gap:
                out.append(k)
            on = True
        elif not f and on:
            on = False
            last_end = t[k]
    return out


def recovery(tag):
    s = np.load(RES / f"run_{tag}_series.npz")
    meta = json.loads((RES / f"run_{tag}.json").read_text())
    rec = np.load(HERE.parents[3] / meta["rec"] if not Path(meta["rec"]).is_absolute() else meta["rec"])
    rows, cols = s["rows"], [str(c) for c in s["cols"]]
    c = {n: j for j, n in enumerate(cols)}
    t_all = rec["t"]
    out = {"bump": [], "spin": []}
    for i in sorted(set(rows[:, c["i"]].astype(int))):
        R = rows[rows[:, c["i"]] == i]
        k = R[:, c["k"]].astype(int)
        t = t_all[k]
        err, yerr = R[:, c["err"]], R[:, c["yerr"]]
        # bumps: robot-robot contact in the recording (at control-step resolution), mapped to frames
        touch = rec["touch_robot"][:, i]
        tele = rec["teleport"][:, i]
        tk = episodes(touch, t_all, 1.0)
        for k0 in tk:
            # push size: the largest gap between the setpoint and the true velocity within 1 s
            win = slice(k0, min(k0 + 100, len(t_all)))
            push = float(np.max(np.hypot(*(rec["vsp"][win, i, :2] - rec["vel"][win, i, :2]).T)))
            if tele[max(0, k0 - 50):k0 + 300].any():
                continue
            f0 = np.searchsorted(k, k0)
            if f0 <= 0 or f0 >= len(k) - 1:
                continue
            t0 = t_all[k0]
            before = err[f0 - 1]
            m3 = (t >= t0) & (t <= t0 + 3)
            a3 = np.searchsorted(t, t0 + 3)
            a10 = np.searchsorted(t, t0 + 10)
            out["bump"].append(dict(robot=i, t=round(float(t0), 2), push_mps=round(push, 2), before=round(float(before), 3),
                                    peak3=round(float(err[m3].max()), 3),
                                    at3=round(float(err[min(a3, len(err) - 1)]), 3),
                                    at10=round(float(err[min(a10, len(err) - 1)]), 3) if a10 < len(err) else None))
        # sharp turns: |yaw rate| > 8 rad/s
        w = np.abs(rec["vel"][:, i, 2])
        for k0 in episodes(w > 8.0, t_all, 1.0):
            f0 = np.searchsorted(k, k0)
            if f0 <= 0 or f0 >= len(k) - 1:
                continue
            k1 = k0
            while k1 < len(w) and w[k1] > 2.0:
                k1 += 1
            turned = float(np.sum(w[k0:k1]) * 0.01)
            t0, t1 = t_all[k0], t_all[min(k1, len(t_all) - 1)]
            m = (t >= t0) & (t <= t1 + 1.0)
            a3 = np.searchsorted(t, t1 + 3)
            out["spin"].append(dict(robot=i, t=round(float(t0), 2), turned_rad=round(turned, 1),
                                    yaw_before_deg=round(math.degrees(yerr[f0 - 1]), 2),
                                    yaw_peak_deg=round(math.degrees(yerr[m].max()), 2) if m.any() else None,
                                    pos_before=round(float(err[f0 - 1]), 3), pos_peak=round(float(err[m].max()), 3) if m.any() else None,
                                    pos_at3=round(float(err[min(a3, len(err) - 1)]), 3)))
    def summ(L, key_peak, key_after, big):
        if not L:
            return {"n": 0}
        pk = np.array([x[key_peak] for x in L if x[key_peak] is not None])
        af = np.array([x[key_after] for x in L if x[key_after] is not None])
        return {"n": len(L), "peak_median": round(float(np.median(pk)), 3), "peak_p90": round(float(np.percentile(pk, 90)), 3),
                "peak_max": round(float(pk.max()), 3), "after_median": round(float(np.median(af)), 3),
                "after_p90": round(float(np.percentile(af, 90)), 3), f"after_over_{big}": int((af > big).sum())}
    hard = [x for x in out["bump"] if x["push_mps"] > 0.5]
    return dict(bump_all=summ(out["bump"], "peak3", "at3", 0.3), bump_hard=summ(hard, "peak3", "at3", 0.3),
                spin=summ(out["spin"], "yaw_peak_deg", "pos_at3", 0.3), spin_events=out["spin"][:40],
                hard_bumps=sorted(hard, key=lambda x: -x["peak3"])[:15])


def figure(tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110)
    for tag, lab in tags:
        p = RES / f"run_{tag}_series.npz"
        if not p.exists():
            continue
        s = np.load(p)
        rows, cols = s["rows"], [str(c) for c in s["cols"]]
        c = {n: j for j, n in enumerate(cols)}
        t = s["t"][rows[:, c["k"]].astype(int)]
        tb = np.round(t).astype(int)
        for key, ls, l2 in (("err", "-", "particle filter"), ("dr_err", "--", "odometry only")):
            med = [np.median(rows[tb == b, c[key]]) for b in range(tb.max() + 1)]
            p90 = [np.percentile(rows[tb == b, c[key]], 90) for b in range(tb.max() + 1)]
            ln, = ax.plot(range(len(med)), med, ls, label=f"{lab}, {l2}: median")
            if key == "err":
                ax.fill_between(range(len(med)), med, p90, color=ln.get_color(), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("game time (s)")
    ax.set_ylabel("position error (m), 10 robots")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "loc_err_time.png")


def window_stats(tag, k_max):
    """robot and ball detection over the first k_max control steps of a run (to compare runs of different length)."""
    s = np.load(RES / f"run_{tag}_series.npz")
    rows, cols, D = s["rows"], [str(c) for c in s["cols"]], s["det"]
    c = {n: j for j, n in enumerate(cols)}
    rows = rows[rows[:, c["k"]] < k_max]
    D = D[D[:, 0] < k_max]
    out = dict(loc_med=round(float(np.median(rows[:, c["err"]])), 4), loc_p90=round(float(np.percentile(rows[:, c["err"]], 90)), 4),
               fp_per_picture=round(float(rows[:, c["fp"]].mean()), 3), robots_detected=round(float(D[:, 5].mean()), 3))
    for lo, hi in ((0, 2), (2, 4), (4, 10)):
        m = (D[:, 4] >= lo) & (D[:, 4] < hi)
        h = m & (D[:, 5] > 0)
        out[f"det_{lo}-{hi}m"] = round(float(D[m, 5].mean()), 3)
        out[f"err_world_med_{lo}-{hi}m"] = round(float(np.median(D[h, 7])), 3) if h.any() else None
    return out


def main():
    runs = sorted(p.stem[4:] for p in RES.glob("run_*.json"))
    table = {}
    for tag in runs:
        d = json.loads((RES / f"run_{tag}.json").read_text())
        row = dict(seconds=round(d["seconds"]), size=d["size"], hfov=d["hfov"], band=d["band"], drift=d["drift"],
                   lines=d["lines"], boundary=d["boundary"],
                   loc_med=round(d["loc_err_m"]["median"], 3), loc_p90=round(d["loc_err_m"]["p90"], 3),
                   loc_p99=round(d["loc_err_m"]["p99"], 3), loc_max=round(d["loc_err_m"]["max"], 3),
                   over_0_3=round(d["loc_err_m"]["frac_over_0_3"], 4), over_1=round(d["loc_err_m"]["frac_over_1"], 4),
                   yaw_med=round(d["yaw_err_deg"]["median"], 2), yaw_p90=round(d["yaw_err_deg"]["p90"], 2),
                   dr_med=round(d["deadreckon_err_m"]["median"], 3), dr_p90=round(d["deadreckon_err_m"]["p90"], 3),
                   ball_visible=round(d["ball"]["visible"], 3), ball_det=d["ball"]["detected_when_visible"],
                   ball_team=d["ball"].get("seen_by_team_any"), ball_gaps=d["ball"].get("unseen_gaps"),
                   ball_by_range=d["ball"]["by_range"], robots=d.get("robots"), timing=d["timing_ms_per_robot_frame"],
                   by_time=d["loc_err_by_time"], per_robot=d["per_robot"])
        try:
            row["recovery"] = recovery(tag)
        except Exception as e:  # noqa: BLE001
            row["recovery"] = f"n/a: {e}"
        table[tag] = row
    for tag in ("L2L3_s1", "L2L3_s1_det2", "L2L3_s1_noband", "L2L3_s1_jitter05", "L2L3_s1_jitter1", "L2L3_s1_noise8"):
        if tag in table:
            table[tag]["window60"] = window_stats(tag, 6000)
            print("window60", tag, table[tag]["window60"])
    (RES / "analysis.json").write_text(json.dumps(table, indent=1, default=float))
    figure([("L2L3_s1", "L2 v L3 seed 1"), ("L3L2_s2", "L3 v L2 seed 2")])
    for tag, r in table.items():
        print(tag, {k: r[k] for k in ("seconds", "size", "hfov", "band", "drift", "lines", "boundary", "loc_med", "loc_p90",
                                      "loc_p99", "loc_max", "over_0_3", "over_1", "yaw_med", "dr_med", "dr_p90",
                                      "ball_visible", "ball_det", "timing")})
        rcv = r["recovery"]
        if isinstance(rcv, dict):
            print("   bumps", rcv["bump_all"], "hard", rcv["bump_hard"], "spins", rcv["spin"])


if __name__ == "__main__":
    main()
