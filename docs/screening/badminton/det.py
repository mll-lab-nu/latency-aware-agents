"""Camera detection error of the reference's pipeline vs the true shuttle position (one perception-free match)."""
import base64, json, os, sys
sys.path.insert(0, "/repo/docs/benchmarks/badminton-minicourt-20261008")
import numpy as np
import study as S
PL, BM, BP, BD = S.PL, S.BM, S.BP, S.BD
from core.sport import MatchConfig, FINISHED, UNIT_OVER
cfg = MatchConfig(sport="badminton", format="match-wallclock", seats=("A",), seed=101, level=1, slowdown=20)
m = PL.SPORT.build(cfg); layer = PL.SPORT.action_layer("high")
ref = S.PerfectRef(S.seat_spec(cfg, m, "A"), "A", 101)
cams = ref.cams
errs, miss, n = [], 0, 0
while m.phase().state != FINISHED and m.k < 4000:
    if m.phase().state == UNIT_OVER:
        m.resume()
    if m.k % 5 == 0:
        st = {"seat": "A", "phase": "running", "world": "running", "step": m.k, "t_sim": m.k * 0.01, "done": False, **m.observe("A")}
        body = ref.act({"state": st, "observation": ref.perceive(m, "A")})
        if body is not None:
            try: S.act_apply(layer, m, "A", body)
            except Exception: pass
    if m.k % 20 == 0 and m.g.s.sh_live:
        job, meta = m.camera_job("A")
        png = PL.SPORT.worker_call(None, "render", job)["png"]
        img = BP.decode(png)
        det = dict((round(t, 3), X) for t, X in BP.detect_views(img, cams, meta["view_sim_times"]))
        for sn, t in zip(job["snaps"], meta["view_sim_times"]):
            X = BD.to_side("A", sn["sh"]) if hasattr(BD, "to_side") else np.array(sn["sh"])
            X = np.asarray(sn["sh"], float)
            n += 1
            if round(float(t), 3) in det:
                errs.append(float(np.linalg.norm(det[round(float(t), 3)] - X)))
            else:
                miss += 1
    m.step()
e = np.array(errs)
print(json.dumps({"frames": n, "detected": len(e), "missed": miss, "median_cm": round(100 * float(np.median(e)), 2),
                  "p90_cm": round(100 * float(np.percentile(e, 90)), 2), "max_cm": round(100 * float(e.max()), 2)}))
