"""Probe: from the ready pose at rest (base at home), can the stroke controller deliver a racket state at contact
T s ahead? For a grid of contact points (relative to the forehand stance of home) and T, the rehearsed position and
face errors and the delivered head speed. Prints one row per case and a summary."""
import os, sys, json, time
import numpy as np
import knobs as KN
BM = KN.BM
if os.environ.get("NOFF"):
    BM.COUPLING_FF = False
if os.environ.get("STANCE"):
    BM.STANCE_DIR = np.array([float(x) for x in os.environ["STANCE"].split(",")])
if os.environ.get("PLAN_A"):
    BM.BASE_PLAN_AMAX = float(os.environ["PLAN_A"]); BM.BASE_TJ = BM.BASE_PLAN_AMAX / BM.BASE_JMAX
Ts = [float(x) for x in os.environ.get("PROBE_T", "0.4,0.5,0.6,0.8,1.0").split(",")]
rows = []
home = BM.HOME.copy()
pts = []
for dx in (-0.9, 0.0, 0.6):
    for dy in (-0.8, 0.0, 0.8):
        for z in (0.9, 1.4, 2.0):
            pts.append((dx, dy, z))
vel_dir = np.array([float(x) for x in os.environ.get("VDIR", "0,0.94,0.34").split(",")]); vel_dir /= np.linalg.norm(vel_dir)
speed = float(os.environ.get("HEAD", "2.5"))
t0 = time.time()
for T in Ts:
    ok = 0; n = 0
    for dx, dy, z in pts:
        m = BM.Match(1)
        m._go_home()
        base_st = BM.stance(np.r_[home, z])        # where the base stands for a contact at...
        p = np.r_[home + BM.stance_offset(z) + np.array([dx, dy]), z]
        m.leg = {"launch": m.k}
        m._ik_seed = None
        plan, info = m.plan_racket(T, p, vel_dir, vel_dir * speed)
        good = info["blade_error_m"] <= 0.03 and info["normal_error_deg"] <= 5
        ok += good; n += 1
        rows.append(dict(T=T, dx=dx, dy=dy, z=z, err=info["blade_error_m"], nerr=info["normal_error_deg"],
                         head=float(np.linalg.norm(info["racket_vel_delivered"])),
                         v=info["racket_vel_delivered"], scale=info["speed_scale"], base=info["base"], good=good))
    hs = [r["head"] for r in rows if r["T"] == T and r["good"]]
    print(f"T={T}: delivered {ok}/{n}, head speed of the delivered: median {np.median(hs) if hs else 0:.2f} max {max(hs) if hs else 0:.2f}", round(time.time() - t0), flush=True)
out = os.path.join(os.environ.get("BM_OUT", "results"), f"probe_{os.environ.get('TAG', 'x')}.json")
json.dump(rows, open(out, "w"))
for r in rows:
    if not r["good"]:
        print(r["T"], r["dx"], r["dy"], r["z"], r["err"], r["nerr"], r["base"])
