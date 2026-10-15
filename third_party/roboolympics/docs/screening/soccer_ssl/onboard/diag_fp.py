"""Diagnose the robot detector's unmatched detections (false positives) on a few recorded frames."""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3])); sys.path.insert(0, str(HERE))
import perceive as PC
from sports.soccer_ssl.sim import onboard as OB
import mujoco as mj
from sports.soccer_ssl.sim.world import World
R = np.load(HERE / "results" / "rec_L2L3_s1.npz")
with OB.world_builder() as cams:
    w = World(seed=1)
m, d = w.model, w.data
ids = [f"{t}{n}" for t, n in w.ids]
cam = PC.Camera()
from collections import Counter
cnt = Counter(); n = 0
for k in range(300, 12000, 600):
    d.qpos[:] = R["qpos"][k]; mj.mj_kinematics(m, d); mj.mj_camlight(m, d)
    for i in (1, 3, 6, 8):
        tp = R["pose"][k, i]
        img = OB.render(m, d, [cams[f"{ids[i]}_robot"]])[0]
        C = PC.classes(img)
        for r in PC.robots(C, cam, clean="--v1" not in sys.argv):
            c, s = math.cos(tp[2]), math.sin(tp[2])
            wx, wy = tp[0] + c * r["x"] - s * r["y"], tp[1] + s * r["x"] + c * r["y"]
            dmin = min(math.hypot(wx - R["pose"][k, j, 0], wy - R["pose"][k, j, 1]) for j in range(10) if j != i)
            if abs(wx) > 4.4 and abs(wy) < 0.75:
                kind = "goalframe"
            elif dmin < 0.5:
                kind = "near_robot"
            elif abs(wx) > 4.6 or abs(wy) > 3.1:
                kind = "at_wall"
            else:
                kind = "other"
            cnt[kind] += 1
            if kind in ("other", "at_wall") and n < 25:
                n += 1
                print(k, ids[i], kind, round(wx, 2), round(wy, 2), "range", round(r["range"], 2), "w", r["width"], "exp", round(r["exp_width"], 1), "dmin", round(dmin, 2), "u", r["u0"], r["u1"], "vb", r["vb"])
print(cnt)
