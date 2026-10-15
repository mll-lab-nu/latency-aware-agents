"""Search a ready pose: the arm configuration from which typical contacts (forehand stance, 0.9-2.2 m high, faces
towards the net, up and down) are quickest to reach under the joint speed and acceleration limits (bm.move_time)."""
import numpy as np, itertools, time
import knobs as KN
BM = KN.BM
ph = BM.Phys(); ph.reset(np.r_[BM.HOME, BM.Q0_ARM])
home = BM.HOME
contacts = []
for z in (0.6, 0.85, 1.1, 1.35, 1.6, 1.9):
    st_off = 0.62 * max(0.30, 0.95 - 0.35 * abs(z - BM.SHOULDER_Z))
    p = np.r_[home + st_off, z]
    for nrm in ((0, 1, 0.5), (0, 1, 0.0), (0.4, 1, 0.3), (-0.4, 1, 0.3), (0, 1, -0.3)):
        contacts.append((p, np.array(nrm, float) / np.linalg.norm(nrm)))
def score(q0):
    ts = []
    for p, n in contacts:
        q, r, sg, alts = BM.ik_multi(ph, p, n, home, [q0], q0)
        ts.append(BM.move_time(q - q0) if r < BM.IK_OK else 2.0)
    return float(np.mean(ts)), float(np.max(ts))
seeds = [np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.9, 0.785]), np.array([0.0, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785]),
         np.array([0.5, 0.0, 0.0, -1.9, 0.0, 1.9, 0.0]), np.array([0.0, 0.0, 0.0, -1.5, 1.5, 1.5, 0.0]),
         np.array([0.3, 0.3, -0.3, -1.7, 0.0, 2.2, 0.785]), BM.Q0_ARM.copy(), np.array([1.4279, 1.0341, -1.7203, -1.2615, -0.2316, 1.5099, -0.0213])]
res = []
t0 = time.time()
print("current Q0", BM.Q0_ARM, score(BM.Q0_ARM), flush=True)
for h in (1.0, 1.15, 1.3, 1.45):
    for off in ((0.3, 0.6), (0.45, 0.45), (0.2, 0.7)):
        for nrm in ((0, 1, 0.25), (0, 1, 0.0)):
            for sd in seeds:
                q, err = BM.ik(ph, np.r_[home + np.array(off), h], np.array(nrm, float), home, sd, iters=400)
                if err > 1e-3:
                    continue
                m, mx = score(q)
                res.append((m, mx, h, off, nrm, np.round(q, 4).tolist()))
    print(h, len(res), round(time.time() - t0), flush=True)
res.sort(key=lambda r: r[0])
for r in res[:10]:
    print(r)
