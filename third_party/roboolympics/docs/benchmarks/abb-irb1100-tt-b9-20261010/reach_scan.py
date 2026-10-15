"""Geometry only: for every library serve, can the ABB IRB 1100 + gantry put the blade on ANY of the stage-A strokes'
contact poses (position + face normal) without touching the table, net or floor, with the gantry anywhere in its
travel (no time limit)? An upper bound of the oracle's 'reach' and 'pose' stages, cheap enough to compare the
what-ifs (abbrig.variant: ABB_PADDLE, ABB_DROP).

    run.sh reach_scan.py OUT.json [part i of n]       env N_OPTS (default 24 options per serve, evenly spread)"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("RECV_DIR", "/recv") + "/prototypes/tabletennis_receive")
import oracle as O  # noqa: E402
import abbrig  # noqa: E402


def main():
    out = sys.argv[1]
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    abbrig.setup_readies()
    body = abbrig.AbbGantry()
    ST = O.load_strokes()
    ids = sorted(ST)[part::nparts]
    k = int(os.environ.get("N_OPTS", "24"))
    rows, t0 = [], time.time()
    for sid in ids:
        st = ST[sid]
        opts = st["opts"]
        if len(opts) > k:
            opts = [opts[int(i)] for i in np.linspace(0, len(opts) - 1, k)]
        why = {}
        ok = False
        for o in opts:
            sols, w = O.solutions(body, o, k_max=1)
            if sols:
                ok = True
                break
            why[w] = why.get(w, 0) + 1
        P = np.array([o["p"] for o in st["opts"]]) if st["opts"] else np.zeros((0, 3))
        rows.append(dict(id=sid, cls=st["cls"], reachable=ok, why=why, n_opts=len(st["opts"]),
                         y_max=float(P[:, 1].max()) if len(P) else None))
        print(sid, ok, why, round(time.time() - t0), flush=True)
        Path(out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
