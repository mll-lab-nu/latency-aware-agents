"""Stage B of the upper-bound receiver for the ABB IRB 1100 + gantry: the serve-receivability study's oracle.py
(branch tt-receive-study 329ff155, unchanged) with this body (abbrig.AbbGantry) and its per-axis gantry reach box.
Everything else is the Franka study's method: the 284 legal serves of the library, the stage-A strokes, reaction 0,
act lead LEAD (default 0.03 s), the limits-aware stroke planner (bodies/trajopt.stroke), the path check, 8 scattered
executions in the contact lab (>= 6 land on the server's half).

    run.sh oracle_abb.py OUT.json [part i of n]        env: REACT, LEAD, READY, SEL, N_EVAL"""
import os
import sys

sys.path.insert(0, os.environ.get("RECV_DIR", "/recv") + "/prototypes/tabletennis_receive")
import oracle as O  # noqa: E402
import bodyrig as BR  # noqa: E402
import abbrig  # noqa: E402

abbrig.setup_readies()
BR.make = lambda key: abbrig.AbbGantry()
O.reach_box = abbrig.reach_box

if __name__ == "__main__":
    sys.argv[1:1] = ["abb_gantry"]
    O.main()
