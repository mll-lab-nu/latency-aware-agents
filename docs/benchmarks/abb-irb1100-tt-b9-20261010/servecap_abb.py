"""The serve-receivability study's serve capability estimate (servecap.py of branch tt-receive-study, unchanged: the
largest blade speed along the serve direction (0, 0.85, 0.5) at contacts behind the end line, each joint's speed
reachable from rest within 0.5 s and 1.2 rad under its limits) for the ABB IRB 1100 + gantry.

    run.sh servecap_abb.py OUT_PREFIX     (one part)"""
import os
import sys

sys.path.insert(0, os.environ.get("RECV_DIR", "/recv") + "/prototypes/tabletennis_receive")
import numpy as np  # noqa: E402
import bodyrig as BR  # noqa: E402
import servecap as SC  # noqa: E402
import abbrig  # noqa: E402

abbrig.setup_readies()
BR.make = lambda key: abbrig.AbbGantry()
_np_zeros = np.zeros

if __name__ == "__main__":
    out = sys.argv[1]
    # servecap.main puts a non-Franka body's base at zeros(nb): for the gantry that is its home (y = -2.0); the Franka's
    # serve stance is (0, -1.75): use the same
    sys.argv = [sys.argv[0], "franka_rail", out]
    SC.BR.make = lambda key: abbrig.AbbGantry()
    orig = abbrig.AbbGantry.fk

    def fk(self, b, q, _o=orig):
        if np.allclose(b, [0.0, -1.75]):
            b = np.array(abbrig.READIES["home"], float)
        return _o(self, b, q)
    abbrig.AbbGantry.fk = fk
    orig_ik = abbrig.AbbGantry.ik

    def ik(self, p, n, b0, q0, **kw):
        if np.allclose(b0, [0.0, -1.75]):
            b0 = np.array(abbrig.READIES["home"], float)
        return orig_ik(self, p, n, b0, q0, **kw)
    abbrig.AbbGantry.ik = ik
    SC.main()
