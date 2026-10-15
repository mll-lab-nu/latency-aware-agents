"""Weakened copies of the low level's reference act (sports/badminton/lowlevel/reference/act.py) for the "measurable"
check of the low level (docs/screening/badminton.md, low level; the walkthrough's step 4): each one drops one piece of
the engineering, so the API and competition tests should rank them between the reference and the boundary.

  python3 docs/screening/badminton/lowlevel_variants.py OUT_DIR      writes OUT_DIR/<variant>/act.py

  norehearse   no correction from rollouts on the servo model (the first swing is sent as planned)
  noservo      rollouts from the observed joints only, without the servo's state (a teleported robot)
  nobase       the base stays where its target is (no stance)
  nearestik    the IK solution with the least travel, not the one whose swing reaches the requested head speed
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "sports" / "badminton" / "lowlevel" / "reference" / "act.py"
SUBS = {
    "norehearse": [("        for it in range(PASSES):", "        for it in range(1):")],
    "noservo": [('S.reset(state["q"], state.get("qd"), state["q_target"], state.get("servo"))',
                 'S.reset(state["q"], state.get("qd"), state["q_target"], None)')],
    "nobase": [("    brows = _base(state, pos, n_c, n_tot, base)",
                "    brows = np.tile(np.asarray(state['q_target'][:2], float), (n_tot, 1))")],
    "nearestik": [("    cands.sort(key=lambda c: c[0])", "    cands.sort(key=lambda c: (c[0][0], c[0][3]))")],
}


def main(out):
    src = SRC.read_text()
    for name, subs in SUBS.items():
        s = src
        for a, b in subs:
            assert a in s, (name, a)
            s = s.replace(a, b)
        d = Path(out) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "act.py").write_text(s)
        print(d)


if __name__ == "__main__":
    main(sys.argv[1])
