"""Checks of the air-hockey two-mode pilot (run in the sport's image: python3 -m pytest pilots/airhockey_modes/tests)."""
import sys
from pathlib import Path

import mujoco
import numpy as np

PG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PG))
sys.path.insert(0, str(PG / "programs"))
import world as W  # noqa: E402
import ahkin  # noqa: E402


def test_ahref_is_the_sport_reference():
    a = (PG / "programs" / "ahref.py").read_text().split("\n", 3)[3]
    b = (PG.parents[1] / "sports" / "airhockey" / "builtin_players.py").read_text()
    assert a == b[3:]          # the copy moved the opening quotes to its header


def test_toolkit_hashes():
    import toolkit_check
    sys.argv = ["toolkit_check.py"]
    assert toolkit_check.main() == 0


def test_toolkit_model_matches_the_world():
    h = W.ModesHockey(seed=1, duel=True)
    kin = ahkin.Kin(str(PG / "toolkit" / "iiwa_mallet.xml"))
    q = h.qtarget["A"]
    assert np.allclose(kin.fk(q)[0], h.mallet_of("A", q), atol=1e-9)
    assert np.allclose(ahkin.VEL[:7], W.H.ARM.vel) and np.allclose(ahkin.ACC[:7], W.H.ARM.acc)


def test_low_plans_are_checked():
    h = W.ModesHockey(seed=1, duel=True)
    seat = W.Seat(h, "low")
    q = np.array(h.qtarget["A"])
    assert seat.command({"type": "plan", "q": [list(q + np.array([0.05] + [0] * 8))]}).startswith("plan refused: row 0: joint 1 velocity")
    assert seat.command({"type": "plan", "q": [list(q)] * 3})["ok"]
    assert seat.command({"type": "move", "x": -0.5, "y": 0}).startswith("unknown command")


def test_platform_strike_hits_a_resting_puck_on_time():
    h = W.ModesHockey(seed=1, duel=True)
    h.park("B", W.PARK_B)
    h.data.qpos[h.px], h.data.qpos[h.py] = -0.5, 0.1
    h.data.qvel[h.pvx:h.pvx + 3] = 0
    mujoco.mj_forward(h.model, h.data)
    seat = W.Seat(h, "high")
    d = np.array([0.974 + 0.5, -0.1]); d /= np.linalg.norm(d)
    r = seat.command({"type": "strike", "t": 0.8, "x": -0.5, "y": 0.1, "dx": d[0], "dy": d[1], "speed": 1.0})
    assert isinstance(r, dict) and r["late_s"] == 0
    k = None
    for _ in range(300):
        if "A" in h.step():
            k = h.k
            break
    assert k is not None and abs(k * W.DT - 0.8) <= 0.02


if __name__ == "__main__":                      # without pytest (the sport's image has none)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
