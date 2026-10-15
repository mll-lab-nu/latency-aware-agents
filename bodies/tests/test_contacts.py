"""bodies/contacts.py: robots touch each other and each other's tools, never themselves through the bits, and the
broad phase's body bits follow. Run: python bodies/tests/test_contacts.py (or pytest)."""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from bodies import contacts as CT  # noqa: E402
from bodies import load_body  # noqa: E402
from bodies.conformance import WORLD  # noqa: E402


def scene(gap, hand="epee"):
    B = load_body("unitree_g1")
    root = ET.fromstring(WORLD)
    for s, (x, yaw) in {"A": (-gap / 2, 0.0), "B": (gap / 2, math.pi)}.items():
        B.mount(root, f"{s}_", (x, 0.0, 0.0), yaw, hand=hand)
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    for s in "AB":
        r = B.rig(m, f"{s}_", hand=hand)
        r.place(d, r.rest)
    mj.mj_forward(m, d)
    return m, d


def body_pairs(m, d):
    out = set()
    for a, b in CT.contacts_between(m, d, "A_", "B_"):
        if "hand_" not in a and "hand_" not in b:
            out.add((a, b))
    return out


def own(m, d, p):
    return sorted((int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon]
                  if all((mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").startswith(p)
                         for g in (c.geom1, c.geom2)))


def test_bodies_touch_only_after_between():
    m, d = scene(0.15, hand=None)
    assert not body_pairs(m, d)                      # the library's default: robots pass through each other
    before = own(m, d, "A_")
    info = CT.between(m, ["A_", "B_"])
    mj.mj_forward(m, d)
    assert body_pairs(m, d)
    assert own(m, d, "A_") == before                 # its own contacts: unchanged (self-collision pairs)
    assert info["A_"]["body_bit"] != info["B_"]["body_bit"]


def test_body_bits_follow_the_geoms():
    m, d = scene(0.15, hand=None)
    CT.between(m, ["A_", "B_"])
    for b in range(m.nbody):
        sel = m.geom_bodyid == b
        if sel.any():
            assert m.body_contype[b] == int(np.bitwise_or.reduce(m.geom_contype[sel]))
            assert m.body_conaffinity[b] == int(np.bitwise_or.reduce(m.geom_conaffinity[sel]))


def test_a_blade_touches_the_other_body_not_its_own():
    m, d = scene(4.0)
    CT.between(m, ["A_", "B_"])
    blade = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "A_hand_blade")
    tip = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "A_hand_tip")
    torso_b = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("B_torso_link").id and m.geom_contype[g]]
    torso_a = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("A_torso_link").id and m.geom_contype[g]]
    fil = lambda g1, g2: bool(m.geom_contype[g1] & m.geom_conaffinity[g2]) or bool(m.geom_contype[g2] & m.geom_conaffinity[g1])
    assert all(fil(blade, g) and fil(tip, g) for g in torso_b)           # the bits let them collide
    assert not any(fil(blade, g) or fil(tip, g) for g in torso_a)        # its own body: only through its pairs
    kick = [g for g in range(m.ngeom) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "").endswith("_strike")]
    assert kick and all(m.geom_contype[g] == 8 for g in kick)           # sensing boxes untouched


def test_bits_stay_above_the_sports_bits():
    for k in range(CT.MAX_ROBOTS):
        a, b = CT.bits(k)
        assert a >= 1 << CT.BASE and b < 1 << 31


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
