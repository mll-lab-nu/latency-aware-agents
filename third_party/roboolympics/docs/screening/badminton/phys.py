"""Physical consistency on the live models: contacts not disabled, the self-collision pairs present and active,
the real-limit and self-collision exploit tests (E12, E13)."""
import os, sys, json
sys.path[:0] = ["/repo", "/repo/sports/badminton", "/repo/sports/badminton/sim", "/repo/sports/badminton/tests"]
os.chdir("/repo")
import mujoco
import test_badminton as T
import bm as BM, bduel as BD
out = {}
for name, path in (("match", "sports/badminton/sim/scene/bm_scene.xml"), ("duel", "sports/badminton/sim/scene/bm_duel.xml")):
    m = mujoco.MjModel.from_xml_path(path)
    out[name] = {"contact_disabled": bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT),
                 "npair": int(m.npair), "ngeom": int(m.ngeom),
                 "geoms_with_contype": int(sum(1 for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]))}
g = BM.Match(1, lv_override={"seed": 1})
out["live_match_contact_disabled"] = bool(g.s.m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
out["live_match_npair"] = int(g.s.m.npair)
print(json.dumps(out), flush=True)
for t in ("test_exploit_real_robot_limits", "test_exploit_self_collision", "test_skill_contract_is_the_simulator",
          "test_opponent_levels_are_placement_sensitive"):
    getattr(T, t)(); print("PASS", t, flush=True)
