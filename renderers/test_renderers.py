"""Unit tests of the renderer plugins (stdlib runner; MuJoCo needed):

    docker run --rm -v $PWD:/w -w /w -e MUJOCO_GL=osmesa ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 \
        python3 renderers/test_renderers.py
"""
import copy
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]   # renderers/mujoco must not shadow mujoco
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from renderers import CameraDef, RendererUnavailable, Scene, available, load  # noqa: E402
from renderers.budget import check_budget, frame_ms  # noqa: E402

DOT = """<mujoco>
  <visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/><global offwidth="640" offheight="480"/></visual>
  <asset><material name="glow" emission="1" rgba="1 1 1 1"/></asset>
  <worldbody>
    <body name="dot"><freejoint/><geom type="sphere" size="0.012" material="glow" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>"""

ARENA = """<mujoco>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1=".3 .32 .35" rgb2=".25 .27 .3" width="64" height="64"/>
    <material name="floor" texture="grid" texrepeat="4 4"/></asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 .1" material="floor"/>
    <geom name="target" type="cylinder" size=".3 .005" pos="1.5 0 .005" rgba=".9 .1 .1 1"/>
    <body name="ball" pos="0 0 .5"><freejoint/><geom name="ball" type="sphere" size=".05" rgba="1 .85 .1 1"/></body>
    <geom name="post" type="box" size=".05 .05 .4" pos="1 1 .4"/>
  </worldbody>
</mujoco>"""

CAMS = [CameraDef("oblique", (-1.2, -0.9, 0.8), (0.1, 0.05, 0.0), fovy_deg=50, width=320, height=240),
        CameraDef("side", (0.0, -2.0, 0.15), (0.0, 0.0, 0.1), fovy_deg=35, width=400, height=300),
        CameraDef("top", (0.02, -0.01, 2.0), (0.0, 0.0, 0.0), fovy_deg=60, width=256, height=256),
        CameraDef("top_y_up", (0.0, 0.0, 1.5), (0.0, 0.0, 0.0), up=(0, 1, 0), fovy_deg=45, width=300, height=200)]
POINTS = [(0.0, 0.0, 0.0), (0.12, -0.08, 0.05), (-0.1, 0.09, -0.04), (0.2, 0.15, 0.1)]


def dot_state(p):
    return {"qpos": [p[0], p[1], p[2], 1.0, 0.0, 0.0, 0.0]}


def test_projection_matches_the_rendered_image_within_one_pixel():
    R = load("mujoco")(Scene(xml=DOT))
    worst = 0.0
    for cam in CAMS:
        for p in POINTS:
            img = R.render(dot_state(p), [cam])[0]
            v, u = np.nonzero(img.max(axis=2) > 100)
            assert len(u), f"{cam.name}: dot {p} not visible"
            got = (u.mean() + 0.5, v.mean() + 0.5)
            want = cam.project(p)
            err = float(np.hypot(got[0] - want[0], got[1] - want[1]))
            worst = max(worst, err)
            assert err <= 1.0, f"{cam.name} {p}: drawn {got}, projected {want} ({err:.2f} px)"
    R.close()
    print(f"     worst projection error {worst:.3f} px over {len(CAMS)} cameras x {len(POINTS)} points")


def test_intrinsics_and_dict_round_trip():
    c = CAMS[0]
    assert CameraDef.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    k = c.intrinsics()
    assert abs(k["focal_px"] - 120 / np.tan(np.radians(25))) < 1e-9 and k["principal_point_px"] == [160, 120]
    r, u, f = c.basis()
    assert abs(r @ u) < 1e-12 and abs(r @ f) < 1e-12 and abs(u @ f) < 1e-12


def test_rendering_is_deterministic():
    s = Scene(xml=ARENA)
    st = {"qpos": [0.3, 0.2, 0.05, 1, 0, 0, 0]}
    a = load("mujoco")(s)
    b = load("mujoco")(s)
    x1, x2, y = a.render(st, CAMS[:2]), a.render(st, CAMS[:2]), b.render(st, CAMS[:2])
    assert all(np.array_equal(p, q) and np.array_equal(p, r) for p, q, r in zip(x1, x2, y))
    a.close()
    b.close()


def test_render_does_not_change_the_state():
    st = {"qpos": [0.3, 0.2, 0.05, 1, 0, 0, 0]}
    before = copy.deepcopy(st)
    R = load("mujoco")(Scene(xml=ARENA))
    img = R.render(st, [CAMS[0]])[0]
    img[:] = 0                                   # the caller owns its image
    again = R.render(st, [CAMS[0]])[0]
    assert st == before and again.any()
    R.close()


def test_more_and_larger_cameras_than_the_spare_set():
    R = load("mujoco")(Scene(xml=ARENA), n_cams=1, size=(64, 48))
    big = CameraDef("big", (-2, 0, 1), (0, 0, 0), width=800, height=600)
    imgs = R.render({"qpos": [0, 0, .05, 1, 0, 0, 0]}, [CAMS[0], CAMS[1], big])
    assert [i.shape for i in imgs] == [(240, 320, 3), (300, 400, 3), (600, 800, 3)]
    R.close()


def test_showcase_draws_the_same_scene():
    from renderers.mujoco_showcase import ShowcaseRenderer
    from renderers.scene_check import same_scene
    s = Scene(xml=ARENA)
    st = {"qpos": [0.3, 0.2, 0.05, 1, 0, 0, 0]}
    S = load("mujoco-showcase")(s, ssaa=2)
    assert S.scene_diffs == [] and S.label == "re-rendered: MuJoCo showcase"
    std = load("mujoco")(s).render(st, [CAMS[0]])[0]
    sho = S.render(st, [CAMS[0]])[0]
    assert sho.shape == std.shape and not np.array_equal(sho, std)       # another look of the same scene
    # the geometry is the same: the dot projects to the same pixel with the showcase renderer
    D = ShowcaseRenderer(Scene(xml=DOT))
    for cam in CAMS[:3]:
        img = D.render(dot_state(POINTS[1]), [cam])[0]
        v, u = np.nonzero(img.max(axis=2) > 100)
        want = cam.project(POINTS[1])
        assert np.hypot(u.mean() + .5 - want[0], v.mean() + .5 - want[1]) <= 1.0
    # the check catches a changed scene
    import mujoco as mj
    m0 = S.m
    sp = mj.MjSpec.from_string(ARENA.replace('size=".05 .05 .4"', 'size=".05 .05 .5"'))
    from renderers.mujoco import compile_spec
    assert "geom_size" in same_scene(m0, compile_spec(sp, S._n, S._size))
    sp2 = mj.MjSpec.from_string(ARENA.replace('rgba=".9 .1 .1 1"', 'rgba=".1 .9 .1 1"'))
    diff2 = same_scene(m0, compile_spec(sp2, S._n, S._size))
    assert any("colour" in b for b in diff2), diff2
    stamped = load("mujoco-showcase")(s, stamp=True).render(st, [CAMS[0]])[0]
    assert not np.array_equal(stamped, sho)


def tt_scene():
    p = os.path.join(ROOT, "tasks", "tabletennis-duel-wallclock-program-standard", "environment", "sim", "scene",
                     "tt_duel.xml")
    return Scene.from_file(p) if os.path.exists(p) else None


def test_showcase_preset_look_of_a_sport_scene():
    s = tt_scene()
    if s is None:
        print("     (skipped: no table-tennis task in this checkout)")
        return
    s.look = {"preset": "tabletennis"}
    S = load("mujoco-showcase")(s, size=(480, 360))
    assert S.scene_diffs == []
    q = load("mujoco")(Scene(xml=s.xml, assets=s.assets)).m.qpos0.tolist()
    img = S.render({"qpos": q}, [CameraDef("broadcast", (0, -3.6, 1.2), (0, 0, .2), fovy_deg=45, width=480,
                                           height=360)])[0]
    assert img.shape == (360, 480, 3) and img.std() > 5


def test_budget_measurement():
    s = Scene(xml=ARENA)
    st = {"qpos": [0.3, 0.2, 0.05, 1, 0, 0, 0]}
    ms = frame_ms(load("mujoco"), s, st, CAMS[:2], n=5)
    ok, ms2 = check_budget(load("mujoco"), s, st, CAMS[:2], budget_ms=10_000)
    bad, _ = check_budget(load("mujoco"), s, st, CAMS[:2], budget_ms=0.0)
    assert ms > 0 and ok and not bad
    print(f"     mujoco: {ms:.1f} ms per observation (2 cameras, 320x240 + 400x300)")
    t = tt_scene()
    if t is not None:
        R = load("mujoco")
        q = R(t).m.qpos0.tolist()
        cams = [CameraDef(f"c{i}", (0, -3.6 + i, 1.2), (0, 0, .2), width=480, height=360) for i in range(2)]
        print(f"     mujoco, table-tennis scene: {frame_ms(R, t, {'qpos': q}, cams):.1f} ms per observation "
              f"(2 cameras 480x360); mujoco-showcase: "
              f"{frame_ms(load('mujoco-showcase'), Scene(xml=t.xml, assets=t.assets), {'qpos': q}, cams):.1f} ms")


def test_path_tracers_report_unavailability_cleanly():
    av = available()
    assert av["cycles"]["needs_gpu"] and not av["cycles"]["realtime"] and av["mujoco"]["realtime"]
    for name in ("cycles", "isaac"):
        cls = load(name)
        assert cls.realtime is False
        os.environ["RO_RENDERER_OFFLINE"] = "1"                 # forced: same path as a host without them
        try:
            cls(Scene(xml=ARENA))
            raise AssertionError("should be unavailable")
        except RendererUnavailable as e:
            assert e.requirements and "GPU" in e.requirements
        finally:
            del os.environ["RO_RENDERER_OFFLINE"]
        try:
            cls(Scene(xml=ARENA))
            print(f"     {name}: available here")
        except RendererUnavailable as e:
            print(f"     {name}: unavailable here: {e.why}")


def test_path_tracer_export_is_written():
    import mujoco as mj
    from renderers.export import export
    from renderers.mujoco_showcase import ShowcaseRenderer
    m = ShowcaseRenderer(Scene(xml=ARENA), n_cams=1).m
    with tempfile.TemporaryDirectory() as d:
        st = {"qpos": [0.3, 0.2, 0.05, 1, 0, 0, 0]}
        info = export(m, [(st, CAMS[0]), (st, CAMS[0])], d)
        sc = json.load(open(os.path.join(d, "scene.json")))
        poses = np.load(os.path.join(d, "poses.npz"))
        assert info["frames"] == 2 and sc["fovy"] == CAMS[0].fovy_deg and len(sc["cameras"]) == 2
        assert poses["pos"].shape == (2, len(sc["geoms"]), 3)
        ball = [i for i, g in enumerate(sc["geoms"]) if g["name"] == "ball"][0]
        assert np.allclose(poses["pos"][0, ball], [0.3, 0.2, 0.05], atol=1e-6)
        lookat, dist, az, el = sc["cameras"][0]
        fwd = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)),
                        np.sin(np.radians(el))])
        assert np.allclose(np.array(lookat) - dist * fwd, CAMS[0].pos, atol=1e-9)
    assert mj is not None


# ------------------------------------------------------------------------------------- presentation scenes (rule 6)
ARM = """<mujoco><compiler angle="radian"/>
  <worldbody><light pos="0 0 3" dir="0 0 -1"/><geom name="floor" type="plane" size="2 2 .1" rgba=".3 .3 .3 1"/>
    <body name="base" pos="0 0 0.1"><geom name="base_c" type="box" size=".1 .1 .1" rgba=".8 .2 .2 1"/>
      <body name="upper" pos="0 0 0.2"><joint name="j1" axis="0 0 1"/>
        <geom name="upper_c" type="capsule" fromto="0 0 0 0 0 .4" size=".04" rgba=".8 .2 .2 1"/>
        <body name="tool" pos="0 0 0.4"><joint name="j2" axis="0 1 0"/>
          <geom name="tool_c" type="capsule" fromto="0 0 0 .3 0 0" size=".03" rgba=".2 .8 .2 1"/></body>
      </body></body></worldbody></mujoco>"""
ARM_MODEL = """<mujoco><compiler meshdir="parts"/>
  <asset><material name="blue" rgba=".1 .3 .9 1" specular=".5"/><mesh file="cube.obj" scale=".12 .12 .12"/></asset>
  <worldbody><body name="base"><geom type="mesh" mesh="cube" material="blue" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size=".1 .1 .1" group="3"/>
      <body name="upper" pos="0 0 UPPER_Z"><joint axis="0 0 1"/>
        <geom type="mesh" mesh="cube" pos="0 0 .2" material="blue" contype="0" conaffinity="0" group="2"/></body>
  </body></worldbody></mujoco>"""
CUBE = "".join(f"v {x} {y} {z}\n" for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)) + "".join(
    f"f {a} {b} {c}\n" for a, b, c in ((1, 3, 4), (1, 4, 2), (5, 6, 8), (5, 8, 7), (1, 2, 6), (1, 6, 5), (3, 7, 8),
                                       (3, 8, 4), (1, 5, 7), (1, 7, 3), (2, 4, 8), (2, 8, 6)))


def presentation_dir(tmp, upper_z="0.2"):
    d = Path(tmp) / "presentation"
    (d / "arm" / "parts").mkdir(parents=True, exist_ok=True)
    (d / "arm" / "arm.xml").write_text(ARM_MODEL.replace("UPPER_Z", upper_z))
    (d / "arm" / "parts" / "cube.obj").write_text(CUBE)
    (d / "presentation.toml").write_text('[presentation]\nversion = 1\nlabel = "test arm"\n[[presentation.robot]]\n'
                                         'model = "./arm/arm.xml"\nbodies = \'^(?P<n>base|upper)$\'\nbody = "{n}"\n')
    return d


def test_presentation_scene_keeps_the_kinematics():
    from renderers.label import frame_label
    from renderers.presentation import dress
    from renderers.scene_check import pose_diff, same_kinematics, visual_geoms
    match = Scene(xml=ARM)
    with tempfile.TemporaryDirectory() as tmp:
        pres = dress(match, presentation_dir(tmp))
    assert pres.is_presentation and not match.is_presentation and pres.match is match
    R0, R1 = load("mujoco")(match), load("mujoco")(pres)
    m0, m1 = R0.m, R1.m
    assert same_kinematics(m0, m1) == [] and len(visual_geoms(m1)) == 2
    rng = np.random.default_rng(1)
    assert pose_diff(m0, m1, [{"qpos": rng.uniform(-3, 3, m0.nq)} for _ in range(50)]) == 0.0
    group = {m1.geom(g).name: int(m1.geom_group[g]) for g in range(m1.ngeom)}
    assert group["base_c"] == group["upper_c"] == 3 and group["tool_c"] == 0       # only dressed bodies are hidden
    st = {"qpos": [0.4, 0.3]}
    a, b = R0.render(st, [CAMS[0]])[0], R1.render(st, [CAMS[0]])[0]
    assert a.shape == b.shape and not np.array_equal(a, b)
    S = load("mujoco-showcase")(pres)
    assert S.label.endswith("presentation meshes, spectators only") and S.render(st, [CAMS[0]])[0].std() > 1
    assert frame_label(R1, pres).endswith("spectators only") and frame_label(R0, match) is None
    for R in (R0, R1, S):
        R.close()


def test_presentation_scene_with_other_kinematics_is_refused():
    from renderers.presentation import dress
    from renderers.scene_check import same_kinematics
    match = Scene(xml=ARM)
    with tempfile.TemporaryDirectory() as tmp:
        pres = dress(match, presentation_dir(tmp))
        try:
            dress(match, presentation_dir(tmp, upper_z="0.3"))         # the model's frames are not the scene's
            raise AssertionError("a declaration naming bodies in other frames must fail")
        except ValueError as e:
            assert "frame" in str(e)
    moved = Scene(xml=pres.xml.replace('pos="0 0 0.2"', 'pos="0 0 0.25"'), assets=pres.assets, match=match)
    try:
        load("mujoco")(moved)
        raise AssertionError("a presentation scene with a moved body must be refused")
    except RuntimeError as e:
        assert "body_pos" in str(e)
    import mujoco as mj
    m0 = load("mujoco")(match).m
    hit = mj.MjModel.from_xml_string(pres.xml.replace('name="_vis:base:0"', 'name="_vis:base:0" contype="1"'),
                                     pres.assets)
    assert any("collides" in b for b in same_kinematics(mj.MjModel.from_xml_string(ARM), hit))
    assert m0 is not None


def test_players_images_never_use_a_presentation_scene():
    from core.observe import CameraWorker
    from renderers.presentation import dress
    with tempfile.TemporaryDirectory() as tmp:
        pres = dress(Scene(xml=ARM), presentation_dir(tmp))
    try:
        CameraWorker(pres, None)
        raise AssertionError("the players' camera worker must refuse a presentation scene")
    except ValueError as e:
        assert "spectators only" in str(e)
    assert CameraWorker(Scene(xml=ARM), None).name == "mujoco"


def test_players_presentation_scene_when_declared():
    """renderer rule 6 (core 0.2.9): `players` in presentation.toml = the players' images use the presentation scene
    (Scene.players), the same kinematics; the default renderer draws it untagged (the competition picture)."""
    from core.observe import CameraWorker
    from renderers.label import frame_label
    from renderers.presentation import players_model, players_scene
    from renderers.scene_check import same_kinematics
    match = Scene(xml=ARM)
    with tempfile.TemporaryDirectory() as tmp:
        d = presentation_dir(tmp)
        assert players_scene(tmp, match) is match and players_model(tmp, match) is None      # spectators only
        t = (d / "presentation.toml").read_text()
        (d / "presentation.toml").write_text(t.replace('version = 1\n', 'version = 2\nplayers = "full"\n'))
        sc = players_scene(tmp, match)
        m = players_model(tmp, match)
    assert sc.is_presentation and sc.players and sc.match is match
    R = CameraWorker(sc, None).renderer
    assert same_kinematics(load("mujoco")(match).m, R.m) == [] and frame_label(R, sc) is None
    assert m.nq == R.m.nq and m.ngeom == R.m.ngeom
    R.close()


def test_players_meshes_bundle_and_levels_of_detail():
    """The players' meshes bundled into a simulator (assets.bundle) verify by sha256 against the manifests, and a
    changed file is refused; every level of detail of lod.json is complete and pinned."""
    from renderers.presentation import PresentationUnavailable, assets
    levels = assets.lods().get("models", {})
    for name, lv in levels.items():
        for level in lv:
            assert assets.missing(name, level) == [], f"{name} {level}: lod files missing or changed"
    name, level = next((n, lev) for n, lv in levels.items() for lev in lv)
    with tempfile.TemporaryDirectory() as tmp:
        files = assets.bundle(name, level, Path(tmp))
        assert assets._find(name)[1]["license_file"] in files
        x = assets.model_xml(name, level, Path(tmp))
        assert x.is_file() and x.parent.parent == Path(tmp)
        victim = next(Path(tmp) / f for f in files if f.endswith((".obj", ".stl")))
        victim.write_bytes(victim.read_bytes() + b"\n")
        assets._verified.clear()
        try:
            assets.model_xml(name, level, Path(tmp))
            raise AssertionError("a changed bundled file must be refused")
        except PresentationUnavailable as e:
            assert "sha256" in str(e)


def test_sport_presentations_keep_the_kinematics():
    """Every sports/<s>/presentation: each configuration's match scene dressed, same kinematics, identical poses."""
    import subprocess
    from renderers.presentation import assets, declared, load_decl
    for d in sorted(Path(ROOT, "sports").glob("*/presentation")):
        if not declared(d):
            continue
        miss = [r["model"] for r in load_decl(d)["robot"] if not r["model"].startswith("./") and assets.missing(r["model"])]
        if miss:
            print(f"     {d.parent.name}: skipped (assets of {miss} not fetched: python3 -m renderers.presentation.assets fetch)")
            continue
        r = subprocess.run([sys.executable, "-c", SPORT_CHECK, d.parent.name], cwd=ROOT, capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, f"{d.parent.name}: {(r.stderr or r.stdout)[-800:]}"
        print(f"     {r.stdout.strip()}")


SPORT_CHECK = """
import sys
sys.path.insert(0, '.')
import numpy as np
from core import spec as spec_mod
from core.episode import load_plugin
from core.testkit import configs
from renderers import load
from renderers.presentation import dress, match_scenes, players_meshes
from renderers.scene_check import pose_diff, same_kinematics, visual_geoms
s = sys.argv[1]
players = players_meshes(f'sports/{s}/presentation')
plugin = f'sports/{s}/plugin.py'
sport, decl = load_plugin(plugin), spec_mod.load(plugin)
scenes = {}
for sc in ([sport.scene(c) for c in configs(decl, sport)] if hasattr(sport, 'scene') else match_scenes(f'sports/{s}')):
    scenes.setdefault(sc.xml, sc)
out = []
for sc in scenes.values():
    p = dress(sc, f'sports/{s}/presentation')
    m0, m1 = load('mujoco')(sc).m, load('mujoco')(p).m
    bad = same_kinematics(m0, m1)
    rng = np.random.default_rng(0)
    worst = pose_diff(m0, m1, [{'qpos': m0.qpos0 + 0.5 * rng.standard_normal(m0.nq)} for _ in range(30)])
    assert not bad and worst == 0.0, (bad, worst)
    out.append(f'{len(visual_geoms(m1))} visual geoms')
    if players:                                   # the players' scene (its level of detail) too
        m2 = load('mujoco')(dress(sc, f'sports/{s}/presentation', meshes=players, players=True)).m
        assert not same_kinematics(m0, m2) and pose_diff(m0, m2, [{'qpos': m0.qpos0}]) == 0.0
        out.append(f'players: {players} meshes')
print(f'{s}: {len(scenes)} match scene(s) dressed ({", ".join(out)}), same kinematics, identical poses')
"""


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("ok  ", n)
        except Exception:  # noqa: BLE001
            bad += 1
            print("FAIL", n)
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
