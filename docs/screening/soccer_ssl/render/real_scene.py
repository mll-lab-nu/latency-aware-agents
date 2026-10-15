"""The camera tests on the real robot: A's robots (bodies/ssl_robot) on B's field (sports/soccer_ssl/sim/
field.build_scene) with SSL_CAMERA_FIELD=b, or on the stand-in field of stub_scene.py (default; used before B's field
was ready). Same interface as stub_scene.py (build, random_state, BALL_R, TOP_Z), so bench.py and detect.py
take --scene real. This branch carries lines A and B since the merge of ssl-world 274cd650."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import stub_scene  # noqa: E402

FIELD = os.environ.get("SSL_CAMERA_FIELD", "stub")
_decl = stub_scene.ssl_body().load_decl()["geometry"]
ROBOT_H = float(_decl["height"])
TOP_Z = ROBOT_H + 0.001                      # the pattern's paper on the top plate
BALL_R = stub_scene.BALL_R
BODY = "bodies.ssl_robot"


def build(layout="ssl4", size=None, lights=True, offsamples=None, shadows=False):
    if FIELD != "b":
        return stub_scene.build(layout, size=size, lights=lights, offsamples=offsamples, shadows=shadows, robot="ssl")
    import camera
    from sports.soccer_ssl.sim import field as F
    spec, handles = F.build_scene(stub_scene.N_PER_TEAM, seed=0, body=stub_scene.ssl_body())
    # SSL_CAMERA_RAW_LIGHT=1: B's lights and carpet as they are (the report's "before"); default: camera.py's venue
    # lighting (directional, matte carpet)
    names = camera.add_cameras(spec, handles, layout=layout, size=size, lights=lights,
                               venue_lighting=not os.environ.get("SSL_CAMERA_RAW_LIGHT"))
    spec.visual.quality.offsamples = 4 if offsamples is None else int(offsamples)
    if shadows:
        for lt in spec.lights:
            lt.castshadow = True
    return spec, names, sorted(handles["robots"])


def random_state(model, rng, ball_at_robot=0.3):
    return stub_scene.random_state(model, rng, ball_at_robot)
