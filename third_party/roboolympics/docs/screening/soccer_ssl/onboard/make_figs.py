"""Example pictures for the report: front-camera views from a recorded match with what the prototype detector reads
drawn on top (floor boundary per column: green = wall, red = tall object, cyan = goal foam; white line points;
the ball's box in orange; robots' boxes in their seen team colour, grey when no colour was read).

    docker run --rm --cpus 1 -v $REPO:/srv -w /srv ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 \\
        python docs/screening/soccer_ssl/onboard/make_figs.py  -> results/examples.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
import perceive as PC  # noqa: E402
from sports.soccer_ssl.sim import onboard as OB  # noqa: E402

SHOTS = [("rec_L2L3_s1.npz", 15.0, "blue2"), ("rec_L2L3_s1.npz", 45.0, "yellow3"), ("rec_L2L3_s1.npz", 75.0, "blue4"),
         ("rec_L3L2_s2.npz", 30.0, "blue0"), ("rec_L3L2_s2.npz", 60.0, "yellow1"), ("rec_drill_s5.npz", 21.0, "blue2")]


def main():
    import mujoco as mj
    from sports.soccer_ssl.sim.world import World
    tiles = []
    cam = PC.Camera()
    worlds = {}
    for rec_name, t, rid in SHOTS:
        R = np.load(HERE / "results" / rec_name)
        meta = json.loads(str(R["meta"]))
        if meta["seed"] not in worlds:
            with OB.world_builder() as cams:
                worlds[meta["seed"]] = (World(seed=meta["seed"], first_kickoff="blue" if meta["seed"] % 2 else "yellow"), cams)
        w, cams = worlds[meta["seed"]]
        k = int(round(t / 0.01)) - 1
        w.data.qpos[:] = R["qpos"][k]
        mj.mj_kinematics(w.model, w.data)
        mj.mj_camlight(w.model, w.data)
        img = OB.render(w.model, w.data, [cams[f"{rid}_robot"]])[0]
        C = PC.classes(img)
        us, vs, kinds = PC.floor_boundary(C, cam)
        lines = PC.line_points(C, cam)
        b = PC.ball(C, cam)
        rob = PC.robots(C, cam)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        col = {"wall": (0, 255, 0), "tall": (255, 0, 0), "goal": (0, 255, 255)}
        for u, v, kd in zip(us, vs, kinds):
            dr.ellipse([u - 2, v - 2, u + 2, v + 2], outline=col[kd])
        if len(lines):
            lu, lv = cam.project(lines[:, 0], lines[:, 1], 0.0)
            for u, v in zip(lu, lv):
                dr.point((u, v), fill=(255, 0, 255))
        if b is not None:
            r = max(b["w"], b["h"]) / 2 + 3
            dr.rectangle([b["u"] - r, b["v"] - r, b["u"] + r, b["v"] + r], outline=(255, 140, 0), width=2)
        for r in rob:
            c = {"blue": (40, 90, 255), "yellow": (255, 230, 0), None: (160, 160, 160)}[r["team"]]
            dr.rectangle([r["u0"], cam.hrow - 25, r["u1"], r["vb"]], outline=c, width=2)
        dr.line([(0, cam.hrow), (cam.W, cam.hrow)], fill=(255, 255, 255), width=1)
        dr.text((6, 6), f"{rec_name[4:-4]} t={t:.0f}s {rid}", fill=(255, 255, 255))
        tiles.append(im)
    W, H = tiles[0].size
    sheet = Image.new("RGB", (3 * W, 2 * H))
    for n, im in enumerate(tiles):
        sheet.paste(im, ((n % 3) * W, (n // 3) * H))
    sheet = sheet.resize((3 * W // 2, H))
    sheet.save(HERE / "results" / "examples.png")


if __name__ == "__main__":
    main()
