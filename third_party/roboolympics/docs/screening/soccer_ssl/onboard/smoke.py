"""Smoke test: World with onboard cameras, a few pictures, the projection check (ball centroid vs robot_to_pixel)."""
import math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
from PIL import Image
from sports.soccer_ssl.sim import onboard as OB
from sports.soccer_ssl.sim.world import World
out = HERE.parent / "results"
band = "--band" in sys.argv
with OB.world_builder(team_band=band) as cams:
    w = World(seed=1)
m, d = w.model, w.data
import mujoco as mj
print("extent", m.stat.extent, "znear", m.vis.map.znear * m.stat.extent)
# put the ball 0.8 m ahead-left of blue1 (the kicker), and render
p, _ = w.robot_pose(("blue", 1))
x, y, yaw = p
bx, by = x + 0.8 * math.cos(yaw) - 0.2 * math.sin(yaw), y + 0.8 * math.sin(yaw) + 0.2 * math.cos(yaw)
w.ball.place(bx, by); mj.mj_forward(m, d)
names = [cams[f"{t}{n}_robot"] for t, n in w.ids]
imgs = OB.render(m, d, names)
for (t, n), im in zip(w.ids, imgs):
    Image.fromarray(im).save(out / f"smoke_{t}{n}{'_band' if band else ''}.png")
im = imgs[w.ids.index(("blue", 1))].astype(int)
r, g, b = im[..., 0], im[..., 1], im[..., 2]
mask = (r > 150) & (g > 50) & (g < 170) & (b < 80)
vs, us = np.nonzero(mask)
c, s = math.cos(-yaw), math.sin(-yaw)
lx, ly = c * (bx - x) - s * (by - y), s * (bx - x) + c * (by - y)
print("ball px", len(us), "centroid", us.mean() if len(us) else None, vs.mean() if len(us) else None,
      "predicted", OB.robot_to_pixel(lx, ly, 0.0215))
print("horizon row", OB.horizon_row())
