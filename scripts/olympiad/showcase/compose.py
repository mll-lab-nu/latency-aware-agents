"""Three-way comparison of one replay window (runs in the MuJoCo image, CPU):
standard MuJoCo (competition) | showcase MuJoCo | showcase Cycles, as a 2x2 1080p clip and still triptychs.

    python3 compose.py <sport> <standard.mp4> <showcase.mp4> <cycles frame dir> <out prefix> [--fps 50] [--stills 1,5]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import font, label, writer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sport"); ap.add_argument("standard"); ap.add_argument("showcase"); ap.add_argument("cycles")
    ap.add_argument("out"); ap.add_argument("--fps", type=float, default=50.0); ap.add_argument("--stills", default="")
    a = ap.parse_args()
    import imageio.v2 as imageio
    rs, rw = imageio.get_reader(a.standard), imageio.get_reader(a.showcase)
    cyc = sorted(glob.glob(os.path.join(a.cycles, "frame_*.png")))
    stills = {int(round(float(t) * a.fps)) for t in a.stills.split(",") if t.strip()}
    W, H = 1920, 1080
    w = writer(a.out + "_three_way.mp4", a.fps)
    for i, (fs, fw, fc) in enumerate(zip(rs, rw, cyc)):
        imc = Image.open(fc).convert("RGB")
        panels = [label(Image.fromarray(fs).resize((W // 2, H // 2), Image.LANCZOS), "standard (competition)"),
                  label(Image.fromarray(fw).resize((W // 2, H // 2), Image.LANCZOS), "showcase: MuJoCo raster"),
                  label(imc.resize((W // 2, H // 2), Image.LANCZOS), "showcase: Blender Cycles")]
        frame = Image.new("RGB", (W, H), (12, 14, 20))
        for p, xy in zip(panels, ((0, 0), (W // 2, 0), (0, H // 2))):
            frame.paste(p, xy)
        d = ImageDraw.Draw(frame)
        f, fs_ = font(34), font(22)
        x0, y0 = W // 2 + 40, H // 2 + 60
        d.text((x0, y0), f"RoboOlympics {a.sport}", font=f, fill=(240, 240, 240))
        lines = ["One recorded match, re-rendered after the fact.",
                 "Same replayed states, scene geometry, colours",
                 "and broadcast camera in all three panels;",
                 "only lights, materials, resolution and",
                 "anti-aliasing / light transport differ.",
                 f"t = {i / a.fps:5.2f} s"]
        for j, ln in enumerate(lines):
            d.text((x0, y0 + 70 + 34 * j), ln, font=fs_, fill=(190, 195, 205))
        w.append_data(np.asarray(frame))
        if i in stills:
            tri = Image.new("RGB", (W, H * 3))
            tri.paste(label(Image.fromarray(fs), "standard (competition)"), (0, 0))
            tri.paste(label(Image.fromarray(fw), "showcase: MuJoCo raster"), (0, H))
            tri.paste(label(imc, "showcase: Blender Cycles"), (0, 2 * H))
            tri.save(f"{a.out}_t{int(round(i / a.fps * 100)):04d}_triptych.jpg", quality=92)
    w.close()


if __name__ == "__main__":
    main()
