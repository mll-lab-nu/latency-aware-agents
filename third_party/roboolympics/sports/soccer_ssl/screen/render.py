"""Contact sheet of a worldrun trace (development aid): frames of the field every `--every` seconds from `--t0`."""
import argparse
import json

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--every", type=float, default=0.5)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--out", required=True)
a = ap.parse_args()
r = json.load(open(a.trace))
tr = [e for e in r["trace"] if e["t"] >= a.t0]
S = 40  # px per m
W, H = int(10 * S), int(7 * S)
frames = []
nxt = a.t0
for e in tr:
    if e["t"] + 1e-6 < nxt:
        continue
    nxt = e["t"] + a.every
    im = Image.new("RGB", (W, H), (30, 110, 40))
    d = ImageDraw.Draw(im)
    X = lambda x: W / 2 + x * S
    Y = lambda y: H / 2 - y * S
    d.rectangle([X(-4.5), Y(3), X(4.5), Y(-3)], outline="white")
    d.line([X(0), Y(3), X(0), Y(-3)], fill="white")
    for s in (-1, 1):
        d.rectangle([X(s * 4.5 if s < 0 else 3.5), Y(1), X(-3.5 if s < 0 else 4.5), Y(-1)], outline="white")
        d.line([X(s * 4.55), Y(0.5), X(s * 4.55), Y(-0.5)], fill=(255, 255, 255), width=4)
    for k, (x, y, th) in e["robots"].items():
        c = (40, 80, 255) if k[0] == "b" else (240, 220, 0)
        d.ellipse([X(x) - 4, Y(y) - 4, X(x) + 4, Y(y) + 4], fill=c)
        import math
        d.line([X(x), Y(y), X(x + 0.18 * math.cos(th)), Y(y + 0.18 * math.sin(th))], fill="black")
        d.text((X(x) + 4, Y(y) - 10), k[1:], fill="white")
    bx, by = e["ball"]
    d.ellipse([X(bx) - 3, Y(by) - 3, X(bx) + 3, Y(by) + 3], fill=(255, 120, 0))
    d.text((4, 2), f"t={e['t']:.1f} {e['state']} {e['score']}", fill="white")
    frames.append(im)
    if len(frames) >= a.n:
        break
cols = 3
rows = (len(frames) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * H), "black")
for i, f in enumerate(frames):
    sheet.paste(f, ((i % cols) * W, (i // cols) * H))
sheet.save(a.out)
