"""Stamp the re-render tag (render.presentation_label) on every PNG frame of a directory, in place: path-traced
frames (Blender, Isaac Sim) are written without it.

    python3 stamp.py <frame dir> <engine: cycles|isaac> [<decor>]
"""
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import looks  # noqa: E402
from render import presentation_label  # noqa: E402

text = looks.output_label(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
for p in sorted(glob.glob(os.path.join(sys.argv[1], "frame_*.png"))):
    presentation_label(Image.open(p).convert("RGB"), text).save(p)
