"""The PNG encoding share of an observation: core/observe.png compresses at zlib level 6; how much would a lower level
save on the SSL tiles (mostly flat green)? Same container settings as bench.py; -> /out/pngcost_<tag>.json."""
from __future__ import annotations

import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parents[4] / "sports" / "soccer_ssl" / "sim"))


def png(img, level):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level)) + chunk(b"IEND", b""))


def main():
    import mujoco as mj
    import camera
    import stub_scene
    from core.observe import compose
    tag = sys.argv[1] if len(sys.argv) > 1 else "stub"
    out = []
    for layout, size in (("ssl4", (800, 600)), ("ssl4", (640, 480)), ("single", (1200, 840))):
        spec, names, _ = stub_scene.build(layout, size=size, offsamples=0)
        m = spec.compile()
        d = mj.MjData(m)
        mj.mj_forward(m, d)
        q, _ = stub_scene.random_state(m, np.random.default_rng(3))
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        sheet, _ = compose(camera.render(m, d, names, size), names, 2)
        for level in (1, 3, 6):
            ts = []
            for _ in range(10):
                t = time.perf_counter()
                b = png(sheet, level)
                ts.append((time.perf_counter() - t) * 1e3)
            ts.sort()
            out.append({"layout": layout, "size": size, "level": level, "ms": round(ts[5], 1), "kb": round(len(b) / 1024, 1)})
            print(out[-1], flush=True)
    Path(f"/out/pngcost_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
