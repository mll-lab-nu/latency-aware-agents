"""Broadcast render helper for the turn-based BILLIARDS (8-ball) tasks: runs in a helper container from the task's own
image (eai-olympiad-billiards: pooltool + Pillow, CPU only, no network), started and fed by scripts/olympiad/broadcast.py
over stdin/stdout (length-prefixed pickles, as render_helper.py).

It never talks to a match. It gets the copied sim directory of a match (the task's own eightball.py, duel_core.py,
render.py) and the decisions the host read from the sim's /data/episode.jsonl, and REPLAYS the game with the task's own
code (the game is a deterministic function of the seed, the breaker and the decisions; the verifier's replay.py does
the same): duel_core.Duel(seed, first).play(seat, decision) for a duel, eightball.Match(level, seed).shot(decision) /
.follow() (the scripted opponent) for the single-player task, with keep_systems=True so each shot's pooltool system
(the ball trajectories) is kept. Frames are drawn with the task's own pinhole renderer (render.render) from broadcast
cameras (overhead, behind the cue ball along the cue line, at a pocket), and the program feed's graphics
(program_gfx.py) go on top.

Requests (tuples) and replies:
  ("scene", key, simdir, sport, mode)             -> ("ok", info)       load a match's sim code
  ("sync", key, cfg, steps)                       -> ("ok", [shot info]) apply the decisions (only the new ones)
  ("frame", key, fs, overlay, (w, h))             -> ("ok", jpeg)       one frame (fs: what to draw, see Table.draw)
  ("still", key, fs, overlay, (w, h))             -> ("ok", png)
  ("clip", key, [fs...], [overlay...], (w, h), fps) -> ("ok", mp4)
cfg: {"mode": "duel", "seed", "first"} or {"mode": "single", "level", "seed"}; steps: [{"seat", "dec"}] (duel) or
[{"dec"} | {"follow": 1}] (single: the opponent plays after the player's review).
"""
from __future__ import annotations

import importlib
import io
import math
import os
import pickle
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import program_gfx  # noqa: E402

import pooltool as pt  # noqa: E402,F401  (imported once here: fresh_import must not drop and re-JIT it per match)
from pooltool.events import EventType  # noqa: E402
from PIL import ImageDraw  # noqa: E402


def _read_exact(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def recv(f):
    n = struct.unpack("<Q", _read_exact(f, 8))[0]
    return pickle.loads(_read_exact(f, n))


def send(f, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack("<Q", len(b)) + b)
    f.flush()


def fresh_import(simdir, names):
    """Import the match's sim modules and forget them again (two matches may run different versions)."""
    before = set(sys.modules)
    sys.path.insert(0, simdir)
    try:
        return [importlib.import_module(n) for n in names]
    finally:
        sys.path.remove(simdir)
        for n in set(sys.modules) - before:
            sys.modules.pop(n, None)


def look_at(eye, target, f, size):
    """A pinhole camera in render.py's convention: (position, R_world_to_camera, f, (w, h))."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return eye, np.stack([x, y, z]), float(f), tuple(size)


class Table:
    """One match: the task's own code, the replayed game, each shot's trajectories."""

    def __init__(self, simdir, sport, mode):
        names = ["eightball", "render"] + (["duel_core"] if mode == "duel" else [])
        mods = fresh_import(simdir, names)
        self.E8, self.R = mods[0], mods[1]
        self.D = mods[2] if mode == "duel" else None
        self.mode = mode
        self.cfg, self.m, self.applied = None, None, 0
        self.after, self.info, self.paths = [], [], []
        self.steps_seen = []

    # ---- the game (the task's own code)
    def _new(self, cfg):
        self.cfg = dict(cfg)
        if self.mode == "duel":
            self.m = self.D.Duel(int(cfg["seed"]), cfg["first"], keep_systems=True)
        else:
            self.m = self.E8.Match.__new__(self.E8.Match)            # Match.__init__ without its opening follow():
            self.m.level, self.m.seed = int(cfg["level"]), int(cfg["seed"])   # the shoot hook must see those shots
            lv = self.E8.LEVELS[self.m.level]
            self.m.g = self.E8.Game(self.m.seed, 0 if lv["breaks"] == "you" else 1, keep_systems=True)
        g = self.m.g
        self.initial = dict(g.pos)
        self.after, self.info, self.paths, self.applied = [], [], [], 0
        orig = g.shoot

        def shoot(p, dec):                     # every shot: remember the table after it
            rec = orig(p, dec)
            self.after.append(dict(g.pos))
            return rec
        g.shoot = shoot
        if self.mode != "duel":
            self.m.opening = self.m.follow()

    def sync(self, cfg, steps):
        if self.cfg != dict(cfg) or self.m is None or len(steps) < self.applied:
            self._new(cfg)
        for st in steps[self.applied:]:
            if self.m.g.done:
                break
            if self.mode == "duel":
                self.m.play(st["seat"], st["dec"])
            elif st.get("follow"):
                self.m.follow()
            else:
                self.m.shot(st["dec"])
        self.applied = len(steps)
        self.steps_seen = list(steps)
        g = self.m.g
        while len(self.info) < len(g.systems):
            self._add(len(self.info))
        return self.info

    def _add(self, i):
        k, p, pos0, s = self.m.g.systems[i]
        c = pt.continuize(s, dt=0.01, inplace=False)
        path, T = {}, 0.0
        for bid, b in c.balls.items():
            rvw, st, ts = b.history_cts.vectorize()
            path[bid] = (np.asarray(rvw[:, 0, :2], float), np.asarray(st), np.asarray(ts, float))
            T = max(T, float(ts[-1]) if len(ts) else 0.0)
        pots, t_first = [], None
        for e in s.events:
            if e.event_type == EventType.BALL_POCKET:
                pots.append((e.ids[0], e.ids[1], float(e.time)))
            elif e.event_type == EventType.BALL_BALL and t_first is None and "cue" in e.ids:
                t_first = float(e.time)
        rec = self.m.g.shots[i]
        self.paths.append(path)
        self.info.append(dict(n=i, k=int(k), player=int(p), T=round(T, 3), phi=float(s.cue.phi), V=float(s.cue.V0),
                              pos0={b: tuple(map(float, v)) for b, v in pos0.items()},
                              pos1={b: tuple(map(float, v)) for b, v in self.after[i].items()},
                              pots=pots, t_first=t_first,
                              rec=dict(rec)))

    # ---- positions
    def pos_at(self, i, t):
        """The balls of shot i at t seconds after the stroke (pocketed balls left out)."""
        out = {}
        for bid, (xy, st, ts) in self.paths[i].items():
            j = int(np.searchsorted(ts, t))
            if j <= 0:
                a = xy[0]
                s_ = st[0]
            elif j >= len(ts):
                a = xy[-1]
                s_ = st[-1]
            else:
                w = (t - ts[j - 1]) / max(1e-9, ts[j] - ts[j - 1])
                a = xy[j - 1] + (xy[j] - xy[j - 1]) * w
                s_ = st[j] if w > 0.5 else st[j - 1]
            if s_ >= 4:                                   # pocketed
                continue
            out[bid] = (float(a[0]), float(a[1]))
        return out

    # ---- cameras
    def camera(self, spec, i, size):
        E8, R = self.E8, self.R
        w, h = size
        kind = (spec or {}).get("kind", "top")
        if kind == "top":                                 # the task's top camera, refitted to the broadcast frame
            pos, Rwc, _, _ = R.top_camera()
            sc = min(0.78 * w / (E8.L + 0.26), 0.72 * h / (E8.W + 0.26))
            return np.array([pos[0] - spec.get("shift", 0.03), pos[1], pos[2]]), Rwc, sc * pos[2], (w, h)
        info = self.info[i] if i is not None and i < len(self.info) else None
        if kind == "cue" and info is not None:            # behind the cue ball, along the executed cue line
            c = np.array(info["pos0"].get("cue", E8.HEAD_SPOT), float)
            d = np.array([math.cos(math.radians(info["phi"])), math.sin(math.radians(info["phi"]))])
            back, up, ahead = spec.get("back", 0.95), spec.get("height", 0.68), spec.get("ahead", 0.75)
            eye = [c[0] - back * d[0], c[1] - back * d[1], up]
            tgt = [c[0] + ahead * d[0], c[1] + ahead * d[1], 0.0]
            return look_at(eye, tgt, 0.68 * w, (w, h))
        if kind == "pocket" and info is not None:         # just beyond the pocket, facing the ball rolling in
            pk = np.array(E8.POCKETS[spec["pocket"]], float)
            tp = spec.get("t_pot", 0.0)
            pts = self.paths[i].get(spec.get("ball"))
            src = None
            if pts is not None:
                xy, st, ts = pts
                src = xy[min(int(np.searchsorted(ts, max(0.0, tp - 0.25))), len(xy) - 1)]
            d = pk - (src if src is not None and np.linalg.norm(pk - src) > 0.05 else np.array([E8.W / 2, E8.L / 2]))
            d /= np.linalg.norm(d)                         # the ball's approach direction
            eye = [pk[0] + 0.36 * d[0], pk[1] + 0.36 * d[1], 0.42]
            tgt = [pk[0] - 0.12 * d[0], pk[1] - 0.12 * d[1], 0.0]
            return look_at(eye, tgt, 0.62 * w, (w, h))
        return self.camera({"kind": "top"}, i, size)

    # ---- frames
    def draw(self, fs, ov, size):
        """fs: {"after": n} the table after shot n (-1: the rack before the first shot); or {"shot": i, "t": s} shot i
        at t s after the stroke (t < 0: lined up, the cue stick drawn); "cam": a camera spec; "marks": gold
        rings around the called ball ({"ball": "5"}) and the called pocket ({"pocket": "lt"})."""
        if ov.get("blank"):
            return program_gfx.decorate_program(program_gfx.blank(size), self._ov(ov))
        i = fs.get("shot")
        stick = None
        if i is not None and i < len(self.info):
            t = float(fs.get("t", 0.0))
            if t < 0:
                pos = dict(self.info[i]["pos0"])
                stick = self.info[i]["phi"]
            else:
                pos = self.pos_at(i, min(t, self.info[i]["T"]))
        else:
            n = fs.get("after", -1)
            pos = dict(self.initial) if n < 0 or not self.after else dict(self.after[min(n, len(self.after) - 1)])
        # The task's CPU rasterizer defaults to 2x supersampling. At 1080p that
        # creates a 4K intermediate every frame; use native HD for live/replay.
        self.R.SS = 1 if size[1] >= 1080 else 2
        cam = self.camera(fs.get("cam"), i, size)
        im = self.R.render(cam, pos, stick_phi=stick if "cue" in pos else None)
        if fs.get("marks"):
            mk = [(*pos[m["ball"]], "ball") for m in fs["marks"] if m.get("ball") in pos] + \
                 [(*self.E8.POCKETS[m["pocket"]], "pocket") for m in fs["marks"] if m.get("pocket") in self.E8.POCKETS]
            self.marks(im, cam, mk)
        program_gfx.decorate_program(im, self._ov(ov))
        return im

    def _ov(self, ov):
        ov = dict(ov)
        ov.setdefault("ball_rgb", self.R.BALL_RGB)
        return ov

    def marks(self, im, cam, marks):
        """Gold rings on the cloth around the called ball and the called pocket (projected with the same camera)."""
        pr = self.R._Proj(cam)
        dr = ImageDraw.Draw(im, "RGBA")
        for x, y, kind in marks:
            r = (self.E8.R * 1.9) if kind == "ball" else 0.085
            ring = self.R._circle((x, y), r, 48, z=0.001 if kind != "ball" else self.E8.R)
            pc = pr.cam(ring)
            if (pc[:, 2] <= 0.05).any():
                continue
            px = pr.pix(pc) / self.R.SS
            pts = [tuple(p) for p in px] + [tuple(px[0])]
            col = (255, 200, 40, 235)
            dr.line(pts, fill=col, width=max(2, int(im.size[1] / 180)))


def jpeg(im, q=90):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=0)
    return b.getvalue()


def clip(tb, fss, ovs, size, fps):
    import imageio.v2 as imageio
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        w = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2, quality=None,
                               ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                                              "-movflags", "+faststart", "-threads", "1"])
        for fs, ov in zip(fss, ovs):
            w.append_data(np.asarray(tb.draw(fs, ov, size)))
        w.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def main():
    fin = sys.stdin.buffer
    fout = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                              # anything printed goes to stderr, the protocol keeps the real stdout
    sys.stdout = sys.stderr
    tables = {}
    send(fout, ("ok", "ready"))
    while True:
        try:
            req = recv(fin)
        except EOFError:
            return
        t0 = time.time()
        try:
            op = req[0]
            if op == "scene":
                _, key, simdir, sport, mode = req
                tables[key] = Table(simdir, sport, mode)
                out = {"sport": sport, "mode": mode}
            elif op == "drop":
                tables.pop(req[1], None)
                out = None
            elif op == "sync":                 # steps None: the host asks only for the facts already computed
                _, key, cfg, steps = req
                if steps is None:
                    steps = tables[key].steps_seen
                out = [{k: v for k, v in x.items()} for x in tables[key].sync(cfg, steps)]
            elif op == "frame":
                _, key, fs, ov, size = req
                out = jpeg(tables[key].draw(fs, ov, tuple(size)))
            elif op == "still":
                _, key, fs, ov, size = req
                b = io.BytesIO()
                tables[key].draw(fs, ov, tuple(size)).save(b, "PNG")
                out = b.getvalue()
            elif op == "clip":
                _, key, fss, ovs, size, fps = req
                out = clip(tables[key], fss, ovs, tuple(size), fps)
            else:
                raise ValueError(f"unknown request {op}")
            send(fout, ("ok", out, round(time.time() - t0, 4)))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            send(fout, ("error", repr(exc)[:300], round(time.time() - t0, 4)))


if __name__ == "__main__":
    main()
