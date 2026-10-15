#!/usr/bin/env python3
"""实时比赛画面: live broadcast video of the running real-time RoboOlympics matches, plus an instant replay of every point.

    python3 scripts/olympiad/broadcast.py [--port 8048] [--hosts mll-4090-2.cs.northwestern.edu]

Endpoints: /program/<match> (MJPEG, the PROGRAM FEED: a TV-style broadcast with an automatic director, see "the
program feed" below; /program/<match>.jpg one frame of it), /stream/<match> (MJPEG, the plain live camera, kept for the
live board), /replay/<match>/<point|last> (MP4 of one finished point at real speed), /snapshot/<match>.jpg,
/api/matches (JSON, CORS open: the live board on :8047 embeds the streams) and / (the viewer page: the program feed
big, a small match switcher).

Program feed: live while the world runs; when it freezes after a point (review pause / match end) a RoboOlympics
stinger wipe (0.8 s) and a highlight package of that point (main camera at real speed, then the camera behind the
player who struck last and the overhead camera at 0.5x around the decisive moment, then a close-up at 0.25x of the
last contact or the winning bounce), a stinger back to live as soon as play resumes, calm loops while a long pause
lasts (live hold with captions, then the package again; after the match: every point in turn). Lower thirds per
seat carry the agents' latest public message / action, or their submitted review (WHY -> PLAN), read from the live
board (:8047, which reads the seats' Claude Code / Codex logs); never hidden reasoning, secrets redacted. The score
bug, LIVE/REPLAY tag and speed note are always on; everything is drawn into the frames by the render helper.
Test offline: program_harness.py (a finished match's log replayed as live) and test_broadcast_program.py.

Turn-based BILLIARDS (8-ball, duel and single player) is on the same page and program feed, read from the sim's own
episode log and redrawn with the task's own code in a helper from the task's image (see "turn-based: billiards" below;
test_broadcast_billiards.py, program_harness.py with a billiards trial).

SOCCER (RoboOlympics 2 v 2 Booster T1: single, duel, four-agent team) is a real-time sport with its own match, director
packages per stoppage and captions (team channel, model output), rendered in a helper from the soccer task's image
(see "real-time: soccer" below; test_broadcast_soccer.py).

3D RACING (two cars, heats; single vs the scripted car and duel) and AIR HOCKEY (two KUKA iiwa arms; single and duel)
are real-time sports on the standard image with their own matches, director packages (racing: crashes as instant
replays while the race goes on, heat ends; air hockey: goals and stoppages) and graphics (see "real-time: 3D racing and
air hockey" below; test_broadcast_racing.py, test_broadcast_airhockey.py, program_harness.py with a finished trial).

Read-only access to the matches: the sims (round 2 on) append a compact state log, /data/broadcast.jsonl
(broadcast/broadcast_log.py), and nothing else changes there. This process finds the running matches like the live
board does (live_board.discover: `docker ps` and the Harbor job directories, locally and over ssh), reads only the new
bytes of each log with `docker exec <sim> tail -c +N` (never a game endpoint), copies the match's sim directory (scene
files) once, and draws the frames OUTSIDE the sim: a render helper container from the eai-olympiad image (MuJoCo,
OSMesa CPU or optional NVIDIA EGL, no network, --cpus capped, nice 10) rebuilds the MuJoCo state from the logged qpos.
Nothing is read or drawn while nobody watches (no page or stream request within IDLE_S seconds).

Live video: the world runs slowed by S, so the live picture is slow motion (shown on screen); it trails the sim by
about DELAY_S seconds (the log is read every TAIL_S s) and is interpolated between logged frames (every 2 control
steps). The frame rate is up to MAX_FPS per stream, lower when several streams share the helper or the host is busy.
Replays are rendered once per point on demand (or as soon as the point ends while someone watches) and cached on disk.
"""
from __future__ import annotations

import argparse
import functools
import socket
from collections import deque
import bisect
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
import time
import pickle
import urllib.request
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "broadcast"))
import audience  # noqa: E402
from delivery import Delivery, PROFILES  # noqa: E402
from review_editor import ReviewEditor  # noqa: E402
import live_board as lb  # noqa: E402  (discovery, docker exec / ssh helpers, job names, match ids)

IMAGE = "ghcr.io/mll-lab-nu/eai-olympiad:0.1.0"
# Spectator picture only (never a match's or an agent's camera): "standard" = the competition scene as is,
# "showcase" = the same scene with better lights/materials/anti-aliasing (scripts/olympiad/showcase/looks.py;
# table-tennis and tennis duels). Costs more CPU per frame on OSMesa; meant for demos, not the Season feed.
RENDER_QUALITY = os.environ.get("BROADCAST_RENDER_QUALITY", "standard")
if RENDER_QUALITY not in ("standard", "showcase"):
    raise SystemExit("BROADCAST_RENDER_QUALITY must be standard or showcase")
CACHE = Path(os.environ.get("BROADCAST_CACHE", Path.home() / ".cache" / "olympiad_broadcast"))
REVIEW_EDITOR = ReviewEditor(
    os.environ.get("BROADCAST_REVIEW_CACHE", str(CACHE / "review-summaries")),
    enabled=os.environ.get("BROADCAST_SUMMARIZE_REVIEWS") == "1", api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    model=os.environ.get("BROADCAST_SUMMARY_MODEL", "openai/gpt-6-luna"))
SUPPORTED = {(sp, mode) for sp in ("tabletennis", "tennis", "badminton", "baseball") for mode in ("single", "duel")}
# turn-based sports: no broadcast log; the sim's own episode log (/data/episode.jsonl) is read and every shot is
# recomputed with the task's own code in a helper container from the task's own image (see "turn-based" below)
TURN_BASED = {"billiards"}
SUPPORTED |= {("billiards", "duel"), ("billiards", "single")}
BILLIARDS_IMAGE = "ghcr.io/mll-lab-nu/eai-olympiad-billiards:0.1.0"
# RoboOlympics soccer (2 v 2 Booster T1): single (vs the program team), duel (one seat per team), team (four seats)
SOCCER = {"soccer"}
SUPPORTED |= {("soccer", m) for m in ("single", "duel", "team")}
SOCCER_IMAGE = "ghcr.io/mll-lab-nu/eai-olympiad-soccer:0.1.0"
UNIT = {"baseball": "pitch", "billiards": "shot", "soccer": "stoppage"}   # what a replay covers (a point, a pitch in baseball, a shot)
TAIL_S, DISCOVER_S, IDLE_S, DELAY_S = 1.0, 15.0, 60.0, 2.0
# CPU remains available on machines without the viewer-only EGL images.
RENDER_BACKEND = os.environ.get("BROADCAST_RENDERER", "osmesa")
if RENDER_BACKEND not in {"osmesa", "egl"}:
    raise ValueError("BROADCAST_RENDERER must be osmesa or egl")
QUALITY = os.environ.get("BROADCAST_QUALITY", "720p" if RENDER_BACKEND == "egl" else "540p")
SIZES = {"540p": (960, 540), "720p": (1280, 720), "1080p": (1920, 1080)}
SIZE = SIZES[QUALITY]
MAX_FPS, MIN_FPS = min(60.0, max(3.0, float(os.environ.get("BROADCAST_FPS", "30")))), 3.0
GPU_DEVICE = os.environ.get("BROADCAST_GPU", "0")
EGL_IMAGE = os.environ.get("BROADCAST_EGL_IMAGE", "eai-broadcast-egl:0.1.0")
SOCCER_EGL_IMAGE = os.environ.get("BROADCAST_SOCCER_EGL_IMAGE", "eai-broadcast-soccer-egl:0.1.0")
TRAIL = 8                 # logged frames of the ball's recent path drawn behind it
REPLAY_FPS = 30
FONT_DIR = next((d for d in (os.environ.get("BC_FONT_DIR", ""), "/usr/share/fonts/opentype/noto",
                              "/usr/share/fonts/truetype/wqy", "/usr/share/fonts/truetype/droid") if d and os.path.isdir(d)), "")
LOCK = threading.Lock()
STATE = {"matches": {}, "last_view": 0.0, "hosts": [], "errors": {}, "discovered": 0.0}

BC_SCRIPT = r"""
cd /data 2>/dev/null || { echo "@@BC -1"; exit 0; }
off=$1; cap=$2; f=${3:-broadcast.jsonl}
s=$(stat -c %s "$f" 2>/dev/null || echo -1)
echo "@@BC $s"
[ "$s" -gt "$off" ] && tail -c +$((off+1)) "$f" | head -c "$cap"
exit 0
"""


# ---------------------------------------------------------------- the render helper container

def egl_args():
    return ["--gpus", f"device={GPU_DEVICE}", "-e", "NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute",
            "-e", "MUJOCO_GL=egl", "-e", "PYOPENGL_PLATFORM=egl"]


@functools.lru_cache(maxsize=4)
def probe_egl(image):
    """Do a real draw: an EGL context alone may silently be Mesa software rendering."""
    name = f"eai-broadcast-probe-{os.getuid()}-{os.getpid()}-{threading.get_ident()}"
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "1", "--memory", "1g",
             *egl_args(), "-v", f"{HERE / 'broadcast'}:/bc:ro", image, "python3", "/bc/egl_probe.py"],
            capture_output=True, timeout=25)
        if result.returncode == 0:
            return result.stdout.decode().strip()
        print(f"Broadcast EGL unavailable ({image}); using CPU. See renderer image/driver setup.", file=sys.stderr)
        return ""
    except (OSError, subprocess.TimeoutExpired):
        return ""
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


class Helper:
    """One render helper process (a container); requests are serialized."""

    def __init__(self, role, cpus, image=None, script="render_helper.py"):
        self.role, self.cpus, self.image, self.script = role, cpus, image, script
        self.lock = threading.Lock()
        self.proc = None
        self.scenes = set()
        self.rt = 0.05                     # running mean seconds per frame
        self.busy_s = 0.0                  # helper seconds spent (CPU use estimate)
        self.backend, self.device = "osmesa", "CPU"
        self.synced = {}                   # billiards: match id -> decisions replayed in this helper

    def start(self):
        name = f"eai-broadcast-{self.role}-{os.getuid()}-{os.getpid()}"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        render_image = self.image or IMAGE
        gl_args = ["-e", "MUJOCO_GL=osmesa", "-e", "PYOPENGL_PLATFORM=osmesa"]
        self.backend, self.device = "osmesa", "CPU"
        if RENDER_BACKEND == "egl" and self.script == "render_helper.py":
            candidate = SOCCER_EGL_IMAGE if render_image == SOCCER_IMAGE else EGL_IMAGE
            device = probe_egl(candidate)
            if device:
                render_image, gl_args = candidate, egl_args()
                self.backend, self.device = "egl", device
        self.proc = subprocess.Popen(
            ["docker", "run", "-i", "--rm", "--name", name, "--network", "none", "--cpus", str(self.cpus),
             "--memory", "3g", "--pids-limit", "64", *gl_args,
             "-e", f"LP_NUM_THREADS={max(1, int(self.cpus))}", "-e", "OMP_NUM_THREADS=1",
             "-e", f"RENDER_QUALITY={RENDER_QUALITY}", "-v", f"{HERE / 'showcase'}:/showcase:ro",
             "-v", f"{CACHE / 'sims'}:/cache/sims:ro", "-v", f"{HERE / 'broadcast'}:/bc:ro", "-w", "/bc",
             *(["-v", f"{FONT_DIR}:/fonts:ro", "-e", "BC_FONT_DIR=/fonts"] if FONT_DIR else []),
             render_image, "nice", "-n", "10", "python3", f"/bc/{self.script}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(CACHE / f"helper-{self.role}.log", "ab"))
        self.scenes = set()
        self.synced = {}
        if self._recv()[1] != "ready":
            raise RuntimeError("render helper did not start")

    def _recv(self):
        h = self.proc.stdout.read(8)
        if len(h) < 8:
            raise EOFError("render helper exited")
        n = struct.unpack("<Q", h)[0]
        b = self.proc.stdout.read(n)
        return pickle.loads(b)

    def call(self, req, timeout_hint=None):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.proc is None or self.proc.poll() is not None:
                        self.start()
                    if req[0] != "scene" and req[1] not in self.scenes:
                        raise KeyError("scene")
                    b = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
                    self.proc.stdin.write(struct.pack("<Q", len(b)) + b)
                    self.proc.stdin.flush()
                    rep = self._recv()
                    self.busy_s += rep[2] if len(rep) > 2 else 0.0
                    if rep[0] != "ok":
                        raise RuntimeError(rep[1])
                    if req[0] == "scene":
                        self.scenes.add(req[1])
                    return rep[1], (rep[2] if len(rep) > 2 else 0.0)
                except KeyError:
                    raise
                except (EOFError, BrokenPipeError, OSError):
                    if self.proc is not None:
                        self.proc.kill()
                    self.proc = None
                    if attempt:
                        raise
        raise RuntimeError("unreachable")

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()


LIVE = Helper("live", 2)
CLIP = Helper("clip", 1)
BILL = Helper("billiards", 2, image=BILLIARDS_IMAGE, script="billiards_helper.py")        # started on first use only
BILL_CLIP = Helper("billiards-clip", 1, image=BILLIARDS_IMAGE, script="billiards_helper.py")
SOC = Helper("soccer", 2, image=SOCCER_IMAGE)                  # soccer: the task's image (T1 meshes at /opt/sai_t1)
SOC_CLIP = Helper("soccer-clip", 1, image=SOCCER_IMAGE)


def helper_call(helper, bm, req):
    """A request for match bm; (re)loads its scene into the helper first if needed (with the match's scene options,
    e.g. racing's track seed from the log header)."""
    try:
        return helper.call(req)
    except KeyError:
        opts = bm.scene_opts() if hasattr(bm, "scene_opts") else None
        helper.call(("scene", bm.id, f"/cache/sims/{bm.id}", bm.sport, bm.mode) + ((opts,) if opts else ()))
        return helper.call(req)


# ---------------------------------------------------------------- per match
class BMatch:
    LOG = "broadcast.jsonl"
    NO_LOG = "This match has no broadcast log (older simulator version)"
    turn_based = False

    def __init__(self, d):
        self.__dict__.update(d)
        self.info = lb.parse_job(self.job, self.proj)
        self.id = lb.match_id(self.proj, self.host)
        self.sport, self.mode = self.info["sport"], "duel" if self.info["duel"] else "single"
        self.off, self.buf, self.hdr = 0, b"", None
        self.ks, self.recs, self.events, self.points = [], [], [], []
        self.size = -1
        self.simdir_ok = False
        self.err = ""
        self.last_tail = 0.0
        self.tail_lock = threading.Lock()
        self.stream = None
        self.program = None                # the program feed (Program), while watched
        self.replays = {}                  # point n -> path
        self.replay_lock = threading.Lock()
        self.gone = None                   # time the match left the discovery (kept a while for its replays)

    # ---- reading the log (incremental)
    def tail(self, force=False):
        if not self.tail_lock.acquire(blocking=False):
            return
        try:
            if not force and time.time() - self.last_tail < TAIL_S or self.gone:
                return
            self.last_tail = time.time()
            out = lb.dexec(self.host, self.sim, BC_SCRIPT, self.off, 4_000_000, self.LOG, timeout=20)
            if out is None:
                self.err = "Could not read the broadcast log"
                return
            i = out.find(b"@@BC ")
            j = out.find(b"\n", i)
            if i < 0 or j < 0:
                self.err = "Could not read the broadcast log"
                return
            header = out[i + 5:j].strip()
            if not re.fullmatch(rb"-?\d+", header):
                self.err = "Incomplete broadcast log response, retrying"
                return
            self.size = int(header)
            if self.size < 0:
                self.err = self.NO_LOG
                return
            data = out[j + 1:]
            if self.size < self.off:          # replaced: start over
                self.reset()
                return
            self.off += len(data)
            lines = (self.buf + data).split(b"\n")
            self.buf = lines.pop()
            for ln in lines:
                self.ingest(ln)
            self.err = ""
            if not self.simdir_ok:
                self.copy_simdir()
        except (OSError, ValueError, tarfile.TarError) as ex:
            self.err = f"Could not read the broadcast log, retrying: {ex}"[:200]
        finally:
            self.tail_lock.release()

    def reset(self):
        self.off, self.buf, self.ks, self.recs, self.events, self.points, self.hdr = 0, b"", [], [], [], [], None

    def ready(self):
        """Something to draw: frames read and the scene files copied."""
        return bool(self.ks) and self.simdir_ok

    def ingest(self, ln):
        try:
            e = json.loads(ln)
        except ValueError:
            return
        if "q" in e and "ev" not in e:
            k = int(e["k"])
            if self.ks and k <= self.ks[-1]:
                return
            self.recs.append(e)
            self.ks.append(k)
            return
        ev = e.get("ev")
        if ev == "hdr":
            self.hdr = e
            self.mode = e.get("mode", self.mode)
            self.sport = e.get("sport", self.sport)
            return
        self.events.append(e)
        if ev == "point":
            k0 = self.points[-1]["k"] if self.points else (self.ks[0] - 1 if self.ks else -1)
            self.points.append(dict(n=len(self.points) + 1, k0=k0, k=int(e["k"]), score=e.get("score"),
                                    winner=e.get("winner"), how=e.get("how") or "", note=e.get("note"),
                                    flight=e.get("flight")))

    def copy_simdir(self):
        dst = CACHE / "sims" / self.id
        if (dst / ".ok").exists():
            self.simdir_ok = True
            return
        raw = lb.run(self.host, ["docker", "exec", self.sim, "tar", "-C", "/srv", "-cf", "-", "--exclude=__pycache__",
                                 "--exclude=think_proxy.py", "."], timeout=60)
        if not raw:
            self.err = "Could not copy the scene files"
            return
        tmp = dst.with_name(self.id + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for mem in t.getmembers():
                if mem.issym() or mem.islnk() or mem.name.startswith("/") or ".." in Path(mem.name).parts:
                    continue
                t.extract(mem, tmp)
        (tmp / ".ok").write_text(time.strftime("%F %T"))
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
        self.simdir_ok = True

    # ---- what is shown
    @property
    def slowdown(self):
        try:
            return float((self.hdr or {}).get("slowdown") or self.info.get("slow") or 1)
        except (TypeError, ValueError):
            return 1.0

    @property
    def dt(self):
        return float((self.hdr or {}).get("dt") or 0.01)

    def labels(self):
        p = self.info["players"]
        if self.mode == "duel":
            return tuple(audience.names({s: nice_name(p.get(s, s)) for s in ("A", "B")}).values())
        if self.sport == "baseball":
            return f"{p.get('A', 'Player')}  bases", f"target L{self.info.get('level', '?')}"
        return f"{p.get('A', 'Player')}", f"CPU L{self.info.get('level', '?')}"

    @property
    def unit(self):
        return UNIT.get(self.sport, "point")

    def note_at(self, k):
        e = next((p for p in reversed(self.points) if p["k"] <= k), None)
        return e.get("note") if e else None

    def score_at(self, k):
        sc = None
        for p in self.points:
            if p["k"] <= k:
                sc = p["score"]
        if sc is None:
            return 0, 0
        if isinstance(sc, dict):
            return sc.get("A", 0), sc.get("B", 0)
        return sc[0], sc[1]

    def who(self, w):
        a, b = self.labels()
        return {"you": a, "A": a, "opponent": b, "B": b, "let": "let", None: "let"}.get(w, str(w))

    def audience_reason(self, p):
        return audience.referee_reason(p["how"], dict(zip(("A", "B"), self.labels())), self.sport)

    def result_text(self, p):
        if self.unit == "pitch" or p["winner"] is None and self.sport == "baseball":
            return f"Pitch {p['n']}: {self.audience_reason(p)}"
        return f"Point {p['n']} to {self.who(p['winner'])}: {self.audience_reason(p)}"

    def trail(self, i, n=TRAIL):
        """The records before record i in the same point (the ball's recent path)."""
        k = self.ks[i]
        kmin = max([p["k"] for p in self.points if p["k"] < k], default=-1)
        return [self.recs[j] for j in range(max(0, i - n), i) if self.ks[j] > kmin]

    def rows(self, k):
        a, b = self.labels()
        sa, sb = self.score_at(k)
        return [(a, sa), (b, sb)]

    def phase_at(self, k):
        """'pause' (review), 'end' or 'run' at the latest log position, for the live caption."""
        ph = "wait"
        for e in self.events:
            if e.get("k", 0) <= k and e["ev"] in ("run", "pause", "end", "start"):
                ph = e["ev"]
        return ph

    def live_overlay(self, k, at_end):
        S = self.slowdown
        tag = "LIVE  x%g slow motion" % S if S != 1 else "LIVE  real speed"
        ov = dict(rows=self.rows(k), tag=tag, live=True, note=self.note_at(k),
                  sub=(f"the game runs at 1/{S:g} of real speed" if S != 1 else ""),
                  clock=f"game time {k * self.dt:7.2f} s · {self.unit} {sum(1 for p in self.points if p['k'] <= k) + 1}")
        last = next((p for p in reversed(self.points) if p["k"] <= k), None)
        ph = self.phase_at(k) if at_end else "run"
        if ph == "end":
            ov["caption"] = "Match over"
        elif ph == "pause":
            ov["caption"] = "Review: the world is frozen until the players are ready"
        elif ph in ("wait", "hdr"):
            ov["caption"] = "Waiting for the players to start"
        if last and (k - last["k"]) * self.dt < 1.5 and ph != "end":
            ov["caption"] = self.result_text(last)[:110]
        return ov

    def info_json(self):
        a, b = self.labels()
        return dict(id=self.id, sport=self.sport, mode=self.mode, host=lb.short(self.host), job=self.job,
                    players=[a, b], slowdown=self.slowdown, has_log=self.hdr is not None, err=self.err,
                    supported=(self.sport, self.mode) in SUPPORTED, frames=len(self.ks),
                    dt=self.dt, k=self.ks[-1] if self.ks else None, phase=self.phase_at(self.ks[-1]) if self.ks else None,
                    score=list(self.score_at(self.ks[-1] if self.ks else 0)), gone=bool(self.gone),
                    unit=self.unit,
                    points=[dict(n=p["n"], winner=(self.who(p["winner"]) if p["winner"] is not None else ""),
                                 how=self.audience_reason(p), score=p["score"]) for p in self.points],
                    viewers=(self.stream.clients if self.stream else 0) + (self.program.clients if self.program else 0),
                    playback=(self.program or self.stream).playback_info() if (self.program or self.stream) else None,
                    program=f"/program/{self.id}")

    def live_helper(self):
        return LIVE

    def clip_helper(self):
        return CLIP

    program_cls = None                     # the program feed's class (None: Program)

    # ---- replays
    def replay(self, n):
        """The MP4 of finished point n (rendered once, cached)."""
        with self.replay_lock:
            if n in self.replays and self.replays[n].exists():
                return self.replays[n]
            p = next((x for x in self.points if x["n"] == n), None)
            if p is None:
                return None
            lo = bisect.bisect_right(self.ks, p["k0"])
            hi = bisect.bisect_right(self.ks, p["k"])
            if hi <= lo:
                return None
            step = max(1, round(1.0 / REPLAY_FPS / self.dt))          # control steps per video frame (real speed)
            recs, ovs = [], []
            a, b = self.labels()
            before = self.score_at(p["k"] - 1) if p["k0"] < p["k"] else (0, 0)
            cap = self.result_text(p)[:110]
            tag = f"REPLAY  {self.unit} {n}"
            note0 = self.note_at(p["k0"]) if p["k0"] >= 0 else None
            k, kend = self.ks[lo], p["k"]
            while k <= kend:
                i = min(bisect.bisect_left(self.ks, k), hi - 1)
                recs.append(self.recs[i])
                ovs.append(dict(rows=[(a, before[0]), (b, before[1])], tag=tag, live=False, note=note0,
                                sub="real speed", clock=f"game time {k * self.dt:7.2f} s", trail=self.trail(i)))
                k += step
            after = self.score_at(p["k"])
            fl = p.get("flight") or []
            if len(fl) > 1:                                             # baseball: the batted ball's flight
                last = self.recs[hi - 1]
                tr = []
                for t, x, y, z in fl:
                    r = dict(last, ball=[x, y, z], t=10.0)
                    recs.append(r)
                    ovs.append(dict(rows=[(a, before[0]), (b, before[1])], tag=tag, live=False, cam="wide", halo=0.9,
                                    sub="real speed", clock=f"batted ball {t:5.2f} s", trail=list(tr[-TRAIL:])))
                    tr.append(r)
            hold, htrail = (recs[-1], tr[-TRAIL - 1:-1]) if len(fl) > 1 else (self.recs[hi - 1], self.trail(hi - 1))
            for _ in range(int(REPLAY_FPS * 1.6)):                      # hold the end with the result
                recs.append(hold)
                ovs.append(dict(rows=[(a, after[0]), (b, after[1])], tag=tag, live=False, note=self.note_at(p["k"]),
                                sub="real speed", caption=cap, trail=htrail, cam="wide" if len(fl) > 1 else None,
                                halo=0.9 if len(fl) > 1 else None))
            mp4, _ = helper_call(self.clip_helper(), self, ("clip", self.id, recs, ovs, SIZE, REPLAY_FPS, "main"))
            out = CACHE / "replays" / self.id / f"{n}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(mp4)
            self.replays[n] = out
            return out


# ---------------------------------------------------------------- live streams
class Stream(threading.Thread):
    """The live broadcast of one match while at least one client watches it."""

    def __init__(self, bm):
        super().__init__(daemon=True)
        self.bm = bm
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0
        self.delivery = Delivery()
        self.clients = 0
        self.last_client = time.time()
        self.pos = None
        self.fps = MAX_FPS
        self.last_key = None
        self.frames = 0
        self.reader_stop = threading.Event()
        self.frame_times = deque(maxlen=160)
        self.display_step = None
        self.display_phase = "wait"
        self.buffering = False

    def read_log(self):
        """Docker/SSH reads must never block the render clock."""
        while not self.reader_stop.is_set():
            try:
                self.bm.tail()
            except Exception as ex:
                self.bm.err = f"Could not read the broadcast log, retrying: {ex}"[:200]
            if self.bm.gone:
                return
            self.reader_stop.wait(TAIL_S)

    def run(self):
        threading.Thread(target=self.read_log, daemon=True).start()
        try:
            self.render_loop()
        finally:
            self.reader_stop.set()

    def playback_info(self):
        now = time.monotonic()
        stamps = list(self.frame_times)
        recent = [t for t in stamps if t >= now - 3.0]
        return dict(step=self.display_step, phase=self.display_phase, buffering=self.buffering,
                    fps=round(len(recent) / 3.0, 1),
                    behind_s=round(max(0, self.bm.ks[-1] - self.display_step) * self.bm.dt * self.bm.slowdown, 2)
                    if self.display_step is not None and self.bm.ks else None)

    def render_loop(self):
        bm = self.bm
        last = time.monotonic()
        last_sent = 0.0
        while True:
            if self.clients <= 0 and time.time() - self.last_client > 10:
                with LOCK:
                    if bm.stream is self:
                        bm.stream = None
                return
            t0 = time.monotonic()
            if not bm.ks or not bm.simdir_ok:
                time.sleep(0.5)
                continue
            now = time.monotonic()
            rate = 1.0 / (bm.slowdown * bm.dt)                       # control steps per wall second
            latest = bm.ks[-1]
            delay = DELAY_S * rate
            if self.pos is None or latest - self.pos > 3 * delay + 4 * rate:
                self.pos = max(bm.ks[0], latest - delay)
            else:
                source_phase = bm.phase_at(latest)
                if self.buffering and latest - self.pos < delay and source_phase == "run":
                    last = now
                    time.sleep(1.0 / MAX_FPS)
                    continue
                self.buffering = False
                desired = self.pos + (now - last) * rate
                self.pos = min(latest, desired)
                if desired > latest and source_phase == "run":
                    self.buffering = True
            last = now
            i = max(0, bisect.bisect_right(bm.ks, self.pos) - 1)
            ra = bm.recs[i]
            rb, alpha = None, 0.0
            if i + 1 < len(bm.ks):
                gap = bm.ks[i + 1] - bm.ks[i]
                if gap <= 2 * int((bm.hdr or {}).get("every", 2)) and gap > 0:
                    rb, alpha = bm.recs[i + 1], (self.pos - bm.ks[i]) / gap
            k_show = bm.ks[i] + alpha * (bm.ks[i + 1] - bm.ks[i]) if rb is not None else bm.ks[i]
            ov = bm.live_overlay(int(k_show), at_end=self.pos >= latest - 1e-6)
            key = (round(k_show, 2), json.dumps(ov, sort_keys=True))
            if key != self.last_key or time.monotonic() - last_sent > 2.0:
                try:
                    if key != self.last_key:
                        ov["trail"] = bm.trail(i)
                        hp = getattr(bm, "live_helper", lambda: LIVE)()
                        jpg, rt = helper_call(hp, bm, ("frame", bm.id, ra, rb, alpha, ov, SIZE, "main"))
                        hp.rt = 0.8 * hp.rt + 0.2 * rt
                        self.jpeg = jpg
                        self.last_key = key
                        self.frames += 1
                        self.frame_times.append(time.monotonic())
                        self.display_step = k_show
                        self.display_phase = bm.phase_at(latest) if self.pos >= latest - 1e-6 else "run"
                    with self.cond:
                        self.seq += 1
                        self.cond.notify_all()
                    last_sent = time.monotonic()
                except Exception as ex:  # noqa: BLE001
                    bm.err = f"Render error: {ex}"[:200]
                    time.sleep(1.0)
            self.fps = target_fps(SOC) if getattr(bm, "sport", None) in SOCCER else target_fps()
            time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - t0)))


def target_fps(helper=None):
    """Budget only feeds sharing this helper, including soccer's plain camera."""
    helper = helper or LIVE
    with LOCK:
        n = 0
        for m in STATE["matches"].values():
            pool = BILL if m.turn_based else SOC if m.sport in SOCCER else LIVE
            if pool is helper:
                n += sum(st is not None and st.clients > 0 for st in (m.stream, m.program))
    fps = min(MAX_FPS, 0.85 / (max(1, n) * max(helper.rt, 1e-3)))
    try:
        if os.getloadavg()[0] > 0.9 * os.cpu_count():
            fps /= 2
    except OSError:
        pass
    return max(MIN_FPS, fps)


def get_stream(bm):
    with LOCK:
        if bm.stream is None or not bm.stream.is_alive():
            bm.stream = Stream(bm)
            bm.stream.start()
        return bm.stream


# ---------------------------------------------------------------- the program feed (automatic director)
# One TV-style program stream per match (/program/<match>): the live picture while the world runs; when the world
# freezes after a point (review pause, match end, or no new frame for FREEZE_S), a "RoboOlympics" stinger wipe and a
# highlight package of the point just finished (real speed, then other cameras in slow motion around the decisive
# moment), a stinger back to live as soon as play resumes, and, while a long pause lasts, calm loops (live hold with
# the agents' captions, the package again). The agents' latest public messages / actions / reviews run as lower thirds
# on the picture; the score bug, LIVE/REPLAY tag and speed note are always on. Everything is drawn into the JPEG frames
# by the render helper (broadcast/program_gfx.py), so the MJPEG stream itself carries it.
BOARD_URL = os.environ.get("BOARD_URL", "http://127.0.0.1:8047")
STINGER_S = 0.8            # the wipe; the cut happens under the fully closed wipe at its middle
FREEZE_S = 2.5             # no new frame for this long after a point: treat the world as frozen
LIVE_HOLD_S = 8.0          # in a long pause: live (frozen picture, agents' captions) between highlight loops
CAPTION_S = 4.0            # how often the agents' activity is read (from the live board)
REVIEW_HOLD_S = 45.0       # a submitted review (WHY / PLAN) stays the seat's caption this long
SPORT_CN = {"tabletennis": "Table tennis", "tennis": "Tennis", "badminton": "Badminton", "baseball": "Baseball"}
BALL_KEY = {"tennis": "b", "badminton": "sh", "baseball": "ball"}           # table tennis: qpos[0:3]
BOUNCE_Z = 0.12            # a ball turning upwards below this height (m) bounced on the table / court surface
COURT_HALF = {"tabletennis": 1.37, "tennis": 11.9, "badminton": 6.7}      # half length of the table / court (y)
CLOSE_DIST = {"tabletennis": 1.6, "tennis": 4.2, "badminton": 3.8, "baseball": 5.0}
SHOT_CAMS = {"baseball": dict(main="main", end="behind", end_b="behind", high="wide", close="plate_side")}
TOKENISH = re.compile(r"\b(?=[A-Za-z0-9+/_=-]*\d)(?=[A-Za-z0-9+/_=-]*[A-Za-z])[A-Za-z0-9+/_=-]{32,}\b")


def redact(s):
    """Secrets out (the live board's patterns plus any long token-like string), whitespace collapsed."""
    return TOKENISH.sub("[redacted]", lb.clean(s, 2000))


def short_cmd(c):
    c = next((ln for ln in str(c).splitlines() if ln.strip()), "").strip()
    c = re.sub(r"^(?:/bin/)?(?:ba)?sh -l?c ", "", c).strip("'\" ")
    if "<<" in c:                                       # an inline script: its command line only
        c = c.split("<<")[0].strip() + " ‹inline script›"
    c = re.sub(r"^(?:cd \S+\s*(?:&&|;)\s*)+", "", c)
    c = re.sub(r"^(?:(?:timeout|nice)\s+(?:-\S+\s+)*\S+\s+|nohup\s+|env\s+(?:\S+=\S+\s+)+|[A-Z_]+=\S+\s+)+", "", c)
    c = re.sub(r"/app/output/|/app/", "", c)
    c = re.sub(r"\s*(?:>|2>|&>)\s*\S+|\s*2>&1|\s*&\s*$", "", c)
    return c.strip()[:90]


def caption_of(kind, text):
    """A harness step (live_board.harness_steps: kind, text) as a lower-third line (label, text), or None for what is
    not shown: hidden reasoning ("思考"), tool results, turn bookkeeping, errors."""
    if kind == "说":
        return "Says", text
    if kind in ("运行命令", "调用 Bash", "调用 exec_command", "调用 shell", "调用 exec"):
        c = audience.command_activity(text)
        return ("Action", c) if c else None
    if kind in ("修改文件", "调用 Edit", "调用 Write", "调用 MultiEdit", "调用 NotebookEdit", "调用 apply_patch"):
        return "Action", "Updating the control program"
    if kind in ("调用 Read", "调用 view_image"):
        return "Action", "Inspecting a saved image" if "image" in kind else "Reading the control program"
    if kind in ("调用 Grep", "调用 Glob"):
        return "Action", "Looking up control-program details"
    if kind.startswith("调用 ") and kind not in ("调用 TodoWrite",):
        return None
    return None


REV = re.compile(r"^复盘 (\S+?)(?:（[^）]*）)*\s*WHY：(.*?)\s*PLAN：(.*)$")


def bj_age(t, now):
    """Age of a live-board clock string (Beijing HH:MM:SS of today)."""
    try:
        h, m, s = map(int, t.split(":"))
    except (AttributeError, ValueError):
        return None
    sec = (now + lb.BJT) % 86400
    return (sec - (h * 3600 + m * 60 + s)) % 86400


class Captions:
    """The agents' lower thirds of one match. Source: the live board's state (/api/state on :8047; live_board.py reads
    each seat's Claude Code / Codex log and the episode log's reviews, read-only). Per seat the newest public item:
    a submitted review (WHY → PLAN, kept REVIEW_HOLD_S), else the latest assistant message or tool action. Never the
    hidden reasoning; secrets redacted."""
    shared = {"t": 0.0, "data": None}
    shared_lock = threading.Lock()

    def __init__(self, bm):
        self.bm = bm
        self.seat = {}           # seat -> dict(kind, text, t)
        self.review = {}         # seat -> dict(text, t, key)
        self.seen = set()

    def card(self, now):
        """The live board's card of this match (one fetch per CAPTION_S for all programs)."""
        with Captions.shared_lock:
            if now - Captions.shared["t"] >= CAPTION_S:
                Captions.shared["t"] = now
                try:
                    with urllib.request.urlopen(BOARD_URL + "/api/state", timeout=3) as r:
                        Captions.shared["data"] = json.load(r)
                except (OSError, ValueError):
                    Captions.shared["data"] = None
            data = Captions.shared["data"]
        return next((c for c in (data or {}).get("cards", []) if c.get("id") == self.bm.id), None)

    def update(self, now):
        card = self.card(now)
        if not card:
            return
        for seat in card.get("seats", []):
            s = str(seat.get("seat") or "A")
            for st in seat.get("steps", []):          # newest first
                cap = caption_of(st.get("k", ""), st.get("t", ""))
                if cap is None:
                    continue
                text = redact(cap[1])
                old = self.seat.get(s)
                if old is None or old["text"] != text:
                    age = seat.get("h_age")
                    self.seat[s] = dict(kind=cap[0], text=text,
                                        t=(now - age) if (old is None and age is not None) else now)
                break
        for e in reversed(card.get("events", [])):    # oldest first
            if e.get("c") != "rev":
                continue
            m = REV.match(e.get("x", ""))
            if not m:
                continue
            who, why, plan = m.groups()
            s = who[0] if who[:1] in ("A", "B") else "A"
            key = (e.get("t"), e.get("x"))
            if key in self.seen:
                continue
            self.seen.add(key)
            age = e.get("age") if e.get("age") is not None else bj_age(e.get("t"), now)
            text = redact(f"{why} → {plan}" if why and plan else (why or plan))
            self.review[s] = dict(text=text, t=now - (age or 0))

    def prepare_reviews(self):
        for review in list(self.review.values()):
            summary = REVIEW_EDITOR.request(review["text"], self.bm.sport)
            if summary:
                review["summary"] = summary

    def lower(self, now, seats):
        out = []
        for s, name in seats:
            rv, it = self.review.get(s), self.seat.get(s)
            if rv and now - rv["t"] < REVIEW_HOLD_S and (not it or it["t"] <= rv["t"] + REVIEW_HOLD_S):
                kind, text, t = "Review", rv["text"], rv["t"]
            elif it:
                kind, text, t = it["kind"], it["text"], it["t"]
            else:
                continue
            age = max(0, int(now - t))
            out.append(dict(seat=s if self.bm.mode == "duel" else "", name=name, kind=kind, text=text,
                            age=f"{age} s ago" if age < 120 else f"{age // 60} min ago",
                            summary=rv.get("summary") if kind == "Review" and rv else None))
        return out


def nice_name(n):
    n = re.sub(r"^[AB]\s+", "", str(n or "")).strip()
    return n[:1].upper() + n[1:] if n.islower() else n


def ball_of(bm, rec):
    v = rec.get("q")[0:3] if bm.sport == "tabletennis" and rec.get("q") else rec.get(BALL_KEY.get(bm.sport, ""))
    return v if v is not None and len(v) >= 3 else None


def key_moments(bm, lo, hi):
    """Contacts and bounces of the ball in records lo..hi-1: (k, kind, pos) where its flight turns sharply."""
    out = []
    pts = []
    for i in range(lo, hi):
        b = ball_of(bm, bm.recs[i])
        pts.append(None if b is None or b[2] < -1.0 else (bm.ks[i], [float(x) for x in b]))
    for j in range(1, len(pts) - 1):
        a, b, c = pts[j - 1], pts[j], pts[j + 1]
        if not (a and b and c) or b[0] - a[0] <= 0 or c[0] - b[0] <= 0:
            continue
        v1 = [(b[1][x] - a[1][x]) / ((b[0] - a[0]) * bm.dt) for x in range(3)]
        v2 = [(c[1][x] - b[1][x]) / ((c[0] - b[0]) * bm.dt) for x in range(3)]
        n1, n2 = sum(x * x for x in v1) ** 0.5, sum(x * x for x in v2) ** 0.5
        if n1 < 0.3 or n2 < 0.3 or n1 > 120 or n2 > 120:                     # parked, or a teleport (reset)
            continue
        cos = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
        if cos > 0.94:                                                       # < ~20 degrees: plain flight
            continue
        kind = "bounce" if v1[2] < 0 < v2[2] and b[1][2] < BOUNCE_Z else "contact"   # on the table / ground
        if out and b[0] - out[-1][0] <= 3 * int((bm.hdr or {}).get("every", 2)):
            continue
        out.append((b[0], kind, b[1]))
    return out


def build_package(bm, p, recap=False):
    """The highlight package of finished point p: a list of shots dict(cam, k0, k1, speed, label, hold)."""
    lo = bisect.bisect_right(bm.ks, p["k0"])
    hi = bisect.bisect_right(bm.ks, p["k"])
    if hi - lo < 2:
        return []
    dt, kend, kfirst = bm.dt, bm.ks[hi - 1], bm.ks[lo]
    sec = lambda s: int(round(s / dt))
    moments = [m for m in key_moments(bm, lo, hi) if m[0] < kend]
    contacts = [m for m in moments if m[1] == "contact"]
    key = moments[-1] if moments else (max(kfirst, kend - sec(0.5)), "contact", ball_of(bm, bm.recs[hi - 1]))
    hit = contacts[-1] if contacts else key
    half = COURT_HALF.get(bm.sport)
    if key[1] == "bounce" and contacts and (half is None or key[2] is None or abs(key[2][1]) > half):
        key = hit                       # a bounce off the court (behind a player): show the last stroke instead
    cams = SHOT_CAMS.get(bm.sport, {})
    end_cam = cams.get("end_b" if (hit[2] is not None and hit[2][1] > 0) else "end", "end_b" if (hit[2] is not None and hit[2][1] > 0) else "end")
    who = "B" if end_cam.endswith("_b") else "A"
    clip = lambda a, b: (max(kfirst, a), min(kend, b))
    kmoment = "bounce" if key[1] == "bounce" else "last hit"
    look = list(key[2]) if key[2] is not None else [0.0, 0.0, 0.5]
    look[2] = max(0.15, look[2])
    side = -1.0 if look[1] < 0 else 1.0               # the camera comes from the side, towards the court's centre
    close = cams.get("close") or (look, CLOSE_DIST.get(bm.sport, 2.0), 180.0 - 40.0 * side, -28.0)
    shots = []
    if not recap:
        a, b = clip(kend - sec(4.0 if bm.sport == "tabletennis" else 6.0), kend)
        shots.append(dict(cam=cams.get("main", "main"), k0=a, k1=b, speed=1.0, label="Main camera · real speed", hold=0.4))
    a, b = clip(key[0] - sec(1.2), key[0] + sec(0.9))
    shots.append(dict(cam=end_cam, k0=a, k1=b, speed=0.5,
                      label=("Catcher view" if bm.sport == "baseball" else f"Behind {'B' if who == 'B' else 'A'}") + " · 0.5×", hold=0.2))
    if not recap:
        a, b = clip(key[0] - sec(0.6), key[0] + sec(0.6))
        shots.append(dict(cam=cams.get("high", "high"), k0=a, k1=b, speed=0.5, label="Overhead · 0.5×", hold=0.2))
    a, b = clip(key[0] - sec(0.35), key[0] + sec(0.45))
    shots.append(dict(cam=close, k0=a, k1=b, speed=0.25, label=f"Close-up · {kmoment} · 0.25×", hold=1.2))
    return [s for s in shots if s["k1"] > s["k0"]]


class Program(Stream):
    """The program feed of one match while at least one client watches it (see the comment above)."""

    def __init__(self, bm):
        super().__init__(bm)
        self.caps = Captions(bm)
        self.mode = "live"                 # live | replay
        self.stinger = None                # dict(t0, to, pkg, point) while the wipe runs
        self.pkg, self.shot_i, self.shot_t0, self.pkg_point = [], 0, 0.0, None
        self.replayed = set()              # points whose package has started
        self.live_since = time.monotonic()
        self.n_loop, self.n_end = 0, 0
        self.last_n, self.last_new = 0, time.monotonic()
        self.log = deque(maxlen=200)       # director decisions (time, text), for /api/matches and the harness
        self.shot_label = ""
        self.clock = time.monotonic

    # ---- the live playhead (as Stream.render_loop)
    def live_step(self, now):
        bm = self.bm
        rate = 1.0 / (bm.slowdown * bm.dt)
        latest = bm.ks[-1]
        delay = DELAY_S * rate
        if self.pos is None or latest - self.pos > 3 * delay + 4 * rate:
            self.pos = max(bm.ks[0], latest - delay)
        else:
            source_phase = bm.phase_at(latest)
            if self.buffering and latest - self.pos < delay and source_phase == "run":
                pass
            else:
                self.buffering = False
                desired = self.pos + (now - self.last_t) * rate
                self.pos = min(latest, desired)
                if desired > latest and source_phase == "run":
                    self.buffering = True
        self.last_t = now

    def frozen(self, now):
        bm = self.bm
        if len(bm.ks) != self.last_n:
            self.last_n, self.last_new = len(bm.ks), now
        ph = bm.phase_at(bm.ks[-1])
        return ph in ("pause", "end") or (now - self.last_new > FREEZE_S and self.pos >= bm.ks[-1] - 1e-6)

    def note(self, text):
        self.log.append((time.time(), text))

    # ---- the director
    def direct(self, now):
        bm = self.bm
        if self.stinger:
            sp = (now - self.stinger["t0"]) / STINGER_S
            if sp >= 0.5 and not self.stinger.get("cut"):
                self.stinger["cut"] = True
                if self.stinger["to"] == "replay":
                    self.mode, self.pkg, self.shot_i, self.shot_t0 = "replay", self.stinger["pkg"], 0, now
                    self.pkg_point = self.stinger["point"]
                else:
                    self.mode, self.live_since = "live", now
            if sp >= 1.0:
                self.stinger = None
            return
        last = bm.points[-1] if bm.points else None
        frozen = self.frozen(now)
        if self.mode == "live":
            if not (last and frozen and self.pos >= last["k"] and bm.ks[-1] <= last["k"] + 1):
                return
            ended = bm.phase_at(bm.ks[-1]) == "end"
            if last["n"] not in self.replayed:
                p, recap = last, False
            elif now - self.live_since >= LIVE_HOLD_S:   # a long pause: loop calmly (at the end: the whole match)
                if ended:
                    self.n_end += 1
                    p, recap = bm.points[(self.n_end - 1) % len(bm.points)], False
                else:
                    self.n_loop += 1
                    p, recap = last, self.n_loop % 2 == 1
            else:
                return
            pkg = self.package(p, recap=recap)
            if not pkg:
                self.replayed.add(p["n"])
                return
            self.replayed.add(p["n"])
            self.stinger = dict(t0=now, to="replay", pkg=pkg, point=p)
            self.note(f"cut to replay of {bm.unit} {p['n']} ({'recap' if recap else 'package'}: "
                      + ", ".join(f"{s['label']}" for s in pkg) + ")")
            return
        # replay
        resumed = last is not None and bm.ks[-1] > last["k"] + 1 and bm.phase_at(bm.ks[-1]) == "run"
        if resumed:
            self.stinger = dict(t0=now, to="live")
            self.note("play resumed: back to live")
            return
        shot = self.pkg[self.shot_i]
        dur = (shot["k1"] - shot["k0"]) * bm.dt / shot["speed"] + shot["hold"]
        if now - self.shot_t0 >= dur:
            self.shot_i += 1
            self.shot_t0 = now
            if self.shot_i >= len(self.pkg):
                self.shot_i = len(self.pkg) - 1
                self.stinger = dict(t0=now, to="live")
                self.note("package done: back to live")

    def package(self, p, recap=False):
        return build_package(self.bm, p, recap=recap)

    # ---- what is drawn
    def seats(self):
        p = self.bm.info["players"]
        if self.bm.mode == "duel":
            return list(audience.names({s: nice_name(p.get(s, s)) for s in ("A", "B")}).items())
        return [("A", nice_name(p.get("A", "Player")))]

    def base_overlay(self, now, k_score):
        bm = self.bm
        sa, sb = bm.score_at(k_score)
        if bm.mode == "duel":
            rows = [["A", self.seats()[0][1], sa], ["B", self.seats()[1][1], sb]]
        elif bm.sport == "baseball":
            rows = [["A", self.seats()[0][1] + " bases", sa], ["", f"Target L{bm.info.get('level', '?')}", sb]]
        else:
            rows = [["A", self.seats()[0][1], sa], ["", f"Program L{bm.info.get('level', '?')}", sb]]
        n_done = sum(1 for p in bm.points if p["k"] <= k_score)
        unit = "Pitch" if bm.unit == "pitch" else "Point"
        ph = bm.phase_at(bm.ks[-1])
        phase = {"pause": "Review pause · world frozen", "end": "Match over", "wait": "Waiting for players", "start": "Waiting for players"}.get(ph, "")
        return dict(program=True, bug=dict(sport=SPORT_CN.get(bm.sport, bm.sport) + (" · Duel" if bm.mode == "duel" else ""),
                                           unit=f"{unit} {n_done + (0 if ph == 'end' else 1)}", rows=rows),
                    phase=phase, lower=self.caps.lower(time.time(), self.seats()))

    def result_cn(self, p):
        if self.bm.unit == "pitch" or p["winner"] is None:
            return f"Pitch {p['n']} · {self.bm.audience_reason(p)}"
        w = dict(self.seats()).get(p["winner"]) if self.bm.mode == "duel" else (
            self.seats()[0][1] if p["winner"] in ("you", "A") else "Program")
        return f"Point {p['n']} · {w or p['winner']} scores — {self.bm.audience_reason(p)}"

    def record_at(self, kf):
        bm = self.bm
        i = max(0, bisect.bisect_right(bm.ks, kf) - 1)
        ra, rb, alpha = bm.recs[i], None, 0.0
        if i + 1 < len(bm.ks):
            gap = bm.ks[i + 1] - bm.ks[i]
            if 0 < gap <= 2 * int((bm.hdr or {}).get("every", 2)):
                rb, alpha = bm.recs[i + 1], min(1.0, max(0.0, (kf - bm.ks[i]) / gap))
        return i, ra, rb, alpha

    def compose(self, now):
        """(record a, record b, alpha, overlay, frame key) of the frame to show now."""
        bm = self.bm
        S = bm.slowdown
        if self.mode == "replay" and self.pkg:
            shot = self.pkg[self.shot_i]
            kf = min(shot["k1"], shot["k0"] + (now - self.shot_t0) * shot["speed"] / bm.dt)
            i, ra, rb, alpha = self.record_at(kf)
            ov = self.base_overlay(now, bm.ks[-1])
            if bm.unit == "point":
                ov["bug"]["unit"] = f"Point {self.pkg_point['n']} · Replay"
            shot_label = re.sub(r"\bBehind ([AB])\b", lambda m: "Behind " + dict(self.seats()).get(m[1], m[1]), shot["label"])
            ov.update(tag="REPLAY", speed="Real speed 1×" if shot["speed"] == 1 else f"Slow motion {shot['speed']:g}×",
                      shot=shot_label, banner=self.result_cn(self.pkg_point), cam=shot["cam"])
            ov["trail"] = bm.trail(i)
            if shot["cam"] != "main" and isinstance(shot["cam"], (list, tuple)):
                ov["halo"] = 0.6 * ({"tabletennis": 0.045, "tennis": 0.16, "badminton": 0.11}.get(bm.sport, 0.1))
            self.shot_label = shot["label"]
        else:
            i, ra, rb, alpha = self.record_at(self.pos)
            k_show = bm.ks[i] + alpha * (bm.ks[i + 1] - bm.ks[i]) if rb is not None else bm.ks[i]
            ov = self.base_overlay(now, int(k_show))
            ov.update(tag="LIVE", speed=f"World slowed ×{S:g}" if S != 1 else "Real time")
            last = next((p for p in reversed(bm.points) if p["k"] <= k_show), None)
            if last and (k_show - last["k"]) * bm.dt < 1.5 and self.pos < bm.ks[-1] - 1e-6:
                ov["banner"] = self.result_cn(last)
            elif last and self.pos >= last["k"] and bm.phase_at(bm.ks[-1]) in ("pause", "end"):
                ov["banner"] = self.result_cn(last)
            ov["trail"] = bm.trail(i)
            self.display_step = k_show
            self.shot_label = "Live"
        if self.stinger:
            sp = min(1.0, (now - self.stinger["t0"]) / STINGER_S)
            ov["stinger"] = round(sp, 3)
            if abs(sp - 0.5) < 0.04:
                ov["blank"] = True
        key = (self.mode, round(alpha, 3), id(ra), json.dumps({k: v for k, v in ov.items() if k != "trail"},
                                                               sort_keys=True, default=str))
        return ra, rb, alpha, ov, key

    def caption_loop(self):
        while not self.reader_stop.is_set():
            try:
                self.caps.update(time.time())
                self.caps.prepare_reviews()
            except Exception as ex:  # noqa: BLE001
                self.bm.err = f"Caption error: {ex}"[:200]
            self.reader_stop.wait(CAPTION_S)

    def run(self):
        threading.Thread(target=self.read_log, daemon=True).start()
        threading.Thread(target=self.caption_loop, daemon=True).start()
        try:
            self.render_loop()
        finally:
            self.reader_stop.set()

    def playback_info(self):
        info = super().playback_info()
        info.update(mode=self.mode, stinger=bool(self.stinger), shot=self.shot_label,
                    replay_of=self.pkg_point["n"] if self.mode == "replay" and self.pkg_point else None,
                    director=[dict(t=round(t, 1), x=x) for t, x in list(self.log)[-6:]])
        return info

    def tick(self):
        """One frame decision + render (returns True when a new JPEG was made)."""
        bm = self.bm
        now = self.clock()
        if not hasattr(self, "last_t"):
            self.last_t = now
        self.live_step(now)
        self.direct(now)
        ra, rb, alpha, ov, key = self.compose(now)
        if key == self.last_key:
            return False
        hp = bm.live_helper()
        jpg, rt = helper_call(hp, bm, ("frame", bm.id, ra, rb, alpha, ov, SIZE, "main"))
        hp.rt = 0.8 * hp.rt + 0.2 * rt
        self.jpeg, self.last_key = jpg, key
        self.frames += 1
        self.frame_times.append(time.monotonic())
        self.display_phase = "replay" if self.mode == "replay" else bm.phase_at(bm.ks[-1])
        return True

    def render_loop(self):
        bm = self.bm
        last_sent = 0.0
        while True:
            if self.clients <= 0 and time.time() - self.last_client > 10:
                with LOCK:
                    if bm.program is self:
                        bm.program = None
                return
            t0 = time.monotonic()
            if not bm.ready():
                time.sleep(0.5)
                continue
            try:
                new = self.tick()
                if new or time.monotonic() - last_sent > 2.0:
                    with self.cond:
                        self.seq += 1
                        self.cond.notify_all()
                    last_sent = time.monotonic()
            except Exception as ex:  # noqa: BLE001
                bm.err = f"Render error: {ex}"[:200]
                time.sleep(1.0)
            self.fps = self.target_fps()
            time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - t0)))

    def target_fps(self):
        return target_fps()


def get_program(bm):
    with LOCK:
        if bm.program is None or not bm.program.is_alive():
            bm.program = (BilliardsProgram if bm.turn_based else (bm.program_cls or Program))(bm)
            bm.program.start()
        return bm.program


# ---------------------------------------------------------------- turn-based: billiards (8-ball)
# The billiards tasks (billiards-duel-program-standard, billiards-eightball-program-standard) have no broadcast log: the
# sim appends one "decision" and one "shot" event per stroke and a "review" per review to /data/episode.jsonl (read
# incrementally like the broadcast log). The game is deterministic (the seed, the breaker and the decisions), so a helper
# container from the task's own image (broadcast/billiards_helper.py: pooltool + the copied sim code, CPU only, no
# network) replays it with the task's own code and draws each shot with the task's own renderer. The director:
#   * a new shot: LIVE on the overhead camera — the cue lined up with the call marked, the stroke at real speed, the
#     final layout with the result; several shots in a row are shown back to back (2x when far behind);
#   * then (nothing pending) a RoboOlympics stinger, the instant replay — behind the cue ball along the cue line at
#     0.5x, then at the pocket of the (called) ball at 0.5x — and a stinger back to the live table;
#   * between shots: the live table still with whose turn it is, groups and balls left, ball in hand, the last result,
#     the reviews (WHY -> PLAN, from the episode log) and the agents' public actions (live board) as lower thirds; while
#     a long think lasts, the last shot's replay again every B_IDLE_S (after the match: the shots that potted in turn).
POCKET_EN = {"lb": "top-left corner", "lc": "top side pocket", "lt": "top-right corner",       # as on the overhead
             "rb": "bottom-left corner", "rc": "bottom side pocket", "rt": "bottom-right corner"}  # camera (x down, y right)
SOLIDS, STRIPES = tuple(str(i) for i in range(1, 8)), tuple(str(i) for i in range(9, 16))
ALL_BALLS = SOLIDS + ("8",) + STRIPES
B_AIM_S, B_HOLD_S = 1.6, 2.2       # live: the cue lined up before the stroke; the final layout with the result
B_IDLE_S = 16.0                    # between shots: replay the last shot again after this long on the still table
B_BACKLOG_FAST, B_BACKLOG_SKIP = 2, 6    # shots waiting to be shown: above these, 2x; skip to the latest few
B_REPLAY_MAX_T = 4.5               # the cue-line replay covers at most this much of the shot (s, game time)
B_REPLAY_MIN_S = 3.0               # a replay runs at least this long before a new shot cuts it (no flash cuts)
SPORT_CN["billiards"] = "Billiards (8-ball)"


def seat_of(p):
    return "A" if p in (0, "A", "0") else "B"


def bill_call(helper, bm, req):
    """A request for billiards match bm; (re)loads its scene and replays its new decisions into the helper first if
    needed (a ("sync", ...) request only does that and returns the shot facts)."""
    for attempt in (0, 1):
        try:
            if req[1] not in helper.scenes:
                raise KeyError("scene")
            if helper.synced.get(bm.id) != len(bm.steps):
                bm.sync(helper)
            return helper.call(req) if req[0] != "sync" else (bm.binfo, 0.0)
        except KeyError:
            if attempt:
                raise
            helper.call(("scene", bm.id, f"/cache/sims/{bm.id}", bm.sport, bm.mode))
            helper.synced.pop(bm.id, None)


class BilliardsMatch(BMatch):
    LOG = "episode.jsonl"
    NO_LOG = "Waiting for the match's episode log"
    turn_based = True

    def __init__(self, d):
        super().__init__(d)
        self.reset_b()

    def reset(self):
        super().reset()
        self.reset_b()

    def reset_b(self):
        self.cfg, self.steps, self.shots, self.reviews, self.binfo = None, [], [], [], []
        self.end_ev, self.begun, self.clock_left = None, False, {}
        self._states = None

    def ready(self):
        return self.simdir_ok and self.cfg is not None

    @property
    def slowdown(self):
        return 1.0

    @property
    def dt(self):
        return 1.0

    def ingest(self, ln):
        try:
            e = json.loads(ln)
        except ValueError:
            return
        ev = e.get("ev")
        if ev == "service":                                   # duel
            self.cfg = dict(mode="duel", seed=int(e["seed"]), first=e.get("first_player", "A"))
            self.hdr, self.mode = e, "duel"
        elif ev == "start" and "level" in e and "seed" in e:  # single player
            self.cfg = dict(mode="single", level=int(e["level"]), seed=int(e["seed"]))
            self.hdr, self.mode = e, "single"
        elif ev in ("match_begins", "match_start"):
            self.begun = True
        elif ev == "decision":
            self.steps.append(dict(seat=e.get("seat", "A"), dec=e["decision"]) if self.mode == "duel" else dict(dec=e["decision"]))
        elif ev == "shot":
            self.shots.append(dict(e, seat=seat_of(e.get("player"))))
            self._states = None
        elif ev == "review":
            self.reviews.append(e)
            if self.mode != "duel":                           # single: the scripted opponent plays after the review
                self.steps.append(dict(follow=1))
        elif ev == "end":
            self.end_ev = e
        if isinstance(e.get("clock_left_s"), (int, float)):
            self.clock_left[e.get("seat") or "A"] = float(e["clock_left_s"])
        if ev:
            self.events.append(e)
        if ev in ("shot", "end"):
            self.refresh_points()

    def refresh_points(self):
        """One replay entry per shot (the page's list and /replay/<match>/<n>)."""
        self.points = [dict(n=i + 1, k0=i, k=i + 1, score=list(self.left(self.state(i))), winner=r["seat"],
                            how=self.summary(i, r), note=None, flight=None) for i, r in enumerate(self.recs_b())]

    def sync(self, helper=None):
        """Replay the decisions in the helper (only the new ones are simulated) and keep its shot facts."""
        helper = helper or BILL
        steps = list(self.steps)
        info, _ = helper.call(("sync", self.id, self.cfg, steps))       # (a request's own steps are ignored)
        helper.synced[self.id] = len(steps)
        if len(info) >= len(self.binfo):
            self.binfo = info
            self._states = None
            self.refresh_points()
            logged = {r.get("k"): r for r in self.shots}
            for x in info:                                   # the replay must agree with the sim's own log
                r = logged.get(x["rec"]["k"])
                if r is not None and (x["rec"]["potted"], x["rec"]["foul"]) != (r.get("potted"), r.get("foul")):
                    self.err = f"Replay differs from the episode log at shot {x['rec']['k'] + 1}"
                    break

    # ---- the game state, from the shot records
    def recs_b(self):
        """The shot records: the helper's (the task's own Game, which also holds the single player task's unlogged
        opening shots) once synced, else the episode log's."""
        if len(self.binfo) >= len(self.shots) and self.binfo:
            return [dict(x["rec"], seat=seat_of(x["rec"]["player"])) for x in self.binfo]
        return self.shots

    def state(self, n):
        """The table after shot n (0-based; -1: before the first shot): balls on the table, groups, turn, ball in hand."""
        recs = self.recs_b()
        if self._states is None or len(self._states) != len(recs):
            first = recs[0]["seat"] if recs else (self.cfg or {}).get("first", "A")
            st = dict(on=list(ALL_BALLS), group={"A": None, "B": None}, turn=first, bih="kitchen", done=False,
                      winner=None, result=None, shot=0)
            out = []
            for r in recs:
                st = dict(st, group=dict(st["group"]), on=list(st["on"]), shot=st["shot"] + 1)
                me, other = r["seat"], ("B" if r["seat"] == "A" else "A")
                if r.get("rerack"):
                    st["on"] = list(ALL_BALLS)
                else:
                    gone = [b for b in (r.get("potted") or []) if not (b == "8" and r.get("eight_respotted"))]
                    gone += [b for b in (r.get("off_table") or []) if b != "8" or not r.get("eight_respotted")]
                    st["on"] = [b for b in st["on"] if b not in gone]
                if r.get("group_assigned"):
                    g = r["group_assigned"]
                    st["group"][me], st["group"][other] = g, ("stripes" if g == "solids" else "solids")
                if r.get("game_over"):
                    res = r.get("result") or ""
                    st.update(done=True, result=res, bih=None,
                              winner=me if res.startswith("8 potted legally") else (None if res.startswith("shot limit") else other))
                else:
                    st["turn"] = me if r.get("keeps_turn") else other
                    st["bih"] = ("kitchen" if r.get("break") or r.get("rerack") else "anywhere") if r.get("foul") else None
                out.append(st)
            if out and self.end_ev is not None and self.end_ev.get("winner") is not None:
                out[-1] = dict(out[-1], winner=seat_of(self.end_ev["winner"]))
            self._states = out
            self._st0 = dict(on=list(ALL_BALLS), group={"A": None, "B": None}, turn=first, bih="kitchen", done=False,
                             winner=None, result=None, shot=0)
        if n < 0 or not self._states:
            return self._st0
        return self._states[min(n, len(self._states) - 1)]

    @staticmethod
    def group_balls(st, seat):
        g = st["group"][seat]
        if g is None:
            return None
        left = [b for b in (SOLIDS if g == "solids" else STRIPES) if b in st["on"]]
        return left or (["8"] if "8" in st["on"] else [])

    def left(self, st):
        """Balls left in each player's group (7 while the table is open), as the task's own score."""
        return tuple(len([b for b in (SOLIDS if st["group"][s] == "solids" else STRIPES) if b in st["on"]])
                     if st["group"][s] else 7 for s in ("A", "B"))

    # ---- names and texts (English)
    def names(self):
        p = self.info["players"]
        if self.mode == "duel":
            return {"A": nice_name(p.get("A", "A")), "B": nice_name(p.get("B", "B"))}
        lv = (self.cfg or {}).get("level") or self.info.get("level", "?")
        return {"A": nice_name(p.get("A", "Player")), "B": f"Program L{lv}"}

    def labels(self):
        n = self.names()
        return (f"A  {n['A']}", f"B  {n['B']}") if self.mode == "duel" else (n["A"], n["B"])

    def who(self, w):
        return self.names().get(seat_of(w), str(w)) if w is not None else ""

    def summary(self, i, r=None):
        """One shot in a line, e.g. "Potted 5 in the top-right corner as called · stays at the table"."""
        r = r or self.recs_b()[i]
        pot = [b for b in (r.get("potted") or []) if b != "cue"]
        call = r.get("call")
        if r.get("break"):
            s = "Break: " + ("potted " + ", ".join(pot) if pot else "nothing potted")
        elif call == "safety":
            s = "Safety" + (" (potted " + ", ".join(pot) + ")" if pot else "")
        elif call and ":" in call:
            b, pk = call.split(":", 1)
            where = POCKET_EN.get(pk, pk)
            if r.get("call_made"):
                s = f"Potted {b} in the {where} as called"
                rest = [x for x in pot if x != b]
                s += f" (also {', '.join(rest)})" if rest else ""
            else:
                s = f"Missed the call {b} in the {where}" + (f" (potted {', '.join(pot)})" if pot else "")
        else:
            s = "Potted " + ", ".join(pot) if pot else "No ball potted"
        if r.get("foul"):
            s = f"Foul: {r['foul']}" + (f" (potted {', '.join(pot)})" if pot and not r.get('break') else "")
            if r.get("break") and pot:
                s += f" (potted {', '.join(pot)})"
        if r.get("eight_respotted"):
            s += " · the 8 is spotted again"
        if r.get("rerack"):
            s += " · re-rack"
        if r.get("game_over"):
            s += f" · {r.get('result')}"
        elif r.get("keeps_turn"):
            s += " · stays at the table"
        elif r.get("foul"):
            s += " · ball in hand for the opponent"
        else:
            s += " · turn passes"
        return s

    def result_text(self, p):
        return f"Shot {p['n']}: {p['how']}"

    def phase_at(self, k=None):
        if self.end_ev is not None or (self.recs_b() and self.state(len(self.recs_b()) - 1)["done"]):
            return "end"
        return "run" if (self.begun or self.shots) else "wait"

    def score_at(self, k=None):
        return self.left(self.state(len(self.recs_b()) - 1))

    def info_json(self):
        a, b = self.labels()
        pr = self.program
        st = self.state(len(self.recs_b()) - 1)
        return dict(id=self.id, sport=self.sport, mode=self.mode, host=lb.short(self.host), job=self.job,
                    players=[a, b], slowdown=1, has_log=self.hdr is not None, err=self.err, supported=True,
                    frames=(len(self.recs_b()) + 1) if self.cfg else 0, dt=1, k=len(self.recs_b()),
                    phase=self.phase_at(), score=list(self.left(st)), gone=bool(self.gone), unit=self.unit,
                    turn_based=True, sport_name=SPORT_CN["billiards"], turn=None if st["done"] else st["turn"],
                    groups=st["group"], winner=st["winner"],
                    points=[dict(n=p["n"], winner=self.who(p["winner"]), how=p["how"], score=p["score"]) for p in self.points],
                    viewers=pr.clients if pr else 0, playback=pr.playback_info() if pr else None,
                    program=f"/program/{self.id}")

    # ---- the MP4 of one shot (overhead, real speed), for the page's replay list
    def replay(self, n):
        with self.replay_lock:
            if n in self.replays and self.replays[n].exists():
                return self.replays[n]
            if not (1 <= n <= len(self.recs_b())):
                return None
            bill_call(BILL_CLIP, self, ("sync", self.id, self.cfg, None))
            if n > len(self.binfo):
                return None
            i, info = n - 1, self.binfo[n - 1]
            fss, ovs = [], []
            prog = BilliardsProgram(self)
            for f in range(int(REPLAY_FPS * 1.0)):
                fss.append(dict(shot=i, t=-1, cam={"kind": "top"}, marks=prog.marks(i)))
                ovs.append(prog.overlay_for(i - 1, "REPLAY", "Real speed 1×", call=prog.call_chip(i)))
            t = 0.0
            while t <= info["T"]:
                fss.append(dict(shot=i, t=round(t, 3), cam={"kind": "top"}, marks=prog.marks(i)))
                ovs.append(prog.overlay_for(i - 1, "REPLAY", "Real speed 1×", call=prog.call_chip(i)))
                t += 1.0 / REPLAY_FPS
            for f in range(int(REPLAY_FPS * 1.6)):
                fss.append(dict(after=i, cam={"kind": "top"}))
                ovs.append(prog.overlay_for(i, "REPLAY", "Real speed 1×", banner=prog.banner(i)))
            for o in ovs:
                o["lower"] = []
            mp4, _ = bill_call(BILL_CLIP, self, ("clip", self.id, fss, ovs, SIZE, REPLAY_FPS))
            out = CACHE / "replays" / self.id / f"{n}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(mp4)
            self.replays[n] = out
            return out


class BilliardsCaptions(Captions):
    """Lower thirds of a billiards match: the reviews straight from the episode log (WHY → PLAN) and, as for the other
    sports, the agents' public messages / actions from the live board."""

    def __init__(self, bm):
        super().__init__(bm)
        self.n_rev = 0

    def update(self, now):
        super().update(now)
        revs = list(self.bm.reviews)
        for e in revs[self.n_rev:]:
            why, plan = (e.get("why") or "").strip(), (e.get("plan") or "").strip()
            if why or plan:
                text = redact(f"{why} → {plan}" if why and plan else (why or plan))
                self.review[e.get("seat") or "A"] = dict(text=text, t=float(e.get("ts") or now))
        self.n_rev = len(revs)


class BilliardsProgram(Program):
    """The program feed of a billiards match (see the comment above BilliardsMatch)."""

    def __init__(self, bm):
        super().__init__(bm)
        self.caps = BilliardsCaptions(bm)
        self.mode = "table"                # table | shot | replay
        self.shown = None                  # shots shown live so far (set when the feed starts: it joins at present)
        self.cur, self.speed, self.aim, self.hold = None, 1.0, B_AIM_S, B_HOLD_S
        self.replay_t0 = 0.0

    def live_step(self, now):
        self.last_t = now

    def note(self, text):
        self.log.append((time.time(), text))

    # ---- the director
    def n_ready(self):
        """Shots that can be shown: replayed by the helper and (duel) already in the sim's log as a shot."""
        bm = self.bm
        return len(bm.binfo) if bm.mode != "duel" else min(len(bm.binfo), len(bm.shots))

    def start_shot(self, now):
        n = self.n_ready()
        if n - self.shown > B_BACKLOG_SKIP:                 # far behind: skip to the latest shots
            self.note(f"skipped shots {self.shown + 1}-{n - 3} (far behind)")
            self.shown = n - 3
        self.cur = self.shown
        self.speed = 2.0 if n - self.shown > B_BACKLOG_FAST else 1.0
        self.aim, self.hold = B_AIM_S / self.speed, B_HOLD_S / self.speed
        self.mode, self.shot_t0 = "shot", now
        self.note(f"live: shot {self.cur + 1} by {self.bm.who(self.bm.binfo[self.cur]['player'])}"
                  + (" (2x, catching up)" if self.speed != 1 else ""))

    def package(self, i, recap=False):
        """The instant replay of shot i: behind the cue ball along the cue line (0.5x), then at the pocket (0.5x)."""
        info = self.bm.binfo[i]
        T = float(info["T"])
        segs = [dict(cam={"kind": "cue"}, t0=-0.4, t1=min(T, B_REPLAY_MAX_T), speed=0.5, label="Cue line · 0.5×",
                     hold=0.5, shot=i)]
        if not recap and info.get("pots"):
            call = (info["rec"].get("call") or "")
            called = call.split(":")[0] if ":" in call else None
            pots = [p for p in info["pots"] if p[1] in POCKET_EN]
            pick = next((p for p in pots if p[0] == called), None) or next((p for p in pots if p[0] != "cue"), None) \
                or (pots[0] if pots else None)
            if pick:
                b, pk, tp = pick
                segs.append(dict(cam={"kind": "pocket", "pocket": pk, "ball": b, "t_pot": tp}, t0=max(0.0, tp - 1.4),
                                 t1=min(T, tp + 0.5), speed=0.5, shot=i, hold=0.8,
                                 label=f"{'Scratch' if b == 'cue' else 'Pocket'} cam · {POCKET_EN[pk]} · 0.5×"))
        return [s for s in segs if s["t1"] > s["t0"]]

    def direct(self, now):
        bm = self.bm
        n = self.n_ready()
        if self.shown is None:                                # join at present (the single player task before
            self.shown = 0 if bm.mode != "duel" and not bm.begun else n   # its start: show the opponent's break)
            self.live_since = now
        if self.stinger:
            sp = (now - self.stinger["t0"]) / STINGER_S
            if sp >= 0.5 and not self.stinger.get("cut"):
                self.stinger["cut"] = True
                to = self.stinger["to"]
                if to == "replay" and n > self.shown:          # a new shot came in during the wipe: live instead
                    self.start_shot(now)
                    self.note("new shot during the wipe: live instead of the replay")
                elif to == "replay":
                    self.mode, self.pkg, self.shot_i, self.shot_t0 = "replay", self.stinger["pkg"], 0, now
                    self.pkg_point, self.replay_t0 = self.stinger["point"], now
                elif to == "shot" and n > self.shown:
                    self.start_shot(now)
                else:
                    self.mode, self.live_since = "table", now
            if sp >= 1.0:
                self.stinger = None
            return
        if self.mode == "shot":
            info = bm.binfo[self.cur]
            if now - self.shot_t0 < self.aim + info["T"] / self.speed + self.hold:
                return
            self.shown = self.cur + 1
            if n > self.shown:                                # the next shot is already in: straight on, live
                self.start_shot(now)
                return
            pkg = self.package(self.cur) if self.speed == 1.0 else []
            if pkg:
                self.stinger = dict(t0=now, to="replay", pkg=pkg, point=dict(n=self.cur + 1, i=self.cur))
                self.note(f"cut to replay of shot {self.cur + 1} (" + ", ".join(s["label"] for s in pkg) + ")")
            else:
                self.mode, self.live_since = "table", now
            return
        if self.mode == "replay":
            if n > self.shown and now - self.replay_t0 >= B_REPLAY_MIN_S:
                self.stinger = dict(t0=now, to="shot")
                self.note("new shot: back to live")
                return
            seg = self.pkg[self.shot_i]
            if now - self.shot_t0 >= (seg["t1"] - seg["t0"]) / seg["speed"] + seg["hold"]:
                self.shot_i += 1
                self.shot_t0 = now
                if self.shot_i >= len(self.pkg):
                    self.shot_i = len(self.pkg) - 1
                    self.stinger = dict(t0=now, to="table")
                    self.note("replay done: back to the live table")
            return
        # the live table, between shots
        if n > self.shown:
            self.start_shot(now)
            return
        if n and now - self.live_since >= B_IDLE_S:           # a long think: the last shot again (after the match:
            if bm.phase_at() == "end":                        # the shots that potted, in turn)
                cand = [i for i in range(n) if bm.binfo[i].get("pots")] or list(range(n))
                i, recap = cand[self.n_end % len(cand)], False
                self.n_end += 1
            else:
                self.n_loop += 1
                i, recap = n - 1, self.n_loop % 2 == 0
            pkg = self.package(i, recap=recap)
            if pkg:
                self.stinger = dict(t0=now, to="replay", pkg=pkg, point=dict(n=i + 1, i=i))
                self.note(f"replay loop: shot {i + 1} ({'recap' if recap else 'package'})")
            else:
                self.live_since = now

    # ---- what is drawn
    def call_chip(self, i):
        r = self.bm.binfo[i]["rec"]
        who = self.bm.who(self.bm.binfo[i]["player"])
        if r.get("break"):
            return ["BREAK", f"{who} breaks"]
        c = r.get("call") or ""
        if c == "safety":
            return ["CALL", "safety"]
        if ":" in c:
            b, pk = c.split(":", 1)
            return ["CALL", f"{b} in the {POCKET_EN.get(pk, pk)}"]
        return None

    def marks(self, i):
        c = self.bm.binfo[i]["rec"].get("call") or ""
        if ":" not in c:
            return []
        b, pk = c.split(":", 1)
        return [dict(ball=b), dict(pocket=pk)]

    def banner(self, i):
        r = self.bm.binfo[i]
        return f"Shot {i + 1} · {self.bm.who(r['player'])} · {self.bm.summary(i)}"

    def seats(self):
        nm = self.bm.names()
        return [("A", nm["A"]), ("B", nm["B"])] if self.bm.mode == "duel" else [("A", nm["A"])]

    def overlay_for(self, n, tag, speed, **kw):
        """The program graphics with the table after shot n: score bug (balls left, whose turn), groups and balls."""
        bm = self.bm
        st = bm.state(n)
        nm = bm.names()
        la, lb_ = bm.left(st)
        rows = [["A", nm["A"], la], ["B" if bm.mode == "duel" else "", nm["B"], lb_]]
        if st["group"]["A"] is None:
            rack = [dict(seat="", label="Open table", balls=[b for b in ALL_BALLS if b in st["on"] and b != "8"])]
        else:
            rack = [dict(seat=s if bm.mode == "duel" or s == "A" else "",
                         label=("" if bm.mode == "duel" or s == "A" else "Program · ")
                         + ("On the 8" if bm.group_balls(st, s) == ["8"] else st["group"][s].capitalize()
                            + ("" if bm.group_balls(st, s) else " · cleared")),
                         balls=bm.group_balls(st, s)) for s in ("A", "B")]
        if st["done"]:
            phase = f"Match over · {nm.get(st['winner'], 'draw')} wins" if st["winner"] else "Match over"
        elif not (bm.begun or bm.shots):
            phase = "Waiting for the players"
        else:
            cl = bm.clock_left.get(st["turn"]) if bm.mode == "duel" else bm.clock_left.get("A")
            phase = f"{nm[st['turn']]} to shoot" + (" · ball in hand" if st["bih"] else "") + \
                (f" · clock {int(cl // 60)}:{int(cl % 60):02d}" if cl is not None and (bm.mode == "duel" or st["turn"] == "A") else "")
        ov = dict(program=True, bug=dict(sport=SPORT_CN["billiards"] + (" · Duel" if bm.mode == "duel" else ""),
                                         unit=(f"Shot {st['shot']}" if st["shot"] else "Rack") + " · balls left", rows=rows,
                                         turn=None if st["done"] else (st["turn"] if bm.mode == "duel" or st["turn"] == "A" else "")),
                  rack=rack, phase=phase, tag=tag, speed=speed, lower=self.caps.lower(time.time(), self.seats()))
        ov.update({k: v for k, v in kw.items() if v is not None})
        return ov

    def compose(self, now):
        bm = self.bm
        top = {"kind": "top"}
        if self.mode == "shot" and self.cur is not None:
            i, info = self.cur, bm.binfo[self.cur]
            e = now - self.shot_t0
            spd = "Real speed 1×" if self.speed == 1 else "Catching up 2×"
            if e < self.aim:
                fs = dict(shot=i, t=-1, cam=top, marks=self.marks(i))
                ov = self.overlay_for(i - 1, "LIVE", spd, call=self.call_chip(i))
            elif e < self.aim + info["T"] / self.speed:
                fs = dict(shot=i, t=round((e - self.aim) * self.speed, 2), cam=top, marks=self.marks(i))
                ov = self.overlay_for(i - 1, "LIVE", spd, call=self.call_chip(i))
            else:
                fs = dict(after=i, cam=top)
                ov = self.overlay_for(i, "LIVE", spd, banner=self.banner(i))
            self.shot_label = f"Live · shot {i + 1}"
        elif self.mode == "replay" and self.pkg:
            seg = self.pkg[self.shot_i]
            i = seg["shot"]
            t = min(seg["t1"], seg["t0"] + (now - self.shot_t0) * seg["speed"])
            fs = dict(shot=i, t=round(t, 3) if t >= 0 else -1, cam=seg["cam"], marks=self.marks(i))
            ov = self.overlay_for(i, "REPLAY", f"Slow motion {seg['speed']:g}×", shot=seg["label"],
                                  banner=self.banner(i), call=self.call_chip(i))
            self.shot_label = seg["label"]
        else:
            n = (self.shown or 0) - 1
            fs = dict(after=n, cam=top)
            st = bm.state(n)
            chip = None
            if st["bih"] and not st["done"] and n >= 0:
                chip = ["BALL IN HAND", bm.names()[st["turn"]] + (" · behind the head string" if st["bih"] == "kitchen" else
                                                                  " · anywhere")]
            ov = self.overlay_for(n, "LIVE", "Turn-based · between shots", banner=self.banner(n) if n >= 0 else None,
                                  call=chip)
            self.shot_label = "Live · table"
        if self.stinger:
            sp = min(1.0, (now - self.stinger["t0"]) / STINGER_S)
            ov["stinger"] = round(sp, 3)
            if abs(sp - 0.5) < 0.04:
                ov["blank"] = True
        key = (self.mode, json.dumps(fs, sort_keys=True, default=str), json.dumps(ov, sort_keys=True, default=str))
        return fs, ov, key

    def tick(self):
        bm = self.bm
        bill_call(BILL, bm, ("sync", bm.id, bm.cfg, None))      # replays only the decisions it has not seen
        now = self.clock()                                      # (after it: the first shot compiles pooltool)
        if not hasattr(self, "last_t"):
            self.last_t = now
        self.direct(now)
        fs, ov, key = self.compose(now)
        if key == self.last_key:
            return False
        jpg, rt = bill_call(BILL, bm, ("frame", bm.id, fs, ov, SIZE))
        BILL.rt = 0.8 * BILL.rt + 0.2 * rt
        self.jpeg, self.last_key = jpg, key
        self.frames += 1
        self.frame_times.append(time.monotonic())
        self.display_phase = "replay" if self.mode == "replay" else bm.phase_at()
        return True

    def playback_info(self):
        info = super().playback_info()
        info.update(shown=self.shown, live_shot=(self.cur + 1) if self.mode == "shot" and self.cur is not None else None)
        return info

    def target_fps(self):
        return target_fps(BILL)


# ---------------------------------------------------------------- real-time: soccer (RoboOlympics 2 v 2)
# The soccer sims (scripts/olympiad/shared/soccer/sim/server.py) append the same kind of broadcast log: a frame every
# 5 control steps (20 ms each) with the full qpos (ball + four Booster T1 robots), the half's clock, the period (1, 2,
# "pso") and the score, plus the referee's events (ref_goal, ref_penalty, ref_fall, ref_pso_kick, ref_stoppage,
# ref_end) and run / pause / end. Every stoppage (goal, ball out, half time, a shoot-out kick...) freezes the world
# for a review, so a stoppage is the soccer "point": the director cuts to its package there (main camera at real
# speed, behind the scorer or a reverse angle at 0.5x, overhead at 0.5x). The scene is built with the task's own
# soccer_scene.build_model() in a helper from the task's image. Captions come from the sim's /data (read-only, as
# the log): the team channel ("say" in episode.jsonl, public to spectators, never shown to the opponent), the seats'
# ready reviews (WHY -> PLAN), and each seat's latest model output (think_calls_<seat>.jsonl: tool calls and text,
# never the reasoning). Test offline: test_broadcast_soccer.py.
SOC_HALF_L = 5.0                     # soccer_scene.py: field 10 m x 7 m, team A attacks +x
SOC_LOOK_X, SOC_LOOK_Y, SOC_EMA = 2.4, 0.9, 0.22
SOC_REPLAY_S = 14.0                  # the per-stoppage MP4 (/replay) covers at most this much game time
SOC_RGB = {"A1": (40, 90, 245), "A2": (30, 200, 225), "B1": (230, 40, 40), "B2": (150, 60, 215),
           "A": (40, 90, 245), "B": (230, 40, 40)}
SOC_KIND = {"goal": "Goal", "no_goal": "No goal", "ball_out": "Ball out", "half_time": "Half time",
            "full_time": "Full time", "global_game_stuck": "Dropped ball (play stuck)", "pso_kick": "Shoot-out kick"}
SOC_SCRIPT = r"""
cd /data 2>/dev/null || exit 0
off=$1
s=$(stat -c %s episode.jsonl 2>/dev/null || echo 0)
[ "$s" -lt "$off" ] && off=0
echo "@@EP $off $s"
[ "$s" -gt "$off" ] && tail -c +$((off+1)) episode.jsonl | head -c $((s-off)) | grep -E '"ev": "(say|ready)"' | cut -c1-3000
for f in think_calls*.jsonl; do [ -f "$f" ] || continue; echo "@@TC $f"; tail -n 1 "$f" | cut -c1-16000; echo; done
exit 0
"""
SPORT_CN["soccer"] = "Soccer"


def soc_period(p, clock=None):
    if p == "pso":
        return "Shoot-out"
    c = "" if clock is None else f" {int(clock // 60)}:{int(clock % 60):02d}"
    return ({1: "1st half", 2: "2nd half"}.get(p, str(p))) + c


class SoccerMatch(BMatch):
    """A soccer match (see the comment above)."""

    def __init__(self, d):
        super().__init__(d)
        self.mode = "team" if self.info.get("team") else self.mode
        self.goals, self.pso, self.flags = [], [], []      # ref_goal, ref_pso_kick, ref_penalty / ref_fall events
        self._pending = {}                                  # the goal / kick of the stoppage about to be logged

    def live_helper(self):
        return SOC

    def clip_helper(self):
        return SOC_CLIP

    @property
    def program_cls(self):
        return SoccerProgram

    def reset(self):
        super().reset()
        self.goals, self.pso, self.flags, self._pending = [], [], [], {}

    def ingest(self, ln):
        try:
            e = json.loads(ln)
        except ValueError:
            return
        if "q" in e and "ev" not in e:
            k = int(e["k"])
            if self.ks and k <= self.ks[-1] or len(e["q"]) < 3:
                return
            b = e["q"]
            tgt = [max(-SOC_LOOK_X, min(SOC_LOOK_X, b[0])), max(-SOC_LOOK_Y, min(SOC_LOOK_Y, 0.7 * b[1])), 0.0]
            prev = self.recs[-1].get("look") if self.recs else None
            jump = prev is None or abs(tgt[0] - prev[0]) + abs(tgt[1] - prev[1]) > 2.5 or k - self.ks[-1] > 20
            e["look"] = tgt if jump else [round(p + SOC_EMA * (t - p), 4) for p, t in zip(prev, tgt)]
            self.recs.append(e)
            self.ks.append(k)
            return
        ev = e.get("ev")
        if ev == "hdr":
            self.hdr = e
            self.mode = e.get("mode", self.mode)
            return
        self.events.append(e)
        k = int(e.get("k", 0))
        if ev == "ref_goal":
            self.goals.append(e)
            self._pending = dict(goal=e)
        elif ev == "ref_pso_kick":
            self.pso.append(e)
            self._pending = dict(kick=e)
        elif ev in ("ref_penalty", "ref_fall"):
            self.flags.append(e)
        elif ev in ("ref_stoppage", "ref_end"):
            if self.points and self.points[-1]["k_ev"] == k:   # ref_end after the deciding kick's stoppage: one entry
                self.points[-1].update(final=True, how=self.points[-1]["how"] + " · " + self.end_text(e))
                return
            k0 = self.points[-1]["k"] if self.points else (self.ks[0] - 1 if self.ks else -1)
            k0 = max(k0, k - int(SOC_REPLAY_S / self.dt))
            pend, self._pending = self._pending, {}
            kind = e.get("kind") or ("goal" if pend.get("goal") else "pso_kick" if pend.get("kick") else "end")
            g, kick = pend.get("goal"), pend.get("kick") or e.get("kick")
            winner = (g or {}).get("team") or (e.get("team") if kind == "goal" else None)
            if kick and kick.get("result") == "goal":
                winner = kick.get("team")
            # the stoppage's last frame is the step before it (the sim logs the moment play stopped)
            kf = min(k, self.ks[-1]) if self.ks and self.ks[-1] > k0 else k
            p = dict(n=len(self.points) + 1, k0=k0, k=kf, k_ev=k, score=e.get("score"), winner=winner, kind=kind,
                     goal=g, kick=kick, period=e.get("period"), final=ev == "ref_end", note=None, flight=None)
            p["how"] = self.stop_text(p, e)
            if ev == "ref_end":
                p["how"] += " · " + self.end_text(e)
            self.points.append(p)

    # ---- texts
    def team_name(self, t):
        pl = self.info["players"]
        if self.mode == "single" and t == "B":
            return f"Program L{(self.hdr or {}).get('level', self.info.get('level', '?'))}"
        n = nice_name(pl.get(t, t)) if self.mode != "single" or t == "A" else t
        return f"{n} ×2" if self.mode == "team" else n

    def stop_text(self, p, e):
        kind, g, kick = p["kind"], p.get("goal"), p.get("kick")
        if kind == "goal" or g:
            g = g or e
            who = g.get("robot")
            if g.get("own_goal"):
                return f"Goal for {g.get('team')} · own goal by {who}"
            return f"Goal · {who or g.get('team')} scores for {g.get('team')}"
        if kind == "pso_kick" and kick:
            res = {"goal": "scores", "saved": f"saved by {kick.get('keeper')}", "post": "hits the post",
                   "wide": "wide", "other": "no goal"}.get(kick.get("result"), kick.get("result"))
            return f"Shoot-out kick {kick.get('n')} · {kick.get('striker')} {res}"
        if kind == "no_goal":
            return f"No goal for {e.get('team')} · kick-off not played outside the centre circle"
        return SOC_KIND.get(kind, str(kind).replace("_", " ").capitalize())

    def end_text(self, e):
        w, how = e.get("winner"), e.get("how") or ""
        sc = e.get("score") or {}
        tail = {"pso_coin_toss": " (shoot-out level, coin toss)", "mercy": " (mercy rule)"}.get(how, "")
        return f"Full time {sc.get('A', 0)}–{sc.get('B', 0)} · {w} wins{tail}" if w else "Full time"

    def result_text(self, p):
        return f"{p['n']}. {p['how']}"

    def labels(self):
        return f"A  {self.team_name('A')}", f"B  {self.team_name('B')}"

    def rec_at(self, k):
        i = bisect.bisect_right(self.ks, k) - 1
        return self.recs[max(0, i)] if self.recs else {}

    def score_at(self, k):
        sc = self.rec_at(k).get("score") or {}
        a, b = sc.get("A", 0), sc.get("B", 0)
        for p in self.points:                               # a goal counts from its stoppage on (scores only grow)
            if p["k"] <= k and isinstance(p.get("score"), dict):
                a, b = max(a, p["score"].get("A", 0)), max(b, p["score"].get("B", 0))
        return a, b

    def pso_at(self, k):
        out = {"A": 0, "B": 0}
        for x in self.pso:
            if x.get("k", 0) <= k + 1 and x.get("result") == "goal":
                out[x.get("team")] = out.get(x.get("team"), 0) + 1
        return out, sum(1 for x in self.pso if x.get("k", 0) <= k + 1)

    def clock_text(self, k):
        r = self.rec_at(k)
        per = r.get("period", 1)
        if per == "pso":
            ps, n = self.pso_at(k)
            return f"Shoot-out · pens {ps['A']}–{ps['B']}" + (f" · kick {n + 1}" if self.phase_at(self.ks[-1]) != "end" else "")
        return soc_period(per, float(r.get("clock") or 0.0))

    def flag_at(self, k, window_s=4.0):
        """A penalty (removal) shortly before game step k: the live banner."""
        e = next((x for x in reversed(self.flags) if x["ev"] == "ref_penalty" and 0 <= k - x["k"] <= window_s / self.dt), None)
        if e is None:
            return None
        why = f" ({e['why']})" if e.get("why") else ""
        return f"Penalty · {e.get('robot')} removed for {e.get('seconds', 0):g} s{why}"

    def phase_at(self, k):
        """As BMatch, but the sim's last frame of a stoppage is the step before its pause / end event."""
        return super().phase_at(k + 1)

    def live_overlay(self, k, at_end):
        ov = super().live_overlay(k, at_end)
        ov["clock"] = self.clock_text(k)
        return ov

    def info_json(self):
        out = super().info_json()
        out.update(mode=self.mode, players=list(self.labels()), clock=self.clock_text(self.ks[-1]) if self.ks else None,
                   points=[dict(n=p["n"], winner=(self.who(p["winner"]) if p["winner"] else ""), how=p["how"],
                                score=p["score"]) for p in self.points])
        return out


def soc_build_package(bm, p, recap=False):
    """The package of stoppage p: main camera at real speed, then behind the scorer (a goal / shoot-out kick) or a
    reverse angle from the far touchline at 0.5x, then overhead at 0.5x (half time / full time: the main camera only)."""
    lo = bisect.bisect_right(bm.ks, p["k0"])
    hi = bisect.bisect_right(bm.ks, p["k"])
    if hi - lo < 2:
        return []
    dt, kend, kfirst = bm.dt, bm.ks[hi - 1], bm.ks[lo]
    sec = lambda x: int(round(x / dt))
    clip = lambda a, b: (max(kfirst, a), min(kend, b))
    ball = [float(x) for x in bm.recs[hi - 1]["q"][0:3]]
    shots = []
    if not recap:
        a, b = clip(kend - sec(6.0), kend)
        shots.append(dict(cam="main", k0=a, k1=b, speed=1.0, label="Main camera · real speed", hold=0.5))
    if p["kind"] in ("half_time", "full_time") and not p.get("goal"):
        return shots or [dict(cam="main", k0=max(kfirst, kend - sec(4.0)), k1=kend, speed=1.0,
                              label="Main camera · real speed", hold=0.5)]
    if p["kind"] in ("goal", "pso_kick", "no_goal") or p.get("goal"):
        team = (p.get("goal") or {}).get("team") or (p.get("kick") or {}).get("team") or p.get("winner")
        sx = (1.0 if team == "A" else -1.0) if team in ("A", "B") else (1.0 if ball[0] >= 0 else -1.0)
        if p["kind"] == "pso_kick" or (p.get("kick") and not p.get("goal")):
            sx = 1.0 if ball[0] >= 0 else -1.0
        look = [sx * (SOC_HALF_L - 1.3), 0.0, 0.35]
        cam1 = (look, 6.4, 0.0 if sx > 0 else 180.0, -26.0)
        lab1 = "Behind the scorer · 0.5×"
    else:
        look = [max(-SOC_HALF_L, min(SOC_HALF_L, ball[0])), max(-3.0, min(3.0, ball[1])), 0.3]
        cam1 = (look, 5.2, 270.0, -24.0)
        lab1 = "Reverse angle · 0.5×"
    a, b = clip(kend - sec(3.0), kend + sec(0.4))
    shots.append(dict(cam=cam1, k0=a, k1=b, speed=0.5, label=lab1, hold=0.3))
    if not recap:
        a, b = clip(kend - sec(2.6), kend)
        shots.append(dict(cam="high", k0=a, k1=b, speed=0.5, label="Overhead · 0.5×", hold=1.2))
    return [s for s in shots if s["k1"] > s["k0"]]


def soc_caption(excerpt):
    """A seat's newest model output (the think proxy's excerpt: tool calls and text, no reasoning) as (kind, text)."""
    x = str(excerpt or "")
    head = x.split("[tool ")[0].split("[shell]")[0].strip()
    tools = re.findall(r"\[tool ([^\]]+)\]\s*(.*?)(?=\s\[tool |\s\[shell\]|$)", x, re.S)
    if tools:
        name, arg = tools[-1]
        m = re.search(r'"?(?:cmd|command)"?\s*:\s*"((?:[^"\\]|\\.)*)"', arg)
        if m:
            try:
                c = json.loads('"' + m.group(1) + '"')
            except ValueError:
                c = m.group(1)
            if re.match(r"\s*game\s+say\b", c):
                return None                                 # shown as the team message itself
            c = audience.command_activity(c)
            if not c:    # a plain wait: keep the previous caption
                return None
            return "Action", c
        name = name.strip()
        if name in ("Edit", "Write", "MultiEdit", "apply_patch"):
            m = re.search(r'"file_path"\s*:\s*"([^"]+)"', arg)
            return "Action", "Updating the control program"
        if name in ("Read", "view_image"):
            m = re.search(r'"(?:file_)?path"\s*:\s*"([^"]+)"', arg)
            return "Action", "Inspecting a saved image" if name == "view_image" else "Reading the control program"
        if name not in ("TodoWrite",):
            return None
    if head:
        return "Says", head
    return None


class SoccerCaptions(Captions):
    """Lower thirds of a soccer match, read from the sim's /data like the broadcast log (see the soccer comment):
    team-channel messages ("Team", sender -> mate), ready reviews, the seats' latest model output. Per seat the newest
    item; the feed shows the two most recent / relevant seats."""
    RANK = {"Team": 20.0, "Review": 25.0, "Says": 10.0, "Action": 0.0}
    SCRIPT = SOC_SCRIPT

    def __init__(self, bm):
        super().__init__(bm)
        self.ep_off = 0
        self.tc_seen = {}

    def update(self, now):
        bm = self.bm
        if bm.gone:
            return
        out = lb.dexec(bm.host, bm.sim, self.SCRIPT, self.ep_off, timeout=20)
        if not out:
            return
        txt = out.decode("utf-8", "replace")
        m = re.search(r"@@EP (\d+) (\d+)\n", txt)
        if not m:
            return
        self.ep_off = int(m.group(2))
        body, _, rest = txt[m.end():].partition("@@TC ")
        for ln in body.splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            seat = str(e.get("seat") or "A")
            t = float(e.get("ts") or e.get("wall") or now)
            ev = e.get("ev") or e.get("event")                 # soccer "ev"; racing / air hockey "event"
            if ev == "say" and e.get("text"):
                text = re.sub(rf"^\s*{re.escape(seat)}\s*:\s*", "", str(e["text"]))
                self.seat[seat] = dict(kind="Team", text=redact(text), t=t, to=e.get("to"))
            elif ev in ("ready", "review") and (e.get("why") or e.get("plan")):
                why, plan = str(e.get("why") or ""), str(e.get("plan") or "")
                self.review[seat] = dict(text=redact(f"{why} → {plan}" if why and plan else (why or plan)), t=t)
        for chunk in ("@@TC " + rest).split("@@TC ")[1:]:
            fname, _, line = chunk.partition("\n")
            mm = re.match(r"think_calls_?([A-Za-z0-9]*)\.jsonl", fname.strip())
            if not mm:
                continue
            seat = mm.group(1) or "A"
            try:
                e = json.loads(line.strip().splitlines()[0]) if line.strip() else None
            except (ValueError, IndexError):
                continue
            if not e or e.get("seq") == self.tc_seen.get(seat):
                continue
            self.tc_seen[seat] = e.get("seq")
            cap = soc_caption(e.get("excerpt"))
            if cap is None:
                continue
            t = float(e.get("upstream_terminal_at") or e.get("t1") or e.get("t0") or now)
            old = self.seat.get(seat)
            if old and old["kind"] == "Team" and t - old["t"] < 12.0:
                continue                                    # a fresh team message stays up a little longer
            self.seat[seat] = dict(kind=cap[0], text=redact(cap[1]), t=t)

    def lower(self, now, seats):
        items = []
        for s, name in seats:
            rv, it = self.review.get(s), self.seat.get(s)
            if rv and now - rv["t"] < REVIEW_HOLD_S and (not it or it["t"] <= rv["t"] + 10):
                kind, text, t, to = "Review", rv["text"], rv["t"], None
            elif it:
                kind, text, t, to = it["kind"], it["text"], it["t"], it.get("to")
            else:
                continue
            age = max(0, int(now - t))
            items.append((t + self.RANK.get(kind, 0.0), s, dict(
                seat=s, name=name, kind=f"Team → {to}" if kind == "Team" and to else kind, text=text,
                age=f"{age} s ago" if age < 120 else f"{age // 60} min ago",
                summary=rv.get("summary") if kind == "Review" and rv else None)))
        top = sorted(items, key=lambda x: -x[0])[:2]
        order = [s for s, _ in seats]
        return [c for _, s, c in sorted(top, key=lambda x: order.index(x[1]))]


class SoccerProgram(Program):
    """The program feed of a soccer match: Program's director with soccer packages, graphics and captions."""

    def __init__(self, bm):
        super().__init__(bm)
        self.caps = SoccerCaptions(bm)

    def package(self, p, recap=False):
        return soc_build_package(self.bm, p, recap=recap)

    def seats(self):
        bm = self.bm
        pl = bm.info["players"]
        nm = lambda t: nice_name(pl.get(t, t))
        if bm.mode == "team":
            return [("A1", nm("A")), ("A2", nm("A")), ("B1", nm("B")), ("B2", nm("B"))]
        if bm.mode == "duel":
            return [("A", nm("A")), ("B", nm("B"))]
        return [("A", nm("A"))]

    def base_overlay(self, now, k_score):
        bm = self.bm
        sa, sb = bm.score_at(k_score)
        rows = [["A", "A · " + bm.team_name("A"), sa], ["B", "B · " + bm.team_name("B"), sb]]
        ph = bm.phase_at(bm.ks[-1])
        mode = {"team": "Team (4 agents)", "duel": "Duel", "single": "Single"}.get(bm.mode, bm.mode)
        phase = {"pause": "Stoppage · world frozen for the review", "end": "Full time",
                 "wait": "Waiting for the players", "start": "Waiting for the players"}.get(ph, "")
        return dict(program=True, bug=dict(sport=f"Soccer · {mode}", unit=bm.clock_text(k_score), rows=rows),
                    phase=phase, lower=self.caps.lower(time.time(), self.seats()), seat_rgb=SOC_RGB)

    def result_cn(self, p):
        return p["how"]

    def compose(self, now):
        ra, rb, alpha, ov, key = super().compose(now)
        if self.mode == "live" and not ov.get("banner") and self.display_step is not None:
            fl = self.bm.flag_at(int(self.display_step))
            if fl:
                ov["banner"] = fl
                key = key[:3] + (key[3] + fl,)
        if ov.get("halo"):
            ov["halo"] = 0.13
        return ra, rb, alpha, ov, key

    def target_fps(self):
        return target_fps(SOC)


# ---------------------------------------------------------------- real-time: 3D racing and air hockey
# Both sims (scripts/olympiad/shared/{racing,airhockey}/sim/server.py) append the usual broadcast log, on the standard
# eai-olympiad image (the LIVE / CLIP helpers draw them):
#   * RACING: header with the track (straight_half_m, radius_m, gates), seed, qpos_layout; a frame (full qpos, heat)
#     every 4 control steps and at every referee event; events start, run, checkpoint, contact, crash (at fault, rule,
#     warning), disqualified, heat_end (public result), pause, review, end. The scene is race.scene_xml(seed): the seed
#     is a scene option of the helper. The main camera chases the leading car (both cars while they are close; the host
#     computes it per frame from the track geometry and the checkpoints, smoothed). The director: a crash is an instant
#     replay WHILE the race goes on (the world is slowed, the replay at real speed costs little game time): main 1x,
#     trackside 0.5x, back to live; at a heat end main 1x, trackside at the finish line 0.5x, overhead 0.5x (played in
#     full even when the next heat starts at once; both only while the world runs slowed >= x4, else as Program);
#     heat banner (winner, how decided, times), crash / disqualification banners, checkpoints and warnings in the bug.
#   * AIR HOCKEY: a frame (puck x, y, yaw, the two arms' 9 joints each) every 2 control steps; events run, point (a
#     goal), stoppage (seven-second foul, dead puck, time cap), pause, end. The scene is hockey.scene_xml() from the
#     sim's own directory. Every rally end freezes the world: package main 1x, goal-line camera (a goal) or puck
#     close-up (a stoppage) 0.5x, overhead 0.5x; banners "Goal for ...", fouls (-> the puck to the other player),
#     face-offs; the bug has goals, fouls, the play clock and the seven-second count.
# Captions come from the sim's /data like soccer's (the seats' reviews in episode.jsonl, their latest model output in
# think_calls*.jsonl). Offline tests: test_broadcast_racing.py, test_broadcast_airhockey.py; replay as live:
# program_harness.py --trial <finished racing / air hockey trial> (broadcast/regen_rt.py rebuilds the log).
RACING, HOCKEY = {"racing"}, {"airhockey"}
SUPPORTED |= {(sp, m) for sp in ("racing", "airhockey") for m in ("single", "duel")}
UNIT.update(racing="heat", airhockey="rally")
SPORT_CN.update(racing="Racing", airhockey="Air hockey")
RT_SCRIPT = SOC_SCRIPT.replace(r'''grep -E '"ev": "(say|ready)"' ''', r'''grep -E '"event": "review"' ''')
assert RT_SCRIPT != SOC_SCRIPT
RACE_GATES, RACE_WARN_DQ = 16, 3
RACE_CLOSE_M = 5.0                   # both cars in the main camera while the gap is below this (m)
RACE_EMA_LOOK, RACE_EMA_AZ = 0.2, 0.14
RACE_LIVE_MIN_SLOW = 4.0             # crash instant replays while the race goes on only when the world is slowed
RACE_BANNER_S = 3.0                  # a crash / heat start banner stays this long (game seconds) on the live picture
RACE_HOW = {"finish": "first across the line", "progress": "further round the lap at the 25 s limit",
            "reached_first": "same progress, reached it first", "lot": "coin flip",
            "disqualification": "disqualification"}
HOCKEY_GOAL_X, HOCKEY_PUCK_R, HOCKEY_CAP_S, HOCKEY_SEVEN_S = 0.974, 0.03165, 120.0, 7.0      # hockey.py
HOCKEY_REPLAY_S = 20.0               # a rally's MP4 (/replay) covers at most this much play


def mmss(t):
    t = max(0, int(t))
    return f"{t // 60}:{t % 60:02d}"


def stadium(a, r):
    """Host copy of race.stadium (the counter-clockwise centre line): length, point(s) -> (xy, tangent), project(xy) ->
    (s, lateral). Used for the race order and the chase camera."""
    import math
    length = 4 * a + 2 * math.pi * r

    def point(s):
        s = s % length
        if s < a:
            return (s, -r), (1.0, 0.0)
        s -= a
        if s < math.pi * r:
            th = -math.pi / 2 + s / r
            return (a + r * math.cos(th), r * math.sin(th)), (-math.sin(th), math.cos(th))
        s -= math.pi * r
        if s < 2 * a:
            return (a - s, r), (-1.0, 0.0)
        s -= 2 * a
        if s < math.pi * r:
            th = math.pi / 2 + s / r
            return (-a + r * math.cos(th), r * math.sin(th)), (-math.sin(th), math.cos(th))
        return (-a + s - math.pi * r, -r), (1.0, 0.0)

    def project(p):
        x, y = float(p[0]), float(p[1])
        if x > a:
            th = math.atan2(y, x - a)
            s, lat = a + r * (th + math.pi / 2), r - math.hypot(x - a, y)
        elif x < -a:
            th = math.atan2(y, x + a)
            if th < 0:
                th += 2 * math.pi
            s, lat = 3 * a + math.pi * r + r * (th - math.pi / 2), r - math.hypot(x + a, y)
        elif y < 0:
            s, lat = x % length, y + r
        else:
            s, lat = a + math.pi * r + (a - x), r - y
        return s % length, lat

    return length, point, project


class RTMatch(BMatch):
    """Common to racing and air hockey: seat names, the cut between non-continuous frames, the point list."""

    def __init__(self, d):
        super().__init__(d)
        self.end_ev = None

    def reset(self):
        super().reset()
        self.end_ev = None

    def ready(self):
        return super().ready() and self.hdr is not None

    def names(self):
        p = self.info["players"]
        if self.mode == "duel":
            return audience.names({s: nice_name(p.get(s, s)) for s in ("A", "B")})
        lv = (self.hdr or {}).get("level", self.info.get("level", "?"))
        return {"A": nice_name(p.get("A", "Player")), "B": f"Program L{lv}"}

    def labels(self):
        n = self.names()
        return n["A"], n["B"]

    def who(self, w):
        return self.names().get(w, "" if w is None else str(w))

    def audience_reason(self, p):
        return p["how"]

    def result_text(self, p):
        return f"{p.get('label') or p['n']}: {p['how']}"

    def rec_at(self, k):
        i = bisect.bisect_right(self.ks, k) - 1
        return self.recs[max(0, i)] if self.recs else {}

    def add_point(self, **p):
        p = dict(dict(n=len(self.points) + 1, note=None, flight=None, final=False, winner=None), **p)
        self.points.append(p)
        return p

    def info_json(self):
        out = super().info_json()
        out.update(points=[dict(n=p["n"], label=p.get("label"), winner=self.who(p["winner"]) if p["winner"] else "",
                                how=p["how"], score=p["score"]) for p in self.points])
        return out


class RacingMatch(RTMatch):
    """A 3D racing match (see the comment above)."""

    def __init__(self, d):
        super().__init__(d)
        self.reset_r()

    def reset(self):
        super().reset()
        self.reset_r()

    def reset_r(self):
        self.geo = None
        self.gates = {}                    # (heat, seat) -> checkpoints passed (as read so far)
        self.cp = []                       # (k, heat, seat, gate)
        self.heat_k0 = {1: 0}
        self.crashes = []
        self.wins = [(0, {"A": 0, "B": 0})]
        self.grid = {}                     # heat -> {seat: inner/outer}
        self._cut_k = None

    @property
    def program_cls(self):
        return RacingProgram

    def scene_opts(self):
        return {"seed": int((self.hdr or {}).get("seed") or self.info.get("seed") or 1)}

    # ---- the log
    def ingest(self, ln):
        try:
            e = json.loads(ln)
        except ValueError:
            return
        if "q" in e and "ev" not in e:
            k = int(e["k"])
            if self.ks and k <= self.ks[-1]:
                return
            prev = self.recs[-1] if self.recs else None
            if prev is not None and (int(prev.get("heat") or 1) != int(e.get("heat") or 1) or k == self._cut_k):
                e["cut"] = True                    # a new heat's grid / a crash placement: no interpolation into it
            self.chase(e, prev)
            self.recs.append(e)
            self.ks.append(k)
            return
        ev = e.get("ev")
        if ev == "hdr":
            self.hdr = e
            self.mode = e.get("mode", self.mode)
            tr = e.get("track") or {}
            if tr.get("straight_half_m"):
                self.geo = stadium(float(tr["straight_half_m"]), float(tr["radius_m"]))
            return
        self.events.append(e)
        k, heat = int(e.get("k", 0)), int(e.get("heat") or 1)
        if isinstance(e.get("heats_won"), dict):
            self.wins.append((k, dict(e["heats_won"])))
        nm = self.names()
        if ev == "run":
            self.heat_k0[heat] = k
            if isinstance(e.get("grid"), dict):
                self.grid[heat] = e["grid"]
        elif ev == "checkpoint":
            s = e.get("seat")
            self.gates[(heat, s)] = max(self.gates.get((heat, s), 0), int(e.get("gate") or 0))
            self.cp.append((k, heat, s, int(e.get("gate") or 0)))
        elif ev == "crash":
            self._cut_k = k
            self.crashes.append(e)
            f = e.get("at_fault")
            o = "B" if f == "A" else "A"
            rule = {"drove_into": f"drove into {nm.get(o, o)}", "trailing": "car behind",
                    "lot": "coin flip"}.get(e.get("fault_rule"), str(e.get("fault_rule") or ""))
            if e.get("kind") == "stuck":
                rule = "cars wedged 2 s · " + rule
            how = (f"Crash · {nm.get(f, f)} at fault ({rule}) · warning {e.get('warning', '?')} of {RACE_WARN_DQ}"
                   f" · placed 2 m behind")
            n_c = sum(1 for p in self.points if p["kind"] == "crash") + 1
            self.add_point(k0=max(self.heat_k0.get(heat, 0), k - int(8.0 / self.dt)), k=k, kind="crash", seat=f,
                           heat=heat, score=dict(self.wins[-1][1]), how=how, label=f"Crash {n_c}", call=e)
        elif ev == "heat_end":
            r = e.get("result") or {}
            w = r.get("winner")
            lo = "B" if w == "A" else "A"
            cars = r.get("cars") or {}
            dec = r.get("decided_by")
            if dec == "disqualification":
                how = f"Heat {r.get('heat', heat)} · {nm.get(lo, lo)} disqualified (third warning) · {nm.get(w, w)} wins the race"
            else:
                how = f"Heat {r.get('heat', heat)} · {nm.get(w, w)} wins — {RACE_HOW.get(dec, dec)}"
                ft = (cars.get(w) or {}).get("finish_time")
                if ft is not None:
                    how += f" · {ft:.2f} s"
                lc = cars.get(lo) or {}
                how += (f" · {nm.get(lo, lo)} {lc['finish_time']:.2f} s" if lc.get("finish_time") is not None
                        else f" · {nm.get(lo, lo)} {lc.get('checkpoints', 0)}/{RACE_GATES} checkpoints")
            self.add_point(k0=self.heat_k0.get(heat, 0), k=k, kind="dq" if dec == "disqualification" else "heat",
                           heat=heat, winner=w, score=dict(e.get("heats_won") or self.wins[-1][1]), how=how,
                           label=f"Heat {r.get('heat', heat)}", result=r)
        elif ev == "end":
            self.end_ev = e
            w = e.get("winner")
            hw = e.get("heats_won") or self.wins[-1][1]
            if w in ("A", "B"):
                tail = f"{nm.get(w, w)} wins the race {hw.get(w, 0)}–{hw.get('B' if w == 'A' else 'A', 0)}"
            else:
                tail = "Race over" + (f" ({e.get('reason')})" if e.get("reason") else "")
            last = self.points[-1] if self.points else None
            if last and last["k"] == k and last["kind"] in ("heat", "dq"):
                if last["kind"] == "heat":
                    last["how"] += " · " + tail
                last["final"] = True

    def chase(self, e, prev):
        """The race order and the main camera of frame e: the cars' progress (checkpoints + distance past the last),
        the leader, the gap; the chase camera behind the leader (both cars while close), smoothed."""
        import math
        if self.geo is None or not self.hdr:
            return
        L, point, project = self.geo
        lay = self.hdr.get("qpos_layout") or {"A": 0, "B": 13}
        q, heat = e["q"], int(e.get("heat") or 1)
        prog, sv = {}, {}
        for s in ("A", "B"):
            a = int(lay.get(s, 0))
            sp, _ = project((q[a], q[a + 1]))
            g = self.gates.get((heat, s), 0)
            base = g * L / RACE_GATES
            if sp - base > L / 2:
                sp -= L
            if sp - base < -L / 2:
                sp += L
            prog[s] = L if g >= RACE_GATES else min(L, max(base, min(sp, base + L / RACE_GATES)))
            sv[s] = sp
        lead = "A" if prog["A"] >= prog["B"] else "B"
        other = "B" if lead == "A" else "A"
        gap = prog[lead] - prog[other]
        ia, ib = int(lay.get("A", 0)), int(lay.get("B", 13))
        pos = {"A": (q[ia], q[ia + 1]), "B": (q[ib], q[ib + 1])}
        if gap < RACE_CLOSE_M:                           # both cars: between them, a little further back
            s_c, off, dist = (sv[lead] + sv[other]) / 2, 0.6, 3.9 + 0.6 * gap
            px, py = (pos["A"][0] + pos["B"][0]) / 2, (pos["A"][1] + pos["B"][1]) / 2
        else:                                            # the leader alone
            (px, py), s_c, off, dist = pos[lead], sv[lead], 0.9, 3.9
        _, (tx, ty) = point(s_c)
        x, y = px + off * tx, py + off * ty              # a little ahead of the car(s), on their actual line
        az = math.degrees(math.atan2(ty, tx))
        if prev is not None and prev.get("look") and not e.get("cut"):
            pl, pa = prev["look"], float(prev["az"])
            d = (az - pa + 180.0) % 360.0 - 180.0
            az = pa + RACE_EMA_AZ * d
            x, y = pl[0] + RACE_EMA_LOOK * (x - pl[0]), pl[1] + RACE_EMA_LOOK * (y - pl[1])
            dist = float(prev["dist"]) + 0.1 * (dist - float(prev["dist"]))
        e["look"] = [round(x, 4), round(y, 4), 0.12]
        e["az"], e["dist"] = round(az, 3), round(dist, 3)
        e["prog"] = {s: round(v, 3) for s, v in prog.items()}
        e["lead"], e["gap"] = lead, round(gap, 3)

    # ---- state at a game step (the bug)
    def heats_won(self, k):
        hw = {"A": 0, "B": 0}
        for kk, w in self.wins:
            if kk <= k:
                hw = w
        return hw

    def score_at(self, k):
        hw = self.heats_won(k)
        return hw.get("A", 0), hw.get("B", 0)

    def cp_at(self, k, heat, seat):
        return max([g for kk, h, s, g in self.cp if kk <= k and h == heat and s == seat] + [0])

    def warnings_at(self, k, seat):
        return sum(1 for c in self.crashes if c.get("k", 0) <= k and c.get("at_fault") == seat)

    def heat_at(self, k):
        return int(self.rec_at(k).get("heat") or 1)

    def heat_clock(self, k):
        h = self.heat_at(k)
        return max(0.0, (k - self.heat_k0.get(h, 0)) * self.dt)


class HockeyMatch(RTMatch):
    """An air hockey match (see the comment above)."""

    def __init__(self, d):
        super().__init__(d)
        self.reset_h()

    def reset(self):
        super().reset()
        self.reset_h()

    def reset_h(self):
        self.runs = {}                     # rally -> k of its start
        self._restart, self._side, self._side_t = False, None, 0.0

    @property
    def program_cls(self):
        return HockeyProgram

    def ingest(self, ln):
        try:
            e = json.loads(ln)
        except ValueError:
            return
        if "q" in e and "ev" not in e:
            k = int(e["k"])
            if self.ks and k <= self.ks[-1] or len(e["q"]) < 3:
                return
            x = float(e["q"][0])
            side = "A" if x < -HOCKEY_PUCK_R else "B" if x > HOCKEY_PUCK_R else None
            if self._restart or not self.ks:
                if self.ks:
                    e["cut"] = True                    # the referee's placement for the new rally: no interpolation
                self._restart, self._side, self._side_t = False, side, 0.0
            elif side is not None and side == self._side:
                self._side_t += (k - self.ks[-1]) * self.dt
            elif side is not None:
                self._side, self._side_t = side, 0.0
            e["side"], e["side_t"] = self._side, round(self._side_t, 3)
            self.recs.append(e)
            self.ks.append(k)
            return
        ev = e.get("ev")
        if ev == "hdr":
            self.hdr = e
            self.mode = e.get("mode", self.mode)
            return
        self.events.append(e)
        k = int(e.get("k", 0))
        nm = self.names()
        sc = e.get("score") or {}
        if ev == "run":
            self.runs[int(e.get("rally") or len(self.points) + 1)] = k
            self._restart = bool(self.points)
        elif ev in ("point", "stoppage"):
            how_k, w, g, nx = e.get("how"), e.get("winner"), e.get("guilty"), e.get("next")
            if how_k == "goal":
                how = f"Goal for {nm.get(w, w)} · {sc.get('A', 0)}–{sc.get('B', 0)}"
            elif how_k == "seven_second_foul":
                how = f"Foul · seven-second rule · {nm.get(g, g)} kept the puck in its half · puck to {nm.get(nx, nx)}"
            elif how_k == "dead_puck":
                how = "Dead puck · face-off at the centre"
            elif how_k == "time_cap":
                how = f"Time cap · {mmss(HOCKEY_CAP_S)} of play"
            else:
                how = str(how_k or "Stoppage").replace("_", " ").capitalize()
            rally = int(e.get("rally") or len(self.points) + 1)
            k0 = self.runs.get(rally, self.points[-1]["k"] if self.points else 0)
            self.add_point(k0=max(k0, k - int(HOCKEY_REPLAY_S / self.dt)), k=k,
                           kind="goal" if how_k == "goal" else str(how_k), winner=w, guilty=g, next=nx,
                           score=dict(sc), how=how, label=f"Rally {rally}", rally=rally)
        elif ev == "end":
            self.end_ev = e
            w, a, b = e.get("winner"), sc.get("A", 0), sc.get("B", 0)
            if e.get("forfeiting") in ("A", "B"):
                tail = f"{nm.get(e['forfeiting'], e['forfeiting'])} forfeits · {nm.get(w, w)} wins"
            elif w not in ("A", "B"):
                tail = "Match over · no result"
            elif a != b:
                tail = f"Full time {a}–{b} · {nm.get(w, w)} wins"
            elif any(p["kind"] == "goal" for p in self.points):
                tail = f"Full time {a}–{b} · {nm.get(w, w)} wins on the last goal (time-cap tie-break)"
            else:
                tail = f"Full time {a}–{b} · {nm.get(w, w)} wins on territory (time-cap tie-break)"
            last = self.points[-1] if self.points else None
            if last and last["k"] == k:
                last["how"] += " · " + tail
                last["final"] = True
            else:
                self.add_point(k0=last["k"] if last else 0, k=k, kind="end", score=dict(sc), how=tail, label="Full time",
                               winner=w, final=True)

    def fouls_at(self, k, seat):
        return sum(1 for p in self.points if p["k"] <= k and p.get("guilty") == seat)

    def rally_at(self, k):
        """The rally on the picture at game step k (a finished rally while the world is frozen after it)."""
        if self.phase_at(k) in ("pause", "end"):
            return max(1, sum(1 for p in self.points if p["k"] <= k))
        return 1 + sum(1 for p in self.points if p["k"] < k)


def race_build_package(bm, p, recap=False, live=False):
    """The package of racing point p (a crash or a heat end): shots dict(cam, k0, k1, speed, label, hold).
    A crash: main camera 1x, then trackside at the crash 0.5x (live=True: the short instant replay while the race goes
    on). A heat end: main 1x, trackside at the finish line 0.5x (a finish), overhead 0.5x."""
    import math
    dt = bm.dt
    sec = lambda x: int(round(x / dt))
    after = sec(0.5) if p["kind"] == "crash" else 0
    lo = bisect.bisect_right(bm.ks, p["k0"])
    hi = bisect.bisect_right(bm.ks, min(p["k"] + after, bm.ks[-1] if bm.ks else p["k"]))
    if hi - lo < 2:
        return []
    kfirst, kend, kp = bm.ks[lo], bm.ks[hi - 1], min(p["k"], bm.ks[hi - 1])
    clip = lambda a, b: (max(kfirst, a), min(kend, b))
    lay = (bm.hdr or {}).get("qpos_layout") or {"A": 0, "B": 13}
    shots = []
    if p["kind"] == "crash":
        i = max(lo, bisect.bisect_right(bm.ks, kp) - 1)
        q = bm.recs[i]["q"]
        f = p.get("seat") or "A"
        v = int(lay.get("B" if f == "A" else "A", 0))            # the car that was hit stays where it was
        x, y = q[v], q[v + 1]
        if bm.geo:
            L, point, project = bm.geo
            s0, _ = project((x, y))
            _, (tx, ty) = point(s0)
            out = (ty, -tx)                                       # the outside of a counter-clockwise track
        else:
            out = (0.0, -1.0)
        az = math.degrees(math.atan2(-out[1], -out[0]))
        trackside = ([x, y, 0.12], 4.2, az + 20.0, -30.0)
        if not recap:
            a, b = clip(kp - sec(3.0 if live else 5.0), kp + sec(0.4))
            shots.append(dict(cam="main", k0=a, k1=b, speed=1.0, label="Main camera · real speed", hold=0.3))
        a, b = clip(kp - sec(1.2), kp + sec(0.45))
        shots.append(dict(cam=trackside, k0=a, k1=b, speed=0.5, label="Trackside · 0.5×", hold=0.8 if live else 0.4))
        if not recap and not live:
            a, b = clip(kp - sec(1.2), kp + sec(0.45))
            shots.append(dict(cam="high", k0=a, k1=b, speed=0.5, label="Overhead · 0.5×", hold=0.8))
        return [s for s in shots if s["k1"] > s["k0"]]
    if not recap:
        a, b = clip(kend - sec(6.0), kend)
        shots.append(dict(cam="main", k0=a, k1=b, speed=1.0, label="Main camera · real speed", hold=0.4))
    r = p.get("result") or {}
    if p["kind"] == "heat" and r.get("decided_by") == "finish" and bm.geo:
        L, point, project = bm.geo
        (fx, fy), _ = point(0.0)                                  # the finish line: s = 0, on the -y straight
        a, b = clip(kend - sec(1.4), kend + sec(0.3))
        shots.append(dict(cam=([fx + 0.4, fy, 0.1], 5.2, 75.0, -30.0), k0=a, k1=b, speed=0.5, label="Finish line · 0.5×",
                          hold=0.4))
    elif p["kind"] == "dq":
        c = (bm.crashes or [{}])[-1]
        q = bm.recs[hi - 1]["q"]
        v = int(lay.get("B" if c.get("at_fault") == "A" else "A", 0))
        a, b = clip(kend - sec(1.6), kend + sec(0.3))
        shots.append(dict(cam=([q[v], q[v + 1], 0.12], 4.2, 60.0, -30.0), k0=a, k1=b, speed=0.5,
                          label="Trackside · 0.5×", hold=0.4))
    if not recap:
        a, b = clip(kend - sec(2.4), kend)
        shots.append(dict(cam="high", k0=a, k1=b, speed=0.5, label="Overhead · 0.5×", hold=1.2))
    return [s for s in shots if s["k1"] > s["k0"]]


def hockey_build_package(bm, p, recap=False):
    """The package of air hockey rally p: main 1x, then the goal-line camera (a goal) or a close-up of the puck (a
    stoppage) at 0.5x, then overhead 0.5x (the time cap / the end: the main camera only)."""
    dt = bm.dt
    sec = lambda x: int(round(x / dt))
    lo = bisect.bisect_right(bm.ks, p["k0"])
    hi = bisect.bisect_right(bm.ks, p["k"])
    if hi - lo < 2:
        return []
    kfirst, kend = bm.ks[lo], bm.ks[hi - 1]
    clip = lambda a, b: (max(kfirst, a), min(kend, b))
    shots = []
    if not recap or p["kind"] in ("time_cap", "end"):
        a, b = clip(kend - sec(6.0), kend)
        shots.append(dict(cam="main", k0=a, k1=b, speed=1.0, label="Main camera · real speed", hold=0.5))
    if p["kind"] in ("time_cap", "end"):
        return [s for s in shots if s["k1"] > s["k0"]]
    q = bm.recs[hi - 1]["q"]
    if p["kind"] == "goal":
        sx = 1.0 if p.get("winner") == "A" else -1.0              # A scores into B's goal at +x
        cam1 = ([sx * (HOCKEY_GOAL_X - 0.12), 0.0, 0.0], 1.25, 90.0 + sx * 28.0, -34.0)
        a, b = clip(kend - sec(1.6), kend)
        lab = "Goal-line camera · 0.5×"
    else:
        x = max(-0.9, min(0.9, float(q[0])))
        cam1 = ([x, float(q[1]) * 0.6, 0.0], 1.05, 90.0, -36.0)
        a, b = clip(kend - sec(2.2), kend)
        lab = "Puck close-up · 0.5×"
    shots.append(dict(cam=cam1, k0=a, k1=b, speed=0.5, label=lab, hold=0.4))
    if not recap:
        a, b = clip(kend - sec(2.0), kend)
        shots.append(dict(cam="high", k0=a, k1=b, speed=0.5, label="Overhead · 0.5×", hold=1.2))
    return [s for s in shots if s["k1"] > s["k0"]]


class RTCaptions(SoccerCaptions):
    """Lower thirds of a racing / air hockey match from the sim's /data (as soccer's): each seat's reviews (episode.jsonl
    "review": WHY -> PLAN) and latest model output (think_calls*.jsonl: tool calls and text, never the reasoning)."""
    SCRIPT = RT_SCRIPT


class RTProgram(Program):
    """Program's director with the sport's packages, graphics, captions; no interpolation across a cut."""
    SPORT = ""

    def __init__(self, bm):
        super().__init__(bm)
        self.caps = RTCaptions(bm)

    def seats(self):
        n = self.bm.names()
        return [("A", n["A"]), ("B", n["B"])] if self.bm.mode == "duel" else [("A", n["A"])]

    def result_cn(self, p):
        return p["how"]

    def record_at(self, kf):
        i, ra, rb, alpha = super().record_at(kf)
        if rb is not None and rb.get("cut"):
            rb, alpha = None, 0.0
        return i, ra, rb, alpha

    def target_fps(self):
        return target_fps(LIVE)

    def bug(self, now, k):
        raise NotImplementedError

    def base_overlay(self, now, k_score):
        sport, unit, rows, phase = self.bug(now, k_score)
        return dict(program=True, bug=dict(sport=sport, unit=unit, rows=rows), phase=phase,
                    lower=self.caps.lower(time.time(), self.seats()))

    def live_banner(self, k):
        return None

    def compose(self, now):
        ra, rb, alpha, ov, key = super().compose(now)
        changed = False
        if self.mode == "replay" and self.pkg and ra.get("k") is not None:
            # the bug shows the replayed moment (heat clock, checkpoints, the seven-second count), not the present
            sport, unit, rows, phase = self.bug(now, int(ra["k"]))
            ov["bug"] = dict(ov["bug"], unit=unit, rows=rows)
            changed = True
        if self.mode == "live" and self.display_step is not None and not ov.get("banner"):
            b = self.live_banner(self.display_step)
            if b:
                ov["banner"] = b
                changed = True
        if changed:
            key = (self.mode, round(alpha, 3), id(ra), json.dumps({x: v for x, v in ov.items() if x != "trail"},
                                                                  sort_keys=True, default=str))
        return ra, rb, alpha, ov, key


class RacingProgram(RTProgram):
    """The program feed of a racing match (see the comment above RACING)."""

    def package(self, p, recap=False):
        return race_build_package(self.bm, p, recap=recap)

    def bug(self, now, k):
        bm = self.bm
        n = bm.names()
        heat = bm.heat_at(k)
        hw = bm.heats_won(k)
        rows = []
        for s in ("A", "B"):
            cp = bm.cp_at(k, heat, s)
            w = bm.warnings_at(k, s)
            note = ("Finished" if cp >= RACE_GATES else f"CP {cp}/{RACE_GATES}") + (
                f" · {w} warning{'s' if w > 1 else ''}" if w else "")
            rows.append([s, n[s], hw.get(s, 0), note])
        ph = bm.phase_at(bm.ks[-1])
        unit = f"Heat {heat} · {bm.heat_clock(k):4.1f} s"
        phase = ""
        if ph == "end":
            phase = "Race over"
        elif ph == "pause":
            phase = "Heat over · world frozen for the review"
            unit = f"Heat {heat} · over"
        elif ph in ("wait", "start"):
            phase = "Waiting for the players"
        else:
            r = bm.rec_at(k)
            if r.get("lead") and r.get("gap", 0) > 0.05:
                phase = f"{n[r['lead']]} leads by {r['gap']:.1f} m"
        return f"Racing · {'Duel' if bm.mode == 'duel' else 'Single'}", unit, rows, phase

    def live_banner(self, k):
        bm = self.bm
        c = next((p for p in reversed(bm.points) if p["kind"] == "crash" and 0 <= k - p["k"] <= RACE_BANNER_S / bm.dt), None)
        if c:
            return c["how"]
        h = bm.heat_at(k)
        k0 = bm.heat_k0.get(h)
        g = bm.grid.get(h) or {}
        if k0 is not None and 0 <= k - k0 <= RACE_BANNER_S / bm.dt and bm.phase_at(k) == "run" and g:
            inner = next((s for s, v in g.items() if v == "inner"), None)
            return f"Heat {h} · go! · {bm.names().get(inner, inner)} starts on the inside" if inner else f"Heat {h} · go!"
        return None

    def direct(self, now):
        """Program's director, plus (world slowed enough) a crash as an instant replay while the race goes on, and a
        heat end's package played in full even when the next heat starts at once (no review wait)."""
        bm = self.bm
        if self.stinger or bm.slowdown < RACE_LIVE_MIN_SLOW:
            return super().direct(now)
        sec = lambda x: int(round(x / bm.dt))
        if self.mode == "live":
            for c in bm.points:
                wait = sec(0.4) if c["kind"] == "crash" else 0
                if c["n"] in self.replayed or self.pos < c["k"] + wait:
                    continue
                self.replayed.add(c["n"])
                if self.pos - c["k"] > sec(2.5):                  # joined late: too old for an instant replay
                    continue
                pkg = race_build_package(bm, c, live=c["kind"] == "crash")
                if pkg:
                    self.stinger = dict(t0=now, to="replay", pkg=pkg, point=c)
                    self.hold_n = c["n"]                          # this package is not cut by play resuming
                    self.note((f"crash {c['n']}: instant replay while the race goes on (" if c["kind"] == "crash"
                               else f"{c['label']} over: package (") + ", ".join(s["label"] for s in pkg) + ")")
                    return
        elif self.mode == "replay" and self.pkg and self.pkg_point and self.pkg_point["n"] == getattr(self, "hold_n", None) \
                and bm.phase_at(bm.ks[-1]) == "run":
            shot = self.pkg[self.shot_i]                          # the world runs: the package is not cut short
            if now - self.shot_t0 >= (shot["k1"] - shot["k0"]) * bm.dt / shot["speed"] + shot["hold"]:
                self.shot_i += 1
                self.shot_t0 = now
                if self.shot_i >= len(self.pkg):
                    self.shot_i = len(self.pkg) - 1
                    self.stinger = dict(t0=now, to="live")
                    self.hold_n = None
                    self.note("replay done: back to live")
            return
        return super().direct(now)


class HockeyProgram(RTProgram):
    """The program feed of an air hockey match (see the comment above RACING)."""

    def package(self, p, recap=False):
        return hockey_build_package(self.bm, p, recap=recap)

    def bug(self, now, k):
        bm = self.bm
        n = bm.names()
        sa, sb = bm.score_at(k)
        rows = []
        for s, sc in (("A", sa), ("B", sb)):
            f = bm.fouls_at(k, s)
            rows.append([s, n[s], sc, f"{f} foul{'s' if f > 1 else ''}" if f else ""])
        ph = bm.phase_at(bm.ks[-1])
        unit = f"Rally {bm.rally_at(k)} · {mmss(k * bm.dt)} / {mmss(HOCKEY_CAP_S)}"
        phase = ""
        if ph == "end":
            phase = "Match over"
        elif ph == "pause":
            phase = "Stoppage · world frozen for the review"
        elif ph in ("wait", "start"):
            phase = "Waiting for the players"
        else:
            r = bm.rec_at(k)
            if r.get("side") in ("A", "B") and r.get("side_t", 0) >= 2.0:
                phase = f"Seven-second count · {n[r['side']]} · {min(HOCKEY_SEVEN_S, r['side_t']):.1f} s"
        return f"Air hockey · {'Duel' if bm.mode == 'duel' else 'Single'}", unit, rows, phase

    def live_banner(self, k):
        bm = self.bm
        prev = next((p for p in reversed(bm.points) if p["k"] <= k), None)
        if prev is None or bm.phase_at(k) != "run" or not 0 <= k - prev["k"] <= 2.5 / bm.dt:
            return None
        nx = prev.get("next")
        if nx == "faceoff":
            return f"Rally {prev.get('rally', 0) + 1} · face-off at the centre"
        n = bm.names()
        if prev["kind"] == "goal":
            return f"Rally {prev.get('rally', 0) + 1} · {n.get(nx, nx)} serves (scored upon)"
        if prev["kind"] == "seven_second_foul":
            return f"Rally {prev.get('rally', 0) + 1} · possession change · puck to {n.get(nx, nx)}"
        return None

def make_match(d):
    """The match object for a discovered sim: billiards reads the episode log, the real-time sports the broadcast log."""
    sport = d["proj"].split("-")[0]
    if sport == "billiards":
        return BilliardsMatch(d)
    if sport in SOCCER:
        return SoccerMatch(d)
    if sport in RACING:
        return RacingMatch(d)
    if sport in HOCKEY:
        return HockeyMatch(d)
    return BMatch(d)


# ---------------------------------------------------------------- discovery
def discoverer(hosts):
    while True:
        time.sleep(1.0)
        if time.time() - STATE["last_view"] > IDLE_S and not any_clients():
            continue
        if time.time() - STATE["discovered"] < DISCOVER_S:
            continue
        try:
            found, counts, errs = lb.discover(hosts)
        except Exception as ex:  # noqa: BLE001
            found, counts, errs = None, {}, {"local": str(ex)}
        now = time.time()
        with LOCK:
            STATE["discovered"], STATE["errors"] = now, errs
            if found is None:
                continue
            cur = STATE["matches"]
            new = {}
            for (h, proj), d in found.items():
                if proj.split("-")[0] not in lb.REALTIME | TURN_BASED | SOCCER:
                    continue
                mid = lb.match_id(proj, h)
                new[mid] = cur.get(mid) or make_match(d)
                new[mid].gone = None
            for mid, m in cur.items():           # finished: keep an hour for the replays
                if mid not in new:
                    m.gone = m.gone or now
                    archive_match(m)
                    if now - m.gone < 3600:
                        new[mid] = m
            STATE["matches"] = new
        for m in list(new.values()):            # read the logs of the matches now (replays need the frames)
            if not m.gone:
                threading.Thread(target=m.tail, daemon=True).start()


def any_clients():
    with LOCK:
        return any((m.stream is not None and m.stream.clients > 0) or (m.program is not None and m.program.clients > 0)
                   for m in STATE["matches"].values())


def eager_replays():
    """While someone watches, render each newly finished point's replay right away (one at a time)."""
    while True:
        time.sleep(2.0)
        if time.time() - STATE["last_view"] > IDLE_S and not any_clients():
            continue
        with LOCK:
            ms = [m for m in STATE["matches"].values() if (m.sport, m.mode) in SUPPORTED and m.simdir_ok
                  and (not m.turn_based or (m.program is not None and m.program.clients > 0))]  # billiards: if watched
        for m in ms:
            if not m.gone:
                m.tail()
            if m.points and m.points[-1]["n"] not in m.replays:
                try:
                    m.replay(m.points[-1]["n"])
                except Exception as ex:  # noqa: BLE001
                    m.err = f"Replay render error: {ex}"[:200]


# Finished broadcasts survive a spectator-server restart. They are explicitly
# marked FINAL and use a saved still, while point replays remain real-speed MP4s.
def archive_match(bm):
    st = bm.stream if bm.stream and bm.stream.jpeg else bm.program
    if not st or not st.jpeg:
        return
    out = CACHE / "finished"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{bm.id}.jpg").write_bytes(st.jpeg)
    info = bm.info_json()
    info.update(gone=True, phase="end", viewers=0, playback=None,
                frame_url=f"/archive/frame/{bm.id}", replay_base=f"/archive/replay/{bm.id}",
                archived_at=time.time())
    (out / f"{bm.id}.json").write_text(json.dumps(info, ensure_ascii=False))


def archived_matches():
    out = []
    for p in (CACHE / "finished").glob("*.json"):
        try:
            m = json.loads(p.read_text())
            if time.time() - m["archived_at"] < 3600:
                out.append(m)
        except (OSError, ValueError, KeyError):
            continue
    return out


# ---------------------------------------------------------------- page
def public_activity(data):
    """Only public remarks, submitted reviews and coarse observable actions."""
    cards = []
    for card in data.get("cards", []):
        seats = []
        for seat in card.get("seats", []):
            steps = []
            for step in seat.get("steps", []):
                kind, content = step.get("k", ""), step.get("t", "")
                if kind == "说":
                    steps.append(dict(kind="Says", text=lb.clean(content, 600)))
                elif kind == "修改文件":
                    steps.append(dict(kind="Program", text="Updating the control program"))
                elif kind == "运行命令":
                    text = "Running a tool or the control program"
                    if re.search(r"game\s+ready", content):
                        text = "Reporting ready"
                    elif re.search(r"game\s+(status|observe)", content):
                        text = "Checking the game state and observations"
                    elif re.search(r"(tail|cat).*\.log", content):
                        text = "Checking the control program log"
                    elif re.search(r"(pytest|py_compile)", content):
                        text = "Checking the program"
                    steps.append(dict(kind="Action", text=text))
                elif kind.startswith("命令结束"):
                    steps.append(dict(kind="Tool", text="The last tool call returned"))
                elif kind == "一轮结束":
                    steps.append(dict(kind="Model", text="Model turn finished"))
                elif kind == "错误":
                    steps.append(dict(kind="Model", text="Model call reported an error; waiting"))
            seats.append(dict(seat=seat.get("seat"), label=seat.get("label"),
                              age=seat.get("h_age"), steps=steps))
        reviews = [dict(time=e.get("t"), text=lb.clean(e.get("x", ""), 1200))
                   for e in card.get("events", []) if e.get("c") == "rev"][:4]
        cards.append(dict(id=card.get("id"), age=card.get("updated_age"), seats=seats, reviews=reviews))
    return dict(cards=cards)


PAGE_PATH = HERE / "broadcast" / "page.html"
PAGE = PAGE_PATH.read_text(encoding="utf-8")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def match(self, mid):
        with LOCK:
            return STATE["matches"].get(mid)

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        STATE["last_view"] = time.time()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/activity":
            try:
                with urllib.request.urlopen("http://127.0.0.1:8047/api/state", timeout=2) as r:
                    data = json.load(r)
                return self.send(200, json.dumps(public_activity(data), ensure_ascii=False).encode(),
                                 "application/json; charset=utf-8")
            except (OSError, ValueError):
                return self.send(503, b'{"cards":[],"error":"progress unavailable"}', "application/json")
        if path == "/api/matches":
            with LOCK:
                ms = list(STATE["matches"].values())
            out = dict(now=time.time(), matches=[m.info_json() for m in ms if (m.sport, m.mode) in SUPPORTED],
                       other=[m.info_json() for m in ms if (m.sport, m.mode) not in SUPPORTED],
                       helper=dict(live_rt_s=round(LIVE.rt, 4), live_busy_s=round(LIVE.busy_s, 1),
                                   clip_busy_s=round(CLIP.busy_s, 1), soccer_rt_s=round(SOC.rt, 4),
                                   soccer_busy_s=round(SOC.busy_s + SOC_CLIP.busy_s, 1)),
                       video=dict(width=SIZE[0], height=SIZE[1], max_fps=MAX_FPS,
                                  helpers={h.role: dict(backend=h.backend, device=h.device,
                                                       started=h.proc is not None)
                                           for h in (LIVE, SOC, BILL)}),
                       errors=STATE["errors"])
            known = {m["id"] for m in out["matches"]}
            out["matches"].extend(m for m in archived_matches() if m["id"] not in known)
            return self.send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        mm = re.match(r"^/archive/frame/([a-z0-9]+)$", path)
        if mm:
            p = CACHE / "finished" / f"{mm.group(1)}.jpg"
            return self.send(200, p.read_bytes(), "image/jpeg") if p.exists() else self.send(404, b"no saved frame", "text/plain")
        mm = re.match(r"^/archive/replay/([a-z0-9]+)/(\d+)$", path)
        if mm:
            p = CACHE / "replays" / mm.group(1) / f"{int(mm.group(2))}.mp4"
            return self.file(p, "video/mp4") if p.exists() else self.send(404, b"no saved replay", "text/plain")
        mm = re.match(r"^/stream/([a-z0-9]+)$", path)
        if mm:
            return self.mjpeg(mm.group(1))
        mm = re.match(r"^/program/([a-z0-9]+)$", path)
        if mm:
            return self.mjpeg(mm.group(1), program=True)
        mm = re.match(r"^/program/([a-z0-9]+)\.jpg$", path)
        if mm:
            bm = self.match(mm.group(1))
            if bm is None or (bm.sport, bm.mode) not in SUPPORTED:
                return self.send(404, b"no such match", "text/plain")
            st = get_program(bm)
            st.last_client = time.time()
            for _ in range(100):
                if st.jpeg:
                    break
                time.sleep(0.1)
            return self.send(200, st.jpeg, "image/jpeg") if st.jpeg else self.send(503, b"no frame yet", "text/plain")
        mm = re.match(r"^/snapshot/([a-z0-9]+)\.jpg$", path)
        if mm:
            bm = self.match(mm.group(1))
            if bm is None:
                return self.send(404, b"no such match", "text/plain")
            st = get_program(bm) if bm.turn_based else get_stream(bm)
            st.last_client = time.time()
            for _ in range(100):
                if st.jpeg:
                    break
                time.sleep(0.1)
            return self.send(200, st.jpeg, "image/jpeg") if st.jpeg else self.send(503, b"no frame yet", "text/plain")
        mm = re.match(r"^/replay/([a-z0-9]+)/(\d+|last)(?:\.mp4)?$", path)
        if mm:
            bm = self.match(mm.group(1))
            if bm is None or (bm.sport, bm.mode) not in SUPPORTED:
                return self.send(404, b"no such match", "text/plain")
            bm.tail()
            if not bm.points:
                return self.send(404, b"no finished point yet", "text/plain")
            n = bm.points[-1]["n"] if mm.group(2) == "last" else int(mm.group(2))
            try:
                p = bm.replay(n)
            except Exception as ex:  # noqa: BLE001
                return self.send(500, f"replay failed: {ex}".encode(), "text/plain")
            if p is None:
                return self.send(404, b"no such point", "text/plain")
            return self.file(p, "video/mp4")
        return self.send(404, b"not found", "text/plain")

    def file(self, p, ctype):
        data = p.read_bytes()
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            a = int(m.group(1)) if m.group(1) else len(data) - int(m.group(2))
            b = int(m.group(2)) if m.group(1) and m.group(2) else len(data) - 1
            a, b = max(0, a), min(len(data) - 1, b)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def mjpeg(self, mid, program=False):
        quality = parse_qs(urlsplit(self.path).query).get("quality", ["original"])[0]
        if quality not in PROFILES:
            return self.send(400, b"Unknown quality profile", "text/plain")
        bm = self.match(mid)
        if bm is None or (bm.sport, bm.mode) not in SUPPORTED:
            return self.send(404, b"no such match", "text/plain")
        # Without a broadcast log there can be no first frame. Report this
        # before opening an MJPEG response rather than leaving an empty stream.
        if not bm.hdr and (bm.size < 0 or bm.gone):
            return self.send(503, (bm.err or "Waiting for the broadcast log, retry shortly").encode(), "text/plain; charset=utf-8")
        program = program or bm.turn_based              # a turn-based match has only the program feed
        st = get_program(bm) if program else get_stream(bm)
        with st.cond:
            st.clients += 1
        self.close_connection = True
        try:
            # Bound slow-client backlog; each iteration already takes the newest frame.
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
            self.connection.settimeout(2.0)
            self.send_response(200)
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.send_header("X-Video-Quality", quality)
            self.end_headers()
            seen = -1
            sent = -1
            while True:
                with st.cond:
                    st.cond.wait_for(lambda: st.seq != seen, timeout=5.0)
                    seq, jpg = st.seq, st.jpeg
                STATE["last_view"] = time.time()
                st.last_client = time.time()
                unchanged = seq == seen
                seen = seq
                if unchanged or jpg is None:
                    if bm.gone or st is not (bm.program if program else bm.stream):
                        return
                    continue
                delivery_seq, jpg = st.delivery.frame(quality, seq, jpg) if quality != "original" else (seq, jpg)
                if delivery_seq == sent:
                    continue
                sent = delivery_seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with st.cond:
                st.clients -= 1
            st.last_client = time.time()


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8048)
    ap.add_argument("--hosts", default="mll-4090-2.cs.northwestern.edu")
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    hosts = [h for h in a.hosts.split(",") if h]
    threading.Thread(target=discoverer, args=(hosts,), daemon=True).start()
    threading.Thread(target=eager_replays, daemon=True).start()
    try:
        Server(("0.0.0.0", a.port), H).serve_forever()
    finally:
        for h in (LIVE, CLIP, BILL, BILL_CLIP, SOC, SOC_CLIP):
            h.stop()


if __name__ == "__main__":
    main()
