#!/usr/bin/env python3
"""The match service of the air-hockey two-mode pilot tasks (sim sidecar): one game of the official format (first to
7 goals, the 120 s cap, USAA rules of sports/airhockey, revision v6 with the real iiwa 14 limits) against the scripted
opponent, in real time slowed by RO_SLOWDOWN, TRUE-state observation (pilot: no camera), the world frozen after
every rally until the player sends its review (ready). HTTP + JSON on port 8000:

  GET  /spec      the level, the commands, the rules, the timing, the toolkit, the deliverable, (low) the contract
  GET  /status    {state: waiting_start | running | review | over, score, rally, k}
  POST /start     the game begins
  POST /observe   the true state (world.ModesHockey.observe) + {state}
  POST /act       {"commands": [command, ...]} (or one command): applied at the next control step in order; replies
                  {took_effect: k, results: [reply | reason refused]}; 409 if not running, 429 if another act is in
                  flight
  POST /wait      {"timeout_s": 60}: block while the world is frozen for your review (or until the game is over)
  POST /ready     {"why": "...", "plan": "..."}: your review after a rally; the world resumes
At every review the deliverable directory (/agent_output/skill) is snapshot to /data/skill_snapshots/rally_NN.
SIGUSR1 (the verifier's collect): write /data/final.json and stop."""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import world as W  # noqa: E402

DATA = Path(os.environ.get("RCB_DATA", "/data"))
AGENT_OUT = Path(os.environ.get("AGENT_OUTPUT", "/agent_output"))
LEVEL = os.environ.get("RCB_ACTION_LEVEL", "low")
OPP = int(os.environ.get("RO_LEVEL") or 1)
SEED = int(os.environ.get("RO_SEED") or 101)
SLOWDOWN = float(os.environ.get("RO_SLOWDOWN") or 10)
START_TIMEOUT_S = float(os.environ.get("START_TIMEOUT_S", 1800))
READY_TIMEOUT_S = float(os.environ.get("READY_TIMEOUT_S", 600))
SPEC_FILE = HERE / f"spec_{LEVEL}.json"
REVISION = "airhockey-modes-pilot-1"


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


class Match:
    def __init__(self):
        self.h = W.ModesHockey(seed=SEED, level=OPP, duel=False, first="A")
        self.seat = W.Seat(self.h, LEVEL)
        self.state = "waiting_start"
        self.t_created = time.time()
        self.lock = threading.Lock()
        self.queue = []
        self.in_flight = False
        self.reviews = []
        self.review_since = None
        self.n_rallies_seen = 0
        self.no_show = self.abandoned = False
        self.ready_timeouts = 0
        self.acts = 0
        self.final_written = False
        self.snapshots = []

    def loop(self):
        period = W.DT * SLOWDOWN
        nxt = time.time()
        while True:
            with self.lock:
                st = self.state
            if st == "over":
                return
            if st == "waiting_start":
                if time.time() - self.t_created > START_TIMEOUT_S:
                    with self.lock:
                        self.no_show, self.state = True, "over"
                    self.write_final()
                    return
                time.sleep(0.05)
                nxt = time.time()
                continue
            if st == "review":
                if time.time() - self.review_since > READY_TIMEOUT_S:
                    with self.lock:
                        self.ready_timeouts += 1
                        self.reviews.append({"rally": self.n_rallies_seen, "why": None, "plan": None, "timeout": True})
                        self.h.ready()
                        self.state = "running"
                time.sleep(0.05)
                nxt = time.time()
                continue
            with self.lock:
                q, self.queue = self.queue, []
                for cmds, ev, holder in q:
                    holder["took_effect"] = self.h.k + 1
                    holder["results"] = [self.seat.command(c) for c in cmds]
                    ev.set()
                self.h.step()
                if len(self.h.results) > self.n_rallies_seen:
                    self.n_rallies_seen = len(self.h.results)
                    if not self.h.done:
                        self.state, self.review_since = "review", time.time()
                if self.h.done:
                    self.state = "over"
            if self.state == "over":
                self.snapshot("final")
                self.write_final()
                return
            nxt += period
            time.sleep(max(0.0, nxt - time.time()))

    def snapshot(self, tag):
        src = AGENT_OUT / "skill"
        if not src.is_dir():
            return
        dst = DATA / "skill_snapshots" / tag
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            self.snapshots.append(tag)
        except Exception as e:  # noqa: BLE001
            self.snapshots.append(f"{tag}: failed {e}"[:120])

    def metrics(self):
        h = self.h
        sc = h.scores()
        goals = [r for r in h.results if r["how"] == "goal"]
        return {"score": [sc["A"]["points"], sc["B"]["points"]], "won": bool(h.done and h.winner() == "A"),
                "decided_by": h.decided_by, "rallies": len(h.results), "t_play": round(h.k * W.DT, 2),
                "goals_for": sum(g["winner"] == "A" for g in goals), "goals_against": sum(g["winner"] == "B" for g in goals),
                "own_goals_against": sum(g["winner"] == "B" and bool(g.get("own_goal")) for g in goals),
                "fouls_A": sc["A"]["fouls"], "fouls_B": sc["B"]["fouls"], "strikes": self.seat.n_strikes,
                "plans": self.seat.n_plans, "moves": self.seat.n_moves, "refusals": self.seat.refusals,
                "results": h.results}

    def write_final(self, collected=False):
        with self.lock:
            if not self.final_written and self.state != "over":
                self.abandoned = self.state != "waiting_start"
                self.no_show = self.state == "waiting_start"
                self.state = "over"
            m = self.metrics()
            won = m["won"] and not (self.abandoned or self.no_show)
            out = {"graded_ready": True, "level": LEVEL, "opponent_level": OPP, "seed": SEED, "slowdown": SLOWDOWN,
                   "score": m["score"], "won": won, "complete": bool(self.h.done), "no_show": self.no_show,
                   "abandoned": self.abandoned, "ready_timeouts": self.ready_timeouts, "acts": self.acts, "match": m,
                   "reviews": self.reviews, "snapshots": self.snapshots, "revision": REVISION,
                   "hockey_revision": W.H.REVISION, "collected": collected}
            self.final_written = True
        DATA.mkdir(parents=True, exist_ok=True)
        tmp = DATA / "final.json.tmp"
        tmp.write_text(json.dumps(out, default=_j))
        tmp.replace(DATA / "final.json")


M: Match = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, default=_j).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return None

    def _path(self):
        parts = [x for x in self.path.split("?")[0].rstrip("/").split("/") if x]
        if parts and parts[0] == "A":
            parts = parts[1:]
        return parts[0] if parts else ""

    def do_GET(self):
        p = self._path()
        if p == "spec":
            return self._send(200, json.loads(SPEC_FILE.read_text()))
        if p == "status":
            with M.lock:
                return self._send(200, {"state": M.state, "score": M.metrics()["score"], "rally": len(M.h.results) + 1,
                                        "k": M.h.k, "slowdown": SLOWDOWN})
        return self._send(404, {"error": "unknown"})

    def do_POST(self):
        p, body = self._path(), self._body()
        if body is None:
            return self._send(400, {"error": "the body must be JSON"})
        if p == "start":
            with M.lock:
                if M.state == "waiting_start":
                    M.state = "running"
            return self._send(200, {"state": M.state})
        if p == "observe":
            with M.lock:
                o = M.h.match_view("A")
                o["state"] = M.state
            return self._send(200, o)
        if p == "wait":
            t_end = time.time() + min(float(body.get("timeout_s", 60)), 600)
            st = None
            while time.time() < t_end:
                with M.lock:
                    st = M.state
                if st in ("running", "over"):
                    break
                time.sleep(0.05)
            return self._send(200, {"state": st})
        if p == "ready":
            with M.lock:
                if M.state != "review":
                    return self._send(409, {"error": f"no review due (state {M.state})"})
                tag = f"rally_{M.n_rallies_seen:02d}"
            M.snapshot(tag)
            with M.lock:
                M.reviews.append({"rally": M.n_rallies_seen, "why": str(body.get("why", ""))[:2000],
                                  "plan": str(body.get("plan", ""))[:2000], "score": M.metrics()["score"]})
                M.h.ready()
                M.state = "running"
            return self._send(200, {"state": "running"})
        if p == "act":
            cmds = body.get("commands", body)
            cmds = cmds if isinstance(cmds, list) else [cmds]
            with M.lock:
                if M.state != "running":
                    return self._send(409, {"error": f"not now (state {M.state})"})
                if M.in_flight:
                    return self._send(429, {"error": "one act in flight"})
                M.in_flight = True
                ev, holder = threading.Event(), {}
                M.queue.append((cmds, ev, holder))
                M.acts += 1
            ok = ev.wait(30)
            with M.lock:
                M.in_flight = False
            if not ok:
                return self._send(409, {"error": "the world stopped before your act took effect"})
            return self._send(200, holder)
        return self._send(404, {"error": "unknown"})


def main():
    global M
    M = Match()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "server.pid").write_text(str(os.getpid()))

    def on_usr1(*_):
        M.snapshot("collect")
        M.write_final(collected=True)

    signal.signal(signal.SIGUSR1, on_usr1)
    threading.Thread(target=M.loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8000))), Handler).serve_forever()


if __name__ == "__main__":
    main()
