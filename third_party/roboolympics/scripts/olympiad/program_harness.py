#!/usr/bin/env python3
"""Replay-as-live test of the program feed (broadcast.py Program: director, stinger, highlight packages, captions).

    python3 scripts/olympiad/program_harness.py --trial <finished table-tennis duel trial dir> \
        --sim ~/.cache/olympiad_broadcast/sims/<match id> --out <dir> [--max-pause 25] [--max-wait 12] [--tail 45]

No match is touched: the trial's recorded joint commands are replayed offline through the sim's own code in a
throw-away container (broadcast/regen_log.py) into a broadcast log with the original wall times; long waits are
compressed (pre-match to --max-wait s, reviews to --max-pause s). The log is then fed at real pace into a match
object of broadcast.py, the seats' model logs (Codex rollout / Claude Code session files, which carry timestamps) and
the episode log's reviews are fed at the same pace as the live board's card, and the program feed is recorded:
<out>/program.mp4 (--record-fps, default 30 fps), stills at every director decision and every 15 s (<out>/stills), <out>/timeline.json
(director decisions, CPU of the render helper and of this process).

Racing / air hockey (a trial dir named racing-* / airhockey-*, --sim = the task's environment/sim): the log is rebuilt
with broadcast/regen_rt.py, the program class is the match's own (RacingProgram / HockeyProgram), the captions read
the trial's artifacts/data (reviews, think_calls) at the same pace.

Billiards (a trial dir named billiards-*, --sim = the task's environment/sim or a copy): the turn-based path. Its
episode log (artifacts/data/episode.jsonl: decisions, shots, reviews) IS what the live server reads, so it is fed as is
(its own wall times, long thinking gaps compressed to --max-pause), "ts" moved to the harness clock; the billiards
render helper replays the decisions with the task's own code. --http-capture also records the MJPEG stream as served
over HTTP (needs --serve) into <out>/program_http.mp4.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def regen(trial, sim, out, sport="tabletennis"):
    data = Path(trial) / "artifacts" / "data"
    if not (data / "trajectory.json").exists():
        data = Path(trial) / "verifier"
    out.mkdir(parents=True, exist_ok=True)
    script = ["/bc/regen_rt.py", sport] if sport in ("racing", "airhockey") else ["/bc/regen_log.py"]
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "1", "-u", f"{os.getuid()}:{os.getgid()}",
                    "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "-v", f"{sim}:/sim:ro", "-v", f"{data}:/data_in:ro", "-v", f"{HERE / 'broadcast'}:/bc:ro",
                    "-v", f"{out}:/out", "-e", "MUJOCO_GL=osmesa", IMAGE, "nice", "-n", "10", "python3",
                    *script, "/sim", "/data_in", "/out/broadcast.jsonl"], check=True)
    return [json.loads(ln) for ln in open(out / "broadcast.jsonl")]


class Clock:
    """Original wall time <-> harness (virtual) seconds, long gaps compressed."""

    def __init__(self, lines, max_wait, max_pause):
        ts = sorted(x["_ts"] for x in lines)
        self.o, self.v = [ts[0]], [0.0]
        for a, b in zip(ts, ts[1:]):
            cap = max_wait if len(self.o) <= 2 else max_pause
            self.o.append(b)
            self.v.append(self.v[-1] + min(b - a, cap))

    def virt(self, t):
        i = bisect.bisect_right(self.o, t) - 1
        if i < 0:
            return self.v[0] - (self.o[0] - t)
        if i >= len(self.o) - 1:
            return self.v[-1] + (t - self.o[-1])
        g = self.o[i + 1] - self.o[i]
        return self.v[i] + (t - self.o[i]) * ((self.v[i + 1] - self.v[i]) / g if g > 0 else 0)


def iso(t):
    from datetime import datetime
    return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def seat_items(seat_dir):
    """(ts, kind, text) of one seat's public model activity, from the timestamped session files."""
    out = []
    for f in sorted(glob.glob(str(seat_dir / "sessions" / "**" / "rollout-*.jsonl"), recursive=True)):   # Codex
        for ln in open(f, errors="replace"):
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            p = e.get("payload") or {}
            if e.get("type") != "response_item" or "timestamp" not in e:
                continue
            ts = iso(e["timestamp"])
            if p.get("type") == "message" and p.get("role") == "assistant":
                txt = " ".join(c.get("text", "") for c in p.get("content") or [] if c.get("type") == "output_text")
                if txt.strip():
                    out.append((ts, "说", txt))
            elif p.get("type") == "custom_tool_call":
                inp = p.get("input") or ""
                m = re.search(r'tools\.exec_command\(\{\s*cmd:\s*"((?:[^"\\]|\\.)*)"', inp)
                if m:
                    out.append((ts, "运行命令", json.loads('"' + m.group(1) + '"')))
                elif "tools.apply_patch" in inp:
                    files = re.findall(r"\*\*\* (?:Update|Add|Delete) File: (\S+)", inp)
                    out.append((ts, "修改文件", ", ".join(files) or "patch"))
                elif "tools.view_image" in inp:
                    m = re.search(r'path:\s*"([^"]+)"', inp)
                    out.append((ts, "调用 view_image", m.group(1) if m else "image"))
    for f in sorted(glob.glob(str(seat_dir / "sessions" / "projects" / "**" / "*.jsonl"), recursive=True)):  # Claude
        for ln in open(f, errors="replace"):
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("type") != "assistant" or "timestamp" not in e:
                continue
            ts = iso(e["timestamp"])
            for kind, text, _ in lb.harness_steps(ln.encode(), "claude"):
                out.append((ts, kind, text))
    return sorted(out, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wait", type=float, default=12.0)
    ap.add_argument("--max-pause", type=float, default=25.0)
    ap.add_argument("--tail", type=float, default=45.0, help="seconds recorded after the last log line")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0: whole match)")
    ap.add_argument("--serve", type=int, default=0, help="also serve the viewer page / and /program/<id> on this port")
    ap.add_argument("--http-capture", action="store_true", help="also record /program/<id> over HTTP (with --serve)")
    ap.add_argument("--record-fps", type=int, default=30, choices=(24, 25, 30, 60),
                    help="recording frame rate; use the broadcast rate to inspect transitions")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "stills").mkdir(parents=True, exist_ok=True)
    os.environ["BROADCAST_CACHE"] = str(out / "cache")
    global IMAGE, lb
    import broadcast as bc  # noqa: E402  (after BROADCAST_CACHE is set)
    lb, IMAGE = bc.lb, bc.IMAGE
    trial = Path(a.trial).resolve()
    proj = trial.name.lower()
    job = trial.parent.name
    mid = lb.match_id(proj, "local")
    dst = out / "cache" / "sims" / mid
    if not (dst / ".ok").exists():
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(Path(a.sim).expanduser(), dst, ignore=shutil.ignore_patterns("__pycache__", "think_proxy.py"))
        (dst / ".ok").write_text("harness")
    billiards = proj.startswith("billiards")
    if billiards:                                       # turn-based: the episode log is the live input
        lines = [dict(json.loads(ln)) for ln in open(trial / "artifacts" / "data" / "episode.jsonl") if ln.strip()]
        for x in lines:
            x["_ts"] = x["ts"]
    else:
        lines = regen(trial, Path(a.sim).expanduser().resolve(), out, proj.split("-")[0])
    clock = Clock(lines, a.max_wait, a.max_pause)
    feed = sorted(((clock.virt(x.pop("_ts")), x) for x in lines), key=lambda x: x[0])

    bm = bc.make_match(dict(job=job, proj=proj, host="local", sim="-", seats={}))
    bm.simdir_ok = True
    t0 = time.time() + 1.0
    state = {"i": 0}

    def tail(force=False):
        now = time.time() - t0
        while state["i"] < len(feed) and feed[state["i"]][0] <= now:
            v, x = feed[state["i"]]
            if "ts" in x:
                x = dict(x, ts=round(t0 + v, 3))         # the line as written now (captions' ages)
            bm.ingest(json.dumps(x).encode())
            state["i"] += 1
        bm.size = 1
    bm.tail = tail

    # the live board's card of this match, at the same pace
    info = bm.info
    items = {s: [(clock.virt(t), k, x) for t, k, x in seat_items(trial / "agents" / s)]
             for s in ("A", "B") if (trial / "agents" / s).is_dir()}
    epi = trial / "artifacts" / "data" / "episode.jsonl"
    revs = []
    for ln in open(epi):
        e = json.loads(ln)
        if billiards:                                   # its reviews come from the episode log itself
            break
        if e.get("ev") in ("ready", "review") and (e.get("why") or e.get("plan")):
            r = lb.fmt_event(e, info)
            if r:
                revs.append((clock.virt(e["ts"]), r[1], r[2]))

    class FakeCaptions(bc.Captions):
        def card(self, now):
            v = now - t0
            seats = []
            for s, its in items.items():
                seen = [x for x in its if x[0] <= v]
                seats.append(dict(seat=s, steps=[dict(k=k, t=lb.clean(x, 600)) for _, k, x in seen[-3:]][::-1],
                                  h_age=(v - seen[-1][0]) if seen else None))
            evs = [dict(t=lb.bj(t0 + t), c=c, x=x, age=v - t) for t, c, x in revs if t <= v][-12:][::-1]
            return dict(id=bm.id, seats=seats, events=evs)

    prog = (bc.BilliardsProgram if billiards else (bm.program_cls or bc.Program))(bm)
    if isinstance(prog.caps, bc.RTCaptions):            # racing / air hockey: the sim's /data at the same pace
        data = trial / "artifacts" / "data"
        reviews = [json.loads(ln) for ln in open(data / "episode.jsonl") if '"event": "review"' in ln]
        calls = {}
        for f in sorted(data.glob("think_calls*.jsonl")):
            m = re.match(r"think_calls_?([A-Za-z0-9]*)\.jsonl", f.name)
            calls[m.group(1) or "A"] = [json.loads(ln) for ln in open(f) if ln.strip()]

        class FakeRTCaptions(bc.RTCaptions):
            def update(self, now):
                v = now - t0
                for e in reviews:
                    tv = clock.virt(e["wall"])
                    if tv <= v and (e["wall"], e["seat"]) not in self.seen:
                        self.seen.add((e["wall"], e["seat"]))
                        self.review[e["seat"]] = dict(text=bc.redact(f"{e['why']} → {e['plan']}"), t=t0 + tv)
                for seat, cs in calls.items():
                    done = [c for c in cs if clock.virt(c.get("t1") or c.get("t0") or 0) <= v]
                    if not done or done[-1].get("seq") == self.tc_seen.get(seat):
                        continue
                    self.tc_seen[seat] = done[-1].get("seq")
                    cap = bc.soc_caption(done[-1].get("excerpt"))
                    if cap:
                        self.seat[seat] = dict(kind=cap[0], text=bc.redact(cap[1]),
                                               t=t0 + clock.virt(done[-1].get("t1") or done[-1]["t0"]))
        prog.caps = FakeRTCaptions(bm)
    elif billiards:
        class FakeBCaptions(FakeCaptions, bc.BilliardsCaptions):
            pass
        prog.caps = FakeBCaptions(bm)
    else:
        prog.caps = FakeCaptions(bm)
    # Measure actual new renders, not duplicated frames in the constant-rate recording.
    rendered_frames = []
    original_tick = prog.tick

    def measured_tick():
        began = time.monotonic()
        new = original_tick()
        if new:
            rendered_frames.append(dict(t=round(time.time() - t0, 4),
                                        render_ms=round((time.monotonic() - began) * 1000, 3),
                                        mode=prog.mode, stinger=bool(prog.stinger), shot=prog.shot_label,
                                        point=prog.pkg_point["n"] if prog.pkg_point else None))
        return new

    prog.tick = measured_tick
    prog.clients = 1
    with bc.LOCK:
        bc.STATE["matches"][bm.id] = bm
        bm.program = prog
    t_end_feed = feed[-1][0]
    while time.time() < t0:
        time.sleep(0.05)
    prog.start()
    if a.serve:                                         # the real HTTP handler, on the harness match
        threading.Thread(target=bc.Server(("0.0.0.0", a.serve), bc.H).serve_forever, daemon=True).start()

    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(a.record_fps), "-c:v", "mjpeg",
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-threads", "2", "-movflags", "+faststart", str(out / "program.mp4")], stdin=subprocess.PIPE)
    helper_name = f"eai-broadcast-{'billiards' if billiards else 'live'}-{os.getuid()}-{os.getpid()}"
    hff = None
    if a.http_capture and a.serve:                      # the same feed as a viewer gets it, over HTTP
        hff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-use_wallclock_as_timestamps", "1", "-f", "mjpeg",
                                "-i", f"http://127.0.0.1:{a.serve}/program/{bm.id}", "-vf", f"fps={a.record_fps}", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart",
                                str(out / "program_http.mp4")], stdin=subprocess.DEVNULL)
    cpu = []
    stop = threading.Event()

    def cpu_sampler():
        last = (time.time(), sum(os.times()[:2]))
        while not stop.wait(5.0):
            r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", helper_name],
                               capture_output=True, text=True)
            now, me = time.time(), sum(os.times()[:2])
            try:
                h = float(r.stdout.strip().rstrip("%"))
            except ValueError:
                h = None
            cpu.append(dict(t=round(now - t0, 1), helper_pct=h, harness_pct=round(100 * (me - last[1]) / (now - last[0]), 1),
                            mode=prog.mode, fps=prog.playback_info()["fps"], loadavg=os.getloadavg()[0]))
            last = (now, me)
    threading.Thread(target=cpu_sampler, daemon=True).start()

    stills, seen_log, last_sig, next_periodic, n_frames = [], 0, None, 15.0, 0
    pending = []
    tick = 1.0 / a.record_fps
    first_frame_s = None
    chapters = []
    last_chapter = None
    tn = time.time()
    limit = a.seconds or (t_end_feed + a.tail)
    while time.time() - t0 < limit:
        jpg = prog.jpeg
        if jpg:
            if first_frame_s is None:
                first_frame_s = time.time() - t0
            chapter_key = (prog.mode, bool(prog.stinger), prog.shot_label,
                           prog.pkg_point["n"] if prog.mode == "replay" and prog.pkg_point else None)
            if chapter_key != last_chapter:
                chapters.append(dict(video_s=round(n_frames / a.record_fps, 3),
                                     t=round(time.time() - t0, 3), mode=chapter_key[0],
                                     stinger=chapter_key[1], shot=chapter_key[2], point=chapter_key[3]))
                last_chapter = chapter_key
            ff.stdin.write(jpg)
            n_frames += 1
            v = time.time() - t0
            sig = (prog.mode, bool(prog.stinger), prog.shot_label)
            if sig != last_sig:
                pending.append((v + (0.25 if prog.stinger else 0.6), sig))
                last_sig = sig
            if v >= next_periodic:
                pending.append((v, ("periodic",)))
                next_periodic += 15.0
            for item in [p for p in pending if p[0] <= v]:
                pending.remove(item)
                name = f"{len(stills):03d}_{v:06.1f}s_" + re.sub(r"[^A-Za-z0-9.]+", "-", "_".join(map(str, item[1])))[:60] + ".jpg"
                (out / "stills" / name).write_bytes(prog.jpeg)
                stills.append(dict(t=round(v, 1), file=name, sig=list(map(str, item[1]))))
        tn += tick
        time.sleep(max(0.0, tn - time.time()))
    stop.set()
    ff.stdin.close()
    ff.wait()
    if hff is not None:
        hff.terminate()                                 # our own ffmpeg (by its Popen handle)
        hff.wait(timeout=20)
    if a.serve:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{a.serve}/api/matches", timeout=10) as r:
                (out / "api_matches.json").write_bytes(r.read())
        except OSError as ex:
            print("api/matches:", ex)
    prog.clients = 0
    prog.last_client = 0
    replay_mp4 = None
    if billiards and bm.points:                         # the page's per-shot MP4 (/replay/<match>/<n>)
        try:
            p = bm.replay(bm.points[-1]["n"])
            replay_mp4 = str(out / "replay_last_shot.mp4")
            shutil.copy(p, replay_mp4)
        except Exception as ex:  # noqa: BLE001
            print("replay:", ex)
    for h in (bc.LIVE, bc.CLIP, bc.BILL, bc.BILL_CLIP):
        h.stop()
    hel = [c["helper_pct"] for c in cpu if c["helper_pct"] is not None]
    summary = dict(match=mid, job=job, frames_written=n_frames, seconds=round(time.time() - t0, 1),
                   rendered=prog.frames, helper_busy_s=round((bc.BILL if billiards else bc.LIVE).busy_s, 1),
                   helper_cpu_pct_mean=round(sum(hel) / len(hel), 1) if hel else None,
                   helper_cpu_pct_max=max(hel) if hel else None,
                   director=[dict(t=round(t - t0, 1), x=x) for t, x in prog.log], stills=stills, cpu=cpu,
                   feed_end_s=round(t_end_feed, 1), replay_mp4=replay_mp4, err=bm.err)
    summary.update(record_fps=a.record_fps, first_frame_s=first_frame_s, chapters=chapters,
                   points=[dict(n=p["n"], winner=p["winner"], score=p["score"]) for p in bm.points])
    (out / "rendered_frames.json").write_text(json.dumps(rendered_frames))
    (out / "timeline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cpu", "stills", "chapters")}, ensure_ascii=False, indent=1))


IMAGE = lb = None
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    main()
