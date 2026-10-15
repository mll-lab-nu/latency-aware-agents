#!/usr/bin/env python3
"""直播看板: a read-only live view of the Olympiad matches running right now (Chinese page, Beijing time).

    python3 scripts/olympiad/live_board.py [--port 8047] [--hosts mll-4090-2.cs.northwestern.edu]

One card per running match: score and state, the latest events (points, shots, fouls, reviews with WHY/PLAN), the
newest image the player saved in /app/output, the model's latest step from its harness log and the tail of its own
program log. The page fetches /api/state every 3 s.

It never touches a match: no game/sim HTTP endpoint is called (a spectator `observe` could move a clock or a meter).
Everything is read from files: `docker exec <container> sh -c 'stat|tail|cat|find'` on this host and through
`ssh -o BatchMode=yes <host> docker exec ...` on the others, and the Harbor job directories (matched to containers by
compose project name). The episode log and the harness log are read incrementally (only the new bytes). Each container
is read at most once every POLL_S seconds, images at most once every IMG_S seconds, and nothing at all is read while
nobody has loaded the page in the last IDLE_S seconds.

实时比赛画面: each real-time match card embeds the live broadcast video and the replay of the last point from the
broadcast server (broadcast.py, --broadcast URL). The page's browser fetches those; this server never contacts it.
"""
import argparse
import calendar
import collections
import io
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# labels shared with the progress monitor on :8046; the sports and their protocols: sports/<sport>/sport.toml (plan.py)
from monitor import MODEL, NAMES, plan  # noqa: E402

try:
    from PIL import Image
except ImportError:  # thumbnails need Pillow; without it the original image is served
    Image = None

TREES = {"local": ["/data/guichi/rcb-oly-run", "/data/guichi/rcb-oly-run2", "/data/guichi/rcb-oly-run3", "/data/guichi/rcb-oly-run4",
                   "/data/guichi/rcb-olympiad", "/data/guichi/rcb-oly-broadcast", "/data/guichi/rcb-oly-soccer",
                   "/data/guichi/rcb-oly-racing-fix", "/data/guichi/rcb-oly-airhockey-fix"],
         "mll-4090-2.cs.northwestern.edu": ["/data/guichi/rcb-olympiad", "/data/guichi/rcb-oly-run3", "/data/guichi/rcb-oly-run4"]}
REALTIME = {s for s in NAMES if plan.realtime(s)}         # [sport] protocol = "realtime"
POLL_S, DISCOVER_S, IMG_S, IDLE_S = 3.0, 15.0, 8.0, 60.0
BJT = 8 * 3600
MONITOR_URL = "http://mll-4090-1.cs.northwestern.edu:8046/"
SITE_URL = "http://mll-4090-1.cs.northwestern.edu:8042/agent-sports-site/"
# 实时比赛画面: the broadcast server (broadcast.py); the page embeds its streams and replays (the board itself never
# talks to it, so the board's own reads stay as they are). --broadcast "" hides the video.
BROADCAST_URL = "http://mll-4090-1.cs.northwestern.edu:8048"
SSH_DIR = Path("/tmp") / f"live_board_ssh_{Path.home().name}"
LOCK = threading.Lock()
STATE = {"matches": {}, "hosts": {}, "errors": {}, "last_view": 0.0, "discovered": 0.0}
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{12,}|eyJ[A-Za-z0-9_-]{20,}[A-Za-z0-9_.-]*|gh[po]_[A-Za-z0-9]{20,}|"
                    r"\b[A-Z0-9_]*(?:TOKEN|API_KEY|SECRET|PASSWORD|ACCOUNT_ID)[A-Z0-9_]*\s*[=:]\s*\S+)")


def short(h):
    return "4090-1" if h == "local" else h.split(".")[0].replace("mll-", "")


def bj(ts, fmt="%H:%M:%S"):
    return time.strftime(fmt, time.gmtime(ts + BJT))


def clean(s, n=400):
    s = SECRET.sub("[已隐去]", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n - 1] + "…"


# ---------------------------------------------------------------- reading containers (read-only)
def run(host, argv, timeout=20):
    """Run argv on host (local or over ssh); returns stdout bytes or None on failure."""
    if host != "local":
        SSH_DIR.mkdir(mode=0o700, exist_ok=True)
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ControlMaster=auto",
                "-o", f"ControlPath={SSH_DIR}/%C", "-o", "ControlPersist=120", f"guichi@{host}", shlex.join(argv)]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return p.stdout if p.returncode == 0 or p.stdout else None


def dexec(host, container, script, *args, timeout=20):
    return run(host, ["docker", "exec", container, "sh", "-c", script, "sh", *map(str, args)], timeout=timeout)


SIM_SCRIPT = r"""
cd /data 2>/dev/null || exit 0
off=$1; fm=$2; cap=$3
s=$(stat -c %s episode.jsonl 2>/dev/null || echo 0)
st=$off; [ "$s" -lt "$off" ] && st=0
[ $((s-st)) -gt "$cap" ] && st=$((s-cap))
echo "@@EP $st $s"
[ "$s" -gt "$st" ] && tail -c +$((st+1)) episode.jsonl | head -c $((s-st))
f=$(stat -c %Y final.json 2>/dev/null || echo 0)
printf '\n@@FIN %s\n' "$f"
[ "$f" != "$fm" ] && [ "$f" != 0 ] && cat final.json
exit 0
"""

SEAT_SCRIPT = r"""
off=$1; cap=$2
echo "@@IMG"
find /app/output -maxdepth 2 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
  -not -path '*/__pycache__/*' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | tail -n 1
echo "@@PLOG"
for x in $(ls -t /app/output/*.log /app/output/*.out 2>/dev/null); do
  if [ -s "$x" ]; then stat -c '%Y %n' "$x"; tail -n 8 "$x" | cut -c1-240; break; fi
done
h=""
for x in /logs/agent/claude-code.txt /logs/agent/codex.txt; do [ -f "$x" ] && h=$x && break; done
if [ -n "$h" ]; then
  s=$(stat -c %s "$h"); m=$(stat -c %Y "$h")
  st=$off; [ "$s" -lt "$off" ] && st=0
  [ $((s-st)) -gt "$cap" ] && st=$((s-cap))
  echo "@@H $h $st $s $m"
  [ "$s" -gt "$st" ] && tail -c +$((st+1)) "$h" | head -c $((s-st))
fi
exit 0
"""


def take_section(out, tag):
    """Split `@@TAG a b c\n<n bytes>` where n = c - b (the byte range of an incremental read)."""
    i = out.find(tag)
    if i < 0:
        return None, None, out
    j = out.find(b"\n", i)
    head = out[i + len(tag):j].decode().split()
    return head, j + 1, out


# ---------------------------------------------------------------- job names
SPORTS = "|".join(NAMES)
RE_SINGLE = re.compile(rf"^(?P<tag>.+?)-(?P<sport>{SPORTS})-(?P<agent>[a-z0-9]+)(?:-L(?P<level>\d+))?(?:-x(?P<slow>[\d.]+))?"
                       rf"-s(?P<seed>h?\d+)-(?P<ts>\d{{8}}T\d{{6}})Z$")
RE_DUEL = re.compile(rf"^(?P<tag>.+?)-(?P<sport>{SPORTS})-(?P<fmt>duel|team)(?:-x(?P<slow>[\d.]+))?-s(?P<seed>h?\d+)-(?P<a>[a-z0-9]+?)A-"
                     rf"(?P<b>[a-z0-9]+?)B-(?P<ts>\d{{8}}T\d{{6}})Z$")


def parse_job(job, project):
    m = RE_DUEL.match(job or "")
    if m:
        d = m.groupdict()
        d.update(duel=True, team=d.pop("fmt") == "team", players={"A": MODEL.get(d["a"], d["a"]), "B": MODEL.get(d["b"], d["b"])})
    else:
        m = RE_SINGLE.match(job or "")
        d = m.groupdict() if m else {"sport": project.split("-")[0], "tag": "", "seed": "?", "slow": None}
        d["duel"] = "duel" in project.split("__")[0]
        d["players"] = ({"A": "A", "B": "B"} if d["duel"] else {"A": MODEL.get(d.get("agent"), d.get("agent") or "选手")})
    if d.get("ts"):
        d["start"] = calendar.timegm(time.strptime(d["ts"], "%Y%m%dT%H%M%S"))
    d["sport_cn"] = NAMES.get(d["sport"], d["sport"])
    cfg = [] if d["duel"] else [f"对程序对手 L{d['level']}" if d.get("level") else "单人"]
    if d.get("slow"):
        cfg.append(f"慢放 ×{d['slow']}")
    cfg.append(f"种子 {d.get('seed')}")
    if d.get("tag") and d["tag"] not in ("r1", "r2", "r3"):
        cfg.append(d["tag"])
    elif d.get("tag"):
        cfg.append(f"第 {d['tag'][1:]} 轮")
    d["cfg"] = " · ".join(cfg)
    return d


# ---------------------------------------------------------------- discovery
DISC = r"""
docker ps --format '{{.Names}}'
echo @@JOBS
for t in "$@"; do ls -d "$t"/jobs/*/*__*/ 2>/dev/null; ls -d "$t"/jobs/*/*__*/result.json 2>/dev/null; done
exit 0
"""


def discover(hosts):
    found, counts, errs = {}, {}, {}
    for h in ["local"] + hosts:
        out = run(h, ["sh", "-c", DISC, "sh", *TREES.get(h, TREES["local"])], timeout=30)
        if out is None:
            errs[short(h)] = "无法连接"
            continue
        names, _, jobs = out.decode("utf-8", "replace").partition("@@JOBS")
        names = set(names.split())
        done, trials = set(), {}
        for ln in jobs.split():
            ln = ln.rstrip("/")
            if ln.endswith("/result.json"):
                done.add(ln[:-len("/result.json")])
            else:
                trials[ln] = None
        n = 0
        for c in sorted(names):
            if not c.endswith("-sim-1") or "__" not in c:
                continue
            proj = c[:-len("-sim-1")]
            tdir = next((t for t in trials if Path(t).name.lower() + "__env" == proj), None)
            if tdir is None or tdir in done:
                continue                     # not an Olympiad trial of these run trees / already finished
            if not any(proj.startswith(s) for s in NAMES):
                continue
            job = Path(tdir).parent.name
            seats = {"A": f"{proj}-player-1"} if f"{proj}-player-1" in names else {}
            for s in ("A", "B", "A1", "A2", "B1", "B2"):     # duel seats; soccer team seats
                if f"{proj}-seat_{s.lower()}-1" in names:
                    seats[s] = f"{proj}-seat_{s.lower()}-1"
            found[(h, proj)] = dict(host=h, proj=proj, trial=tdir, job=job, sim=c, seats=seats)
            n += 1
        counts[short(h)] = n
    return found, counts, errs


# ---------------------------------------------------------------- per-match state
class Seat:
    def __init__(self, container):
        self.c = container
        self.h_off, self.h_buf, self.h_kind, self.h_mtime = 0, b"", "", 0
        self.steps = collections.deque(maxlen=6)
        self.plog, self.plog_name, self.plog_mtime = [], "", 0
        self.img_path, self.img_mtime, self.img_fetch = "", 0.0, 0.0
        self.thumb, self.thumb_v, self.thumb_name, self.thumb_mtime = None, "", "", 0.0


def match_id(proj, host):
    """The match's id on this board and on the broadcast server (broadcast.py)."""
    return re.sub(r"[^a-z0-9]", "", proj.split("__")[1]) + short(host).replace("-", "")


class Match:
    def __init__(self, d):
        self.__dict__.update(d)
        self.info = parse_job(self.job, self.proj)
        self.id = match_id(self.proj, self.host)
        self.ep_off, self.ep_buf, self.fin_mtime, self.final = 0, b"", "0", {}
        self.events = collections.deque(maxlen=60)
        self.last = {}           # latest event of each type (score fallbacks, turn, frozen state)
        self.clock_left = {}     # seat -> latest clock_left_s seen in an event
        self.seat = {s: Seat(c) for s, c in self.seats.items()}
        self.updated, self.err = 0.0, ""
        self.busy = False


SKIP = re.compile(rb'"ev": ?"(observe|act_[a-z]+|plan_ready|barrier_scheduled|resolved)"|"kind": ?"observe"')
SKIP_EV = {"observe", "act_applied", "act_received", "act_snapshot", "plan_ready", "barrier_scheduled", "resolved", "call"}


def ingest(m, ln):
    """One episode-log line: remember the latest event of each type; add a feed line if it is meaningful."""
    if SKIP.search(ln[:200]):
        return                         # high-rate protocol events (observe calls, act bookkeeping)
    try:
        e = json.loads(ln)
    except ValueError:
        return
    if not isinstance(e, dict) or e.get("ev") in SKIP_EV:
        return
    ev = e.get("ev")
    m.last[ev] = e
    if e.get("seat") in ("A", "B") and e.get("clock_left_s") is not None:
        m.clock_left[e["seat"]] = e["clock_left_s"]
    if ev in ("pause_for_review", "run", "end", "match_start", "game_start", "shootout_start", "round_start"):
        m.last["_phase"] = e
    r = fmt_event(e, m.info)
    if r:
        m.events.append(r)


def poll_sim(m):
    out = dexec(m.host, m.sim, SIM_SCRIPT, m.ep_off, m.fin_mtime, 3_000_000, timeout=30)
    if out is None:
        m.err = "读取模拟器日志失败"
        return
    head, pos, _ = take_section(out, b"@@EP ")
    if head is None:
        m.err = "模拟器还没有日志"
        return
    st, s = int(head[0]), int(head[1])
    data = out[pos:pos + (s - st)]
    if st != m.ep_off:
        m.ep_buf = b""                # skipped ahead (first read of a long log) or the file was replaced
        if st > 0:
            data = data[data.find(b"\n") + 1:]
    m.ep_off = s
    buf = m.ep_buf + data
    lines = buf.split(b"\n")
    m.ep_buf = lines.pop()
    for ln in lines:
        ingest(m, ln)
    i = out.find(b"\n@@FIN ", pos + (s - st) - 1)
    if i >= 0:
        j = out.find(b"\n", i + 1)
        fm = out[i + 7:j].decode().strip()
        body = out[j + 1:].strip()
        if body:
            try:
                m.final = json.loads(body)
                m.fin_mtime = fm
            except ValueError:
                pass                   # caught mid-write; read again next time
    m.err = ""
    m.updated = time.time()


def poll_seat(m, s, seat):
    out = dexec(m.host, seat.c, SEAT_SCRIPT, seat.h_off, 2_000_000, timeout=30)
    if out is None:
        return
    a = out.find(b"@@IMG\n")
    b = out.find(b"@@PLOG\n")
    hh = out.find(b"\n@@H ")
    img = out[a + 6:b].decode("utf-8", "replace").strip() if a >= 0 and b > a else ""
    if img:
        t, _, p = img.split(" ", 2)
        seat.img_path, seat.img_mtime = p, float(t)
    plog = out[b + 7:hh if hh >= 0 else len(out)].decode("utf-8", "replace").splitlines() if b >= 0 else []
    if plog:
        mt, _, name = plog[0].partition(" ")
        seat.plog_mtime, seat.plog_name = int(mt or 0), name
        seat.plog = [clean(x, 240) for x in plog[1:] if x.strip()][-8:]
    if hh >= 0:
        j = out.find(b"\n", hh + 1)
        h, st, size, mt = out[hh + 5:j].decode().split()
        st, size = int(st), int(size)
        data = out[j + 1:j + 1 + (size - st)]
        seat.h_kind = "codex" if h.endswith("codex.txt") else "claude"
        seat.h_mtime = int(mt)
        if st != seat.h_off:
            seat.h_buf = b""
            if st > 0:
                data = data[data.find(b"\n") + 1:]
        seat.h_off = size
        lines = (seat.h_buf + data).split(b"\n")
        seat.h_buf = lines.pop()
        if len(seat.h_buf) > 4_000_000:
            seat.h_buf = b""           # a giant line (an image in a tool result): drop it
        for ln in lines:
            for step in harness_steps(ln, seat.h_kind):
                seat.steps.append(step)
    if seat.img_path and (seat.img_path != seat.thumb_name or seat.img_mtime != seat.thumb_mtime) \
            and time.time() - seat.img_fetch >= IMG_S:
        seat.img_fetch = time.time()
        raw = dexec(m.host, seat.c, 'head -c 12000000 "$1"', seat.img_path, timeout=40)
        if raw:
            seat.thumb = thumbnail(raw)
            seat.thumb_name, seat.thumb_mtime = seat.img_path, seat.img_mtime
            seat.thumb_v = f"{int(seat.img_mtime * 1000)}"


def thumbnail(raw):
    if Image is None:
        return raw
    try:
        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        im.thumbnail((900, 900))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=80)
        return b.getvalue()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- harness logs (Claude Code / Codex)
def harness_steps(ln, kind):
    if not ln.strip() or len(ln) > 3_000_000:
        return []
    try:
        e = json.loads(ln)
    except ValueError:
        return []
    out = []
    now = time.time()
    if kind == "claude":
        if e.get("type") == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text", "").strip():
                    out.append(("说", clean(c["text"], 600), now))
                elif c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    arg = inp.get("command") or inp.get("file_path") or inp.get("description") or inp.get("pattern") \
                        or inp.get("url") or json.dumps(inp, ensure_ascii=False)
                    out.append((f"调用 {c.get('name')}", clean(arg, 500), now))
                elif c.get("type") == "thinking" and c.get("thinking", "").strip():
                    out.append(("思考", clean(c["thinking"], 400), now))
        elif e.get("type") == "result":
            out.append(("会话结束", clean(e.get("result") or e.get("subtype") or "", 300), now))
    else:
        it = e.get("item") or {}
        t = it.get("type")
        if e.get("type") == "item.completed" and t == "agent_message":
            out.append(("说", clean(it.get("text", ""), 600), now))
        elif e.get("type") == "item.completed" and t == "reasoning" and it.get("text"):
            out.append(("思考", clean(it.get("text", ""), 400), now))
        elif e.get("type") == "item.started" and t == "command_execution":
            cmd = re.sub(r"^/bin/(ba)?sh -lc ", "", it.get("command", ""))
            out.append(("运行命令", clean(cmd, 500), now))
        elif e.get("type") == "item.completed" and t == "command_execution":
            res = clean(it.get("aggregated_output", ""), 300)
            out.append((f"命令结束（退出码 {it.get('exit_code')}）", res or "（无输出）", now))
        elif e.get("type") == "item.completed" and t == "file_change":
            ch = it.get("changes") or []
            out.append(("修改文件", clean(", ".join(c.get("path", "") for c in ch), 300), now))
        elif e.get("type") == "item.completed" and t in ("web_search", "mcp_tool_call"):
            out.append((t, clean(json.dumps(it, ensure_ascii=False), 300), now))
        elif e.get("type") == "turn.completed":
            out.append(("一轮结束", "", now))
        elif e.get("type") in ("error", "turn.failed"):
            out.append(("错误", clean(json.dumps(e, ensure_ascii=False), 300), now))
    return out


# ---------------------------------------------------------------- events -> Chinese lines
def who(seat, info):
    if seat in (None, ""):
        return ""
    if info["duel"]:
        return f"{seat}（{info['players'].get(seat, seat)}）" if seat in ("A", "B") else str(seat)
    return {"you": "选手", "A": "选手", "opponent": "对手", 0: "选手", 1: "对手", "0": "选手", "1": "对手"}.get(seat, str(seat))


def sc(v, info):
    if isinstance(v, dict):
        return " : ".join(str(v.get(k)) for k in ("A", "B") if k in v) if ("A" in v or "B" in v) else json.dumps(v)
    if isinstance(v, (list, tuple)):
        return " : ".join(map(str, v))
    return str(v)


def fmt_event(e, info):
    """One feed line (ts, css class, text) for a meaningful event, else None."""
    ev, ts = e.get("ev"), e.get("ts") or time.time()
    seat = e.get("seat")
    if ev == "point":
        pts = e.get("points") or ([e["point"]] if isinstance(e.get("point"), dict) else [])
        lines = []
        for p in pts:
            n = p.get("point")
            lines.append(f"得分 → {who(p.get('winner'), info)}：{clean(p.get('how', ''), 200)}（比分 {sc(p.get('score'), info)}）"
                         + (f" 第 {n} 分" if isinstance(n, int) and info["duel"] else ""))
        return (ts, "pt", "；".join(lines)) if lines else None
    if ev == "shot" and isinstance(e.get("shots"), list):           # real-time singles: our shot results
        ss = [clean(x.get("summary", ""), 160) for x in e["shots"] if x.get("summary")]
        return (ts, "shot", "击球：" + "；".join(ss)) if ss else None
    if ev in ("review", "ready"):
        why, plan = clean(e.get("why", ""), 260), clean(e.get("plan", ""), 260)
        if not (why or plan):
            return (ts, "sys", f"{who(seat, info)} 准备就绪") if ev == "ready" else None
        tag = ""
        for k in ("call", "result", "arrow_score", "score"):
            if k in e and not isinstance(e[k], (dict, list)):
                tag = f"（{e[k]}）"
                break
        return (ts, "rev", f"复盘 {who(seat or 'A', info)}{tag} WHY：{why} PLAN：{plan}")
    if ev == "pause_for_review":
        return (ts, "sys", "暂停：世界冻结，等待复盘")
    if ev == "run":
        return (ts, "sys", "继续比赛" + (f"（击球手 {who(e['batter'], info)}）" if e.get("batter") else ""))
    if ev == "pitch_over":
        r = e.get("result") or {}
        bat = f"{who(r.get('batter'), info)} " if r.get("batter") else ""
        cnt = r.get("count_after")
        return (ts, "shot", f"{bat}第 {r.get('pitch')} 球 {r.get('pitch_type', '')} {r.get('radar_mps', '')} m/s → "
                            f"{r.get('outcome', '')}" + (f"（球数 {cnt[0]}-{cnt[1]}）" if isinstance(cnt, list) and len(cnt) == 2 else "")
                + (f" · 垒数 {sc(e['score'], info)}" if "score" in e else (f" · 垒数 {e['bases']}" if "bases" in e else "")))
    if ev == "dart":
        ev_ = f" · {e['event']}" if e.get("event") else ""
        return (ts, "shot", f"{who(seat or 'A', info)}{' 第 %s 局' % e['leg'] if e.get('leg') else ''} 第 {e.get('visit')} 轮 第 {e.get('dart')} 镖："
                            f"{e.get('call')}（{e.get('value')} 分，剩 {e.get('after')}）{ev_}")
    if ev in ("arrow",) or (ev == "shot" and "distance_m" in e):
        return (ts, "shot", f"{who(seat or 'A', info)} 第 {e.get('end')} 组 第 {e.get('arrow')} 箭（{e.get('distance_m'):.0f} 米）："
                            f"{e.get('call')}（{e.get('score')} 环）" + (f" 脱靶 {e['miss']}" if e.get("miss") else ""))
    if ev == "shot" and "made_total" in e or (ev == "shot" and "touched" in e):
        return (ts, "shot", f"{who(seat or 'A', info)} 第 {e.get('shot')} 投（点位 {e.get('spot')}）：{e.get('result')}"
                            f"（碰到 {'、'.join(e.get('touched') or [])}）")
    if ev == "shot" and isinstance(e.get("result"), dict) and "pins_down" in e["result"]:
        r = e["result"]
        x = "全中！" if r.get("strike") else ("补中" if r.get("spare") else ("洗沟" if r.get("gutter") else ""))
        return (ts, "shot", f"{who(seat, info)} 第 {r.get('frame')} 格 第 {r.get('ball')} 球：击倒 {r.get('pins_down')} 瓶 {x}"
                            f"（累计 {r.get('total_so_far')}）")
    if ev == "shot" and isinstance(e.get("result"), dict) and "hole" in e["result"]:
        r = e["result"]
        return (ts, "shot", f"{who(seat, info)} 第 {r.get('hole')} 洞 第 {r.get('stroke')} 杆：{'进洞！' if r.get('holed') else r.get('result')}"
                            f"（碰墙/障碍 {r.get('wall_or_block_hits', 0)} 次，此前各洞共 {r.get('total_so_far')} 杆）")
    if ev == "shot" and "potted" in e:
        pl = who(e.get("player"), info)
        pot = "、".join(map(str, e.get("potted") or [])) or "无"
        foul = f" 犯规：{e['foul']}" if e.get("foul") else ""
        call = f" 叫袋 {e['call']}{'（成功）' if e.get('call_made') else ''}" if e.get("call") else ""
        return (ts, "shot", f"{pl} 第 {e.get('k', 0) + 1} 杆{'（开球）' if e.get('break') else ''}：进球 {pot}{call}{foul}"
                            + ("，继续击球" if e.get("keeps_turn") else "") + (f" · {e['result']}" if e.get("result") else ""))
    if ev in ("shot", "throw") and ("v_actual" in e or "collisions_heard" in e or "removed" in e):
        by = who(seat or e.get("by"), info)
        rm = e.get("removed") or e.get("removed_why") or []
        line = f"{by} 第 {e.get('end', 0) + (0 if ev == 'throw' else 1)} 局 第 {e.get('throw', (e.get('k', 0) + 1))} 壶" \
               f"（{e.get('spin', '')}）" + (f"：碰撞 {e.get('collisions', e.get('collisions_heard'))} 次" if e.get('collisions', e.get('collisions_heard')) else "：无碰撞") \
               + (f"，移出 {len(rm)} 壶" if rm else "")
        ef = e.get("end_finished")
        if isinstance(ef, dict):
            line += f" · 本局结束：{who(['A', 'B'][ef.get('team', 0)] if info['duel'] else ef.get('team'), info)} 得 {ef.get('points')} 分（总分 {sc(ef.get('score'), info)}）"
        return (ts, "shot", line)
    if ev == "turn":
        cl = e.get("clock_left_s")
        return (ts, "sys", f"轮到 {who(seat, info)}" + (f"（剩余时钟 {mmss(cl)}）" if cl is not None else ""))
    if ev in ("match_start", "match_begins", "game_start", "shootout_start", "round_start"):
        return (ts, "sys", "比赛开始")
    if ev == "start" and seat:
        return (ts, "sys", f"{who(seat, info)} 已就位")
    if ev == "end":
        w = e.get("winner")
        return (ts, "end", f"比赛结束（{e.get('reason')}）" + (f"，胜者 {who(w, info)}" if w not in (None, "") else ""))
    if ev in ("clock_forfeit", "clock_expired", "clock_out", "lost", "start_timeout", "tiebreak", "error",
              "refused_not_model", "act_dropped", "video_error", "meter_error"):
        rest = {k: v for k, v in e.items() if k not in ("ts", "ev", "wall", "step")}
        return (ts, "bad", f"{ev} {clean(json.dumps(rest, ensure_ascii=False), 200)}")
    return None


def mmss(s):
    try:
        s = int(float(s))
    except (TypeError, ValueError):
        return "—"
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------- score / state per sport
def summarize(m):
    F, info, last = m.final or {}, m.info, m.last
    if isinstance(F.get("sport_fields"), dict):     # a v0.2 match (core/service.py COMPAT): the sport's own fields win
        F = {**F, **F["sport_fields"]}
    sp, duel = info["sport"], info["duel"]
    L, R = ("A " + info["players"]["A"], "B " + info["players"]["B"]) if duel else (info["players"]["A"], f"程序对手 L{info.get('level', '?')}")
    main, unit, det = None, "", []

    def AB(d):
        return (d.get("A"), d.get("B")) if isinstance(d, dict) else (None, None)

    if sp in ("tabletennis", "tennis", "badminton"):
        unit = "分"
        if duel:
            main = AB(F.get("score") or {})
            if F.get("hits") is not None:
                det.append(f"击球 {F['hits']} 次")
        else:
            main = (F.get("points_for"), F.get("points_against"))
            if F.get("win_points"):
                det.append(f"先得 {F['win_points']} 分者胜")
            if F.get("returns") is not None:
                det.append(f"回球成功 {F['returns']} 次")
    elif sp == "baseball":
        unit = "垒"
        if duel:
            main = AB(F.get("score") or {})
            det += [f"已投 {F.get('pitches', '—')} 球", f"打席 {F.get('plate_appearances', '—')}"]
            if (last.get("run") or {}).get("batter"):
                det.append(f"击球手 {who(last['run']['batter'], info)}")
        else:
            L, R = info["players"]["A"], f"目标 ≥{F.get('threshold', '?')} 垒（L{info.get('level', '?')}）"
            main = (F.get("bases"), None)
            det += [f"打席 {F.get('pa_done', '—')}/{F.get('n_pa', '—')}", f"安打 {F.get('hits', 0)}",
                    f"本垒打 {F.get('home_runs', 0)}", f"保送 {F.get('walks', 0)}", f"三振 {F.get('strikeouts', 0)}"]
        r = (last.get("pitch_over") or {}).get("result") or {}
        if isinstance(r.get("count_after"), list):
            det.append(f"当前球数 {r['count_after'][0]}-{r['count_after'][1]}")
    elif sp == "darts":
        unit = "局"
        if duel:
            main = AB(F.get("legs") or {})
            a, b = AB(F.get("remaining") or {})
            det.append(f"本局剩余 A {a} · B {b}")
        else:
            lg = F.get("legs") or [None, None]
            main = (lg[0], lg[1])
            det.append(f"本局剩余 选手 {F.get('you_remaining')} · 对手 {F.get('opponent_remaining')}")
    elif sp == "archery":
        unit = "环"
        if duel:
            main = AB(F.get("totals") or {})
            a, b = AB(F.get("n_x") or {})
            det.append(f"X 环 A {a} · B {b}")
        else:
            L, R = info["players"]["A"], f"达标 ≥{F.get('success_min_total', '?')} 环"
            main = (F.get("total"), None)
            det += [f"已射 {F.get('arrows_shot', 0)}/{(F.get('ends') or 4) * (F.get('arrows_per_end') or 3)} 箭",
                    "各组 " + " / ".join(map(str, F.get("end_scores") or [])) if F.get("end_scores") else ""]
    elif sp == "basketball":
        unit = "中"
        if duel:
            main = AB(F.get("made") or {})
            det.append(f"已投 {F.get('shots_taken', 0)} 球")
        else:
            L, R = info["players"]["A"], f"需命中 ≥{F.get('makes_needed', '?')}"
            main = (F.get("made"), None)
            det.append(f"已投 {F.get('shots_taken', 0)}/{F.get('shots_total', '?')}")
    elif sp in ("bowling", "golf"):
        unit = "分" if sp == "bowling" else "杆"
        S = F.get("scores") or {}
        if duel:
            main = ((S.get("A") or {}).get("total"), (S.get("B") or {}).get("total"))
        else:
            L, R = info["players"]["A"], ("（越高越好）" if sp == "bowling" else "（杆数越少越好）")
            main = ((S.get("A") or {}).get("total"), None)
        for s in (["A", "B"] if duel else ["A"]):
            x = S.get(s) or {}
            if sp == "bowling" and x.get("frames"):
                det.append(f"{s if duel else ''} 各格 " + " ".join("-".join(map(str, f)) for f in x["frames"]))
            if sp == "golf" and x.get("holes"):
                det.append(f"{s if duel else ''} 各洞 " + " ".join(map(str, x["holes"])) + (f"（标准杆 {x.get('par')}）" if x.get("par") else ""))
        if not any(main):
            sh = last.get("shot") or {}
            if isinstance(sh.get("result"), dict):
                r = sh["result"]
                det.append(f"最近：{who(sh.get('seat'), info)} 累计 {r.get('total_so_far')}")
    elif sp == "billiards":
        unit = "剩余球"
        if duel:
            main = AB(F.get("score") or (last.get("review") or {}).get("score") or {})
        else:
            main = (F.get("your_balls_left"), F.get("opponent_balls_left"))
            if F.get("your_group"):
                det.append(f"选手打{'全色球' if F['your_group'] == 'solids' else '花色球'}")
            det.append(f"出杆 选手 {F.get('our_shots', 0)} · 对手 {F.get('opponent_shots', 0)}")
    elif sp == "curling":
        unit = "分"
        if duel:
            main = AB(F.get("score") or {})
            det.append(f"已投 {F.get('n_shots', 0)} 壶")
        else:
            main = (F.get("points_for"), F.get("points_against"))
            det.append(f"已完成 {F.get('n_ends_played', 0)} 局 · 已投 {F.get('throws', 0)} 壶")
    if not main or all(x is None for x in main):     # fallback: the latest event carrying a score
        for e in reversed(list(last.values())):
            if isinstance(e, dict) and "score" in e and isinstance(e["score"], (dict, list)):
                v = e["score"]
                main = AB(v) if isinstance(v, dict) else tuple(v[:2])
                break
    det = [x for x in det if x]
    # state: frozen / turn / clocks
    status = ""
    ph = F.get("phase")
    if F.get("finalized") or ph == "over" or F.get("done") or "end" in last:
        status = "比赛已结束，正在收尾（回放校验 / 评分）"
    elif sp in REALTIME:
        p = (last.get("_phase") or {}).get("ev")
        if ph == "review" or p == "pause_for_review":
            status = "世界冻结：等待复盘（ready）"
        elif p in ("run", "match_start", "game_start") or ph == "running":
            status = "比赛进行中（实时）"
        else:
            status = "等待选手就位"
    else:
        t = last.get("turn")
        if t and duel:
            status = f"轮到 {who(t.get('seat'), info)}"
        elif F.get("started") is False or (isinstance(F.get("started"), dict) and not all(F["started"].values())):
            status = "等待选手就位"
        else:
            status = "比赛进行中（回合制）"
    clocks = clock_text(F, last, info, m.clock_left)
    return dict(L=L, R=R, main=[None if x is None else x for x in (main or (None, None))], unit=unit, det=det,
                status=status, clocks=clocks)


def clock_text(F, last, info, clock_left=None):
    total = F.get("clock_s") or F.get("match_clock_s") or F.get("clock_budget_s")
    if info["duel"]:
        used = F.get("clock_used_s")
        if not isinstance(used, dict) and isinstance(F.get("seat_stats"), dict):
            used = {s: v.get("clock_used_s") for s, v in F["seat_stats"].items()}
        if isinstance(used, dict) and total:
            return " · ".join(f"{s} 剩余 {mmss(total - (used.get(s) or 0))}" for s in ("A", "B") if s in used)
        left = clock_left or {}
        if left:
            return " · ".join(f"{s} 剩余 {mmss(v)}（最近一次）" for s, v in sorted(left.items()))
        return ""
    if F.get("clock_left_s") is not None:
        return f"思考时钟剩余 {mmss(F['clock_left_s'])}"
    used = F.get("clock_used_s")
    if isinstance(F.get("seat_stats"), dict):
        used = (F["seat_stats"].get("A") or {}).get("clock_used_s")
    if total and isinstance(used, (int, float)):
        return f"思考时钟剩余 {mmss(total - used)}"
    return ""


def snapshot():
    now = time.time()
    with LOCK:
        ms = list(STATE["matches"].values())
        hosts = dict(STATE["hosts"])
        errs = dict(STATE["errors"])
    cards = []
    for m in sorted(ms, key=lambda m: (m.info["sport"] not in REALTIME, m.info["sport"], m.proj)):
        info = m.info
        seats = []
        for s, st in sorted(m.seat.items()):
            seats.append(dict(
                seat=s, label=(f"{s} · {info['players'].get(s, s)}" if info["duel"] else info["players"]["A"]),
                img=(f"/img/{m.id}/{s}?v={st.thumb_v}" if st.thumb else ""),
                img_name=Path(st.thumb_name).name if st.thumb else "",
                img_age=(now - st.thumb_mtime) if st.thumb else None,
                steps=[dict(k=k, t=t) for k, t, _ in list(st.steps)[-3:]][::-1],
                h_age=(now - st.h_mtime) if st.h_mtime else None,
                h_kind={"claude": "Claude Code", "codex": "Codex"}.get(st.h_kind, ""),
                plog=st.plog, plog_name=Path(st.plog_name).name if st.plog_name else "",
                plog_age=(now - st.plog_mtime) if st.plog_mtime else None))
        try:
            summ = summarize(m)
        except Exception as ex:  # noqa: BLE001
            summ = dict(L="", R="", main=[None, None], unit="", det=[f"解析出错：{ex}"], status="", clocks="")
        cards.append(dict(
            id=m.id, sport=info["sport"], sport_cn=info["sport_cn"], duel=info["duel"], realtime=info["sport"] in REALTIME,
            players=info["players"], cfg=info["cfg"], host=short(m.host), job=m.job,
            elapsed=(now - info["start"]) if info.get("start") else None,
            updated_age=(now - m.updated) if m.updated else None, err=m.err, summary=summ,
            events=[dict(t=bj(ts), c=c, x=x) for ts, c, x in list(m.events)[-12:]][::-1], seats=seats))
    return dict(now=bj(now, "%Y-%m-%d %H:%M:%S"), hosts=hosts, errors=errs, cards=cards,
                discovered_age=(now - STATE["discovered"]) if STATE["discovered"] else None)


# ---------------------------------------------------------------- poller
def poller(hosts):
    pool = ThreadPoolExecutor(max_workers=10)
    last_disc, last_poll = 0.0, 0.0

    def work(m):
        try:
            poll_sim(m)
            for s, st in m.seat.items():
                poll_seat(m, s, st)
        except Exception as ex:  # noqa: BLE001
            m.err = f"读取出错：{ex}"
        finally:
            m.busy = False

    while True:
        time.sleep(0.5)
        now = time.time()
        if now - STATE["last_view"] > IDLE_S:
            continue                                   # nobody is watching: read nothing
        if now - last_disc >= DISCOVER_S:
            last_disc = now
            try:
                found, counts, errs = discover(hosts)
            except Exception as ex:  # noqa: BLE001
                found, counts, errs = None, {}, {"本机": str(ex)}
            if found is not None:
                with LOCK:
                    cur = STATE["matches"]
                    STATE["matches"] = {k: cur.get(k) or Match(v) for k, v in found.items()}
                    STATE["hosts"], STATE["errors"], STATE["discovered"] = counts, errs, time.time()
        if now - last_poll >= POLL_S:
            last_poll = now
            with LOCK:
                ms = list(STATE["matches"].values())
            for m in ms:
                if not m.busy:                         # a slow read is never stacked: one read per container at a time
                    m.busy = True
                    pool.submit(work, m)


# ---------------------------------------------------------------- page
PAGE = r"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>RoboOlympics 直播看板</title>
<style>
:root{--bg:#f6f5f1;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2dfd7;--card:#fff;--soft:#f0eee8;--acc:#2d5fa8;--ok:#2f7d4f;
--bad:#b3412f;--warn:#a26a00;--rev:#6b4fa8;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#141413;--fg:#ecebe6;--mut:#a3a19a;--line:#33322e;
--card:#1d1d1b;--soft:#262623;--acc:#7ea7e8;--ok:#5fbf86;--bad:#e2765f;--warn:#e0b050;--rev:#b9a2ef;color-scheme:dark}}
:root[data-theme=dark]{--bg:#141413;--fg:#ecebe6;--mut:#a3a19a;--line:#33322e;--card:#1d1d1b;--soft:#262623;--acc:#7ea7e8;
--ok:#5fbf86;--bad:#e2765f;--warn:#e0b050;--rev:#b9a2ef;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:0;padding:12px 16px 40px}
a{color:var(--acc)} .mut{color:var(--mut)} main{max-width:1500px;margin:0 auto}
header.top{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;margin-bottom:12px}
h1{font-size:21px;margin:0} .pill{display:inline-block;padding:1px 8px;border-radius:10px;background:var(--soft);margin-right:6px;font-size:13px}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--bad);margin-right:6px;animation:b 1.6s infinite}
@keyframes b{50%{opacity:.25}}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,640px),1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.hd{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 10px;align-items:baseline}
.hd b{font-size:17px} .who{font-size:15px;font-weight:600}
.score{display:flex;align-items:center;gap:12px;margin:10px 0 4px;flex-wrap:wrap}
.side{flex:1 1 120px;min-width:0;font-size:13px;color:var(--mut);overflow-wrap:anywhere} .side.r{text-align:right}
.num{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:1px;white-space:nowrap}
.num small{font-size:13px;font-weight:400;color:var(--mut);margin-left:4px}
.st{font-size:13px;margin:2px 0} .st .frz{color:var(--warn);font-weight:600} .st .end{color:var(--ok);font-weight:600}
.det{font-size:12.5px;color:var(--mut)}
h3{font-size:13px;color:var(--mut);margin:12px 0 4px;font-weight:600}
ul.ev{list-style:none;margin:0;padding:0;font-size:12.5px;max-height:300px;overflow-y:auto;border-top:1px solid var(--line)}
ul.ev li{padding:3px 0;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
ul.ev .t{color:var(--mut);font-variant-numeric:tabular-nums;margin-right:6px}
.pt{color:var(--acc)} .rev{color:var(--rev)} .bad{color:var(--bad)} .end{color:var(--ok);font-weight:600} .sys{color:var(--mut)}
.seats{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:10px}
.seat{background:var(--soft);border-radius:8px;padding:8px;min-width:0}
.seat img{width:100%;height:auto;max-height:340px;object-fit:contain;border-radius:6px;background:#000;display:block}
.noimg{padding:24px 8px;text-align:center;color:var(--mut);font-size:12px;border:1px dashed var(--line);border-radius:6px}
.step{font-size:12.5px;margin-top:6px;overflow-wrap:anywhere} .step b{color:var(--acc);font-weight:600}
.step.old{opacity:.65}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11.5px;background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:6px;margin:4px 0 0;max-height:180px;overflow-y:auto}
details summary{cursor:pointer;font-size:12.5px;color:var(--mut);margin-top:6px}
.empty{background:var(--card);border:1px dashed var(--line);border-radius:10px;padding:40px 16px;text-align:center;color:var(--mut)}
.bcv{display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.bcv img,.bcv video{width:100%;aspect-ratio:16/9;background:#000;border-radius:6px;display:block;object-fit:contain}
.bcv .mut{font-size:11.5px;margin-top:3px;overflow-wrap:anywhere}
.tbtn{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:1px 8px;font-size:12px;cursor:pointer}
</style></head><body><main>
<header class=top><h1><span class=live></span>RoboOlympics · 直播看板</h1>
<span id=hosts class=mut></span>
<span class=mut><a href="__MON__">实验进度监控</a> · <a href="__SITE__">奥赛网站</a> · <button class=tbtn id=theme>深色/浅色</button></span></header>
<p class="mut" style="margin:0 0 12px;font-size:12.5px">只读观战：只读取容器里的文件（比赛日志、选手保存的图片、模型日志），从不调用比赛服务，不影响比赛。
每 3 秒刷新；时间为北京时间。<span id=upd></span></p>
<div id=cards></div></main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ago=s=>s==null?'':(s<60?Math.round(s)+' 秒前':s<3600?Math.round(s/60)+' 分钟前':(s/3600).toFixed(1)+' 小时前');
const dur=s=>{if(s==null)return '—';s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return (h?h+' 小时 ':'')+m+' 分钟'};
try{const t=localStorage.getItem('lb-theme');if(t)document.documentElement.dataset.theme=t}catch(e){}
$('#theme').onclick=()=>{const d=document.documentElement;const dark=d.dataset.theme?d.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
 d.dataset.theme=dark?'light':'dark';try{localStorage.setItem('lb-theme',d.dataset.theme)}catch(e){}};
const open=new Set();
document.addEventListener('toggle',e=>{if(e.target.id){e.target.open?open.add(e.target.id):open.delete(e.target.id)}},true);
function num(v){return v==null?'–':esc(v)}
function card(c){
 const s=c.summary||{};const m=s.main||[];
 const st=(s.status||'');const cls=st.includes('冻结')?'frz':(st.includes('结束')?'end':'');
 const two=m[1]!=null;
 let h=`<div class=card id="c-${c.id}"><div class=hd><span><b>${esc(c.sport_cn)}</b> · <span class=who>${c.duel?esc(c.players.A)+' 对 '+esc(c.players.B):esc(c.players.A)}</span></span>
 <span class=mut style="font-size:12.5px">${esc(c.cfg)} · ${esc(c.host)} · 已进行 ${dur(c.elapsed)}</span></div>
 <div class=score><div class=side>${esc(s.L)}</div><div class=num>${num(m[0])}${two?' : '+num(m[1]):''}<small>${esc(s.unit)}</small></div>
 <div class="side r">${esc(s.R)}</div></div>
 <div class=st><span class=${cls}>${esc(st)}</span>${s.clocks?' · <span class=mut>'+esc(s.clocks)+'</span>':''}</div>
 <div class=det>${(s.det||[]).map(esc).join(' · ')}</div>`;
 if(c.err)h+=`<div class=bad style="font-size:12px">${esc(c.err)}</div>`;
 if(BC&&c.realtime)h+=bcast(c);
 h+=`<h3>赛事动态</h3><ul class=ev>`+(c.events.length?c.events.map(e=>`<li class=${e.c}><span class=t>${e.t}</span>${esc(e.x)}</li>`).join(''):'<li class=sys>还没有事件</li>')+`</ul>`;
 h+=`<h3>选手视角与模型动态</h3><div class=seats>`;
 for(const p of c.seats){
  const id=`pl-${c.id}-${p.seat}`;
  h+=`<div class=seat><div style="font-weight:600;font-size:13px;margin-bottom:4px">${esc(p.label)}</div>`;
  h+=p.img?`<img src="${p.img}" alt="选手保存的最新图片"><div class=mut style="font-size:11.5px">${esc(p.img_name)} · ${ago(p.img_age)}</div>`
          :`<div class=noimg>选手还没有保存图片</div>`;
  h+=`<div class=mut style="font-size:11.5px;margin-top:6px">${esc(p.h_kind)} 日志${p.h_age!=null?' · 最近更新 '+ago(p.h_age):''}</div>`;
  h+=p.steps.length?p.steps.map((x,i)=>`<div class="step${i?' old':''}"><b>${esc(x.k)}</b> ${esc(i&&x.t.length>160?x.t.slice(0,159)+'…':x.t)}</div>`).join(''):'<div class="step mut">（还没有读到模型的动作）</div>';
  if(p.plog.length)h+=`<details id="${id}"${open.has(id)?' open':''}><summary>程序日志 ${esc(p.plog_name)} · ${ago(p.plog_age)}</summary><pre>${p.plog.map(esc).join('\n')}</pre></details>`;
  h+=`</div>`;
 }
 return h+`</div></div>`;
}
const BC='__BC__';let bcm={};
async function bctick(){if(!BC)return;try{const d=await (await fetch(BC+'/api/matches',{cache:'no-store'})).json();
 const o={};for(const m of d.matches.concat(d.other||[]))o[m.id]=m;bcm=o}catch(e){bcm=null}}
function bcast(c){
 const b=bcm&&bcm[c.id];let h='<h3>实时比赛画面</h3>';
 if(bcm===null)return h+`<div class=noimg>连不上广播服务器 ${esc(BC)}</div>`;
 if(!b)return h+'<div class=noimg>正在接入直播画面…</div>';
 if(!b.supported)return h+'<div class=noimg>这个项目的直播画面还没有接入</div>';
 if(!b.has_log)return h+`<div class=noimg>${esc(b.err||'这场比赛的模拟器没有广播日志（第 2 轮起的版本才有）')}</div>`;
 const n=b.points.length,p=n?b.points[n-1]:null;
 h+=`<div class=bcv><div><img data-keep="s-${c.id}" src="${BC}/stream/${c.id}" alt="直播画面">
  <div class=mut>直播 · 世界慢放 ×${esc(b.slowdown)}，画面是慢动作；在比赛容器外渲染，只读模拟器追加的状态日志</div></div>`;
 h+=n?`<div><video data-keep="r-${c.id}-${n}" src="${BC}/replay/${c.id}/${n}" controls autoplay muted playsinline preload=auto></video>
  <div class=mut>上一分回放（第 ${n} 分，实速）：${esc(p.winner)} 得分 · ${esc(p.how)}</div></div>`
  :'<div class=noimg>第一分结束后这里播放上一分的回放</div>';
 return h+'</div>';
}
let imgs={};
function render(d){
  const hs=Object.entries(d.hosts).map(([h,n])=>`<span class=pill>${esc(h)}：${n} 场</span>`).join('')
   +Object.entries(d.errors).map(([h,e])=>`<span class="pill bad">${esc(h)}：${esc(e)}</span>`).join('');
  $('#hosts').innerHTML=hs||'正在查找比赛…';
  $('#upd').textContent=' 页面更新于 '+d.now+'。';
  if(!d.cards.length){$('#cards').innerHTML=`<div class=empty style="grid-column:1/-1">${d.discovered_age==null?'正在查找正在进行的比赛…':'当前没有比赛在进行。'}</div>`;return}
  const keep={},kept={};document.querySelectorAll('#cards img:not([data-keep])').forEach(i=>keep[i.getAttribute('src')]=i);
  document.querySelectorAll('#cards [data-keep]').forEach(i=>kept[i.dataset.keep]=i);   // live streams / replays keep playing
  $('#cards').innerHTML=d.cards.map(card).join('');
  document.querySelectorAll('#cards img:not([data-keep])').forEach(i=>{const o=keep[i.getAttribute('src')];if(o&&o.complete)i.replaceWith(o)});
  document.querySelectorAll('#cards [data-keep]').forEach(i=>{const o=kept[i.dataset.keep];if(o)i.replaceWith(o)});
}
async function tick(){
 await bctick();
 try{const r=await fetch('/api/state',{cache:'no-store'});render(await r.json())}
 catch(e){$('#upd').textContent=' 连接中断，重试中…'}
}
const INIT=__INIT__;render(INIT);bctick().then(()=>render(INIT));setInterval(tick,3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ctype, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        STATE["last_view"] = time.time()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            init = json.dumps(snapshot(), ensure_ascii=False).replace("</", "<\\/")
            return self.send(200, PAGE.replace("__MON__", MONITOR_URL).replace("__SITE__", SITE_URL)
                             .replace("__BC__", BROADCAST_URL.rstrip("/"))
                             .replace("__INIT__", init).encode(),
                             "text/html; charset=utf-8")
        if path.startswith("/bc/"):           # the broadcast server, relayed so the page needs only this port
            return self.relay(self.path[3:])
        if path == "/api/state":
            return self.send(200, json.dumps(snapshot(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
        mm = re.match(r"^/img/([a-z0-9]+)/([AB])$", path)
        if mm:
            with LOCK:
                m = next((x for x in STATE["matches"].values() if x.id == mm.group(1)), None)
            st = m.seat.get(mm.group(2)) if m else None
            if st and st.thumb:
                return self.send(200, st.thumb, "image/jpeg" if Image else "image/png", "max-age=3600")
        return self.send(404, b"not found", "text/plain")

    def relay(self, rest):
        """Pass a broadcast-server request through (MJPEG streams included), so viewers who can reach only the board's
        port still see the video."""
        import urllib.request
        try:
            up = urllib.request.urlopen(BROADCAST_UPSTREAM.rstrip("/") + rest, timeout=20)
        except Exception as ex:  # noqa: BLE001
            return self.send(502, f"broadcast server unreachable: {ex}".encode(), "text/plain")
        self.send_response(up.status)
        for k in ("Content-Type", "Content-Length", "Cache-Control", "Access-Control-Allow-Origin"):
            if up.headers.get(k):
                self.send_header(k, up.headers[k])
        self.end_headers()
        try:
            while True:
                chunk = up.read1(65536) if hasattr(up, "read1") else up.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            up.close()

    def log_message(self, *a):
        pass


BROADCAST_UPSTREAM = "http://127.0.0.1:8048"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8047)
    ap.add_argument("--hosts", default="mll-4090-2.cs.northwestern.edu")
    ap.add_argument("--broadcast", default="/bc", help="broadcast URL the page uses ('/bc' = relayed by this server; "
                    "'' = no video)")
    ap.add_argument("--broadcast-upstream", default="http://127.0.0.1:8048", help="the broadcast server, for /bc/ relaying")
    a = ap.parse_args()
    globals()["BROADCAST_URL"] = a.broadcast
    globals()["BROADCAST_UPSTREAM"] = a.broadcast_upstream
    threading.Thread(target=poller, args=([h for h in a.hosts.split(",") if h],), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
