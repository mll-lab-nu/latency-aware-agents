#!/usr/bin/env python3
"""Live monitor of the Olympiad experiments: what has been played, what is running, what is left.

    python3 scripts/olympiad/monitor.py [--port 8046] [--hosts mll-4090-2.cs.northwestern.edu] [--results DIR]

Every 60 s it copies the result files (<results>/<sport>-round<r>.jsonl) of the other hosts into a local mirror and
lists the matches running on every host (their run_single.sh / run_duel.sh / run_team.sh / run_race.sh processes). The
page (Chinese) is rebuilt on each request: progress per sport and round against the plan, the matches running now, and
the latest results. The plan (planned()): rounds 1-3 of run_round.py, i.e. of the sports' declarations (plan.py:
sports/<sport>/sport.toml and tasks/), for every published sport with run_round's default players; a sport with a
head-to-head format plays its duels, a sport of solo formats only (a score, no interaction) its singles (Season 1,
plan A). A sport whose plan cannot be made shows the error on the page. Nothing here changes an experiment.
"""
import argparse
import calendar
import html
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_round  # noqa: E402

plan = run_round.plan
# Season 1 (decided 2026-10-04, plan A): every sport with all the players; the head-to-head sports play a round robin of
# duels (both seat orders), the scoring sports (no interaction between players) play singles only.
SPORTS = plan.sports()                                   # sports/*/sport.toml, published (tasks/)
PLAN = {s: list(run_round.PLAYERS) for s in SPORTS}
SCORING = {s for s in SPORTS if all(t["kind"] == "single" for t in plan.tasks(s))}   # singles only; the others: duels
# Chinese labels for the page (display only; a sport without one shows its sport.toml title)
LABELS = {"tabletennis": "乒乓球", "tennis": "网球", "badminton": "羽毛球", "baseball": "棒球", "darts": "飞镖",
          "archery": "射箭", "basketball": "篮球投篮", "bowling": "保龄球", "golf": "高尔夫推杆", "billiards": "台球（八球）",
          "curling": "冰壶", "soccer": "足球（2v2）", "racing": "3D 赛车", "airhockey": "空气曲棍球", "fencing": "击剑",
          "foosball": "桌上足球", "drone": "无人机竞速", "discgolf": "飞盘高尔夫"}
NAMES = {s: LABELS.get(s) or plan.decl(s)["sport"].get("title", s) for s in SPORTS}
MODEL = {"opus": "Opus 5.5", "fable": "Fable 5.1", "astra": "Astra", "sol": "SOL", "luna": "luna", "oracle": "oracle",
         "nop": "nop"}
ROUNDS = (1, 2, 3)
RESULTS = pathlib.Path("/data/guichi/olympiad_results")
MIRROR = pathlib.Path("/data/guichi/olympiad_results_mirror")
STATE = {"running": {}, "updated": 0, "errors": []}
BJT = 8 * 3600          # times on the page are Beijing time (UTC+8)


def bj(ts, fmt):
    return time.strftime(fmt, time.gmtime(ts + BJT))


def planned(sport, r):
    """The matches of a sport and round in the plan (run_round.matches at each task's default slowdown); raises when
    the plan cannot be made (the page shows the error instead of an empty plan)."""
    kind = "single" if sport in SCORING else "duel"
    return [m for m in run_round.matches(r, sport, None, PLAN[sport]) if m["kind"] == kind]


key = run_round.key


def records(sport, r):
    """All attempts of a sport and round, from this host and the mirrored hosts."""
    out = []
    for d in [RESULTS] + sorted(MIRROR.glob("*")):
        f = d / f"{sport}-round{r}.jsonl"
        if f.exists():
            host = "4090-1" if d == RESULTS else d.name.split(".")[0].replace("mll-", "")
            for line in f.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                rec["host"] = host
                out.append(rec)
    return out


def poll(hosts):
    while True:
        errs, running = [], {}
        for h in hosts:
            dst = MIRROR / h
            dst.mkdir(parents=True, exist_ok=True)
            p = subprocess.run(["rsync", "-a", "--include=*-round*.jsonl", "--exclude=*", f"guichi@{h}:{RESULTS}/", f"{dst}/"],
                               capture_output=True, text=True, timeout=60)
            if p.returncode:
                errs.append(f"{h}: rsync {p.stderr.strip()[:200]}")
        cmd = "ps -eo etimes,args | grep -E 'run_(single|duel|team|race)\\.sh --task' | grep -v grep"
        for h in ["local"] + hosts:
            try:
                out = subprocess.run(cmd if h == "local" else ["ssh", "-o", "BatchMode=yes", f"guichi@{h}", cmd],
                                     shell=(h == "local"), capture_output=True, text=True, timeout=30).stdout
            except subprocess.TimeoutExpired:
                errs.append(f"{h}: ps timed out")
                continue
            name = "4090-1" if h == "local" else h.split(".")[0].replace("mll-", "")
            running[name] = [ln.strip() for ln in out.splitlines() if ln.strip()]
        STATE.update(running=running, updated=time.time(), errors=errs)
        time.sleep(60)


def parse_run(line):
    m = re.match(r"(\d+)\s+/bin/bash scripts/olympiad/run_(single|duel|team|race)\.sh (.*)", line)
    if not m:
        return None
    secs, kind, args = int(m.group(1)), m.group(2), m.group(3).split()
    a = {args[i][2:]: args[i + 1] for i in range(0, len(args) - 1) if args[i].startswith("--")}
    sport = a.get("task", "").split("-")[0]
    who = (MODEL.get(a.get("agent"), a.get("agent")) if kind == "single"
           else " / ".join(MODEL.get(p, p) for p in a.get("players", "").split(",")) if kind == "race"
           else f"{MODEL.get(a.get('a'), a.get('a'))} 对 {MODEL.get(a.get('b'), a.get('b'))}")
    cfg = ((f"L{a['level']}" if "level" in a else "单人") if kind == "single" else
           {"duel": "对打", "team": "团体", "race": "多车赛"}[kind]) + \
        (f" ×{a['slowdown']}" if "slowdown" in a else "") + f" 种子 {a.get('seed')}"
    return dict(sport=NAMES.get(sport, sport), who=who, cfg=cfg, tag=a.get("tag", ""), minutes=secs // 60)


def score(rec):
    w = rec.get("reward") or {}
    if rec["kind"] == "duel":
        a, b = w.get("A_points"), w.get("B_points")
        if a is None:
            for k in ("A_legs", "A_total", "A_score", "A_bases", "A_balls_left", "A_strokes", "A_made"):
                if k in w:
                    a, b = w[k], w.get("B" + k[1:])
                    break
        won = "A 胜" if w.get("A_won") else ("B 胜" if w.get("B_won") else "")
        return f"{a}:{b} {won}".strip() if a is not None else won
    for x, y in (("points_for", "points_against"), ("you_legs", "opponent_legs"), ("your_score", "opponent_score")):
        if x in w:
            return f"{w[x]}:{w.get(y)}" + (" 胜" if w.get("success") else " 负")
    for k in ("bases", "total", "made", "score", "strokes", "pins"):
        if k in w:
            return f"{k} {w[k]}" + (" 达标" if w.get("success") else " 未达标")
    return "成功" if w.get("success") else "未成功"


def who(rec):
    if rec["kind"] == "single":
        lv = f" L{rec['level']}" if rec.get("level") is not None else ""
        return f"{MODEL.get(rec.get('player'), rec.get('player'))}{lv}"
    if rec["kind"] == "race":
        return " / ".join(MODEL.get(p, p) for p in rec.get("players", []))
    return f"{MODEL.get(rec.get('a'), rec.get('a'))}(A) 对 {MODEL.get(rec.get('b'), rec.get('b'))}(B)"


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2dfd7;--ok:#2f7d4f;--bad:#b3412f;--run:#2d5fa8;--pend:#a7a39a;--card:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#a3a19a;--line:#33322e;--ok:#5fbf86;--bad:#e2765f;
--run:#7ea7e8;--pend:#6d6a63;--card:#1f1f1c}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif;margin:0;padding:16px}
main{max-width:1100px;margin:0 auto} h1{font-size:22px;margin:4px 0} h2{font-size:17px;margin:22px 0 8px}
.mut{color:var(--mut)} .wrap{overflow-x:auto} table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}
th{font-weight:600;color:var(--mut)} .bar{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--line);min-width:160px}
.bar i{display:block;height:100%} .ok{background:var(--ok)} .bad{background:var(--bad)} .run{background:var(--run)}
.tok{color:var(--ok)} .tbad{color:var(--bad)} .trun{color:var(--run)} .legend span{margin-right:14px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle}
"""


def page():
    now = time.time()
    running = STATE["running"]
    run_rows, run_keys = [], set()
    for host, lines in running.items():
        for ln in lines:
            r = parse_run(ln)
            if r:
                run_rows.append((host, r))
    prog_rows, total = [], dict(plan=0, valid=0, failed=0)
    latest = []
    plan_errors = []
    for sport in PLAN:
        for r in ROUNDS:
            try:
                ms = planned(sport, r)
            except Exception as e:  # noqa: BLE001  (shown on the page, never an empty plan in silence)
                plan_errors.append(f"{NAMES[sport]} 第 {r} 轮计划出错：{type(e).__name__}: {e}")
                continue
            if not ms:
                continue
            recs = records(sport, r)
            latest += [dict(rec, sport=sport, round=r) for rec in recs]
            valid = {key(x) for x in recs if (x.get("reward") or {}).get("valid_trial") == 1}
            tried = {key(x) for x in recs}
            n, v = len(ms), sum(key(m) in valid for m in ms)
            f = sum(key(m) in tried and key(m) not in valid for m in ms)
            total["plan"] += n; total["valid"] += v; total["failed"] += f
            if r > 1 and not tried:
                continue
            prog_rows.append((NAMES[sport], r, n, v, f, ", ".join(MODEL.get(p, p) for p in PLAN[sport])))
    latest.sort(key=lambda x: x.get("job", "")[-16:], reverse=True)
    h = [f"<!doctype html><html lang=zh><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<meta http-equiv=refresh content=60><title>RoboOlympics Live</title><style>{CSS}</style></head><body><main>",
         "<h1>RoboOlympics · 实验实时监控</h1>",
         f"<p class=mut>每 60 秒刷新；数据更新于 {bj(STATE['updated'] or now, '%Y-%m-%d %H:%M:%S')}（北京时间）。"
         f"计划共 {total['plan']} 场（3 轮），已完成有效 {total['valid']} 场，失败待重跑 {total['failed']} 场，"
         f"正在进行 {len(run_rows)} 场。</p>"]
    if STATE["errors"] or plan_errors:
        h.append("<p class=tbad>" + "；".join(html.escape(e) for e in STATE["errors"] + plan_errors) + "</p>")
    h.append("<h2>正在进行</h2><div class=wrap><table><tr><th>机器</th><th>项目</th><th>选手</th><th>配置</th><th>已进行</th></tr>")
    for host, r in sorted(run_rows, key=lambda x: -x[1]["minutes"]):
        h.append(f"<tr><td>{host}</td><td>{r['sport']}</td><td>{html.escape(r['who'])}</td><td>{r['cfg']}"
                 f"{' <span class=mut>(' + r['tag'] + ')</span>' if r['tag'] not in ('r1', 'r2', 'r3') else ''}</td>"
                 f"<td class=trun>{r['minutes']} 分钟</td></tr>")
    if not run_rows:
        h.append("<tr><td colspan=5 class=mut>当前没有比赛在进行</td></tr>")
    h.append("</table></div>")
    h.append("<h2>进度（按项目和轮次）</h2><p class=legend mut><span><i class='dot ok'></i>有效完成</span>"
             "<span><i class='dot bad'></i>失败待重跑</span><span><i class='dot' style='background:var(--line)'></i>未开始</span></p>"
             "<div class=wrap><table><tr><th>项目</th><th>轮</th><th>参赛模型</th><th>有效 / 计划</th><th>进度</th></tr>")
    for name, r, n, v, f, players in prog_rows:
        h.append(f"<tr><td>{name}</td><td>第 {r} 轮</td><td class=mut>{players}</td><td>{v} / {n}"
                 f"{f' <span class=tbad>（失败 {f}）</span>' if f else ''}</td><td><div class=bar>"
                 f"<i class=ok style='width:{100 * v / n:.1f}%'></i><i class=bad style='width:{100 * f / n:.1f}%'></i></div></td></tr>")
    h.append("</table></div>")
    h.append("<h2>最近的比赛结果</h2><div class=wrap><table><tr><th>开始时间（北京时间）</th><th>项目</th><th>轮</th><th>选手</th>"
             "<th>种子</th><th>结果</th><th>有效</th><th>用时</th><th>机器</th></tr>")
    for x in latest[:80]:
        w = x.get("reward") or {}
        t = re.search(r"(\d{8}T\d{6})Z", x.get("job", ""))
        ts = bj(calendar.timegm(time.strptime(t.group(1), "%Y%m%dT%H%M%S")), "%m-%d %H:%M") if t else "—"
        ok = w.get("valid_trial") == 1
        why = "" if ok else ("模型额度用完" if w.get("proxy_error") else ("技术故障" if w else "没有结果"))
        h.append(f"<tr><td>{ts}</td><td>{NAMES.get(x['sport'], x['sport'])}</td><td>{x['round']}</td><td>{html.escape(who(x))}</td>"
                 f"<td>{x.get('seed')}</td><td>{html.escape(score(x)) if ok else '—'}</td>"
                 f"<td class={'tok' if ok else 'tbad'}>{'有效' if ok else '无效 · ' + why}</td>"
                 f"<td>{x.get('minutes')} 分钟</td><td class=mut>{x['host']}</td></tr>")
    h.append("</table></div><p class=mut>规则与任务说明见 <a href='http://mll-4090-1.cs.northwestern.edu:8042/agent-sports-site/'>奥赛网站</a>。"
             "单人赛结果为 选手:对手；对打为 A:B。</p></main></body></html>")
    return "".join(h)


class H(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    global RESULTS, MIRROR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8046)
    ap.add_argument("--hosts", default="mll-4090-2.cs.northwestern.edu")
    ap.add_argument("--results", default=str(RESULTS), help="this host's result files (run_round.py --results)")
    ap.add_argument("--mirror", default=str(MIRROR), help="the local copy of the other hosts' result files")
    a = ap.parse_args()
    RESULTS, MIRROR = pathlib.Path(a.results), pathlib.Path(a.mirror)
    threading.Thread(target=poll, args=([h for h in a.hosts.split(",") if h],), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
