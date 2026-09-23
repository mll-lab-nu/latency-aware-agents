"""Wake the existing Codex task for failures and completed datasets/evaluations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time

from latency_bench.utils.io import write_json_atomic


HOSTS = {
    "h200-1": ("34.131.223.74", "/home/ubuntu/codex/hist8-vla-v2-20260923"),
    "h200-2": ("34.131.91.104", "/home/ubuntu/codex/hist8-vla-v2-20260923"),
    "h100": ("34.6.91.253", "/mnt/ssd/codex/hist8-vla-v2-20260923"),
}
REMOTE_STATUS = """
import json,pathlib,time
p=pathlib.Path('outputs/tasks/decision_history_vla_hist8_v2')
q=p/'queue'; state=json.loads((q/'state.json').read_text())
pid=int((q/'controller.pid').read_text())
proc=pathlib.Path(f'/proc/{pid}/cmdline')
controller_alive=proc.exists() and str((q/'manifest.json').resolve()).encode() in proc.read_bytes().split(b'\\0')
for name,s in state.items():
 if s['status']=='running':
  job=q/'jobs'/name/f"attempt_{s['attempt']}"/'job.json'
  proc=pathlib.Path(f"/proc/{s['pid']}/cmdline")
  s['process_alive']=proc.exists() and str(job.resolve()).encode() in proc.read_bytes().split(b'\\0')
  s['log_age_seconds']=time.time()-(job.parent/'output.log').stat().st_mtime
print(json.dumps({'jobs':state,'controller_pid':pid,'controller_alive':controller_alive}))
"""


def collect_host(name: str) -> tuple[str, dict]:
    ip, root = HOSTS[name]
    command = f"cd {shlex.quote(root)} && python3 -c {shlex.quote(REMOTE_STATUS)}"
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", f"ubuntu@{ip}", command],
            capture_output=True, text=True, timeout=45,
        )
    except subprocess.TimeoutExpired:
        return name, {"error": "SSH status query exceeded 45 seconds"}
    if result.returncode:
        return name, {"error": result.stderr[-2000:]}
    try:
        return name, json.loads(result.stdout)
    except json.JSONDecodeError:
        return name, {"error": "Invalid status response: " + result.stdout[-2000:]}


def pending_events(snapshots: dict, notified: set[str]) -> list[dict]:
    events = []
    for host, snapshot in snapshots.items():
        if "error" in snapshot:
            events.append({"id": f"connection_failed/{host}", "detail": f"{host}: {snapshot['error']}"})
            continue
        jobs = snapshot["jobs"]
        if not snapshot["controller_alive"] and any(s["status"] in {"pending", "running"} for s in jobs.values()):
            events.append({"id": f"controller_stopped/{host}/{snapshot['controller_pid']}",
                           "detail": f"{host} controller stopped with unfinished jobs"})
        for job_id, state in jobs.items():
            identity = f"{host}/{job_id}/{state['attempt']}"
            if state["status"] in {"failed", "needs_diagnosis"}:
                events.append({"id": f"failed/{identity}", "detail": f"{identity}: {state['status']}"})
            elif state["status"] == "running":
                if not state["process_alive"] or state["log_age_seconds"] > 1800:
                    events.append({"id": f"inspect_running/{identity}",
                                   "detail": f"{identity}: live={state['process_alive']}, log age={state['log_age_seconds']:.0f}s"})
            elif state["status"] == "completed" and job_id.startswith(("eval_", "materialize_zero__")):
                events.append({"id": f"evaluation_completed/{identity}", "detail": f"{identity}: evaluation or dataset conversion completed"})
    return [event for event in events if event["id"] not in notified]


def watch(output: Path, thread_id: str, codex: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    notified = set(json.loads(state_path.read_text())["notified"]) if state_path.exists() else set()
    while True:
        with ThreadPoolExecutor(max_workers=3) as pool:
            snapshots = dict(pool.map(collect_host, HOSTS))
        events = pending_events(snapshots, notified)
        if events:
            prompt = (
                "【hist8 自动监控唤醒】这是用户要求的当前任务监控。请处理以下已发生事件：\n"
                + "\n".join(event["detail"] for event in events)
                + "\n先读取 program/DECISION_HISTORY_VLA_H8.md 和 program/logs/decision-history-vla-v2-2026-09-23.md。"
                "实时核对服务器状态。当前目标是九路 profile 与九路 zero 训练，以及 zero→zero、zero→profile、"
                "profile→profile 共 27 组 100 局评测；parallel 32，Flappy/Demon/Deadly 上限分别为 "
                "3600/7200/3600 原始帧。每 rank workers 16，prefetch 4，每卡 batch 64，global 128，"
                "底座 LR 统一为已确定的 3e-6。两台 H200 优先双卡训练，常规评测整对占用释放；"
                "第 1 台 GPU 4 为另一任务占用时 GPU 7 可供独立采集/评测。"
                "核验进程和实际训练更新，不重复启动。失败先定位再按授权修复；已完成评测核对 100 seeds、SHA、"
                "动作分布并更新汇总。未发生可行动变化时保持安静；只报告结论、失败、完成或需要用户处理的事项。"
            )
            result = subprocess.run([codex, "queue", "--thread", thread_id, "--message", prompt],
                                    check=True, capture_output=True, text=True, timeout=30)
            receipt = {"time": time.time(), "events": events, "queue_receipt": result.stdout.strip()}
            with (output / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(receipt) + "\n")
            notified.update(event["id"] for event in events)
            print(json.dumps(receipt), flush=True)
        complete = all("error" not in snapshot and all(s["status"] == "completed" for s in snapshot["jobs"].values())
                       for snapshot in snapshots.values())
        write_json_atomic(state_path, {"last_checked": time.time(), "thread_id": thread_id,
                               "status": "completed" if complete else "watching",
                               "notified": sorted(notified), "hosts": snapshots})
        if complete:
            return
        time.sleep(1200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--codex", required=True)
    args = parser.parse_args()
    watch(args.output_dir, args.thread_id, args.codex)


if __name__ == "__main__":
    main()
