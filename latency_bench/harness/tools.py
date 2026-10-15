"""Run information tools with public snapshots and delayed result visibility."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time


class ToolTasks:
    """Own tool futures. Tools have no environment or action-submission handle."""

    def __init__(self, functions, *, clock, mode):
        self.functions = functions
        self.clock = clock
        self.mode = mode
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = []
        self.next_id = 0

    def submit(self, request, snapshot, frame):
        function = self.functions[request["name"]]
        task_id = self.next_id
        self.next_id += 1
        started = time.perf_counter()
        future = self.pool.submit(self._run, function, deepcopy(snapshot), deepcopy(request["input"]), started)
        self.pending.append((task_id, request["name"], frame, future))
        return task_id

    @staticmethod
    def _run(function, snapshot, arguments, started):
        value = function(snapshot, arguments)
        return value, (time.perf_counter() - started) * 1000

    def poll(self, frame):
        visible = []
        remaining = []
        for task_id, name, source_frame, future in self.pending:
            if self.mode == "realtime" and not future.done():
                remaining.append((task_id, name, source_frame, future))
                continue
            value, wall_ms = future.result()
            ready = source_frame + self.clock.latency_ms_to_raw_frames(wall_ms)
            if self.mode == "simulated" and ready > frame:
                remaining.append((task_id, name, source_frame, future))
                continue
            visible.append({"task_id": task_id, "name": name, "source_frame": source_frame,
                            "visible_frame": ready if self.mode == "simulated" else frame,
                            "wall_ms": wall_ms, "result": value})
        self.pending = remaining
        return visible

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def discard_pending(self):
        task_ids = []
        for task_id, _, _, future in self.pending:
            future.cancel()
            task_ids.append(task_id)
        self.pending.clear()
        return task_ids
