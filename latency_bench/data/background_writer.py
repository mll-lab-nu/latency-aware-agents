"""Bounded background I/O shared by image and episode collectors."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


class BackgroundWriter:
    """Own single-producer writes, apply backpressure, and propagate worker failures."""

    def __init__(self, *, max_workers=1, max_pending=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending = deque()
        self.max_pending = max_pending

    def poll(self):
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()

    def submit(self, function, *args, **kwargs):
        self.poll()
        if len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(function, *args, **kwargs))

    def flush(self):
        while self.pending:
            self.pending.popleft().result()

    def close(self):
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)
