from __future__ import annotations

from collections import deque

from latency_bench.core.types import Observation


class HistoryBuffer:
    def __init__(self, max_len: int = 16):
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.max_len = int(max_len)
        self._items: deque[Observation] = deque(maxlen=self.max_len)

    def append(self, observation: Observation) -> None:
        self._items.append(observation)

    def last(self) -> Observation | None:
        if not self._items:
            return None
        return self._items[-1]

    def last_k(self, k: int) -> list[Observation]:
        if k <= 0:
            return []
        return list(self._items)[-int(k):]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

