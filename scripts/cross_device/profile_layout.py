from __future__ import annotations

import sys

from latency_bench.cross_device import layout as _impl

sys.modules[__name__] = _impl
