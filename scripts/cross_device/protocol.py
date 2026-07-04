from __future__ import annotations

import sys

from latency_bench.cross_device import protocol as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
