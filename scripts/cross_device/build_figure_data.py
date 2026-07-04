from __future__ import annotations

import sys

from latency_bench.analysis import cross_device_figure_data as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
