import numpy as np

from latency_bench.data.state_normalization import min_max_normalize_state


def test_min_max_state_normalization_maps_constant_dimensions_to_zero():
    normalized = min_max_normalize_state(
        np.asarray([2.0, 5.0, 10.0], dtype=np.float32),
        np.asarray([0.0, 5.0, 0.0], dtype=np.float32),
        np.asarray([4.0, 5.0, 10.0], dtype=np.float32),
    )

    assert np.array_equal(normalized, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
