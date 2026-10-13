import numpy as np

from latency_bench.data.state_normalization import min_max_normalize_state


def test_min_max_state_normalization_maps_constant_dimensions_to_zero():
    normalized = min_max_normalize_state(
        np.asarray([2.0, 5.0, 10.0], dtype=np.float32),
        np.asarray([0.0, 5.0, 0.0], dtype=np.float32),
        np.asarray([4.0, 5.0, 10.0], dtype=np.float32),
    )

    assert np.array_equal(normalized, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def test_normalization_avoids_intermediate_multiplication_overflow():
    # Long-lived numeric contract shared by generic state and task inputs.
    from latency_bench.integrations.starvla.task_contracts import normalize

    values = np.asarray([[2e38, 4.0], [0.0, 4.0]], dtype=np.float32)
    minimum = np.asarray([0.0, 4.0], dtype=np.float32)
    maximum = np.asarray([3e38, 4.0], dtype=np.float32)
    expected = [[1.0 / 3.0, 0.0], [-1.0, 0.0]]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        generic = min_max_normalize_state(values, minimum, maximum)
        task = normalize(values, {"min": minimum, "max": maximum})
    assert generic.dtype == task.dtype == np.float32
    np.testing.assert_allclose(generic, expected, atol=1e-7)
    np.testing.assert_array_equal(task, generic)
