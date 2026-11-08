import numpy as np

from scripts.experiments.temporal_profiles.compare import burst_lengths, describe, sample_blocks, session_acf
from scripts.experiments.temporal_profiles.scores import bootstrap_indices, evaluate_windows, interval, timing_diagnostics, window_scores, windows
from scripts.experiments.temporal_profiles.study import inner_split


def test_slow_spells_do_not_bridge_sessions():
    first, second = np.array([1., 5., 5.]), np.array([5., 1., 5.])
    actual = np.concatenate([burst_lengths(x, 3.) for x in (first, second)])
    np.testing.assert_array_equal(actual, [2, 1, 1])


def test_acf_centers_sessions_and_constant_is_undefined():
    values = np.tile([1., -1.], 100)
    result = session_acf([values + 50, values - 50], 2)
    np.testing.assert_allclose(result, [-1., 1.])
    assert np.isnan(session_acf([np.full(100, 7.)])).all()


def test_conditional_slow_probability_excludes_session_boundary():
    arrays = [np.r_[np.ones(99), 5.], np.r_[5., np.ones(99)]]
    metrics, _, _ = describe(arrays, arrays, 3.)
    assert metrics["next_slow_given_slow"] == 0


def test_blocks_preserve_joint_pairs_and_are_seeded():
    sources = [np.array([[i, i - .5] for i in range(1, 20)]),
               np.array([[i, i - .5] for i in range(100, 120)])]
    result = sample_blocks(sources, 500, 42, 32)
    np.testing.assert_array_equal(result, sample_blocks(sources, 500, 42, 32))
    assert set(map(tuple, result)) <= set(map(tuple, np.concatenate(sources)))
    np.testing.assert_allclose(result[:, 0] - result[:, 1], .5)


def test_inner_split_uses_only_given_fitting_runs():
    training, tuning = inner_split(list(range(25)))
    assert len(training) == 20 and len(tuning) == 5
    assert set(training).isdisjoint(tuning)
    assert set(training + tuning) == set(range(25))
    assert (training, tuning) == inner_split(list(range(25)))


def test_windows_preserve_order_and_cannot_wrap():
    actual = windows(np.arange(100), 8, 20, 42)
    assert actual.shape == (20, 8)
    assert np.all(np.diff(actual, axis=1) == 1)
    assert actual.min() >= 0 and actual.max() < 100


def test_energy_and_fair_variogram_match_hand_calculation():
    scores = window_scores(np.array([[0., 2.]]), np.zeros((2, 2)), 0., 1.)
    np.testing.assert_allclose(scores["energy"], np.sqrt(2))
    np.testing.assert_allclose(scores["variogram"], 2.)
    corrected = window_scores(np.array([[0., 1.]]), np.array([[0., 0.], [0., 4.]]), 0., 1.)
    np.testing.assert_allclose(corrected["variogram"], -1.)


def test_variogram_detects_order_with_same_marginal_values():
    observed = np.array([[0., 0., 1., 1.]])
    matching = window_scores(observed, np.repeat(observed, 5, axis=0), 0., 1.)
    reordered = window_scores(observed, np.tile([0., 1., 0., 1.], (5, 1)), 0., 1.)
    assert matching["variogram"] == 0
    assert reordered["variogram"] > 0


def test_paired_bootstrap_retains_exact_difference():
    a = np.arange(30).reshape(3, 10) / 10
    result = interval((a + .3) - a, bootstrap_indices(3, 10, draws=100))
    np.testing.assert_allclose([result["mean"], *result["ci95"]], .3)


def test_deadline_boundary_and_streak_are_physical_frames():
    values = np.full(256, 100.)
    values[:4] = [100., 100.1, 200., 200.1]
    result = timing_diagnostics(values, values, 110., 100.)
    assert result["miss_1_frame"] == 3 / 256
    assert result["miss_2_frames"] == 1 / 256
    assert result["max_miss_streak"] == 3


def test_short_sessions_remain_valid_without_fabricating_long_windows():
    values = np.arange(26, dtype=float) + 1
    diagnostics = timing_diagnostics(values, values, 24., 20.)
    assert diagnostics["acf_error"] == 0
    assert diagnostics["slow_count_error_8"] == 0
    assert diagnostics["slow_count_error_32"] is None
    assert diagnostics["slow_count_error_128"] is None
    session = {"run_id": 3, "pairs": np.column_stack((values, values / 2))}
    ensembles = {h: np.repeat(values[None, :h], 3, axis=0) for h in (8, 16)}
    scores = evaluate_windows([session], ensembles, 0., 1., 42, horizons=(8, 16))[0]
    assert np.isfinite(scores["window_score"])
    assert "energy_128" not in scores


def test_panel_aggregation_weights_cases_equally_and_pairs_methods():
    from scripts.experiments.temporal_profiles.matrix_report import group_summary
    from scripts.experiments.temporal_profiles.study import METHODS

    records = []
    for value, run_count in ((1., 25), (3., 5)):
        data = {method: {"window_score": {"mean": value + (0.5 if method == "hmm" else 0),
                                         "samples": np.full(100, value + (0.5 if method == "hmm" else 0)),
                                         "run_count": run_count}} for method in METHODS}
        records.append({"data": data})
    summary = group_summary(records, "window_score")
    assert summary["methods"]["temporal"]["mean"] == 2.
    assert summary["methods"]["temporal"]["runs"] == 30
    np.testing.assert_allclose(summary["paired"]["hmm_minus_temporal"]["ci95"], [.5, .5])


def test_secondary_bootstrap_preserves_run_identity_with_missing_short_runs():
    from scripts.experiments.temporal_profiles.matrix_report import bootstrap_supported

    # Draws select global IDs 10/30, with ID 20 excluded by the horizon rule.
    indices = (np.array([[0, 1], [1, 1]]), np.array([[0, 1, 2], [2, 1, 2]]))
    matrix = np.array([[1., 5.], [3., 7.]])
    actual = bootstrap_supported(matrix, [10, 30], [10, 20, 30], indices)
    np.testing.assert_allclose(actual, [4., 7.])
