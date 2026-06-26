from __future__ import annotations


def test_clean_v1_accepts_deadly_corridor_by_raw_return():
    from latency_bench.data.filters import clean_v1_accepts_episode_summary

    low_return_summary = {"episode_length": 12, "episode_raw_return": 2199.0}
    high_return_summary = {"episode_length": 12, "episode_raw_return": 2200.0}

    assert not clean_v1_accepts_episode_summary("deadly_corridor", low_return_summary, max_steps_per_episode=1000)
    assert clean_v1_accepts_episode_summary("deadly_corridor", high_return_summary, max_steps_per_episode=1000)


def test_clean_v1_accepts_flappy_by_length_fraction():
    from latency_bench.data.filters import clean_v1_accepts_episode_summary

    # Flappy keeps episodes that survive >= 0.7 * max_steps_per_episode.
    assert not clean_v1_accepts_episode_summary(
        "flappy", {"episode_length": 699, "episode_raw_return": 0.0}, max_steps_per_episode=1000
    )
    assert clean_v1_accepts_episode_summary(
        "flappy", {"episode_length": 700, "episode_raw_return": 0.0}, max_steps_per_episode=1000
    )


def test_clean_v1_accepts_demon_attack_by_raw_score():
    from latency_bench.data.filters import clean_v1_accepts_episode_summary

    assert not clean_v1_accepts_episode_summary(
        "demon_attack",
        {"episode_length": 12, "episode_raw_return": 7309.0, "episode_clipped_return": 90.0},
        max_steps_per_episode=1000,
    )
    assert clean_v1_accepts_episode_summary(
        "demon_attack",
        {"episode_length": 12, "episode_raw_return": 7310.0, "episode_clipped_return": 1.0},
        max_steps_per_episode=1000,
    )


def test_demon_raw_score_bins_match_eval_distribution_breaks():
    from latency_bench.data.filters import demon_raw_score_bin_counts

    counts = demon_raw_score_bin_counts(
        [
            {"episode_raw_return": 6270.0},
            {"episode_raw_return": 6550.0},
            {"episode_raw_return": 6920.0},
            {"episode_raw_return": 7310.0},
            {"episode_raw_return": 8460.0},
        ]
    )

    assert counts == {
        "<6460": 1,
        "6460-6889": 1,
        "6890-7309": 1,
        "7310-7999": 1,
        ">=8000": 1,
    }


def test_flappy_clean_v1_train_filter_keeps_flap_context_and_samples_background():
    from latency_bench.data.filters import FLAPPY_CONTEXT_WINDOW, filter_train_rows_clean_v1

    rows = [
        {"decision_step": step, "action_text": action_text, "raw_reward": 0.0}
        for step, action_text in enumerate(["NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "FLAP", "NOOP", "NOOP", "NOOP"])
    ]

    kept_steps = [row["decision_step"] for row in filter_train_rows_clean_v1("flappy", rows, seed=123)]
    expected_context = {
        step
        for step in range(len(rows))
        if abs(step - 5) <= FLAPPY_CONTEXT_WINDOW
    }

    assert 5 in kept_steps
    assert expected_context <= set(kept_steps)


def test_contiguous_v1_keeps_all_rows_in_order():
    """KV-memory rollouts need contiguous per-episode decision sequences, so the
    contiguous_v1 preset must not drop interior rows (unlike clean_v1)."""
    from latency_bench.data.filters import FILTER_PRESET_CONTIGUOUS_V1, filter_train_rows

    rows = [
        {"decision_step": step, "action_text": "FLAP" if step == 5 else "NOOP", "raw_reward": 0.0}
        for step in range(9)
    ]

    kept_steps = [
        row["decision_step"]
        for row in filter_train_rows(FILTER_PRESET_CONTIGUOUS_V1, "flappy", rows, seed=123)
    ]

    assert kept_steps == list(range(9))


def test_contiguous_v1_still_uses_episode_quality_gate():
    """Quality control moves to the episode level: contiguous_v1 keeps the same
    episode accept/reject gate as clean_v1."""
    from latency_bench.data.filters import FILTER_PRESET_CONTIGUOUS_V1, accepts_episode

    low = {"episode_length": 12, "episode_raw_return": 2199.0}
    high = {"episode_length": 12, "episode_raw_return": 2200.0}

    assert not accepts_episode(FILTER_PRESET_CONTIGUOUS_V1, "deadly_corridor", low, max_steps_per_episode=1000)
    assert accepts_episode(FILTER_PRESET_CONTIGUOUS_V1, "deadly_corridor", high, max_steps_per_episode=1000)


def test_demon_clean_v1_train_filter_keeps_reward_context_and_survival_anchors():
    from latency_bench.data.filters import filter_train_rows_clean_v1

    rows = [
        {"decision_step": step, "action_text": "NOOP", "raw_reward": 1.0 if step == 20 else 0.0}
        for step in range(40)
    ]

    kept_steps = [row["decision_step"] for row in filter_train_rows_clean_v1("demon_attack", rows, seed=123)]

    assert set(range(12, 29)) <= set(kept_steps)
    assert {0, 16, 32} <= set(kept_steps)


def test_density_weights_flappy_events_full_background_downweighted():
    from latency_bench.data.filters import (
        FLAPPY_BACKGROUND_KEEP_PROB,
        FLAPPY_CONTEXT_WINDOW,
        density_weights_clean_v1,
    )

    rows = [
        {"decision_step": step, "action_text": "FLAP" if step == 5 else "NOOP", "raw_reward": 0.0}
        for step in range(12)
    ]
    weights = density_weights_clean_v1("flappy", rows)

    assert len(weights) == len(rows)
    for step, weight in enumerate(weights):
        if abs(step - 5) <= FLAPPY_CONTEXT_WINDOW:
            assert weight == 1.0
        else:
            assert weight == FLAPPY_BACKGROUND_KEEP_PROB


def test_density_weights_deadly_uniform():
    from latency_bench.data.filters import density_weights_clean_v1

    rows = [{"decision_step": step, "action_text": "NOOP", "raw_reward": 0.0} for step in range(10)]
    assert density_weights_clean_v1("deadly_corridor", rows) == [1.0] * 10


def test_density_weights_demon_reward_context_and_anchors_full():
    from latency_bench.data.filters import (
        DEMON_BACKGROUND_KEEP_PROB,
        DEMON_CONTEXT_WINDOW,
        DEMON_SURVIVAL_ANCHOR_STRIDE,
        density_weights_clean_v1,
    )

    rows = [
        {"decision_step": step, "action_text": "NOOP", "raw_reward": 1.0 if step == 20 else 0.0}
        for step in range(40)
    ]
    weights = density_weights_clean_v1("demon_attack", rows)

    for step, weight in enumerate(weights):
        near_reward = abs(step - 20) <= DEMON_CONTEXT_WINDOW
        anchor = step % DEMON_SURVIVAL_ANCHOR_STRIDE == 0
        if near_reward or anchor:
            assert weight == 1.0
        else:
            assert weight == DEMON_BACKGROUND_KEEP_PROB
