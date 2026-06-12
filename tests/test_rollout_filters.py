from __future__ import annotations


def test_clean_v1_accepts_deadly_corridor_by_raw_return():
    from latency_bench.data.filters import clean_v1_accepts_episode_summary

    low_return_summary = {"episode_length": 12, "episode_raw_return": 2199.0}
    high_return_summary = {"episode_length": 12, "episode_raw_return": 2200.0}

    assert not clean_v1_accepts_episode_summary("deadly_corridor", low_return_summary, max_steps_per_episode=1000)
    assert clean_v1_accepts_episode_summary("deadly_corridor", high_return_summary, max_steps_per_episode=1000)


def test_clean_v1_accepts_demon_attack_by_clipped_return():
    from latency_bench.data.filters import clean_v1_accepts_episode_summary

    assert not clean_v1_accepts_episode_summary(
        "demon_attack",
        {"episode_length": 12, "episode_clipped_return": 50.0},
        max_steps_per_episode=1000,
    )
    assert clean_v1_accepts_episode_summary(
        "demon_attack",
        {"episode_length": 12, "episode_clipped_return": 51.0},
        max_steps_per_episode=1000,
    )


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


def test_demon_clean_v1_train_filter_keeps_reward_context_and_survival_anchors():
    from latency_bench.data.filters import filter_train_rows_clean_v1

    rows = [
        {"decision_step": step, "action_text": "NOOP", "clipped_reward": 1.0 if step == 20 else 0.0}
        for step in range(40)
    ]

    kept_steps = [row["decision_step"] for row in filter_train_rows_clean_v1("demon_attack", rows, seed=123)]

    assert set(range(12, 29)) <= set(kept_steps)
    assert {0, 16, 32} <= set(kept_steps)
