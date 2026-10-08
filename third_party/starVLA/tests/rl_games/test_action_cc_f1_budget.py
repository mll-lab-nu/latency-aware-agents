from starVLA.training.rl_games.action_cc_f1_budget import build_frame_records


def test_shared_frame_budget_truncates_long_episode() -> None:
    episodes = [
        ("flappy", 0, "episode-0", 0, [(flat_idx, flat_idx) for flat_idx in range(3600)]),
    ]

    frame_records = build_frame_records(
        episodes=episodes,
        num_procs=1,
        rank=0,
        shared_frame_budget=400,
        per_latency_frame_budget=None,
        cross_task_mode=False,
    )

    assert len(frame_records) == 400
    assert frame_records[0] == (0, 0, "episode-0", 0, "flappy", 0)
    assert frame_records[-1] == (0, 399, "episode-0", 399, "flappy", 0)


def test_per_latency_frame_budget_truncates_each_latency() -> None:
    episodes = [
        ("flappy", 0, "latency-0", 0, [(flat_idx, flat_idx) for flat_idx in range(20)]),
        ("flappy", 0, "latency-1", 1, [(flat_idx, flat_idx) for flat_idx in range(20, 40)]),
    ]

    frame_records = build_frame_records(
        episodes=episodes,
        num_procs=1,
        rank=0,
        shared_frame_budget=100,
        per_latency_frame_budget=5,
        cross_task_mode=False,
    )

    assert len(frame_records) == 10
    assert [record[5] for record in frame_records] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
