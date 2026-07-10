from latency_bench.envs.atari import is_true_atari_episode_end


def test_episode_end_contract_prioritizes_explicit_terminal_metadata():
    assert not is_true_atari_episode_end(False, False, {"is_true_episode_end": True})
    assert is_true_atari_episode_end(False, True, {"is_true_episode_end": False})
    assert not is_true_atari_episode_end(True, False, {"is_true_episode_end": False, "lives": 0})
    assert not is_true_atari_episode_end(True, False, {"game_over": False, "lives": 0})
    assert not is_true_atari_episode_end(True, False, {"lives": 2})
    assert is_true_atari_episode_end(True, False, {"lives": 0})
    assert is_true_atari_episode_end(True, False, None)
