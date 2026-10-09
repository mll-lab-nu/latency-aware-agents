from __future__ import annotations

import numpy as np
import pytest


def test_ghost_trail_image_builds_single_causal_composite():
    from latency_bench.data.ghost_trail import build_ghost_trail_image

    ghost = np.zeros((4, 4, 3), dtype=np.uint8)
    ghost[1, 1] = [255, 0, 0]
    current = np.full((4, 4, 3), 10, dtype=np.uint8)

    def segment_red(frame):
        return frame[..., 0] > 200

    out = build_ghost_trail_image(
        [ghost, current],
        [0, 1],
        segment_fn=segment_red,
        gamma=1.3,
        min_alpha=255,
        scroll_px_per_step=0.0,
    )

    assert out[1, 1].tolist() == [255, 0, 0]
    assert out[0, 0].tolist() == [10, 10, 10]


@pytest.mark.parametrize("history_frames", [1, 5, 30])
def test_demon_trail_stream_matches_causal_windows_and_reset(history_frames):
    """Long-lived ghost-trail contract: cached frames preserve causal pixels and reset isolation."""
    from latency_bench.data.ghost_trail import (
        DemonGhostTrailStream, GhostTrailConfig,
        build_demon_attack_ghost_trail_episode_frames, build_demon_attack_ghost_trail_window,
    )
    frames = []
    for step in range(12):
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[177:180, 30 + step:38 + step] = [184, 70, 162]
        frame[60:65, 20 + step:28 + step] = [80, 180, 90]
        frame[140 - step * 4:142 - step * 4, 80:82] = [210, 170, 120]
        frames.append(frame)
    episode = build_demon_attack_ghost_trail_episode_frames(
        frames, range(len(frames)), config=GhostTrailConfig(history_frames=history_frames),
    )
    stream = DemonGhostTrailStream(history_frames)
    for step, frame in enumerate(frames):
        stream.append(frame)
        rendered = stream.render()
        window = frames[max(0, step - history_frames + 1):step + 1]
        np.testing.assert_array_equal(rendered, build_demon_attack_ghost_trail_window(window))
        np.testing.assert_array_equal(rendered, episode[step])
        foreground = np.any(frame != 0, axis=-1)
        np.testing.assert_array_equal(rendered[foreground], frame[foreground])
    if history_frames > 1:
        assert np.any(episode[11] != frames[11])
    stream = DemonGhostTrailStream(history_frames)
    stream.append(frames[0])
    np.testing.assert_array_equal(stream.render(), frames[0])
