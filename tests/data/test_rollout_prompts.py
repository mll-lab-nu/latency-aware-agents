from __future__ import annotations


def test_flappy_ghost_trail_prompt_context_precedes_latency_text():
    from latency_bench.data.prompts import BASE_PROMPTS, render_teacher_prompt

    prompt = render_teacher_prompt(
        "flappy",
        latency_raw_frames=2,
        latency_ms=66.67,
        env_fps=30.0,
        obs_fps=30.0,
        image_transform="flappy_ghost_trail",
    )

    ghost_text = "You can see the past motion of the bird in the image. Use it to make your decision."
    latency_text = "Current action latency is"
    assert prompt.startswith(BASE_PROMPTS["flappy"])
    assert ghost_text in prompt
    assert prompt.index(ghost_text) < prompt.index(latency_text)
