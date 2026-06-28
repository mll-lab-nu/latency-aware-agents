"""Prompt templates for canonical rollout datasets."""

from __future__ import annotations

BASE_PROMPTS = {
    "flappy": "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. Choose the action: NOOP, FLAP.",
    "demon_attack": "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE. Survive, avoid enemy fire, and shoot enemies to maximize score.",
    "deadly_corridor": "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. The available actions are: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK. Choose the best next action from the current visual observation.",
}

TIMING_SUFFIX = (
    " Current action latency is {latency_raw_frames} raw frames ({latency_ms:.2f} ms). "
    " The environment runs at {env_fps:g} FPS and observations are emitted at {obs_fps:g} FPS. "
    "Choose the best next action."
)

IMAGE_TRANSFORM_CONTEXTS = {
    "flappy_ghost_trail": " You can see the past motion of the bird in the image. Use it to make your decision.",
}


def render_teacher_prompt(
    env_name: str,
    *,
    latency_raw_frames: int,
    latency_ms: float,
    env_fps: float,
    obs_fps: float,
    image_transform: str = "raw_rgb",
) -> str:
    """Render the prompt stored on each rollout decision row.

    Args:
        env_name: Canonical benchmark environment name.
        latency_raw_frames: Action latency measured in raw environment frames.
        latency_ms: Action latency in milliseconds.
        env_fps: Raw environment frame rate.
        obs_fps: Observation decision rate.
        image_transform: Optional exported-image transform name. Some transforms
            add visual context that should be reflected in the prompt.

    Returns:
        Base environment prompt with timing context appended.
    """

    transform_context = IMAGE_TRANSFORM_CONTEXTS.get(str(image_transform).strip().lower(), "")
    return BASE_PROMPTS[str(env_name)] + transform_context + TIMING_SUFFIX.format(
        latency_raw_frames=int(latency_raw_frames),
        latency_ms=float(latency_ms),
        env_fps=float(env_fps),
        obs_fps=float(obs_fps),
    )
