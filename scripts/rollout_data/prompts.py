from __future__ import annotations

BASE_PROMPTS = {
    "flappy": "You are playing Flappy Bird. Your goal is to pass through the pipes, avoid crashing, and maximize score. Choose exactly one action from: NOOP, FLAP.",

    "demon_attack": "You are playing Demon Attack. Your goal is to survive, avoid enemy fire, and shoot enemies to maximize score. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.",

    "deadly_corridor": "You are playing Deadly Corridor. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. Choose one option from each action group: turn (TURN_NOOP, TURN_LEFT, TURN_RIGHT), move (MOVE_NOOP, MOVE_FORWARD, MOVE_BACKWARD), strafe (STRAFE_NOOP, MOVE_LEFT, MOVE_RIGHT), and attack (ATTACK_NOOP, ATTACK). The four choices are combined into the next action.",
}

TIMING_SUFFIX = (
    " Current action latency is {latency_raw_frames} raw frames ({latency_ms:.2f} ms). "
    " The environment runs at {env_fps:g} FPS and observations are emitted at {obs_fps:g} FPS. "
    "Choose the best next action."
)


def render_teacher_prompt(
    env_name: str,
    *,
    latency_raw_frames: int,
    latency_ms: float,
    env_fps: float,
    obs_fps: float,
) -> str:
    return BASE_PROMPTS[str(env_name)] + TIMING_SUFFIX.format(
        latency_raw_frames=int(latency_raw_frames),
        latency_ms=float(latency_ms),
        env_fps=float(env_fps),
        obs_fps=float(obs_fps),
    )
