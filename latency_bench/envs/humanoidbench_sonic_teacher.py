"""Deterministic SONIC reference teacher for the HumanoidBench Hurdle task."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

RUN_STATE = "run"
JUMP_STATE = "jump"
DEFAULT_HURDLE_POSITIONS = tuple(float(index * 7) for index in range(1, 11))
DEFAULT_TRIGGER_DISTANCES = tuple(np.arange(0.5, 3.01, 0.25).tolist())
SONIC_CONTROL_HZ = 50.0
FOOT_CONTACT_HEIGHT = 0.02

# These fields are the episode-level quality contract supplied by the task
# evaluator.  Per-step HumanoidBench reward is intentionally not part of it.
COURSE_COMPLETE_INFO = "course_complete"
HURDLES_PASSED_INFO = "hurdles_passed"
HURDLE_CONTACTS_INFO = "hurdle_contacts"
QUALITY_FAILURE_INFO = "falls_or_nonfoot_ground_contact"
TEACHER_ACCEPTED_INFO = "teacher_accepted"
REFERENCE_TRACKING_PREFIX = "reference_tracking_"


def reference_phase_cost(
    actual_joint_positions: np.ndarray,
    actual_joint_velocities: np.ndarray,
    actual_foot_contacts: np.ndarray,
    reference_qpos: np.ndarray,
    reference_joint_velocities: np.ndarray,
    reference_foot_contacts: np.ndarray,
) -> np.ndarray:
    """Return the measured-state matching cost for each reference phase."""

    position_error = np.sqrt(
        np.mean((np.asarray(reference_qpos)[:, 7:] - actual_joint_positions) ** 2, axis=1)
    )
    velocity_error = np.sqrt(
        np.mean(
            (np.asarray(reference_joint_velocities) - actual_joint_velocities) ** 2,
            axis=1,
        )
    )
    contact_error = np.mean(
        np.asarray(reference_foot_contacts) != np.asarray(actual_foot_contacts),
        axis=1,
    )
    return position_error + 0.1 * velocity_error + 0.25 * contact_error


def match_reference_phase(
    actual_joint_positions: np.ndarray,
    actual_joint_velocities: np.ndarray,
    actual_foot_contacts: np.ndarray,
    reference_qpos: np.ndarray,
    reference_joint_velocities: np.ndarray,
    reference_foot_contacts: np.ndarray,
    candidate_phases: Iterable[int] | None = None,
) -> int:
    """Return the lowest-cost reference phase for a measured robot state."""

    phases = (
        np.arange(len(reference_qpos), dtype=np.int64)
        if candidate_phases is None
        else np.asarray(tuple(candidate_phases), dtype=np.int64)
    )
    costs = reference_phase_cost(
        actual_joint_positions,
        actual_joint_velocities,
        actual_foot_contacts,
        np.asarray(reference_qpos)[phases],
        np.asarray(reference_joint_velocities)[phases],
        np.asarray(reference_foot_contacts)[phases],
    )
    return int(phases[np.argmin(costs)])


class SonicHurdleTeacher:
    """Playback-only run/jump teacher with no residual or learned action."""

    def __init__(
        self,
        run_tokens: np.ndarray,
        jump_tokens: np.ndarray,
        run_reference_qpos: np.ndarray,
        jump_reference_qpos: np.ndarray,
        run_reference_foot_contacts: np.ndarray,
        jump_reference_foot_contacts: np.ndarray,
        neutral_hand_action: np.ndarray,
        *,
        trigger_distance: float,
        hurdle_positions: Iterable[float] = DEFAULT_HURDLE_POSITIONS,
    ) -> None:
        self.run_tokens = np.asarray(run_tokens, dtype=np.float32)
        self.jump_tokens = np.asarray(jump_tokens, dtype=np.float32)
        self.run_reference_qpos = np.asarray(run_reference_qpos, dtype=np.float32)
        self.jump_reference_qpos = np.asarray(jump_reference_qpos, dtype=np.float32)
        self.run_reference_joint_velocities = (
            np.gradient(self.run_reference_qpos[:, 7:], axis=0) * SONIC_CONTROL_HZ
        )
        self.jump_reference_joint_velocities = (
            np.gradient(self.jump_reference_qpos[:, 7:], axis=0) * SONIC_CONTROL_HZ
        )
        self.run_reference_foot_contacts = np.asarray(run_reference_foot_contacts)
        self.jump_reference_foot_contacts = np.asarray(jump_reference_foot_contacts)
        self.neutral_hand_action = np.asarray(neutral_hand_action, dtype=np.float32)
        self.trigger_distance = trigger_distance
        self.hurdle_positions = tuple(hurdle_positions)
        self.resume_run_phase = match_reference_phase(
            self.jump_reference_qpos[-1, 7:],
            self.jump_reference_joint_velocities[-1],
            self.jump_reference_foot_contacts[-1],
            self.run_reference_qpos,
            self.run_reference_joint_velocities,
            self.run_reference_foot_contacts,
        )
        self.reset()

    def reset(self) -> None:
        self.state = RUN_STATE
        self.run_phase = 0
        self.jump_phase = 0
        self.next_hurdle_index = 0

    def _action(
        self,
        token: np.ndarray,
        reference_qpos: np.ndarray,
        reference_joint_velocity: np.ndarray,
        reference_foot_contacts: np.ndarray,
    ) -> np.ndarray:
        self.last_reference_qpos = reference_qpos
        self.last_reference_joint_velocity = reference_joint_velocity
        self.last_reference_foot_contacts = reference_foot_contacts
        return np.concatenate((token, self.neutral_hand_action)).astype(np.float32)

    def _jump_action(self) -> np.ndarray:
        action = self._action(
            self.jump_tokens[self.jump_phase],
            self.jump_reference_qpos[self.jump_phase],
            self.jump_reference_joint_velocities[self.jump_phase],
            self.jump_reference_foot_contacts[self.jump_phase],
        )
        self.jump_phase += 1
        if self.jump_phase == len(self.jump_tokens):
            self.state = RUN_STATE
            self.run_phase = self.resume_run_phase
            self.jump_phase = 0
        return action

    def action(self, root_x: float) -> np.ndarray:
        """Return the executed 78D action for the current measured root x."""

        while (
            self.next_hurdle_index < len(self.hurdle_positions)
            and root_x > self.hurdle_positions[self.next_hurdle_index]
        ):
            self.next_hurdle_index += 1

        if self.state == JUMP_STATE:
            return self._jump_action()

        if self.next_hurdle_index < len(self.hurdle_positions):
            distance = self.hurdle_positions[self.next_hurdle_index] - root_x
            if 0.0 <= distance <= self.trigger_distance:
                self.state = JUMP_STATE
                return self._jump_action()

        action = self._action(
            self.run_tokens[self.run_phase],
            self.run_reference_qpos[self.run_phase],
            self.run_reference_joint_velocities[self.run_phase],
            self.run_reference_foot_contacts[self.run_phase],
        )
        self.run_phase = (self.run_phase + 1) % len(self.run_tokens)
        return action

    __call__ = action


def hard_valid_episode(info: Mapping[str, Any]) -> bool:
    """Return whether an episode has no task/contact/fall quality violation."""

    return not (
        info[HURDLE_CONTACTS_INFO]
        or info[QUALITY_FAILURE_INFO]
    )


def lexicographic_score(
    info: Mapping[str, Any],
    *,
    reference_metric_keys: Iterable[str] | None = None,
) -> tuple[Any, ...]:
    """Score an episode: validity, completion, progress, then tracking error."""

    metric_keys = (
        tuple(reference_metric_keys)
        if reference_metric_keys is not None
        else tuple(
            sorted(
                key
                for key in info
                if key.startswith(REFERENCE_TRACKING_PREFIX)
            )
        )
    )
    tracking = tuple(
        -info[key]
        for key in metric_keys
    )
    return (
        hard_valid_episode(info),
        info[COURSE_COMPLETE_INFO],
        info[HURDLES_PASSED_INFO],
        *tracking,
    )


def select_best_result(
    results: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select the best diagnostic sweep result using the teacher score."""

    results = list(results)
    metric_keys = sorted(
        set.intersection(
            *(
                {
                    key
                    for key in result["info"]
                    if key.startswith(REFERENCE_TRACKING_PREFIX)
                }
                for result in results
            )
        )
    )
    return max(
        results,
        key=lambda result: lexicographic_score(
            result["info"], reference_metric_keys=metric_keys
        ),
    )


def reference_tracking_errors(env: Any, teacher: SonicHurdleTeacher) -> dict[str, float]:
    """Compare the measured G1 state with the token's physical reference."""

    native = env.unwrapped
    actual_joint_positions = native.data.qpos[native._body_qpos_addresses]
    actual_joint_velocities = native.data.qvel[native._body_qvel_addresses]
    reference_qpos = teacher.last_reference_qpos
    quaternion_dot = abs(
        np.dot(native.data.qpos[3:7], reference_qpos[3:7])
        / (
            np.linalg.norm(native.data.qpos[3:7])
            * np.linalg.norm(reference_qpos[3:7])
        )
    )
    actual_foot_contacts = np.asarray(
        [
            native.data.site_xpos[native.model.site(name).id, 2]
            <= FOOT_CONTACT_HEIGHT
            for name in ("left_foot", "right_foot")
        ]
    )
    return {
        "reference_tracking_joint_position_rmse": float(
            np.sqrt(np.mean((actual_joint_positions - reference_qpos[7:]) ** 2))
        ),
        "reference_tracking_joint_velocity_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        actual_joint_velocities
                        - teacher.last_reference_joint_velocity
                    )
                    ** 2
                )
            )
        ),
        "reference_tracking_root_height_error": float(
            abs(native.data.qpos[2] - reference_qpos[2])
        ),
        "reference_tracking_root_orientation_error": float(
            2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
        ),
        "reference_tracking_contact_phase_mismatch": float(
            np.mean(actual_foot_contacts != teacher.last_reference_foot_contacts)
        ),
    }


def evaluate_episode(
    env: Any,
    teacher: SonicHurdleTeacher,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run one deterministic teacher episode and return its episode summary."""

    teacher.reset()
    env.reset(seed=seed)
    total_reward = 0.0
    info: Mapping[str, Any] = {}
    tracking_totals: dict[str, float] = {}
    for step in range(env.spec.max_episode_steps):
        action = teacher(env.unwrapped.data.qpos[0])
        _, reward, terminated, truncated, info = env.step(action)
        for key, value in reference_tracking_errors(env, teacher).items():
            tracking_totals[key] = tracking_totals.get(key, 0.0) + value
        total_reward += reward
        if terminated or truncated:
            break
    info = {
        **info,
        **{key: value / (step + 1) for key, value in tracking_totals.items()},
    }
    return {
        "trigger_distance": teacher.trigger_distance,
        "steps": step + 1,
        "return": total_reward,
        "info": info,
    }


def evaluate_trigger_distances(
    env_factory: Callable[[], Any],
    run_tokens: np.ndarray,
    jump_tokens: np.ndarray,
    run_reference_qpos: np.ndarray,
    jump_reference_qpos: np.ndarray,
    run_reference_foot_contacts: np.ndarray,
    jump_reference_foot_contacts: np.ndarray,
    neutral_hand_action: np.ndarray,
    trigger_distances: Iterable[float],
    *,
    seed: int = 0,
    hurdle_positions: Iterable[float] = DEFAULT_HURDLE_POSITIONS,
) -> dict[str, Any]:
    """Evaluate each trigger distance from the same deterministic reset."""

    hurdle_positions = tuple(hurdle_positions)
    results = []
    for trigger_distance in trigger_distances:
        teacher = SonicHurdleTeacher(
            run_tokens,
            jump_tokens,
            run_reference_qpos,
            jump_reference_qpos,
            run_reference_foot_contacts,
            jump_reference_foot_contacts,
            neutral_hand_action,
            trigger_distance=trigger_distance,
            hurdle_positions=hurdle_positions,
        )
        env = env_factory()
        try:
            results.append(evaluate_episode(env, teacher, seed=seed))
        finally:
            env.close()
    best = select_best_result(results)
    return {
        TEACHER_ACCEPTED_INFO: best["info"][TEACHER_ACCEPTED_INFO],
        "best": best,
        "results": results,
    }
