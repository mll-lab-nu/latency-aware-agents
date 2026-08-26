"""Generate SONIC planner references and v1.1 encoder token caches.

The pinned planner emits MuJoCo G1 ``qpos`` frames at 30 Hz.  This module
repeats planner inference, resamples the result to SONIC's 50 Hz control rate,
and runs the matching v1.1 encoder.  The encoder output already includes FSQ;
the saved array therefore contains the 64-dimensional base tokens consumed by
the residual controller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

SONIC_PLANNER_RUN_MODE = 3
SONIC_PLANNER_DEFAULT_SPEED = -1.0
SONIC_PLANNER_FORWARD = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
SONIC_PLANNER_FACING = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
SONIC_PLANNER_ALLOWED_TOKENS = np.ones((1, 11), dtype=np.int64)

SONIC_ENCODER_INPUT_DIM = 1751
SONIC_TOKEN_DIM = 64
SONIC_CONTROL_HZ = 50.0
SONIC_PLANNER_HZ = 30.0
SONIC_ENCODER_REFERENCE_FRAMES = 10
SONIC_ENCODER_REFERENCE_STEP = 5
SONIC_REFERENCE_LOOKAHEAD = (
    SONIC_ENCODER_REFERENCE_FRAMES - 1
) * SONIC_ENCODER_REFERENCE_STEP

# MuJoCo joint order -> encoder/IsaacLab joint order.  This is the inverse of
# the order used by the SONIC decoder to consume its 29 body outputs.
_MUJOCO_INDEX_FOR_POLICY = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)
_DEFAULT_ANGLES = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def default_initial_qpos() -> np.ndarray:
    """Return the standard SONIC G1 MuJoCo qpos used by the planner."""

    qpos = np.zeros(36, dtype=np.float32)
    qpos[:7] = [0.0, 0.0, 0.788740, 1.0, 0.0, 0.0, 0.0]
    qpos[7:] = _DEFAULT_ANGLES
    return qpos


def _planner_inputs(context_qpos: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    return {
        "context_mujoco_qpos": context_qpos[None, ...].astype(np.float32),
        "target_vel": np.asarray([SONIC_PLANNER_DEFAULT_SPEED], dtype=np.float32),
        "mode": np.asarray([SONIC_PLANNER_RUN_MODE], dtype=np.int64),
        "movement_direction": SONIC_PLANNER_FORWARD,
        "facing_direction": SONIC_PLANNER_FACING,
        "height": np.asarray([-1.0], dtype=np.float32),
        "random_seed": np.asarray([seed], dtype=np.int64),
        "has_specific_target": np.asarray([[0]], dtype=np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
        "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
        "allowed_pred_num_tokens": SONIC_PLANNER_ALLOWED_TOKENS,
    }


def _run_planner(
    planner_session: object, context_qpos: np.ndarray, seed: int
) -> np.ndarray:
    outputs = planner_session.run(None, _planner_inputs(context_qpos, seed))
    names = [output.name for output in planner_session.get_outputs()]
    output_by_name = dict(zip(names, outputs))
    num_frames = int(np.asarray(output_by_name["num_pred_frames"]).reshape(-1)[0])
    return np.asarray(output_by_name["mujoco_qpos"], dtype=np.float32)[0, :num_frames]


def generate_planner_qpos(
    planner_session: object,
    horizon_steps: int,
    *,
    seed: int = 1234,
) -> np.ndarray:
    """Generate enough 30 Hz MuJoCo qpos frames for an encoded horizon."""

    qpos = default_initial_qpos()
    required_30hz_frames = (
        int(
            np.ceil(
                (horizon_steps + SONIC_REFERENCE_LOOKAHEAD)
                * SONIC_PLANNER_HZ
                / SONIC_CONTROL_HZ
            )
        )
        + 1
    )
    context = np.repeat(qpos[None, :], 4, axis=0)
    chunks: list[np.ndarray] = []
    generated_frames = 0
    while generated_frames < required_30hz_frames:
        chunk = _run_planner(planner_session, context, seed)
        new_chunk = chunk if not chunks else chunk[4:]
        chunks.append(new_chunk)
        generated_frames += len(new_chunk)
        context = chunk[-4:]
    return np.concatenate(chunks, axis=0)[:required_30hz_frames]


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, weight: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return (q0 + weight * (q1 - q0)) / np.linalg.norm(q0 + weight * (q1 - q0))
    theta = np.arccos(dot)
    sine_theta = np.sin(theta)
    return (
        np.sin((1.0 - weight) * theta) / sine_theta * q0
        + np.sin(weight * theta) / sine_theta * q1
    )


def resample_qpos_to_50hz(qpos_30hz: np.ndarray, num_frames: int) -> np.ndarray:
    """Resample planner qpos from 30 Hz to 50 Hz using quaternion SLERP."""

    source = np.asarray(qpos_30hz, dtype=np.float64)
    result = np.empty((num_frames, 36), dtype=np.float32)
    for target_frame, target_time in enumerate(
        np.arange(num_frames) / SONIC_CONTROL_HZ
    ):
        source_position = target_time * SONIC_PLANNER_HZ
        frame0 = min(int(np.floor(source_position)), len(source) - 1)
        frame1 = min(frame0 + 1, len(source) - 1)
        weight = min(source_position - frame0, 1.0)
        result[target_frame, :3] = (1.0 - weight) * source[
            frame0, :3
        ] + weight * source[frame1, :3]
        result[target_frame, 3:7] = _quat_slerp(
            source[frame0, 3:7], source[frame1, 3:7], weight
        )
        result[target_frame, 7:] = (1.0 - weight) * source[
            frame0, 7:
        ] + weight * source[frame1, 7:]
    return result


def _heading_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ]
    )


def build_encoder_observation(
    qpos_50hz: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    """Pack the official 1751D v1.1 encoder input for G1 mode."""

    qpos = np.asarray(qpos_50hz, dtype=np.float64)
    base_heading = _heading_quaternion(qpos[frame_index, 3:7])
    frame_indices = np.minimum(
        frame_index
        + np.arange(SONIC_ENCODER_REFERENCE_FRAMES) * SONIC_ENCODER_REFERENCE_STEP,
        len(qpos) - 1,
    )
    reference = qpos[frame_indices]
    joint_positions = reference[:, 7:][:, _MUJOCO_INDEX_FOR_POLICY]
    joint_positions_all = qpos[:, 7:]
    joint_velocities_all = np.empty_like(joint_positions_all)
    joint_velocities_all[:-1] = np.diff(joint_positions_all, axis=0) * SONIC_CONTROL_HZ
    joint_velocities_all[-1] = joint_velocities_all[-2]
    joint_velocities = joint_velocities_all[frame_indices][:, _MUJOCO_INDEX_FOR_POLICY]
    anchor_heading = np.asarray(
        [
            _quat_to_rotation_6d(
                _quat_multiply(_quat_conjugate(base_heading), frame[3:7])
            )
            for frame in reference
        ]
    )

    observation = np.zeros(SONIC_ENCODER_INPUT_DIM, dtype=np.float32)
    observation[4 : 4 + 10 * 29] = joint_positions.reshape(-1)
    observation[4 + 10 * 29 : 4 + 20 * 29] = joint_velocities.reshape(-1)
    observation[4 + 20 * 29 : 4 + 20 * 29 + 10 * 6] = anchor_heading.reshape(-1)
    return observation


def encode_reference_tokens(
    planner_session: object,
    encoder_session: object,
    horizon_steps: int,
    *,
    seed: int = 1234,
) -> np.ndarray:
    """Generate ``horizon_steps + 1`` FSQ-compressed base tokens."""

    qpos_30hz = generate_planner_qpos(
        planner_session,
        horizon_steps,
        seed=seed,
    )
    qpos_50hz = resample_qpos_to_50hz(
        qpos_30hz,
        horizon_steps + SONIC_REFERENCE_LOOKAHEAD + 1,
    )
    tokens = []
    for frame_index in range(horizon_steps + 1):
        encoder_input = build_encoder_observation(qpos_50hz, frame_index)
        output = encoder_session.run(
            ["encoded_tokens"], {"obs_dict": encoder_input[None, :]}
        )[0]
        tokens.append(output[0])
    return np.asarray(tokens, dtype=np.float32).reshape(-1, SONIC_TOKEN_DIM)


def write_reference_tokens(
    planner_path: Path,
    encoder_path: Path,
    output_path: Path,
    horizon_steps: int,
    *,
    seed: int = 1234,
) -> None:
    """Run pinned ONNX planner/encoder and save the token cache."""

    planner_session = ort.InferenceSession(
        str(planner_path), providers=["CPUExecutionProvider"]
    )
    encoder_session = ort.InferenceSession(
        str(encoder_path), providers=["CPUExecutionProvider"]
    )
    tokens = encode_reference_tokens(
        planner_session,
        encoder_session,
        horizon_steps,
        seed=seed,
    )
    np.save(output_path, tokens)
