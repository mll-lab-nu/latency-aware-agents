"""Generate 50 Hz SONIC references and pack v1.1 encoder observations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_DEFAULT_ANGLES,
    MUJOCO_INDEX_FOR_POLICY,
    SONIC_TOKEN_DIM,
)

SONIC_PLANNER_RUN_MODE = 3
SONIC_PLANNER_DEFAULT_SPEED = 3.0
SONIC_PLANNER_FORWARD = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
SONIC_PLANNER_FACING = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
SONIC_PLANNER_ALLOWED_TOKENS = np.ones((1, 11), dtype=np.int64)

SONIC_ENCODER_INPUT_DIM = 1751
SONIC_CONTROL_HZ = 50.0
SONIC_PLANNER_HZ = 30.0
SONIC_ENCODER_REFERENCE_FRAMES = 10
SONIC_ENCODER_REFERENCE_STEP = 5
SONIC_REFERENCE_LOOKAHEAD = (
    SONIC_ENCODER_REFERENCE_FRAMES - 1
) * SONIC_ENCODER_REFERENCE_STEP

BONES_SEED_SOURCE_HZ = 120.0


def default_initial_qpos() -> np.ndarray:
    """Return the standard SONIC G1 MuJoCo qpos used by the planner."""

    qpos = np.zeros(36, dtype=np.float32)
    qpos[:7] = [0.0, 0.0, 0.788740, 1.0, 0.0, 0.0, 0.0]
    qpos[7:] = G1_DEFAULT_ANGLES
    return qpos


def _planner_inputs(
    context_qpos: np.ndarray,
    seed: int,
    *,
    target_velocity: float = SONIC_PLANNER_DEFAULT_SPEED,
) -> dict[str, np.ndarray]:
    return {
        "context_mujoco_qpos": context_qpos[None, ...].astype(np.float32),
        "target_vel": np.asarray([target_velocity], dtype=np.float32),
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
    planner_session: object,
    context_qpos: np.ndarray,
    seed: int,
    *,
    target_velocity: float,
) -> np.ndarray:
    outputs = planner_session.run(
        None,
        _planner_inputs(context_qpos, seed, target_velocity=target_velocity),
    )
    names = [output.name for output in planner_session.get_outputs()]
    output_by_name = dict(zip(names, outputs))
    num_frames = int(np.asarray(output_by_name["num_pred_frames"]).reshape(-1)[0])
    return np.asarray(output_by_name["mujoco_qpos"], dtype=np.float32)[0, :num_frames]


def load_bones_seed_qpos(csv_path: Path) -> np.ndarray:
    """Load one BONES-SEED G1 CSV into MuJoCo qpos order and SI units."""

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        joint_columns = [name for name in reader.fieldnames if name.endswith("_dof")]

    root_position = (
        np.asarray(
            [
                [row["root_translateX"], row["root_translateY"], row["root_translateZ"]]
                for row in rows
            ],
            dtype=np.float32,
        )
        / 100.0
    )
    root_euler = np.asarray(
        [
            [row["root_rotateX"], row["root_rotateY"], row["root_rotateZ"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    root_quat_xyzw = Rotation.from_euler("xyz", root_euler, degrees=True).as_quat()
    joint_positions = np.asarray(
        [[row[name] for name in joint_columns] for row in rows],
        dtype=np.float32,
    )

    qpos = np.empty((len(rows), 36), dtype=np.float32)
    qpos[:, :3] = root_position
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = np.deg2rad(joint_positions)
    return qpos


def generate_planner_qpos(
    planner_session: object,
    horizon_steps: int,
    *,
    seed: int = 1234,
    target_velocity: float = SONIC_PLANNER_DEFAULT_SPEED,
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
        chunk = _run_planner(
            planner_session,
            context,
            seed,
            target_velocity=target_velocity,
        )
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


def resample_qpos_to_50hz(
    qpos_30hz: np.ndarray,
    num_frames: int,
    *,
    source_hz: float = SONIC_PLANNER_HZ,
) -> np.ndarray:
    """Resample qpos to 50 Hz using quaternion SLERP and linear joints."""

    source = np.asarray(qpos_30hz, dtype=np.float64)
    result = np.empty((num_frames, 36), dtype=np.float32)
    for target_frame, target_time in enumerate(
        np.arange(num_frames) / SONIC_CONTROL_HZ
    ):
        source_position = target_time * source_hz
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
    *,
    actual_root_quaternion: np.ndarray | None = None,
) -> np.ndarray:
    """Pack the official 1751D v1.1 encoder input for G1 mode."""

    qpos = np.asarray(qpos_50hz, dtype=np.float64)
    root_quaternion = (
        qpos[frame_index, 3:7]
        if actual_root_quaternion is None
        else actual_root_quaternion
    )
    base_heading = _heading_quaternion(root_quaternion)
    frame_indices = np.minimum(
        frame_index
        + np.arange(SONIC_ENCODER_REFERENCE_FRAMES) * SONIC_ENCODER_REFERENCE_STEP,
        len(qpos) - 1,
    )
    reference = qpos[frame_indices]
    joint_positions = reference[:, 7:][:, MUJOCO_INDEX_FOR_POLICY]
    joint_positions_all = qpos[:, 7:]
    joint_velocities_all = np.empty_like(joint_positions_all)
    joint_velocities_all[:-1] = np.diff(joint_positions_all, axis=0) * SONIC_CONTROL_HZ
    joint_velocities_all[-1] = joint_velocities_all[-2]
    joint_velocities = joint_velocities_all[frame_indices][:, MUJOCO_INDEX_FOR_POLICY]
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


def encode_qpos_tokens(
    encoder_session: object, qpos_50hz: np.ndarray, horizon_steps: int
) -> np.ndarray:
    """Encode a 50 Hz qpos sequence with the SONIC v1.1 encoder."""

    tokens = []
    for frame_index in range(horizon_steps + 1):
        observation = build_encoder_observation(qpos_50hz, frame_index)
        tokens.append(
            encoder_session.run(["encoded_tokens"], {"obs_dict": observation[None, :]})[
                0
            ][0]
        )
    return np.asarray(tokens, dtype=np.float32).reshape(-1, SONIC_TOKEN_DIM)


def load_bones_seed_reference_qpos(csv_path: Path) -> tuple[np.ndarray, int]:
    """Return the 50 Hz physical reference and its encoded token count."""

    qpos_120hz = load_bones_seed_qpos(csv_path)
    num_tokens = int(np.ceil(len(qpos_120hz) * SONIC_CONTROL_HZ / BONES_SEED_SOURCE_HZ))
    return (
        resample_qpos_to_50hz(
            qpos_120hz,
            num_tokens + SONIC_REFERENCE_LOOKAHEAD,
            source_hz=BONES_SEED_SOURCE_HZ,
        ),
        num_tokens,
    )


def canonicalize_reference_heading(qpos: np.ndarray) -> np.ndarray:
    """Remove a motion's global starting yaw while preserving pose dynamics."""

    canonical = np.asarray(qpos, dtype=np.float32).copy()
    inverse_heading = _quat_conjugate(_heading_quaternion(canonical[0, 3:7]))
    canonical[:, 3:7] = np.asarray(
        [_quat_multiply(inverse_heading, frame[3:7]) for frame in canonical]
    )
    origin = np.asarray([canonical[0, 0], canonical[0, 1], 0.0])
    canonical[:, :3] = Rotation.from_quat(
        inverse_heading[[1, 2, 3, 0]]
    ).apply(canonical[:, :3] - origin)
    return canonical


def straighten_reference_heading(qpos: np.ndarray) -> np.ndarray:
    """Keep each pose's tilt while removing global yaw and lateral drift."""

    straight = np.asarray(qpos, dtype=np.float32).copy()
    straight[:, 3:7] = np.asarray(
        [
            _quat_multiply(_quat_conjugate(_heading_quaternion(frame[3:7])), frame[3:7])
            for frame in straight
        ]
    )
    straight[:, 1] = straight[0, 1]
    return straight
