"""SONIC-aligned G1 task environments.

The robot, PD controller, and 64D SONIC interface are shared by all tasks.
The raw baseline uses the same scene and controller with a direct 29D body
target, so the two policies differ only at the policy interface.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
import onnxruntime as ort
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from humanoid_bench.dmc_deps import dmc_index
from humanoid_bench.dmc_deps.dmc_wrapper import MjDataWrapper, MjModelWrapper
from humanoid_bench.env import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_RANDOMNESS,
    NamedIndexStructs,
    TASKS,
)
from humanoid_bench.robots import G1

# Apply the HumanoidBench family's MuJoCo 3.3 named-index compatibility patch.
from latency_bench.envs import humanoidbench as humanoidbench_family

G1_CONTROL_MODE_RAW = "raw_body_target"
G1_CONTROL_MODE_SONIC = "sonic_token"
G1_CONTROL_MODE_SONIC_RESIDUAL = "sonic_token_residual"
SONIC_DECODER_MODE_ENVIRONMENT = "environment"
SONIC_DECODER_MODE_INFERENCE_WORKER = "inference_worker"

_NO_RESET_NOISE_TASKS = {
    "bookshelf_hard",
    "bookshelf_simple",
    "cube",
    "kitchen",
}

G1_SONIC_UNSUPPORTED_TASKS = ("highbar_simple", "highbar_hard")
G1_SONIC_SUPPORTED_TASKS = tuple(
    task_key for task_key in TASKS if task_key not in G1_SONIC_UNSUPPORTED_TASKS
)


def _task_id_suffix(task_key: str) -> str:
    return "".join(word.capitalize() for word in task_key.split("_"))


G1_SONIC_ENV_SPECS = {
    f"g1_sonic_{task_key}": {
        "env_id": f"LatencyBench/G1Sonic{_task_id_suffix(task_key)}-v0",
        "scene_filename": f"sonic_g1/envs/scene_{task_key}.xml",
        "horizon": TASKS[task_key].max_episode_steps,
    }
    for task_key in G1_SONIC_SUPPORTED_TASKS
}

_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

_HAND_KP = np.ones(14, dtype=np.float64)
_HAND_KD = np.full(14, 0.1, dtype=np.float64)

# MuJoCo joint i consumes the corresponding IsaacLab-indexed decoder output.
_POLICY_INDEX_FOR_MUJOCO = np.asarray(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)
# Reorder MuJoCo state vectors into the decoder's IsaacLab ordering.
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
    dtype=np.float64,
)

_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_NATURAL_FREQ = 10.0 * 2.0 * np.pi
_DAMPING_RATIO = 2.0
_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ
_ACTION_SCALE = np.asarray(
    [
        0.25 * 139.0 / _STIFFNESS_7520_22,
        0.25 * 139.0 / _STIFFNESS_7520_22,
        0.25 * 88.0 / _STIFFNESS_7520_14,
        0.25 * 139.0 / _STIFFNESS_7520_22,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 139.0 / _STIFFNESS_7520_22,
        0.25 * 139.0 / _STIFFNESS_7520_22,
        0.25 * 88.0 / _STIFFNESS_7520_14,
        0.25 * 139.0 / _STIFFNESS_7520_22,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 88.0 / _STIFFNESS_7520_14,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 5.0 / _STIFFNESS_4010,
        0.25 * 5.0 / _STIFFNESS_4010,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 25.0 / _STIFFNESS_5020,
        0.25 * 5.0 / _STIFFNESS_4010,
        0.25 * 5.0 / _STIFFNESS_4010,
    ],
    dtype=np.float64,
)
_KP = np.asarray(
    [
        _STIFFNESS_7520_22,
        _STIFFNESS_7520_22,
        _STIFFNESS_7520_14,
        _STIFFNESS_7520_22,
        2 * _STIFFNESS_5020,
        2 * _STIFFNESS_5020,
        _STIFFNESS_7520_22,
        _STIFFNESS_7520_22,
        _STIFFNESS_7520_14,
        _STIFFNESS_7520_22,
        2 * _STIFFNESS_5020,
        2 * _STIFFNESS_5020,
        _STIFFNESS_7520_14,
        2 * _STIFFNESS_5020,
        2 * _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_4010,
        _STIFFNESS_4010,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_5020,
        _STIFFNESS_4010,
        _STIFFNESS_4010,
    ],
    dtype=np.float64,
)
_KD = np.asarray(
    [
        _DAMPING_7520_22,
        _DAMPING_7520_22,
        _DAMPING_7520_14,
        _DAMPING_7520_22,
        2 * _DAMPING_5020,
        2 * _DAMPING_5020,
        _DAMPING_7520_22,
        _DAMPING_7520_22,
        _DAMPING_7520_14,
        _DAMPING_7520_22,
        2 * _DAMPING_5020,
        2 * _DAMPING_5020,
        _DAMPING_7520_14,
        2 * _DAMPING_5020,
        2 * _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_4010,
        _DAMPING_4010,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_5020,
        _DAMPING_4010,
        _DAMPING_4010,
    ],
    dtype=np.float64,
)


def _rotate_by_quaternion(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    qvec = np.asarray([x, y, z], dtype=np.float64)
    return vector + 2.0 * (
        w * np.cross(qvec, vector) + np.cross(qvec, np.cross(qvec, vector))
    )


class G1SonicEnv(MujocoEnv):
    """G1 task with either direct body targets or SONIC tokens."""

    metadata: ClassVar[dict[str, Any]] = humanoidbench_family.HumanoidBenchEnv.metadata

    def __init__(
        self,
        *,
        scene_path: str,
        task_name: str,
        control_mode: str = G1_CONTROL_MODE_SONIC,
        sonic_decoder_path: str | None = None,
        sonic_decoder_mode: str = SONIC_DECODER_MODE_ENVIRONMENT,
        sonic_base_token_path: str | None = None,
        sonic_residual_scale: float | None = None,
        render_mode: str,
        width: int,
        height: int,
    ) -> None:
        self._task_name = task_name
        self._task_key = task_name.removeprefix("g1_sonic_")
        self._control_mode = control_mode
        self.robot = G1(self)
        # HumanoidBench's robot.dof means qpos length, not actuator count.
        self.robot.dof = 50
        task_class = TASKS[self._task_key]
        task_info = task_class(self.robot, None)
        policy_observation_space = task_info.observation_space
        residual_control = control_mode == G1_CONTROL_MODE_SONIC_RESIDUAL
        if residual_control:
            self._base_tokens = np.load(sonic_base_token_path)
            self._residual_scale = sonic_residual_scale
            self._base_token_cursor = 0
            packed_observation_space = Box(
                -np.inf,
                np.inf,
                shape=(policy_observation_space.shape[0] + 930 + 64,),
                dtype=policy_observation_space.dtype,
            )
            observation_space = DictSpace(
                {
                    "obs": packed_observation_space,
                    "sonic_state": Box(-np.inf, np.inf, shape=(930,), dtype=np.float32),
                    "base_token": Box(-np.inf, np.inf, shape=(64,), dtype=np.float32),
                }
            )
        else:
            observation_space = (
                DictSpace(
                    {
                        "obs": policy_observation_space,
                        "sonic_state": Box(
                            -np.inf, np.inf, shape=(930,), dtype=np.float32
                        ),
                    }
                )
                if sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER
                else policy_observation_space
            )
        super().__init__(
            model_path=scene_path,
            frame_skip=task_info.frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )
        if control_mode == G1_CONTROL_MODE_RAW:
            action_dim = 43
        elif residual_control:
            action_dim = 64
        else:
            action_dim = 78
        self.action_space = Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = observation_space
        self._sonic_decoder_mode = sonic_decoder_mode
        self.action_low = np.full(action_dim, -1.0, dtype=np.float64)
        self.action_high = np.full(action_dim, 1.0, dtype=np.float64)

        if (
            control_mode in (G1_CONTROL_MODE_SONIC, G1_CONTROL_MODE_SONIC_RESIDUAL)
            and sonic_decoder_mode == SONIC_DECODER_MODE_ENVIRONMENT
        ):
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._decoder = ort.InferenceSession(
                sonic_decoder_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._decoder_input_name = self._decoder.get_inputs()[0].name
            self._decoder_output_name = self._decoder.get_outputs()[0].name

        self._body_qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in _BODY_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._body_qvel_addresses = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in _BODY_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._hand_qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in _HAND_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._hand_qvel_addresses = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in _HAND_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._body_actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name.removesuffix("_joint"),
                )
                for name in _BODY_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._hand_actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name.removesuffix("_joint"),
                )
                for name in _HAND_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._hand_joint_ranges = self.model.jnt_range[
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in _HAND_JOINT_NAMES
            ]
        ].copy()
        self._body_joint_ranges = self.model.jnt_range[
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in _BODY_JOINT_NAMES
            ]
        ].copy()
        self._pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        named_data = MjDataWrapper(self.data)
        named_model = MjModelWrapper(self.model)
        axis_indexers = dmc_index.make_axis_indexers(named_model)
        self.named = NamedIndexStructs(
            model=dmc_index.struct_indexer(named_model, "mjmodel", axis_indexers),
            data=dmc_index.struct_indexer(
                named_data,
                "mjdata",
                axis_indexers,
            ),
        )
        self._task = task_class(self.robot, self)
        self.randomness = (
            0 if self._task_key in _NO_RESET_NOISE_TASKS else DEFAULT_RANDOMNESS
        )
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=10)
        self._last_action = np.zeros(29, dtype=np.float64)
        self._pending_decoded_action = None

    def _body_state(self) -> tuple[np.ndarray, np.ndarray]:
        q_mu = self.data.qpos[self._body_qpos_addresses].copy()
        dq_mu = self.data.qvel[self._body_qvel_addresses].copy()
        return (q_mu - _DEFAULT_ANGLES)[_MUJOCO_INDEX_FOR_POLICY], dq_mu[
            _MUJOCO_INDEX_FOR_POLICY
        ]

    def _frame(self) -> dict[str, np.ndarray]:
        body_q, body_dq = self._body_state()
        root_quaternion = self.data.qpos[3:7]
        gravity = _rotate_by_quaternion(
            np.asarray(
                [
                    root_quaternion[0],
                    -root_quaternion[1],
                    -root_quaternion[2],
                    -root_quaternion[3],
                ]
            ),
            np.asarray([0.0, 0.0, -1.0]),
        )
        base_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._pelvis_body_id,
            base_velocity,
            1,
        )
        return {
            "angular_velocity": base_velocity[:3].copy(),
            "body_q": body_q,
            "body_dq": body_dq,
            "last_action": self._last_action.copy(),
            "gravity": gravity,
        }

    def _decoder_state(self) -> np.ndarray:
        frames = list(self._history)
        frames = [
            {
                "angular_velocity": np.zeros(3, dtype=np.float64),
                "body_q": np.zeros(29, dtype=np.float64),
                "body_dq": np.zeros(29, dtype=np.float64),
                "last_action": np.zeros(29, dtype=np.float64),
                "gravity": np.zeros(3, dtype=np.float64),
            }
        ] * (10 - len(frames)) + frames
        return np.concatenate(
            [
                np.concatenate([frame["angular_velocity"] for frame in frames]),
                np.concatenate([frame["body_q"] for frame in frames]),
                np.concatenate([frame["body_dq"] for frame in frames]),
                np.concatenate([frame["last_action"] for frame in frames]),
                np.concatenate([frame["gravity"] for frame in frames]),
            ],
        ).astype(np.float32)

    def _decoder_input(self, token: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(token, dtype=np.float32), self._decoder_state()]
        )

    def _decode(self, token: np.ndarray) -> np.ndarray:
        decoder_input = self._decoder_input(token)
        return np.asarray(
            self._decoder.run(
                [self._decoder_output_name],
                {self._decoder_input_name: decoder_input[None, :]},
            )[0][0],
            dtype=np.float64,
        )

    def _controller(
        self,
        body_action_mu: np.ndarray,
        hand_action: np.ndarray | None,
        *,
        raw_body: bool = False,
    ) -> np.ndarray:
        if raw_body:
            target_mu = self._body_joint_ranges[:, 0] + (body_action_mu + 1.0) * 0.5 * (
                self._body_joint_ranges[:, 1] - self._body_joint_ranges[:, 0]
            )
        else:
            target_mu = _DEFAULT_ANGLES + body_action_mu * _ACTION_SCALE
        q_mu = self.data.qpos[self._body_qpos_addresses]
        dq_mu = self.data.qvel[self._body_qvel_addresses]
        hand_target = (
            np.zeros(14, dtype=np.float64)
            if hand_action is None
            else self._hand_joint_ranges[:, 0]
            + (hand_action + 1.0)
            * 0.5
            * (self._hand_joint_ranges[:, 1] - self._hand_joint_ranges[:, 0])
        )
        hand_q = self.data.qpos[self._hand_qpos_addresses]
        hand_dq = self.data.qvel[self._hand_qvel_addresses]
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[self._body_actuator_ids] = _KP * (target_mu - q_mu) - _KD * dq_mu
        ctrl[self._hand_actuator_ids] = (
            _HAND_KP * (hand_target - hand_q) - _HAND_KD * hand_dq
        )
        return np.clip(
            ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def set_decoded_action(self, action: np.ndarray) -> None:
        """Supply the body action decoded by Sample Factory's inference worker."""

        self._pending_decoded_action = np.asarray(action, dtype=np.float64)

    def _observation(self, task_observation: np.ndarray):
        if self._control_mode == G1_CONTROL_MODE_SONIC_RESIDUAL:
            sonic_state = self._decoder_state()
            base_token = self._base_tokens[self._base_token_cursor]
            return {
                "obs": np.concatenate([task_observation, sonic_state, base_token]),
                "sonic_state": sonic_state,
                "base_token": base_token,
            }
        if self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
            return {"obs": task_observation, "sonic_state": self._decoder_state()}
        return task_observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        return super().reset(seed=seed, options=options)

    def reset_model(self) -> Any:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        if self._control_mode == G1_CONTROL_MODE_SONIC_RESIDUAL:
            init_qpos[2] = 0.76
            init_qpos[self._body_qpos_addresses] = _DEFAULT_ANGLES
            init_qpos[self._hand_qpos_addresses] = 0.0
            randomness = 0.0
            self._base_token_cursor = 0
        else:
            randomness = self.randomness
        self.set_state(
            init_qpos
            + self.np_random.uniform(-randomness, randomness, size=self.model.nq),
            init_qvel,
        )
        self._last_action.fill(0.0)
        self._pending_decoded_action = None
        self._history.clear()
        task_observation = self._task.reset_model()
        self._history.append(self._frame())
        return self._observation(task_observation)

    def step(self, action: Any):
        action = np.asarray(action, dtype=np.float32)
        if self._control_mode == G1_CONTROL_MODE_RAW:
            body_action_mu = action[:29].astype(np.float64)
            decoder_action = body_action_mu[_MUJOCO_INDEX_FOR_POLICY]
            hand_action = action[29:43].astype(np.float64)
            raw_body = True
        elif self._control_mode == G1_CONTROL_MODE_SONIC_RESIDUAL:
            if self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
                decoder_action = self._pending_decoded_action
                self._pending_decoded_action = None
            else:
                token = (
                    self._base_tokens[self._base_token_cursor]
                    + self._residual_scale * action
                )
                decoder_action = self._decode(token)
            body_action_mu = decoder_action[_POLICY_INDEX_FOR_MUJOCO]
            hand_action = None
            raw_body = False
        elif self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
            decoder_action = self._pending_decoded_action
            self._pending_decoded_action = None
            body_action_mu = decoder_action[_POLICY_INDEX_FOR_MUJOCO]
            hand_action = action[64:78].astype(np.float64)
            raw_body = False
        else:
            decoder_action = self._decode(action[:64])
            body_action_mu = decoder_action[_POLICY_INDEX_FOR_MUJOCO]
            hand_action = action[64:78].astype(np.float64)
            raw_body = False
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._controller(
                body_action_mu, hand_action, raw_body=raw_body
            )
            mujoco.mj_step(self.model, self.data)
        self._last_action[:] = decoder_action
        self._history.append(self._frame())
        if self._control_mode == G1_CONTROL_MODE_SONIC_RESIDUAL:
            self._base_token_cursor += 1
        task_observation = self._task.get_obs()
        reward, info = self._task.get_reward()
        terminated, termination_info = self._task.get_terminated()
        return (
            self._observation(task_observation),
            reward,
            bool(terminated),
            False,
            {"per_timestep_reward": reward, **info, **termination_info},
        )

    def render(self):
        return self._task.render()


for _task_name, _spec in G1_SONIC_ENV_SPECS.items():
    gym.register(
        id=_spec["env_id"],
        entry_point=G1SonicEnv,
        max_episode_steps=_spec["horizon"],
        kwargs={
            "task_name": _task_name,
            "render_mode": "rgb_array",
            "width": 256,
            "height": 256,
        },
    )
