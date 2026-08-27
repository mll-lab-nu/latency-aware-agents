"""SONIC-aligned G1 task environments.

The robot, PD controller, and 64D SONIC interface are shared by all tasks.
The raw baseline uses the same scene and controller with a direct 29D body
target and optional hand controls, so the two policies differ only at the
policy interface.
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
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_BODY_JOINT_NAMES,
    G1_DEFAULT_ANGLES,
    MUJOCO_INDEX_FOR_POLICY,
    POLICY_INDEX_FOR_MUJOCO,
)
from latency_bench.envs.humanoidbench_sonic_teacher import (
    DEFAULT_HURDLE_POSITIONS,
    JUMP_STATE,
    RUN_STATE,
    SONIC_CONTROL_HZ,
    match_reference_phase,
)
from latency_bench.envs.humanoidbench_sonic_reference import (
    build_encoder_observation,
)

G1_CONTROL_MODE_RAW = "raw_body_target"
G1_CONTROL_MODE_SONIC = "sonic_token"
G1_CONTROL_MODE_SONIC_REFERENCE = "sonic_token_reference"
G1_CONTROL_MODE_SONIC_COMPOSED = "sonic_token_composed"
SONIC_DECODER_MODE_ENVIRONMENT = "environment"
SONIC_DECODER_MODE_INFERENCE_WORKER = "inference_worker"
SONIC_REFERENCE_RESIDUAL_SCALE = 0.25
SONIC_COMPOSED_PHYSICAL_RESIDUAL_SCALE = 0.15
SONIC_COMPOSED_TRIGGER_DISTANCE = 1.0
SONIC_COMPOSED_JUMP_START_PHASE = 30
SONIC_COMPOSED_RUN_MATCH_RADIUS = 25

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


def g1_raw_reset_action(scene_path: str, *, learn_hands: bool) -> list[float]:
    """Return normalized raw targets for the pose used by ``reset_model``."""

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    joint_names = G1_BODY_JOINT_NAMES + (_HAND_JOINT_NAMES if learn_hands else ())
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    reset_positions = (
        np.concatenate(
            [G1_DEFAULT_ANGLES, np.zeros(len(_HAND_JOINT_NAMES), dtype=np.float64)]
        )
        if learn_hands
        else G1_DEFAULT_ANGLES
    )
    ranges = model.jnt_range[joint_ids]
    return (
        2.0 * (reset_positions - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1.0
    ).tolist()


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
        sonic_encoder_path: str | None = None,
        sonic_decoder_mode: str = SONIC_DECODER_MODE_ENVIRONMENT,
        sonic_reference_bundle_path: str | None = None,
        sonic_initial_reference_qpos: np.ndarray | None = None,
        raw_learn_hands: bool = True,
        render_mode: str,
        width: int,
        height: int,
        hurdle_height: float = 0.375,
        hurdle_count: int = 10,
        hurdle_horizon: int = 1000,
    ) -> None:
        self._task_name = task_name
        self._task_key = task_name.removeprefix("g1_sonic_")
        self._control_mode = control_mode
        self._sonic_initial_reference_qpos = sonic_initial_reference_qpos
        self._raw_learn_hands = raw_learn_hands
        self.robot = G1(self)
        # HumanoidBench's robot.dof means qpos length, not actuator count.
        self.robot.dof = 50
        task_class = TASKS[self._task_key]
        task_info = task_class(self.robot, None)
        policy_observation_space = task_info.observation_space
        composed_control = control_mode == G1_CONTROL_MODE_SONIC_COMPOSED
        reference_control = control_mode == G1_CONTROL_MODE_SONIC_REFERENCE
        if composed_control:
            reference_bundle = np.load(sonic_reference_bundle_path)
            self._run_encoder_qpos = np.asarray(
                reference_bundle["run_encoder_qpos"], dtype=np.float32
            )
            self._jump_encoder_qpos = np.asarray(
                reference_bundle["jump_encoder_qpos"], dtype=np.float32
            )
            self._run_reference_qpos = np.asarray(
                reference_bundle["run_reference_qpos"], dtype=np.float32
            )
            self._jump_reference_qpos = np.asarray(
                reference_bundle["jump_reference_qpos"], dtype=np.float32
            )
            self._run_reference_foot_contacts = np.asarray(
                reference_bundle["run_reference_foot_contacts"]
            )
            self._run_reference_joint_velocities = (
                np.gradient(self._run_reference_qpos[:, 7:], axis=0) * SONIC_CONTROL_HZ
            )
            self._reference_mode = RUN_STATE
            self._run_phase = 0
            self._jump_phase = 0
            self._paused_run_phase = 0
            self._next_hurdle_index = 0
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._encoder = ort.InferenceSession(
                sonic_encoder_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._encoder_input_name = self._encoder.get_inputs()[0].name
            self._encoder_output_name = self._encoder.get_outputs()[0].name
            packed_observation_space = Box(
                -np.inf,
                np.inf,
                shape=(policy_observation_space.shape[0] + 930 + 64 + 2,),
                dtype=policy_observation_space.dtype,
            )
            observation_space = DictSpace(
                {
                    "obs": packed_observation_space,
                    "sonic_state": Box(-np.inf, np.inf, shape=(930,), dtype=np.float32),
                    "base_token": Box(-np.inf, np.inf, shape=(64,), dtype=np.float32),
                }
            )
        elif reference_control:
            reference_bundle = np.load(sonic_reference_bundle_path)
            self._reference_tokens = np.asarray(
                reference_bundle["reference_tokens"], dtype=np.float32
            )
            self._reference_qpos = np.asarray(
                reference_bundle["reference_qpos"], dtype=np.float32
            )
            self._reference_phase = 0
            packed_observation_space = Box(
                -np.inf,
                np.inf,
                shape=(policy_observation_space.shape[0] + 930 + 64 + 1,),
                dtype=policy_observation_space.dtype,
            )
            observation_space = DictSpace(
                {
                    "obs": packed_observation_space,
                    "sonic_state": Box(-np.inf, np.inf, shape=(930,), dtype=np.float32),
                    "reference_token": Box(
                        -np.inf, np.inf, shape=(64,), dtype=np.float32
                    ),
                    "reference_phase": Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
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
            action_dim = 29 + (14 if raw_learn_hands else 0)
            self.action_space = Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        elif composed_control:
            action_dim = len(G1_BODY_JOINT_NAMES)
            self.action_space = Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        else:
            action_dim = 78
            self.action_space = Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = observation_space
        self._sonic_decoder_mode = sonic_decoder_mode
        self.action_low = np.full(action_dim, -1.0, dtype=np.float64)
        self.action_high = np.full(action_dim, 1.0, dtype=np.float64)

        if (
            control_mode
            in (
                G1_CONTROL_MODE_SONIC,
                G1_CONTROL_MODE_SONIC_REFERENCE,
                G1_CONTROL_MODE_SONIC_COMPOSED,
            )
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
                for name in G1_BODY_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._body_qvel_addresses = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in G1_BODY_JOINT_NAMES
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
                for name in G1_BODY_JOINT_NAMES
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
                for name in G1_BODY_JOINT_NAMES
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
        self._task = (
            task_class(
                self.robot,
                self,
                hurdle_count=hurdle_count,
                hurdle_horizon=hurdle_horizon,
            )
            if self._task_key == "hurdle"
            else task_class(self.robot, self)
        )
        if self._task_key == "hurdle":
            for geom_id in self._task._hurdle_geom_ids:
                self.model.geom_size[geom_id, 2] = hurdle_height * 0.5
                self.model.geom_pos[geom_id, 2] = hurdle_height * 0.5
            for body_id in range(self.model.nbody):
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if body_name is not None and body_name.startswith("starting_hurdle_"):
                    geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
                    self.model.geom_size[geom_ids[2:4], 2] = hurdle_height * 0.5
                    self.model.geom_pos[geom_ids[2:4], 2] = hurdle_height * 0.5
                    self.model.geom_pos[geom_ids[4], 2] = (
                        hurdle_height - self.model.geom_size[geom_ids[4], 2]
                    )
            mujoco.mj_forward(self.model, self.data)
        self.randomness = (
            0 if self._task_key in _NO_RESET_NOISE_TASKS else DEFAULT_RANDOMNESS
        )
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=10)
        self._last_action = np.zeros(29, dtype=np.float64)
        self._pending_decoded_action = None
        self._composed_residual_abs_sum = 0.0
        self._composed_residual_abs_max = 0.0
        self._composed_residual_component_count = 0
        self._composed_residual_saturated_count = 0
        self._previous_composed_residual = np.zeros(
            len(G1_BODY_JOINT_NAMES), dtype=np.float64
        )
        self._base_token = np.zeros(64, dtype=np.float32)

    def _body_state(self) -> tuple[np.ndarray, np.ndarray]:
        q_mu = self.data.qpos[self._body_qpos_addresses].copy()
        dq_mu = self.data.qvel[self._body_qvel_addresses].copy()
        return (q_mu - G1_DEFAULT_ANGLES)[MUJOCO_INDEX_FOR_POLICY], dq_mu[
            MUJOCO_INDEX_FOR_POLICY
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
            target_mu = G1_DEFAULT_ANGLES + body_action_mu * _ACTION_SCALE
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

    def _foot_contacts(self) -> np.ndarray:
        return np.asarray(
            [
                self.data.site_xpos[self.model.site(name).id, 2] <= 0.02
                for name in ("left_foot", "right_foot")
            ]
        )

    def _reference_state(self) -> tuple[str, int]:
        if self._reference_mode == RUN_STATE:
            return RUN_STATE, self._run_phase
        return JUMP_STATE, self._jump_phase

    def _reference_phase_features(self) -> np.ndarray:
        mode, phase = self._reference_state()
        phase_count = (
            len(self._run_reference_qpos)
            if mode == RUN_STATE
            else len(self._jump_reference_qpos)
        )
        return np.asarray(
            [float(mode == JUMP_STATE), phase / phase_count],
            dtype=np.float32,
        )

    def _reference_observation(self) -> tuple[np.ndarray, np.ndarray]:
        token = self._reference_tokens[self._reference_phase]
        phase = np.asarray(
            [self._reference_phase / len(self._reference_tokens)], dtype=np.float32
        )
        return token, phase

    def _reference_info(self) -> dict[str, float]:
        reference_qpos = self._reference_qpos[self._reference_phase]
        actual_qpos = self.data.qpos[self._body_qpos_addresses]
        return {
            "reference_phase": float(self._reference_phase),
            "reference_joint_rmse": float(
                np.sqrt(np.mean((actual_qpos - reference_qpos[7:]) ** 2))
            ),
        }

    def _online_reference_token(self) -> np.ndarray:
        mode, phase = self._reference_state()
        reference_qpos = (
            self._run_encoder_qpos if mode == RUN_STATE else self._jump_encoder_qpos
        )
        encoder_observation = build_encoder_observation(
            reference_qpos,
            phase,
            actual_root_quaternion=self.data.qpos[3:7],
        )
        return np.asarray(
            self._encoder.run(
                [self._encoder_output_name],
                {self._encoder_input_name: encoder_observation[None, :]},
            )[0][0],
            dtype=np.float32,
        )

    def _match_reference_phase(self, candidate_phases=None) -> int:
        actual_joint_positions = self.data.qpos[self._body_qpos_addresses]
        actual_joint_velocities = self.data.qvel[self._body_qvel_addresses]
        actual_foot_contacts = self._foot_contacts()
        return match_reference_phase(
            actual_joint_positions,
            actual_joint_velocities,
            actual_foot_contacts,
            self._run_reference_qpos,
            self._run_reference_joint_velocities,
            self._run_reference_foot_contacts,
            candidate_phases=candidate_phases,
        )

    def _start_jump(self) -> None:
        self._paused_run_phase = self._run_phase
        self._jump_phase = SONIC_COMPOSED_JUMP_START_PHASE
        self._reference_mode = JUMP_STATE

    def _start_run(self) -> None:
        self._reference_mode = RUN_STATE
        phase_start = max(0, self._paused_run_phase - SONIC_COMPOSED_RUN_MATCH_RADIUS)
        phase_stop = min(
            len(self._run_reference_qpos),
            self._paused_run_phase + SONIC_COMPOSED_RUN_MATCH_RADIUS + 1,
        )
        self._run_phase = self._match_reference_phase(
            range(phase_start, phase_stop)
        )

    def _advance_hurdle_index(self) -> None:
        root_x = self.data.qpos[0]
        while (
            self._next_hurdle_index < len(DEFAULT_HURDLE_POSITIONS)
            and root_x > DEFAULT_HURDLE_POSITIONS[self._next_hurdle_index]
        ):
            self._next_hurdle_index += 1

    def _prepare_composed_action(self) -> np.ndarray:
        if self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
            decoder_action = self._pending_decoded_action
            self._pending_decoded_action = None
        else:
            decoder_action = self._decode(self._base_token)
        return decoder_action

    def _maybe_trigger_jump(self) -> None:
        self._advance_hurdle_index()
        if self._reference_mode == RUN_STATE and self._next_hurdle_index < len(
            DEFAULT_HURDLE_POSITIONS
        ):
            distance = (
                DEFAULT_HURDLE_POSITIONS[self._next_hurdle_index] - self.data.qpos[0]
            )
            if 0.0 <= distance <= SONIC_COMPOSED_TRIGGER_DISTANCE:
                self._start_jump()

    def _composed_residual_info(self) -> dict[str, float]:
        component_count = self._composed_residual_component_count
        return {
            "residual_abs_mean": self._composed_residual_abs_sum / component_count,
            "residual_abs_max": self._composed_residual_abs_max,
            "residual_saturation_fraction": (
                self._composed_residual_saturated_count / component_count
            ),
        }

    def _advance_composed_reference(self) -> None:
        if self._reference_mode == RUN_STATE:
            self._run_phase += 1
            return
        self._jump_phase += 1
        if self._jump_phase == len(self._jump_reference_qpos):
            self._start_run()
            self._jump_phase = 0

    def _observation(self, task_observation: np.ndarray):
        if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED:
            sonic_state = self._decoder_state()
            self._base_token = self._online_reference_token()
            return {
                "obs": np.concatenate(
                    [task_observation, sonic_state, self._base_token, self._reference_phase_features()]
                ),
                "sonic_state": sonic_state,
                "base_token": self._base_token,
            }
        if self._control_mode == G1_CONTROL_MODE_SONIC_REFERENCE:
            sonic_state = self._decoder_state()
            reference_token, reference_phase = self._reference_observation()
            return {
                "obs": np.concatenate(
                    [task_observation, sonic_state, reference_token, reference_phase]
                ),
                "sonic_state": sonic_state,
                "reference_token": reference_token,
                "reference_phase": reference_phase,
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
        if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED:
            init_qpos[:7] = self._run_reference_qpos[0, :7]
            init_qpos[self._body_qpos_addresses] = self._run_reference_qpos[0, 7:]
            init_qpos[self._hand_qpos_addresses] = 0.0
        elif self._control_mode == G1_CONTROL_MODE_SONIC_REFERENCE:
            init_qpos[self._body_qpos_addresses] = self._reference_qpos[0, 7:]
            init_qpos[self._hand_qpos_addresses] = 0.0
        elif self._sonic_initial_reference_qpos is not None:
            init_qpos[2:7] = self._sonic_initial_reference_qpos[2:7]
            init_qpos[self._body_qpos_addresses] = (
                self._sonic_initial_reference_qpos[7:]
            )
            init_qpos[self._hand_qpos_addresses] = 0.0
        elif self._control_mode != G1_CONTROL_MODE_SONIC:
            init_qpos[self._body_qpos_addresses] = G1_DEFAULT_ANGLES
            init_qpos[self._hand_qpos_addresses] = 0.0
        randomness = self.randomness
        if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED:
            self._reference_mode = RUN_STATE
            self._run_phase = 0
            self._jump_phase = 0
            self._paused_run_phase = 0
            self._next_hurdle_index = 0
        if self._control_mode == G1_CONTROL_MODE_SONIC_REFERENCE:
            self._reference_phase = 0
        self.set_state(
            init_qpos
            + self.np_random.uniform(-randomness, randomness, size=self.model.nq),
            init_qvel,
        )
        self._last_action.fill(0.0)
        self._pending_decoded_action = None
        self._composed_residual_abs_sum = 0.0
        self._composed_residual_abs_max = 0.0
        self._composed_residual_component_count = 0
        self._composed_residual_saturated_count = 0
        self._previous_composed_residual.fill(0.0)
        self._history.clear()
        task_observation = self._task.reset_model()
        self._history.append(self._frame())
        return self._observation(task_observation)

    def step(self, action: Any):
        action = np.asarray(action, dtype=np.float32)
        residual_smoothness_penalty = 0.0
        if self._control_mode == G1_CONTROL_MODE_RAW:
            body_action_mu = action[:29].astype(np.float64)
            decoder_action = body_action_mu[MUJOCO_INDEX_FOR_POLICY]
            hand_action = (
                action[29:43].astype(np.float64) if self._raw_learn_hands else None
            )
            raw_body = True
        elif self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED:
            decoder_action = self._prepare_composed_action()
            body_action_mu = decoder_action[POLICY_INDEX_FOR_MUJOCO]
            joint_target_residual = action * SONIC_COMPOSED_PHYSICAL_RESIDUAL_SCALE
            residual_smoothness_penalty = 0.001 * np.mean(
                (action - self._previous_composed_residual) ** 2
            )
            self._previous_composed_residual = action
            body_action_mu += joint_target_residual / _ACTION_SCALE
            self._composed_residual_abs_sum += np.abs(action).sum()
            self._composed_residual_abs_max = max(
                self._composed_residual_abs_max,
                np.abs(action).max(),
            )
            self._composed_residual_component_count += action.size
            self._composed_residual_saturated_count += np.count_nonzero(
                np.abs(action) >= 0.999
            )
            hand_action = None
            raw_body = False
        elif self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
            decoder_action = self._pending_decoded_action
            self._pending_decoded_action = None
            body_action_mu = decoder_action[POLICY_INDEX_FOR_MUJOCO]
            hand_action = action[64:78].astype(np.float64)
            raw_body = False
        else:
            token = action[:64]
            if self._control_mode == G1_CONTROL_MODE_SONIC_REFERENCE:
                token = self._reference_tokens[self._reference_phase] + (
                    SONIC_REFERENCE_RESIDUAL_SCALE * token
                )
            decoder_action = self._decode(token)
            body_action_mu = decoder_action[POLICY_INDEX_FOR_MUJOCO]
            hand_action = action[64:78].astype(np.float64)
            raw_body = False
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._controller(
                body_action_mu,
                hand_action,
                raw_body=raw_body,
            )
            mujoco.mj_step(self.model, self.data)
            if self._task_key == "hurdle":
                self._task._record_contacts()
        if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED:
            self._last_action[:] = body_action_mu[MUJOCO_INDEX_FOR_POLICY]
        else:
            self._last_action[:] = decoder_action
        self._history.append(self._frame())
        reference_info = {}
        if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED:
            was_running = self._reference_mode == RUN_STATE
            self._advance_composed_reference()
            if was_running:
                self._maybe_trigger_jump()
        elif self._control_mode == G1_CONTROL_MODE_SONIC_REFERENCE:
            reference_info = self._reference_info()
            self._reference_phase = (self._reference_phase + 1) % len(
                self._reference_tokens
            )
        task_observation = self._task.get_obs()
        reward, info = self._task.get_reward()
        if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED and (
            info["failure_penalty"]
            or info["quality_failure_penalty"]
            or info["timeout_penalty"]
        ):
            residual_smoothness_penalty = 0.0
        else:
            reward -= residual_smoothness_penalty
        terminated, termination_info = self._task.get_terminated()
        return (
            self._observation(task_observation),
            reward,
            bool(terminated),
            False,
            {
                "per_timestep_reward": reward,
                **info,
                **termination_info,
                "residual_smoothness_penalty": residual_smoothness_penalty,
                **(
                    self._composed_residual_info()
                    if self._control_mode == G1_CONTROL_MODE_SONIC_COMPOSED
                    else {}
                ),
                **reference_info,
            },
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
