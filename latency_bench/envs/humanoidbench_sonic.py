"""SONIC-aligned G1 task environments.

The robot, PD controller, and 64D SONIC interface are shared by all tasks.
The raw baseline uses the same scene and controller with a direct body
target and optional hand controls, so the two policies differ only at the
policy interface.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
import onnxruntime as ort
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco import mujoco_rendering
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from humanoid_bench.dmc_deps import dmc_index, dmc_sizes
from humanoid_bench.dmc_deps.dmc_wrapper import MjDataWrapper, MjModelWrapper
from humanoid_bench.env import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_RANDOMNESS,
    HumanoidEnv,
    NamedIndexStructs,
    TASKS,
)
from humanoid_bench.robots import G1

from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_BODY_JOINT_NAMES,
    G1_DEFAULT_ANGLES,
    G1_GR00T_STATE_DIM,
    G1_HAND_JOINT_NAMES,
    MUJOCO_INDEX_FOR_POLICY,
    POLICY_INDEX_FOR_MUJOCO,
)


# HumanoidBench's generated table targets MuJoCo 3.1.6. These fields were
# removed or renamed in MuJoCo 3.3 before the named indexers are constructed.
for _struct_name, _field_names in {
    "mjmodel": ("flex_xvert0", "tex_rgb"),
    "mjdata": ("qLDiagSqrtInv",),
}.items():
    for _field_name in _field_names:
        del dmc_sizes.array_sizes[_struct_name][_field_name]


def _add_marker_to_mujoco_33_scene(self, marker: dict[str, Any]) -> None:
    """Add one Gymnasium marker using MuJoCo 3.3's MjvGeom fields."""

    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError(f"Ran out of geoms. maxgeom: {self.scn.maxgeom}")

    geom = self.scn.geoms[self.scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.full(3, 0.1),
        np.zeros(3),
        np.eye(3).ravel(),
        np.ones(4, dtype=np.float32),
    )
    geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    geom.objid = -1
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    for key, value in marker.items():
        attribute = getattr(geom, key)
        if isinstance(attribute, np.ndarray):
            attribute[:] = value
        else:
            setattr(geom, key, value)
    self.scn.ngeom += 1


mujoco_rendering.BaseRender._add_marker_to_scene = _add_marker_to_mujoco_33_scene

G1_CONTROL_MODE_RAW = "raw_body_target"
G1_CONTROL_MODE_DIRECT_ABSOLUTE = "direct_absolute_position"
G1_CONTROL_MODE_SONIC = "sonic_token"
SONIC_DECODER_MODE_ENVIRONMENT = "environment"
SONIC_DECODER_MODE_INFERENCE_WORKER = "inference_worker"
GR00T_PROPRIO_DIM = G1_GR00T_STATE_DIM
GR00T_PROPRIO_INFO = "official_gr00t_proprio"
CONDITIONED_BODY_TARGET_INFO = "conditioned_body_target"
ACTION_TARGET_INFO = "action_target"

_NO_RESET_NOISE_TASKS = {
    "bookshelf_hard",
    "bookshelf_simple",
    "cube",
    "kitchen",
}

G1_SONIC_UNSUPPORTED_TASKS = ("highbar_simple", "highbar_hard", "run", "hurdle")
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

_HAND_KP = np.ones(14, dtype=np.float64)
_HAND_KD = np.full(14, 0.1, dtype=np.float64)

# Unitree G1 ``g1_29dof_with_hand.urdf`` velocity fields, in the MuJoCo and
# hand-name orders above. These cap the 50 Hz position target, not measured dq.
_BODY_TARGET_VELOCITY_LIMITS = np.asarray(
    [32, 32, 32, 20, 30, 30] * 2
    + [32, 30, 30]
    + [37, 37, 37, 37, 37, 22, 22] * 2,
    dtype=np.float64,
)
_HAND_TARGET_VELOCITY_LIMITS = np.asarray(
    [3.14, 12, 12, 12, 12, 12, 12] * 2,
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


def g1_absolute_request_from_action(
    body_action: np.ndarray,
    hand_action: np.ndarray | None,
    *,
    body_joint_ranges: np.ndarray,
    hand_joint_ranges: np.ndarray,
    raw_body: bool,
    absolute_position: bool = False,
) -> np.ndarray:
    """Convert one policy command into an absolute request without stateful limiting.

    Collectors and the PD controller consume the same request. Joint and torque
    bounds are enforced by the upstream MuJoCo XML, not by a command filter.
    """
    if absolute_position:
        body_target_request = body_action
    elif raw_body:
        body_target_request = body_joint_ranges[:, 0] + (
            body_action + 1.0
        ) * 0.5 * (body_joint_ranges[:, 1] - body_joint_ranges[:, 0])
    else:
        body_target_request = G1_DEFAULT_ANGLES + body_action * _ACTION_SCALE
    if hand_action is None:
        hand_target_request = np.zeros(14, dtype=np.float64)
    elif absolute_position:
        hand_target_request = hand_action
    else:
        hand_target_request = hand_joint_ranges[:, 0] + (
            hand_action + 1.0
        ) * 0.5 * (hand_joint_ranges[:, 1] - hand_joint_ranges[:, 0])
    return np.concatenate([body_target_request, hand_target_request])


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
    joint_names = G1_BODY_JOINT_NAMES + (G1_HAND_JOINT_NAMES if learn_hands else ())
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    reset_positions = (
        np.concatenate(
            [G1_DEFAULT_ANGLES, np.zeros(len(G1_HAND_JOINT_NAMES), dtype=np.float64)]
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

    metadata: ClassVar[dict[str, Any]] = HumanoidEnv.metadata

    def __init__(
        self,
        *,
        scene_path: str,
        task_name: str,
        control_mode: str = G1_CONTROL_MODE_SONIC,
        sonic_decoder_path: str | None = None,
        sonic_decoder_mode: str = SONIC_DECODER_MODE_ENVIRONMENT,
        raw_learn_hands: bool = True,
        sonic_learn_hands: bool = True,
        gr00t_balance_teacher_observation_state: bool = False,
        render_mode: str,
        width: int,
        height: int,
        camera_name: str | None = None,
    ) -> None:
        self._task_name = task_name
        self._task_key = task_name.removeprefix("g1_sonic_")
        self._control_mode = control_mode
        self._raw_learn_hands = raw_learn_hands
        self._sonic_learn_hands = sonic_learn_hands
        self._gr00t_balance_teacher_observation_state = (
            gr00t_balance_teacher_observation_state
        )
        self.robot = G1(self)
        # HumanoidBench's robot.dof means qpos length, not actuator count.
        self.robot.dof = 50
        task_class = TASKS[self._task_key]
        task_info = task_class(self.robot, None)
        policy_observation_space = Box(
            -np.inf,
            np.inf,
            shape=(task_info.observation_space.shape[0] + 86,),
            dtype=task_info.observation_space.dtype,
        )
        if sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
            observation_space = DictSpace(
                {
                    "obs": policy_observation_space,
                    "sonic_state": Box(-np.inf, np.inf, shape=(930,), dtype=np.float32),
                }
            )
        else:
            observation_space = policy_observation_space
        super().__init__(
            model_path=scene_path,
            frame_skip=task_info.frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name if camera_name is None else camera_name,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )
        self.observation_space = observation_space
        self._sonic_decoder_mode = sonic_decoder_mode

        if (
            control_mode == G1_CONTROL_MODE_SONIC
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
                for name in G1_HAND_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._hand_qvel_addresses = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in G1_HAND_JOINT_NAMES
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
                for name in G1_HAND_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._hand_joint_ranges = self.model.jnt_range[
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in G1_HAND_JOINT_NAMES
            ]
        ].copy()
        self._body_joint_ranges = self.model.jnt_range[
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in G1_BODY_JOINT_NAMES
            ]
        ].copy()
        if control_mode == G1_CONTROL_MODE_RAW:
            action_dim = 29 + (14 if raw_learn_hands else 0)
            self.action_space = Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        elif control_mode == G1_CONTROL_MODE_DIRECT_ABSOLUTE:
            action_dim = 43
            self.action_space = Box(
                np.concatenate(
                    [
                        self._body_joint_ranges[:, 0][MUJOCO_INDEX_FOR_POLICY],
                        self._hand_joint_ranges[:, 0],
                    ]
                ),
                np.concatenate(
                    [
                        self._body_joint_ranges[:, 1][MUJOCO_INDEX_FOR_POLICY],
                        self._hand_joint_ranges[:, 1],
                    ]
                ),
                dtype=np.float32,
            )
        elif control_mode == G1_CONTROL_MODE_SONIC:
            action_dim = 64 + (14 if sonic_learn_hands else 0)
            self.action_space = Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        if control_mode == G1_CONTROL_MODE_DIRECT_ABSOLUTE:
            self.action_low = self.action_space.low.astype(np.float64)
            self.action_high = self.action_space.high.astype(np.float64)
        else:
            self.action_low = np.full(action_dim, -1.0, dtype=np.float64)
            self.action_high = np.full(action_dim, 1.0, dtype=np.float64)
        # Observation and diagnostic scales only; these do not limit commands.
        self._target_velocity_limits = np.concatenate(
            [_BODY_TARGET_VELOCITY_LIMITS, _HAND_TARGET_VELOCITY_LIMITS]
        )
        self._joint_ranges = np.concatenate(
            [self._body_joint_ranges, self._hand_joint_ranges]
        )
        self._q_cmd = np.zeros(43, dtype=np.float64)
        self._v_cmd = np.zeros(43, dtype=np.float64)
        self._command_tracking_error = 0.0
        self._target_velocity_max_ratio = 0.0
        self._target_acceleration_mean_ratio_per_s = 0.0
        self._target_acceleration_max_ratio_per_s = 0.0
        self._target_velocity_reversal_fraction = 0.0
        self._actual_joint_velocity_max = 0.0
        self._torque_saturation_fraction = 0.0
        self._torque_saturation_count = 0
        self._torque_sample_count = 0
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

    def copy_balance_state_from(self, source):
        """Restore an independent Balance branch, including MuJoCo derived buffers."""
        self.data = copy.copy(source.data)
        axes = dmc_index.make_axis_indexers(MjModelWrapper(self.model))
        self.named = NamedIndexStructs(
            model=self.named.model,
            data=dmc_index.struct_indexer(MjDataWrapper(self.data), "mjdata", axes),
        )
        for field in ("_q_cmd", "_v_cmd", "_last_action", "_history", "_np_random"):
            setattr(self, field, copy.deepcopy(getattr(source, field)))

    def _body_state(self) -> tuple[np.ndarray, np.ndarray]:
        q_mu = self.data.qpos[self._body_qpos_addresses].copy()
        dq_mu = self.data.qvel[self._body_qvel_addresses].copy()
        return (q_mu - G1_DEFAULT_ANGLES)[MUJOCO_INDEX_FOR_POLICY], dq_mu[
            MUJOCO_INDEX_FOR_POLICY
        ]

    def _frame(self) -> dict[str, np.ndarray]:
        body_q, body_dq = self._body_state()
        gravity = self._projected_gravity()
        base_angular_velocity = self._base_angular_velocity()
        return {
            "angular_velocity": base_angular_velocity,
            "body_q": body_q,
            "body_dq": body_dq,
            "last_action": self._last_action.copy(),
            "gravity": gravity,
        }

    def _base_angular_velocity(self) -> np.ndarray:
        base_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._pelvis_body_id,
            base_velocity,
            1,
        )
        return base_velocity[:3].copy()


    def _projected_gravity(self) -> np.ndarray:
        root_quaternion = self.data.qpos[3:7]
        return _rotate_by_quaternion(
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

    def _conditioned_body_target(self) -> np.ndarray:
        return self._q_cmd[:29][MUJOCO_INDEX_FOR_POLICY].copy()

    def _conditioned_target(self) -> np.ndarray:
        return np.concatenate(
            [self._conditioned_body_target(), self._q_cmd[29:]]
        )

    def _official_gr00t_proprio(self) -> np.ndarray:
        if self._gr00t_balance_teacher_observation_state:
            state = np.concatenate(
                [self._task.get_obs(), self._command_observation()]
            )
        else:
            q = np.concatenate(
                [
                    self.data.qpos[self._body_qpos_addresses][MUJOCO_INDEX_FOR_POLICY],
                    self.data.qpos[self._hand_qpos_addresses],
                ]
            )
            dq = np.concatenate(
                [
                    self.data.qvel[self._body_qvel_addresses][MUJOCO_INDEX_FOR_POLICY],
                    self.data.qvel[self._hand_qvel_addresses],
                ]
            )
            state = np.concatenate(
                [
                    self._base_angular_velocity(),
                    q,
                    dq,
                    self._conditioned_target(),
                    self._projected_gravity(),
                ]
            )
        return state.astype(np.float32)

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

    def g1_absolute_request_from_raw_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized MuJoCo-order controls to policy-order joint requests."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        request = g1_absolute_request_from_action(
            action[:29],
            action[29:43] if self._raw_learn_hands else None,
            body_joint_ranges=self._body_joint_ranges,
            hand_joint_ranges=self._hand_joint_ranges,
            raw_body=True,
        )
        return np.concatenate([request[:29][MUJOCO_INDEX_FOR_POLICY], request[29:]])

    def _condition_targets(
        self,
        body_action_mu: np.ndarray,
        hand_action: np.ndarray | None,
        *,
        raw_body: bool,
        absolute_position: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = g1_absolute_request_from_action(
            body_action_mu,
            hand_action,
            body_joint_ranges=self._body_joint_ranges,
            hand_joint_ranges=self._hand_joint_ranges,
            raw_body=raw_body,
            absolute_position=absolute_position,
        )
        q_cmd = request
        v_cmd = (q_cmd - self._q_cmd) / self.dt
        acceleration = (v_cmd - self._v_cmd) / self.dt
        self._command_tracking_error = float(
            np.mean(
                np.abs(request - q_cmd)
                / (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
            )
        )
        self._target_velocity_max_ratio = float(
            np.max(np.abs(v_cmd) / self._target_velocity_limits)
        )
        self._target_acceleration_mean_ratio_per_s = float(
            np.mean(np.abs(acceleration) / self._target_velocity_limits)
        )
        self._target_acceleration_max_ratio_per_s = float(
            np.max(np.abs(acceleration) / self._target_velocity_limits)
        )
        self._target_velocity_reversal_fraction = float(
            np.count_nonzero(v_cmd * self._v_cmd < 0.0) / 43
        )
        self._q_cmd = q_cmd
        self._v_cmd = v_cmd
        return q_cmd[:29], q_cmd[29:]

    def _command_observation(self) -> np.ndarray:
        q_norm = 2.0 * (
            self._q_cmd - self._joint_ranges[:, 0]
        ) / (self._joint_ranges[:, 1] - self._joint_ranges[:, 0]) - 1.0
        v_norm = self._v_cmd / self._target_velocity_limits
        return np.concatenate([q_norm, v_norm])

    def _controller(
        self,
        body_target: np.ndarray,
        hand_target: np.ndarray,
    ) -> np.ndarray:
        q_mu = self.data.qpos[self._body_qpos_addresses]
        dq_mu = self.data.qvel[self._body_qvel_addresses]
        hand_q = self.data.qpos[self._hand_qpos_addresses]
        hand_dq = self.data.qvel[self._hand_qvel_addresses]
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[self._body_actuator_ids] = _KP * (body_target - q_mu) - _KD * dq_mu
        ctrl[self._hand_actuator_ids] = (
            _HAND_KP * (hand_target - hand_q) - _HAND_KD * hand_dq
        )
        self._torque_saturation_count += np.count_nonzero(
            (ctrl < self.model.actuator_ctrlrange[:, 0])
            | (ctrl > self.model.actuator_ctrlrange[:, 1])
        )
        self._torque_sample_count += self.model.nu
        return ctrl

    def set_decoded_action(self, action: np.ndarray) -> None:
        """Supply the body action decoded by Sample Factory's inference worker."""

        self._pending_decoded_action = np.asarray(action, dtype=np.float64)

    def _observation(self, task_observation: np.ndarray):
        policy_observation = np.concatenate(
            [task_observation, self._command_observation()]
        )
        if self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
            return {"obs": policy_observation, "sonic_state": self._decoder_state()}
        return policy_observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        observation, info = super().reset(seed=seed, options=options)
        info[GR00T_PROPRIO_INFO] = self._official_gr00t_proprio()
        info[CONDITIONED_BODY_TARGET_INFO] = self._conditioned_body_target()
        info[ACTION_TARGET_INFO] = self._conditioned_target()
        return observation, info

    def reset_model(self) -> Any:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        if self._control_mode != G1_CONTROL_MODE_SONIC:
            init_qpos[self._body_qpos_addresses] = G1_DEFAULT_ANGLES
            init_qpos[self._hand_qpos_addresses] = 0.0
        randomness = self.randomness
        self.set_state(
            init_qpos
            + self.np_random.uniform(-randomness, randomness, size=self.model.nq),
            init_qvel,
        )
        self._actual_joint_velocity_max = 0.0
        self._torque_saturation_fraction = 0.0
        self._torque_saturation_count = 0
        self._torque_sample_count = 0
        self._last_action.fill(0.0)
        self._pending_decoded_action = None
        self._history.clear()
        task_observation = self._task.reset_model()
        self._q_cmd = np.concatenate(
            [
                self.data.qpos[self._body_qpos_addresses],
                self.data.qpos[self._hand_qpos_addresses],
            ]
        ).copy()
        self._v_cmd.fill(0.0)
        self._history.append(self._frame())
        return self._observation(task_observation)

    def step(self, action: Any):
        action = np.asarray(action, dtype=np.float32)
        absolute_position = False
        if self._control_mode in (
            G1_CONTROL_MODE_RAW,
            G1_CONTROL_MODE_DIRECT_ABSOLUTE,
        ):
            if self._control_mode == G1_CONTROL_MODE_DIRECT_ABSOLUTE:
                decoder_action = action[:29].astype(np.float64)
                body_action_mu = decoder_action[POLICY_INDEX_FOR_MUJOCO]
                hand_action = action[29:43].astype(np.float64)
                absolute_position = True
                raw_body = False
            else:
                body_action_mu = action[:29].astype(np.float64)
                decoder_action = body_action_mu[MUJOCO_INDEX_FOR_POLICY]
                hand_action = (
                    action[29:43].astype(np.float64) if self._raw_learn_hands else None
                )
                raw_body = True
        elif self._control_mode == G1_CONTROL_MODE_SONIC:
            if self._sonic_decoder_mode == SONIC_DECODER_MODE_INFERENCE_WORKER:
                decoder_action = self._pending_decoded_action
                self._pending_decoded_action = None
                body_action_mu = decoder_action[POLICY_INDEX_FOR_MUJOCO]
            else:
                decoder_action = self._decode(action[:64])
                body_action_mu = decoder_action[POLICY_INDEX_FOR_MUJOCO]
            hand_action = (
                action[64:78].astype(np.float64)
                if self._sonic_learn_hands
                else None
            )
            raw_body = False
        body_target, hand_target = self._condition_targets(
            body_action_mu,
            hand_action,
            raw_body=raw_body,
            absolute_position=absolute_position,
        )
        self._torque_saturation_count = 0
        self._torque_sample_count = 0
        self._actual_joint_velocity_max = 0.0
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._controller(body_target, hand_target)
            mujoco.mj_step(self.model, self.data)
            self._actual_joint_velocity_max = max(
                self._actual_joint_velocity_max,
                float(
                    np.max(
                        np.abs(
                            self.data.qvel[
                                np.concatenate(
                                    [
                                        self._body_qvel_addresses,
                                        self._hand_qvel_addresses,
                                    ]
                                )
                            ]
                        )
                    )
                ),
            )
        self._torque_saturation_fraction = float(
            self._torque_saturation_count / self._torque_sample_count
        )
        self._last_action[:] = decoder_action
        self._history.append(self._frame())
        task_observation = self._task.get_obs()
        reward, info = self._task.get_reward()
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
                "command_tracking_error": self._command_tracking_error,
                "target_velocity_max_ratio": self._target_velocity_max_ratio,
                "target_acceleration_mean_ratio_per_s": self._target_acceleration_mean_ratio_per_s,
                "target_acceleration_max_ratio_per_s": self._target_acceleration_max_ratio_per_s,
                "target_velocity_reversal_fraction": self._target_velocity_reversal_fraction,
                "actual_joint_velocity_max": self._actual_joint_velocity_max,
                "torque_saturation_fraction": self._torque_saturation_fraction,
                GR00T_PROPRIO_INFO: self._official_gr00t_proprio(),
                CONDITIONED_BODY_TARGET_INFO: self._conditioned_body_target(),
                ACTION_TARGET_INFO: self._conditioned_target(),
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
