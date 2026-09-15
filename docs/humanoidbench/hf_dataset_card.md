---
pretty_name: HumanoidBench Balance Simple
task_categories:
  - reinforcement-learning
tags:
  - robotics
  - humanoid
  - lerobot
---

# HumanoidBench Balance Simple

LeRobot v2 image-parquet exports for the `g1_sonic_balance_simple` task in
HumanoidBench. The task prompt is:

```text
Keep the G1 humanoid balanced on the support board.
```

This dataset is published in the `latency-sensitive-bench/humanoidbench`
dataset repository. HumanoidBench scene assets are not duplicated here; use
`latency-sensitive-bench/g1-sonic-assets` at revision
`7c0a6616c78106adf9fa54dfc40e56ea07dc7d5d`.

## Dataset contents

The repository layout is:

```text
balance_simple/
├── lerobot/       # train: 90 episodes, 86,048 frames
└── lerobot__val/  # validation: 10 episodes, 9,736 frames
```

Each split is a self-contained LeRobot v2 dataset with `data/`, `videos/`,
and `meta/`. Data is sampled at 50 Hz; one row is one decision step. The
episodes contain 100 total teacher rollouts split 90/10 with split seed 0.

## Schema

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `observation.image` | video, `256 × 256 × 3`, H.264, 50 Hz | head-camera RGB observation |
| `observation.state` | `float32[198]` | native G1 teacher observation |
| `action` | `float32[43]` | absolute native joint target |
| `timestamp` | `float64` | seconds from episode start |
| `episode_index` | `int64` | episode id |
| `frame_index` | `int64` | frame id within episode |
| `task_index` | `int64` | index into `meta/tasks.jsonl` |
| `latency` | `int64` | recorded latency metadata |
| `latency_raw_frames` | `int64` | recorded latency in raw frames |
| `done` | `bool` | episode termination flag |
| `reward` | `float32` | teacher environment reward |
| `density_weight` | `float32` | rollout density weight |

The 43 action entries are ordered as follows:

```text
left_hip_pitch_joint, right_hip_pitch_joint, waist_yaw_joint,
left_hip_roll_joint, right_hip_roll_joint, waist_roll_joint,
left_hip_yaw_joint, right_hip_yaw_joint, waist_pitch_joint,
left_knee_joint, right_knee_joint, left_shoulder_pitch_joint,
right_shoulder_pitch_joint, left_ankle_pitch_joint, right_ankle_pitch_joint,
left_shoulder_roll_joint, right_shoulder_roll_joint, left_ankle_roll_joint,
right_ankle_roll_joint, left_shoulder_yaw_joint, right_shoulder_yaw_joint,
left_elbow_joint, right_elbow_joint, left_wrist_roll_joint,
right_wrist_roll_joint, left_wrist_pitch_joint, right_wrist_pitch_joint,
left_wrist_yaw_joint, right_wrist_yaw_joint, left_hand_thumb_0_joint,
left_hand_thumb_1_joint, left_hand_thumb_2_joint, left_hand_middle_0_joint,
left_hand_middle_1_joint, left_hand_index_0_joint, left_hand_index_1_joint,
right_hand_thumb_0_joint, right_hand_thumb_1_joint, right_hand_thumb_2_joint,
right_hand_index_0_joint, right_hand_index_1_joint, right_hand_middle_0_joint,
right_hand_middle_1_joint
```

State entries are ordered `state_0` through `state_197`; the authoritative
feature metadata is in each split's `meta/info.json` and `meta/modality.json`.

## Provenance

The data is generated from the published G1 SONIC Balance Simple FastTD3
teacher with deterministic rollout, 100 maximum episodes, validation fraction
`0.1`, and split seed `0`. The converted output is intended for StarVLA
training.

The state is the teacher's 198D pre-decision observation. Actions are 43D
absolute native joint targets. No normalization or action-dimension padding is
applied in this dataset.

## Intended use

Use `balance_simple/lerobot` for training and keep `balance_simple/lerobot__val`
for validation. The StarVLA modality maps these fields to `state.state`,
`action.action`, and `video.ego_view`; Balance uses the H8 action contract.

This dataset is for research and benchmarking in the HumanoidBench Balance
Simple environment. It does not include model checkpoints or W&B artifacts.

No license metadata is asserted here.
