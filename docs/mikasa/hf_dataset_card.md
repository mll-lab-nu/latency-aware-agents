---
pretty_name: MIKASA InterceptGrabFast H1 Rollouts
task_categories:
  - reinforcement-learning
tags:
  - reinforcement-learning
  - robotics
  - lerobot
  - mikasa
---

# MIKASA InterceptGrabFast H1 rollouts

Successful deterministic rollouts from the PPO teacher for
`InterceptGrabFast-VLA-v0`. The language instruction is:

```text
Intercept the rolling ball and grasp it to stop it.
```

The task uses the H1 embodiment, a 7D `pd_ee_delta_pose` action, top and wrist
RGB observations, and a native 20 Hz control rate. The full reproduction guide
is the [MIKASA cookbook](https://github.com/ZihanWang314/latency-sensitive-bench/blob/master/docs/mikasa/cookbook.md).

## Contents

The published rollouts are retained in `MLL-Lab/LAGEN-datasets`
at `4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a`, under
`zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/`. The paths below are relative
to that directory.

```text
raw/                              # 275 raw episodes
lerobot/                          # 250 training episodes
lerobot__val/                     # 25 validation episodes
mikasa_intercept_grab_fast_h1.json
```

The raw rollout contains 275 successful episodes, 9,859 transitions, and 275
unique seeds. Episode length has mean/min/max `35.851/24/57`. Actions are clipped
to `[-1, 1]`. The LeRobot v3 conversion contains 8,911 train frames and 948
validation frames at 20 FPS. The split preserves collection order: episodes
`0:250` train and `250:275` validation.

The RLDS conversion is an intermediate duplicate and is not published.

## Raw NPZ schema

Each `train_data_*.npz` file stores one episode:

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `rgb` | `uint8[T, 128, 128, 6]` | concatenated top and wrist RGB |
| `proprio` | `float32[T, 7]` | robot proprioception |
| `action` | `float32[T, 7]` | end-effector delta pose action |
| `reward` | `float32[T]` | environment reward |
| `success` | `int32[T]` | per-step success signal |
| `done` | `int32[T]` | per-step termination signal |
| `language_instruction` | scalar string | task instruction |
| `success_once` | scalar bool | episode success |
| `episode_length` | scalar int32 | number of transitions |
| `episode_seed` | scalar int64 | rollout seed |

## LeRobot schema

Each split is self-contained with `data/`, `videos/`, and `meta/`:

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `observation.images.top` | AV1 video, `128 × 128 × 3` | top camera RGB |
| `observation.images.wrist` | AV1 video, `128 × 128 × 3` | wrist camera RGB |
| `observation.state` | `float32[7]` | absolute proprioception |
| `action` | `float32[7]` | non-absolute end-effector action |
| `timestamp` | `float32[1]` | seconds from episode start |
| `episode_index` | `int64[1]` | episode id |
| `frame_index` | `int64[1]` | frame id within episode |
| `task_index` | `int64[1]` | task id |

Use `lerobot` for training and keep
`lerobot__val` for held-out action-loss evaluation.
The mixture JSON records both exact StarVLA dataset names.

## Provenance

The teacher uses a 49D privileged state and deterministic mean actions. It was
trained with normalized dense reward, seed 123, learning rate `1e-4`, and eight
PPO update epochs. The selected checkpoint passed `100/100` canonical episodes
before collection. The MIKASA integration revision is
`cf4f96f319022f89c9d7cfbd639d19bc10ed44fb` on vendor baseline
`16634db18bef08128ed79346469c86fc12169aed`. The dataset contains only complete
successful trajectories.

This dataset is for research and benchmarking in MIKASA. No license metadata is
asserted here.
