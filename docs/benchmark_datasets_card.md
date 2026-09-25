---
license: other
task_categories:
- robotics
tags:
- LeRobot
- image
- video
configs:
- config_name: humanoidbench_balance_simple
  default: true
  data_files:
  - split: train
    path: zero-latency/humanoidbench-balance-simple/teacher-rollouts-v1/lerobot/data/chunk-*/*.parquet
  - split: validation
    path: zero-latency/humanoidbench-balance-simple/teacher-rollouts-v1/lerobot__val/data/chunk-*/*.parquet
- config_name: mikasa_intercept_grab_fast
  data_files:
  - split: train
    path: viewer/mikasa-intercept-grab-fast/train.parquet
  - split: validation
    path: viewer/mikasa-intercept-grab-fast/validation.parquet
dataset_info:
- config_name: humanoidbench_balance_simple
  features:
  - name: observation.image
    dtype: image
  - name: observation.state
    list: float32
    length: 198
  - name: action
    list: float32
    length: 43
  - name: timestamp
    dtype: float64
  - name: episode_index
    dtype: int64
  - name: frame_index
    dtype: int64
  - name: task_index
    dtype: int64
  - name: latency
    dtype: int64
  - name: latency_raw_frames
    dtype: int64
  - name: done
    dtype: bool
  - name: reward
    dtype: float32
  - name: density_weight
    dtype: float32
- config_name: mikasa_intercept_grab_fast
  features:
  - name: observation.images.top
    dtype: image
  - name: observation.images.wrist
    dtype: image
  - name: observation.state
    list: float32
    length: 7
  - name: action
    list: float32
    length: 7
  - name: timestamp
    dtype: float32
  - name: frame_index
    dtype: int64
  - name: episode_index
    dtype: int64
  - name: index
    dtype: int64
  - name: task_index
    dtype: int64
---

# Latency-Sensitive Bench datasets

Accepted zero-latency teacher rollouts for the supported benchmark tasks.

## Viewer subsets

- `humanoidbench_balance_simple`: 90 training and 10 validation episodes. The
  `observation.image` values are PNG bytes declared as the Hugging Face `Image`
  feature, so the Dataset Viewer renders them instead of showing their encoded
  representation. Canonical LeRobot MP4 files remain under each split's
  `videos/` directory.
- `mikasa_intercept_grab_fast`: fixed 250/25 StarVLA split. Every Viewer row
  includes its corresponding `observation.images.top` and
  `observation.images.wrist` frame. The canonical LeRobot data remains under
  the task path with its RGB observations stored in MP4 files.

## Repository layout

- `zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/raw`: 275
  successful source NPZ episodes.
- `zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/lerobot`,
  `lerobot__val`, and `mikasa_intercept_grab_fast_h1.json`: converted dataset
  and StarVLA mixture.
- `viewer/mikasa-intercept-grab-fast`: browser-oriented Parquet with the same
  transitions and decoded top/wrist frames.
- `zero-latency/humanoidbench-balance-simple/teacher-rollouts-v1/lerobot` and
  `lerobot__val`: published 90/10 split, videos, and metadata.

Extreme Parkour and HAIC datasets are intentionally not published until their
current VLA pipelines pass their documented gates.

Historical source repositories remain unchanged: MIKASA dataset revision
`28312021c634fa97f8068077dfd6eb0efe6d45ce` and Balance dataset revision
`dbb954842942196e127c9acc0d776b62aa679990`.

See the repository [artifact registry](https://github.com/placeholder/placeholder/blob/master/docs/artifacts.md)
for exact task contracts and reproduction commands.
