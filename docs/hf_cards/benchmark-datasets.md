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
- config_name: deadly_corridor_fixed_latency_6_1000ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fixed_latency_6_1000ep_7k2steps/train-00000-of-00002.parquet
    - latency-aware/deadly-corridor/deadly_corridor_fixed_latency_6_1000ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fixed_latency_6_1000ep_7k2steps/val-00000-of-00001.parquet
- config_name: demon_attack_fixed_latency_6_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps/train-00000-of-00003.parquet
    - latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps/train-00001-of-00003.parquet
    - latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps/train-00002-of-00003.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: flappy_fixed_latency_3_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00000-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00001-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00002-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00003-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00004-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00005-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00006-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00007-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00008-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00009-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00010-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00011-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00012-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00013-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00014-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00015-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00016-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00017-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00018-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00019-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00020-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00021-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00022-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00023-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00024-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00025-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00026-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00027-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00028-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00029-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00030-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00031-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00032-of-00034.parquet
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/train-00033-of-00034.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: flappy_fix_latency_2_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00000-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00001-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00002-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00003-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00004-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00005-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00006-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00007-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00008-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00009-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00010-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00011-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00012-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00013-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00014-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00015-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00016-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00017-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00018-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00019-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00020-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00021-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00022-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00023-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00024-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00025-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00026-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00027-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00028-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00029-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00030-of-00032.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/train-00031-of-00032.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/val-00000-of-00004.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/val-00001-of-00004.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/val-00002-of-00004.parquet
    - latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/val-00003-of-00004.parquet
- config_name: deadly_corridor_fixed_latency_0_1000ep_7k2steps
  data_files:
  - split: train
    path:
    - zero-latency/deadly-corridor/deadly_corridor_fixed_latency_0_1000ep_7k2steps/train-00000-of-00002.parquet
    - zero-latency/deadly-corridor/deadly_corridor_fixed_latency_0_1000ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - zero-latency/deadly-corridor/deadly_corridor_fixed_latency_0_1000ep_7k2steps/val-00000-of-00001.parquet
- config_name: demon_attack_fixed_latency_0_200ep_7k2steps
  data_files:
  - split: train
    path:
    - zero-latency/demon-attack/demon_attack_fixed_latency_0_200ep_7k2steps/train-00000-of-00004.parquet
    - zero-latency/demon-attack/demon_attack_fixed_latency_0_200ep_7k2steps/train-00001-of-00004.parquet
    - zero-latency/demon-attack/demon_attack_fixed_latency_0_200ep_7k2steps/train-00002-of-00004.parquet
    - zero-latency/demon-attack/demon_attack_fixed_latency_0_200ep_7k2steps/train-00003-of-00004.parquet
  - split: val
    path:
    - zero-latency/demon-attack/demon_attack_fixed_latency_0_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: flappy_fixed_latency_0_200ep_7k2steps
  data_files:
  - split: train
    path:
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00000-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00001-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00002-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00003-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00004-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00005-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00006-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00007-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00008-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00009-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00010-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00011-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00012-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00013-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00014-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00015-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00016-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00017-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00018-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00019-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00020-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00021-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00022-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00023-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00024-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00025-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00026-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00027-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00028-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00029-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00030-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00031-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00032-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00033-of-00035.parquet
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/train-00034-of-00035.parquet
  - split: val
    path:
    - zero-latency/flappy/flappy_fixed_latency_0_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: demon_attack_fixed_latency_6_200ep_7k2steps_ghost15
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps_ghost15/train.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps_ghost15/val.parquet
- config_name: flappy_fixed_latency_3_200ep_7k2steps_ghost15
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps_ghost15/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps_ghost15/val.parquet
- config_name: inverted_pendulum_200ep_2__inverted_pendulum_fixed_latency_0_100ep_1ksteps_og
  data_files:
  - split: train
    path:
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep_2/inverted_pendulum_fixed_latency_0_100ep_1ksteps_og/train.parquet
  - split: val
    path:
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep_2/inverted_pendulum_fixed_latency_0_100ep_1ksteps_og/val.parquet
- config_name: inverted_pendulum_200ep_2__inverted_pendulum_fixed_latency_2_1000ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_2_1000ep_7k2steps/train-00000-of-00002.parquet
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_2_1000ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_2_1000ep_7k2steps/val-00000-of-00001.parquet
- config_name: inverted_pendulum_200ep_2__inverted_pendulum_fixed_latency_4_1000ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_4_1000ep_7k2steps/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_4_1000ep_7k2steps/val-00000-of-00001.parquet
- config_name: inverted_pendulum_200ep__inverted_pendulum_fixed_latency_0_200ep_7k2steps
  data_files:
  - split: train
    path:
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00000-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00001-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00002-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00003-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00004-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00005-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00006-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00007-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00008-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00009-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00010-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00011-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00012-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00013-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00014-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00015-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00016-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00017-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00018-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00019-of-00021.parquet
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/train-00020-of-00021.parquet
  - split: val
    path:
    - zero-latency/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_0_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: inverted_pendulum_200ep__inverted_pendulum_fixed_latency_2_1000ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_2_1000ep_7k2steps/train-00000-of-00002.parquet
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_2_1000ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_2_1000ep_7k2steps/val-00000-of-00001.parquet
- config_name: inverted_pendulum_200ep__inverted_pendulum_fixed_latency_4_1000ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_4_1000ep_7k2steps/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/inverted-pendulum/inverted_pendulum_200ep/inverted_pendulum_fixed_latency_4_1000ep_7k2steps/val-00000-of-00001.parquet
- config_name: flappy_fix_latency_0_200ep_context5
  data_files:
  - split: train
    path:
    - zero-latency/flappy/flappy_fix_latency_0_200ep_context5/train.parquet
  - split: val
    path:
    - zero-latency/flappy/flappy_fix_latency_0_200ep_context5/val.parquet
- config_name: flappy_fix_latency_1_200ep_context5
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_1_200ep_context5/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_1_200ep_context5/val.parquet
- config_name: flappy_fix_latency_2_200ep_context5
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_2_200ep_context5/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_2_200ep_context5/val.parquet
- config_name: flappy_fix_latency_3_200ep_context5
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_3_200ep_context5/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_3_200ep_context5/val.parquet
- config_name: flappy_fix_latency_4_200ep_context5
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_4_200ep_context5/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_4_200ep_context5/val.parquet
- config_name: deadly_corridor_fix_latency_0
  data_files:
  - split: train
    path:
    - zero-latency/deadly-corridor/deadly_corridor_fix_latency_0/data/train-00000-of-00001.parquet
  - split: val
    path:
    - zero-latency/deadly-corridor/deadly_corridor_fix_latency_0/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_1
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_1/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_1/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_10
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_10/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_10/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_12
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_12/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_12/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_13
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_13/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_13/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_14
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_14/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_14/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_15
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_15/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_15/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_2
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_2/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_2/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_3
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_3/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_3/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_4
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_4/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_4/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_5
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_5/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_5/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_6
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_6/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_6/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_7
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_7/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_7/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_8
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_8/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_8/data/val-00000-of-00001.parquet
- config_name: deadly_corridor_fix_latency_9
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_9/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_corridor_fix_latency_9/data/val-00000-of-00001.parquet
- config_name: demon_attack_200ep__demon_attack_fix_latency_0_200ep
  data_files:
  - split: train
    path:
    - zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep/train.parquet
  - split: val
    path:
    - zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep/val.parquet
- config_name: demon_attack_200ep__demon_attack_fix_latency_2_200ep
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_2_200ep/train.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_2_200ep/val.parquet
- config_name: demon_attack_200ep__demon_attack_fix_latency_4_200ep
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_4_200ep/train.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_4_200ep/val.parquet
- config_name: demon_attack_200ep__demon_attack_fix_latency_6_200ep
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_6_200ep/train.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_6_200ep/val.parquet
- config_name: demon_attack_200ep__demon_attack_fix_latency_8_200ep
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_8_200ep/train.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_200ep/demon_attack_fix_latency_8_200ep/val.parquet
- config_name: demon_attack_fix_latency_0
  data_files:
  - split: train
    path:
    - zero-latency/demon-attack/demon_attack_fix_latency_0/data/train-00000-of-00001.parquet
  - split: val
    path:
    - zero-latency/demon-attack/demon_attack_fix_latency_0/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_1
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_1/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_1/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_10
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_10/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_10/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_11
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_11/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_11/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_12
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_12/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_12/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_13
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_13/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_13/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_14
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_14/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_14/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_15
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_15/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_15/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_2
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_2/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_2/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_3
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_3/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_3/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_4
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_4/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_4/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_5
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_5/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_5/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_6
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_6/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_6/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_7
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_7/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_7/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_8
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_8/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_8/data/val-00000-of-00001.parquet
- config_name: demon_attack_fix_latency_9
  data_files:
  - split: train
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_9/data/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/demon-attack/demon_attack_fix_latency_9/data/val-00000-of-00001.parquet
- config_name: flappy_200ep__flappy_fix_latency_0_200ep
  data_files:
  - split: train
    path:
    - zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep/train.parquet
  - split: val
    path:
    - zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep/val.parquet
- config_name: flappy_200ep__flappy_fix_latency_1_200ep
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_1_200ep/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_1_200ep/val.parquet
- config_name: flappy_200ep__flappy_fix_latency_2_200ep
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_2_200ep/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_2_200ep/val.parquet
- config_name: flappy_200ep__flappy_fix_latency_3_200ep
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_3_200ep/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_3_200ep/val.parquet
- config_name: flappy_200ep__flappy_fix_latency_4_200ep
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_4_200ep/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_200ep/flappy_fix_latency_4_200ep/val.parquet
- config_name: flappy_fix_latency_0_small
  data_files:
  - split: train
    path:
    - zero-latency/flappy/flappy_fix_latency_0_small/data/train-00000-of-00002.parquet
    - zero-latency/flappy/flappy_fix_latency_0_small/data/train-00001-of-00002.parquet
  - split: val
    path:
    - zero-latency/flappy/flappy_fix_latency_0_small/data/val-00000-of-00001.parquet
- config_name: flappy_fix_latency_1_small
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_1_small/data/train-00000-of-00003.parquet
    - latency-aware/flappy/flappy_fix_latency_1_small/data/train-00001-of-00003.parquet
    - latency-aware/flappy/flappy_fix_latency_1_small/data/train-00002-of-00003.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_1_small/data/val-00000-of-00001.parquet
- config_name: flappy_fix_latency_2_small
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_2_small/data/train-00000-of-00003.parquet
    - latency-aware/flappy/flappy_fix_latency_2_small/data/train-00001-of-00003.parquet
    - latency-aware/flappy/flappy_fix_latency_2_small/data/train-00002-of-00003.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_2_small/data/val-00000-of-00001.parquet
- config_name: flappy_fix_latency_3_small
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_3_small/data/train-00000-of-00003.parquet
    - latency-aware/flappy/flappy_fix_latency_3_small/data/train-00001-of-00003.parquet
    - latency-aware/flappy/flappy_fix_latency_3_small/data/train-00002-of-00003.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_3_small/data/val-00000-of-00001.parquet
- config_name: flappy_fix_latency_4_small
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fix_latency_4_small/data/train-00000-of-00002.parquet
    - latency-aware/flappy/flappy_fix_latency_4_small/data/train-00001-of-00002.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fix_latency_4_small/data/val-00000-of-00001.parquet
- config_name: mean_vs_profile_flappy_mean
  data_files:
  - split: train
    path: latency-aware/flappy/mean_rtx3090_s42_200ep_v1/demonstrations/raw/train.parquet
  - split: validation
    path: latency-aware/flappy/mean_rtx3090_s42_200ep_v1/demonstrations/raw/val.parquet
- config_name: mean_vs_profile_deadly_corridor_mean
  data_files:
  - split: train
    path: latency-aware/deadly-corridor/mean_rtx3090_s42_8500ep_v1/demonstrations/raw/train.parquet
  - split: validation
    path: latency-aware/deadly-corridor/mean_rtx3090_s42_8500ep_v1/demonstrations/raw/val.parquet
- config_name: mean_vs_profile_ant_mean
  data_files:
  - split: train
    path: latency-aware/ant/mean_rtx3090_s2222_100ep_v1/demonstrations/raw/train.parquet
  - split: validation
    path: latency-aware/ant/mean_rtx3090_s2222_100ep_v1/demonstrations/raw/val.parquet
- config_name: mean_vs_profile_intercept_mean_episode_manifest
  data_files:
  - split: train
    path: latency-aware/intercept-grab-fast/mean_rtx3090_h1_s123_275ep_v1/demonstrations/conversion/expected/lerobot/meta/episodes/**/*.parquet
  - split: validation
    path: latency-aware/intercept-grab-fast/mean_rtx3090_h1_s123_275ep_v1/demonstrations/conversion/expected/lerobot__val/meta/episodes/**/*.parquet
- config_name: flappy_fixed_l2_fs4
  data_files:
  - split: train
    path:
    - latency-aware/flappy/flappy_fixed_l2_fs4/train.parquet
  - split: val
    path:
    - latency-aware/flappy/flappy_fixed_l2_fs4/val.parquet
- config_name: factor-sweeps--deadly_corridor__context_window
  data_files:
  - split: test
    path:
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs10:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs10:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs10:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs10:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs10:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs12:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs12:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs12:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs12:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs12:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs14:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs14:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs14:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs14:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs14:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs16:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs16:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs16:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs16:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs16:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs1:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs1:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs1:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs1:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs1:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs2:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs2:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs2:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs2:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs2:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs6:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs6:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs6:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs6:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs6:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs8:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs8:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs8:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs8:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs8:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs10:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs10:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs10:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs10:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs10:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs12:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs12:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs12:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs12:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs12:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs14:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs14:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs14:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs14:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs14:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs16:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs16:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs16:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs16:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs16:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs1:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs1:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs1:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs1:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs1:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs2:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs2:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs2:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs2:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs2:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs4:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs4:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs4:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs4:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs4:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs6:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs6:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs6:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs6:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs6:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs8:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs8:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs8:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs8:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:frame_stack:uniform_u2_6:fs8:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
- config_name: factor-sweeps--deadly_corridor__observation_stride
  data_files:
  - split: test
    path:
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs11p6667:stride3:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs11p6667:stride3:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs11p6667:stride3:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs11p6667:stride3:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs11p6667:stride3:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs17p5:stride2:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs17p5:stride2:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs17p5:stride2:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs17p5:stride2:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs17p5:stride2:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs35:stride1:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs35:stride1:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs35:stride1:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs35:stride1:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs35:stride1:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs5p83333:stride6:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs5p83333:stride6:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs5p83333:stride6:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs5p83333:stride6:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs5p83333:stride6:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs7:stride5:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs7:stride5:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs7:stride5:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs7:stride5:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs7:stride5:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs8p75:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs8p75:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs8p75:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs8p75:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:fixed_l4:fs4:obs8p75:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs11p6667:stride3:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs11p6667:stride3:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs11p6667:stride3:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs11p6667:stride3:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs11p6667:stride3:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs17p5:stride2:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs17p5:stride2:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs17p5:stride2:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs17p5:stride2:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs17p5:stride2:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs35:stride1:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs35:stride1:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs35:stride1:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs35:stride1:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs35:stride1:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs5p83333:stride6:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs5p83333:stride6:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs5p83333:stride6:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs5p83333:stride6:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs5p83333:stride6:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs7:stride5:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs7:stride5:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs7:stride5:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs7:stride5:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs7:stride5:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs8p75:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs8p75:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs8p75:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs8p75:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/deadly-corridor/factor_sweep:deadly_corridor:observation_stride:uniform_u2_6:fs4:obs8p75:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
- config_name: factor-sweeps--demon_attack__context_window
  data_files:
  - split: test
    path:
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs10:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs10:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs10:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs10:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs10:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs11:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs11:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs11:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs11:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs11:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs12:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs12:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs12:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs12:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs12:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs4:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs4:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs4:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs4:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs4:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs5:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs5:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs5:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs5:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs5:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs6:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs6:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs6:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs6:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs6:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs7:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs7:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs7:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs7:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs7:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs8:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs8:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs8:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs8:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs8:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs9:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs9:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs9:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs9:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:fixed_l4:fs9:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs10:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs10:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs10:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs10:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs10:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs11:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs11:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs11:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs11:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs11:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs12:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs12:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs12:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs12:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs4:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs4:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs4:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs4:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs4:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs5:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs5:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs5:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs5:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs5:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs6:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs6:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs6:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs6:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs6:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs7:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs7:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs7:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs7:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs7:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs8:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs8:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs8:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs8:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs8:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs9:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs9:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs9:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs9:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:frame_stack:uniform_u2_6:fs9:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
- config_name: factor-sweeps--demon_attack__observation_stride
  data_files:
  - split: test
    path:
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs10:stride6:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs10:stride6:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs10:stride6:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs10:stride6:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs10:stride6:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs12:stride5:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs12:stride5:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs12:stride5:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs12:stride5:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs12:stride5:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs20:stride3:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs20:stride3:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs20:stride3:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs20:stride3:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs20:stride3:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs30:stride2:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs30:stride2:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs30:stride2:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs30:stride2:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs30:stride2:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6:stride10:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6:stride10:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6:stride10:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6:stride10:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6:stride10:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6p66667:stride9:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6p66667:stride9:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6p66667:stride9:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6p66667:stride9:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs6p66667:stride9:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs7p5:stride8:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs7p5:stride8:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs7p5:stride8:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs7p5:stride8:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs7p5:stride8:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs8p57143:stride7:seed10/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs8p57143:stride7:seed11/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs8p57143:stride7:seed12/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs8p57143:stride7:seed13/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs8p57143:stride7:seed14/fixed_4/eval_latency_4/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs10:stride6:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs10:stride6:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs10:stride6:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs10:stride6:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs10:stride6:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs12:stride5:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs12:stride5:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs12:stride5:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs12:stride5:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs12:stride5:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs15:stride4:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs15:stride4:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs15:stride4:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs15:stride4:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs15:stride4:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs20:stride3:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs20:stride3:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs20:stride3:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs20:stride3:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs20:stride3:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs30:stride2:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs30:stride2:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs30:stride2:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs30:stride2:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs30:stride2:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6:stride10:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6:stride10:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6:stride10:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6:stride10:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6:stride10:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6p66667:stride9:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6p66667:stride9:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6p66667:stride9:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6p66667:stride9:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs6p66667:stride9:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs7p5:stride8:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs7p5:stride8:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs7p5:stride8:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs7p5:stride8:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs7p5:stride8:seed14/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs8p57143:stride7:seed10/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs8p57143:stride7:seed11/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs8p57143:stride7:seed12/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs8p57143:stride7:seed13/uniform_min_2_max_6/episode_metrics.jsonl
    - evaluations/demon-attack/factor_sweep:demon_attack:observation_stride:uniform_u2_6:fs4:obs8p57143:stride7:seed14/uniform_min_2_max_6/episode_metrics.jsonl
- config_name: factor-sweeps--flappy__context_window
  data_files:
  - split: test
    path:
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs2:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs2:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs2:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs2:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs2:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs3:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs3:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs3:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs3:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs3:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs4:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs4:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs4:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs4:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs4:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs5:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs5:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs5:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs5:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs5:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs6:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs6:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs6:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs6:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs6:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs7:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs7:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs7:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs7:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs7:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs8:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs8:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs8:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs8:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:fixed_l2:fs8:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs1:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs1:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs1:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs1:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs1:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs2:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs2:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs2:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs2:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs2:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs3:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs3:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs3:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs3:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs3:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs4:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs4:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs4:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs4:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs4:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs5:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs5:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs5:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs5:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs5:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs6:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs6:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs6:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs6:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs6:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs7:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs7:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs7:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs7:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs7:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs8:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs8:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs8:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs8:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:frame_stack:uniform_u1_3:fs8:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
- config_name: factor-sweeps--flappy__observation_stride
  data_files:
  - split: test
    path:
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs10:stride3:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs10:stride3:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs10:stride3:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs10:stride3:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs10:stride3:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs30:stride1:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs30:stride1:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs30:stride1:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs30:stride1:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs30:stride1:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs6:stride5:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs6:stride5:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs6:stride5:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs6:stride5:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs6:stride5:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs7p5:stride4:seed10/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs7p5:stride4:seed11/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs7p5:stride4:seed12/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs7p5:stride4:seed13/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs7p5:stride4:seed14/fixed_2/eval_latency_2/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs10:stride3:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs10:stride3:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs10:stride3:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs10:stride3:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs10:stride3:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs15:stride2:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs15:stride2:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs15:stride2:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs15:stride2:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs15:stride2:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs30:stride1:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs30:stride1:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs30:stride1:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs30:stride1:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs30:stride1:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs6:stride5:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs6:stride5:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs6:stride5:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs6:stride5:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs6:stride5:seed14/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs7p5:stride4:seed10/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs7p5:stride4:seed11/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs7p5:stride4:seed12/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs7p5:stride4:seed13/uniform_min_1_max_3/episode_metrics.jsonl
    - evaluations/flappy/factor_sweep:flappy:observation_stride:uniform_u1_3:fs4:obs7p5:stride4:seed14/uniform_min_1_max_3/episode_metrics.jsonl
- config_name: task-transfer--asterix_fixed_latency_2_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/asterix/asterix_200ep/asterix_fixed_latency_2_200ep_7k2steps/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/asterix/asterix_200ep/asterix_fixed_latency_2_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: task-transfer--asterix_fixed_latency_4_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/asterix/asterix_200ep/asterix_fixed_latency_4_200ep_7k2steps/train-00000-of-00001.parquet
  - split: val
    path:
    - latency-aware/asterix/asterix_200ep/asterix_fixed_latency_4_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: task-transfer--atlantis_fixed_latency_2_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/atlantis/atlantis_200ep/atlantis_fixed_latency_2_200ep_7k2steps/train-00000-of-00002.parquet
    - latency-aware/atlantis/atlantis_200ep/atlantis_fixed_latency_2_200ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - latency-aware/atlantis/atlantis_200ep/atlantis_fixed_latency_2_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: task-transfer--atlantis_fixed_latency_4_200ep_7k2steps
  data_files:
  - split: train
    path:
    - latency-aware/atlantis/atlantis_200ep/atlantis_fixed_latency_4_200ep_7k2steps/train-00000-of-00002.parquet
    - latency-aware/atlantis/atlantis_200ep/atlantis_fixed_latency_4_200ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - latency-aware/atlantis/atlantis_200ep/atlantis_fixed_latency_4_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: task-transfer--deadly_corridor_fix_latency_2_1000ep
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_2_1000ep/train.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_2_1000ep/val.parquet
- config_name: task-transfer--deadly_corridor_fix_latency_4_1000ep
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_4_1000ep/train.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_4_1000ep/val.parquet
- config_name: task-transfer--deadly_corridor_fix_latency_6_1000ep
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_6_1000ep/train.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_6_1000ep/val.parquet
- config_name: task-transfer--deadly_corridor_fix_latency_8_1000ep
  data_files:
  - split: train
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_8_1000ep/train.parquet
  - split: val
    path:
    - latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_8_1000ep/val.parquet
- config_name: task-transfer--asterix_fixed_latency_0_200ep_7k2steps
  data_files:
  - split: train
    path:
    - zero-latency/asterix/asterix_200ep/asterix_fixed_latency_0_200ep_7k2steps/train-00000-of-00001.parquet
  - split: val
    path:
    - zero-latency/asterix/asterix_200ep/asterix_fixed_latency_0_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: task-transfer--atlantis_fixed_latency_0_200ep_7k2steps
  data_files:
  - split: train
    path:
    - zero-latency/atlantis/atlantis_200ep/atlantis_fixed_latency_0_200ep_7k2steps/train-00000-of-00002.parquet
    - zero-latency/atlantis/atlantis_200ep/atlantis_fixed_latency_0_200ep_7k2steps/train-00001-of-00002.parquet
  - split: val
    path:
    - zero-latency/atlantis/atlantis_200ep/atlantis_fixed_latency_0_200ep_7k2steps/val-00000-of-00001.parquet
- config_name: task-transfer--deadly_corridor_fix_latency_0_1000ep
  data_files:
  - split: train
    path:
    - zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep/train.parquet
  - split: val
    path:
    - zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep/val.parquet
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
- config_name: deadly_corridor_fixed_latency_6_1000ep_7k2steps
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 684389145
  splits:
  - name: train
    num_bytes: 703563241
    num_examples: 42484
  - name: val
    num_bytes: 6147218
    num_examples: 368
- config_name: demon_attack_fixed_latency_6_200ep_7k2steps
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 763841275
  splits:
  - name: train
    num_bytes: 1223165610
    num_examples: 1055141
  - name: val
    num_bytes: 11386581
    num_examples: 9843
- config_name: flappy_fixed_latency_3_200ep_7k2steps
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 16591979414
  splits:
  - name: train
    num_bytes: 16894171290
    num_examples: 1393742
  - name: val
    num_bytes: 174776300
    num_examples: 14400
- config_name: flappy_fix_latency_2_200ep_7k2steps
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 16936709594
- config_name: deadly_corridor_fixed_latency_0_1000ep_7k2steps
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 701558795
  splits:
  - name: train
    num_bytes: 720184275
    num_examples: 43511
  - name: val
    num_bytes: 7153745
    num_examples: 433
- config_name: demon_attack_fixed_latency_0_200ep_7k2steps
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 1026147636
  splits:
  - name: train
    num_bytes: 1633484649
    num_examples: 1405067
  - name: val
    num_bytes: 16701818
    num_examples: 14395
- config_name: flappy_fixed_latency_0_200ep_7k2steps
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 16755505055
  splits:
  - name: train
    num_bytes: 17106878045
    num_examples: 1414816
  - name: val
    num_bytes: 128567571
    num_examples: 10653
- config_name: demon_attack_fixed_latency_6_200ep_7k2steps_ghost15
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 993608224
- config_name: flappy_fixed_latency_3_200ep_7k2steps_ghost15
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 25946160513
- config_name: inverted_pendulum_200ep_2__inverted_pendulum_fixed_latency_0_100ep_1ksteps_og
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  - name: state
    list: float32
  download_size: 311624367
- config_name: inverted_pendulum_200ep_2__inverted_pendulum_fixed_latency_2_1000ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  - name: state
    list: float32
  download_size: 613208881
- config_name: inverted_pendulum_200ep_2__inverted_pendulum_fixed_latency_4_1000ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  - name: state
    list: float32
  download_size: 209654575
- config_name: inverted_pendulum_200ep__inverted_pendulum_fixed_latency_0_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  - name: state
    list: float32
  download_size: 9666412181
- config_name: inverted_pendulum_200ep__inverted_pendulum_fixed_latency_2_1000ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  - name: state
    list: float32
  download_size: 613208881
- config_name: inverted_pendulum_200ep__inverted_pendulum_fixed_latency_4_1000ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  - name: state
    list: float32
  download_size: 209654575
- config_name: flappy_fix_latency_0_200ep_context5
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: context_images
    list: image
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 10602038543
- config_name: flappy_fix_latency_1_200ep_context5
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: context_images
    list: image
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 10841983057
- config_name: flappy_fix_latency_2_200ep_context5
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: context_images
    list: image
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 11351791009
- config_name: flappy_fix_latency_3_200ep_context5
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: context_images
    list: image
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 11092199608
- config_name: flappy_fix_latency_4_200ep_context5
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: context_images
    list: image
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 11016605946
- config_name: deadly_corridor_fix_latency_0
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 67742624
- config_name: deadly_corridor_fix_latency_1
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 36651540
- config_name: deadly_corridor_fix_latency_10
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 36553730
- config_name: deadly_corridor_fix_latency_12
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 44786426
- config_name: deadly_corridor_fix_latency_13
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 43609149
- config_name: deadly_corridor_fix_latency_14
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 40536512
- config_name: deadly_corridor_fix_latency_15
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 36304256
- config_name: deadly_corridor_fix_latency_2
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 42663026
- config_name: deadly_corridor_fix_latency_3
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 31973089
- config_name: deadly_corridor_fix_latency_4
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 43798230
- config_name: deadly_corridor_fix_latency_5
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 40272253
- config_name: deadly_corridor_fix_latency_6
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 36065188
- config_name: deadly_corridor_fix_latency_7
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 38885794
- config_name: deadly_corridor_fix_latency_8
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 29953330
- config_name: deadly_corridor_fix_latency_9
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 32283639
- config_name: demon_attack_200ep__demon_attack_fix_latency_0_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 295983006
- config_name: demon_attack_200ep__demon_attack_fix_latency_2_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 235259928
- config_name: demon_attack_200ep__demon_attack_fix_latency_4_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 229138153
- config_name: demon_attack_200ep__demon_attack_fix_latency_6_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 206983292
- config_name: demon_attack_200ep__demon_attack_fix_latency_8_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 181576540
- config_name: demon_attack_fix_latency_0
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 254540959
- config_name: demon_attack_fix_latency_1
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 256665766
- config_name: demon_attack_fix_latency_10
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 128828158
- config_name: demon_attack_fix_latency_11
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 91934511
- config_name: demon_attack_fix_latency_12
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 147323835
- config_name: demon_attack_fix_latency_13
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 133535114
- config_name: demon_attack_fix_latency_14
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 128934140
- config_name: demon_attack_fix_latency_15
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 110904516
- config_name: demon_attack_fix_latency_2
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 257547911
- config_name: demon_attack_fix_latency_3
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 251769191
- config_name: demon_attack_fix_latency_4
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 234514561
- config_name: demon_attack_fix_latency_5
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 232359621
- config_name: demon_attack_fix_latency_6
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 200936987
- config_name: demon_attack_fix_latency_7
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 205590797
- config_name: demon_attack_fix_latency_8
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 147339258
- config_name: demon_attack_fix_latency_9
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 140837088
- config_name: flappy_200ep__flappy_fix_latency_0_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 4031947259
- config_name: flappy_200ep__flappy_fix_latency_1_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 4232073507
- config_name: flappy_200ep__flappy_fix_latency_2_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 4609014545
- config_name: flappy_200ep__flappy_fix_latency_3_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 4430345648
- config_name: flappy_200ep__flappy_fix_latency_4_200ep
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  download_size: 4376744936
- config_name: flappy_fix_latency_0_small
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 1230554693
- config_name: flappy_fix_latency_1_small
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 1266677976
- config_name: flappy_fix_latency_2_small
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 1271808270
- config_name: flappy_fix_latency_3_small
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 1272004102
- config_name: flappy_fix_latency_4_small
  features:
  - name: image
    dtype: image
  - name: split
    dtype: string
  - name: prompt
    dtype: string
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: latency
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: reward
    dtype: float64
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: t
    dtype: int64
  - name: seed
    dtype: int64
  - name: deterministic
    dtype: bool
  download_size: 1211318389
- config_name: flappy_fixed_l2_fs4
  features:
  - name: action
    list: float32
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: deterministic
    dtype: bool
  - name: done
    dtype: bool
  - name: episode_idx
    dtype: int64
  - name: image_stack
    list: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  - name: t
    dtype: int64
  download_size: 11243675137
- config_name: task-transfer--asterix_fixed_latency_2_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  splits:
  - name: train
    num_examples: 256820
  - name: val
    num_examples: 2598
  download_size: 329687901
- config_name: task-transfer--asterix_fixed_latency_4_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  splits:
  - name: train
    num_examples: 222525
  - name: val
    num_examples: 2154
  download_size: 283416109
- config_name: task-transfer--atlantis_fixed_latency_2_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  splits:
  - name: train
    num_examples: 356400
  - name: val
    num_examples: 3600
  download_size: 577290972
- config_name: task-transfer--atlantis_fixed_latency_4_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  splits:
  - name: train
    num_examples: 356400
  - name: val
    num_examples: 3600
  download_size: 588296037
- config_name: task-transfer--deadly_corridor_fix_latency_2_1000ep
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_examples: 32500
  - name: val
    num_examples: 4000
  download_size: 558841872
- config_name: task-transfer--deadly_corridor_fix_latency_4_1000ep
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_examples: 28737
  - name: val
    num_examples: 3178
  download_size: 501369715
- config_name: task-transfer--deadly_corridor_fix_latency_6_1000ep
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_examples: 34290
  - name: val
    num_examples: 3890
  download_size: 597152880
- config_name: task-transfer--deadly_corridor_fix_latency_8_1000ep
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_examples: 28895
  - name: val
    num_examples: 3245
  download_size: 517062764
- config_name: task-transfer--asterix_fixed_latency_0_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  splits:
  - name: train
    num_examples: 272266
  - name: val
    num_examples: 2768
  download_size: 355282674
- config_name: task-transfer--atlantis_fixed_latency_0_200ep_7k2steps
  features:
  - name: image
    dtype: image
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: decision_step
    dtype: int64
  - name: split
    dtype: string
  - name: seed
    dtype: int64
  - name: prompt
    dtype: string
  - name: latency_raw_frames
    dtype: int64
  - name: latency_ms
    dtype: float64
  - name: raw_reward
    dtype: float64
  - name: action_id
    dtype: int64
  - name: action
    list: float32
  - name: action_text
    dtype: string
  splits:
  - name: train
    num_examples: 356400
  - name: val
    num_examples: 3600
  download_size: 599995003
- config_name: task-transfer--deadly_corridor_fix_latency_0_1000ep
  features:
  - name: action_id
    dtype: int64
  - name: action_text
    dtype: string
  - name: decision_step
    dtype: int64
  - name: env_name
    dtype: string
  - name: episode_idx
    dtype: int64
  - name: image
    dtype: image
  - name: latency_ms
    dtype: float64
  - name: latency_raw_frames
    dtype: int64
  - name: prompt
    dtype: string
  - name: raw_reward
    dtype: float64
  - name: seed
    dtype: int64
  - name: split
    dtype: string
  splits:
  - name: train
    num_examples: 38308
  - name: val
    num_examples: 4295
  download_size: 661234054

---

# LAGEN datasets

Project resources: [LAGEN collection](https://huggingface.co/collections/MLL-Lab/lagen-6ac750f21139008b8114fff5).

Training data, episode-level evaluations and paper experiment manifests.

| Experiment | Entry |
|---|---|
| Sim2Real calibration | [30-case held-out calibration](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/50d8f0aef64749a849760fc580ce3efb3c5c5e76/experiments/sim2real-calibration) |
| Visual history | [Visual history](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/visual-history) |
| Latency in prompt | [Latency in prompt](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/latency-in-prompt) |
| Latency transfer | [Latency transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/latency-transfer) |
| Task transfer | [Task transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/task-transfer) |
| VLA fine-tuning scope | [VLA fine-tuning scope](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/vla-finetuning-scope) |
| Mean vs. profile training | [Mean vs. profile training](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/mean-vs-profile) |
| Observation stride | [Observation stride](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/observation-stride) |
| Context window | [Context window](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/context-window) |

| Resource | Repository |
|---|---|
| benchmark-models | [MLL-Lab/LAGEN-models](https://huggingface.co/MLL-Lab/LAGEN-models) |
| profiles | [MLL-Lab/LAGEN-profiles](https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles) |

[Benchmark release inventory](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/blob/main/reports/benchmark-release/README.md)

## Reproduction

Select the exact task, checkpoint, dataset, and profile from the linked experiment manifest.
Each result identifies its episode count, clocks, latency settings, and seeds.
The manifest records any missing source bindings.
