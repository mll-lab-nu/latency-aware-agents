# Benchmark artifact registry

This is the human-readable index for the four supported zero-latency task
pipelines. New artifacts use two central Hugging Face repositories:

- models: [`latency-sensitive-bench/benchmark-models`](https://huggingface.co/latency-sensitive-bench/benchmark-models)
- datasets: [`latency-sensitive-bench/benchmark-datasets`](https://huggingface.co/datasets/latency-sensitive-bench/benchmark-datasets)
- profiles: [`latency-sensitive-bench/profiles`](https://huggingface.co/datasets/latency-sensitive-bench/profiles)

Paths below are stable task prefixes. The task-owned scripts write local files;
publishing is an explicit developer operation. A status of `planned` means the
central path is reserved but must not be treated as downloadable.

The checked-in source for the central dataset card and its explicit Viewer
configs is [`benchmark_datasets_card.md`](benchmark_datasets_card.md).

## Model artifacts

| Task | Role | Local source | Central path | Status |
| --- | --- | --- | --- | --- |
| MIKASA Intercept Grab Fast | Small-policy PPO teacher | `outputs/tasks/mikasa_intercept_grab_fast/strict_teacher/oracle_checkpoints/InterceptGrabFast-VLA-v0/final_success_ckpt.pt` | `zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/` | Published |
| MIKASA Intercept Grab Fast | QwenOFT VLA inference bundle | `third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable/` | `zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-v1/` | Published; 5k SFT and eval accepted |
| Balance Simple | FastTD3 teacher | `results/humanoidbench_g1_balance_control_comparison_fasttd3_25m/checkpoints/g1_sonic_balance_simple_raw_ramp_0p1s_fasttd3_25000000steps_s1/checkpoint_p0/best_000325980_20864000_reward_702.976.pth` | `zero-latency/humanoidbench-balance-simple/small-policy/fasttd3-v1/` | Published |
| Balance Simple | Official GR00T inference bundle | `outputs/tasks/humanoidbench_balance_simple/vla_obs198_h40/checkpoint-15000/` | `zero-latency/humanoidbench-balance-simple/vla/gr00t-n1.7-15k-v1/` | Fresh 15k run pending; do not fall back to 10k |
| Extreme Parkour Go2 | Privileged PPO teacher | `third_party/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour/2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k/model_14999.pt` | `zero-latency/extreme-parkour-go2/small-policy/ppo-v1/` | Teacher verified; publication pending VLA pipeline |
| Extreme Parkour Go2 | H40 action+yaw GR00T VLA | `outputs/official_gr00t/extreme_parkour_go2_action_yaw_h40_official_flow_aux_dagger/training/extreme_parkour_go2_vla_dagger_action_yaw_h40_official_flow_aux_5k/checkpoint-5000/` | `zero-latency/extreme-parkour-go2/vla/gr00t-action-yaw-v1/` | Pending fresh flow + auxiliary run; do not publish |
| HAIC Pull Cart G1 | Native PPO teacher | `outputs/haic/phase1_teacher/checkpoint_final.pt` | `zero-latency/haic-pull-cart/small-policy/ppo-v1/` | Teacher verified; publication pending VLA pipeline |
| HAIC Pull Cart G1 | Action-only GR00T VLA | `outputs/official_gr00t/haic_pull_cart_vla_action_h40_dagger/training/haic_pull_cart_vla_dagger_action_h40_official_flow_aux_5k/checkpoint-5000/` | `zero-latency/haic-pull-cart/vla/gr00t-action-only-v1/` | Pending fresh flow + auxiliary run; do not publish |

Do not replace an exact source path with a `latest` symlink in a publication
record.

## Dataset artifacts

| Task | Contents | Local source | Central path | Status |
| --- | --- | --- | --- | --- |
| MIKASA Intercept Grab Fast | 275 successful episodes; raw NPZ and LeRobot 250/25 split | `outputs/mikasa/grabfast_h1_dataset/` | `zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/` | Published |
| Balance Simple | LeRobot 90/10 split; scene assets referenced separately | `outputs/official_gr00t/teachers/g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198/` | `zero-latency/humanoidbench-balance-simple/teacher-rollouts-v1/` | Published |
| Extreme Parkour Go2 | Action-rich 250/25 bootstrap and completed DAgger rounds | `outputs/official_gr00t/extreme_parkour_go2_action_data/` and task DAgger output | `zero-latency/extreme-parkour-go2/teacher-rollouts-v1/` | Bootstrap rebuild/pipeline in progress |
| HAIC Pull Cart G1 | Action-only 250/25 bootstrap and completed DAgger rounds | `outputs/official_gr00t/haic_pull_cart_vla_action_dagger/bootstrap/` | `zero-latency/haic-pull-cart/teacher-rollouts-v1/` | Bootstrap in progress; do not publish |

Do not put optimizer state, W&B directories, temporary raw shards, or local
absolute-path manifests in either central repository. Keep large scene assets
in their existing dedicated repository.

## Deployment profiles

Completed profiles are published under
`profiles/<model>/<gpu-deployment>/<workload>/<physical-instance>/` in the
profiles dataset repository. The associated raw five-session run is published
under `runs/` only after all five 100-episode sessions succeed. Model provenance
records the central model path and resolved revision, not a machine-local
absolute path.

## Historical sources

These task-specific repositories remain unchanged and are listed so an older
result can be traced to its original source. They are not the new publication
layout.

| Task | Historical artifact | Source repository |
| --- | --- | --- |
| MIKASA | PPO teacher | [`latency-sensitive-bench/mikasa-robo`](https://huggingface.co/latency-sensitive-bench/mikasa-robo) |
| MIKASA | Teacher rollouts | [`latency-sensitive-bench/mikasa-robo`](https://huggingface.co/datasets/latency-sensitive-bench/mikasa-robo) |
| Balance Simple | Official GR00T 10k bundle | [`latency-sensitive-bench/humanoidbench`](https://huggingface.co/latency-sensitive-bench/humanoidbench) |
| Balance Simple | Teacher rollouts | [`latency-sensitive-bench/humanoidbench`](https://huggingface.co/datasets/latency-sensitive-bench/humanoidbench) |
| Balance Simple | Scene assets | [`latency-sensitive-bench/g1-sonic-assets`](https://huggingface.co/datasets/latency-sensitive-bench/g1-sonic-assets) |

## StarVLA H1 models and shared data

The four StarVLA models (`qwenoft`, `qwenpi_v3`, `qwengr00t`, `wanoft`) use
`zero-latency/<task>/vla/starvla-<model>-h1/<run-id>/` in the central model
repository. Formal pipelines upload after final checkpoint L0 evaluation;
smoke runs stay local. See [SCALE_MODELS](../program/SCALE_MODELS.md) for commands.

Each task shares `zero-latency/<task>/teacher-rollouts-h1-state-v1/` across all
four models. Parkour and HAIC additionally publish model-specific DAgger deltas
at `zero-latency/<task>/dagger/starvla-<model>-h1/<run-id>/round-NN/`.
These are new H1 datasets; HAIC data contains a fresh 50 Hz image, state, and
teacher latent at each control step. Existing issued-H40 data is not reused.
The dataset README and task contract describe fields and fixed train/validation
normalization. Model provenance references the exact shared and delta revisions.

Teacher checkpoint downloads for all four tasks are pinned to model repository
revision `46abd9ece5745d808d16aa8013766a26b4f6c371` by the H1 pipeline, including
the Parkour and HAIC teachers now present in that revision.
