# Humanoid latency research proposal

[English](./humanoidbench.md) | [中文](./humanoidbench.zh.md)

> Status: working proposal
>
> Last updated: 2026-08-26

## Conclusion

The default route is:

```text
HumanoidBench task/scene/reward
        +
official SONIC-aligned Unitree G1
        ↓
per-task FastTD3 teacher: proprioception + privileged task state → 64D body token + 14D hands
        ↓
frozen SONIC decoder at 50 Hz → 29D body position targets + 14D hand targets
        ↓
quality-gated rollouts: RGB + language + proprioception → executed 78D command
        ↓
StarVLA training infrastructure + official NVIDIA GR00T N1.7 model/processor
        ↓
VLA at about 2.5 Hz → 40 × (64D body token + 14D hands)
        ↓
SONIC at 50 Hz, with the benchmark injecting VLA observation/inference latency
```

The corrected first experiment uses the same continuous `Box(78)` teacher contract for four supported tasks: 64 SONIC body-token dimensions followed by 14 learned hand targets. The matching flat baseline uses `Box(43)`: 29 direct body targets followed by the same 14 hand targets. Each task may have its own teacher checkpoint, but not its own action abstraction. A higher-level planner, motion reference, or skill library is considered only for a task that still fails after the mapping, reward, controller contract, and training budget have been checked.

This route deliberately gives up native HumanoidBench comparability. The objective is a controlled latency benchmark for our own embodied VLA pipeline, not a leaderboard entry under the original robot and actuator definitions.

## Goals and non-goals

### Goals

- Study how delayed visual observation and VLA inference affect whole-body task performance.
- Produce useful teacher data for a VLA rather than merely maximize the original scalar reward.
- Reuse one body command representation across locomotion, balance, and basketball.
- Keep the entire experiment in MuJoCo simulation.
- Retain the existing fast off-policy training path where possible.

### Non-goals

- Native HumanoidBench score comparability.
- Real-robot deployment, VR teleoperation, sim-to-real transfer, or human demonstrations.
- Reimplementing SONIC or NVIDIA GR00T inside this repository.
- Building AMP, DeepMimic, PULSE, WIMLE, and a planner in parallel before the minimal route has been tested.
- Proving that a task-reward-optimal policy is human-like.

## What the existing experiments established

### Flat joint-space RL learns task reward, not necessarily useful motion

The first Sample Factory APPO baseline used 32 workers, rollout 64, batch 2048, one epoch, and 10M environment steps:

| Task | Return | Episode length |
|---|---:|---:|
| Balance Hard | 33.74 | 37.9 |
| Hurdle | 14.43 | 76.7 |
| Run | 12.86 | 93.5 |
| Basketball | 12.12 | 31.5 |
| High Bar Hard | 2.81 | 56.8 |

Run configuration: [baseline experiment summary](../results/humanoidbench_10m_baseline/generated/summary.json).

An official-aligned APPO attempt changed the optimization schedule to rollout 256, batch 512, ten epochs, 16 batches per epoch, and learning rate `3e-5`, still at 10M steps:

| Task | Return | Episode length |
|---|---:|---:|
| Balance Hard | 73.20 | 92.3 |
| Hurdle | 12.64 | 56.0 |
| Run | 20.25 | 96.3 |
| Basketball | 31.46 | 60.5 |
| High Bar Hard | 2.57 | 25.4 |

Run configuration: [official-aligned experiment summary](../results/humanoidbench_sf_official_10m_wandb/generated/summary.json). High Bar was aggregated directly from its [ten evaluation episodes](../results/humanoidbench_sf_official_10m_wandb/h1strong_highbar_hard/eval_best_l0/eval_latency_0/episode_metrics.jsonl), because that run has no queue summary.

More optimizer work helped some tasks but did not solve the family. The completed flat-action FastTD3 runs were much stronger:

| Task | Return | Episode length | Training FPS |
|---|---:|---:|---:|
| Run | 907.21 ± 4.72 | 1000 | 4109 |
| Hurdle | 503.72 ± 163.57 | 930.8 | 4864 |

Run configuration: [FastTD3 experiment summary](../results/humanoidbench_fasttd3/generated/summary.json).

Manual inspection of their evaluation videos nevertheless found visibly unnatural motion. We have not yet computed quantitative motion-quality metrics, so this is a qualitative finding rather than a measured human-likeness score:

- [Run evaluation video](../results/humanoidbench_fasttd3/h1hand_run/eval_best_l0/eval_latency_0/videos/bin_00_episode_000003.mp4)
- [Hurdle evaluation video](../results/humanoidbench_fasttd3/h1hand_hurdle/eval_best_l0/eval_latency_0/videos/bin_00_episode_000007.mp4)

This is consistent with the task definition: the original reward can favor forward progress, balance, and task completion without specifying natural arm pose, gait symmetry, foot slip, impact, or action smoothness. [FastTD3's HumanoidBench report](https://younggyo.me/fast_td3/) and [MuJoCo MPC experiments](https://arxiv.org/abs/2408.00342) report the same reward-design failure mode.

### The MuJoCo environment is not the main flat-RL throughput bottleneck

The quick single-measurement benchmark was:

| Task | Raw env, 1 env | Raw env, 32 envs | SF aggregate, 32 envs | SF steady 10 s, 32 envs |
|---|---:|---:|---:|---:|
| Balance Hard | 471 | 15024 | 4103 | 7373 |
| Hurdle | 532 | 16617 | 4007 | 6758 |
| Run | 692 | 22466 | 4815 | 7782 |
| Basketball | 632 | 20412 | 4714 | 7782 |
| High Bar Hard | 201 | 6445 | 2189 | 5325 |

Raw measurements: [throughput CSV](../results/humanoidbench_throughput/quick/throughput.csv).

The raw environments scale with parallelism, and flat FastTD3 reaches roughly 4k–5k FPS. A SONIC decoder adds a new cost, so this result must not be treated as a promise that token-space training will retain the same throughput.

The earlier memory incident was also not a `/dev/shm` limit: the host exposed about 1.84 TiB, while the job cgroup had a limit and observed peak of roughly 483 GB.

### Abandoned and inconclusive experiments

- WIMLE is not a current implementation route. Its historical prototype performed about 4.4–4.7 steps/s because every environment step synchronously triggered many small optimizer updates. There is no retained implementation or reproducible checkpoint in the current tree.
- The previous squash experiment is inconclusive: its saved configuration does not prove that the intended `continuous_tanh_scale` behavior was active. It is not evidence for or against action squashing.
- Increasing APPO `num_epochs` improved early reward but reduced throughput and reached another plateau. It does not identify a single universal hyperparameter fix.

### FastTD3 + SONIC Run contract smoke

The first Run-only integration now uses the pinned SONIC v1.1 decoder and its exact 29-DoF G1 MuJoCo asset. FastTD3 sees a continuous `Box(64)` action; the environment builds the official 994D decoder input, holds the decoded 29D body target for one 50 Hz control interval, recomputes PD control over ten 2 ms MuJoCo substeps, and fixes the 14 hand targets at the relaxed pose.

The repository-side contract checks pass: deterministic seeded reset, decoder history layout and zero padding, IsaacLab-to-MuJoCo joint mapping, 64D action and 99D privileged observation spaces, one-step execution, RGB rendering, and a 1000-step zero-token rollout. The local dynamics intentionally use the deployment-style `0.002 × 10` timing rather than SONIC's training simulator `0.005 × 4`; both have a 50 Hz policy interval, but they are not trajectory-identical dynamics.

A 100k-step FastTD3 smoke completed end to end and produced final, best, and periodic checkpoints plus deterministic evaluation artifacts. It is a pipeline check, not a learning result:

| Measurement | Result |
|---|---:|
| Collected environment steps | 100,224 |
| End-to-end wall-clock FPS, including startup and compilation | 865.7 |
| Best-checkpoint evaluation return, 10 episodes | 16.60 ± 1.31 |
| Mean evaluation episode length | 132.5 |
| Evaluation video | [MP4](../results/humanoidbench_sonic_fasttd3/smoke_100k/g1_sonic_run/eval_best_l0/eval_latency_0/videos/bin_00_episode_000002.mp4) |

The short policy still falls and is not yet a teacher candidate, but its early learning signal is strong. At about 100k steps, the original 61D FastTD3 Run training had a rolling return of 1.72, while SONIC reached 6.56. The original run first reached a comparable 6.63 at about 649k steps. Its rolling return first crossed 16.6 near 2.65M steps, although that number must not be compared directly with SONIC's deterministic 16.6 evaluation return. The evidence supports continuing after performance optimization; it does not yet establish final task success or natural motion.

W&B was disabled because this was a short debug run; generated long-run configurations enable the existing `latency-sensitive-bench` project and `humanoidbench-sonic-fasttd3` group.

The current per-environment, single-threaded CPU ONNX Runtime decoder scales as follows in one measurement per count:

| Environments | Aggregate environment FPS | FPS per environment |
|---:|---:|---:|
| 1 | 149.9 | 149.9 |
| 4 | 441.0 | 110.3 |
| 8 | 890.3 | 111.3 |
| 16 | 1515.7 | 94.7 |
| 32 | 2519.3 | 78.7 |

This was enough to validate the design but not enough to justify a 10M-step run. In that historical Run-only experiment, the optimized path moved one dynamic-batch ONNX decoder into the Sample Factory inference worker; actor/replay stored 64D while only the decoded 29D target reached MuJoCo. A later 100k verification reached 5.06k–5.38k steady-state FPS; startup-inclusive FPS was 1.52k because 128 workers take roughly 40 seconds to start. The five-task correction below supersedes that action contract with learned hands and full 78D replay.

The resulting 1M-step W&B run completed successfully:

| Measurement | Result |
|---|---:|
| Collected environment steps | 1,000,192 |
| Total FPS including startup and shutdown | 4,329.6 |
| Mature 60-second FPS | 5,495.5 |
| Best-checkpoint deterministic return, 10 episodes | 139.16 ± 20.51 |
| Mean episode length | 922.8 / 1000 |
| Full-horizon episodes | 3 / 10 |
| W&B | [run](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/humanoidbench_sonic_fasttd3_gpu_decoder_g1_sonic_run) |
| Evaluation video | [MP4](../results/humanoidbench_sonic_fasttd3_gpu_decoder_1m/g1_sonic_run/eval_best_l0/eval_latency_0/videos/bin_00_episode_000005.mp4) |

This proves fast token-space learning on Run, but the video still determines whether the learned motion is acceptable teacher data. It does not justify 10M steps or generalize to the other four tasks by itself.

### Historical parity correction before long training

The old G1 comparison mixed a different robot with simplified task scenes and approximate local reward code. That is the root cause of the uninterpretable H1/G1 gap; optimizer tuning on those runs would not answer the intended question. The corrected candidate now:

- includes the official HumanoidBench floor, hurdle, balance, and basketball task assets unchanged;
- calls the official `Run`, `Hurdle`, `BalanceHard`, and `Basketball` reward/reset/termination classes directly;
- changes only the robot/controller, G1 keyframes, observation/action dimensions, and required task sites/sensors/cameras;
- exposes raw `43D = 29 body + 14 hands` and SONIC `78D = 64 token + 14 hands` actions;
- keeps raw actions in MuJoCo joint order and applies the Isaac-to-MuJoCo permutation only to SONIC decoder output;
- keeps the original `0.002 × 10 = 20 ms` timing and task-specific tracking cameras.

The historical candidate XMLs loaded and rendered with the expected `(nq, nv, nu)`, and the eight family contract tests passed. A two-GPU, no-W&B Hurdle smoke ran 65,792 steps per variant through real learner updates: raw43 completed at 1,201.7 total FPS and sonic78 at 1,197.1 total FPS while both jobs shared host CPUs; their mature 10-second rates were about 1.6k and 1.7k respectively. This is a pipeline check, not the later dedicated throughput benchmark.

The remaining historical gate was manual review of paired H1/G1 videos. No corrected asset revision was published from that candidate.

The previously published [`latency-sensitive-bench/g1-sonic-assets`](https://huggingface.co/datasets/latency-sensitive-bench/g1-sonic-assets) revision is invalid for the historical comparison: it used simplified local scenes rather than the official HumanoidBench task assets and logic. The supported replacement now omits both High Bar tasks. The decoder remains pinned separately to NVIDIA's `nvidia/GEAR-SONIC` revision `1157dbafbae49ba17b8b53ace04cd8191b6a6697`.

## Why SONIC is the preferred body interface

[SONIC](https://arxiv.org/html/2511.07820) is a learned whole-body controller for Unitree G1. Its public VLA workflow exposes a compact motion-token command while the low-level policy handles the high-rate body action. For this project it provides three useful properties:

1. The teacher and VLA can use the same body command representation.
2. A learned whole-body prior may prevent some degenerate raw-joint solutions before adding imitation machinery.
3. VLA latency can be studied at the command-chunk boundary while the low-level controller continues at 50 Hz.

SONIC does not by itself guarantee task success, human motion, or natural fingers. Those remain empirical properties of the teacher, task reward, and hand command.

### The body command is a continuous 64D vector

SONIC uses finite scalar quantization internally during training, but the official deployment client sends a checkpoint-specific 64D `float32` motion token. It also linearly blends commands, which would not make sense for a mandatory runtime token-ID interface. Therefore:

- The SONIC body prefix should remain a continuous `Box(64)`; the complete teacher action is `Box(78)` after appending learned hands.
- Do not add a quantization wrapper or switch to MultiDiscrete APPO.
- Treat the 64 dimensions as a checkpoint-specific latent contract, not individually meaningful controls.

The supporting public contracts are in the [SONIC VLA workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html), [inference guide](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html), and [`features_sonic_vla.py`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data/features_sonic_vla.py).

### Start every task at the 64D layer

The smallest useful teacher is:

```text
observation: robot proprioception + privileged task/object state
action:      64D continuous SONIC body token + 14D hand targets
algorithm:   FastTD3
decoder:     frozen SONIC policy
```

This does not mean one actor is shared by all tasks. It means per-task actors share one action contract, replay format, decoder, and eventual VLA target.

If a task fails, first check:

1. the G1 scene and reset state;
2. reward and termination semantics;
3. token scaling, history, proprioception, and SONIC checkpoint compatibility;
4. whether the frozen decoder can express the required movement;
5. training stability and sufficient updates.

Only after those checks should that task reopen a higher-level planner, reference tracker, or skill interface. Adding such a layer now would create an untested second command language and make the teacher/VLA contract harder to interpret.

## Robot and task adaptation

### Use the exact official SONIC G1

Native HumanoidBench G1 is not wire-compatible with the public SONIC controller:

| Property | Native HumanoidBench G1 | Official SONIC-aligned G1 |
|---|---:|---:|
| Body DoF | 23 | 29 |
| Hand DoF | 14 | 14 |
| Body actuation | torque | position targets through the SONIC/PD contract |
| Intended command | raw actuator action | 64D motion token |

Because native score comparability is not required, the clean solution is to bring the official SONIC-aligned G1 model and controller contract into the HumanoidBench scenes. Do not retarget SONIC to the native 23-DoF torque robot and do not retrain SONIC before there is evidence that the official body cannot support the tasks.

HumanoidBench upstream has 32 tasks, but this SONIC G1 adaptation supports 30. `highbar_simple` and `highbar_hard` are explicitly unsupported: they are not registered, generated, compared, or trained. The supported tasks reuse one shared official G1/controller definition rather than forking a robot per task.

### Initial four tasks

| Task | Scene adaptation | Initial hand policy |
|---|---|---|
| Balance Hard | official task asset/class; replace only the robot and G1 keyframe | learned 14D targets |
| Run | official floor and `Run` task class; replace only the robot/keyframe | learned 14D targets |
| Hurdle | official hurdle assets and `Hurdle` task class; replace only the robot/keyframe | learned 14D targets |
| Basketball | official court, ball reset, reward, and termination; replace only the robot/keyframe | learned 14D targets |

No custom reward, termination, obstacle, ball, board, bar geometry, camera semantics, or timing change is allowed in this parity pass. Task-specific adaptation is limited to morphology-specific reset. Native HumanoidBench numbers remain historical smoke references rather than directly comparable scores because the robot and controller differ.

### Decision: align the supported 30-task family before more long training

The initial correction exposed family contracts that must be fixed once before extending training:

- HumanoidBench's `robot.dof` means the length of robot `qpos`, not the actuator count. The SONIC G1 must therefore expose `dof=50`, `nv=49`, and 43 actuators to every task. Leaving the native G1 value of 44 causes task observations and rewards to misclassify the final six robot coordinates as task state.
- Policy observations must come from each official `task.get_obs()`. A flat model-state concatenation omits goals or phases used by Reach, Push, Cube, Package, Bookshelf, and Spoon, and leaves Kitchen reward state stale.
- The shared robot asset must supply the complete HumanoidBench root, site, sensor, and eight-camera contract. These names belong to the robot adapter, not to 30 task patches.
- Initial-state parity means task-space parity, not equality between H1 and G1 joint vectors. Task objects, root placement, feet, palms, contacts, COM, and camera semantics are aligned while each morphology keeps legal joint coordinates.

The implementation registers the 30 supported task keys with the same raw `43D` and SONIC `78D` interfaces. Each supported task uses its native G1 keyframe as the morphology-specific seed. Room is the reset exception: its task objects use the fork's non-overlapping, floor-clear placement, while SONIC G1 keeps the candidate's stable body pose and inherits only the reference scene placement. This is a semantic joint retarget, not an indexed copy: the free root and 12 leg joints map directly; native `torso_joint` maps to SONIC `waist_yaw_joint`; elbow pitch/roll map to elbow/wrist roll; native hand joints zero–two map to thumb, three–four to index, and five–six to middle. SONIC-only waist roll/pitch and wrist pitch/yaw stay neutral. Task-object qpos after the native 44-qpos robot boundary is copied exactly.

The current gate result is 30/30 supported task-space initializations. High Bar Simple and High Bar Hard are intentionally outside the SONIC G1 support boundary because the G1 hand morphology does not provide a validated equivalent grasp. They have no SONIC registry entry, scene, comparison, or training job.

Generic upstream bugs are fixed as separate commits in [`mindorigin150/humanoid-bench`](https://github.com/mindorigin150/humanoid-bench): Kitchen's MuJoCo 3.3 default nesting, the G1 `free_base` contract, Room's non-overlapping floor-clear object reset, Truck episode bookkeeping, the Truck package-density omission, Spoon target phase reset, High Bar reset noise ordering, and Package's stale randomized-destination observation. The main repository pins that fork as a submodule. Upstream [issue #62](https://github.com/carlosferrazza/humanoid-bench/issues/62) independently reports the exact G1 Truck `free_base` failure; [issue #11](https://github.com/carlosferrazza/humanoid-bench/issues/11) also documents the recurring confusion that `robot.dof` includes free-root qpos rather than meaning actuator count. We adopt [Truck density report #60](https://github.com/carlosferrazza/humanoid-bench/issues/60) as an explicit physics-contract fix: all ten movable Truck package visual/collision geoms use `density="5"`, matching the existing Package task, instead of MuJoCo's default density. This changes Truck physics and therefore invalidates earlier Truck training results; it is tracked separately from robot/reset parity.

The corrected asset bundle is built locally first and includes source revisions and all upstream licenses. Manual review receives exactly one side-by-side MP4 and one start/mid/end PNG per supported task under `comparisons/<task>/`. No corrected Hugging Face revision and no 10M training starts until all 30 supported environments compile, reset, step, render, and the comparisons are approved. After that gate, long training is the four-task `raw43` versus `sonic78` matrix.

Local use is explicit and does not modify HumanoidBench's native registry: import `latency_bench.envs.humanoidbench_sonic`, choose `G1_SONIC_ENV_SPECS["g1_sonic_<task>"]`, and pass that task's generated `scene_path` to `gym.make`. Raw control selects `raw_body_target`; SONIC selects `sonic_token` plus the pinned decoder. After manual approval, `prepare_g1_sonic_assets(cache_dir)` will download the pinned Hugging Face asset revision and decoder, and the FastTD3 experiment generator will resolve the selected task scenes automatically. Until the pending revision is replaced, the helper intentionally cannot generate a long-run job from the obsolete public bundle.

## Hands are a separate control problem

The official VLA action is:

```text
64D body motion token + 7D left hand + 7D right hand = 78D
```

The 64D body token does not constrain the 14 hand joints. If a flat RL actor emits unconstrained fingers for Run or Hurdle, they can remain visually strange even when body motion improves. The public deployment path treats hands as a separate command stream; an optional training-time hand decoder does not change that external contract.

For the strict task-parity experiment, all four actors learn the full 14D hand stream. This keeps the comparison honest: H1Hand exposes controllable hands and Basketball requires them. Run and Hurdle do not reward natural fingers, so visually strange hand motion is an expected data-quality risk; do not hide it with a scripted hand fallback or a custom hand reward before the parity result is measured.

The rollout dataset and replay must store the full executed 78D command. If unconstrained fingers later make otherwise successful locomotion unusable as teacher data, the smallest follow-up is an explicit hand smoothness/neutral-pose cost reported separately from raw task reward.

## Sample Factory integration

### Minimal learner boundary

FastTD3 remains responsible for actor, critic, replay, and optimization. The HumanoidBench G1 environment presents `Box(78)`, while the matching raw baseline presents `Box(43)`. The environment decodes the token prefix when required, applies the policy's hand suffix, and advances MuJoCo.

FastTD3 replay keeps the complete 78D learnable action. The inference worker sends only the first 64 dimensions through SONIC and injects the decoded 29D body target into MuJoCo; the original 78D action, including its 14D hand suffix, remains in the trajectory and replay. Do not replace it with the decoded 29D body target.

### Throughput constraint

The official C++/ONNX MuJoCo demo is useful for correctness. Evaluation and nonzero-latency runs use one in-process, single-threaded CPU ONNX Runtime session per environment because decoder execution must remain inside the delayed environment boundary.

Zero-latency FastTD3 training instead uses one CUDA ONNX Runtime session in the Sample Factory inference worker. It concatenates the first 64 dimensions of each 78D actor action with the 930D controller state on GPU, binds GPU input/output buffers directly, and injects the decoded 29D body target into the environment. Actor output, replay action, and scheduler-visible action remain 78D. Batches of 1, 8, 32, and 64 execute successfully; GPU versus CPU output differs by less than `0.01` maximum absolute error.

The decoder must receive the real controller history and proprioception at each 50 Hz policy step. Holding its target across that step's ten 2 ms MuJoCo substeps is intentional; reusing it across multiple policy steps would change the SONIC control contract.

Before long training, report separately:

- raw MuJoCo FPS;
- SONIC decode time and batch size;
- end-to-end environment FPS;
- learner FPS and GPU utilization.

The optimized 100k verification reached 5.06k–5.38k steady-state FPS and passed the 3.5k gate. Startup-inclusive FPS remains misleading for short runs because 128 environment processes take roughly 40 seconds to initialize. Long-run reports must therefore include both steady-state and total wall-clock FPS.

## Teacher quality and rollout data

### Success is necessary but not sufficient

A teacher checkpoint is eligible only if it satisfies both:

1. task success or raw-return threshold;
2. motion-quality threshold.

The first token-space run should log at least:

- official/raw task return and episode length;
- shaped return, if shaping is introduced;
- token magnitude and temporal difference;
- decoded target velocity/acceleration;
- actuator effort or power;
- foot slip, contact impact, and non-foot contacts for locomotion;
- joint-limit margin;
- task-specific failure reasons.

These metrics diagnose motion but do not establish human-distribution alignment. Human-like motion ultimately needs an external reference distribution.

### When simple regularization is enough

For a VLA teacher, the requirement is not automatically “look exactly human.” Simple regularization is sufficient if the decoded trajectories are:

- dynamically stable and task-successful;
- smooth enough not to teach jitter or controller abuse;
- free of obvious contact exploitation and joint-limit abuse;
- diverse enough for the task;
- robust under closed-loop state deviations.

SONIC already supplies a motion prior, so the first token-space experiment should not add AMP or motion tracking. If high-reward token policies still exploit the scene, add the smallest task-aware penalties and keep raw reward separate for reporting.

### When reference motion becomes necessary

Reference tracking is warranted only if downstream use requires a recognizable motion style or if token-space regularization cannot remove a repeated failure mode. Examples include a specific human running gait, a prescribed high-bar technique, or a basketball motion whose body/hand coordination cannot be expressed by the minimal side-channel.

The first escalation should be a fixed, stationary reference reward because it is compatible with off-policy replay. AMP and learned adversarial rewards are later options: their changing discriminator reward complicates replay and can conflict with sparse task reward. [HumanoidOlympics](https://humanoidolympics.github.io/) documents both unnatural task-optimal PPO behavior and cases where style rewards impede task completion.

### Dataset contract

Each training sample should contain:

| Field | Source | Student-visible? |
|---|---|---|
| RGB observation | simulated camera | yes |
| language instruction | task template | yes |
| robot proprioception | official G1 state contract | yes |
| future 40 × 64D body tokens | teacher execution | target |
| future 40 × 14D hand targets | fixed/synergy/task hand policy | target |
| privileged task/object state | simulator | no; debug and quality gating only |
| raw reward, success, quality metrics | simulator/evaluator | no; filtering and analysis only |
| timing and command-age metadata | latency wrapper | no; analysis, optionally future conditioning |

The image must remain necessary for object- and scene-dependent decisions. Privileged object state may help the teacher but must not leak into the VLA input.

Start with offline behavior cloning. Add DAgger-style data aggregation only after closed-loop evaluation shows distribution shift; do not build it speculatively.

## VLA implementation in StarVLA

StarVLA should remain the training shell: dataset loading, distributed launch, trainer, W&B logging, checkpointing, and evaluation. The model and processor should be the official NVIDIA GR00T N1.7 implementation imported through Hugging Face/Isaac-GR00T interfaces.

Do not extend the current `QwenGR00T.py` path and label it N1.7. That file is a lightweight Qwen-VL plus flow head derived from the earlier N1.5 design; the current HumanoidBench configuration also assumes horizon 1 with 61D action and 177D state. N1.7 has a different model, processor, modality configuration, and checkpoint contract.

The intended boundary is:

```text
StarVLA data/training/evaluation
        imports
official AutoModel/processor + Gr00tN1d7 configuration
        loads
official base checkpoint + UNITREE_G1_SONIC post-training initialization
        predicts
40 × 78D SONIC-compatible action chunks
```

Preserve the official processor, normalization statistics, modality configuration, and checkpoint layout. Do not copy model source, rename state-dict keys, or create a private checkpoint conversion unless the official loading path proves impossible.

The G1 SONIC post-training checkpoint aligns the body/action representation; it is not a zero-shot solver for these five tasks. Task behavior still comes from our simulated teacher data.

References:

- [Isaac-GR00T repository](https://github.com/NVIDIA/Isaac-GR00T)
- [Official GR00T policy interface](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/policy/gr00t_policy.py)
- [SONIC model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- [SONIC quickstart](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)

## Latency semantics

The benchmark injects latency at the VLA observation/inference boundary, not inside the 50 Hz low-level controller.

```text
t0: capture RGB + proprioception
    VLA inference is pending
    SONIC continues executing the previous chunk at 50 Hz
t1: new 40-step chunk arrives
    discard the prefix made stale by t1 - t0
    begin the remaining suffix
    if no valid suffix remains, record an underflow/fallback event
```

This matches the public SONIC client behavior, which removes the elapsed prefix rather than replaying commands intended for the past. A 40-step chunk at 50 Hz covers about 0.8 seconds. The nominal VLA request rate is about 2.5 Hz, so requests overlap in time with low-level execution.

Measure:

- observation-to-chunk latency;
- command age when each token is executed;
- stale-prefix length;
- chunk underflow count and duration;
- SONIC decode/control latency separately from VLA latency;
- task return, success, and motion quality as functions of latency.

A fallback action must be explicit in the experiment definition. It must not be silently hidden by the adapter.

## Phased implementation and decision gates

### Gate 0 — contract smoke test

- Import the exact official G1 MuJoCo asset and SONIC checkpoint.
- Replay identical token/proprioception sequences through the official reference and repository integration.
- Compare decoded targets and short MuJoCo rollouts.
- Confirm the 64D body + 14D hand wire contract and state history.

Proceed only when the local integration matches the official contract closely enough for deterministic smoke trajectories.

### Gate 1 — one-task token RL proof

- Adapt Run first because the flat FastTD3 result already establishes task learnability.
- Train one FastTD3 actor in `Box(78)` with learned hands.
- Verify task learning, motion video, controller metrics, and end-to-end FPS.

Proceed if token RL learns useful locomotion and throughput is practical. If it cannot, determine whether the blocker is scene mapping, controller expressivity, or optimization before adding another abstraction.

### Gate 2 — four-task teacher suite

- Reuse the same robot, decoder, action space, and replay contract.
- Train separate task checkpoints for Balance, Run, Hurdle, and Basketball.
- Keep the same learned 14D hand suffix for every task.
- Select checkpoints using task success plus motion quality.

Escalate only the failing task. A task-specific higher layer is justified when direct 64D training is sufficiently trained and the failure is reproducible after contract checks.

### Gate 3 — teacher data and VLA fine-tuning

- Export quality-gated 78D rollouts with RGB, language, and proprioception.
- Fine-tune official N1.7 inside StarVLA using offline behavior cloning.
- Evaluate open-loop chunk prediction and closed-loop task success without latency.

Add data aggregation only if closed-loop states leave the teacher-data distribution.

### Gate 4 — latency study

- Inject controlled observation/inference delays.
- Use stale-prefix removal while SONIC keeps executing at 50 Hz.
- Compare task success and motion degradation across latency distributions.
- Separate VLA delay, controller compute, and environment throughput in reports.

## Routes not currently being implemented

| Route | Why not now | Reopen when |
|---|---|---|
| Direct VLA → raw 43D/61D joints | duplicates low-level control learning and reproduced unnatural flat-action incentives | SONIC token control is shown to be the limiting factor |
| Reward regularization on the old flat actor | can improve symptoms but does not align the teacher with the intended VLA/SONIC interface | useful as a diagnostic or fallback baseline |
| DeepMimic/reference tracking | needs task motion references and retargeting | a specific token-space task requires prescribed style |
| AMP/adversarial imitation | nonstationary style reward complicates off-policy replay and may fight task reward | a reference dataset exists and fixed tracking is insufficient |
| PULSE/large motion-prior stack | changes the action hierarchy and adds major infrastructure | motion priors become a project-wide dependency |
| WIMLE | historical prototype was orders of magnitude too slow and is no longer retained | a batched formulation has a compelling new result |
| Per-task planner/skill layer | prematurely creates several teacher interfaces | one task reproducibly fails direct 64D control |
| Retrain SONIC for H1 | unnecessary while native benchmark comparability is not required | official G1 cannot support a required task |

## Decision log

| Date | Decision | Rationale/evidence | Rejected alternative | Reopen condition |
|---|---|---|---|---|
| 2026-08-26 | Support 30 tasks and explicitly exclude both High Bar tasks | SONIC G1 has no validated equivalent for either High Bar grasp contract | keep blocked High Bar scenes in the registry or training matrix | a separate robot/task contract is approved |
| 2026-08-26 | Align the supported 30-task family before more long training | five-task work exposed shared qpos, observation, reset, camera, and asset contracts | continue tuning only approximate scenes | all supported task-space comparisons and runtime contracts pass |
| 2026-08-26 | Adopt `density="5"` for Truck's ten movable package geoms | issue #60 identifies the omitted density and the Package task provides the intended project convention | preserve MuJoCo's default-density Truck physics | evidence shows a different intended density or untouched physics is required |
| 2026-08-26 | Define H1/G1 parity in task space rather than qpos equality | the morphologies have different joint counts and kinematics, while task objects and contacts remain comparable | copy H1 or native G1 qpos by index | a downstream method explicitly requires joint-wise retargeted demonstrations |
| 2026-08-26 | Fix generic HumanoidBench bugs as separate commits in the project fork | keeps SONIC adaptation local while making Kitchen, Truck, Room, Spoon, Package, and High Bar reset behavior reusable | maintain duplicate patches in the asset generator | fixes are accepted upstream or the fork is retired |
| 2026-08-26 | Keep the post-approval 10M matrix at four tasks × two interfaces | task alignment is an environment contract pass, not evidence for a larger sweep | launch raw43 and sonic78 for every upstream task immediately | extended tasks pass short learning and controller-manifold probes |
| 2026-08-25 | Optimize for our latency study, not native HumanoidBench comparability | tasks and latency semantics matter more than original morphology scores | preserve the native H1/H1Hand robot at all costs | a future result must be submitted to the native benchmark |
| 2026-08-25 | Use the exact official SONIC-aligned G1 | avoids a 23-DoF torque versus 29-DoF position-target contract mismatch | retarget/retrain SONIC for native G1 or H1 now | official G1 cannot express a required task |
| 2026-08-25 | Use direct continuous 64D FastTD3 for every first teacher attempt | smallest shared body interface and matches deployment tokens | quantized/MultiDiscrete policy or immediate high-level planner | direct token control fails after contract and training checks |
| 2026-08-25 | Permit per-task teacher checkpoints but keep one action contract | tasks differ; sharing the wire format provides the useful reuse | assume one universal actor without evidence | multi-task sharing becomes an explicit experiment |
| 2026-08-25 | Keep hands outside the 64D body token | official VLA wire format has independent 7D + 7D hands | allow unconstrained 14D fingers on all tasks | Basketball needs more than the initial hand synergy |
| 2026-08-25 | Historical, superseded: use fixed relaxed hands for Balance/Run/Hurdle, fixed grasp for High Bar, and scripted Basketball grasp | this minimized the first Run-only proof but no longer preserves full task controllability | train a full hand policy from the start | reopened on 2026-08-26; the parity experiment now learns all 14 hand joints |
| 2026-08-25 | Store the full executed 78D action in teacher data | the student must learn the command actually sent to the controller | store only actor outputs or decoded 29D targets | the official VLA action contract changes |
| 2026-08-25 | Allow privileged state only in the teacher | efficient task learning without making vision redundant for the VLA | privileged object state as student input | a separate state-policy baseline is needed |
| 2026-08-25 | Reuse official N1.7 inside StarVLA instead of extending `QwenGR00T.py` | preserves pretrained model/processor/action contracts | copy or approximate N1.7 locally | official loading cannot be integrated after a concrete attempt |
| 2026-08-25 | Start with offline behavior cloning | no evidence yet that DAgger is required | prebuild online data aggregation | closed-loop distribution shift is measured |
| 2026-08-25 | Define latency at the VLA chunk boundary and remove stale prefixes | matches the official deployment logic and isolates high-level latency | freeze the entire controller during inference or replay stale commands | a different deployment architecture is selected |
| 2026-08-25 | Keep all work in MuJoCo | sufficient for the research question and current scope | real robot, VR, or sim-to-real infrastructure | project scope explicitly changes |
| 2026-08-26 | Implement the first proof only for Run with fixed relaxed hands and one CPU ONNX Runtime session per environment | the minimal path was sufficient to validate the 64D token, decoder, controller, FastTD3 checkpoint, and deterministic video pipeline | build a centralized decoder service before measuring local sessions | profiling shows decoder execution or duplication is the dominant bottleneck |
| 2026-08-26 | Do not launch the 10M SONIC run after the 100k smoke | the pipeline completed and showed much stronger early learning than flat FastTD3, but measured throughput was 2519 raw environment FPS at 32 environments and 865.7 end-to-end FPS including startup/compile | treat 2k FPS as adequate and start long training | a dedicated optimization pass reaches an accepted throughput and repeats the smoke |
| 2026-08-26 | Use `0.002 × 10` dynamics for the latency benchmark | matches the 500 Hz deployment control loop while preserving SONIC's 50 Hz policy interval | copy the training simulator's `0.005 × 4` dynamics | deterministic comparison shows the dynamics choice prevents useful SONIC behavior |
| 2026-08-26 | Batch SONIC decoding on CUDA in the Sample Factory inference worker for zero-latency training | one decoder session and GPU I/O binding raised steady-state throughput to 5.06k–5.38k FPS; the first Run-only experiment stored 64D, while the corrected path stores 78D | one CPU session per environment or a standalone decoder service | nonzero latency requires decoder execution inside the environment boundary |
| 2026-08-26 | Historical, partially superseded: host G1 assets in a pinned public Hugging Face dataset and keep the official decoder separately pinned | reproducible assets without adding about 50 MB of binaries to the source repository; the storage choice remains, but the first asset revision is invalid | vendor assets in Git or fork HumanoidBench only to store them | corrected local assets pass review and replace the dataset default revision |
| 2026-08-26 | Stop at a bounded 1M W&B experiment instead of 10M | it completed at 4,329.6 total FPS and evaluated at 139.16 ± 20.51 return with 922.8 mean episode length | infer final teacher quality from the 100k smoke or immediately spend 10M steps | video review confirms motion quality and the other tasks reproduce the result |
| 2026-08-26 | Invalidate the first G1 asset revision and rebuild the four supported scenes from official task assets/classes | the simplified G1 scenes changed task geometry and reward semantics, so H1/G1 results were not interpretable | tune training on the mismatched scenes | four paired H1/G1 reset videos and contract tests pass manual review |
| 2026-08-26 | Compare raw `43D` with SONIC `78D`, with both policies learning the same 14 hand joints | preserves official task controllability and avoids ignored/scripted actions while isolating the 29D-body versus 64D-token difference | fixed hands or Basketball-only scripted synergy | measured hand motion requires an explicit quality intervention |
| 2026-08-26 | Keep direct continuous 64D body-token control for all four first attempts | it is the smallest uniform SONIC interface; the previous failures occurred on mismatched scenes and cannot justify another layer | add a task planner or higher-level skill interface now | a task still fails after corrected parity scenes, sufficient training, and video review |
| 2026-08-26 | Remove the obsolete Run-only CLI and leave the corrected asset revision intentionally unresolved before review | prevents silently regenerating jobs against the invalid first asset bundle | retain a selectable legacy suite that can no longer be reproduced correctly | the paired videos are approved and the corrected Hugging Face revision is pinned |

## Open experimental questions

These are measurements, not new infrastructure commitments:

1. Does direct 64D body-token + 14D hand RL solve all four supported task rewards, and which tasks remain motion-quality failures?
2. Can Basketball learn the full 14D hand stream from the official sparse task signal?
3. Can the official task definitions remain unchanged for G1, as intended, or does one task expose a morphology-specific impossibility?
4. Does offline N1.7 fine-tuning remain stable with simulated RGB and privileged-teacher rollouts?
5. At what command age or stale-prefix length does each task fail?

Each failed gate should update the decision log with evidence before adding another algorithm or abstraction.
