# Latency command-RL — launch and smoke handoff

All four task entrypoints are implemented. Real staged smokes verified native
training/reload, issued-label data conversion, two-update VLA saves/reload and
baseline/new zero/profile closed-loop evaluation. These runs reused explicit
existing profiles; the combined fresh full-protocol profile-to-training launch
and formal long training have not been run. No robustness gain has been established.

Profiling contamination notice: a Balance EGL dataset export ran locally at
approximately 19:15:26–19:18:01 UTC on 2026-09-05 and may have affected the first
Parkour session. Balance VLA smokes also overlapped the
ongoing local Parkour profile on 2026-09-05, approximately 19:23:07.676–19:25:00
UTC and 19:30:31.686–19:31:00.743 UTC. The first trained two steps but failed
during evaluation; the second was stopped without producing a checkpoint.
Profile processes were not touched. Samples from those intervals are not clean latency measurements.
Do not accept that affected profile session without repeating its measurement.
Further GPU validation is restricted to the remote server while profiling runs.

## Fixed contract

Keep native PPO for MIKASA/Parkour/HAIC and FastTD3 for Balance. Train the
command policy, not the downstream decoder. Parkour uses H40×34 latent+yaw;
HAIC H40×256 latent; MIKASA H1×7; Balance 43D absolute requests repeated H40.
Parkour/HAIC decision/control rates are 10/50 Hz, MIKASA 20/20 Hz, Balance
50/50 Hz. Privileged teacher inputs remain; VLA inputs remain RGB/proprio.

Each admitted observation owns its issued label. Do not assemble H40 labels
from subsequent policy decisions or substitute actually-applied motor actions.
Actor auxiliary targets must decode the same issued chunk with the same aligned
actor inputs and frozen decoder. Balance conditioning stays at application time.

## Full-chain launch and recovery

Run from the repository root with the task's existing teacher, baseline VLA,
pretrained base, simulator assets and Hugging Face access already staged. Each
command below measures a profile before training; it does not silently reuse a
different model or profile. Publishing remains a separate operation.

```bash
.venv/bin/python -m scripts.tasks.mikasa_intercept_grab_fast.run_latency_pipeline \
  --config configs/training/mikasa_latency_pipeline.yaml
.venv/bin/python -m scripts.tasks.extreme_parkour_go2.run_latency_pipeline \
  --config configs/training/parkour_latency_pipeline.yaml
.venv/bin/python -m scripts.tasks.haic_pull_cart.run_latency_pipeline \
  --config configs/training/haic_latency_pipeline.yaml
.venv/bin/python -m scripts.tasks.humanoidbench_balance_simple.run_latency_pipeline \
  --config configs/training/humanoidbench_balance_latency_pipeline.yaml
```

Use the corresponding `*_pipeline_smoke.yaml` for short training. Profiling
still follows its complete protocol; append `--profile-path /absolute/path/profile.json`
to explicitly reuse a measurement. Use different output directories for
independent experiments. Stage logs and resolved latency/evaluation YAMLs are
written under each config's `output_dir`; stop at the first failed stage.

For another machine, copy the task YAML and explicitly replace its interpreter,
teacher, base-model, baseline-model and output paths. Keep decoder identity,
observation/action semantics and the latency configuration unchanged. Local
interpreters are listed below; the remote smoke host is accessed through
`ssh xinyuan@204.12.163.107`. The two local GPUs are reserved for the ongoing
formal profile: do not overlap training, export rendering or evaluation there.

Recovery uses native checkpoints, not a new workflow engine:

- MIKASA: pass `--resume <output>/teacher/training.pt` to the native training
  entry shown below; `teacher.pt` alone is initialization, not optimizer resume.
- Parkour: rerun `train_latency_command.sh` with the same teacher/config plus
  native `--resume --load_run <run-name> --checkpoint <model_N.pt>`;
  `--max_iterations` is the additional native learning budget. To continue the
  pipeline after native training, set `native_checkpoint` explicitly and pass
  `--skip-native-train`.
- HAIC: pass the new native `checkpoint_final.pt` as the training checkpoint;
  its `last_phase=latency_command` selects optimizer/progress/RNG restore.
  The pipeline accepts `--skip-native-train --native-checkpoint <path>`.
- Balance: retain the same native run name and set
  `native_restart_behavior: resume`; the selected FastTD3 checkpoint restores
  optimizer/progress state. For VLA periodic recovery use `--start-step` equal
  to the next save boundary, with the original maximum step and interval.

Preserve complete optimizer/scheduler files and the associated datasets when
resuming VLA training. A model-only checkpoint supports inference, not exact
training recovery. Smoke failures, `--keep-failed` data and a two-update VLA
are not accepted demonstrations or evidence of improved robustness.

## Automated checks (2026-09-06)

The final CPU-isolated command/checkpoint, four-pipeline, Balance validation,
executor, evaluation, latency, artifact and profile-protocol selection passed
**354 tests**, with one skip. Two cases were deselected: the Flappy executor
case needs the absent optional `flappy_bird_gymnasium` package, and the hardware
projection case passed separately without `CUDA_VISIBLE_DEVICES=''`. The
Flappy CPU/GPU renderer module was also excluded for that missing dependency.
No dependency was installed and no unrelated test was changed.

The final HAIC bootstrap/pipeline selection passed **14 tests**. Native HAIC
admission/finite-gradient tests passed **2 tests** remotely; TorchRL warned
that its unused C++ replay-buffer extension is unavailable. GR00T's sharded
dataset suite previously passed **23 tests**. These selections overlap and
their counts must not be summed as distinct tests.

Command transport checks two replicas over two episodes against scalar and
batched evaluators frame by frame, including admission, capacity drops, stale
H40 prefixes and independent resets. IID streams use the same global episode
ID derivation; temporal streams retain per-replica state.

## Functional smoke profiles

Three measured short profiles are available for pipeline smoke only:

- `outputs/latency_rl/parkour/smoke_profile_session0/profile.json`: 18,898 admitted
  samples from the current Parkour run's 100-episode session 0. Although it ended
  before the VLA overlap, the EGL export may have affected it. It is retained
  only for functional smoke, not accepted as clean measurement. The formal run
  is unchanged and this partial profile is not published.
- `outputs/latency_rl/balance/smoke_profile_session0/profile.json`: 4,973 admitted
  samples and 2,665 capacity drops from the 2026-08-31 100-episode session using
  the explicit 10k Balance model. This is not a new five-session measurement.
- `outputs/latency_rl/haic/co_located_measure_root/profile/profile.json`: 12
  admitted measurements from one complete 58-control-frame episode, mean
  301.340 ms, following the public evaluator's 100-sample warmup. The simulator
  and the existing HAIC 5k GR00T checkpoint shared remote GPU0. This deliberately
  small, co-located sample is only functional smoke evidence, not a formal
  latency estimate or topology reproduction. Raw records, resolved evaluation
  config and logs are in the same parent directory. The command exited 0 after
  slow native SDK shutdown; no shutdown workaround was added. Remote profile:
  `/home/xinyuan/haic_runs/co_located_measure_root/profile/profile.json`.

### MIKASA accepted smoke

MIKASA was revalidated after the admission, checkpoint and seed changes:
two native updates, a resumed third update, two complete 60-frame episodes,
and RLDS/LeRobot conversion completed. Artifacts and logs are under
`outputs/latency_rl/mikasa/revalidation_remote`. Its fresh two-step VLA run
completed optimization and held-out evaluation on the remote GPU and saved
the model plus full DeepSpeed optimizer state. Fresh checkpoint reload and
zero/IID closed-loop evaluations both completed 60 frames (success 0/1 each;
returns 0.13381242 and 0.27674512 respectively). Local summaries are
`revalidation_remote/eval_zero/queue_eval_latency_0.json` and
`revalidation_remote/eval_iid/queue_eval_latency_profile_sample.json`, relative
to `outputs/latency_rl/mikasa`. The fresh model lives on the remote server at
`/home/xinyuan/latency-sensitive-bench-mikasa-h1/third_party/starVLA/results/Checkpoints/mikasa_latency_revalidation_h1/checkpoints/steps_2_pytorch_model.pt`.
The existing canonical baseline VLA was also reloaded and evaluated against the
same profile: zero/IID returns 4.66181925/0.29341384, 60 frames and success 0/1
each. Its summaries are under `revalidation_remote/eval_baseline_zero` and
`revalidation_remote/eval_baseline_iid`. Four single-episode smoke runs do not
establish robustness or a statistically meaningful comparison.

The final control-trace collector was re-run remotely: two 60-frame episodes
with 28/29 admitted commands. Every non-noop applied command exactly matches
the issued action at `applied_source_raw_frame`, with a ready frame no later
than consumption; H1 chunk indices are 0, and real noops use source/index -1.
RGB, proprio, issued actions, admission, ready frames, done flags and applied
commands are bit-identical to the revalidation dataset already used for VLA
training. Only execution-source logging changed. Files are under
`revalidation_remote/issued_data_control_trace`, with
`revalidation_remote/collect_control_trace.log`; VLA training was not repeated
for this logging-only change.

### Parkour native and pipeline status (2026-09-06)

The accepted remote native latency-command smoke used **1,024 environments**
(not one): two finite updates with the standard five PPO epochs and four
minibatches, after clamping the summed action log-ratio before exponentiation.
Its log is `/home/xinyuan/parkour-latency-smoke/native_latency_fixed_iid_5x4.log`;
update 0 reports value loss `0.0130`, surrogate loss `-0.0595`, and entropy
`1931.1259`, while update 1 reports `0.0083`, `-0.0915`, and `1934.8921`.
The run wrote `model_0.pt` and `model_1.pt` under
`2026-09-05_20-35-07_latency_command_fixed_iid_5x4`. It used
`/home/xinyuan/parkour-latency-smoke/parkour_latency_debug.yaml`, whose
latency method is IID and whose profile is
`/home/xinyuan/parkour-latency-smoke/smoke_profile_session0/profile.json`.
The checked-in formal launcher and smoke config both use 1,024 environments
in `scripts/tasks/extreme_parkour_go2/train_latency_command.sh`; the smoke
limits native training to two iterations.

The earlier checkpoint-contract smoke was separate: the initial run had no
native resume input and wrote
`/home/xinyuan/parkour-latency-smoke/parkour/logs/rsl_rl/unitree_go2_parkour/2026-09-05_19-12-30_latency_command_smoke/model_0.pt`.
It and its successful resume used
`/home/xinyuan/parkour-latency-smoke/repo/configs/training/latency/parkour.yaml`
(fixed zero latency), with the resume loading that exact `model_0.pt` and
writing
`/home/xinyuan/parkour-latency-smoke/parkour/logs/rsl_rl/unitree_go2_parkour/2026-09-05_19-14-01_latency_command_resume_smoke/model_0.pt`.
The later successful `resume_next2` also used that fixed-zero config, loaded
`/home/xinyuan/parkour-latency-smoke/parkour/logs/rsl_rl/unitree_go2_parkour/2026-09-05_19-51-54_latency_command_updates2_smoke/model_1.pt`, and wrote
`/home/xinyuan/parkour-latency-smoke/parkour/logs/rsl_rl/unitree_go2_parkour/2026-09-05_20-19-38_latency_command_resume_next2/model_2.pt`; it was
diagnostic only. These fixed-zero resumes are not interchangeable with the
IID `model_1.pt` used by the collector. No temporal native resume was run.
The latency bundle included scheduler plus RNG state, and decoder freezing is
established by the runtime module contract.

The stabilized `model_1.pt` was used by the remote IID collector with the
original teacher as decoder. It committed two complete `--keep-failed` smoke
episodes (one train and one validation), 1,184 raw frames, and 979/205
control-trace frames. The output is
`/home/xinyuan/parkour-latency-smoke/collection_iid_2ep`; its metadata records
the IID `smoke_profile_session0/profile.json`, H40×34 issued chunks, and the
separate command/decoder checkpoints. The shards contain NPZ and MP4 files.
Conversion completed locally at
`outputs/latency_rl/parkour/collector_remote_iid_2ep`, producing train/val
LeRobot datasets with 98/21 admitted rows, `issued_command_chunk` modality
metadata, and MP4 files. The GR00T sharded loader extracts `(40, 34)` actions
and the aligned `actor_observation`/`termination` objectives.

The clean remote actor-objective bootstrap completed two optimizer updates
with the official GR00T launcher. It used one GPU, global batch size 1,
`max_steps=2`, the official base model
`/home/xinyuan/latency-sensitive-bench-mikasa-h1/outputs/official_gr00t/models/GR00T-N1.7-3B`,
the converted IID dataset
`/home/xinyuan/parkour-latency-smoke/repo/outputs/latency_rl/parkour/collector_remote_iid_2ep/lerobot`,
and the fixed teacher action actor
`/home/xinyuan/parkour-latency-smoke/checkpoints/teacher/model_14999.pt`.
The durable training log is
`/home/xinyuan/parkour-latency-smoke/actor_objective_vla_2updates_complete/train.log`;
the launcher exited 0 and reported `train_runtime=39.7032` and
`train_loss=3.8606886863708496`.

The accepted checkpoint is
`/home/xinyuan/parkour-latency-smoke/actor_objective_vla_2updates_complete/training/parkour_actor_objective_2updates_complete/checkpoint-2`.
It contains both model shards, `optimizer.pt` (14,592,708,947 bytes),
`scheduler.pt`, `rng_state.pth`, `trainer_state.json`, and processor metadata.
The prior
`/home/xinyuan/parkour-latency-smoke/actor_objective_vla_2updates` output is
retained as an invalid failed-save run because its optimizer save failed; it
must not be used as an accepted artifact. The earlier local-backbone wiring
smoke is likewise not an actor-objective result.

The new checkpoint was reloaded for one-episode closed-loop evaluation using
the original teacher as the runtime actor. Both commands exited 0. Configs,
logs, and JSON outputs are durable under
`/home/xinyuan/parkour-latency-smoke/actor_objective_vla_2updates_complete`:

| model | latency | steps | mean return | edge violation |
|---|---|---:|---:|---:|
| baseline | zero | 936 | 4.354336371296085 | 0.0128205 |
| baseline | IID profile sample | 1,500 | 1.2002695962437429 | 0.0026667 |
| new checkpoint-2 | zero | 1,500 | 0.7200697755761212 | 0.0 |
| new checkpoint-2 | IID profile sample | 1,500 | 0.535144853303791 | 0.0 |

The earlier baseline runs use the same seed 2 and profile. Their summaries are
retained locally under `outputs/latency_rl/parkour/real_chain_remote/evals/`
in `original_zero` and `original_iid`; the explicit model wrapper was
`/tmp/gr00t-parkour-baseline`. The new summaries are `eval_zero/queue_eval_latency_0.json`
and `eval_iid/queue_eval_latency_profile_sample.json` under the durable remote
run above, with local metadata copies in
`outputs/latency_rl/parkour/actor_objective_complete_metadata/`.
The new checkpoint expects the external `/tmp/Qwen/Qwen3-VL-cosmos` model/tokenizer
metadata on that evaluation host; it is not a portable publication artifact.

The evaluations used the existing
`/home/xinyuan/parkour-latency-smoke/smoke_profile_session0/profile.json`
for the IID profile and are functional wiring evidence only, not a quality or
robustness claim.

The checked-in pipeline has an explicit handoff at each boundary:
`NATIVE_CHECKPOINT=` identifies the native run's final checkpoint; collection
passes that command checkpoint while retaining the original teacher decoder;
video-backed NPZ shards are converted into distinct train/validation LeRobot
datasets; bootstrap VLA training consumes that dataset and keeps the original
teacher as its fixed action decoder; generated zero and temporal evaluation
configs point at the newly trained VLA model. The Parkour action modality uses
`issued_command_chunk` for the H40×34 carrier. The formal config remains
1,024 native environments/15,000 iterations, 250/25 collection episodes,
5,000 VLA updates, and 100-episode evaluation. The smoke config explicitly
limits native training to two iterations with 1,024 environments, collection to
1/1 episodes, VLA bootstrap to two updates, and evaluation to one 1,500-step
episode.

The remote acceptance run intentionally used the smoke scale and direct
single-GPU GR00T invocation; it is a functional two-update actor-objective
smoke, not a formal quality run. Its exact arguments were one GPU, global batch
1, `--action-objective parkour`, and the explicit base, dataset, and teacher
paths recorded above. The checked-in full pipeline remains available for a
future clean measured profile.

The argument differences are intentional handoff facts, not interchangeable
run names:

| stage | formal launcher | checked-in smoke override | remote chain actually run |
|---|---|---|---|
| native PPO | script default `--seed 1 --num_envs 1024 --max_iterations 15000` | `--num_envs 1024 --max_iterations 2` | `--seed 1 --num_envs 1024 --max_iterations 2`, IID debug config |
| collection | 250/25 episodes, 50 envs | 1/1 episodes, 1 env | 1/1 episodes, 1 env; IID `model_1.pt` command checkpoint |
| VLA | 5,000 updates, global batch 128, two GPUs, Parkour actor objective | two updates, global batch 1, one GPU, Parkour actor objective | two updates, global batch 1, one GPU, Parkour actor objective |
| evaluation | 100 episodes, one backend env, inference batch 8 on configured worker device | one episode, one env, inference batch 1 | one episode, one env, batch 1 with simulator and worker on `cuda:0` |

Available local interpreters are
`/pfs/users/lxy/conda/envs/extreme-parkour/bin/python` (Parkour 3.11),
`/pfs/users/lxy/conda/envs/haic/bin/python` (HAIC 3.11), and
`third_party/Isaac-GR00T/.venv/bin/python` (GR00T 3.12). The local Parkour
teacher, GR00T base model, and VLA source artifact exist in the shared tree.
The local GPU is reserved for the formal Parkour profile; the remote smoke
workspace contains the native and collector artifacts used above. VLA stages
still require an available GPU and explicit staging of the base/model inputs
to the selected execution host.

Once those artifacts and a clean measured profile are staged, the bounded
pipeline command is:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  scripts/tasks/extreme_parkour_go2/run_latency_pipeline.py \
  --config configs/training/parkour_latency_pipeline_smoke.yaml \
  --profile-path outputs/latency_rl/parkour/smoke_profile_session0/profile.json
```

The remote native and collection stages were run individually with equivalent
artifacts, while the accepted VLA/evaluation stages were run directly. A clean
formal profile and full-scale pipeline run remain separate future work.

### HAIC native and pipeline status (2026-09-06)

The accepted HAIC evidence was produced on `xinyuan@204.12.163.107`
(`computeinstance-u00mv1dams6a8gc51r`) in
`/home/xinyuan/latency-sensitive-bench-mikasa-h1`. Set these remote roots when
reproducing it:

```bash
B=/home/xinyuan/haic_runs/co_located_measure_root
P=/home/xinyuan/latency-sensitive-bench-mikasa-h1
```

The ready full-pipeline config is
`outputs/latency_rl/haic/haic_full_pipeline_remote.yaml` (synced to
`$P/outputs/latency_rl/haic/haic_full_pipeline_remote.yaml`). It uses the
original phase-1 teacher for native training, the canonical remote HAIC
latency config, the existing checkpoint-5000 baseline for profiling, a fresh
base-model VLA run, and 100-episode zero/profile evaluations. The full run was
prepared but not launched in this acceptance pass:

```bash
cd "$P"
.venv/bin/python scripts/tasks/haic_pull_cart/run_latency_pipeline.py \
  --config "$P/outputs/latency_rl/haic/haic_full_pipeline_remote.yaml"
```

The checked-in HAIC push code uses IsaacLab's
`Articulation.set_external_force_and_torque` API in `push`, `push_start`, and
`push_per_motion`; no push-deleting monkeypatch was used. The smoke evidence
below is functional integration evidence (small native/VLA runs and one-episode
evaluations), not a quality, robustness, or formal-scale result.

#### Native command and resume

The native acceptance run started from the phase-1 teacher and used the real
latency command config, horizon 40, control repeat 5, zero action delay, and
alpha 1.0. Its exact command and continuation were:

```bash
cd "$P"
export PYTHONPATH="$P:$P/third_party/HAIC:/home/xinyuan/IsaacLab/source/isaaclab"
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=offline
export HAIC_PYTHON=/home/xinyuan/miniconda3/envs/haic/bin/python
bash scripts/tasks/haic_pull_cart/train_latency_command.sh \
  "$P/outputs/haic/phase1_teacher/checkpoint_final.pt" \
  "$B/latency_real.yaml" \
  hydra.run.dir="$B/native_acceptance3" \
  task.num_envs=1 task.max_episode_length=100 \
  algo.train_every=8 algo.num_minibatches=1 total_frames=256 \
  save_interval=8 wandb.mode=offline

bash scripts/tasks/haic_pull_cart/train_latency_command.sh \
  "$B/native_acceptance3/wandb/latest-run/files/checkpoint_final.pt" \
  "$B/latency_real.yaml" \
  hydra.run.dir="$B/native_acceptance3_resume" \
  task.num_envs=1 task.max_episode_length=100 \
  algo.train_every=8 algo.num_minibatches=1 total_frames=128 \
  save_interval=8 wandb.mode=offline
```

The first run completed 32/32 iterations and wrote checkpoints at 8, 16, 24,
and `final` under `$B/native_acceptance3/wandb/latest-run/files/`; the
continuation completed 16/16 iterations and wrote its final checkpoint under
`$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt`.
The inspected actor metadata was finite in both checkpoints: the first ended
at `last_iter=32`, `num_updates=34`, and the continuation at
`last_iter=47`, `num_updates=50`, with 14 additional finite actor updates.
The native W&B logs contain 31 `actor/update_applied` entries; the recorded
actor metrics and tensors were finite.
Both outer commands exited naturally with return code 0.

#### Complete-episode collection and conversion

Using the resumed native checkpoint, the accepted train and validation
collection commands were:

```bash
cd "$P"
export PYTHONPATH="$P:$P/third_party/HAIC:/home/xinyuan/IsaacLab/source/isaaclab"
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=offline
export HAIC_PYTHON=/home/xinyuan/miniconda3/envs/haic/bin/python

"$HAIC_PYTHON" third_party/HAIC/scripts/haic_vla.py bootstrap-collect --headless \
  --device cuda:0 \
  --teacher-checkpoint "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --output-dir "$B/raw_acceptance3_train" --inference-device cuda:0 \
  --inference-batch-size 1 --num-envs 1 --seed 0 --dagger-round 0 \
  --vla-cadence 5 --split train --episode-budget 2 \
  --latency-config "$B/latency_real.yaml" --keep-failed

"$HAIC_PYTHON" third_party/HAIC/scripts/haic_vla.py bootstrap-collect --headless \
  --device cuda:0 \
  --teacher-checkpoint "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --output-dir "$B/raw_acceptance3_val" --inference-device cuda:0 \
  --inference-batch-size 1 --num-envs 1 --seed 1 --dagger-round 0 \
  --vla-cadence 5 --split val --episode-budget 1 \
  --latency-config "$B/latency_real.yaml" --keep-failed
```

The resulting logs are `$B/raw_acceptance3_train.log` and
`$B/raw_acceptance3_val.log`. Train accepted 2 episodes and 217 raw steps;
validation accepted 1 episode and 178 raw steps. The committed shard fields
were `issued_command [40, 256]`, `actor_input [40, 605]`,
`teacher_action [40, 23]`, and `termination [40]` in both splits. The merged
raw root is `$B/haic_vla_acceptance3/raw`; conversion was:

```bash
cd "$P"
"$P/third_party/Isaac-GR00T/.venv/bin/python" \
  scripts/official_gr00t/prepare_haic_dataset.py \
  "$B/haic_vla_acceptance3/raw" \
  "$B/haic_vla_acceptance3/bootstrap/lerobot" \
  --force --round-index 0 \
  --modality-config latency_bench/policy/official_gr00t_haic_issued_modality.py
```

The prepared train dataset has 2 episodes and 45 frames; the sibling
`$B/haic_vla_acceptance3/bootstrap/lerobot__val` has 1 episode and 36 frames.
The actor adapter is `$B/haic_vla_acceptance3/bootstrap/actor_adapt.pt`.

#### Fresh-base VLA smoke

The fresh official GR00T base is
`$P/outputs/official_gr00t/models/GR00T-N1.7-3B`. The two-update smoke used
the prepared dataset and resumed native checkpoint as both teacher and frozen
runtime checkpoint:

```bash
cd "$P"
export HAIC_PYTHON=/home/xinyuan/miniconda3/envs/haic/bin/python
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES
export GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1
export LD_LIBRARY_PATH=/home/xinyuan/.cache/latency-sensitive-bench/ffmpeg7/lib
bash scripts/tasks/haic_pull_cart/train_vla.sh \
  "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --base-model "$P/outputs/official_gr00t/models/GR00T-N1.7-3B" \
  --output-root "$B/haic_vla_acceptance3/vla_smoke2" \
  --bootstrap-dataset "$B/haic_vla_acceptance3/bootstrap/lerobot" \
  --actor-checkpoint "$B/haic_vla_acceptance3/bootstrap/actor_adapt.pt" \
  --policy-config "$P/configs/examples/haic/official_gr00t_eval.yaml" \
  --latency-profile-path "$B/profile/profile.json" \
  --latency-config "$B/latency_real.yaml" \
  --runtime-checkpoint "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --wandb-run-id haic_pull_cart_acceptance3_smoke2 \
  --start-round 0 --end-round 0 --inference-device cuda:0 \
  --inference-batch-size 1 --smoke
```

The resulting checkpoint is
`$B/haic_vla_acceptance3/vla_smoke2/bootstrap/training/haic_pull_cart_acceptance3_smoke2_sft/checkpoint-2`.
It includes both model shards, `optimizer.pt` (14,797,999,507 bytes), the
scheduler/RNG/trainer state, and processor/config metadata; the local mirror
keeps metadata only and excludes weights and optimizer state.
`vla_smoke2.log` records 2/2 updates, `train_runtime=81.4045`, and
`train_loss=1.4915385246`. The initial wrapper marker was return code 1 because
its generated eval config omitted `executor.inference_batch_size`; after that
configuration fix, the direct periodic eval at
`$B/haic_vla_acceptance3/vla_smoke2/bootstrap/eval/` exited naturally with
return code 0 and produced the expected summary.

#### Final zero/profile evaluations

The four final evaluations used the resumed native checkpoint at
`$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt`, the
existing measured profile `$B/profile/profile.json`, and one episode with
`eval_max_steps=1000`. The baseline model was
`/home/xinyuan/haic_runs/haic_pull_cart_vla_action_h40_flow_aux_20260904/training/haic_pull_cart_vla_dagger_action_h40_official_flow_aux_5k/checkpoint-5000`;
the new model was the fresh checkpoint-2 path above. Each direct command was
run with the checked-in HAIC implementation and returned naturally with code
0; its `$B/eval_*/final_exit` marker is `0`.

```bash
cd "$P"
export HAIC_PYTHON=/home/xinyuan/miniconda3/envs/haic/bin/python
export OMNI_KIT_ACCEPT_EULA=YES GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1
export PYTHONPATH="$P:$P/third_party/HAIC:$P/third_party/Isaac-GR00T"
export LD_LIBRARY_PATH=/home/xinyuan/.cache/latency-sensitive-bench/ffmpeg7/lib
for config in eval_baseline_zero eval_baseline_profile eval_new_zero eval_new_profile; do
  "$HAIC_PYTHON" "$P/third_party/HAIC/scripts/haic_vla.py" latency-eval \
    --headless --eval-config "$B/${config}.yaml"
done
```

| model | latency | length | return | motion progress | cart progress | pullcart score | success | output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline checkpoint-5000 | zero | 55 | 3.066511783748865 | 0.4732142985 | 0.0897285491 | 28.1471423805 | 0 | `$B/eval_baseline_zero/queue_eval_latency_0.json` |
| baseline checkpoint-5000 | measured profile | 48 | 2.3801772370934486 | 0.4607142806 | 0.0238152742 | 24.2264777422 | 0 | `$B/eval_baseline_profile/queue_eval_latency_profile_sample.json` |
| fresh checkpoint-2 | zero | 45 | 2.468793924897909 | 0.4553571343 | 0.0725262091 | 26.3941671699 | 0 | `$B/eval_new_zero/queue_eval_latency_0.json` |
| fresh checkpoint-2 | measured profile | 37 | 2.047113075852394 | 0.4410714209 | 0.0119812191 | 22.6526319981 | 0 | `$B/eval_new_profile/queue_eval_latency_profile_sample.json` |

The profile rows reuse `$B/profile/profile.json`; they do not claim a new
latency measurement or a quality/robustness result. Each `queue_eval_*.json`
records its model checkpoint identity.

The four result trees are mirrored locally under
`outputs/latency_rl/haic/acceptance3/evals/eval_{baseline,new}_{zero,profile}`;
the checkpoint-2 metadata (without model weights or optimizer state) is under
`outputs/latency_rl/haic/vla_acceptance3_metadata`.

## MIKASA single launch

```bash
.venv/bin/python -m scripts.tasks.mikasa_intercept_grab_fast.run_latency_pipeline \
  --config configs/training/mikasa_latency_pipeline_smoke.yaml
```

This invokes the registry's full profile protocol, then short PPO, complete-episode
collection, RLDS/LeRobot conversion, VLA training and new-checkpoint zero/profile
evaluation. `profile_arguments: [--hf-artifacts]` selects the pinned canonical
profile model explicitly. Profiling is not shortened by the training smoke config.
To reuse an existing measurement explicitly, add `--profile-path <profile.json>`.
The generated training/evaluation configs and individual stage logs are saved in
`output_dir`. Evaluations run from starVLA's native working directory so saved
relative pretrained-model paths resolve correctly. The orchestration wiring test passes; the newly combined launcher
still awaits a clean GPU slot for a complete run.

## MIKASA reproduction

Run from the repository root. The checked-in smoke config references the local
existing profile explicitly, not a freshly measured profile. Profile path:
`results/profiling/profiles/qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/instance_e906b84a82f9e086/profile.json`.

```bash
export PYTHONPATH="$PWD:$PWD/third_party/MIKASA-Robo${PYTHONPATH:+:$PYTHONPATH}"
export VLA_ACCELERATE=/pfs/users/lxy/conda/envs/starvla_rl_games_openvla/bin/accelerate
export CUDA_VISIBLE_DEVICES=1
third_party/MIKASA-Robo/.venv/bin/python \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --latency-config configs/training/mikasa_latency_smoke.yaml \
  --checkpoint outputs/latency_rl/mikasa/source/teacher.pt \
  --latency-output-dir outputs/latency_rl/mikasa/smoke --no-track

third_party/MIKASA-Robo/.venv/bin/python \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --latency-config configs/training/mikasa_latency_resume_smoke.yaml \
  --resume outputs/latency_rl/mikasa/smoke/training.pt \
  --latency-output-dir outputs/latency_rl/mikasa/resume_smoke --no-track

bash scripts/tasks/mikasa_intercept_grab_fast/collect_latency.sh \
  --latency-config configs/training/mikasa_latency_smoke.yaml \
  --checkpoint outputs/latency_rl/mikasa/smoke/teacher.pt \
  --path-to-save-data outputs/latency_rl/mikasa/issued_data \
  --num-train-data 2 --num-envs 2 --keep-failed
```

Outputs are `teacher.pt` and `training.pt`; initialization loads model weights,
whereas `--resume` restores optimizer/progress/random streams and starts fresh
simulator episodes. `training` config values override native task presets,
allowing genuinely small runs without silently running the preset 1024 envs.
Use fresh output directories for repeat collection/conversion.

Conversion uses the existing commands in `scripts/tasks/mikasa_intercept_grab_fast/train_vla.sh`,
with absolute NPZ/RLDS/LeRobot roots and splits `train[:1]` and `train[1:2]`.
The RLDS builder filters on `command_admitted`, keeping RGB/state/action at the
same original frame. Raw NPZ remains the full control trace.

VLA smoke command after conversion and `scripts/mikasa/prepare_starvla_dataset.py`:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_latency_vla.sh \
  outputs/latency_rl/mikasa/issued_data/data_lerobot \
  third_party/starVLA/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt \
  --run_id mikasa_latency_smoke_h1_ds --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.per_device_batch_size 1 --trainer.gradient_accumulation_steps 1 \
  --trainer.max_train_steps 2 --trainer.num_warmup_steps 0 \
  --trainer.eval_interval 2 --trainer.eval_num_batches 1 --trainer.save_interval 2
```

Use the native Accelerate/DeepSpeed launcher. Direct `python train_starvla.py`
does not establish its distributed environment and failed looking for MPI;
no MPI dependency was added. The completed two-update loop took about 14 seconds,
excluding model initialization and checkpoint saving. Observed GPU occupancy
reached 96,009 MiB during saving; this is a point measurement, not a measured peak.

Closed-loop smoke uses the new checkpoint, not the original profile model:

```bash
CUDA_VISIBLE_DEVICES=0,1 third_party/MIKASA-Robo/.venv/bin/python \
  scripts/mikasa/evaluate.py latency-eval \
  --eval-config configs/training/mikasa_latency_vla_smoke_eval.yaml
CUDA_VISIBLE_DEVICES=0,1 third_party/MIKASA-Robo/.venv/bin/python \
  scripts/mikasa/evaluate.py latency-eval \
  --eval-config configs/training/mikasa_latency_vla_smoke_profile_eval.yaml
```

Local interpreters:

- MIKASA: `third_party/MIKASA-Robo/.venv/bin/python`
- starVLA: `/pfs/users/lxy/conda/envs/starvla_rl_games_openvla/bin/python`
- Parkour: `/pfs/users/lxy/conda/envs/extreme-parkour/bin/python`
- HAIC: `/pfs/users/lxy/conda/envs/haic/bin/python`
- GR00T: `third_party/Isaac-GR00T/.venv/bin/python`

The conversion environments are the existing `uv run --project` projects under
MIKASA-Robo. Their declared dependencies were installed during smoke.
MIKASA native smoke logged roughly 20 aggregate steps/s including initialization;
the resumed single update logged 86 steps/s. These tiny-run rates are not a
reliable basis for formal training budgets. Peak GPU memory was not recorded.

### Balance corrected VLA acceptance (2026-09-06)

Balance's corrected two-update GR00T wiring run completed the underlying
single-GPU launcher with `global_step=2`, `train_loss=1.5560086`, and
`train_runtime=39.008s`. The temporary wrapper used for this remote run
intentionally deviated from the canonical launcher by forcing
`--nproc_per_node=1`, global batch 1, and zero dataloader workers; the direct
torchrun argv is recorded in `program/BALANCE.md`. The resulting checkpoint
contains model shards plus the full optimizer/scheduler/RNG/trainer state:

```text
/home/xinyuan/balance-latency-smoke/outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/vla_final/g1_balance_corrected_vla_final_2updates/checkpoint-2
```

`optimizer.pt` is 14,610,010,451 bytes. The wrapper's optional validation
stage originally stopped because the derived `lerobot__val` directory lacked
`meta/stats.json`. The canonical runner now generates that metadata before
loading validation data, and its post-training validation rerun completed with
body MAE `0.232301`, all-action MAE `0.177278`, and repeat standard deviation
`0.0`; this uses all 19 admitted rows rather than subtracting a future-action
horizon. The four explicit closed-loop evaluations below completed with two
episodes, 1,000 maximum steps, one simulated worker, and the session-0
temporal profile with slot 0:

| checkpoint | latency | mean return | std return | mean length |
|---|---|---:|---:|---:|
| original `checkpoint-10000` | zero | 24.349 | 0.009 | 33.5 |
| fresh `checkpoint-2` | zero | 7.517 | 0.014 | 20.5 |
| original `checkpoint-10000` | real profile | 34.426 | 2.309 | 48.0 |
| fresh `checkpoint-2` | real profile | 11.023 | 0.435 | 23.5 |

Persistent outputs are under
`/home/xinyuan/balance-latency-smoke/outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/evals/final_revalidated/`.
This is accepted as Balance checkpoint/reload and evaluation wiring evidence,
not as a robustness or quality claim; the one-visible-GPU host is not a
2x-topology reproduction. These post-fix result copies are also available at
`outputs/latency_rl/balance/final_revalidated/`; earlier runs remain in the
remote sibling `evals/final/` directory.

Native-to-VLA provenance is preserved under
`/home/xinyuan/balance-latency-smoke/`: the corrected native run first saved
`outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/native/checkpoints/g1_sonic_balance_simple_raw_ramp_fasttd3_real_profile_corrected_s1/checkpoint_p0/checkpoint_000000178_4128.pth`
after 4,096 environment frames, then resumed to
`/home/xinyuan/balance-latency-smoke/outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/native/checkpoints/g1_sonic_balance_simple_raw_ramp_fasttd3_real_profile_corrected_s1/checkpoint_p0/checkpoint_000000434_8224.pth`
after the 8,192-frame target. The logs are
`corrected_native_initial.log` and `corrected_native_resume.log`. Export from
the resumed checkpoint (`corrected_export.log`) produced 57 raw train rows and
57 raw validation rows; command-admitted filtering retained 19 LeRobot rows in
each split. The dataset is under
`outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/teacher_dataset/`.

## Next formal experiment

Stage the configured teacher/base/baseline artifacts on the execution host,
reserve the required GPUs, and use the full-chain commands above. In this
checkout, the HAIC teacher and baseline model are remote-only; use the explicit
remote configuration in the HAIC section or stage them at the configured local
paths. The other three tasks' configured teacher/model assets were found locally.

Run a clean full-protocol profile and the formal native/VLA budgets, then compare
baseline/new policies over the configured episode sets. The tiny failed-episode
smoke datasets establish wiring only; they are not accepted training demonstrations.
Record actual throughput, peak memory and disk growth before committing to
long-run resource estimates. The observed smoke runtimes are not such estimates.
The remote smoke host has only approximately 11 GB free after retaining the
accepted checkpoints; provision sufficient persistent output storage before
starting another full optimizer save or a formal run.

Formal training remains unrun. Use a clean full-protocol profile, explicit model
paths and the four task commands above; the short contaminated or partial
profiles here are functional smoke inputs only. For Balance, explicitly use
the existing 10k GR00T artifact, not the absent 15k path, and keep that choice
a separately named experiment.

Disk recovery removed only weights/optimizer files from obsolete smoke runs:
Balance `vla_new_chain_2updates` and `real_profile_corrected/vla_localmirror`,
Parkour `vla_local_backbone_2updates`, and HAIC `vla_real2_final` plus the
one-step `haic_vla_acceptance3/vla_smoke` attempt. Their logs/configs/metadata
remain remotely; metadata from the initial three runs was also copied to
each task's local `archived_remote_smokes` directory. These deleted weights
are not recoverable and historical paths to them must not be used for reload.

Byte-identical top-level final weight duplicates were also removed from the
accepted Balance `vla_final`, Parkour `actor_objective_vla_2updates_complete`
and HAIC `haic_vla_acceptance3/vla_smoke2` runs. Their complete `checkpoint-2`
directories remain intact; those duplicates
can be recreated by copying the checkpoint shards. Always reload the documented
checkpoint directory, not its parent. Baselines, teachers and datasets were retained.
