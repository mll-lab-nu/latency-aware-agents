# Atlantis / Asterix H8 execution — 2026-09-24

Approved recipe and resume contract:
[`ATLANTIS_ASTERIX_H8.md`](../ATLANTIS_ASTERIX_H8.md).

## Initial implementation and preflight

- Added the two games to history export, scratch VLA training and GPU evaluation;
  the new Asterix models use the existing native 9-class action layout.
- Realtime profiling now joins single-worker outputs to their original decision
  windows and supplies true H8 state; warmup uses reset history. Profiling presets
  explicitly keep simulation and rendering on CPU.
- Independent runtime roots are `/home/ubuntu/codex/atari-hist8-20260924` on the
  two existing H200 machines. Atlantis is assigned to `34.131.223.74`, Asterix to
  `34.131.91.104`; the first machine's occupied GPU 4 is excluded from this task.
- Both pinned CPU teacher checkpoints loaded with the expected H8 inputs.
  Smoke collection first compiled the GPU Atari backend before probe/replay.
- Initial combined tests: 166 passed; four failures are an unavailable `ffmpeg`
  executable and existing unrelated Flappy/generic sweep fixtures (13 versus
  30 Hz, a missing `wandb_tags` field, and a missing Flappy base YAML). Targeted
  Atlantis/Asterix, H8 conversion, realtime history and profiling tests passed.
  These failures were verified against the untouched baseline inputs. The final
  targeted selection passed 166 tests with six unrelated cases deselected.

## Verified data and formal launch

- Atlantis: 200 episodes, 324,000 train / 36,000 validation labels. Asterix:
  300 episodes, 337,557 train / 36,265 validation labels. Both full datasets
  passed a row-by-row check that binary H8 equals the previous eight teacher
  decisions: 360,000 Atlantis rows and 373,822 Asterix rows.
- All six two-step training→bundle→two-episode GPU evaluation preflights passed.
  A separate Atlantis CPU simulator/rendering realtime preflight completed two
  128-frame episodes, including actual H8 transport and warmup. Its H200 timing
  is smoke evidence only, not the requested RTX 3090 profile.
- The real preflight found two missing handoff details: inference bundles needed
  `rl_games.env_eval.image_size`, and student manifests needed frame_stack 1
  while raw teacher provenance remained frame_stack 4. Both are corrected;
  the exported images were already single current RGB.
- Four formal runs were confirmed through the W&B API in the required project:
  Atlantis OFT [`40be052b`](https://wandb.ai/saberrr-zju/starvla_tasks/runs/40be052b),
  GR00T [`60bc68a7`](https://wandb.ai/saberrr-zju/starvla_tasks/runs/60bc68a7),
  PI [`04d16bbd`](https://wandb.ai/saberrr-zju/starvla_tasks/runs/04d16bbd), and
  Asterix OFT [`aa718e0f`](https://wandb.ai/saberrr-zju/starvla_tasks/runs/aa718e0f).
  All were initially confirmed with actual optimizer steps and finite losses.
  Final training/evaluation/publication states are recorded below.

## Verified final training checkpoints

- Atlantis QwenOFT completed 5,000 steps successfully; W&B `40be052b` readback
  confirms global step 5,000. Its final checkpoint and inference bundle are the
  same file, SHA256 `f91d4079a94f779856220e4a954750624228a3da596b5ecdd70861e888a97877`.
  Both ranks' optimizer/RNG and scheduler state are retained. The queue has
  completed its independent 100-episode zero→zero evaluation with the agreed
  GPU/parallel-32/7200-frame protocol and published the model and results.
  Training evidence:
  [`atlantis_qwenoft_training.json`](../../outputs/tasks/atari_hist8_v1/verification/atlantis_qwenoft_training.json).

## Queue placement and continuation

- H200 #1 uses GPUs 0,1,2,3,5,6 for Atlantis only. On the user's subsequent
  explicit side-conversation request, pending Asterix GR00T/PI jobs were removed
  from this queue and returned to H200 #2 with the same run IDs and recipe.
- H200 #2 now runs Asterix GR00T on GPUs 2,3, PI on 4,5 and OFT on 6,7.
  The user requested termination of the former MIKASA Fixed-3/4 processes on
  2–5; their launchers and descendants were stopped and artifacts retained.
  All existing Atari learners were preserved. The user subsequently also
  requested stopping MIKASA Fixed-1 on 0,1; that launcher and its descendants
  are now stopped, with artifacts retained and GPUs 0,1 left unassigned.
  Each queue has a `before_user_gpu23_45.json` backup; the target queue also has
  `user_gpu23_45.json`. The earlier migration receipts remain under
  `outputs/tasks/atari_hist8_v1/relocation/{source,target}.json` locally.
- All three Asterix bundles/evaluations will now reside on H200 #2; publish
  Asterix there with `scripts/tasks/atari_hist8/publish_zero.py`. Atlantis
  publication remains on H200 #1. The previous cross-host Asterix bundle-copy
  instruction is superseded by this user-requested placement.
- Both runtime roots contain `code_snapshot.json` and `code_snapshots/<sha>.json`;
  preserve the archived records named in each training run's metadata. The
  task-generated dataset/weights and formal queue remain under
  `outputs/tasks/atari_hist8_v1` on each host.
- Local service `codex-atari-hist8-watch.service` checks every 20 minutes and
  wakes this task for failures, completed training/evaluation, or a stalled
  worker. State is `outputs/tasks/atari_hist8_v1/monitor/state.json`; unchanged
  state stays quiet. It also monitors the RTX profile queue and publication
  completions. It stops only at the full pipeline completion marker,
  `outputs/tasks/atari_hist8_v1/pipeline.complete.json`.
- SSH hit packet-size-related retransmissions during setup. The task-local
  `outputs/tasks/atari_hist8_v1/ssh_proxy.py` uses TCP MSS 1200; current control
  sockets are `/tmp/codex-atari-h8-mss1200-h200{1,2}.sock`. Direct cross-host
  BatchMode SSH works and was used for the dataset transfer.

## Monitor path correction

The first liveness alert was a false positive: the queue launched workers with
relative `--job` paths, while the monitor compared them with absolute paths.
All four worker processes remained alive, with fresh logs and advancing optimizer
steps. The monitor now matches the actual relative argument; the four false-event
deduplication entries were cleared so later real failures can still notify.
Live readback returned four `process_alive=true` records and no pending events.
Only the local monitoring service was restarted; no learner or training schedule
was restarted. Evidence: `outputs/tasks/atari_hist8_v1/monitor/relative_path_correction.json`.

## Per-model publication and RTX 3090 continuation

The user supplied `root@70.30.252.209:25590` and requested immediate publication
and profiling of each trained model. This supersedes the original stop before
profiling and the grouped three-model publication gate. Existing zero manifests
were augmented with per-model package, training publication and evaluation
publication jobs, preserving active learners and GPU placement. Each host now
has 17 zero-stage jobs; backups are `before_per_model_publication.json`.

Atlantis OFT, GR00T and PI have all completed 5,000 updates, their respective
100-episode zero→zero evaluations and both HF publication stages. Asterix OFT
has also completed both stages; Asterix GR00T/PI are still training at this
check. OFT mean raw scores are Atlantis 39,560.0 and Asterix 5,803.5. Exact
per-model identities and read-back records are in `publications/` on each
runtime. Atlantis OFT evaluation publication is
`6491a03f7da933756ee76f8804d1e0b9b1190fc6`, Asterix OFT is
`6c2cbeb0830a5ce6c8ad7a0c2226c72dce420aca` in `benchmark-models`.

The two-card profiling box uses `/workspace/atari-hist8` with an isolated
`/workspace/atari-runtime` Python 3.10 environment. Torch 2.11.0+cu128,
Transformers 4.57.0 and FlashAttention 2.8.3 match the selected source runtime;
all three model-family modules import. CPU ALE RGB rendering for both games
and CUDA on each card were checked. The complete Vast agent guide was read;
durable task processes use Supervisor. `atari-runtime` and `atari-prefetch`
finished successfully. Models and the pinned backbone were downloaded from HF.

Supervisor `atari-profiles` currently runs Atlantis OFT on physical GPU 0 and
Asterix OFT on physical GPU 1. Both have reached actual episode measurements.
Their resolved contracts show one visible GPU, CPU simulator/rendering, one
environment and one inference worker. Each requires 5 sessions × 10 episodes;
neither profile is declared complete yet. The controller automatically queues
the other four models as their HF publications become available. Raw evidence
is under `results/profiling/runs/`; profile publication/readback receipts go to
`outputs/tasks/atari_hist8_v1/profile_receipts/`.

The new HF artifact revision plumbing passed 28 targeted tests. Profiling
always loads a fixed model revision and records that identity. On completion,
each profile is uploaded to the profiles dataset and read back at a fixed
revision before profile-aware teacher training. The local monitor now includes
this third host and requests continuation through the remaining authorized
stages instead of waiting for hardware. RTX SSH control socket:
`/tmp/codex-atari-h8-rtx3090.sock` (same MSS-1200 proxy).

## H100 asynchronous teacher scheduling

The user assigned the two-H100 machine in `servers.md` to new small teachers.
Both cards on `ubuntu@34.6.91.253` were checked idle before deployment. The
isolated runtime is `/mnt/ssd/codex/atari-hist8-20260924`, using the existing
`/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla` environment. Only HF/W&B
credentials from the local `.env` were installed, with mode 600.

Asterix QwenOFT was the first completed 3090 profile: five sessions and ten
episodes per session, single GPU, CPU simulator/rendering. Its profile dataset
revision is `fbbba76b21fa705cf4e5d05ec1ecfb82a88b12d8`, path
`profiles/qwenoft/1x-rtx3090/asterix/instance_d907cdaf680040e0/profile.json`, SHA256
`4e2e3ed923c7500248ce4ce1508f124faec78282d90d6b99d070a9e29bdb6146`.
The full directory, including temporal distribution and burst attachments, is
downloaded by the teacher preparation job; its read-back receipt is retained in
`outputs/tasks/atari_hist8_v1/profile_receipts/asterix-qwenoft.json`.

The Asterix OFT profile-aware teacher is now actually training on H100 GPU 0.
Live config and W&B readback confirm CPU simulator, seed 0, frame stack 4,
history 8, 15,000,000-frame target, constant LR 0.000303, profile worker slot 0
and capacity 1. W&B run
[`asterix-qwenoft-hist8-profile-cpu15m-s0-v1`](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/asterix-qwenoft-hist8-profile-cpu15m-s0-v1)
was verified running with advancing frame counts. This is training progress;
final training, independent GPU evaluation and teacher publication are pending.

H100 `queue/teachers` contains all six prepare→train→eval→publish chains and
runs in tmux `atari-hist8-teachers`. The local bridge service forwards completed
profile receipts every 30 seconds; the queue starts each available profile
independently, with up to two single-GPU jobs. Published teacher catalogs are
forwarded to the H200 hosts for their profile VLA queues. The bridge recovered
from one SSH session interruption coinciding with a monitor restart; subsequent
readback confirmed forwarding and an active service.

Both H200 profile queues and launchers are installed in separate
`queue/atlantis_profile` / `queue/asterix_profile` directories. Their launchers
wait for the corresponding existing zero queue to finish, then handle paired
profile evaluation and profile VLA training/publication as receipts arrive.
Zero manifests and active learners were preserved. H100 and both profile queues
are included in the local execution monitor.

The H100 targeted suite passed 51 tests, including CPU training/GPU evaluation
configuration, native action dimensions, cross-host queue dependencies and a
causal history audit with rejected commands and delayed applied actions. Flat
training config comparison against the published zero teacher showed only the
intended latency/capacity and run/log identity changes. A live-checkpoint GPU
collection/conversion/evaluation preflight is running separately under
`profile_smoke/`; its results do not count toward formal scores.

During subsequent verification, direct SSH began timing out across all four
hosts, including MSS-1200 connections. Explicit HTTP CONNECT through the local
existing proxy (`ProxyCommand=nc -X connect -x 127.0.0.1:7890 %h %p`) restored
access. Bridge and monitor now use this route with separate control sockets;
both services were checked active and successful forwarding resumed. All six
queue/launcher snapshots returned without connection errors. The existing
monitor now clears each host's connection-alert deduplication entry after a
successful read, so a later independent outage can notify again. The queued
direct-SSH failure alert is historical; current proxy-based checks passed.
The existing
Atlantis zero controller has exited normally because all 17 jobs are complete.
H100 teacher training continued uninterrupted through this local connection
failure and was verified beyond 2.48M frames. Evidence:
[`h100_profile_teacher_running.json`](../../outputs/tasks/atari_hist8_v1/verification/h100_profile_teacher_running.json).

## Six zero checkpoints verified and first profile evaluation

All six zero VLA training jobs completed 5,000 optimizer steps and their HF
training publications passed readback. W&B readback confirms step 5,000 for
all six; both ranks' optimizer/RNG files and scheduler checkpoint are retained.
The final Asterix GR00T and PI run IDs are `42b8e7f3` and `ec20df07`.
Their training publication revisions are
`967a1727724c696322547a5c0bdeac6d2ba52f9b` and
`bf5a01c39d8ab092327a4b5c94594fec24559760`. Evidence:
[`atlantis_zero_final_training_states.json`](../../outputs/tasks/atari_hist8_v1/verification/atlantis_zero_final_training_states.json)
and [`asterix_zero_final_training_states.json`](../../outputs/tasks/atari_hist8_v1/verification/asterix_zero_final_training_states.json).
The two Asterix formal zero evaluations are still running; their scores and
evaluation publication are separate from this training completion.

The real profile preflight finished successfully using the Asterix OFT teacher's
671,744-frame checkpoint. GPU probe/replay used 256 slots and 20 episodes of
256 raw frames; all 1,280 rows passed the causal binary H8 audit, conversion
produced 136D history, and the separate two-episode GPU evaluation completed.
This is smoke evidence only and is excluded from all formal scores:
[`profile_teacher_smoke.json`](../../outputs/tasks/atari_hist8_v1/verification/profile_teacher_smoke.json).

H200 #2 had no pending zero GPU jobs: GR00T/PI evaluations were already on
GPUs 2/3 and 4/5, while 6/7 were verified idle. The Asterix profile queue was
started early on 6/7 to run OFT zero→profile independently; its previous
manifest is `before_early_profile.json`. The first launcher command referenced
the local staging location of `resume_zero.sh`; it exited before creating any
worker. The deployed root-level `./resume_zero.sh asterix_profile` corrected
that path, and the controller then completed profile preparation and launched
the 100-episode evaluation. No zero worker was stopped. GPUs 0/1 are currently
occupied by another task and remain excluded. Resume this existing profile
manifest; check resources before later expanding its GPU pool.

## Zero-stage closeout and three published profiles

All six zero→zero evaluations and their HF result publications are now complete.
The 600 episode records have the agreed actual seeds, GPU/parallel-32/capacity-1
protocol, 60/15 Hz and 7,200-frame limit, with zero invalid actions. W&B means
match all six local summaries. Mean raw returns:

| Game | QwenOFT | QwenGR00T | QwenPI v3 |
|---|---:|---:|---:|
| Atlantis | 39,560.0 | 39,662.0 | 39,615.0 |
| Asterix | 5,803.5 | 5,714.0 | 5,758.5 |

The final Asterix evaluation publication revisions are GR00T
`550ecfbbe35ee7d170d450ff051f297de2097ff8` and PI
`bacdab2bdeb46480955b4ec03d1f097162928c6d`. The zero stage is closed in
[`zero_stage.complete.json`](../../outputs/tasks/atari_hist8_v1/zero_stage.complete.json),
with [results](../../outputs/tasks/atari_hist8_v1/zero_stage_report.md) and
[per-evaluation verification](../../outputs/tasks/atari_hist8_v1/verified_evaluations.json).
The complete pipeline remains active; `pipeline.complete.json` does not exist.

Asterix OFT zero→profile also completed and published at
`d445334970ab7e110c3cacb5c8b790d3df20e1ec`: mean 374.5 over the same 100 seeds,
versus 5,803.5 zero→zero for identical weights. Its checkpoint/profile identities,
episode protocol and W&B readback passed. No profile-trained VLA comparison is
available yet.

Three 3090 profiles are published and verified. Each has exactly five sessions
of ten episodes, CPU simulator/rendering, H8, one visible inference GPU, one
environment and a 7,200-frame cap. Mean observation-to-action latency:

| Game / model | Mean ms | Profile dataset revision |
|---|---:|---|
| Asterix / OFT | 114.939 | `fbbba76b21fa705cf4e5d05ec1ecfb82a88b12d8` |
| Atlantis / GR00T | 212.994 | `a00f633aa3227e2be8648b4f1337d7ccd188d18f` |
| Atlantis / OFT | 116.319 | `7efc9af422864b9a1cb569aa81f1b2b1e216b89e` |

Full contracts and per-session count verification are in
[`completed_profiles.json`](../../outputs/tasks/atari_hist8_v1/verification/completed_profiles.json).
RTX now measures Asterix GR00T and Atlantis PI; Asterix PI remains pending.
The bridge automatically admitted both new Atlantis profiles. H100 is training
Asterix OFT and Atlantis GR00T, while Atlantis OFT waits for a free GPU.
Both new Atlantis zero→profile evaluations are running on H200 #1.

Both Atlantis evaluations subsequently completed and published during this
verification pass: OFT mean 18,588.0 and GR00T mean 8,167.0, each over 100
episodes. Their records, model/profile identities, GPU/32/7200 protocol and
W&B readback passed. This brings verified formal VLA evaluations to 9/18
(900 episodes); profile-trained VLA results remain pending.

H100 queue priority now runs completed teachers' GPU evaluation before another
long training job, allowing prompt publication and VLA collection. Both active
teacher workers were verified to retain their original PIDs after restarting
only the controller. With both zero queues complete and H200 #2 GPUs 2–7
verified free, the Asterix profile queue's pool was restored to 2–7; another
task owns 0/1. The source manifest backups and local verification receipts
record both scheduling changes. Training recipes and running learners were
preserved.

## Fourth profile and queued continuation

Asterix GR00T profiling completed at dataset revision
`15001405a402e30b7706476bc92aea45a61bb193`, profile SHA256
`5e104064d718ac51ce56aefac5318a155fd099f5c714ff8f0bc4985509639fd1`.
Its mean observation-to-action latency is 216.599 ms. The per-session records
confirm 5×10 episodes and the same CPU/H8/single-GPU/7200-frame contract.
`verification/completed_profiles.json` now contains four verified profiles.

The bridge delivered this receipt to H100 and H200 #2. Teacher preparation is
complete and training is pending a free H100; Asterix GR00T zero→profile is
already running on H200 #2 GPUs 2/3. Both PI profiles are measuring on the
RTX host. Existing teacher and evaluation workers remain healthy; no retry
or training recipe change was needed.

## User-authorized H200 teacher scheduling (21:20 CST)

All six 3090 profile jobs completed. Both PI zero→profile evaluations started.
The user requested using idle H200 GPUs instead of waiting for H100. After
checking live GPU and CPU occupancy, moved only the three unstarted teacher
train→eval→publish chains from H100 into their existing H200 game queues:
Atlantis PI on H200 #1 GPU 5, Asterix GR00T on H200 #2 GPU 4, and Asterix PI
on H200 #2 GPU 5. Preparation was already complete on each destination.
The source jobs were removed before destination admission to prevent duplicate
training; the original H100 learner PIDs 978099/997632 were preserved.
The CPU/ALE 15M recipe, profile identity and W&B project were retained.
Migration snapshots are in
[`teacher_h200_relocation/`](../../outputs/tasks/atari_hist8_v1/teacher_h200_relocation/).
H200 teacher publication writes local catalogs consumed by its existing VLA
queue. Live queues, rather than the old six-teacher H100 manifest, own resumes.

Asterix OFT teacher training, independent evaluation and publication completed.
Its subsequent data collection attempt failed at probe/replay validation:
seed 282 had probe length 1221 versus replay length 1001, using 256 slots for
both phases. No collection retry or data-contract change was made during this
scheduling operation; the existing failed attempt remains available for diagnosis.

## Probe/replay fix and RTX retirement (22:28 CST)

The Asterix OFT collection mismatch was reproduced with the same checkpoint,
reset seed 282, image, H8 and temporal samples. At decision 198, compacted
inference batches of 256 versus 50 produced slightly different GPU logits:
the winning action changed from 0 to 7, producing 1001 versus 1221 steps.
The collector now preserves the full slot batch for network inference in both
probe and replay. With the fix, both cases produce 1001 steps and bitwise
identical images, histories, logits and actions. All 64 dataset-export tests
passed. The fix was deployed to both H200 hosts without replacing unrelated
worktree changes. [Verification](../../outputs/tasks/atari_hist8_v1/diagnostics/fixed_batch_verification.json).

Archived the failed raw collection as `raw.failed_variable_batch_attempt1` and
explicitly retried `collect_profile__asterix-qwenoft` with the existing manifest.
Attempt 2 is running under `h8-fixed-batch-asterix`; the two active teacher
workers were preserved. The teacher itself was already published at revision
`da734589fb8669108a8021ea79f823665e1b11c8`, best checkpoint 13,893,632 steps,
independent GPU evaluation mean 3941.0 over 100 episodes.

All six profiles were independently downloaded from their pinned HF revisions
after the user shut down the RTX host. Each passes the CPU/H8/single-GPU contract
and five sessions of ten episodes. The two remaining PI profiles are Atlantis
`0de3f61f3635e7e706487c6cae62a4897bf636d5` (321.823 ms mean) and Asterix
`36f784f257aabf73721d5edd63abc7b3dee1a272` (319.332 ms mean).
[Full verification](../../outputs/tasks/atari_hist8_v1/verification/completed_profiles.json)
and [profile-stage completion](../../outputs/tasks/atari_hist8_v1/profiles.complete.json)
are local; the full pipeline is not complete.

The bridge had incorrectly continued polling the retired RTX host before
forwarding teacher catalogs, blocking later stages on SSH timeout. Bridge and
monitor now skip RTX when the verified profile-stage marker exists. Both
services were restarted; the bridge successfully forwarded all six receipts
and the Asterix OFT / Atlantis GR00T teacher catalogs at 22:25:44. H200 #1 then
started Atlantis GR00T profile collection (attempt 1, PID 3666418, GPU 2).
No RTX restart is needed. Future monitoring prompts reflect this retirement
and the current H100/H200 teacher ownership.

## All zero-model evaluation arms verified; three teachers published

Re-audited both H200 evaluation directories against actual episode seeds,
frame caps, action validity, checkpoint/profile identity, publication receipts
and live `saberrr-zju/starvla_tasks` summaries. All six zero→zero and all six
zero→profile evaluations pass: 12/18 formal groups, 1200/1800 episodes.
Newly closed zero→profile means are Atlantis PI 4369.0, Asterix GR00T 182.0,
and Asterix PI 143.5, each over the agreed 100 seeds. Full identities and results
are in [evaluation progress](../../outputs/tasks/atari_hist8_v1/evaluation_progress.md)
and [verification](../../outputs/tasks/atari_hist8_v1/verified_evaluations.json).
These results demonstrate substantial degradation of zero-trained VLA under
measured latency; profile-trained VLA comparisons remain pending.

H100's existing queue has completed all 15 owned stages. In addition to Asterix
OFT, Atlantis OFT and GR00T teachers completed 15,007,744 training steps and
independent 100-episode GPU evaluations, with W&B readback checked by publication:
OFT mean 38411.0, best 14,508,032 steps, HF revision
`34f68e5162100362de0710d7745edba677fc0957`; GR00T mean 29064.0, best 12,156,928,
revision `de60cabd0f3b7484773374f83be46a777657f74a`.
Their catalogs were forwarded and both H200 #1 collectors are live. H100 is
currently idle; the three migrated teachers continue on their assigned H200
GPUs under the existing manifests. No duplicate jobs were launched.
Asterix OFT collection attempt 2 has progressed past episode 219/350 in replay,
including the previous failing point, with no new probe/replay mismatch so far.
Formal history audit and conversion are still required before VLA training.

## First profile datasets accepted; Atlantis GR00T VLA started

Atlantis GR00T collection completed 250 full episodes, split 225/25, with
365,845 training labels and 39,091 validation labels. Its H8 audit passed all
404,936 rows and conversion retained 4 actions / 56D binary history. The
existing H200 #1 queue started `atlantis-qwengr00t-hist8-profile-s42-v1` on GPUs
2/6 (worker PID 1758953). W&B initialized in `saberrr-zju/starvla_tasks` as
run `0c35d237`; optimizer progress and final outcomes remain to be verified.

Asterix OFT collection attempt 2 completed successfully after the fixed-batch
repair: all 350 episodes replayed, split 315/35, 323,534 training labels and
38,047 validation labels. The H8 audit passed all 361,581 rows; conversion
retained native 9 actions / 136D binary history. The collector exited zero and
the VLA job is pending two free GPUs in its existing H200 #2 pool. No additional
retry or change to the training recipe was needed. Data evidence is retained in
[`data_verification/`](../../outputs/tasks/atari_hist8_v1/data_verification/).

Atlantis OFT collection continues. Atlantis PI teacher finished training and
is in its existing GPU evaluation stage. Both Asterix GR00T/PI teachers remain
in training. No queue has a current failed stage. The earlier monitor events
about attempt 1 and the retired RTX connection are stale queued notifications.

## Five teachers published; two profile VLAs updating

The Asterix GR00T `live=False` event was normal publication-process completion:
its queue now records success, its catalog exists after HF readback, and its
collector is running on H200 #2 GPU 4. Atlantis PI publication likewise passed.
Both completed 15,007,744 training steps and independent 100-episode evaluations:

| Teacher | Best checkpoint steps | GPU mean return | HF revision |
|---|---:|---:|---|
| Atlantis PI | 14,065,664 | 27780.0 | `5ed3e54aac1bd957c882b426218ca2e76db5b720` |
| Asterix GR00T | 13,975,552 | 1852.5 | `b2fe384570e74126afe1b9f75ca523747809bd36` |

W&B means match each publication provenance. Publication code verified every
published file at the returned revision before writing each catalog. Local
configs, per-episode records, provenance and catalogs are retained under
[`teacher_verification/`](../../outputs/tasks/atari_hist8_v1/teacher_verification/).
Five of six profile teachers are now published; Asterix PI remains in training.
Atlantis PI collection awaits a free GPU in its existing pool.

Atlantis OFT collection/conversion also passed: 200 episodes split 180/20,
322,419 train labels, 35,989 validation labels, all 358,408 H8 rows audited.
Its profile VLA is running on GPUs 3/5, alongside GR00T on 2/6. Actual optimizer
updates were verified (OFT step 409 and GR00T step 328, both finite loss); these
are running checks, not completed training or formal evaluation results.
Asterix OFT data remains accepted and its VLA awaits two free H200 #2 GPUs.
The existing queues have no current failed stages; no duplicate launch or
retry was performed for these completion events.

## All six profile teachers closed

Asterix PI completed 15,007,744 training steps; best checkpoint is 14,974,976
steps. Its independent GPU evaluation mean is 1141.0 over 100 episodes,
matching W&B, and HF publication/readback completed at revision
`e90f33e9f069a205f56b8d2e1e64e08d879cd821`. Its collector started automatically
on H200 #2 GPU 5.

All six teachers are now trained, independently evaluated and published.
Rechecked all six original CPU/ALE H8/frame-stack-4/15M/seed-0/constant-LR
configs, temporal slot 0/capacity 1, checkpoint/profile identities and the
600 raw evaluation records: seeds 1000000–1000099, GPU simulator, parallel 32,
60/15 Hz, 7200-frame cap and no invalid actions. All mean returns match the
publication's W&B readback. Their publication catalogs are written only after
every file is read back at the returned HF revision.
[Teacher-stage completion](../../outputs/tasks/atari_hist8_v1/teachers.complete.json)
retains all six model/profile revisions and evidence links. This closes only
the teacher stage; the profile-VLA stage remains active.

| Game | OFT teacher | GR00T teacher | PI teacher |
|---|---:|---:|---:|
| Atlantis | 38411.0 | 29064.0 | 27780.0 |
| Asterix | 3941.0 | 1852.5 | 1141.0 |

Asterix GR00T collection/conversion completed: 550 episodes split 495/55,
345,402 training labels and 36,335 validation labels, with H8 audit passing all
381,737 rows. Its nine-action distribution, termination records and pinned
teacher/profile provenance are retained under
[`data_verification/asterix-qwengr00t`](../../outputs/tasks/atari_hist8_v1/data_verification/asterix-qwengr00t/).
Four profile datasets are now accepted. Both Atlantis OFT/GR00T VLAs remain
running; both ready Asterix VLAs await two free GPUs. Atlantis PI collection
awaits a GPU in its existing pool, while Asterix PI collection is active.
Existing queue controllers and workers are alive with no failed stages.

## 2026-09-25: Atlantis OFT profile VLA trained and published

Atlantis OFT profile VLA completed all 5,000 optimizer updates. Final weights,
both rank optimizer/RNG states and scheduler state are present. Live W&B
readback for `saberrr-zju/starvla_tasks/7ea15469` confirms global step 5000 and
validation action loss 0.1460187435. This validation loss is not a game-score
evaluation. Packaging and training publication completed automatically at HF
revision `13b8c220b8a201357d2418fbc184de123079066d`, checkpoint SHA256
`3fc3190e4151c33d203f69da461d6c56042236200ea9f61369a08c97cb28b88b`.
Readback passed; its formal profile→profile evaluation is still pending two
available GPUs. [Training verification](../../outputs/tasks/atari_hist8_v1/verification/atlantis_qwenoft_profile_training.json).

The released GPU admitted Atlantis PI collection. Atlantis GR00T and Asterix
OFT profile VLAs continue training, with no failed stages in either queue.
Asterix PI data collection/conversion is complete: 800 episodes split 720/80,
337,461 training labels and 37,403 validation labels. All 374,864 rows passed
H8 history audit. Evidence is in
[`data_verification/asterix-qwenpi_v3`](../../outputs/tasks/atari_hist8_v1/data_verification/asterix-qwenpi_v3/).
Five of six profile datasets are now accepted; only Atlantis PI remains in
collection. The two ready Asterix GR00T/PI VLA jobs retain their existing
pending states and will be admitted when their two-GPU requirement is met.

## All six profile datasets accepted; first formal profile→profile eval running

Atlantis PI collection/conversion completed 250 episodes split 225/25,
362,699 training labels and 39,022 validation labels. Its H8 audit passed all
401,721 rows. Rechecked all six dataset reports against their teacher catalogs
and pinned profile revisions: every dataset has at least 320,000 training
labels, full-episode 90/10 split, no reward filtering, matching action counts,
7200 raw-frame cap, seed 42, and GPU probe/replay/inference batches of 256.
All six H8 audits passed. [Dataset-stage completion](../../outputs/tasks/atari_hist8_v1/profile_data.complete.json)
records the exact identities and per-dataset evidence; this is not pipeline
completion.

The existing queues automatically admitted Atlantis PI VLA on GPUs 3/5 and
Asterix GR00T VLA on GPUs 6/7. Alongside Atlantis GR00T and Asterix OFT, four
profile VLAs are running. Asterix PI remains pending a free pair.
Atlantis OFT training publication remains readback-verified at
`13b8c220b8a201357d2418fbc184de123079066d`; its formal profile→profile evaluation
started on H200 #1 GPUs 0/1. Its resolved protocol is 100 episodes, GPU/32,
capacity 1, 60/15 Hz, 7200 raw frames and the pinned OFT temporal profile;
22 episodes had completed at inspection. Results are not yet accepted or
published. No failed stage or duplicate start was observed.

## First profile-trained VLA result verified and published

Atlantis OFT profile→profile finished all 100 episodes and published at HF
revision `3bbc0bfd7369c818456525bfbdac9580a382c739`. Mean raw score is 39229.0.
Rechecked seeds 1000000–1000099, GPU/32/capacity-1/60–15Hz/7200 protocol,
invalid actions, actual step-5000 checkpoint identity, identical pinned profile
SHA to the zero→profile arm, live W&B metrics and publication receipt.
The corresponding zero→profile mean is 18588.0, versus zero→zero 39560.0.
Profile training therefore recovers near the zero-latency baseline for this
model, with a 2.11× mean versus zero-trained weights under the same profile.
This is a one-training-seed comparison over the shared 100 episode seeds.
Formal evaluation progress is 13/18 groups (1300/1800 episodes).
[Results and identities](../../outputs/tasks/atari_hist8_v1/verified_evaluations.json)
and [raw evaluation evidence](../../outputs/tasks/atari_hist8_v1/evaluation_verification/atlantis-qwenoft-profile-to-profile/).

Asterix OFT profile VLA finished 5000 optimizer updates, saved both ranks'
optimizer/RNG state and scheduler state, and published/read back at revision
`82b7e1a07db83f94b54e4a141ce09026e35354d3`, checkpoint SHA
`b96bf1398d3d767b13be1cc76eefdfe70fdab349231ef5a4636495c548fe09ac`.
Live W&B `ac01280d` confirms step 5000 and validation loss 0.1989223361.
Its formal evaluation is running on GPUs 4/5. Asterix PI training automatically
started on 2/3; GR00T remains on 6/7. Atlantis GR00T training has completed and
packaging is running, while Atlantis PI remains in training. No failed stages
were observed. [Asterix OFT training verification](../../outputs/tasks/atari_hist8_v1/verification/asterix_qwenoft_profile_training.json).

## Fifteen formal evaluation groups verified

Atlantis GR00T and Asterix OFT profile→profile evaluations each completed the
agreed 100 shared seeds. Independent audit passed raw scores, lengths, action
validity, GPU/32/capacity-1/60–15Hz/7200 protocol, step-5000 checkpoint identity,
matching profile SHA across the two profile-environment arms, live W&B metrics
and fixed-revision HF publication receipts.

| Model | zero→zero | zero→profile | profile→profile | Evaluation publication revision |
|---|---:|---:|---:|---|
| Atlantis GR00T | 39662.0 | 8167.0 | 28488.0 | `b3a6fd68fc00812311dc7feeaccc51b3093dc5e7` |
| Asterix OFT | 5803.5 | 374.5 | 3653.5 | `a4dd7774b90faf83cfec251a3a23249e3bbac080` |

Both improve over zero-trained weights in the same profile environment but
remain below their zero→zero baselines. These are single training-seed results.
Formal completion is now 15/18 groups, 1500/1800 episodes. Updated
[comparison](../../outputs/tasks/atari_hist8_v1/evaluation_progress.md),
[identities](../../outputs/tasks/atari_hist8_v1/verified_evaluations.json), and
[per-episode evidence](../../outputs/tasks/atari_hist8_v1/evaluation_verification/).

Atlantis GR00T final training identity was also independently checked: both
rank optimizer/RNG states and scheduler state exist; live W&B `0c35d237`
reports step 5000 and validation action loss 0.0225690883. Training publication
revision is `ce75d0a7829b0b096f9af3817758c21a467790ff`, weight SHA256
`408625af9e210710db6b3b20a6053967e225cbf3abedff0073a231dda6a05e22`.
[Training verification](../../outputs/tasks/atari_hist8_v1/verification/atlantis_qwengr00t_profile_training.json).

Three profile VLAs are fully trained/evaluated/published. Atlantis PI, Asterix
GR00T and Asterix PI remain in their original live training attempts. No failed
stage was found and no restart was necessary; their dependent evaluation and
publication jobs remain in the existing manifests.

## Atlantis PI profile training/publication verified

Atlantis PI completed 5000 optimizer updates with both ranks' optimizer/RNG
state and scheduler state saved. Live W&B `saberrr-zju/starvla_tasks/e58d89ce`
confirms global step 5000 and validation action loss 0.0421042815. HF training
publication/readback completed at `4e954396c055ab2a0ef7c6097dce946ffcb8141c`,
checkpoint SHA256
`8da09e91d1c6c3908cd1536d6c072e3b6f669b40c82fe9e98705f363691e10ba`.
[Training verification](../../outputs/tasks/atari_hist8_v1/verification/atlantis_qwenpi_v3_profile_training.json).
The existing queue admitted its formal 100-episode profile→profile evaluation
on H200 #1 GPUs 0/1; 56 episodes had completed at inspection. No final score is
yet accepted. All three Atlantis profile VLAs are now trained and published.

Asterix GR00T and PI retain their original live training attempts, with loss
and learning-rate updates continuing. No failed stages or duplicate workers
were found. Overall profile-VLA training/publication is 4/6; formal evaluation
remains 15/18 until Atlantis PI's result is audited and published.

## Atlantis complete; all six profile VLA trainings verified

Atlantis PI profile→profile completed 100 episodes with mean score 26447.0,
compared with zero→profile 4369.0 and zero→zero 39615.0. Raw records, actual
seeds, action validity, agreed GPU/32/capacity-1/60–15Hz/7200 protocol,
checkpoint/profile identity and live W&B readback passed. Evaluation publication
revision is `9502b332f984a867aa753d425d510775a389e0e3`.
Atlantis now has all nine formal evaluation groups (900 episodes) accepted and
published; its profile queue completed all 30 stages. The task's models,
teachers, profiles and datasets are indexed by
[`atlantis.complete.json`](../../outputs/tasks/atari_hist8_v1/atlantis.complete.json).
Combined formal progress is 16/18 groups, 1600/1800 episodes.

Both final Asterix trainings reached 5000 optimizer updates; independent W&B
readback confirms GR00T run `946c608a`, validation loss 0.0473545417, and PI
run `f9d438a5`, validation loss 0.0489547253. Both ranks' optimizer/RNG states
and scheduler files are present. PI training publication/readback is complete
at `04e2633f430ab8e0a111078ad27a896a51721f43`; GR00T training publication was
still running at inspection. Both final formal evaluations are running under
the existing queue, GR00T on 4/5 and PI on 2/3. No failed stage was found.

[`profile_training.complete.json`](../../outputs/tasks/atari_hist8_v1/profile_training.complete.json)
closes all six profile trainings only. The complete pipeline remains active
until the final two Asterix evaluations and their publication readbacks pass.

## Complete pipeline accepted and comparison published

The final Asterix profile→profile results passed the full episode, seed,
frame-cap, action-validity, checkpoint/profile-identity, W&B and HF-receipt
audit: GR00T mean 1440.0, PI mean 1015.0, each over 100 shared seeds.
GR00T evaluation publication is `6d11a6d4946f278d9b165d260496cd757b330c3d`;
PI evaluation publication is `b53f7bbd530c986912433448d7dc52952349bd99`.
GR00T training publication/readback is
`69ea38788064e1ab65fd24809ef235f37476d8be`.
All 18 groups (1800 VLA episodes) are verified. Both profile queues are fully
complete: Atlantis 30/30 stages, Asterix 33/33, with no unfinished or failed
jobs. All 1800 episode records, action audits, configs and provenance are
retained locally in `evaluation_evidence/` as well as the published model
artifacts. Teachers separately contribute 600 verified evaluation episodes.

The complete comparison and fixed-revision model inventory were published to
`benchmark-models/reports/atlantis-asterix-h8-v1` at revision
`ea2ad5a90c28de1297793956a93dd3a7ed605c4b`; every report file was downloaded
again and compared byte-for-byte. [Report receipt](../../outputs/tasks/atari_hist8_v1/final_report_publication.json).
[`pipeline.complete.json`](../../outputs/tasks/atari_hist8_v1/pipeline.complete.json)
now records completion across training, evaluation and publication separately.
All six profile-trained comparisons improve over zero-trained weights under
the corresponding measured profile; only Atlantis OFT closely approaches its
zero-latency baseline. These are single-training-seed comparisons, not an
estimate of training-seed variability.

Task-specific local watch and receipt forwarding services were stopped
at closeout; systemd confirms both inactive/dead. The retired RTX host was not contacted. Other machine workloads
and services are outside this closeout.

## Post-shutdown artifact check and paper update (2026-09-25)

After the user shut down the compute servers, independently queried HF fixed
revisions for all 20 referenced checkpoint objects: 12 VLA bundles, six
profile teachers and two original zero teachers. All object LFS SHA256 values
match the recorded checkpoint identities; all download endpoints returned
success. No compute server was contacted. This check verifies Hub object
identity and availability, not a second full download of every weight file.
All 1800 VLA episode records with configs/action audits/provenance and all
600 teacher episode records remain local. [Post-shutdown verification](../../outputs/tasks/atari_hist8_v1/paper_update_20260925/verification_summary.json).

Updated the Atlantis/Asterix section of `program/RESULTS.md` and the six
profile-trained cells in `paper/figures/teaser/data.csv`. Regenerated Table 1
and the main Figure 1 (PDF/300-DPI PNG, plus editable bar-chart SVG). Updated
provenance and the manuscript's affected counts: 37 complete comparisons,
36 improvements, 13 environments with a completed trained endpoint. The
remaining five endpoints retain their missing status. The 38-page manuscript
compiled successfully; both changed visuals were inspected and diff checks
passed in the root and nested paper repositories.
