# Atlantis / Asterix H8 pipeline

## Completed: 2026-09-25

The complete authorized pipeline is finished: six zero VLAs, six profile-aware
teachers, six profile VLAs, six verified RTX 3090 profiles and all 18 formal
VLA evaluation groups (1800 episodes). All weights, results and profiles were
published and read back; teacher evaluations add 600 independently scored
episodes. Both profile queues completed every stage. Do not restart them.

[Final comparison on HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/ea2ad5a90c28de1297793956a93dd3a7ed605c4b/reports/atlantis-asterix-h8-v1/README.md),
[local comparison](../outputs/tasks/atari_hist8_v1/evaluation_progress.md), and
[completion evidence](../outputs/tasks/atari_hist8_v1/pipeline.complete.json).
The RTX host remains retired. The task-specific watch and receipt bridge are
stopped after closeout. Commands below document the completed protocol and
historical execution; they are not instructions to relaunch it.

This task extends the accepted three-game H8 recipe to Atlantis and Asterix.
The authorized execution covers zero VLA training, publication, RTX 3090
profiling, profile-aware teachers, profile VLA training and all three evaluation
conditions. The user has supplied the profiling machine. Publish each model as
soon as training finishes and start its profile independently of other models
and performance evaluation; attach formal evaluation results when available.

## Models and protocols

The zero teachers are pinned by
[`zero_teachers.json`](../configs/experiments/atari_hist8/zero_teachers.json) to
`benchmark-models@dfaeecb84cd6b5dfcc3adc0935d0ae86d0b504d2`, using the published
CPU/ALE 15M, seed-0 H8 best checkpoints. Atlantis uses 4 action classes and
56 history values; Asterix uses 9 action classes and 136 history values.
VLA input is current RGB plus eight completed decision windows, encoded as
admission, admitted command and last applied action, with raw binary 0/1 state.
Action prediction horizon is one.

Each zero dataset is shared by QwenOFT, QwenGR00T and QwenPI v3. GPU probe and
replay use 256 slots, 60/15 environment/observation Hz and 7200 raw frames per
episode. Start with 200 episodes and add 50 until the training split contains
at least 320,000 labels. Split complete episodes 90/10, split seed 0, rollout
seed 42; retain all returns. The teacher keeps its four-frame network input,
and export reads its actual pre-decision `transport`. Keep identical GPU slot
counts and seed assignment during probe/replay, including resumed collection.
Interrupted GPU H8 replay must restart the full probe slot layout; it cannot
continue from only the remaining episode shards. Retain `probe_state.json`,
archive partial `episode_rows` and images, then replay all probe slots again.

VLA training starts from
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` with seed
42 and a random action head. Use two GPUs, per-device batch 64, global batch
128, accumulation 1, bf16, ZeRO-2, gradient checkpointing, 5000 optimizer
steps, 100 warmup steps, cosine minimum ratio 1/30, backbone LR 3e-6, action
head LR 1e-4 and other parameters 3e-5. Save and validate every 500 steps;
each rank uses 16 data workers and prefetch 4. Resume optimizer/scheduler/RNG
state without resetting the schedule.

VLA training, validation and formal evaluation belong to
[`saberrr-zju/starvla_tasks`](https://wandb.ai/saberrr-zju/starvla_tasks).
Small-policy runs belong to
[`saberrr-zju/latency-sensitive-bench`](https://wandb.ai/saberrr-zju/latency-sensitive-bench).
Read `WANDB_API_KEY` and `HF_TOKEN` from the local `.env`; remote credentials
are stored mode 600 and excluded from snapshots/publication.

## Execution

passing the runtime root, Python environments root, pinned backbone path and
explicit GPU pool. Run its manifest through
trains three models, packages and publishes their step-5000 bundles, evaluates
zero→zero and publishes evaluation results. Profiling has a separate two-GPU
controller. Failures require diagnosis
before explicit retry; verify process identity before relaunching a controller.
Current local SSH access uses the existing HTTP proxy:
`-o 'ProxyCommand=nc -X connect -x 127.0.0.1:7890 %h %p'`.
Monitor and receipt-bridge services keep separate control sockets.
After a queue relocation, resume the existing manifest through the shared
runner (or the runtime's `resume_zero.sh`); do not regenerate it with

Use independent `outputs/tasks/atari_hist8_v1` directories. Store smoke data
and models under `smoke/`; these do not establish formal training or performance
results. Operational notes and current execution evidence are in
[`logs/atlantis-asterix-h8-2026-09-24.md`](logs/atlantis-asterix-h8-2026-09-24.md).

## RTX 3090 stage and subsequent training

All six profiles have been published and independently read back from their
fixed HF revisions, including five sessions of ten episodes per profile.
`outputs/tasks/atari_hist8_v1/profiles.complete.json` closes this stage. The user
has shut down the RTX host; do not reconnect or request a restart. The bridge
and monitor skip it after this marker, while H100/H200 continue using the
published profiles. Verification is in `verification/completed_profiles.json`.

Use the task `profile.sh` wrappers or `scripts/profile.sh` to run the shared
profiler with the selected task and model. Completed results retain their pinned
profiles and HF read-back revisions in `profile_receipts/`.
Use CPU simulator and CPU rendering, one inference worker, batch 1, existing
warmup, 5 sessions × 10 episodes and 7200 raw frames at 60/15 Hz. The GPU is
reserved for VLA inference. Realtime H8 joins late outputs to their source
decision windows before the next admitted observation; this protocol uses a
single worker. Publish traces, profiles and machine identity to
`latency-sensitive-bench/profiles`, then pin the read-back revision.

For each game/model profile, train a fresh small teacher with the corresponding
zero teacher's CPU/ALE 15M seed-0 recipe, four-frame input and H8, changing only
the temporal latency/worker-capacity condition. Use worker slot 0, capacity 1,
and GPU network updates. Independently evaluate each best checkpoint with the
GPU simulator and publish its exact training config and profile provenance.
Collect six profile-conditioned datasets with the same data protocol and
per-reset-seed temporal streams, then train six independent profile VLAs with
the same Qwen recipe.

All six profile teachers have completed training, independent GPU evaluation
and HF publication/readback. `outputs/tasks/atari_hist8_v1/teachers.complete.json`
records their fixed model/profile identities and verification. Continue the
existing profile-VLA queues; do not restart completed teacher jobs.

Small teachers use H100 and available H200 GPUs. The two-H100 host is `ubuntu@34.6.91.253`, in
`/mnt/ssd/codex/atari-hist8-20260924`, with
`/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla/bin/python`. The user explicitly
authorized H200 capacity as well: each verified profile starts independently.
The H100 `queue/teachers/manifest.json` retains Atlantis OFT/GR00T and completed
Asterix OFT. Atlantis PI's train→eval→publish jobs are in H200 #1's existing
`queue/atlantis_profile` (GPU 5); Asterix GR00T/PI are in H200 #2's existing
`queue/asterix_profile` (GPUs 4/5). Their prepare jobs were already complete on
those hosts. Each teacher still uses CPU/ALE simulation and one GPU for network
updates. Resume the existing manifests; do not regenerate a live graph.
Controllers after this transfer use tmux sessions `h8-relocated-teachers`,
`h8-relocated-atlantis_profile`, and `h8-relocated-asterix_profile`.
Completed teachers' short GPU evaluations take priority over starting the next
long training job, so their publication and VLA collection can proceed promptly.

Local service `codex-atari-hist8-bridge.service` forwards completed RTX receipts
from the local verified receipt directory every 30 seconds to H100 and H200.
It also forwards published teacher catalogs
from H100 to H200. H200-trained teachers publish their catalogs locally for
their same-host VLA queues. No partial profile or checkpoint is a readiness marker.
`profile_teacher.py prepare` downloads the pinned profile directory, including
its distribution/burst attachments. Keep the receipt fixed when resuming a run;
a replacement measurement requires a separate versioned run.

Each H200 has a separate `<game>_profile` queue. Both zero queues are complete;
resume the existing profile manifest with `bash ./resume_zero.sh <game>_profile`
in that host's runtime root. Atlantis's candidate GPU pool is 0/1/2/3/5/6;
Asterix's is 2–7, excluding occupied GPUs 0/1. The runner admits jobs only on
free GPUs. Check actual occupancy before each resume.
The profile queue performs
zero→profile evaluation and, as each teacher publication arrives,
collection→profile VLA training→publication→profile→profile evaluation.
Collection audits binary history, zero reset, causal H8 shifts and admitted
teacher commands across every exported row. Teacher publication includes best
checkpoint identity, original CPU config, independent GPU evaluation and the
pinned profile; the collector records deployment overrides separately.

All six profile datasets have passed collection, conversion and H8 audit;
`outputs/tasks/atari_hist8_v1/profile_data.complete.json` records their identities
and label counts. Resume the existing VLA queues without recollecting these data.

GPU teacher inference keeps the full 256-slot batch in both probe and replay,
including inactive slots. Compacting active observations changes numerical
results and can flip nearly tied actions, breaking exact replay. Asterix OFT
collection attempt 1 is archived; attempt 2 completed with the fixed batch,
350 episodes and a passed 361,581-row H8 audit. Controller
`h8-fixed-batch-asterix` continues the existing manifest.

For the two Atari tasks, `collect_vla_hist8.py --teacher-catalog <catalog.json>`
reads the publication identities explicitly. The catalog has `revision` for
the model repo, `profile_revision` for the profile repo, and
`teachers[task][model]` entries containing `path`, `checkpoint`, `sha256`,
`profile_path` and `profile_sha256`. `profile_path` is the repo-relative
`profiles/.../profile.json` path. Collection keeps the original CPU training
config and records the local profile relocation and rollout seed separately.

## Formal evaluation and publication

For each game/model, evaluate zero→zero, zero→profile and profile→profile:
18 runs × 100 episodes, actual seeds 1,000,000–1,000,099, GPU simulator,
parallel 32, capacity 1, 60/15 Hz, 7200 raw-frame cap and raw game scores.
Use the same pinned profile in both profile-environment arms. Save per-episode
returns, lengths, termination facts and action distributions; write evaluations
to the corresponding VLA training run's separate condition namespace.

Publish VLA bundles to `benchmark-models` at
`<zero-latency|latency-aware>/<game>/vla/starvla-<model>-h1/<run-id>`;
publish teachers under each game's `latency-aware/.../small-policy/` subtree.
Read back the published revision and match checkpoint/config/result identities.
For zero bundles, run `scripts/tasks/atari_hist8/publish_zero.py --task TASK
--model MODEL --stage training --output-root outputs/tasks/atari_hist8_v1` after
packaging, then `--stage evaluation` after the formal zero→zero evaluation.
The source queues already contain these per-model publication jobs.
Report training completion, evaluation completion, performance conclusions and
publication separately. Keep all three model families' measured outcomes.
