# StarVLA resident DAgger migration — 2026-09-19

## Root cause and decision

The failed QwenPI profile run reached DAgger step 500, then the controller created a
second `ProcessInferencePool` on the training GPUs.  The training ranks still owned
the model, gradients, and optimizer state, so loading another policy caused CUDA OOM.
The controller exited and the remaining training ranks later timed out in a collective.

Intermediate DAgger boundaries now serve inference from the live training model.  All
ranks enter `eval()` and inference mode, serve controller requests through rank-owned
Unix sockets, and return to `train()` only after the controller writes the boundary's
continue file.  `ProcessInferencePool` remains only for collection before a DAgger
trainer exists and for evaluation after that trainer has exited.

## Verification

Three two-round smoke runs completed with `result.json`:

- QwenPI profile, two ranks;
- QwenGR00T profile, two ranks;
- QwenOFT profile, one rank with gradient checkpointing.

At the resident boundaries, GPU processes were the original training ranks; no second
model pool was loaded.  Controller clients disconnected between evaluation and
collection without releasing the boundary, both two-rank runs resumed training after
their continue files, and all three runs completed their final external evaluation.

The focused protocol, live-policy, controller, persistent-worker refresh, and trainer
boundary tests passed on the training host.  Local syntax and diff checks also passed.

## Formal migration

The original QwenPI zero and QwenOFT controllers were paused only after complete
bootstrap step 5000 model and trainer-state checkpoints existed.  They were then
replaced by new `--resume --publish` controllers in the same output directories, so
bootstrap training is not repeated and the original W&B run IDs are reused.

The QwenPI profile controller resumed from its complete DAgger step 500 checkpoint.
It entered the resident step-500 boundary before step 501 and began the missing
evaluation and round-2 collection with the two live training ranks on GPUs 6–7.
The continue file then released both ranks, which trained continuously through a
complete step 1000 model and trainer-state checkpoint.  The following step-1000
development evaluation again used only those two resident rank processes and
completed all 32 episodes without CUDA OOM or an NCCL timeout.

QwenPI zero and QwenOFT reused their completed bootstrap checkpoints, finished the
first external collection, and launched their new DAgger trainers directly.  The
generated QwenOFT DAgger configuration retained one process, micro-batch 64,
gradient accumulation 2, global batch 128, and gradient checkpointing.

The formal assignments remain:

- QwenOFT profile: GPU 3, gradient checkpointing, micro-batch 64, accumulation 2;
- QwenPI zero: GPUs 4–5, micro-batch 64 per rank;
- QwenPI profile: GPUs 6–7, micro-batch 64 per rank.

Matching zero/profile watchdogs require a valid source `result.json` and successful
publication before launching the corresponding QwenGR00T job with the same resident
boundary mechanism.  No formal result is recorded in `PROGRESS.md` until training,
evaluation, and publication are independently verified.
