# Worker Capacity Contract Audit (2026-09-23)

## Decision

The nine historical profile-small results for Flappy Bird, Demon Attack, and
Deadly Corridor are invalidated. Their training transport admitted every
observation, while formal evaluation used `simulated_worker_capacity: 1` and
dropped observations while the inference worker was busy. Historical HF
revisions remain unchanged for audit, but are not accepted results.

W&B authentication against `placeholder/starvla_tasks` succeeded. Checkpoint
selection and W&B credentials were excluded as primary causes. With capacity
disabled only for diagnosis, the original Flappy checkpoints recovered
20-episode deterministic means of 2185.485 (GR00T), 1565.540 (OFT), and 261.725
(PI). Under capacity one, GR00T and PI dropped about 49.6% and 56.3% of
observations, respectively.

## Unified Contract

- Every Sample Factory integration receives latency arguments from
  `training/common/latency_config.py`, including the required normalized
  `executor.simulated_worker_capacity` value.
- Scalar, CPU-batched, and fused CUDA transports select the lowest-numbered idle
  worker, sample latency only for admitted observations, and continue physics
  for dropped observations.
- Worker occupancy uses `worker_service_raw_frames`; action-ready latency is not
  substituted for service time. Episode reset clears occupancy and admission
  counters.
- Decision info and episode metrics expose submitted/dropped observations,
  worker/service metadata, and latest, rolling, and cumulative drop ratios.
- Automatic Sample Factory checkpoint resolution writes the actual absolute
  `best_*.pth` path back into runtime metadata used by episode JSON, summary,
  and provenance.

The implementation keeps the existing scalar, CPU-batched, and fused CUDA
paths. It does not introduce a universal environment wrapper or duplicate
`ActionLatencyBatch`.

## Verification

CPU regression tests passed: 122 passed, 7 GPU tests deselected. H100 extension
tests passed for the complete Flappy CUDA suite (19 tests), complete Demon Attack
CUDA core suite (13 tests), and representative Demon Attack/Asterix/Atlantis
shared-CUDA cases. The CuLE and Flappy CUDA extensions compiled with CUDA 12.9.

Temporary verification checkpoints, logs, and compiled caches were removed
after the contract tests.

## Corrected Runs

The corrected protocol reserves new `capacity1_seed42` experiment, checkpoint,
output, and W&B identifiers. No formal training is authorized in this change.

| Task | Profile | FPS | Training | Evaluation horizon |
|---|---|---:|---:|---:|
| Flappy Bird | temporal | 10/10 | 25M | 18,000 |
| Demon Attack | IID | 60/15 | 25M | 14,400 |
| Deadly Corridor | IID | 60/15 | 10M | 3,600 |

Each task/model pair uses training seed 42 and its own `best_*.pth`, followed by
100 deterministic episodes with capacity one and the established formal seed
schedule. No score threshold is invented. Acceptance requires matching profile,
seed, transport, drop ratio, and checkpoint provenance.

No corrected checkpoint, log, or evaluation is uploaded to Hugging Face before
explicit review approval.
