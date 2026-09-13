# Walker2D continuous-action adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260912). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the Walker2D adaptation record. The task
uses the generic `env.name: gymnasium` path and preserves the native
six-dimensional continuous torque space.

The zero-latency APPO run using the vendored Sample Factory MuJoCo recipe
completed 10,006,528 frames on H200. Its best checkpoint reached a training
statistic of 4,212.219 and was evaluated for ten episodes at each of latency 0
through 4. L0 achieved 3,743.507 mean return, while one or more frames of
latency caused the zero-latency policy to fail quickly. The six native torque
values remain unchanged through OpenVLA training and evaluation. With online
observations aligned to the model's saved image size, the 5k continuous-projector
QwenVLA checkpoint reaches 983.915 mean return and 469.5 mean steps at L0; 4 of
20 episodes reach the full 1,000-step horizon. The report uses this post-fix L0
result directly and does not retain the superseded evaluation comparison.

## September 12 full pipeline

Completed at 10 / 10 FPS. Matched 20-seed realtime return improves from **103.650 ± 4.904** to **857.437 ± 971.038** (8.27×; 17/20 seeds improve). Final mean length is 400.25 steps, with no full 1,000-step episodes. Simulated IID teacher evaluation is separate: 4102.871 ± 593.643 across 10 episodes.

All required data/model uploads are hash-verified. The 100-episode dataset has 96,730 rows; one tiny probe/replay return difference is explicitly documented. See the [report](index.html#profile-pipeline-20260912), [run ledger](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/walker2d/profile_latency/provenance/pipeline/status.md), and [paired results](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/walker2d/profile_latency/provenance/pipeline/paired_B_F.json). One training seed; no commit or push performed.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/walker2d) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.
