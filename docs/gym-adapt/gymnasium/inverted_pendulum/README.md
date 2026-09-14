# Inverted Pendulum agent-assisted adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-latency-new-10). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the complete Inverted Pendulum release
record. It documents the continuous-to-discrete force wrapper, frozen L1 timing
contract, verified APPO training, latency-bench evaluation, synchronized
evidence, and the stages intentionally left `NOT RUN`.

See the [Gymnasium examples index](../README.md) for other adapted tasks and the
[Gymnasium adaptation guide](../../../gymnasium_adapter.md) for the normative
workflow.

The [10 FPS profile-latency run](index.html#profile-latency-new-10) tracks the
measured profile and pipeline. Its
[ledger](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/inverted_pendulum/profile_latency/provenance/pipeline/status.md)
records the run independently.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/inverted_pendulum) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.
