# Humanoid adaptation

Latest experiment: [September 2026 profile-latency results](index.html#humanoid-profile-20260913). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the completed Humanoid-v4 native 17D continuous-action and 376D RGB+state adaptation. The report covers the 10M-frame APPO teacher, 100 gated rollout episodes, 5k QwenOFT run, and the post-fix final-checkpoint L0 evaluation. With the saved 224×224 image contract restored, the VLA still terminates within 66 steps in all 20 episodes.

## September 13 profile-latency pipeline

[Final run ledger](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/humanoid/profile_latency/provenance/pipeline/status.md): **terminated after teacher quality assessment**, at 10 / 10 FPS. The 10M-frame teacher completed normally. Fixed-seed evaluation averaged 1,036.93 return and 196.9 steps; the bounded exporter probe averaged 1,275.39 return and 244.42 steps. All episodes ended early and reward was dominated by the alive bonus. No lower gate was selected; R/C/V/F were not started. Profiling and the complete small model are published and hash-verified; diagnostic evidence and sampling limitations are retained in the report. The monitor was deleted.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/humanoid) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.
