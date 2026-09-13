# HalfCheetah adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260912). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the verified HalfCheetah-v4 native continuous-action record.
The adaptation uses a task-owned wrapper around the native six-dimensional torque space and
trains its zero-latency small model with the vendored Sample Factory MuJoCo recipe. The completed
RGB+state path includes 100 teacher episodes, native LeRobot conversion, 5k-step QwenOFT training,
and L0/L2 latency-bench evaluation of the final checkpoint.

Fresh profiling-only run: [20260911T023128Z-p-only4](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/half_cheetah/profile_latency/provenance/profiling/status.md), env/obs FPS 10 / 10. Profiling and publication are complete; the subsequent full pipeline is recorded below.

- [September 11 realtime baseline](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/half_cheetah/profile_latency/provenance/baseline/status.md): completed 20 episodes at 10 FPS; return 357.85 ± 285.91.

- [September 12 pipeline continuation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/half_cheetah/profile_latency/provenance/pipeline/status.md): reuses completed P/B at 10 FPS; teacher training and publication complete; simulated E return 5,217.47 ± 108.45, all 10 episodes pass the original >3000 gate. R/C is complete: 100 episodes / 100,000 rows (90/10 split), all actual returns >3000 and exactly matched to the probe. All 123 raw/converted files are published and verified. VLA completed5000 steps/exit0; all22 inference files are published and verified. Matched3090 F is complete:288.53±248.11 versus baseline357.85±285.91 (−19.4%;8/20 seeds improve). Full traces verified; no average control-return improvement in this run.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/half_cheetah) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.
