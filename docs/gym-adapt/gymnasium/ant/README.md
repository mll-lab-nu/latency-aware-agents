# Ant adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260911). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the verified Ant-v4 native 8D continuous-action and 27D RGB+state adaptation. The report includes the EGL probe, 10M-frame L0 APPO run, best-checkpoint L0-L4 evaluation, 100 gated teacher episodes, native LeRobot conversion, 5k QwenOFT training, and final-checkpoint L0/L2 VLA evaluation.

Fresh profiling-only run: [20260911T023128Z-p-only4](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/profiling/status.md), env/obs FPS 10 / 10. Profiling is complete; the subsequent pipeline is tracked below.

- [September 11 realtime baseline](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/baseline/status.md): completed 20 episodes at 10 FPS; return 194.33 ± 259.05.

- [Ant pipeline continuation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/pipeline/status.md): S/E completed at 10/10 FPS; E return 2301.03 ± 37.60. E 0/10 and bounded exporter probe 0/12 passed the original >3000 gate; The user subsequently authorized a >2500 data gate; R/C completed 100,000 rows, all 100 actual episode returns pass >2500, and VLA training completed 5,000 steps with exit code 0 using the same teacher lineage. The final model bundle is uploaded and all 22 files verified; the 20-episode measured evaluation completed with exit 0: return 1,853.33 ± 508.69 versus baseline 194.33 ± 259.05, with all 20 paired seeds improving and 16/20 reaching 1,000 steps.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/ant) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.
