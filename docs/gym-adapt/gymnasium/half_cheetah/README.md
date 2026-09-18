# HalfCheetah — GR00T and OpenVLA

[Open the experiment report](index.html). Updated 2026-09-18 from verified checkpoint-selection records and fixed HF revisions.

## GR00T H1 / 5 FPS: stopped at the profile-teacher gate

The fresh APPO teacher completed its **10M-step target** (actual **10,002,432 frames**). Final and training-best each scored **0/20 above the strict return >3000 data gate**. Training-best was selected by mean return. **E10, probe, profile data collection, profile VLA and F were not run.** RTX3090 B20 completed with its interrupted-run recovery audited; the profile branch remains stopped.

| Stage | Episodes | Return mean ± population SD | State |
| --- | ---: | ---: | --- |
| L0 VLA, simulated zero latency | 20 | 5012.47 ± 742.29 | Completed |
| B, measured real-time | 20 | -262.47 ± 74.94 | Complete; recovery audited |
| Profile teacher Final | 20 | 1843.16 ± 29.76 | 0/20 >3000 |
| Profile teacher training-best | 20 | 1847.22 ± 27.63 | Selected; 0/20 >3000 |
| E10 / probe / profile data / profile VLA / F | — | — | Not run: teacher gate |

- B20: seeds 42–61, cap 1000, measured-only RTX3090 latency. Mean episode latency **109.17ms**; mean frame interval **200.000701ms**;0 observation drops, action drops or invalid actions.
- Recovery: container restart at20:44 UTC interrupted the sweep. Complete seeds 42–57 were preserved and core metrics reconstructed using the frozen implementation; only seeds58–61 were evaluated afterward, following a new warmup. Original/resumed traces and ID-remapping provenance are retained. Some aggregate metadata for the first16 (actual_env_fps, end-drain counters) were lost and remain unavailable, without estimation. P and completed episodes were not repeated.
- Teacher selection: deterministic seeds 104771–104790 for each checkpoint, cap 1000, simulated IID from the new RTX3090 profile. All 40,000 steps were audited for seeds, timing, contiguous indices and reward sums; no invalid/dropped actions. The threshold is for teacher-data eligibility, not VLA acceptance.
- Final checkpoint: step 19,536 / 10,002,432 frames. Selected training-best: step 18,736 / 9,592,832 frames; SHA `01e50eb4902c9c6cc99879bbd8bc34333921ffc02a7371f092f51b7e8795bd59`. Target budget, original APPO seed 2222 and strict gate were preserved.
- P: 5×10 sessions, seeds 42–51 in every session, cap 900, no FPS calibration. All 50 episodes reached 900 steps; 45,000 samples, 0 capacity drops. Mean **110.88ms**, P95 **117.14ms**, maximum **140.55ms**, all below the 200 ms frame period. These are separate from B20.
- L0 VLA: fresh 5000 updates, seed 42, global 64=2×micro32×acc1; single RGB224 + state17, native 6D actions, H1. Original APPO H1 data were reused (90,000/10,000 rows, 90/10 episodes), changing only FPS metadata/prompt. No L0 teacher retraining.
- Native physics are unchanged. QwenGR00T uses DiT-B, 4 inference timesteps, 8 diffusion repeats, train-only state minmax once, native action units, VLM/action LR1e-5/1e-4, warmup 100, held-out 200 batches/save every 500. Profile teacher remains original APPO 10M/seed 2222/8×8/LR0.00295, fresh initialization, IID seed 0 and no latency-info feature.

This run did not produce an eligible profile teacher under the agreed seed and budget. No profile VLA checkpoint or GR00T B/F improvement is claimed. A full HalfCheetah horizon measures neither successful locomotion nor data eligibility by itself.

[Teacher checkpoints and gate evidence](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/3b4294bafb55a9ad8a6327c0995e15478e3ac371/half_cheetah/profile_latency/small_model/GR00T) · [Profile / gate archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/223f2106c10b264b820963391ec546d8dce039a5/half_cheetah/profile_latency/GR00T/h1_5fps_20260917) · [L0 model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/3b4294bafb55a9ad8a6327c0995e15478e3ac371/half_cheetah/zero_latency/GR00T) · [L0 data](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/324664672c0d45666d6aded0b81af0c900ca1d28/half_cheetah/zero_latency/GR00T/h1_5fps_20260917) · [Per-seed teacher plot](gr00t-teacher-returns.svg) · [Compact results](results-summary.json) · [Publication index](gr00t-publication.json).

## Historical OpenVLA / 10 FPS

The earlier QwenOFT run completed its full pipeline: B20 **357.85±285.91**, F20 **288.53±248.11**, with 8/20 seeds improving. These within-run results and their original evidence remain in the report. Different FPS, recipes and measured latency distributions prevent a controlled cross-model ranking.

<details>
<summary>Original OpenVLA adaptation and archive notes</summary>

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

</details>


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). 8 wins, 0 ties, 12 losses; mean paired gain -69.32.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
