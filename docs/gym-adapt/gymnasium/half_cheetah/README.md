# HalfCheetah

[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)

| Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T · Scientific stop | 110.88 / 117.14 ms | 5,414.26 ± 740.00 | 5,012.47 ± 742.29 | -262.47 ± 74.94 | — | — |
| OpenVLA · Completed | 71.82 / 73.09 ms | 5,414.26 ± 740.00 | 3,675.25 ± 1,032.71 | 357.85 ± 285.91 | 5,217.47 ± 108.45 | 288.53 ± 248.11 |
| Pi0.5 · Scientific stop | 145.71 / 151.53 ms | 5,414.26 ± 740.00 | 5,357.98 ± 736.99 | -347.44 ± 82.66 | — | — |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## GR00T

Scientific stop

### Protocol

H1; RGB224 + native continuous state/actions; env/obs 5/5 FPS; L0 simulated zero latency and B RTX 3090 measured-only CUDA-synchronized latency, one worker; L0/B n=20, seeds 42–61, cap 1000; F not run. P: 5×10 episodes, seeds 42–51 each session, cap 900, no FPS calibration. State 17/action 6. Historical shared L0 teacher: n=10, original 20 FPS, seed 2222.

### Training

QwenGR00T, 5000 updates, seed 42, global batch 64; DiT-B, 4 inference timesteps, 8 diffusion repeats, VLM/action LR 1e-5/1e-4, warmup 100, held-out 200 batches and saves every 500. L0 batch 2×micro 32×acc 1. Fresh APPO profile teacher: seed 2222, 10M target, actual 10002432 frames, final step 19536; selected training-best step 18736 / 9592832 frames. APPO 8×8 workers, 64×64 tanh, LR 0.00295, IID seed 0, no latency-info feature.

### Data

L0 reused 100 original APPO H1 episodes: 90/10 split, 90000/10000 rows, clock metadata 20→5 FPS; native physics unchanged. Strict profile teacher-data gate >3000; both candidates 0/20; no profile data.

### Evaluation scope

- Scientific stop: profile teacher gate failed; L0/P/B complete; E10/data/profile VLA/F not run.
- One training run; population SD describes episode variation. Different clocks, recipes and realized latency distributions do not establish controlled architecture rankings or equal-latency causal effects.
- Selected training-best selection20: 1847.22 ± 27.63; Final: 1843.16 ± 29.76, seeds 104771–104790. These are selection metrics, not E10.
- B recovered preserved seeds 42–57 and evaluated only 58–61 after fresh warmup. First16 actual_env_fps/end-drain metadata were lost; no estimates substituted.
- Historical L0 teacher result is shared source evidence, not a new 5 FPS evaluation.

### Models, data and evidence

- [Profile, B recovery and gate evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/223f2106c10b264b820963391ec546d8dce039a5/half_cheetah/profile_latency/GR00T/h1_5fps_20260917)

- [teacher archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/half_cheetah)

- [L0 evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/324664672c0d45666d6aded0b81af0c900ca1d28/half_cheetah/zero_latency/GR00T/h1_5fps_20260917)

- [L0 checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/3b4294bafb55a9ad8a6327c0995e15478e3ac371/half_cheetah/zero_latency/GR00T)

## OpenVLA

Completed

### Protocol

Historical P/B/E/F at env/obs 10/10 FPS, one worker; B/F measured RTX 3090, n=20, seeds 42–61, native cap 1000 (configured evaluation cap 3600). P: 5×10 episodes, 60 s per episode. E10: simulated IID, n=10. Historical L0 VLA n=20 at original clock. State 17/action 6. L0 teacher: n=10, original 20 FPS, seed 2222; L0 VLA original 20 FPS.

### Training

QwenOFT discretized-text state, L1 action loss, 5000 steps, batch 16×1, seed 42. Profile teacher: original APPO 10M recipe, seed 2222, actual 10006528 frames, training-best selected at final frame.

### Data

L0: 100 teacher episodes, 90/10 split, 90000/10000 rows. Profile: strict return>3000; 100 accepted/106 attempts, 100000 rows, 90/10 split; all123 data files and22 final inference files verified.

### Evaluation scope

- Completed historical pipeline; F mean return decreased relative to B.
- One training run; population SD describes episode variation. Different clocks, recipes and realized latency distributions do not establish controlled architecture rankings or equal-latency causal effects.
- E10 is the historical simulated teacher evaluation; selection-seed overlap is not established in local evidence (historical HTML describes independent E).
- B→F mean paired gain -69.32 (-19.37%); 8/20 seeds improve. Full horizon alone does not establish successful locomotion.
- Historical L0 teacher and VLA clocks differ from 10 FPS P/B/E/F; their episodes are not a matched B/F block.

### Models, data and evidence

- [Complete fixed-revision archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/half_cheetah)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/half_cheetah/zero_latency/OpenVLA)

- [Profile-trained VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/half_cheetah/profile_latency/OpenVLA)

## Pi0.5

Scientific stop

### Protocol

H1; RGB224 + native continuous state/actions; env/obs 5/5 FPS; L0 simulated zero latency and B RTX 3090 measured-only CUDA-synchronized latency, one worker; L0/B n=20, seeds 42–61, cap 1000; F not run. P: 5×10 episodes, seeds 42–51 each session, cap 900, no FPS calibration. State 17/action 6. Teacher selection n=20 per candidate, seeds 104771–104790. Historical L0 teacher n=10 at 20 FPS, seed 2222. P additionally uses a 60 s sampling wall-time cap excluding reset/init/warmup; 100 warmup samples per session, 500 total, excluded from canonical latency samples.

### Training

QwenPI_v3, 5000 updates, seed 42, global batch 128 = micro 64 × accumulation 1 × 2 GPUs; GC off, ZeRO-2; train-only state min-max applied once. Fresh profile APPO seed 2222, original 10M target, actual 10002432 frames / step 19536; selected Final.

### Data

L0: original 100 APPO H1 episodes, 90/10 split, 90000/10000 rows; clock/prompt metadata20→5 FPS. Strict profile gate>3000: 0/20 for both candidates; no profile data.

### Evaluation scope

- Scientific stop: profile teacher gate failed; L0/P complete, B 20 summary verified; E10/data/profile VLA/F not run.
- One training run; population SD describes episode variation. Different clocks, recipes and realized latency distributions do not establish controlled architecture rankings or equal-latency causal effects.
- B 20 is verified from completed status plus20 episode summaries, not a full-step trace audit.
- Selected Final selection20: 1889.53 ± 44.32; training-best tied. E10 was not run; no substitute selection metric is placed in the E10 column.
- Teacher completion is established by actual Final checkpoint frames and user report; transient-service exit code is not proven.
- L0 audit passed20000 steps/actions; state vectors are not logged per step. Native clipping is preserved. Historical teacher result remains at original 20 FPS.

### Models, data and evidence

- [P verification](assets/evidence/P-verification.json)

- [L0 audit](assets/evidence/l0-audit.json)

- [B summary-only verification](assets/evidence/B-summary-for-table.json)

- [Teacher scientific-stop audit](assets/evidence/teacher-verification.json)

- [Published selected teacher](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/cf5763197b16922b14121ad982c4128e92607e1d/half_cheetah/profile_latency/small_model/Pi05)

- [L0 checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/ae2efeb81414f262e1839874e4691aef62b0b879/half_cheetah/zero_latency/Pi05)
