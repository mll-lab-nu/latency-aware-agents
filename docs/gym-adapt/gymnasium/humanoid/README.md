# Humanoid

[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)

| Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T · Scientific stop | 118.38 / 125.04 ms | 6,331.47 ± 407.20 | 6,225.77 ± 1,077.26 | 1,158.63 ± 646.19 | 2,883.10 ± 1,804.11 | — |
| OpenVLA · Scientific stop | 286.64 / 292.69 ms | 6,331.47 ± 407.20 | 272.06 ± 36.17 | 226.92 ± 56.90 | 1,036.93 ± 280.76 | — |
| Pi0.5 · Scientific stop | 182.02 / 198.61 ms | 6,331.47 ± 407.20 | 6,667.56 ± 46.18 | 891.67 ± 623.80 | — | — |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## GR00T

Scientific stop

### Protocol

VLA 5/5 FPS, H1, RGB224, native state/action 376 / 17; L0/B 20 episodes, seeds 42–61, cap 1000. B RTX 3090 measured-only, one worker. P 5×10 episodes. E10 simulated IID; seeds 104771–104780 repeat first ten selection seeds. Historical L0 teacher 66.6667/66.6667 FPS, n=20. P raw-step cap: 900.

### Training

Fresh VLA 5,000 updates, seed 42, global 64; DiT-B, 4 inference timesteps / 8 diffusion repeats. L0 micro 16 × accumulation 4 on one H200. Fresh APPO 10M; training-best selected at step 19288 / 9875456 frames.

### Data

L0 reused 100 source episodes. Profile strict >5000:16 qualifying specs /120 attempts, target 100; specs were not replayed, no profile dataset.

### Evaluation scope

- L0/P/B/E audited; stopped at 120-attempt collection ceiling; no profile VLA/F. Stop evidence published.
- Returns are mean ± population SD across episodes from one training run; no controlled architecture ranking or equal-latency causal claim.
- L0 teacher is the shared source teacher at its original clock, not reevaluated at the current VLA clock.
- Teacher return gates apply to demonstrations, not VLA acceptance. Native clipping is preserved; logged actions can be preclip commands.
- E10 repeats selection seeds; it is not independent validation.
- Full runtime traces were audited server-side; compact publication does not contain every raw trace.
- Selected teacher selection20:3036.04 ±1802.44, 4/20>5000; E10:2/10>5000. Collection-budget stop does not mean all candidates failed.

### Models, data and evidence

- [Pinned GR00T evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/a260732e0bce9771ca62a3c336bb446ecd0ea5df/humanoid/profile_latency/GR00T/h1_5fps_20260917)

- [teacher archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/humanoid/zero_latency/small_model)

- [L0 VLA checkpoint (profile branch stopped)](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/7aa5afb6f20d52509849e8d8156b5d8583ea288f/humanoid/zero_latency/GR00T)

## OpenVLA

Scientific stop

### Protocol

B/E 10/10 FPS; B 20 episodes, seeds 42–61, cap 1000, measured-only RTX 3090; E10 simulated IID, one worker. H1, RGB224, state/action 376 / 17. Historical L0 teacher 66.6667/66.6667 FPS, n=20; historical L0 VLA 20 episodes at 66.6667/66.6667 FPS.

### Training

QwenOFT / Qwen3-VL-4B, continuous state projector, native action regression, L1, 5,000 updates, seed 42. L0 batch 4 × accumulation 4=16; ZeRO-2 CPU optimizer offload. No profile VLA training.

### Data

L0 strict>5000 accepted 100/119; 90/10 episodes, 88,608/9,614 rows. Profile probe had no qualifying episode; no replay or bulk collection.

### Evaluation scope

- Historical L0/P/B/E complete; teacher quality gate stopped profile collection/VLA/F.
- Returns are mean ± population SD across episodes from one training run; no controlled architecture ranking or equal-latency causal claim.
- L0 teacher is the shared source teacher at its original clock, not reevaluated at the current VLA clock.
- Teacher return gates apply to demonstrations, not VLA acceptance. Native clipping is preserved; logged actions can be preclip commands.
- OpenVLA is this project’s QwenOFT implementation, not an architecture-matched comparison with GR00T/Pi0.5.
- Formal shared L0 teacher6331.47 supersedes the separate5571.3 threshold probe. No F or paired gain.
- Historical E10 is the simulated profile-teacher evaluation; overlap with selection seeds is not established in local evidence.

### Models, data and evidence

- [Pinned complete historical archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/humanoid)

- [Profile teacher E10](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/humanoid/profile_latency/small_model/evaluation/summary.json)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/humanoid/zero_latency/OpenVLA)

## Pi0.5

Scientific stop

### Protocol

5/5 FPS, H1, RGB224, native state/action 376 / 17; L0/B 20 episodes, seeds 42–61, cap 1000. B measured-only CUDA-synchronized latency, one worker, no added IID. P 5×10 episodes, 100 warmups/session, cap 900 and 60-second sampling cap. Historical L0 teacher 66.6667/66.6667 FPS, n=20.

### Training

Fresh QwenPI_v3, 5,000 updates, seed 42, global 128, GC off, ZeRO-2, 4 inference timesteps/2 diffusion repeats. L0 micro 32 × accumulation 2 × two GPUs after approved micro 64 OOM change. Profile APPO 10M, seed 2222; Final step 19536 /10002432 selected on tie.

### Data

L0 source 100 episodes, 90/10 split. Both profile candidates 0/20>5000; no E10, probe, replay, profile data or profile VLA.

### Evaluation scope

- L0/P/B audited; profile branch stopped at teacher selection gate; no E10/data/profile VLA/F. B locally audited, not uploaded.
- Returns are mean ± population SD across episodes from one training run; no controlled architecture ranking or equal-latency causal claim.
- L0 teacher is the shared source teacher at its original clock, not reevaluated at the current VLA clock.
- Teacher return gates apply to demonstrations, not VLA acceptance. Native clipping is preserved; logged actions can be preclip commands.
- Final and training-best selection20 both1538.29 ±686.65; 0/20>5000, max3329.05. Selection20 is not substituted for absent E10.
- P retained warmup drift warnings for sessions 0, 2, 3; 500 warmups excluded.
- B evidence is local/server-only in this update; publication is separate from audit.
- State provenance is verified through manifest/loader/saved reload; no per-step state vectors were logged.
- P uses a 60-second sampling cap per episode, excluding initialization/reset/warmup; this does not change B/F or teacher caps.

### Models, data and evidence

- [Current B full-step audit](assets/evidence/B-verification.json)

- [Pi0.5 stage summary](assets/evidence/pi05-pipeline-summary-20260922.md)

- [L0 pinned model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/eee96ac0fae3889d5108e6ae00701a6502fd5f80/humanoid/zero_latency/Pi05)

- [teacher archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/humanoid/zero_latency/small_model)

- [Published failed teacher evaluation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/3391ea3f8e962dba3d40063f60138b8d4991c3c2/humanoid/profile_latency/Pi05/h1_5fps_g128_20260921/small_model/evaluation)

- [Pinned P evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/8ccefb9dc676bc35e56c253319951b2c2a6934f6/humanoid/profile_latency/Pi05/h1_5fps_g128_20260921/profiling)
