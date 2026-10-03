# InvertedPendulum — Pi0.5, GR00T and OpenVLA

[Open the experiment report](index.html). Final report for the Pi0.5, GR00T and OpenVLA experiments.

## Measured real-time results

Both B (zero-trained) and F (profile-trained) use RTX3090 measured-only inference latency,20 episodes each, seeds 42–61 and a 1000-step cap. Values are mean ± population SD.

| Model / experiment | Env / obs FPS | B | F | Within-run mean gain |
|---|---: |---: |---: |---: |
| Pi0.5 / QwenPI_v3 H1 |5 / 5|20.15 ± 6.48|830.40 ± 339.20|+810.25|
| GR00T / QwenGR00T H1 |5 / 5|15.55 ± 4.79|1000.00 ± 0.00|+984.45|
| OpenVLA / QwenOFT |10 / 10|19.80 ± 6.21|586.85 ± 457.53|+567.05|

These are within-model comparisons from one training run. Different clocks, recipes and measured latency distributions prevent a controlled architecture ranking. The HTML report includes detailed protocols, adaptation, plots and environment images.

## Pi0.5 H1 /5 FPS pipeline — complete

Measured F20 is **830.40 ± 339.20**, compared with B20 **20.15 ± 6.48**: mean gain **810.25 (+4,021.09%)**,20 wins / 0 ties / 0 losses. Sixteen F episodes reached 1,000 steps; the other four returned 149–154. Seeds 42–61, cap 1,000, RTX 3090 measured-only latency; no additional VLA gate or extra training budget.

- Separate L0 simulation 20: 1,000 ± 0. Both fresh VLAs completed 2,000 updates; state4/action5/RGB224/H1/5FPS, continuous projector and train-only minmax once. No L0 continuation or Bridge weights.
- P:5 × 10 episodes, 990 samples, 500 warmup excluded; mean 141.774 ms / P95 148.172 ms.
- APPO teacher:seed 0, target 1M counted env steps / actual 1,007,616. Final20/best20 all 1,000; tie selects Final; E10 all 1,000, reuses first 10 selection seeds. Probe 12/12; 100 accepted / 100 total attempts ≤120, actual replay all 1,000 >900, 90/10 split and 90,000/10,000 rows.
- SFT:seed 42 / global 64 = micro 8 × accum 4 × 2 H100, 4 inference timesteps / 2 diffusion repeats, VLM/head/base LR 1e−5/1e−4/2e−5, warmup 100, held-out 200 batches / save every 500. Profile run saves model weights without optimizer state because of disk capacity; science recipe and budget unchanged.
- F uses prompt key 1 (one 200 ms tick condition label) with measured-only latency; B uses key 0. B/F mean episode latency 137.89891 / 139.39327 ms; zero invalid/dropped actions and observations. F has 16,608 audited steps; within-episode interval 200.00076 ms. Episode-wall-time FPS includes startup overhead, separately recorded.

[Paired chart](pi05-paired-returns.svg) · [Exact results](pi05-results.json) · [Publication index](pi05-publication.json) · [Zero model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/1eb76cfa0299bc778a66452bf248c84dbc24ddc6/inverted_pendulum/zero_latency/Pi05) · [Profile model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/1eb76cfa0299bc778a66452bf248c84dbc24ddc6/inverted_pendulum/profile_latency/Pi05) · [Teacher](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/1eb76cfa0299bc778a66452bf248c84dbc24ddc6/inverted_pendulum/profile_latency/small_model/Pi05) · [Data and evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/1c23cb1c6cde53f07917909dd04a100fbb1160f2/inverted_pendulum/profile_latency/Pi05/h1_5fps_20260918) · [Full F traces](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/e20d12cc7ae30558947f36ab38b886aa89b20c9d/inverted_pendulum/profile_latency/Pi05/h1_5fps_20260918/vla/evaluation/traces).

Model files, datasets and full evaluation records are linked in the HTML report.

## GR00T H1 /5 FPS pipeline — complete

- Contract: native InvertedPendulum-v 4 physics unchanged; discrete forces[-3,-1.5,0,1.5,3], state 4, single RGB224, H1, no action chunks; 5/5 FPS, hold/FIFO and one inference worker.
- L0: reused original APPO H1 data,90/10 episodes and 90,000/10,000 rows. Only prompt/FPS metadata changed 25→5; no L0 teacher retraining. VLA 2000 updates; separate zero-latency L0 simulation (20 episodes) scored 1000 ± 0.
- P: 5×10 episodes,900-step cap with natural termination; 800 canonical samples. Mean 108.09 ms, P95 114.50 ms, max 134.29 ms.
- Teacher: fresh APPO seed 0, 1M requested frames,1,007,616 actual at the batch boundary. Final and training-best each 20 scored 1000 ± 0; tie selectsFinal step 1968. E10 scored 1000 ± 0. Selection uses 104771–104790; E10 repeats its first 10 seeds.
- Data: 12/12 bounded probe; 100 accepted/100 attempts, 0 rejected,90/10 split,90,000/10,000 rows. Every actual replay return 1000, strict gate>900. Train-only minmax state normalization is applied once.
- Profile VLA: fresh 2000 updates, seed 42, global 64=micro 16×accum 4. Qwen3-VL-4B-Instruct +DiT-B,4 inference timesteps,8 diffusion repeats, warmup 100, VLM/action LR1e-5/1e-4,200 held-out batches every 500 updates. Saved reload and real held-out forward[1,1,5] finite passed.
- F20: all 20 reached 1000 steps; 20 wins/0 ties/0 losses against B. B/F mean episode latency 106.07135/106.65249 ms, observation drops 0/311 versus 0/20,000, zero invalid or dropped actions. Trace step intervals average 200.04377/200.00061 ms. No new teacher threshold is used as a VLA acceptance rule.

Prompt key 1 means one raw tick(200 ms at 5 FPS); this is a condition label, not an added 200 ms delay. F uses measured-only latency. L0 source data and measured B/F evaluations retain their stated FPS protocols.

[Zero model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/6358f39350b2ce864479637c4d9cbbf4137343a0/inverted_pendulum/zero_latency/GR00T) · [Profile model / final result](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/df76ab5e19a7b90d49c2a27a8f4133701d6d559f/inverted_pendulum/profile_latency/GR00T) · [APPO teacher](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/964ee0be2ba7348652fff3a5addbf9119e0316c6/inverted_pendulum/profile_latency/small_model/GR00T) · [L0 data](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/a3702d961080ffad9d34f5797d0ccd4cde64f4fb/inverted_pendulum/zero_latency/GR00T/h1_5fps_20260917) · [Profile data and evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917) · [Paired comparison](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917/vla/evaluation/comparison.json).

Model files and complete evaluation evidence are linked in the HTML report.

## Evidence scope

[Compact results](results-summary.json) · [Publication index](gr00t-publication.json) · [Per-seed chart](gr00t-paired-returns.svg). Only display media and compact indices are local. Public artifacts include model bundles, raw demonstrations, QA, profiles, episode metrics and summaries. The audited B/F step, action, latency and observation-attempt records, including the 20,000-step F evaluation and both warmups, are now published: 18 trace files (89,646,252 bytes) plus four manifest/checksum files. All 22 files passed pinned size/hash verification; bulk traces remain off the local workspace.

[Baseline traces](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917/baseline/evaluation/traces) · [Final traces](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917/vla/evaluation/traces).

## Historical references

[OpenVLA 10 FPS detailed report](index.html#openvla) · [Experimental evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/inverted_pendulum). The 25 FPS L0/fixed-latency evaluations are separate from the 10 FPS measured B/F evaluations.



See the [Gymnasium report index](../README.md) and [adaptation guide](../../../gymnasium_adapter.md).


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and sources](openvla-paired-results.json). 19 wins, 1 ties, 0 losses; mean paired gain +567.05.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
