# Atlantis / Asterix GPU training gate — 2026-09-22

## Scope and implementation

`env.name: gymnasium` and the Sample Factory checkpoint format remain unchanged. The
Gymnasium integration now selects `AtlantisBatchedGpuEnv` or `AsterixBatchedGpuEnv`
when `env.simulator: gpu`; an unsupported GPU task fails at the required task mapping
instead of constructing a CPU Gymnasium environment. Static formal eval configs remain
explicitly `simulator: cpu`. Separate `sf_eval_gpu_zero_latency.yaml` configs exercise
the existing batched CuLE eval backends for learning and preprocessing checks.

The GPU candidate keeps the established Atari learning contract: one worker,
rollout/recurrence `32/32`, learner batch `1024`, two epochs, eight batches per epoch,
`max_policy_lag=300`, constant LR `0.000303`, `convnet_simple`, ELU, entropy coefficient
`0.003`, Adam epsilon `1e-6`, no gradient clipping, normalized inputs/returns, raw reward,
four-frame stack, and no life-loss termination.

## H100 batch selection

Both tasks ran concurrently, one H100 per task. The 4096-env probes trained for 1M
frames. The selected 2048-env probes trained to checkpoint step `6,520,832`; the final
FPS column is Sample Factory's complete 300-second rolling window. Lag averages summarize
all non-warmup reports; `max lag` is the largest observed value.

| Task | GPU envs | Frames | FPS | Mean reported lag | Max lag | Consecutive `>=300` |
|---|---:|---:|---:|---:|---:|---:|
| Asterix | 4096 | 1.02M | 20.389k | 137.56 | 479 | 1 |
| Atlantis | 4096 | 1.02M | 18.350k | 154.17 | 495 | 2 |
| Asterix | **2048** | 6.52M | **19.105k** | **41.12** | **239** | **0** |
| Atlantis | **2048** | 6.52M | **18.661k** | **46.09** | **255** | **0** |

The durable candidate decision is therefore 2048 GPU envs for both tasks. It remains in
the same speed class as Flappy while avoiding the recurrent queue/lag spikes seen at
4096. Primary throughput evidence:
[Asterix log](../../outputs/atari_gpu_gate/2026-09-22/logs/asterix_gpu2048_65m.log) and
[Atlantis log](../../outputs/atari_gpu_gate/2026-09-22/logs/atlantis_gpu2048_65m.log).

## Checkpoint compatibility and learning signal

The 6.52M checkpoints were evaluated deterministically for episode IDs `0..99` with
config seed 42, which maps to environment reset seeds `104771..104870`, on both the CuLE
GPU backend and formal CPU/ALE backend.

| Task | GPU eval mean ± SD | CPU/ALE mean ± SD | Mean length |
|---|---:|---:|---:|
| Asterix | 2515.0 ± 765.294 | 2515.0 ± 765.294 | 3188.9 |
| Atlantis | 23227.0 ± 2417.555 | 23399.0 ± 2350.978 | 5000.0 |

Asterix is episode-for-episode identical across backends. Atlantis differs by 0.74% in
mean return; both backends show a learned policy, and the formal CPU path loads and runs
the GPU-trained checkpoint. Evidence:
[GPU eval](../../outputs/atari_gpu_gate/2026-09-22/eval_gpu) and
[CPU/ALE eval](../../outputs/atari_gpu_gate/2026-09-22/eval_cpu).

CuLE/ALE fixed-action core tests passed `5/5`; GPU routing, batch conversion, explicit CPU
eval, and no-fallback contract tests passed `8/8`. The isolated H100 checkout used CUDA
12.9 with an SM90-only extension cache; this was test-host setup and did not add a code
fallback.

## Gate status

Throughput, lag, integration, episode metrics, and GPU-to-CPU checkpoint compatibility
are verified. This 6.52M single-seed probe is not the planned seed-`10,11,12` CPU/GPU
1M/3M sample-efficiency A/B and is not the final 10M acceptance run. Its formal CPU/ALE
scores are below the final 90% thresholds (`26343.9` Atlantis, `3689.1` Asterix), so the
existing memory/small-model CPU training specs remain unchanged and no checkpoint is
accepted or published from this gate. The next decision is the three-seed same-frame A/B;
only a passing result should switch those formal specs to GPU.
