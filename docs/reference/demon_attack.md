# Demon Attack teacher evaluation contract

Demon Attack evaluation measures a full game, capped by the declared raw-frame
budget. Losing a life with lives remaining must continue the same episode and
accumulate its return. A life-loss reset clears policy history and pending actions.
It does not finish the formal episode or clear its accumulated reward.

## Frozen evaluation inputs

Use the published, training-selected best weights from
`MLL-Lab/LAGEN-models@46a33b18a2db8feb5cc6b520a95c2e96662c5e5b`:

| Profile | Checkpoint |
|---|---|
| OpenVLA/OFT | `best_000048224_24690688_reward_9340.850.pth` |
| GR00T | `best_000047744_24444928_reward_4477.300.pth` |
| π₀.₅ | `best_000047936_24543232_reward_2760.000.pth` |

Restore the adjacent `config.json`, including observation normalization and model
architecture. Keep 60/15 Hz, four raw observation frames, eight-decision history,
capacity one, `hold`, and `latest_ready`. Use each architecture's original RTX 3090
temporal profile. The evaluation manifest identifies the immutable profile revision.
Set `evaluation.eval_raw_reward: true`. The GPU evaluation backend must honor that
setting even when the training configuration retains `env.clip_reward: true`.

The original-cohort comparison uses 100 episodes, seeds
104771–104870, a 14,400-frame cap and 16 parallel environments. The student-cohort
comparison uses seeds 1000000–1000099, a 7,200-frame cap, 32 environments and latency
seed 271828, matching the archived student evaluation configuration.

Temporal sampler state belongs to a replica and continues across episodes.
Changing parallelism or episode-completion order can change subsequent latency
draws despite identical environment seeds. A backend equivalence experiment must
therefore replay the same actions and latency draws when isolating simulator
behavior. Report full-evaluation differences separately from these controlled
trajectory comparisons.

## ALE/CuLE observations

The CUDA core follows the installed ALE 0.8.1 initialization before applying
seeded NOOPs and the benchmark's FIRE/RIGHT reset actions. Demon Attack uses
60 NOOP frames and three four-frame console resets. Atlantis uses the same
console reset sequence. Asterix uses two console resets and one startup FIRE.
The hardware RNG uses ALE's fixed system seed 4753849, independently of the
episode seed. Cache construction reproduces power-on RAM, the 60-frame display
format detection, the ROM-loading initialization and the subsequent reset_game
initialization. These internal
frames do not count toward episode action frames.

The reset cache contains emulator state, RAM, render state and the generated
native frame. Restoring a cache entry restores all four. Cache entry zero is
one seeded NOOP after initialization. The core preserves terminal state until
its owner explicitly resets the episode or continues after life loss.

Advancing the environment generates the native frame. Reading that frame and
producing an 84×84 observation only convert the current frame buffer. Repeated
reads cannot change the current frame or the next transition.

CUDA grayscale conversion and area scaling match the CPU OpenCV 4.10.0 path
byte for byte. Area scaling accumulates horizontal float32 sums before vertical
sums, with explicit multiplication, addition and nearest-even rounding. The
same CUDA kernel serves ordinary steps and batched decisions.

## Runtime acceptance

Run the environment regression on the actual evaluation host and checkout:

```bash
python -m pytest -q -o addopts= \
  tests/envs/eval_align/test_latency_cpu_cule_alignment.py \
  tests/training/gpu/test_demon_cule_obs_matches_ale.py \
  tests/training/gpu/test_atlantis_cuda_core.py \
  tests/training/gpu/test_asterix_cuda_core.py
```

These suites cover all 30 NOOP reset counts, RAM and image equality, raw scoring,
life-loss continuation, repeated image reads and staggered resets in a 16-slot
batch. Missing optional runtime dependencies can cause skips.
A skipped check does not establish acceptance.

For the full executor regression, point the environment variable below at a
resolved OFT evaluation YAML with eight-decision history and restored local artifact paths.
The test skips when its configuration variable or CUDA is unavailable.
Missing files in an explicitly supplied configuration fail during loading.
The test does not download model weights.

```bash
DEMON_ATTACK_TEACHER_EVAL_CONFIG=/absolute/path/resolved_config.yaml \
python -m pytest -q -o addopts= \
  tests/envs/eval_align/test_latency_cpu_cule_alignment.py \
  -k frozen_h8
```

This test loads the real checkpoint and evaluates 16 seeded episodes on CPU/ALE
and CuLE through the production executor. Both must reach 7,000 raw frames,
including life-loss continuation, with matching episode returns. It would fail
under the former first-life termination behavior.

Preserve resolved configuration, exact checkpoint/profile identities, runtime
code revision and patch, dependency versions, test results, and per-episode
metrics with each accepted evaluation. Training completion, successful evaluation,
and acceptance of the reported metric are separate states. Preserve superseded
scores as audit evidence rather than overwriting the original records.
