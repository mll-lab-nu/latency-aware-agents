# Held-out sim2real: one experiment per invocation

`run.sh` runs one task/model on one explicitly selected inference GPU, in the
foreground. It has no job queue, GPU allocator, background process manager,
monitor or retry loop. The calling agent owns machine setup, concurrent jobs,
tmux, monitoring, result transfer and decisions to resume a failed experiment.

From a deployed checkout at `/workspace/latency-sensitive-bench`:

```bash
bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --task flappy --model qwenoft --gpu 0

bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --task mikasa_intercept_grab_fast --model qwenpi_v3 \
  --gpu 1 --simulator-gpu 0
```

GPU numbers are physical `nvidia-smi` indices. Ordinary experiments expose only
their inference GPU. InterceptGrab exposes simulator then inference GPU as
logical `cuda:0` and `cuda:1`; it still has exactly one inference worker.
Ant/InvertedPendulum use EGL rendering on the selected GPU. A caller may run
two ordinary experiments concurrently on different cards, or reserve both
cards for one InterceptGrab experiment.

Run from a Python environment containing this repository, the selected StarVLA
model dependencies and the task simulator. InterceptGrab additionally needs
the MIKASA/ManiSkill environment. Asset downloads use the current Hugging Face
credentials; the pinned profile datasets must be readable. The script does
not install environments or authenticate accounts.

## Frozen assets and task contracts

[`configs/experiments/sim2real.yaml`](../../../configs/experiments/sim2real.yaml)
lists all 18 task/model pairs, exact checkpoint/config files, evaluation archive
provenance, and training profile coordinates. Model repository
`latency-sensitive-bench/benchmark-models` is pinned to
`180da0effdaaaeacef218cf5fbaa4db41af67495`.

| Task ID | Models | Original env/obs FPS | Archived maximum steps |
|---|---|---|---|
| `flappy` | `qwenoft`, `qwenpi_v3`, `qwengr00t` | 10/10 | 3600 |
| `demon_attack` | same three | 60/15 | 7200 |
| `asterix` | same three | 60/15 | 7200 |
| `mikasa_intercept_grab_fast` | same three, corrected H1 `evalfix_20260923` | 20/20 | 60 transitions |
| `ant` | `qwenoft` | 10/10 | **3600** |
| `ant` | `qwenpi_v3`, `qwengr00t` | 5/5 | **1000** |
| `inverted_pendulum` | `qwenoft` | 10/10 | 1000 |
| `inverted_pendulum` | `qwenpi_v3`, `qwengr00t` | 5/5 | 1000 |

Eval YAMLs under `configs/experiments/sim2real/eval/` are verbatim archived
snapshots, including original host-local paths. Asset resolution relocates the
explicitly listed model/manifest/contract/prompt-map files and backbone.
Runtime configuration changes devices, mode, clocks, episode count/seeds and
logging output. Environment physics, frame skip, observation stride, model
inputs/history, action mapping, prompts, normalization, scheduler and maximum
steps retain their archived values. InterceptGrab preserves
`hold_last_chunk_action: true`.

Ant/IP prompt maps contain literal training-clock FPS and millisecond annotations.
Those strings and prompt keys remain exactly as trained while wall-clock pacing
is calibrated; they are not rewritten as target-GPU timing claims.

Ant/IP eval snapshots and training profiles come from the **dataset** repository
`latency-sensitive-bench/Standard-Pipeline` at
`ff90f1b6c46a6eaae55bbf341762f03a2be6e013`, distinct from its model repository's
`2208875...` revision. InterceptGrab snapshots are the local corrected-evaluation
archives identified by each published bundle's provenance.

The study pins ordinary `Qwen/Qwen3-VL-4B-Instruct` backbone assets to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. This revision is recorded by the newer
bundles; for historical OFT archives recording only a local backbone path, it
is this study's initialization/processor asset revision, not a claim about an
undocumented training-time revision. Ant OFT's runtime uses ordinary Qwen3-VL,
not the Action tokenizer variant.

## Protocol

Each experiment runs these stages serially:

1. Two independent realtime calibration probes, each warmed once and collecting
   the first 500 admitted observation-to-action latency records. A probe ends
   after completing/draining the episode that reaches the sample budget.
   Warmup records and records after the first 500 are excluded.
2. Fit sessions **0–4**, 20 episodes per session; fit and freeze one profile
   using only their `observation_attempts.jsonl` files.
3. Held-out sessions **5–9**, 20 episodes per session.
4. Five simulation runs for each of **fixed, normal, iid, temporal**. Replay
   always uses this experiment's frozen profile, never a mutable current ref.
5. Run-level summaries, Student-t 95% intervals and paired sim-minus-real errors.

Every real session is a fresh evaluator process with a fresh worker/model and
warmup; its 20 episodes share that runtime. Environment seeds are
`2000000 + 20 * session_id + episode_id`. Simulation run `i` uses validation
session `5+i`'s environment seeds and latency seed `i`. Calibration uses
1999500–1999999 and 1999000–1999499, separate from all fit/valid seeds even when
a probe needs 500 episodes. Simulation capacity and inference batch size are one.

Let `F_train` and `O_train` be the original model's env/obs FPS, and `L_train`
the mean latency in its exact training profile, in milliseconds. Calibration
targets `d = L_train * F_train / 1000` frames. Each probe measures complete
observation-to-action latency through the realtime executor. With probe mean
`L`, the next clock is `F = 1000*d/L`, `O = F/(F_train/O_train)`.

The clocks scale together; physical simulation steps and observation/action
stride do not change. Profile acquisition FPS is recorded separately and is
not substituted for training FPS. Calibration imposes no worker-utilization
FPS cap. After round two, clocks are frozen; fitting or validation observations
never change them. Reports include fractional and scheduler-rounded frame
statistics, observed fit/valid timing and capacity drops.

Each combination produces **200 real plus 400 simulated episodes**, excluding
warmup and calibration. Intervals use five run means, not 100 episodes as
independent machine sessions. They are conditional on one fitted profile and
do not quantify uncertainty from refitting profiles. Task-specific episode
metrics, including InterceptGrab success, also receive run means, intervals and
paired comparisons. Raw records remain available. Cross-task normalization and
paper plotting are separate postprocessing decisions.

## Results and explicit resume

The launcher prints `EXPERIMENT_ID` and `OUTPUT_DIR`. Default layout:

```text
results/sim2real/<task>/<model>/<gpu-instance>/<experiment-id>/
  contract.json
  hardware.json
  archived_eval.yaml
  calibration/{probe_0,probe_1}/
  calibration/calibration.json
  real/{0..9}/
  profile/
  sim/{fixed,normal,iid,temporal}/{0..4}/
  runs.csv
  comparison.json
  DONE
```

Evaluator directories contain `generated.yaml`, `stdout.log`, raw traces,
episode metrics and a `DONE` marker written only after successful completion.
A subprocess failure propagates to the caller. Resume explicitly:

```bash
bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --task flappy --model qwenoft --gpu 0 --resume EXPERIMENT_ID
```

Use the same `--output-root` if overridden. Resume retains completed stages and
the frozen profile, restarting incomplete runs from their beginning. Model,
config, training-profile and device identity must match the saved contract.
Resuming a completed experiment rebuilds summaries without repeating runs.

## Historical figures

`build_figures.sh` belongs to the retired OpenVLA/Flappy paper experiment and is
not called by this runner. Its old path predates the paper reorganization.
Current frozen Figure 4 sources live under
`paper/figures/fig4_sim2real_calibration/source/`; use that experiment's README
when reproducing historical figures.
