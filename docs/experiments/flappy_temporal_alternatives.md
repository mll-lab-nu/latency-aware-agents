# Flappy Bird: Temporal, HMM and block resampling

This protocol compares the timing generator used to train a teacher and its VLA.
All three branches use the same raw hardware measurements, game clocks, inputs,
training budgets and evaluation seeds. The first implementation supports one
inference worker, slot 0, matching the Flappy/OpenVLA (QwenOFT) experiment.

## Download, measure, fit and publish

The end-to-end entrypoint is
[`profile.sh`](../../scripts/experiments/temporal_profiles/profile.sh).
Run it on the GPU machine whose timing you want to model. It reuses `install.sh`
for Flappy/OpenVLA, activates `starvla_rl_games_openvla`, installs the profiling
extras, downloads the zero-latency model and backbone, measures once, fits the
selected generators, exports the evidence and uploads to an existing HF dataset
repository. Authenticate with `hf auth login` or `HF_TOKEN` with access to the
model and write access to the destination repository.

```bash
cd /workspace/latency-sensitive-bench
export WORKSPACE_DIR=/workspace
export OUTPUT_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/profiling/flappy_3090_run1"
export GPU=0
export HF_REPO=MLL-Lab/LAGEN-profiles
bash scripts/experiments/temporal_profiles/profile.sh
```

The defaults are a starting protocol with **explicit, untuned settings**:

| Setting | Default |
| --- | --- |
| Zero-latency checkpoint | `MLL-Lab/LAGEN-models`, revision `a4099f759f71ffa72d967be3695c8e054cfc377c` |
| Model bundle | `zero-latency/flappy/vla/starvla-qwenoft-h1/openvla_flappy_fix_latency_0` |
| Weights | `checkpoints/steps_5000_pytorch_model.pt` |
| Backbone | `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Game / observation clocks | 30 / 30 Hz; FPS calibration disabled |
| Execution | Realtime, one environment, one inference worker on logical `cuda:0` |
| Prompt | Existing recipe's `latency_neutral` mode |
| Measurement budget | `REPEATS=5`, `EPISODES=10`, `MAX_ENV_STEPS=3600`; seed 0 |
| Fitting split | `FIT_SESSION_IDS=0,1,2`; sessions 3 and 4 reserved |
| Methods | `METHODS=temporal,hmm,block` |
| HMM | `HMM_STATES=2`, `HMM_MAX_ITERATIONS=1000`, restarts 7/42/2026 |
| Block resampling | `BLOCK_MEAN_LENGTH=32` admitted requests |

The model is QwenOFT, named `openvla` by the existing installation/profiling
recipe. The checkpoint metadata identifies the Qwen3-VL backbone; the script
binds its downloaded location instead of relying on a training-machine path.
Only the selected checkpoint, its configuration/statistics and the backbone
are downloaded. Warmup follows the existing realtime runner and is recorded
separately from measured observations. Each repeat is an independent process;
the game-seed protocol repeats across those processes.

The script freezes `protocol.json` before measurement. Every method, including
Temporal, is fitted again on exactly `FIT_SESSION_IDS`. The runner's automatic
Temporal fit on all sessions remains in the local raw run but is excluded from
the comparison export. Reserved sessions are included as raw evidence for later
evaluation; this script does not compute held-out scores or select winners.
Choosing HMM states or block lengths using those reserved sessions would make
them tuning data, requiring separate untouched evaluation sessions.

Set `SKIP_INSTALL=1` to reuse an environment with the profiling/analysis extras
already installed. Set `SKIP_UPLOAD=1` for a local export. A single method can be
selected with `METHODS=hmm` or `METHODS=block`. `--help` lists the controls.
The **7,200-frame limit applies to later demonstration collection**, not this
3,600-frame profiling budget.

To resume an interruption, rerun with the same `OUTPUT_DIR` and settings.
Completed measurement sessions and fits are reused. A completed measurement
also lets an upload retry skip GPU execution. To change the fitting split or
settings, use a new output directory and reuse the raw run:

```bash
# Copy RUN_DIR from the previous output's measurement.json.
export EXISTING_RUN_DIR=/absolute/path/to/raw/runs/openvla/GPU/flappy/INSTANCE/RUN_ID
export OUTPUT_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/profiling/flappy_3090_refit"
export SKIP_INSTALL=1
export HMM_STATES=3
export BLOCK_MEAN_LENGTH=16
bash scripts/experiments/temporal_profiles/profile.sh
```

In reuse mode, the raw run's recorded clocks, budgets and checkpoint identity
take precedence over `REPEATS`, `EPISODES` and `MAX_ENV_STEPS`; it downloads no
weights and performs no GPU measurement. It requires completed sessions with
`run_contract.json`, `hardware.json` and `sessions/<id>/` timing records.

Outputs are:

- `profiles/<method>/profile.json` and its companion files: ready for the
  teacher's `latency_profile_path`. Keep each method directory intact.
- `bundle/`: profiles, all raw sessions, warmup evidence, measured configuration,
  hardware, split/settings and model/code provenance, with an explanatory README.
- `upload_receipt.json`: HF repository, exact commit revision and browse link.
- Stage logs and `measurement.json`, including the raw `RUN_DIR` for reuse.

The upload creates a new folder under
`experiments/temporal-alternatives/flappy/openvla/<GPU>/<RUN_ID>/<OUTPUT_DIR_NAME>`
in the dataset repository; `HF_PATH_PREFIX` changes the leading prefix. It
rejects an already existing remote folder rather than replacing an artifact.
It uploads the bundle only, excluding downloaded model weights and HF caches.
To retry publication alone from the activated environment:

```bash
python -m scripts.experiments.temporal_profiles.profiling upload \
  --output-dir "$OUTPUT_DIR" --repo-id "$HF_REPO"
```

## Experiment protocol

1. **Identify the measurement source.** Reuse the raw `observation_attempts.jsonl`
   sessions behind the selected RTX 3090 profile if available. Record the zero
   checkpoint path/revision, software, GPU, worker count and clock. The released
   history experiment's original profile was measured at 30/30 Hz. Its published
   profile summary alone is insufficient to fit HMM or resample blocks. Our
   previous 30-case study uses a different calibration cohort; do not substitute
   those fitted alternatives for this profile without refitting every method on
   the same cohort. If remeasurement is needed, run the selected zero-latency VLA
   once under a common 30/30 Hz, one-worker profiling protocol and share those
   recordings among all methods. Freeze the measurement-session budget beforehand.

2. **Fit and select timing models.** Split independent recordings into fitting,
   tuning and untouched evaluation recordings before examining comparison
   results. Never randomly split individual requests from the same recording.
   Reuse the existing offline study's selection procedure: HMM states 2/3/4,
   three fitting restarts (7, 42, 2026), 1,000 iterations; block mean lengths
   8/16/32/64/128. Select using the prespecified window scores on tuning runs,
   then refit on fitting+tuning runs. Temporal receives the identical final
   fitting set. Save every choice, split, profile version and fit convergence
   diagnostic. The CLI accepts selected settings; its defaults are not evidence
   that a candidate has been tuned. With only the five original sessions, adapt
   the session split explicitly; do not claim the earlier 25/25-run protocol.

3. **Check timing fidelity.** Compare generated sequences with untouched raw
   recordings using the existing window scores, marginal delay distances,
   autocorrelation, slow-run lengths, and deadline exceedances. Report each
   metric separately. This tests timing fidelity; it does not establish which
   trained agent will play better. Include frame-based diagnostics at the
   downstream 10 Hz clock as well as millisecond-space comparisons.

4. **Train three small teachers.** Use APPO, 25 million environment steps,
   10/10 Hz environment/observation rates, four-frame teacher images, eight
   completed-decision history, worker capacity 1, and identical optimizer and
   checkpoint-selection rules. Only the fitted timing generator changes.
   The supplied sweep starts with seed 42. For a final training-variability
   comparison, repeat the complete branches with the same additional seeds;
   evaluation episodes alone do not measure training-seed variability.

5. **Collect fresh demonstrations for every method.** Use each teacher's own
   fitted generator and 10/10 Hz clock. Collect 200 episodes, cap each episode at
   **7,200 raw frames**, keep all episodes, and split 180/20 with split seed 0.
   Death ends an episode earlier. Two-pass collection resets the timing stream
   from the episode seed so probe and image replay agree. Store the issued
   teacher action and the existing binary decision history. Log actual rows and
   return distributions; equal episode limits do not imply equal dataset sizes.
   Recollect Temporal too: the released Temporal dataset had an 18,000-frame cap.

6. **Train three VLAs.** Convert each dataset through the shared converter, then
   use the existing QwenOFT history recipe: one RGB image, 24 history values,
   action horizon 1, pretrained Qwen3-VL backbone and a fresh action head,
   5,000 updates. Match optimizer settings, global batch and initialization
   seeds. The zero-latency VLA supplies the hardware measurement workload;
   the student initialization follows the existing supervised-training recipe.
   No environment runs during these supervised updates.

7. **Evaluate on hardware.** Freeze the trained checkpoints and run each on the
   same RTX 3090, 10/10 Hz clock, one inference worker, inference batch 1, one
   active environment, seeds 1000000–1000099, and a **3,600-raw-frame evaluation
   cap**. Use `executor.mode: realtime`: inference duration determines delivery,
   and there is no HMM/Temporal/block delay injection. Evaluate the zero-latency
   VLA under this same protocol as a reference. Interleave/randomize branch run
   order to reduce machine drift. Record actual achieved FPS, delays, drops,
   per-episode returns and uncertainty over seeds. The existing H8 evaluator
   produces simulated profile replay; label that as a separate diagnostic.

Save raw measurements, split IDs, profiles, selected checkpoints, collection
metadata, VLA configs and evaluation episodes under `outputs/` or the artifact
repository. Conclusions must follow the results; no method is presumed to win.

## Implemented methods

Select either method through the ordinary latency config:

```yaml
executor:
  mode: simulated
  simulated_worker_capacity: 1
latency:
  method: hmm  # or block; Temporal remains method: temporal
  profile_path: outputs/temporal_profiles/policy_comparison/profiles/hmm/profile.json
  profile_worker_slot: 0
  seed: 42
  add_latency_info: false
```

The HMM has full-covariance Gaussian emissions on standardized log worker busy
time and log positive overhead (`action delay - busy time`). It keeps session
boundaries in fitting and hidden state between admitted requests. Fitting uses
[hmmlearn GaussianHMM](https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn.hmm.GaussianHMM).
Its saved parameters replay using NumPy only. Nonpositive overhead is rejected
explicitly. Emissions are not clipped to observed extremes; training queues
grow if needed. Nonconvergence is retained in the saved restart diagnostics.

Block resampling uses geometric block lengths with a configured mean. Each
block starts uniformly over recorded admitted requests and follows consecutive
joint `(action delay, busy time)` pairs, wrapping within that recording only.
It never crosses into another recording inside a block. This is the same
stationary-bootstrap procedure used in the offline comparison.

Both advance only for admitted requests, preserve machine state across ordinary
episode resets, and use explicit seed resets for reproducible teacher
probe/replay. CPU wrappers, batched teacher scheduling and simulated evaluation
consume the same sampler interface. Delays stay in milliseconds and are rounded
up using the current environment clock, so a 76 ms delay becomes three raw
frames at 30 Hz or one at 10 Hz. Multiple workers are rejected rather than
modeled as independent streams with unsupported cross-worker correlations.

## Fitting from an existing local profiling run

Use the normal profiling environment; add the HMM fitting dependency with
`python -m pip install -e '.[profiling,analysis]'`. Runtime sampling itself does
not import hmmlearn. `RUN_DIR` is the directory returned by
`python -m latency_bench profile run`, containing `run_contract.json`,
`hardware.json` and completed `sessions/<id>/` records.

After selecting the fit IDs and hyperparameters on tuning data:

```bash
export WORKSPACE_DIR="$PWD"
export OUTPUT_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/policy_comparison"
export RUN_DIR=/absolute/path/to/completed/profiling/run

# Example IDs/settings only. Replace them with the frozen fitting split and
# selected hyperparameters. Held-out session IDs must not appear here.
python -m latency_bench profile fit --run-dir "$RUN_DIR" \
  --method temporal --session-ids 0 1 2 \
  --output-dir "$OUTPUT_DIR/profiles/temporal"
python -m latency_bench profile fit --run-dir "$RUN_DIR" \
  --method hmm --session-ids 0 1 2 --hmm-states 3 --hmm-max-iterations 1000 \
  --output-dir "$OUTPUT_DIR/profiles/hmm"
python -m latency_bench profile fit --run-dir "$RUN_DIR" \
  --method block --session-ids 0 1 2 --block-mean-length 32 \
  --output-dir "$OUTPUT_DIR/profiles/block"
```

Without an explicit output, alternatives are written to `RUN_DIR/profile_hmm`
and `RUN_DIR/profile_block`. Existing Temporal output is preserved. Reusing a
completed output with different fitting settings fails; use a new directory.
For archived raw logs without a run contract, the shared Python API
`build_profile_from_traces(..., method="hmm"/"block")` accepts the same session
descriptors and provenance as Temporal. Do not invent a measurement contract.

## Teacher training and 7,200-frame collection

Generate the three matching teacher configurations and inspect the commands:

```bash
python scripts/sweeps/generate_sweep_configs.py \
  --spec configs/experiments/flappy/temporal_alternatives_25m.yaml \
  --eval-checkpoint-kind best
cat "$OUTPUT_DIR/configs/train_commands.txt"
```

Each command invokes the existing `python -m training.train --config ...`
entrypoint. GPU environment installation and CUDA extension setup follow the
existing Flappy installation scripts. The spec defaults to the output layout
above; use the generator's path overrides if relocating it.

After a teacher finishes, supply its explicitly selected checkpoint file. For
example, for HMM:

```bash
python -m scripts.experiments.temporal_profiles.prepare_collection \
  --checkpoint /absolute/path/to/hmm_teacher/checkpoint_p0/best_SELECTED.pth \
  --profile-path "$OUTPUT_DIR/profiles/hmm/profile.json" \
  --output-dir "$OUTPUT_DIR/data/hmm"
python -m latency_bench collect --config "$OUTPUT_DIR/data/hmm/collect.yaml"
python -m latency_bench convert --config "$OUTPUT_DIR/data/hmm/convert.yaml"
```

Repeat for Temporal and block with their own checkpoints and profiles. The
helper validates the teacher's Flappy clock/history/capacity, obtains the method
from its checkpoint config, and writes collection/conversion configs. It
defaults to exactly 200 episodes, a 7,200-frame cap, no return filter and 10%
validation. It does not train, collect or overwrite an existing config itself.

Use `scripts/tasks/decision_history/train_vla_hist8.py` for the converted
`data/<method>/lerobot` directories, with distinct run IDs and the same backbone,
learning rates and global batch. Its `--training-condition` accepts `hmm` and
`block` for accurate metadata; Temporal keeps `profile`. The history evaluator's
`--latency-method hmm|block|temporal` selects simulated replay for diagnostics.
Final hardware comparisons use the shared `latency_bench eval` realtime path.

These commands define a runnable protocol after fitting inputs and GPU resources
are supplied. Implementation tests and config generation are not completed
teacher training, dataset collection, VLA training, or hardware evaluation.
