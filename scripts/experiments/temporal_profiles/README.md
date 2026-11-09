# Offline temporal-profile pilot

For **new Flappy/QwenOFT hardware measurements**, use [`profile.sh`](profile.sh).
It installs/activates the model environment, downloads the pinned zero-latency
checkpoint, records timings once, fits Temporal/HMM/block on common sessions,
and exports/uploads the bundle. Run `bash scripts/experiments/temporal_profiles/profile.sh --help`
for settings, resume and reuse. The offline studies below use archived traces.

## Train the Flappy teachers

[`train_temporal.sh`](train_temporal.sh), [`train_hmm.sh`](train_hmm.sh), and
[`train_block.sh`](train_block.sh) each check/create and activate the `latency`
Conda environment, download their own profile, train an APPO teacher, save its
checkpoints, and upload to `latency-sensitive-bench/temporal-profile-teachers`.
Run on Linux/CUDA with an HF login or `HF_TOKEN` that can read the source dataset
and write/create the destination model repository. A new model repository is
private; existing visibility is preserved.

```bash
export WORKSPACE_DIR=/workspace
export OUTPUT_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/flappy_3090_teachers"
export GPU=0
export WANDB_ENTITY="your-wandb-username-or-team"
# Authenticate beforehand with WANDB_API_KEY or wandb login.
bash scripts/experiments/temporal_profiles/train_temporal.sh
bash scripts/experiments/temporal_profiles/train_hmm.sh
bash scripts/experiments/temporal_profiles/train_block.sh
```

Run sequentially from the repository root. All methods use 10/10 FPS, four
image frames, eight decisions of action history, 25M training transitions and
seed 42. The source is `latency-sensitive-bench/temporal-profiles` at revision
`1a4bd7bbc14815e0478810457ff204d4cf70636b`, run `20261109T002620386065Z`:
the original Temporal, two-state HMM and block-32 fits. No refitting occurs.

W&B is enabled in the experiment YAML under project `flappy-teachers`.
`WANDB_ENTITY` is required; `WANDB_PROJECT` optionally overrides the YAML project.
The scripts pass both settings into the training configuration. Credentials stay
in `WANDB_API_KEY` or your W&B login. Each output directory gets a distinct run ID,
reused on resume. Follow `episode_train/rolling_mean_return`,
`episode_train/rolling_mean_length` and `episode_train/completed_decision_steps`.

Outputs are separate under `$OUTPUT_DIR/METHOD/seed_42`. Rerun to resume training
or retry an upload after export. `--help` lists controls including `SEED`,
`HF_MODEL_REPO`, `FORCE_INSTALL=1` and `SKIP_UPLOAD=1`. Bundles retain best/latest
checkpoints, the exact profile, configs and logs. Selection uses training
reward; the automatic final evaluation uses the latest checkpoint with 100
episodes and a 3,600-frame cap. Demonstration collection, including its
7,200-frame cap, is a later stage.

Preparation selects a fresh start when no saved weights exist, and resumes when
a latest checkpoint exists. Failed startup directories are archived by Sample
Factory on a fresh start; corrupt checkpoints still fail explicitly. If upgrading
an output prepared before W&B was enabled, choose a new `OUTPUT_DIR` because the
old protocol froze different logging settings. Stop the old process first.

## Collect VLA demonstrations from local teachers

[`collect_temporal.sh`](collect_temporal.sh), [`collect_hmm.sh`](collect_hmm.sh),
and [`collect_block.sh`](collect_block.sh) use the local `bundle/` produced by
the teacher training scripts. They read `training_complete.json` to select the
recorded best checkpoint and use that bundle's own fitted profile. No model or
profile download is performed.

From the repository root on the Linux/CUDA machine:

```bash
export WORKSPACE_DIR=/workspace
export OUTPUT_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/flappy_3090_rollouts"
export HF_DATASET_REPO=latency-sensitive-bench/temporal-profile-rollouts
export GPU=0
# Authenticate with HF_TOKEN or hf auth login, with dataset write permission.

export TEACHER_BUNDLE=/absolute/path/to/temporal/seed_42/bundle
bash scripts/experiments/temporal_profiles/collect_temporal.sh

export TEACHER_BUNDLE=/absolute/path/to/hmm/seed_42/bundle
bash scripts/experiments/temporal_profiles/collect_hmm.sh

export TEACHER_BUNDLE=/absolute/path/to/block/seed_42/bundle
bash scripts/experiments/temporal_profiles/collect_block.sh
```

Each script activates `latency`, installing it with the existing installer if
absent, then collects, audits, converts and uploads. New HF dataset repositories
are private; existing visibility is preserved. `--help` lists the controls.

The recipe reuses `scripts/tasks/decision_history/collect_vla_hist8.py`: simulated
10/10 Hz, the teacher's fitted timing generator, capacity 1, deterministic teacher
actions, 256 probe/replay environments, two-pass collection, no filtering, 90/10
episode split with split seed 0, and collection seed 42. Start at 200 episodes and
add 50 until there are at least 320,000 training labels. The default cap is the
explicitly chosen **7,200 raw frames**, rather than the current reference source's
3,600-frame Flappy default. `EPISODES`, `MAX_ENV_STEPS` and `MIN_TRAIN_ROWS` expose
those budgets; changes require a new output directory. The game clock is simulated,
so collection does not wait for wall-clock 10 FPS.

Outputs are under `$OUTPUT_DIR/METHOD/seed_42`:

- `raw/`: canonical embedded-image Parquet, episode seeds and collection metadata.
- `lerobot/` and `lerobot__val/`: StarVLA training and validation datasets. Each
  decision has one current RGB image, 24 binary values for eight prior decisions,
  and the teacher-issued action. The teacher still consumes four stacked frames.
- `_generated_mixtures/lerobot.json`: existing StarVLA dataset-mixture handoff.
- `quality_report.json`, `history_audit.json`, configs, profile and teacher
  metadata: source identity, history checks, returns, lengths and row counts.
- `upload_receipt.json`: exact published dataset revision and folder URL.

The H8 route preserves its existing task prompt without adding latency text.
VLA training must use the matching H8 recipe: one image, 24 state values, identity
state normalization and action horizon 1. Set local dataset paths after download;
archived manifests retain original collection-machine paths as provenance.

Completed data is reused on rerun. If upload fails, retry only publication:

```bash
# Keep the same OUTPUT_DIR and HF_DATASET_REPO as above. No teacher/GPU needed.
bash scripts/experiments/temporal_profiles/collect_temporal.sh --upload-only
```

`SKIP_UPLOAD=1` retains a local dataset. Interrupted partial GPU image replay has
the existing collector's restriction against resuming a subset of slots; preserve
the failed output and use a new `OUTPUT_DIR`. No rollout or dataset upload is
performed by merely installing these scripts.

## Complete 30-case cohort

The full-cohort entrypoint downloads the pinned calibration traces, audits all
1,500 sessions, fits/tunes the alternatives separately for every case, and builds
an aggregate report. It covers the complete Cartesian product of:

- Flappy Bird, Inverted Pendulum and MIKASA InterceptGrabFast;
- QwenOFT and QwenGR00T;
- A100, L40S, RTX 3090, RTX 4090 and RTX 5090.

This is the archived calibration cohort, not all checkpoints or games elsewhere
in the Hugging Face organization. Install the requirements below, then run:

```bash
export WORKSPACE_DIR="$PWD"
export SOURCE_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/all_cases_source"
export OUTPUT_DIR="$WORKSPACE_DIR/outputs/temporal_profiles/all_cases"
export WORKERS=4
bash scripts/experiments/temporal_profiles/all_cases.sh
```

`PYTHON` selects the interpreter. `SKIP_DOWNLOAD=1` reuses the downloaded source.
The immutable revisions are the same as the pilot. Downloads include only the
experiment metadata, three profile files, and 50 request logs per case. Auditing
checks all fitting timings against each published profile; outer validation is
initially inspected only for structural session lengths. Each case then learns
and tunes using only the archived fitting partition.

The shared primary score uses **8- and 16-request windows**, because the shortest
recording has 26 accepted requests. This includes every validation run of every
case without padding or concatenating independent sessions. It is a new common
protocol; do not combine its score values with the pilot's 8/32/128 score.
Secondary 32/128-window scores reuse the frozen choices and report their exact
eligible run/case coverage. ACF diagnostics use lags up to `min(64, n-1)`, so ACF
errors do not use identical lag support in every case. Count-window diagnostics
return missing values when a session is too short.

The inner 20/5 split, candidate grids, three restarts, five tuning repetitions,
25-session refit and ten final generation repetitions follow the stronger study
below. The cohort uses a fixed **1,000-iteration HMM budget per restart**. Select
the highest training likelihood, retaining and disclosing convergence failures
rather than dropping cases or changing the budget after seeing validation
scores. The standalone study defaults to 500 iterations and also records any
failure to reach tolerance. This replaces the earlier pilot runner's fail-fast
convergence behavior.

Aggregation weights cases equally, then runs equally within case. Its 2,000
bootstrap draws resample validation run IDs and generation repetitions jointly
across cases and methods, preserving reused seeds and keeping case weights fixed.
Secondary horizons renormalize the same cluster draws over eligible runs.
The intervals condition on fitted generators and selected settings; they are not
uncertainty over new hardware, tasks, checkpoints, or fitting/model selection.
Case and subgroup intervals are exploratory, without multiplicity correction.

Outputs include `report.html`, `report.pdf`, `timing_traces.pdf`,
`explanation.txt`, `case_results.csv/json`, `group_results.csv`, `summary.json`,
`protocol.json`, `cohort_audit.json`, a portable report ZIP, and per-case evidence
under `cases/<gpu>/<task>/<model>/`. Full generated sequences stay in each case's
`sequences/`; the portable report bundle excludes those large archives. Timing
plots always show validation run 39, repetition 0, first 300 requests or the
entire shorter session. No display is selected based on the result.

Completed case directories can be reused by rerunning `matrix run`; an incomplete
directory causes an explicit error and must be inspected before rerunning. The
runner saves each subprocess log and completion status. Report-only rerender:

```bash
python -m scripts.experiments.temporal_profiles.matrix_report --output "$OUTPUT_DIR"
```

## Original single-case entrypoint

Compare timing generators for Flappy Bird / QwenOFT / RTX 3090 using a pinned
Hugging Face calibration case. This runs on CPU; it does not load a VLA, train
a policy, run a game, or model admission decisions. Raw artifacts and results
belong under `outputs/temporal_profiles/`.

```bash
python3.12 -m venv .venv-temporal-profiles
source .venv-temporal-profiles/bin/activate
python -m pip install -r scripts/experiments/temporal_profiles/requirements.txt
export OUTPUT_DIR="$PWD/outputs/temporal_profiles/flappy_qwenoft_rtx3090"
bash scripts/experiments/temporal_profiles/run.sh
```

Use `SKIP_DOWNLOAD=1` to repeat the CPU analysis of downloaded sources.
`PYTHON` selects the interpreter. Plot-only rerender:

```bash
python -m scripts.experiments.temporal_profiles.plot --output "$OUTPUT_DIR"
```

## Protocol

- Profiles: `MLL-Lab/LAGEN-profiles@8531005becf942264a5c8e3965d1a94b657a90df`.
- Experiment: `MLL-Lab/LAGEN-datasets@0b3b64a5d5f4a52e5b1decf1a30b01ec8637becc`.
- Case: `experiments/sim2real-calibration/cases/rtx3090/flappy/qwenoft`.
- Respect the archived random split of 25 fit / 25 validation runs, two episodes
  per run. Never fit across session boundaries. Episodes within a session retain
  the original temporal sampler's continuing state.
- Audit submitted requests, dropped requests, observation ordering, worker IDs,
  positive service/overhead timings, and agreement of the fitting sample count
  and mean with the archived profile. Dropped requests have no latency emission.
- Reuse the repository's Fixed, IID and Temporal samplers; the archive uses v13.
- HMM: two full-covariance Gaussian states on standardized log busy time and log
  observation-to-worker overhead. Exponentiation produces positive busy time and
  overhead; action latency is their sum. Use three EM restarts (7, 42, 2026),
  selecting by fitting likelihood only. Save iterations and convergence evidence.
- Stationary bootstrap: jointly resample action/busy pairs in geometric blocks,
  mean length 32 selected before inspecting validation results. Start uniformly
  over fitting observations; circular continuation stays in its original session.
  This is an initial baseline, not a tuned optimal block length.
- One generated sequence per validation session, matched in accepted-request
  count and episode boundaries. Seed is `20261107 + 1000 * method_index + run_id`.
- Use action-response latency for primary plots. Fixed's existing busy-time
  rounding differs from other samplers; this is not a controlled capacity study.
- Diagnostic slow spells exceed fitting action-latency p95, a common cutoff for
  all methods. This differs from the internal Temporal burst detection rule.
  A separate `worker_spike_metrics.json` repeats the diagnostics on worker busy
  times using the archived Temporal worker-spike threshold. Fixed is excluded
  from that check because its busy time uses frame rounding.
  Spells touching session boundaries retain their observed (possibly censored)
  lengths; spells and autocorrelation pairs never cross independent sessions.
- ACF uses within-session centering. Fixed's ACF is undefined. KS is reported as
  a descriptive distance, without its IID-test p-value. No significance claim.
- The display is the first 1,000 accepted requests of the first validation run
  listed by the archived protocol (39), with a single shared y scale. It is not
  selected for a favorable burst. Generated bursts need not align with real ones.

Outputs: `index.html`, two PNG/PDF figures, `metrics.csv/json`, `audit.json`,
`hmm_fit.json`, `provenance.json`, `sequences.npz` and a code snapshot. The report
distinguishes the quality of timing reproduction from policy/game performance.

## Method references

- [HMM / explicit-duration distinction, Yu (2010)](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf).
- [Stationary bootstrap, Politis and Romano (1994)](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf).
- [hmmlearn GaussianHMM API](https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn.hmm.GaussianHMM).

## Inner-tuned study with repeated generation

After downloading the pilot sources, the stronger procedure reuses them without
network access. Choose a new output directory; existing directories are rejected.

```bash
export SOURCE_DIR="$PWD/outputs/temporal_profiles/flappy_qwenoft_rtx3090/source"
export OUTPUT_DIR="$PWD/outputs/temporal_profiles/stronger_flappy_qwenoft_rtx3090"
bash scripts/experiments/temporal_profiles/study.sh
```

The executable protocol is written before model fitting. The archived 25 fitting
runs are permuted with seed 20261108, then split into 20 inner-training and 5
inner-tuning sessions. Alternatives see only inner-training measurements during
selection. HMMs have 2, 3 or 4 states, three EM restarts each, a maximum of 500
iterations and absolute likelihood tolerance .01. Restarts are selected by
training likelihood. Block candidates have mean lengths 8, 16, 32, 64 or 128.

Each candidate generates five repetitions of the five tuning-session lengths.
Each independent generated session contributes exactly one sampled window per
horizon (8, 32, 128), making 25 ensemble members per horizon. Real tuning sessions
contribute up to 64 uniformly sampled windows each. Windows can overlap within
a real session, but a session remains the statistical unit. A generated window
never spans two independent generated sessions.

Selection minimizes an equal-weight average of six scores: energy and variogram
at each of the three horizons, averaged equally across the five tuning sessions.
An exact tie selects the smaller setting. The choice and every candidate's
session scores are saved before any outer validation measurements are read.
The selected HMM is refitted on all 25 original fitting runs; the selected block
length resamples all 25 fitting runs. Fixed/IID/Temporal use the original profile
fitted on those same 25 runs. Baseline parameters and code are not tuned here.

Both scores use action-response latency standardized by the fitting mean and
standard deviation (inner-training data during selection, all fitting data in
final evaluation). Energy uses mean distance to the observed window minus half
the mean off-diagonal ensemble-pair distance, divided by sqrt(window length).
Variogram uses exponent .5, dyadic lags up to 64, pair weights 1/lag, and divides
by total pair weight. It subtracts ensemble variance / ensemble size from each
squared-error term to remove finite-ensemble bias. Generated ensemble members
come from independent timing sequences; correlated windows from one sequence
are not treated as independent ensemble samples. Finite-sample corrected scores
can be negative. The combination is a declared selection objective, not a claim
of universal optimality for control tasks.
The corrections are exactly unbiased under IID ensemble sampling. This study
stratifies independent generated sessions by recorded length and episode
schedule, so their use here is approximate; this limitation applies to the
window-score comparisons and is recorded in the report.

Final evaluation uses ten repetitions of all 25 archived validation lengths.
For each repetition, each method supplies 25 independent generated windows per
horizon, then is scored against all real validation sessions. Real-window draws
are shared across methods and repetitions. Also report per-run distributions,
autocorrelation, slow-response counts, frame-deadline exceedances, consecutive
miss probabilities and longest miss streaks. Deadline conversion follows the
repository's ceiling/epsilon rule. Every original fitting and validation run
is kept intact, including its two consecutive episodes.

Uncertainty uses 2,000 crossed bootstrap draws of whole real-run IDs and whole
generation-repetition IDs. The same draws are used for all methods and paired
differences against Temporal. No individual request/window is treated as an
independent experiment. Intervals condition on fitted parameters and selected
settings, exclude fitting uncertainty, and are not adjusted for multiple
comparisons. The validation set was already inspected in the earlier pilot:
this is a stronger exploratory check, not new blinded confirmatory evidence.

Outputs include the full protocol, inner selection, final HMM parameters, source
snapshot, ten compressed generated-sequence archives, per-run/repetition metrics,
bootstrap summaries and paired differences, three PNG/PDF figures and `index.html`.
These checks do not perform queue/admission replay, game evaluation, or training.

Rerender the report without repeating fitting/generation:

```bash
python -m scripts.experiments.temporal_profiles.study_plot --output "$OUTPUT_DIR"
```

Package the completed study as a plain-language visual report without fitting or
generating new timings:

```bash
python -m scripts.experiments.temporal_profiles.package_report --output "$OUTPUT_DIR"
```

This adds `report.html`, a four-page `report.pdf`, an overview figure, a trace
comparison using saved repetition 0, `summary_results.csv` and `report_bundle.zip`.
The display uses the first 1,000 accepted requests of the first validation run,
with shared axes. The CSV flattens the saved summary without changing metrics;
deadline fractions remain fractions in the CSV and are scaled per 10,000 only in
the report. The ZIP contains reports, figures, metrics, protocol, selection,
source snapshots and the original small profile files. Full timing arrays stay
in `sequences/`; raw source JSONL files stay in `SOURCE_DIR`. Both are excluded
from the lightweight review bundle. The source directory recorded in the
completed study's protocol must remain available to package its profile files.

Window-score references:
- [Gneiting and Raftery (2007), proper scoring rules](https://doi.org/10.1198/016214506000001437).
- [Scheuerer and Hamill (2015), variogram scores](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf).
