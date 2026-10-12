# MIKASA Intercept Grab Fast

Install the reference task and model environments:

```bash
bash install.sh --env mikasa_intercept_grab_fast --model openvla --tier dev
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latency
```

Run commands from the repository root.
The [installation guide](../QUICK_START.md) describes host graphics libraries and interpreter selection.
The [artifact guide](../artifacts.md) identifies published teachers, datasets, and models.

## Choose your input

| Existing input | Next step |
| --- | --- |
| VLA inference bundle | Evaluate or profile the bundle |
| Prepared LeRobot data | Train with `--dataset-dir` |
| PPO teacher checkpoint | Collect demonstrations, then convert them |
| Published teacher data | Use the shared H1 launcher |

The shared H1 student uses top and wrist RGB views, 7D state, and one 7D action at 20 Hz.
Qwen models use Qwen3-VL-4B-Instruct.
WanOFT uses Wan2.2-TI2V-5B-Diffusers and tiles the simultaneous views.
The original image-only H1 baseline has a different input contract.
Use the contract saved with the selected checkpoint.

## Evaluate an existing model

Prepare the backbone recorded by the selected bundle:

```bash
hf download MODEL_REPO_ID --revision MODEL_COMMIT
```

Use the returned snapshot directory as `policy.backbone_path`.
Save this overlay as `my_eval.yaml` in the repository root:

```yaml
__base__: configs/examples/mikasa_intercept_grab_fast/profile_qwenoft.yaml
policy:
  source: local
  root: /ABSOLUTE/MODEL_BUNDLE
  checkpoint: checkpoints/model.pt
  model_config: config.yaml
  task_contract_path: task_contract.json
  backbone_path: /ABSOLUTE/BACKBONE_SNAPSHOT
executor:
  mode: simulated
  inference_devices: [cuda:1]
  parallel_envs: 1
latency:
  method: zero
evaluation:
  episodes: 100
  max_env_steps: 60
logging:
  output_dir: outputs/mikasa/eval
```

Replace the model and backbone paths with your prepared artifacts.
The bundle must match the state-based H1 recipe.
For another model, inherit its corresponding `profile_<model>.yaml`.

To use a published bundle directly, replace the source block with its exact repository coordinates:

```yaml
policy:
  source: hf
  hf_repo: MLL-Lab/LAGEN-models
  revision: MODEL_ARTIFACT_COMMIT
  root: PATH_WITHIN_MODEL_REPOSITORY
  checkpoint: checkpoints/model.pt
  model_config: config.yaml
  task_contract_path: task_contract.json
  backbone_path: /ABSOLUTE/BACKBONE_SNAPSHOT
```

Select the revision and path from the artifact manifest.
Run evaluation with separate logical simulator and inference GPUs:

```bash
conda activate starvla_rl_games_openvla
CUDA_VISIBLE_DEVICES=0,1 python -m latency_bench eval --config my_eval.yaml
```

The inherited seed is 4242424242. The example evaluates 100 episodes with a 60-step limit at 20/20 Hz.
Results identify the resolved model source and configuration.
This evaluation is independent of training and data collection.

## Train from published or prepared H1 data

Run a short training check using the shared published dataset:

```bash
conda activate latency
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh \
  --smoke --run-id mikasa-qwenoft-smoke
```

The check performs two SFT updates, saves and reloads the checkpoint, and evaluates two episodes.
It does not publish artifacts or establish policy quality.

For a full local run with prepared data, use:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh \
  --dataset-dir /ABSOLUTE/DATASET_PARENT \
  --run-id mikasa-qwenoft-local --no-publish
```

The parent must contain `lerobot/` and `lerobot__val/` with matching task contracts.
Omit `--dataset-dir` to reuse the shared published data.
The full H1 recipe uses 5,000 SFT updates, global batch 128, and 100 zero-delay evaluation episodes.
It defaults to two training GPUs.
The other launchers select `qwenpi_v3`, `qwengr00t`, or `wanoft`.

Use `--output-dir` to select a run directory.
The run saves training configuration, dataset references, evaluation records, and an `inference/` bundle.
Use `--vla-python` and `--task-python` for explicit interpreter paths.
See the [standard pipeline](../standard_pipeline.md) for resources, resume, and publication.

## Collect successful H1 data from a teacher

Use a native fixed-latency PPO checkpoint with matching action and observation contracts.
The recipe selects 250 successful training episodes and 25 successful validation episodes.
It uses 20 Hz control, latency seed 0, and a 60-transition limit.
It labels each student observation with the teacher's issued command.

Set the simulator and conversion interpreters before running the Bash entrypoint:

```bash
conda activate latency
export TASK_PY="$(command -v python)"
export LEROBOT_PY="$TASK_PY"
bash scripts/tasks/mikasa_intercept_grab_fast/collect_fixed_latency_dataset.sh \
  2 3 /ABSOLUTE/TEACHER/checkpoint.pt outputs/mikasa_successful_h1 YOUR_ORG/TEACHER_DATASET
```

`2` selects two raw steps, or 100 ms. `3` selects physical GPU 3.
The output directory is `outputs/mikasa_successful_h1/fixed_l2/`.
The script collects NPZ episodes, converts them to LeRobot, and uploads to the supplied dataset repository.
Publication requires write access to that repository.

To collect and convert locally, prepare the stage configurations separately:

```bash
conda activate latency
python -m scripts.tasks.mikasa_intercept_grab_fast.successful_h1_dataset \
  --stage prepare \
  --config configs/examples/mikasa_intercept_grab_fast/collect_successful_h1.yaml \
  --latency-frames 2 --checkpoint /ABSOLUTE/TEACHER/checkpoint.pt \
  --output outputs/mikasa_successful_h1/fixed_l2 \
  --repo-id YOUR_ORG/TEACHER_DATASET
CUDA_VISIBLE_DEVICES=3 python -m latency_bench collect \
  --config outputs/mikasa_successful_h1/fixed_l2/collect.yaml
python -m latency_bench convert \
  --config outputs/mikasa_successful_h1/fixed_l2/convert.yaml
```

The prepare stage records the publication destination but does not upload.
The generated converter writes train and validation data using training-derived normalization.
Add `--package-only` before the Bash entrypoint's positional arguments to reuse completed NPZ collection.

## Use H8 and conditioned H8 data

H8 predicts eight 7D actions per request.
The recipes use 20/20 Hz and a 60-transition limit.
One inference worker processes admitted requests. The executor drops observations when that worker is busy.
Chunks align with their source observation. The scheduler holds the last chunk action when the chunk expires.

| Task key | State | Training data |
| --- | --- | --- |
| `mikasa_h8` | 7D proprioception | Teacher suffix labels and valid-head masks |
| `mikasa_conditioned_h8` | 63D: 7D proprioception + 8×7 action prefix | Explicit prefix and loss masks |

The shared trainer accepts these keys through `--task`.
For conditioned H8, supply a prepared `--dataset-dir`.
The 63D v3 task cannot consume a 71D v2 dataset.
A fixed condition selects the prefix from its declared delay.
A profile condition selects it from the frozen profile mean.
The prefix never uses the next sampled delay.

Standalone task collection and conversion use the existing recipes:

```bash
conda activate latency
python -m latency_bench collect --config configs/examples/mikasa_intercept_grab_fast/collect.yaml
python -m latency_bench convert --config configs/examples/mikasa_intercept_grab_fast/convert.yaml
```

These recipes produce prefix H8 data for a fixed two-step delay.
Use `collect_conditioned_h8.yaml` for its explicit conditioned recipe.
Check each recipe's teacher, student, and normalization inputs before collection.

## Profile your model

Save `my_profile.yaml` in the repository root:

```yaml
__base__: configs/examples/mikasa_intercept_grab_fast/profile_qwenoft.yaml
policy:
  source: local
  root: /ABSOLUTE/MODEL_BUNDLE
  backbone_path: /ABSOLUTE/BACKBONE_SNAPSHOT
logging:
  output_dir: outputs/mikasa/profiling
```

Run the measurement:

```bash
conda activate starvla_rl_games_openvla
CUDA_VISIBLE_DEVICES=0,1 python -m latency_bench profile run --config my_profile.yaml
```

The inherited recipe uses five repetitions of 50 episodes, with 20/20 Hz and a 60-step limit.
It uses logical GPU 0 for MIKASA and GPU 1 for inference.
The command prints the measurement directory, fitted profile path, and completion status.
Use the [profiling cookbook](../profiling/cookbook.md) to replay the returned profile.
