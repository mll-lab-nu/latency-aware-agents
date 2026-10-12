# Games and Gymnasium

Use a Sample Factory teacher, collect its demonstrations, and train or evaluate a compatible VLA.
Each stage writes local files. You can start with an existing teacher, dataset, or VLA bundle.
The [installation guide](../installation.md) describes the required environments.

## Select a task

Install the simulator and the reference model runtime. Conda must already be installed.
For InvertedPendulum:

```bash
bash install.sh --env inverted_pendulum --model openvla --tier dev
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latency
```

The task recipes are under `configs/examples/gymnasium/<task>/`.
The game recipes are under `configs/examples/<task>/`.
Keep the observation, action, clocks, and episode contract consistent across all stages.
Use the [Gymnasium guide](../guides/gymnasium_adapter.md) to add another task.

## Train and evaluate a teacher

Run the reference InvertedPendulum teacher:

```bash
python -m training.train \
  --config configs/examples/gymnasium/inverted_pendulum/small_model_train.yaml
```

This recipe trains for 1,000,000 raw steps with seed 0 and fixed 40 ms latency.
The environment and observation clocks are both 25 Hz.
The recipe saves checkpoints under `outputs/examples/gymnasium_inverted_pendulum/checkpoints/`.
Select the checkpoint from that run before evaluation:

```bash
python -m latency_bench eval \
  --config configs/examples/gymnasium/inverted_pendulum/small_model_eval.yaml \
  --checkpoint-path /ABSOLUTE/TEACHER/checkpoint.pth
```

The evaluator runs 20 episodes with a 1,000-step cap.
It saves results under `outputs/examples/gymnasium_inverted_pendulum/eval/`.
For shorter checks, use an overlay with smaller training and evaluation budgets.

## Collect and convert demonstrations

Save this as `my_collect.yaml` in the repository root:

```yaml
__base__: configs/examples/gymnasium/inverted_pendulum/small_model_eval.yaml
env:
  make_kwargs:
    render_mode: rgb_array
policy:
  source: local
  root: /ABSOLUTE/TEACHER_RUN
  checkpoint: checkpoint_p0/YOUR_CHECKPOINT.pth
collection:
  student_observation:
    image_key: null
    state_key: gymnasium_state
    state_source: metadata
  val_fraction: 0.1
  split_seed: 42
logging:
  output_dir: outputs/inverted_pendulum/dataset
  video:
    enabled: false
```

Replace the teacher root and checkpoint with the selected run.
The inherited configuration declares the state labels and five-action catalog.
Collection uses the teacher's issued commands as labels.

```bash
python -m latency_bench collect --config my_collect.yaml
```

Save this as `my_convert.yaml`:

```yaml
conversion:
  format: lerobot
  input_dir: outputs/inverted_pendulum/dataset
  output_dir: outputs/inverted_pendulum/lerobot
  action_carrier: native
```

```bash
python -m latency_bench convert --config my_convert.yaml
```

Conversion writes `lerobot/`, `lerobot__val/`, manifests, and a generated mixture.
Use the paths reported by the converter for training.
The [collection guide](../guides/rollout_collection.md) defines the data and normalization contracts.

## Train a VLA

Activate the model environment. The StarVLA launcher accepts the converted dataset and mixture:

```bash
conda activate starvla_rl_games_openvla
cd third_party/starVLA
bash commands/train_gymnasium.sh openvla \
  /ABSOLUTE/DATASET_PARENT lerobot /ABSOLUTE/GENERATED_MIXTURE.json \
  run_id=inverted_pendulum_local trainer.max_train_steps=5000
```

Replace the dataset parent and mixture with the conversion outputs.
The parent contains `lerobot/` and `lerobot__val/`.
The launcher reads the manifest's task, action width, and state contract.
It saves the model and training configuration under `results/Checkpoints/inverted_pendulum_local/` within StarVLA.
Preserve the actual training statistics with the checkpoint.
Pi05 preparation overlays require their declared data and backbone bindings before training.
They are not complete training runs.

## Evaluate and measure a VLA

Return to the repository root.
Use the [artifact guide](../artifacts.md#prepare-a-backbone-for-evaluation) to prepare an evaluation overlay.
Bind the checkpoint, full configuration, dataset manifest, prompt map, and backbone.

```bash
python -m latency_bench eval --config my_eval.yaml
python -m latency_bench profile run --config my_profile.yaml
```

Use `paused` for synchronous evaluation, `simulated` for a declared delay, and `realtime` for deployment timing.
The [profiling cookbook](profiling.md) explains fitting and replay.
Keep checkpoint revisions, episode counts, clocks, latency settings, and seeds with each result.
