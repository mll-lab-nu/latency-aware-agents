# StarVLA WanOFT checkpoint evaluation

The benchmark evaluates the checkpoints produced by the three StarVLA
one-click WanOFT pipelines:

- `third_party/starVLA/scripts/run_flappy_wan_oft_pipeline.sh`
- `third_party/starVLA/scripts/run_demon_attack_wan_oft_pipeline.sh`
- `third_party/starVLA/scripts/run_deadly_corridor_wan_oft_pipeline.sh`

Install the matching runtime once:

```bash
bash install.sh --tier use --model wan_oft --env flappy,demon_attack,deadly_corridor
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starvla_rl_games_wan_oft
```

Run an evaluation by overriding the generated checkpoint and prompt-map paths:

```bash
python -m latency_bench eval \
  --config configs/experiments/flappy/starvla_wan_oft.yaml \
  --checkpoint-path "/ABSOLUTE/RUN/checkpoints/steps_N_pytorch_model.pt" \
  --model-config-path "/ABSOLUTE/RUN/config.full.yaml" \
  --latency-prompt-map-path third_party/starVLA/data/flappy_fix_latency_3_200ep_context5/flappy_train__bridge/latency_prompt_map.json \
  --output-dir outputs/wanoft_eval
```

Use the corresponding config for the other tasks:

```text
configs/experiments/demon_attack/starvla_wan_oft.yaml
configs/experiments/deadly_corridor/starvla_wan_oft.yaml
```

For a short checkpoint smoke test, keep the training pipeline at 50 steps and
limit the rollout:

```bash
python -m latency_bench eval \
  --config configs/experiments/demon_attack/starvla_wan_oft.yaml \
  --checkpoint-path "/ABSOLUTE/RUN/checkpoints/steps_50_pytorch_model.pt" \
  --model-config-path "/ABSOLUTE/RUN/config.full.yaml" \
  --latency-prompt-map-path third_party/starVLA/data/demon_attack_fix_latency_6_200ep_context5/demon_attack_train__bridge/latency_prompt_map.json \
  --output-dir outputs/wanoft_smoke \
  --episode-start 0 \
  --episode-count 1 \
  --max-env-steps 50
```

`--latency-raw-frames` overrides both the benchmark's fixed raw-frame latency
and the prompt-map key. The shipped configs match the current pipelines:
Flappy uses 3 raw frames, while Demon Attack and Deadly Corridor use 6.

## Multi-image contract

WanOFT's saved `config.full.yaml` is the source of truth for
`datasets.vla_data.pack_image_sequence` and `image_sequence_length`.
The compact `config.yaml` saved beside these checkpoints does not contain
those dataset input fields. Pass `--model-config-path` when overriding a
checkpoint so latency-bench reads the full input contract before constructing
the environment and sizes the raw RGB history to cover the requested number
of policy observations.

For the default five-image checkpoints:

| task | raw frames per observation | raw history | selected frame indices |
| --- | ---: | ---: | --- |
| Flappy Bird | 1 | 5 | `0,1,2,3,4` |
| Demon Attack | 4 | 17 | `0,4,8,12,16` |
| Deadly Corridor | 4 | 17 | `0,4,8,12,16` |

The adapter passes selected images to StarVLA oldest-to-newest, matching `starVLA.training.rl_games.eval_core`.
At reset, the raw frame buffer repeats the initial image to match eval_core's initial history.
Packed sequences require raw RGB input.
A ghost-trail transform defines a different model input contract.

The prompt map supplies `latency_raw_frames` for in-process StarVLA training evaluation.
For the two frame-skipped tasks, six raw frames remain 1.5 policy decision intervals.
The evaluator does not round that delay to whole decision intervals.
