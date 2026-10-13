# LAGEN documentation

Start with installation and the workflow guide. Choose a cookbook for your task.
Run checkout commands from the repository root.

## Start here

| Guide | Purpose |
| --- | --- |
| [Installation](installation.md) | Install the package, task dependencies, and model runtime |
| [Workflow](workflow.md) | Select a stage from your existing environment, teacher, dataset, model, or profile |
| [CPU example](guides/custom_components.md#run-the-cpu-example) | Run evaluation with your own components |
| [Published artifacts](artifacts.md) | Download compatible models, datasets, and profiles |

## Cookbooks

| Task | Guide |
| --- | --- |
| Games and Gymnasium | [Teacher training, collection, conversion, and VLA evaluation](cookbooks/games_gymnasium.md) |
| MIKASA manipulation | [Intercept Grab Fast](cookbooks/mikasa.md) |
| HumanoidBench | [Balance Simple](cookbooks/balance.md) |
| Deployment latency | [Measurement, fitting, and replay](cookbooks/profiling.md) |
| MIKASA teacher training | [PPO with a measured profile](cookbooks/mikasa_latency_ppo.md) |
| WanOFT game models | [Checkpoint evaluation and image history](cookbooks/wan_oft.md) |

## Guides

| Guide | Purpose |
| --- | --- |
| [Configuration](guides/configuration.md) | Set tasks, model sources, clocks, delays, and budgets |
| [Custom components](guides/custom_components.md) | Connect environment and policy factories |
| [Gymnasium adaptation](guides/gymnasium_adapter.md) | Register a task and train with the latency wrapper |
| [Demonstration collection](guides/rollout_collection.md) | Align observations and teacher commands, then convert data |
| [Memory](guides/memory.md) | Select temporal images and KV memory |
| [Periodic evaluation](guides/latency_bench_periodic_eval.md) | Evaluate Sample Factory checkpoints during training |

## Reference

| Reference | Contents |
| --- | --- |
| [Execution semantics](reference/latency_execution_semantics_en.md) | Clocks, capacity, ordering, hold, and action chunks |
| [Parallel evaluation](reference/eval_parallelism.md) | Episode slots, inference batches, and environment backends |
| [Realtime traces](reference/realtime_pipeline_logging.md) | Timing fields, measurement boundaries, and statistics |
| [Runtime architecture](reference/infra.md) | Package responsibilities and interfaces |
| [Demon Attack](reference/demon_attack.md) | Full-game scoring and life-loss resets |

## Reproduction and maintenance

- [Paper experiment recipes](experiments/README.md)
- [Artifact contracts and fixed sources](artifacts.md)
- [Hugging Face card sources](hf_cards/README.md)
- [Third-party sources](../third_party/README.md)
- [Test conventions](../tests/README.md)
