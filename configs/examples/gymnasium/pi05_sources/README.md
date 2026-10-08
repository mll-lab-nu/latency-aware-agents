# Pi05 seven-environment preparation contracts

`source_map.json` pins the Standard-Pipeline dataset revision and SHA256/size of every retained metadata file. Files under each environment are unchanged archive bytes, including historical absolute paths; those paths are evidence, not new Pi05 artifact bindings.

`../<environment>/pi05_train.yaml` is a preparation overlay loaded by `latency_bench.core.config.load_config`. Its `env` block comes from the archived actual profiling session and preserves make kwargs, action catalog, reset settings, physics frame skip and teacher frame stack. `hydra_overrides` is the argument list for `third_party/starVLA/commands/pi05/train_gymnasium_pi05.sh` after the three required positional dataset arguments. The launcher binds the newly materialized manifest; the seven tests compose the same overrides against the archived manifests. These overlays are not registered executable profiling workloads.

For an already materialized and verified dataset, invoke the existing entry with the overlay arguments (activate the matching StarVLA Python runtime, including Hydra and PyYAML):

```python
import subprocess
from pathlib import Path
from latency_bench.core.config import load_config
cfg = load_config("configs/examples/gymnasium/hopper/pi05_train.yaml")
starvla_root = Path("third_party/starVLA").resolve()
subprocess.run([
    "bash", "commands/pi05/train_gymnasium_pi05.sh",
    str(Path(dataset_root).resolve()), data_mix, str(Path(custom_mixtures_path).resolve()),
    *cfg["hydra_overrides"],
], cwd=starvla_root, check=True)
```

Supply the three dataset arguments from the actual conversion output. Before any run, also bind the immutable Qwen/Qwen3-VL-4B-Instruct revision, the approved per-environment training recipe and independent run output. The overlay's null bindings deliberately identify missing artifacts; there are no invented Pi05 checkpoints or Hub paths.

## Timing and source boundaries

| Task | Archived L0 env/obs FPS | Target env/obs FPS | Model state/action dim |
| --- | --- | --- | --- |
| AirRaid | 50/12.5 | 50/12.5 | 0/6 |
| InvertedPendulum | 25/25 | 10/10 | 4/5 |
| Hopper | 125/125 | 10/10 | 11/3 |
| HalfCheetah | 20/20 | 10/10 | 17/6 |
| Walker2d | 125/125 | 10/10 | 17/6 |
| Ant | 20/20 | 10/10 | 27/8 |
| Humanoid | 66.6667/66.6667 | 10/10 | 376/17 |

Source clocks are retained verbatim in `source_clock`; they are not silently rewritten in manifests. The six state tasks require explicit timing/prompt derivation and dataset conversion before target-clock training. Preserve raw-tick stride and simulator physics. State labels and normalization stay in the original manifest. Verify train-only min-max plus the selected registry transform offline/online when real samples are available.

AirRaid has no archived zero-latency manifest at the indexed paths. Its retained manifest is the archived L0 inference bundle used for profiling; the raw zero-latency metadata and training config are retained separately. That manifest has `uses_state=false` and a legacy placeholder `state_dim=1`; the Pi05 overlay correctly uses `include_state=false`, model `state_dim=0`. Its PAL mode, six actions, `noop_max=30`, `fire_reset=true`, frame skip 1 and teacher frame stack 4 come from the actual profile environment. VLA remains single-frame 224x224; teacher frame stack is not VLA temporal context.

Hopper's L0 dataset uses `LatencyBench/HopperRgbState-v0`, while the profiling runtime uses `LatencyBench/Hopper-v0`. Both are retained with their own source; the new dataset must be explicitly bound to the intended runtime rather than treating the IDs as interchangeable without verification.

## Reproduction inputs

The archived files describe source contracts. They do not identify a complete
new training run. Supply converted data, its manifest, the backbone revision,
and the selected training budget before using an overlay. Use the
[artifact registry](../../../../docs/artifacts.md) for published model and data bindings.
