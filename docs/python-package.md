# Install the framework package

The wheel contains all `latency_bench` and `training` modules, base configs,
task XML files, and CUDA kernel sources. It does not contain third-party
checkouts, documentation media, experiment outputs, or model checkpoints.

The default installation needs only NumPy and PyYAML. Latency clocks and
samplers work without Gymnasium, Torch, or simulator resources:

```bash
python -m pip install latency_sensitive_bench-0.1.1-py3-none-any.whl
```

Choose extras for the features you use:

| Extra | Features |
| --- | --- |
| `runtime` | Environment adapters, image processing, profiling, and logging |
| `data` | Dataset import and export |
| `training` | Torch, TorchVision, and reinforcement-learning dependencies |
| `starvla` | VLA model loading dependencies |
| `analysis` | Plotting |
| `dev` | Pytest |
| `mujoco`, `atari`, `vizdoom`, `box2d`, `sonic` | Selected simulator dependencies |

For example, install `[runtime,mujoco]` to run an existing MuJoCo task.
Reference training backends also require their pinned third-party packages.
Install these from a source checkout with the existing installer.
The CPU custom-component workflow needs `[runtime]`.

Base configs are package resources. The source checkout keeps
`configs/base` as a link to the same files for existing experiment YAMLs.
Pass your own experiment config by path. Experiment outputs remain outside
the installed package.

## Build and release

Build from a clean worktree with Python 3.10–3.12:

```bash
python -m pip install build
python -m build
```

The build produces a wheel and a source distribution in `dist/`.
Builds do not initialize third-party repositories or download their resources.
Publish both files as GitHub Release attachments with a versioned tag.

The `0.1.1` release uses benchmark revision `31c41252` for sooool.
Its sampling implementation is unchanged. Later framework versions must be
selected explicitly by downstream projects.
