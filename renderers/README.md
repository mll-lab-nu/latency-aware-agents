# renderers/

Renderer plugins: they draw a sport's **scene** from its **camera definitions** for one **world state**. Design and
rules: [docs/framework_zh.md](../docs/framework_zh.md) §3.1 and §9 (renderer rules).

```python
from renderers import load, available, Scene, CameraDef
R = load("mujoco")(Scene.from_file("scene.xml"))                  # or Scene(xml=MJCF_TEXT, assets={path: bytes})
cams = [CameraDef("behind", pos=(0, -3, 1.2), lookat=(0, 0, 0.2), fovy_deg=45, width=320, height=240)]
images = R.render({"qpos": qpos}, cams)                            # one H x W x 3 uint8 array per camera
cams[0].project(world_point)                                       # the pixel every renderer draws that point at
```

| name | module | real time (player images) | GPU | what it is |
|---|---|---|---|---|
| `mujoco` (default) | `renderers/mujoco` | yes | no (osmesa) / optional (egl) | MuJoCo's rasterizer; the competition picture |
| `mujoco-showcase` | `renderers/mujoco_showcase` | yes (slower: shadows, MSAA, `ssaa`) | no | same scene, better lights and materials; `Scene.look = {"preset": "tabletennis"}` uses a sport look of `scripts/olympiad/showcase/looks.py` |
| `cycles` | `renderers/cycles` | no (re-render only) | yes (`device="CPU"`: no, slow) | Blender 4.2 Cycles in `rcb-showcase-blender` (Docker) |
| `isaac` | `renderers/isaac` | no (re-render only) | yes (RTX) | Isaac Sim 4.5 RTX in `nvcr.io/nvidia/isaac-sim:4.5.0` (Docker) |

The interface (`renderers/base.py`): `CameraDef(name, pos, lookat, up=(0,0,1), fovy_deg=45, width=320, height=240)`
with `intrinsics()`, `basis()`, `project(p)`, `to_dict()/from_dict()`; `Scene(engine="mujoco", xml, assets, look)`;
`Renderer(scene, **opts)` with `render(state, cameras) -> [images]`, `close()`, class attributes `name`, `realtime`,
`needs_gpu`, `label`; `RendererUnavailable(name, why, requirements)`. `renderers/budget.py`: `frame_ms(...)` (median
ms per observation, all cameras) and `check_budget(..., budget_ms) -> (ok, ms)`.

## Presentation scenes (spectators only)

A sport may dress its match scene for spectators (renderer rule 6): `sports/<sport>/presentation/presentation.toml`
names robot models of the asset manifest (`renderers/presentation/menagerie.json`: MuJoCo Menagerie at a pinned
commit, sha256 per file) and the scene bodies that get their visual geoms. `renderers/presentation.dress(match_scene,
dir)` (or `for_plugin(plugin_path, match_scene)`) returns a `Scene` with `match` set: the match scene plus visual-only
geoms (name prefix `_vis:`, contype = conaffinity = 0, mass 0) on the same bodies, the dressed bodies' own geoms moved
to the hidden group 3. Every renderer checks `scene_check.same_kinematics(match, presentation)` before drawing one and
refuses otherwise; `scene_check.pose_diff` compares body poses over recorded states (0.0 for every sport so far).
`label.frame_label(renderer, scene)` is the frame tag ("... | presentation meshes, spectators only"). The players'
images do not use it (`core/observe.CameraWorker` refuses) unless the sport declares `players` (below). The model files
are not in the repository:

```bash
python3 -m renderers.presentation.assets fetch      # into $RO_ASSET_CACHE or <repo>/.cache/assets (git-ignored)
python3 -m renderers.presentation.assets verify
```

### Players' images from the presentation scene (core 0.2.9)

`players = "25k"` (or `"full"`) at the top of `[presentation]` makes the players' images use the presentation scene
too (`Scene.players`; `presentation.players_scene(sport_dir, match)`, or `renderers.players_model(...)` for a sport that draws
its own images; `core/observe.players_scene` for the core's camera worker). `tools/sportlib.assemble_sim` bundles the
declaration and the models' files with those meshes (sha256-checked, licences included) into
`<sim>/sport/presentation/` (assets under `assets/`) and copies `renderers/presentation/` (without `lod/`) into the
simulator: the match is offline and the files are part of `source_sha`. The players' frames carry no presentation tag
(they are the competition picture). Levels of detail (`lod.json`, files in `lod/<level>/`, committed: Franka Panda
25k = 0.8 MB, KUKA iiwa 14 25k = 0.8 MB) are the upstream visual meshes decimated by quadric error metrics:

```bash
pip install trimesh==5.1.1 fast-simplification==0.2.0     # a developer step; rebuilding reproduces the files
python3 -m renderers.presentation.lod build franka_emika_panda 25k
```

The sport also sets `[observation] render_threads` (llvmpipe threads per render process: `LP_NUM_THREADS` in the
generated task; without it llvmpipe starts one thread per visible host CPU whatever the container's quota: no measurable
loss for one render process, but the seats' render processes then compete for the quota) and `sim_cpus` (the
sim container's CPUs). Measured (one observation, osmesa, 2026-10-07): full Menagerie
meshes cost 1.3-6.3x the match scene (+300-470 ms for 8 tiles with two Pandas, still +250-290 ms at 8-16 threads:
the extra vertex work barely scales with llvmpipe threads), the 25k level 1.0-2.1x (+80-95 ms at 1 thread, +40-75 ms
at 2). One process per camera tile scales far better than threads: 8 tiles with two full Pandas, 8 CPUs: 1 process x 8
threads 360-540 ms, 8 processes x 1 thread 140-220 ms (not implemented yet).

Never put `renderers/` itself on `sys.path` (the `mujoco` backend package would shadow the `mujoco` library); import
the package `renderers` from its parent directory.

## Adding a renderer

1. A package `renderers/<name>/` whose `__init__.py` defines a `Renderer` subclass and `RENDERER = <class>`; one row in
   `REGISTRY` (`renderers/__init__.py`).
2. The rules (docs/framework_zh.md §9):
   - it never changes physics or match state: draw on your own model copy, from the given state only;
   - same scene: geometry, kinematics, cameras and the colours a viewer sees equal the default renderer's
     (`renderers/scene_check.same_scene`); refuse to draw otherwise;
   - it follows `CameraDef.project` (the pinhole model) to within a pixel;
   - for player images it must meet the task's `frame_budget_ms` (`budget.check_budget`; tools/check measures it) and
     set `realtime`; a GPU requirement is declared in `needs_gpu`, with `RendererUnavailable(requirements)` when missing;
   - results record the player-image renderer; every re-rendered frame carries `label` ("re-rendered: <engine>",
     `renderers/label.stamp`), with the presentation note for a presentation scene (`label.frame_label`);
   - a presentation scene (`Scene.match` set) is drawn only after `scene_check.same_kinematics` (inherited from the
     `mujoco` renderer), and only for spectators.
3. Tests in `renderers/test_renderers.py` (stdlib runner, in the sim image):
   `docker run --rm -v $PWD:/w -w /w -e MUJOCO_GL=osmesa ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 python3 renderers/test_renderers.py`

The path tracers export the scene with `renderers/export.py` (the format of
`scripts/olympiad/showcase/blender/export_scene.py`) and run the existing showcase scripts in their images; they need
Docker, the image and (except Cycles on CPU) an NVIDIA GPU on the calling host, and mujoco in the calling process.
`scripts/olympiad/showcase/rerender.py` (whole-match re-renders of legacy tasks) is unchanged.
