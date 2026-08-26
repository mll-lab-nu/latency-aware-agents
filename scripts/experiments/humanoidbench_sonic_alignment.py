"""Build the supported SONIC G1 HumanoidBench assets and paired hold videos.

The builder deliberately operates on XML/assets and MuJoCo directly.  It does
not import the latency-bench environment registry, so it can validate the
asset/comparison layer independently.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMANOID_BENCH_ROOT = REPO_ROOT / "third_party/humanoid-bench"
HB_ASSETS = HUMANOID_BENCH_ROOT / "humanoid_bench/assets"
DEFAULT_CANDIDATE_ROOT = REPO_ROOT / "results/humanoidbench_g1_official_asset_candidate"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/humanoidbench_g1_all_tasks_alignment"
UNSUPPORTED_TASKS = ("highbar_simple", "highbar_hard")

SUPPORTED_TASKS = (
    "stand",
    "walk",
    "run",
    "kitchen",
    "maze",
    "hurdle",
    "cube",
    "bookshelf_simple",
    "bookshelf_hard",
    "crawl",
    "window",
    "spoon",
    "door",
    "push",
    "reach",
    "basketball",
    "truck",
    "package",
    "cabinet",
    "sit_simple",
    "sit_hard",
    "balance_simple",
    "balance_hard",
    "stair",
    "slide",
    "pole",
    "room",
    "insert_normal",
    "insert_small",
    "powerlift",
)

SONIC_CAMERAS = {
    "cam_kitchen": {"pos": "-0.3 -2.5 2.5", "xyaxes": "1 0 0 0 0.5 1"},
    "cam_default": {
        "pos": "2 3 0.8",
        "mode": "trackcom",
        "xyaxes": "-1 0.5 0 -0.2 -0.2 1",
    },
    "cam_maze": {
        "pos": "3 3 3",
        "mode": "trackcom",
        "xyaxes": "-0.5 0.5 0 -0.4 -0.4 1",
    },
    "cam_tabletop": {
        "pos": "2.5 2 1.5",
        "mode": "trackcom",
        "xyaxes": "-0.5 0.5 0 -0.4 -0.4 1",
    },
    "cam_hurdle": {
        "pos": "-3 3 3",
        "mode": "trackcom",
        "xyaxes": "-0.5 -0.5 0 0.4 -0.4 1",
    },
    "cam_basketball": {
        "pos": "-1.8 3.5 3",
        "mode": "trackcom",
        "xyaxes": "-0.4 -0.5 0 0.4 -0.4 1",
    },
    "cam_hand_visible": {
        "pos": "-1.5 1.5 0.5",
        "mode": "trackcom",
        "xyaxes": "-0.8 -1 0 0 -0.3 1",
    },
    "cam_inhand": {
        "pos": "1.5 1.5 0.5",
        "mode": "trackcom",
        "xyaxes": "-1 0.8 0 0 -0.3 1",
    },
}

CAMERA_BY_TASK = {task: "cam_default" for task in SUPPORTED_TASKS}
CAMERA_BY_TASK.update(
    {
        "kitchen": "cam_kitchen",
        "maze": "cam_maze",
        "hurdle": "cam_hurdle",
        "truck": "cam_hurdle",
        "push": "cam_tabletop",
        "reach": "cam_tabletop",
        "spoon": "cam_tabletop",
        "basketball": "cam_basketball",
        "cube": "cam_inhand",
        "insert_normal": "cam_inhand",
        "insert_small": "cam_inhand",
        "bookshelf_simple": "cam_hand_visible",
        "bookshelf_hard": "cam_hand_visible",
        "window": "cam_hand_visible",
        "cabinet": "cam_hand_visible",
    }
)

REFERENCE_SCENES = {task: f"g1_torque_{task}.xml" for task in SUPPORTED_TASKS}
REFERENCE_ROBOTS = {task: "h1hand" for task in SUPPORTED_TASKS}

NATIVE_G1_TO_SONIC_JOINT = {
    "left_hip_pitch_joint": "left_hip_pitch_joint",
    "left_hip_roll_joint": "left_hip_roll_joint",
    "left_hip_yaw_joint": "left_hip_yaw_joint",
    "left_knee_joint": "left_knee_joint",
    "left_ankle_pitch_joint": "left_ankle_pitch_joint",
    "left_ankle_roll_joint": "left_ankle_roll_joint",
    "right_hip_pitch_joint": "right_hip_pitch_joint",
    "right_hip_roll_joint": "right_hip_roll_joint",
    "right_hip_yaw_joint": "right_hip_yaw_joint",
    "right_knee_joint": "right_knee_joint",
    "right_ankle_pitch_joint": "right_ankle_pitch_joint",
    "right_ankle_roll_joint": "right_ankle_roll_joint",
    "torso_joint": "waist_yaw_joint",
    "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint": "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "left_zero_joint": "left_hand_thumb_0_joint",
    "left_one_joint": "left_hand_thumb_1_joint",
    "left_two_joint": "left_hand_thumb_2_joint",
    "left_three_joint": "left_hand_index_0_joint",
    "left_four_joint": "left_hand_index_1_joint",
    "left_five_joint": "left_hand_middle_0_joint",
    "left_six_joint": "left_hand_middle_1_joint",
    "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint": "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
    "right_zero_joint": "right_hand_thumb_0_joint",
    "right_one_joint": "right_hand_thumb_1_joint",
    "right_two_joint": "right_hand_thumb_2_joint",
    "right_three_joint": "right_hand_index_0_joint",
    "right_four_joint": "right_hand_index_1_joint",
    "right_five_joint": "right_hand_middle_0_joint",
    "right_six_joint": "right_hand_middle_1_joint",
}


def _format_floats(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _model_reset_qpos(model: mujoco.MjModel, task: str) -> np.ndarray:
    return model.qpos0.copy() if task == "pole" else model.key_qpos[0].copy()


@cache
def _sonic_seed_model(candidate_root: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(
        str(candidate_root / "assets/envs/scene_run.xml")
    )


def _scene_qpos(task: str, candidate_root: Path) -> np.ndarray:
    native_model = mujoco.MjModel.from_xml_path(str(_reference_scene_path(task)))
    native_qpos = _model_reset_qpos(native_model, task)
    sonic_model = _sonic_seed_model(candidate_root)
    sonic_qpos = sonic_model.key_qpos[0, :50].copy()
    if task == "room":
        # The native G1 Room keyframe puts the SONIC collision feet below floor.
        # Keep the candidate's stable body pose and inherit only scene placement.
        sonic_qpos[:2] = native_qpos[:2]
        sonic_qpos[3:7] = native_qpos[3:7]
        return np.concatenate((sonic_qpos, native_qpos[44:]))
    sonic_qpos[:7] = native_qpos[:7]
    for native_name, sonic_name in NATIVE_G1_TO_SONIC_JOINT.items():
        native_address = native_model.joint(native_name).qposadr[0]
        sonic_address = sonic_model.joint(sonic_name).qposadr[0]
        sonic_qpos[sonic_address] = native_qpos[native_address]
    return np.concatenate((sonic_qpos, native_qpos[44:]))


def patch_sonic_robot_xml(source: Path, destination: Path) -> None:
    """Add the shared official camera/root contract to the one robot XML."""

    tree = ET.parse(source)
    root = tree.getroot()
    for element in root.iter():
        if element.attrib.get("name") == "floating_base_joint":
            element.set("name", "free_base")

    worldbody = root.find("worldbody")
    ET.SubElement(
        worldbody,
        "camera",
        {"name": "cam_kitchen", **SONIC_CAMERAS["cam_kitchen"]},
    )
    pelvis = next(
        body for body in root.iter("body") if body.attrib.get("name") == "pelvis"
    )
    existing = {camera.attrib["name"] for camera in pelvis.findall("camera")}
    for name, attrs in SONIC_CAMERAS.items():
        if name != "cam_kitchen" and name not in existing:
            ET.SubElement(pelvis, "camera", {"name": name, **attrs})

    ET.indent(tree, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=False)


def _reference_scene_path(task: str) -> Path:
    return HB_ASSETS / "envs" / REFERENCE_SCENES[task]


def build_scene_xml(task: str, candidate_root: Path, destination: Path) -> None:
    """Create one SONIC scene, retaining the official task include and objects."""

    source = _reference_scene_path(task)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", f"g1_sonic_{task}")
    for include in root.iter("include"):
        filename = include.attrib["file"]
        if filename.startswith("../robots/"):
            include.set("file", "../robots/g1_sonic.xml")

    qpos = _scene_qpos(task, candidate_root).tolist()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe):
        keyframe.remove(key)
    ET.SubElement(keyframe, "key", {"name": "qpos0", "qpos": _format_floats(qpos)})

    ET.indent(tree, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=False)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def write_source_and_licenses(asset_root: Path, candidate_root: Path) -> None:
    shutil.copy2(HUMANOID_BENCH_ROOT / "LICENSE", asset_root / "HUMANOIDBENCH_LICENSE")
    shutil.copy2(
        HB_ASSETS / "kitchen/adept_models/LICENSE", asset_root / "ADEPT_LICENSE"
    )
    shutil.copy2(
        HB_ASSETS / "kitchen/third_party/franka/LICENSE", asset_root / "FRANKA_LICENSE"
    )
    shutil.copy2(candidate_root / "SONIC_LICENSE", asset_root / "SONIC_LICENSE")
    humanoidbench_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HUMANOID_BENCH_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (asset_root / "SOURCE.md").write_text(
        f"""# SONIC G1 HumanoidBench all-task asset bundle

This bundle is generated by `scripts/experiments/humanoidbench_sonic_alignment.py`.

- HumanoidBench source revision: `{humanoidbench_revision}`.
- `common/`, `locomotion/`, `kitchen/`, and all supported task geometry are
  copied from the vendored HumanoidBench tree.
- `robots/g1_sonic.xml` and `robots/meshes/` come from the local pinned SONIC
  candidate. The generated XML only adds the shared official camera names and
  renames the free root joint to `free_base`.
- `envs/scene_<task>.xml` keeps the official task include. Root, common joints,
  hands, and task-object qpos are retargeted by joint semantics from the native
  G1 keyframe. SONIC-only waist and wrist axes retain their neutral value.
- `comparison.mp4` and `comparison.png` are generated separately by the
  explicit `compare` command. They use a deterministic 50 Hz, 250-frame
  keyframe-hold rollout and freeze a side after termination.

High Bar Simple and High Bar Hard are intentionally unsupported and are absent
from this bundle, registry, training matrix, and comparisons.

The bundle is local until the 30 comparisons pass manual review. The Kitchen
MuJoCo 3.3 default-class compatibility change belongs to the HumanoidBench
fork and is intentionally not silently applied here.
""",
        encoding="utf-8",
    )


def build_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
) -> Path:
    """Build all supported scene XMLs and the self-contained asset subset."""

    asset_root = output_root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    for directory in ("common", "tasks", "locomotion", "kitchen"):
        _copy_tree(HB_ASSETS / directory, asset_root / directory)
    for task in UNSUPPORTED_TASKS:
        (asset_root / "tasks" / f"{task}.xml").unlink()
        (asset_root / "envs" / f"scene_{task}.xml").unlink(missing_ok=True)
    _copy_tree(candidate_root / "assets/robots/meshes", asset_root / "robots/meshes")
    patch_sonic_robot_xml(
        candidate_root / "assets/robots/g1_sonic.xml",
        asset_root / "robots/g1_sonic.xml",
    )
    write_source_and_licenses(asset_root, candidate_root)
    for task in SUPPORTED_TASKS:
        build_scene_xml(task, candidate_root, asset_root / "envs" / f"scene_{task}.xml")
    return asset_root


def _reference_scene_for_compare(task: str) -> Path:
    return HB_ASSETS / "envs" / f"{REFERENCE_ROBOTS[task]}_pos_{task}.xml"


def _camera_for_task(task: str) -> str:
    return CAMERA_BY_TASK[task]


def _reset_room_objects(data: Any) -> None:
    """Apply HumanoidBench Room.reset_model's seeded object placement."""

    position = data.qpos.copy()
    velocity = data.qvel.copy()
    x_ranges = (
        (-3.5, -3.0),
        (-2.5, -2.0),
        (-1.5, -1.0),
        (-0.5, 0.0),
        (0.5, 1.0),
        (1.5, 2.0),
    )
    object_base = position.size - 42
    object_z = (0.01, 0.01, 0.082, 0.15, 0.11, 0.01)
    for object_index, ((x_low, x_high), z) in enumerate(zip(x_ranges, object_z)):
        qpos_address = object_base + object_index * 7
        position[qpos_address] = np.random.uniform(x_low, x_high)
        position[qpos_address + 1] = np.random.uniform(1.2, 3.5) * np.random.choice(
            [1, -1]
        )
        position[qpos_address + 2] = z
        if object_index == 2:
            position[qpos_address + 3 : qpos_address + 7] = np.array(
                [0.0733422, 0.0519076, -0.240058, -0.966591]
            )
        elif object_index == 3:
            position[qpos_address + 3 : qpos_address + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0]
            )
    data.qpos[:] = position
    data.qvel[:] = velocity


def _reset_data(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    task: str,
    seed: int,
) -> None:
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    if task == "room":
        np.random.seed(seed)
        _reset_room_objects(data)
    mujoco.mj_forward(model, data)


def _hold_control(mujoco: Any, model: Any, data: Any, target: Any) -> None:
    control = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        qvel_address = int(model.jnt_dofadr[joint_id])
        if model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE:
            control[actuator_id] = target[qpos_address]
        else:
            control[actuator_id] = 200.0 * (
                target[qpos_address] - data.qpos[qpos_address]
            ) - 20.0 * data.qvel[qvel_address]
    if model.nu:
        control = np.clip(
            control,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
    data.ctrl[:] = control


def _render_frame(mujoco: Any, renderer: Any, model: Any, data: Any, camera: str) -> Any:
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def _label_frame(frame: Any, label: str) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    ImageDraw.Draw(image).rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    ImageDraw.Draw(image).text((6, 5), label, fill=(255, 255, 255))
    return image


def _compare_task(
    task: str,
    asset_root: Path,
    output_root: Path,
    *,
    seed: int,
    steps: int,
    width: int,
    height: int,
) -> None:
    from PIL import Image

    from latency_bench.logging.video import AsyncVideoWriter

    np.random.seed(seed)
    reference_model = mujoco.MjModel.from_xml_path(str(_reference_scene_for_compare(task)))
    sonic_model = mujoco.MjModel.from_xml_path(str(asset_root / "envs" / f"scene_{task}.xml"))
    reference_data = mujoco.MjData(reference_model)
    sonic_data = mujoco.MjData(sonic_model)
    _reset_data(mujoco, reference_model, reference_data, task=task, seed=seed)
    _reset_data(mujoco, sonic_model, sonic_data, task=task, seed=seed)
    reference_target = reference_data.qpos.copy()
    sonic_target = sonic_data.qpos.copy()
    reference_renderer = mujoco.Renderer(reference_model, height=height, width=width)
    sonic_renderer = mujoco.Renderer(sonic_model, height=height, width=width)
    camera = _camera_for_task(task)
    frames: list[Any] = []
    comparison_dir = output_root / "comparisons" / task
    comparison_dir.mkdir(parents=True, exist_ok=True)
    writer = AsyncVideoWriter(comparison_dir / "comparison.mp4", fps=50.0)
    reference_frame = None
    sonic_frame = None
    reference_done = False
    sonic_done = False
    try:
        for frame_index in range(steps):
            for _ in range(10):
                if not reference_done:
                    _hold_control(mujoco, reference_model, reference_data, reference_target)
                    mujoco.mj_step(reference_model, reference_data)
                    reference_done = bool(
                        not np.isfinite(reference_data.qpos).all()
                        or reference_data.qpos[2] < 0.2
                    )
                if not sonic_done:
                    _hold_control(mujoco, sonic_model, sonic_data, sonic_target)
                    mujoco.mj_step(sonic_model, sonic_data)
                    sonic_done = bool(
                        not np.isfinite(sonic_data.qpos).all() or sonic_data.qpos[2] < 0.2
                    )
            if not reference_done:
                reference_frame = _render_frame(
                    mujoco, reference_renderer, reference_model, reference_data, camera
                )
            if not sonic_done:
                sonic_frame = _render_frame(
                    mujoco, sonic_renderer, sonic_model, sonic_data, camera
                )
            if reference_frame is None:
                reference_frame = _render_frame(
                    mujoco, reference_renderer, reference_model, reference_data, camera
                )
            if sonic_frame is None:
                sonic_frame = _render_frame(
                    mujoco, sonic_renderer, sonic_model, sonic_data, camera
                )
            joined = Image.new("RGB", (width * 2, height))
            joined.paste(_label_frame(reference_frame, f"H1 reference | {frame_index / 50:.2f}s"), (0, 0))
            joined.paste(_label_frame(sonic_frame, f"SONIC G1 | {frame_index / 50:.2f}s"), (width, 0))
            frame = np.asarray(joined)
            writer.write(frame)
            if frame_index in (0, steps // 2, steps - 1):
                frames.append(frame)
    finally:
        writer.close()
        reference_renderer.close()
        sonic_renderer.close()
    contact_sheet = Image.new("RGB", (width * 2, height * len(frames)))
    for index, frame in enumerate(frames):
        contact_sheet.paste(Image.fromarray(frame), (0, height * index))
    contact_sheet.save(comparison_dir / "comparison.png")


def compare_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    seed: int = 7,
    steps: int = 250,
    width: int = 256,
    height: int = 256,
) -> None:
    asset_root = output_root / "assets"
    for task in UNSUPPORTED_TASKS:
        shutil.rmtree(output_root / "comparisons" / task, ignore_errors=True)
    for task in SUPPORTED_TASKS:
        _compare_task(
            task,
            asset_root,
            output_root,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "compare", "build-and-compare"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    if args.command in ("build", "build-and-compare"):
        build_bundle(args.output_root, args.candidate_root)
    if args.command in ("compare", "build-and-compare"):
        compare_bundle(
            args.output_root,
            seed=args.seed,
            steps=args.steps,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
