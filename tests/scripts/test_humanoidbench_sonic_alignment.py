from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scripts.experiments import humanoidbench_sonic_alignment as alignment


def test_supported_task_inventory_and_reference_mapping_are_complete():
    assert len(alignment.SUPPORTED_TASKS) == 30
    assert all(
        task not in alignment.SUPPORTED_TASKS for task in alignment.UNSUPPORTED_TASKS
    )
    assert set(alignment.REFERENCE_ROBOTS) == set(alignment.SUPPORTED_TASKS)
    assert alignment.REFERENCE_ROBOTS["run"] == "h1hand"


def test_patch_robot_xml_adds_shared_camera_contract_and_root_name(tmp_path: Path):
    source = tmp_path / "robot.xml"
    source.write_text(
        "<mujoco><worldbody><body name='pelvis'>"
        "<joint name='floating_base_joint' type='free'/>"
        "<camera name='cam_default' pos='0 0 0'/>"
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    destination = tmp_path / "out/robot.xml"
    alignment.patch_sonic_robot_xml(source, destination)
    root = ET.parse(destination).getroot()
    worldbody = root.find("worldbody")
    pelvis = next(body for body in root.iter("body"))
    assert pelvis.find("joint").attrib["name"] == "free_base"
    assert {camera.attrib["name"] for camera in pelvis.findall("camera")} == set(
        alignment.SONIC_CAMERAS
    ) - {"cam_kitchen"}
    assert worldbody.find("camera").attrib["name"] == "cam_kitchen"


def test_build_scene_preserves_official_task_and_object_qpos(
    tmp_path: Path, monkeypatch
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    # Use an isolated source scene because the builder's official root is fixed;
    # this test exercises the qpos/include transformation directly below.
    source = source_root / "scene.xml"
    source.write_text(
        "<mujoco><include file='../robots/g1_torque.xml'/>"
        "<include file='../tasks/cube.xml'/><keyframe><key qpos='"
        + " ".join(["0"] * 51)
        + "'/></keyframe></mujoco>",
        encoding="utf-8",
    )
    monkeypatch.setattr(alignment, "_reference_scene_path", lambda _task: source)
    monkeypatch.setattr(
        alignment,
        "_scene_qpos",
        lambda _task, _candidate: np.zeros(57, dtype=np.float64),
    )
    destination = tmp_path / "out/scene.xml"
    alignment.build_scene_xml("run", tmp_path / "candidate", destination)
    root = ET.parse(destination).getroot()
    includes = [include.attrib["file"] for include in root.iter("include")]
    assert includes == ["../robots/g1_sonic.xml", "../tasks/cube.xml"]
    qpos = root.find("keyframe/key").attrib["qpos"].split()
    assert len(qpos) == 57
    assert qpos[-7:] == ["0"] * 7


def test_native_g1_joint_retargeting_covers_body_and_hands():
    assert len(alignment.NATIVE_G1_TO_SONIC_JOINT) == 37
    assert alignment.NATIVE_G1_TO_SONIC_JOINT["torso_joint"] == "waist_yaw_joint"
    assert (
        alignment.NATIVE_G1_TO_SONIC_JOINT["left_elbow_roll_joint"]
        == "left_wrist_roll_joint"
    )
    assert (
        alignment.NATIVE_G1_TO_SONIC_JOINT["right_six_joint"]
        == "right_hand_middle_1_joint"
    )


def test_room_reset_model_is_seeded_and_only_repositions_task_objects():
    class _Data:
        def __init__(self):
            self.qpos = np.zeros(92)
            self.qvel = np.arange(85, dtype=np.float64)

    first = _Data()
    second = _Data()
    initial_robot = first.qpos[:50].copy()
    initial_velocity = first.qvel.copy()
    np.random.seed(11)
    alignment._reset_room_objects(first)
    np.random.seed(11)
    alignment._reset_room_objects(second)

    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.qpos[:50], initial_robot)
    np.testing.assert_array_equal(first.qvel, initial_velocity)
    object_qpos = first.qpos[-42:].reshape(6, 7)
    np.testing.assert_array_less(
        object_qpos[:, 0],
        np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]),
    )
    np.testing.assert_array_less(
        np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5]),
        object_qpos[:, 0],
    )
    np.testing.assert_array_equal(
        object_qpos[:, 2], np.array([0.01, 0.01, 0.082, 0.15, 0.11, 0.01])
    )
