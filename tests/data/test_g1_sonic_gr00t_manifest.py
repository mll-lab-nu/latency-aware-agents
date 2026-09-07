from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from latency_bench.data.filters import accepts_episode_contract
from scripts.teachers import export_g1_sonic_gr00t
from scripts.teachers.export_g1_sonic_gr00t import (
    _job_dirs,
    _load_manifest,
    _selected_jobs,
)
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_GR00T_ACTION_DIM,
    G1_GR00T_STATE_DIM,
)


BALANCE_TEACHER = "g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198"
HURDLE_TEACHER = "g1_hurdle_sonic_residual_h025_gamma099_1m_43d_v2"
RUN_TEACHER = "g1_run_sonic_reference_playback_43d_v2"


def test_g1_sonic_manifest_defines_jobs_and_acceptance_contracts():
    jobs = _selected_jobs(
        _load_manifest(Path("configs/teachers/g1_sonic_gr00t.json")), None
    )
    assert {job["name"] for job in jobs} == {
        HURDLE_TEACHER,
        RUN_TEACHER,
        BALANCE_TEACHER,
    }
    by_name = {job["name"]: job for job in jobs}
    assert by_name[HURDLE_TEACHER]["acceptance"] == {
        "type": "final_info_true",
        "key": "teacher_accepted",
    }
    assert by_name[BALANCE_TEACHER]["acceptance"] == {
        "type": "min_duration_s",
        "value": 18.0,
    }
    assert by_name[RUN_TEACHER]["source"] == "reference_playback"


def test_export_parser_overrides_selected_checkpoint_path(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "teachers": [
                    {
                        "name": "balance",
                        "task": "g1_sonic_balance_simple",
                        "source": "checkpoint",
                        "checkpoint": "old.pth",
                    }
                ]
            }
        )
    )

    captured = {}
    monkeypatch.setattr(
        export_g1_sonic_gr00t,
        "export_checkpoint_teacher",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    export_g1_sonic_gr00t.main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "output"),
            "--teacher",
            "balance",
            "--checkpoint",
            str(tmp_path / "accepted.pth"),
            "--no-convert",
        ]
    )
    assert captured["job"]["checkpoint"] == str(tmp_path / "accepted.pth")


def test_g1_sonic_manifest_derives_output_dirs_and_applies_gates():
    raw, official = _job_dirs(Path("outputs"), "teacher")
    assert raw == Path("outputs/teacher/raw")
    assert official == Path("outputs/teacher/lerobot")

    duration_gate = {"type": "min_duration_s", "value": 18.0}
    assert not accepts_episode_contract(
        {"episode_raw_frames": 899, "final_info": {}},
        duration_gate,
        env_fps=50.0,
    )
    assert accepts_episode_contract(
        {"episode_raw_frames": 900, "final_info": {}},
        duration_gate,
        env_fps=50.0,
    )

    final_info_gate = {"type": "final_info_true", "key": "teacher_accepted"}
    assert accepts_episode_contract(
        {"episode_raw_frames": 1, "final_info": {"teacher_accepted": True}},
        final_info_gate,
        env_fps=50.0,
    )
    assert not accepts_episode_contract(
        {"episode_raw_frames": 1, "final_info": {"teacher_accepted": False}},
        final_info_gate,
        env_fps=50.0,
    )


def test_reference_playback_probes_before_writing_rows(monkeypatch, tmp_path):
    class FakeEnv:
        action_space = SimpleNamespace(shape=(29,))

        def reset(self, *, seed):
            self.seed = seed
            self.step_count = 0
            return {}, {
                "official_gr00t_proprio": np.zeros(G1_GR00T_STATE_DIM, dtype=np.float32),
                "action_target": np.zeros(G1_GR00T_ACTION_DIM, dtype=np.float32),
            }

        def step(self, action):
            assert action.shape == self.action_space.shape
            self.step_count += 1
            terminated = self.step_count >= (1 if self.seed == 0 else 2)
            return (
                {},
                1.0,
                terminated,
                False,
                {
                    "official_gr00t_proprio": np.full(
                        G1_GR00T_STATE_DIM, self.step_count, dtype=np.float32
                    ),
                    "action_target": np.full(
                        G1_GR00T_ACTION_DIM, self.step_count, dtype=np.float32
                    ),
                },
            )

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        json.dumps(
            {
                "env": {
                    "task_name": "g1_sonic_run",
                    "env_id": "FakeG1Run-v0",
                    "registration_imports": [],
                    "make_kwargs": {"render_mode": "rgb_array"},
                    "env_fps": 50.0,
                    "obs_fps": 50.0,
                    "frame_stack": 1,
                    "base_prompt": "Run forward.",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(export_g1_sonic_gr00t.gym, "make", lambda *args, **kwargs: FakeEnv())
    raw_dir = tmp_path / "raw"

    result = export_g1_sonic_gr00t.export_reference_playback_dataset(
        job={
            "name": "run",
            "task": "g1_sonic_run",
            "source": "reference_playback",
            "playback_config": str(config_path),
            "max_steps_per_episode": 2,
            "acceptance": {"type": "min_duration_s", "value": 0.04},
        },
        raw_output_dir=raw_dir,
        official_output_dir=tmp_path / "lerobot",
        max_episodes=1,
        val_fraction=0.0,
        split_seed=0,
        convert=False,
        force=False,
    )

    report = result["rollout"]["filter_report"]
    assert report["attempted_episodes"] == 2
    assert report["rejected_episodes"] == 1
    assert report["train_rows_after_row_filter"] == 2
    assert result["rollout"]["base_prompt"] == "Run forward."
    assert (raw_dir / "images/train/episode_000000/decision_step_000001.png").is_file()
