from __future__ import annotations

import json
from pathlib import Path

from latency_bench.data.filters import accepts_episode_contract
from scripts.teachers import export_g1_sonic_gr00t
from scripts.teachers.export_g1_sonic_gr00t import _job_dirs, _load_manifest, _selected_jobs

BALANCE_TEACHER = "g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198"


def test_g1_sonic_manifest_defines_jobs_and_acceptance_contracts():
    jobs = _selected_jobs(_load_manifest(Path("configs/teachers/g1_sonic_gr00t.json")), None)
    assert [job["name"] for job in jobs] == [BALANCE_TEACHER]
    assert jobs[0]["task"] == "g1_sonic_balance_simple"
    assert jobs[0]["source"] == "checkpoint"
    assert jobs[0]["acceptance"] == {"type": "min_duration_s", "value": 18.0}


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
