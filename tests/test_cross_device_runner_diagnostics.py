from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.cross_device import runner


def test_finalize_profile_artifacts_writes_diagnostics_json(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles" / "openvla-flappy" / "flappy" / "L_L40S"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.json").write_text(json.dumps({"profile_id": "L_L40S"}) + "\n", encoding="utf-8")
    fingerprint = {"gpu": {"name": "NVIDIA L40S"}, "packages": {"flash_attn": "2.7.4"}}
    monkeypatch.setattr(runner, "_host_fingerprint", lambda: fingerprint)
    monkeypatch.delenv("HF_PROFILE_REPO", raising=False)
    manifest = {}
    task = SimpleNamespace(profile_output_dir=profile_dir)

    runner._finalize_profile_artifacts(task, tmp_path, manifest)

    assert json.loads((profile_dir / "hardware.json").read_text(encoding="utf-8")) == fingerprint
    assert json.loads((profile_dir / "diagnostics.json").read_text(encoding="utf-8")) == fingerprint
    profile = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    assert profile["hardware"] == fingerprint
    assert profile["diagnostics_path"] == "diagnostics.json"
