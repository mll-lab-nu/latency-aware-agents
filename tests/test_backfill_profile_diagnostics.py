from __future__ import annotations

import json

from scripts.cross_device import backfill_profile_diagnostics as backfill


def test_apply_backfill_writes_profile_and_manifest_diagnostics(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles" / "openvla-flappy" / "flappy" / "L_L40S"
    run_dir = tmp_path / "raw" / "l40s" / "instances" / "instance_abc" / "openvla" / "cross_device" / "runs" / "measure" / "openvla-flappy" / "flappy" / "l40s" / "best_seed_0_n100"
    profile_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (profile_dir / "profile.json").write_text(json.dumps({"profile_id": "L_L40S"}) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    fingerprint = {
        "gpu": {"name": "NVIDIA L40S", "clocks.current.sm": "1800 MHz"},
        "torch": {"version": "2.6.0"},
        "packages": {"flash_attn": "2.7.4"},
    }
    monkeypatch.setattr(backfill, "collect_hardware_fingerprint", lambda: fingerprint)

    written = backfill.apply_backfill(profile_dir=profile_dir, run_dir=run_dir)

    assert written == {
        "diagnostics_path": str(profile_dir / "diagnostics.json"),
        "hardware_path": str(profile_dir / "hardware.json"),
        "manifest_path": str(run_dir / "manifest.json"),
        "profile_path": str(profile_dir / "profile.json"),
    }
    assert json.loads((profile_dir / "hardware.json").read_text(encoding="utf-8")) == fingerprint
    assert json.loads((profile_dir / "diagnostics.json").read_text(encoding="utf-8")) == fingerprint
    assert json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))["hardware"] == fingerprint
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["hardware_fingerprint"] == fingerprint
