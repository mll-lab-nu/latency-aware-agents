from __future__ import annotations

import latency_bench.latency.hardware_info as hardware_info


def test_fingerprint_is_serializable_and_complete(monkeypatch):
    monkeypatch.setattr(
        hardware_info,
        "_run",
        lambda cmd: _fake_run(cmd),
    )
    monkeypatch.setattr(hardware_info, "_torch_gpu", lambda: {"device_count": 1, "torch_cuda_version": "12.4"})
    monkeypatch.setattr(hardware_info, "_collect_torch", lambda: {"version": "2.6.0", "cuda_version": "12.4", "backends": {"cuda_flash_sdp_enabled": True}})
    monkeypatch.setattr(hardware_info, "_collect_packages", lambda: {"flash_attn": "2.7.4"})

    fingerprint = hardware_info.collect_hardware_fingerprint()

    assert fingerprint["gpu"]["name"] == "NVIDIA B200"
    assert fingerprint["gpu"]["power.limit"] == "1000.00 W"
    assert fingerprint["gpu"]["pcie.link.width.current"] == "16x"
    assert fingerprint["gpu"]["clocks.current.sm"] == "1800 MHz"
    assert fingerprint["gpu"]["device_count"] == 1
    assert fingerprint["cpu"]["model_name"] == "AMD EPYC 9554"
    assert fingerprint["cpu"]["numa_nodes"] == "2"
    assert fingerprint["cpu"]["logical_cores"] is not None
    assert "system" in fingerprint
    assert fingerprint["runtime"]["python_executable"]
    assert fingerprint["torch"]["backends"]["cuda_flash_sdp_enabled"] is True
    assert fingerprint["packages"]["flash_attn"] == "2.7.4"
    assert isinstance(fingerprint["captured_at"], str)


def test_probes_degrade_when_tools_missing(monkeypatch):
    monkeypatch.setattr(hardware_info, "_run", lambda cmd: None)  # no nvidia-smi / lscpu
    monkeypatch.setattr(hardware_info, "_torch_gpu", lambda: {})
    monkeypatch.setattr(hardware_info, "_cpuinfo_model", lambda: None)

    fingerprint = hardware_info.collect_hardware_fingerprint()

    assert fingerprint["gpu"] == {}
    # logical_cores still comes from os.cpu_count(); the call must not raise.
    assert "logical_cores" in fingerprint["cpu"]
    assert "runtime" in fingerprint
    assert "torch" in fingerprint
    assert "packages" in fingerprint


def test_nvidia_smi_field_count_mismatch_yields_empty(monkeypatch):
    monkeypatch.setattr(hardware_info, "_run", lambda cmd: "only, three, fields")
    assert hardware_info._nvidia_smi_gpu() == {}


def _fake_run(cmd: list[str]) -> str | None:
    if cmd[0] == "nvidia-smi":
        if "--query-gpu=" in cmd[1]:
            return (
                "GPU-test, NVIDIA B200, 183 GiB, 10 MiB, 182 GiB, 1000.00 W, 1000.00 W, "
                "560.35.05, 00000000:8A:00.0, 5, 5, 16x, 16x, 1965 MHz, 1800 MHz, "
                "405 MHz, 10.0, P0, 65 C, 450.00 W, 75 %, 10 %"
            )
        if cmd[1:] == ["topo", "-m"]:
            return "GPU0 CPU Affinity NUMA Affinity\nGPU0 X 0-63 0"
    if cmd[0] == "lscpu":
        return "Model name:  AMD EPYC 9554\nCPU(s):  128\nCPU max MHz:  3762.9880\nNUMA node(s):  2\nL3 cache:  256 MiB\n"
    return None
