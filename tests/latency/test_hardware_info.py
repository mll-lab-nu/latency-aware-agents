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


def test_fingerprint_records_all_gpus_and_runtime_cpu_scope(monkeypatch):
    monkeypatch.setattr(hardware_info, "_run", lambda cmd: _fake_multi_gpu_run(cmd))
    monkeypatch.setattr(hardware_info, "_torch_gpu", lambda: {"device_count": 2, "torch_cuda_version": "12.4"})
    monkeypatch.setattr(hardware_info, "_collect_torch", lambda: {"version": "2.6.0", "cuda_version": "12.4"})
    monkeypatch.setattr(hardware_info, "_read_text", lambda path: _fake_read_text(path))
    monkeypatch.setattr(hardware_info.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr(
        hardware_info,
        "_collect_packages",
        lambda: {
            "torch": "2.6.0",
            "transformers": "4.57.0",
            "tokenizers": "0.22.1",
            "safetensors": "0.6.2",
        },
    )

    fingerprint = hardware_info.collect_hardware_fingerprint()

    assert fingerprint["gpu"]["uuid"] == "GPU-0"
    assert [gpu["uuid"] for gpu in fingerprint["gpus"]] == ["GPU-0", "GPU-1"]
    assert fingerprint["gpus"][1]["pci.bus_id"] == "00000000:8B:00.0"
    assert fingerprint["runtime"]["cpu_affinity"] == [0, 1, 2, 3]
    assert fingerprint["runtime"]["cgroup"]["cpu_max"] == "200000 100000"
    assert fingerprint["runtime"]["cgroup"]["cpuset_cpus_effective"] == "0-3"
    assert fingerprint["packages"]["tokenizers"] == "0.22.1"
    assert fingerprint["packages"]["safetensors"] == "0.6.2"


def test_participating_gpu_descriptor_canonicalizes_order_and_topology(monkeypatch):
    gpus = [
        {"uuid": "GPU-b", "name": "NVIDIA L40S"},
        {"uuid": "GPU-a", "name": "NVIDIA L40S"},
        {"uuid": "GPU-c", "name": "NVIDIA L40S"},
    ]
    topology = (
        "GPU0 GPU1 GPU2 CPU Affinity NUMA Affinity\n"
        "GPU0 X SYS PXB 0-3 0\n"
        "GPU1 SYS X NV1 4-7 0\n"
        "GPU2 PXB NV1 X 8-11 1\n"
    )

    descriptor = hardware_info.describe_participating_gpus(
        gpus=gpus,
        topology=topology,
        cuda_visible_devices="2,0",
    )

    assert descriptor == {
        "gpu_class": "2x-l40s",
        "gpus": [
            {"uuid": "GPU-b", "name": "NVIDIA L40S", "cuda_index": 1},
            {"uuid": "GPU-c", "name": "NVIDIA L40S", "cuda_index": 0},
        ],
        "topology_links": [
            {"gpu_a": "GPU-b", "gpu_b": "GPU-c", "link": "PXB"},
        ],
    }

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    current = hardware_info.describe_participating_gpus(
        gpus=gpus,
        topology=topology,
    )
    assert [gpu["uuid"] for gpu in current["gpus"]] == ["GPU-a", "GPU-b"]


def test_public_projection_keeps_only_deployment_identity():
    projection = hardware_info.public_hardware_projection(
        {
            "gpus": [
                {
                    "uuid": "GPU-0",
                    "name": "NVIDIA GeForce RTX 4090",
                    "driver_version": "570.1",
                },
            ],
            "gpu": {"topology": "GPU0 X\n", "driver_version": "570.1"},
            "torch": {"version": "2.6.0", "cuda_version": "12.6"},
        },
        cuda_visible_devices="0",
    )

    assert projection == {
        "gpu_class": "1x-rtx4090",
        "gpus": [{"slot": 0, "name": "NVIDIA GeForce RTX 4090"}],
        "topology_links": [],
        "driver_version": "570.1",
        "torch": {"version": "2.6.0", "cuda_version": "12.6"},
    }


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
    assert hardware_info._nvidia_smi_gpus() == []


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


def _fake_multi_gpu_run(cmd: list[str]) -> str | None:
    if cmd[0] == "nvidia-smi":
        if "--query-gpu=" in cmd[1]:
            return "\n".join(
                [
                    "GPU-0, NVIDIA L40S, 46068 MiB, 100 MiB, 45968 MiB, 350.00 W, 350.00 W, "
                    "570.211.01, 00000000:8A:00.0, 4, 4, 16, 16, 2520 MHz, 2100 MHz, "
                    "9000 MHz, 8.9, P0, 55, 280.00 W, 95 %, 40 %",
                    "GPU-1, NVIDIA L40S, 46068 MiB, 10 MiB, 46058 MiB, 350.00 W, 350.00 W, "
                    "570.211.01, 00000000:8B:00.0, 4, 1, 16, 16, 2520 MHz, 210 MHz, "
                    "405 MHz, 8.9, P8, 30, 36.00 W, 0 %, 0 %",
                ]
            )
        if cmd[1:] == ["topo", "-m"]:
            return "GPU0 GPU1 CPU Affinity NUMA Affinity\nGPU0 X PXB 0-3 0\nGPU1 PXB X 4-7 1"
    if cmd[0] == "lscpu":
        return "Model name:  AMD EPYC 9334\nCPU(s):  128\nNUMA node(s):  2\n"
    return None


def _fake_read_text(path: str) -> str:
    return {
        "/sys/fs/cgroup/cpu.max": "200000 100000\n",
        "/sys/fs/cgroup/cpuset.cpus.effective": "0-3\n",
    }[path]


def test_profile_workers_preserve_logical_devices_and_shared_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c,GPU-b,GPU-a")
    fingerprint = {
        "gpus": [{"uuid": f"GPU-{name}", "name": "NVIDIA Future GPU"} for name in "abc"],
        "gpu": {"driver_version": "1"}, "torch": {}, "cpu": {}, "memory": {},
    }
    deployment = hardware_info.describe_inference_deployment(fingerprint, ["cuda:2", "cuda:2"])
    assert deployment["gpu_class"] == "1x-nvidia-future-gpu"
    assert deployment["gpus"] == [{"uuid": "GPU-a", "name": "NVIDIA Future GPU", "cuda_index": 2}]
    assert deployment["workers"] == [
        {"slot": i, "device": "cuda:2"} for i in range(2)
    ]
    public = hardware_info.public_hardware_projection(fingerprint, deployment=deployment)
    assert public["workers"] == [{"slot": i, "device": "cuda:2"} for i in range(2)]
    assert "GPU-a" not in str(public)


def test_cpu_profile_does_not_include_visible_gpus_or_transient_frequency(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fingerprint = {"gpus": [{"uuid": "GPU-visible", "name": "unused"}], "gpu": {},
                   "cpu": {"model_name": "cpu", "cpu_mhz": "1000"}, "system": {"hostname": "host"}}
    first = hardware_info.describe_inference_deployment(fingerprint, ["cpu", "cpu"])
    fingerprint["cpu"]["cpu_mhz"] = "2000"
    second = hardware_info.describe_inference_deployment(fingerprint, ["cpu", "cpu"])
    assert first == second
    assert first["gpu_class"] == "cpu"
    assert first["gpus"] == []
    assert len(first["workers"]) == 2
