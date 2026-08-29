from latency_bench.cross_device.instances import instance_id_from_gpu_deployment


def test_instance_id_uses_all_gpu_uuids_and_topology_links():
    deployment = {
        "gpus": [
            {"uuid": "GPU-a", "name": "NVIDIA L40S", "cuda_index": 0},
            {"uuid": "GPU-b", "name": "NVIDIA L40S", "cuda_index": 1},
        ],
        "topology_links": [{"gpu_a": "GPU-a", "gpu_b": "GPU-b", "link": "NV1"}],
    }
    changed_gpu = {
        **deployment,
        "gpus": [
            {"uuid": "GPU-a", "name": "NVIDIA L40S", "cuda_index": 0},
            {"uuid": "GPU-c", "name": "NVIDIA L40S", "cuda_index": 1},
        ],
    }
    changed_topology = {
        **deployment,
        "topology_links": [{"gpu_a": "GPU-a", "gpu_b": "GPU-b", "link": "SYS"}],
    }

    assert instance_id_from_gpu_deployment(deployment) != instance_id_from_gpu_deployment(changed_gpu)
    assert instance_id_from_gpu_deployment(deployment) != instance_id_from_gpu_deployment(changed_topology)


def test_instance_id_ignores_non_identity_descriptor_fields():
    deployment = {
        "gpus": [
            {"uuid": "GPU-a", "name": "NVIDIA L40S", "cuda_index": 0},
            {"uuid": "GPU-b", "name": "NVIDIA L40S", "cuda_index": 1},
        ],
        "topology_links": [{"gpu_a": "GPU-a", "gpu_b": "GPU-b", "link": "NV1"}],
    }
    renamed = {
        **deployment,
        "gpu_class": "2x-l40s",
        "extra_telemetry": {"temperature": [40, 41]},
    }

    assert instance_id_from_gpu_deployment(deployment) == instance_id_from_gpu_deployment(renamed)


def test_instance_id_is_invariant_to_descriptor_order():
    deployment = {
        "gpus": [
            {"uuid": "GPU-a", "name": "NVIDIA L40S", "cuda_index": 0},
            {"uuid": "GPU-b", "name": "NVIDIA L40S", "cuda_index": 1},
        ],
        "topology_links": [{"gpu_a": "GPU-a", "gpu_b": "GPU-b", "link": "NV1"}],
    }
    reordered = {
        "gpus": list(reversed(deployment["gpus"])),
        "topology_links": [{"gpu_a": "GPU-b", "gpu_b": "GPU-a", "link": "NV1"}],
    }

    assert instance_id_from_gpu_deployment(deployment) == instance_id_from_gpu_deployment(reordered)
