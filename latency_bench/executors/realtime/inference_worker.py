"""Realtime inference subprocess program."""
from __future__ import annotations

import gc
import traceback
from copy import deepcopy
from importlib.util import find_spec
from typing import Any

from latency_bench.executors.realtime.messages import RealtimePolicyResult
from latency_bench.executors.realtime.shared_observation import (
    ObservationAttachmentCache,
    ObservationLocalizer,
)
from latency_bench.latency.hardware_info import safe_set, torch_backend_flags
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
from latency_bench.utils.seeding import seed_everything


# --- inference worker ---------------------------------------------------------

def _worker_runtime_metadata(worker_id: int, start_method: str) -> dict[str, Any]:
    metadata = {
        "worker_id": int(worker_id),
        "worker_start_method": start_method,
    }
    if find_spec("torch") is None:
        return metadata

    import torch

    metadata.update({
        "worker_torch_version": torch.__version__,
        "worker_torch_cuda_version": torch.version.cuda,
        "worker_cuda_device_count": torch.cuda.device_count(),
    })
    safe_set(metadata, "worker_cuda_current_device", lambda: int(torch.cuda.current_device()))
    safe_set(
        metadata,
        "worker_cuda_current_device_name",
        lambda: str(torch.cuda.get_device_name(torch.cuda.current_device())),
    )
    backend_flags = torch_backend_flags(torch)
    for flag_key in ("cuda_flash_sdp_enabled", "cuda_mem_efficient_sdp_enabled", "cuda_math_sdp_enabled"):
        if flag_key in backend_flags:
            metadata[f"worker_{flag_key}"] = backend_flags[flag_key]
    safe_set(metadata, "worker_torch_float32_matmul_precision", torch.get_float32_matmul_precision)
    return metadata


def _inference_worker(connection: Any, config: dict[str, Any], worker_id: int, device: str, start_method: str) -> None:
    gc_was_enabled = gc.isenabled()
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.policy.factory import build_policy

        if device.startswith("cuda"):
            import torch

            torch.cuda.set_device(device)
        seed_everything(config["experiment"]["seed"] + worker_id)
        runtime_metadata = _worker_runtime_metadata(worker_id, start_method)
        worker_config = deepcopy(config)
        worker_config["executor"]["inference_devices"] = [device]
        action_resolver = ActionResolver(worker_config["env"])
        policy = build_policy(worker_config, action_resolver)
        measurer = RuntimeLatencyMeasurer(
            device=device,
            sync_cuda=worker_config["executor"]["sync_cuda"],
        )
        attachment_cache = ObservationAttachmentCache(blocks={})
        observation_localizer = ObservationLocalizer()
        # The policy's object graph is permanent; freeze it out of GC scans and
        # stop automatic collections so gen-2 pauses can't delay a result send.
        gc.freeze()
        gc.disable()
        connection.send(("ready", {"worker_id": int(worker_id), "device": device}))

        def predict_from_attached_observation(attached_observation):
            local_observation = observation_localizer.localize(attached_observation)
            policy_output = policy.predict(local_observation)
            return policy_output

        while True:
            command, payload = connection.recv()
            payload = payload or {}
            if command == "close":
                attachment_cache.close()
                return
            if command == "collect":
                # Sent by the pool only while every worker is idle (post-drain,
                # pre-episode), so this pause can never delay an inference.
                gc.collect()
                continue
            if command == "predict_batch":
                policy_outputs = policy.predict_batch(payload["observations"])
                connection.send(
                    ("batch_result", {"policy_outputs": policy_outputs})
                )
                continue
            if command == "reset_state":
                policy.reset_state(slot_id=payload["slot_id"])
                connection.send(("state_reset", {}))
                continue
            if command == "predict":
                task = payload["task"]
                attached_observation = attachment_cache.attach(task.observation)
                policy_output, latency_record = measurer.run_and_measure(
                    predict_from_attached_observation,
                    attached_observation,
                    obs_id=task.obs_id,
                )
                latency_record.metadata = {
                    **latency_record.metadata,
                    **runtime_metadata,
                }
                result = RealtimePolicyResult(
                    policy_output=policy_output,
                    completion_time_ms=task.submit_time_ms + latency_record.latency_ms,
                    latency_record=latency_record,
                )
                try:
                    connection.send(("result", {"result": result}))
                except (BrokenPipeError, EOFError, OSError):
                    return
    except BaseException:
        try:
            connection.send(("error", {"traceback": traceback.format_exc()}))
        except (BrokenPipeError, EOFError, OSError):
            return
    finally:
        # The real worker process exits with this function; restoring GC state
        # only matters for in-process callers (tests).
        gc.unfreeze()
        if gc_was_enabled:
            gc.enable()
