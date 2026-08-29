"""Realtime inference subprocess program."""
from __future__ import annotations

import gc
import traceback
from copy import deepcopy
from typing import Any

from latency_bench.core.timing import StageProfiler, profiler_scope
from latency_bench.executors.realtime.messages import RealtimePolicyResult
from latency_bench.executors.realtime.shared_observation import (
    ObservationAttachmentCache,
    ObservationLocalizer,
)
from latency_bench.executors.realtime.timing import monotonic_ms
from latency_bench.latency.hardware_info import safe_set, torch_backend_flags
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
from training.common.utils import seed_everything


# --- inference worker ---------------------------------------------------------

def _worker_runtime_metadata(worker_id: int, start_method: str) -> dict[str, Any]:
    import torch

    metadata = {
        "worker_id": int(worker_id),
        "worker_start_method": start_method,
        "worker_torch_version": torch.__version__,
        "worker_torch_cuda_version": torch.version.cuda,
        "worker_cuda_device_count": torch.cuda.device_count(),
    }
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

        seed_everything(config["experiment"]["seed"] + worker_id)
        runtime_metadata = _worker_runtime_metadata(worker_id, start_method)
        worker_config = deepcopy(config)
        worker_config.setdefault("policy", {})["device"] = device
        worker_config.setdefault("latency", {})["device"] = device
        action_resolver = ActionResolver(worker_config["env"])
        policy = build_policy(worker_config, action_resolver)
        latency_cfg = worker_config.get("latency", {})
        measurer = RuntimeLatencyMeasurer(
            device=device,
            sync_cuda=bool(latency_cfg.get("sync_cuda", True)),
        )
        attachment_cache = ObservationAttachmentCache(blocks={})
        observation_localizer = ObservationLocalizer()
        # The policy's object graph is permanent; freeze it out of GC scans and
        # stop automatic collections so gen-2 pauses can't delay a result send.
        gc.freeze()
        gc.disable()
        connection.send(("ready", {"worker_id": int(worker_id), "device": device}))

        def predict_from_attached_observation(attached_observation, profile):
            localize_start_ms = monotonic_ms()
            local_observation = observation_localizer.localize(attached_observation.observation)
            if profile is not None:
                profile.worker_observation_localize_ms = monotonic_ms() - localize_start_ms
            # sync_cuda=False: the realtime path measures end-to-end wall-clock
            # latency, so a CUDA sync inside predict would perturb the measurement.
            profiler = StageProfiler(enabled=profile is not None, sync_cuda=False)
            with profiler_scope(profiler):
                policy_output = policy.predict(local_observation)
            if profile is not None:
                timings = profiler.timings
                profile.policy_build_example_ms = timings.get("policy_build_example_ms", 0.0)
                profile.policy_wrapper_predict_action_ms = timings.get(
                    "policy_wrapper_predict_action_ms", 0.0
                )
                profile.policy_decode_ms = timings.get("policy_decode_ms", 0.0)
            return policy_output

        while True:
            command, payload = connection.recv()
            worker_recv_ms = monotonic_ms()
            payload = payload or {}
            if command == "close":
                attachment_cache.close()
                return
            if command == "collect":
                # Sent by the pool only while every worker is idle (post-drain,
                # pre-episode), so this pause can never delay an inference.
                gc.collect()
                continue
            if command == "predict":
                task = payload["task"]
                attach_start_ms = monotonic_ms()
                attached_observation = attachment_cache.attach(task.observation)
                attach_ms = monotonic_ms() - attach_start_ms
                profile = task.profile
                if profile is not None:
                    profile.worker_recv_ms = worker_recv_ms
                    profile.parent_to_worker_ipc_ms = (
                        worker_recv_ms - profile.parent_to_worker_send_start_ms
                    )
                    inference_start_ms = monotonic_ms()
                    profile.worker_observation_attach_ms = attach_ms
                    profile.worker_recv_overhead_ms = inference_start_ms - worker_recv_ms
                try:
                    policy_output, latency_record = measurer.run_and_measure(
                        predict_from_attached_observation,
                        attached_observation,
                        profile,
                        obs_id=task.obs_id,
                    )
                finally:
                    attached_observation.close()
                latency_record.metadata = {
                    **latency_record.metadata,
                    **runtime_metadata,
                }
                result = RealtimePolicyResult(
                    policy_output=policy_output,
                    completion_time_ms=task.submit_time_ms + latency_record.latency_ms,
                    latency_record=latency_record,
                    profile=profile,
                )
                try:
                    if result.profile is not None:
                        result.profile.worker_result_send_start_ms = monotonic_ms()
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
