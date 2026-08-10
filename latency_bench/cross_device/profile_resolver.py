from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from latency_bench.cross_device.instances import resolve_instance_id
from latency_bench.cross_device.spec import detect_hardware
from latency_bench.cross_device.sync import download_profile
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.utils.io import read_json


PROFILE_JSON_RE = re.compile(
    r"^profiles/(?P<device_id>[^/]+)/(?P<instance_id>[^/]+)/(?P<workload_id>[^/]+)/(?P<revision>[^/]+)/profile\.json$"
)


@dataclass(frozen=True)
class ProfileCandidate:
    ref: Path
    device_id: str
    instance_id: str
    workload_id: str
    revision: str
    profile: dict[str, Any] | None = None
    local_profile_path: Path | None = None


@dataclass(frozen=True)
class ResolvedProfile:
    ref: Path
    local_profile_path: Path
    score: int
    match_reasons: tuple[str, ...]
    detected_device_id: str
    detected_instance_id: str
    hardware: dict[str, Any]
    profile: dict[str, Any]


def discover_remote_profile_refs(workload: dict[str, Any]) -> list[ProfileCandidate]:
    """List published profile refs from the configured HF profile repository."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(
        repo_id=workload["hf_repo"],
        repo_type=workload.get("hf_repo_type", "dataset"),
    )
    return profile_candidates_from_repo_files(files)


def profile_candidates_from_repo_files(files: Iterable[str]) -> list[ProfileCandidate]:
    """Parse `profiles/<device>/<instance>/<workload>/<revision>/profile.json` paths."""
    candidates = []
    for filename in files:
        match = PROFILE_JSON_RE.match(str(filename))
        if match is None:
            continue
        parts = match.groupdict()
        ref = Path(parts["device_id"]) / parts["instance_id"] / parts["workload_id"] / parts["revision"]
        candidates.append(
            ProfileCandidate(
                ref=ref,
                device_id=parts["device_id"],
                instance_id=parts["instance_id"],
                workload_id=parts["workload_id"],
                revision=parts["revision"],
            )
        )
    return sorted(candidates, key=lambda item: item.ref.as_posix())


def resolve_best_profile(
    workload: dict[str, Any],
    *,
    hardware: str = "auto",
    workload_id: str | None = None,
    revision: str = "latest",
    download: bool = True,
) -> ResolvedProfile:
    """Resolve the best published profile for the current machine.

    The resolver fingerprints the local GPU, matches it to the configured device
    aliases, lists HF profiles, downloads small profile manifests for compatible
    candidates, then scores exact-instance and same-device matches.
    """
    fingerprint = collect_hardware_fingerprint()
    detected = detect_hardware(workload, hardware)
    device = detected["device"]
    instance_id = resolve_instance_id(fingerprint)
    target_workload_id = workload_id or str(workload["workload"]["id"])
    candidates = [
        candidate
        for candidate in discover_remote_profile_refs(workload)
        if candidate.device_id == device["id"]
        and candidate.workload_id == target_workload_id
        and (revision == "latest" or candidate.revision == revision)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No compatible published profiles found "
            f"for device={device['id']!r}, workload={target_workload_id!r}, revision={revision!r}"
        )

    loaded = [_load_candidate_profile(workload, candidate) for candidate in candidates]
    scored = [
        _score_candidate(candidate, fingerprint=fingerprint, instance_id=instance_id)
        for candidate in loaded
    ]
    scored.sort(
        key=lambda item: (
            item[0],
            _revision_sort_key(item[1].revision),
            item[1].ref.as_posix(),
        ),
        reverse=True,
    )
    score, selected, reasons = scored[0]
    if selected.local_profile_path is None:
        raise RuntimeError(f"profile candidate was not loaded: {selected.ref}")
    local_profile_path = selected.local_profile_path
    if download:
        local_profile_path = download_profile(workload, selected.ref) / "profile.json"
    return ResolvedProfile(
        ref=selected.ref,
        local_profile_path=local_profile_path,
        score=score,
        match_reasons=tuple(reasons),
        detected_device_id=str(device["id"]),
        detected_instance_id=instance_id,
        hardware=fingerprint,
        profile=selected.profile or {},
    )


def _load_candidate_profile(workload: dict[str, Any], candidate: ProfileCandidate) -> ProfileCandidate:
    from huggingface_hub import hf_hub_download

    filename = f"profiles/{candidate.ref.as_posix()}/profile.json"
    path = Path(
        hf_hub_download(
            repo_id=workload["hf_repo"],
            repo_type=workload.get("hf_repo_type", "dataset"),
            filename=filename,
        )
    )
    return ProfileCandidate(
        ref=candidate.ref,
        device_id=candidate.device_id,
        instance_id=candidate.instance_id,
        workload_id=candidate.workload_id,
        revision=candidate.revision,
        profile=read_json(path),
        local_profile_path=path,
    )


def _score_candidate(
    candidate: ProfileCandidate,
    *,
    fingerprint: dict[str, Any],
    instance_id: str,
) -> tuple[int, ProfileCandidate, list[str]]:
    profile = candidate.profile or {}
    profile_hardware = profile.get("hardware") or {}
    profile_gpu = profile_hardware.get("gpu") or {}
    local_gpu = fingerprint.get("gpu") or {}
    score = 0
    reasons: list[str] = []

    if candidate.instance_id == instance_id:
        score += 1000
        reasons.append("same_gpu_uuid_instance")
    if profile.get("device_id") == candidate.device_id:
        score += 100
        reasons.append("same_device_id")
    if _normalized(profile_gpu.get("name")) and _normalized(profile_gpu.get("name")) == _normalized(local_gpu.get("name")):
        score += 50
        reasons.append("same_gpu_name")
    if _normalized(profile_gpu.get("memory.total")) and _normalized(profile_gpu.get("memory.total")) == _normalized(local_gpu.get("memory.total")):
        score += 20
        reasons.append("same_vram")
    if _normalized(profile_gpu.get("compute_cap")) and _normalized(profile_gpu.get("compute_cap")) == _normalized(local_gpu.get("compute_cap")):
        score += 10
        reasons.append("same_compute_cap")
    if profile.get("workload_id") == candidate.workload_id:
        score += 5
        reasons.append("same_workload_id")
    return score, candidate, reasons


def _revision_sort_key(revision: str) -> tuple[int, ...]:
    numbers = tuple(int(value) for value in re.findall(r"\d+", str(revision)))
    return numbers or (-1,)


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
