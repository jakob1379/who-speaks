from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import ctranslate2
import torch

from st_who_speaks.logging import get_logger

logger = get_logger(__name__)

AMD_ROCM = "AMD ROCm"
NVIDIA_CUDA = "NVIDIA CUDA"
VENDOR_AMD = "0x1002"
VENDOR_NVIDIA = "0x10de"


@dataclass(slots=True)
class AccelerationStatus:
    hardware_available: bool
    torch_gpu_available: bool
    whisper_gpu_available: bool
    backend_label: str | None
    summary: str
    host_gpu_label: str | None = None


@lru_cache(maxsize=1)
def detect_acceleration_status() -> AccelerationStatus:
    host_gpu_label = detect_host_gpu_label()
    torch_gpu_available = bool(torch.cuda.is_available())
    torch_backend = detect_torch_backend_label()
    whisper_gpu_available = detect_whisper_gpu_available(torch_backend, host_gpu_label)
    backend_label = resolve_backend_label(
        torch_backend=torch_backend,
        torch_gpu_available=torch_gpu_available,
        whisper_gpu_available=whisper_gpu_available,
    )
    hardware_available = torch_gpu_available or whisper_gpu_available
    summary = build_acceleration_summary(
        host_gpu_label=host_gpu_label,
        torch_backend=torch_backend,
        backend_label=backend_label,
        hardware_available=hardware_available,
        torch_gpu_available=torch_gpu_available,
        whisper_gpu_available=whisper_gpu_available,
    )

    logger.info(
        "detected acceleration status",
        hardware_available=hardware_available,
        torch_gpu_available=torch_gpu_available,
        whisper_gpu_available=whisper_gpu_available,
        backend_label=backend_label,
        host_gpu_label=host_gpu_label,
    )

    return AccelerationStatus(
        hardware_available=hardware_available,
        torch_gpu_available=torch_gpu_available,
        whisper_gpu_available=whisper_gpu_available,
        backend_label=backend_label,
        summary=summary,
        host_gpu_label=host_gpu_label,
    )


def resolve_execution_settings(*, use_hardware_acceleration: bool) -> dict[str, Any]:
    status = detect_acceleration_status()
    acceleration_enabled = use_hardware_acceleration and status.hardware_available
    transcription_device = resolve_device(
        enabled=acceleration_enabled and status.whisper_gpu_available
    )
    embedding_device = resolve_device(
        enabled=acceleration_enabled and status.torch_gpu_available
    )
    return {
        "hardware_acceleration_enabled": acceleration_enabled,
        "backend_label": status.backend_label,
        "summary": status.summary,
        "transcription_device": transcription_device,
        "transcription_compute_type": resolve_transcription_compute_type(
            transcription_device
        ),
        "embedding_device": embedding_device,
    }


def detect_torch_backend_label() -> str | None:
    if getattr(torch.version, "hip", None):
        return AMD_ROCM
    if getattr(torch.version, "cuda", None):
        return NVIDIA_CUDA
    return None


def detect_host_gpu_label() -> str | None:
    label = None
    if has_amd_runtime_hint():
        label = AMD_ROCM
    elif has_nvidia_runtime_hint():
        label = NVIDIA_CUDA
    else:
        vendor_path = next(Path("/sys/class/drm").glob("card*/device/vendor"), None)
        vendor_id = read_vendor_id(vendor_path) if vendor_path is not None else None
        if vendor_id == VENDOR_AMD:
            label = AMD_ROCM
        elif vendor_id == VENDOR_NVIDIA:
            label = NVIDIA_CUDA
    return label


def has_amd_runtime_hint() -> bool:
    return (
        Path("/dev/kfd").exists()
        or any(Path("/dev/dri").glob("renderD*"))
        and any(path.exists() for path in [Path("/opt/rocm"), Path("/usr/lib/rocm")])
    )


def has_nvidia_runtime_hint() -> bool:
    return any(
        path.exists()
        for path in [
            Path("/dev/nvidia0"),
            Path("/dev/nvidiactl"),
            Path("/proc/driver/nvidia/version"),
        ]
    )


def should_probe_whisper_gpu(
    torch_backend: str | None, host_gpu_label: str | None
) -> bool:
    if torch_backend not in {NVIDIA_CUDA, AMD_ROCM}:
        return False
    if host_gpu_label is None:
        return True
    return host_gpu_label == torch_backend


def detect_whisper_gpu_available(
    torch_backend: str | None, host_gpu_label: str | None
) -> bool:
    if not should_probe_whisper_gpu(torch_backend, host_gpu_label):
        return False
    try:
        whisper_gpu_available = ctranslate2.get_cuda_device_count() > 0
        if whisper_gpu_available:
            ctranslate2.get_supported_compute_types("cuda")
        return whisper_gpu_available
    except Exception:
        return False


def resolve_backend_label(
    *,
    torch_backend: str | None,
    torch_gpu_available: bool,
    whisper_gpu_available: bool,
) -> str | None:
    if torch_gpu_available or whisper_gpu_available:
        return torch_backend or "GPU"
    return None


def build_acceleration_summary(
    *,
    host_gpu_label: str | None,
    torch_backend: str | None,
    backend_label: str | None,
    hardware_available: bool,
    torch_gpu_available: bool,
    whisper_gpu_available: bool,
) -> str:
    summary = f"{backend_label} is available for speaker embeddings. Transcription will remain on CPU."
    if host_gpu_label == AMD_ROCM and torch_backend != AMD_ROCM:
        summary = (
            "AMD GPU appears present, but this Python environment is not ROCm-enabled. "
            "The app will run on CPU until you install a ROCm-enabled PyTorch build. "
            "faster-whisper GPU on AMD also needs a ROCm-enabled CTranslate2 build."
        )
    elif host_gpu_label == NVIDIA_CUDA and torch_backend != NVIDIA_CUDA:
        summary = (
            "NVIDIA GPU appears present, but this Python environment is not CUDA-enabled. "
            "The app will run on CPU until you install CUDA-enabled wheels."
        )
    elif not hardware_available:
        summary = "Hardware acceleration is not available. The app will run on CPU."
    elif whisper_gpu_available and torch_gpu_available:
        summary = (
            f"{backend_label} is available for transcription and speaker embeddings."
        )
    elif whisper_gpu_available:
        summary = f"{backend_label} is available for transcription. Speaker embeddings will use CPU."
    elif backend_label == AMD_ROCM:
        summary = (
            "AMD ROCm is available for speaker embeddings. Whisper transcription will use CPU "
            "unless a ROCm-enabled CTranslate2 build is installed."
        )
    return summary


def resolve_device(*, enabled: bool) -> str:
    return "cuda" if enabled else "cpu"


def resolve_transcription_compute_type(transcription_device: str) -> str:
    return "float16" if transcription_device == "cuda" else "int8"


def read_vendor_id(vendor_path: Path) -> str | None:
    try:
        return vendor_path.read_text().strip().lower()
    except OSError:
        logger.debug("failed to read vendor id", exc_info=True)
        return None
