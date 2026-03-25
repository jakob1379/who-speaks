from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import ctranslate2
import torch

from st_who_speaks.logging import get_logger

logger = get_logger(__name__)


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
    whisper_gpu_available = False
    backend_label = None
    torch_backend = detect_torch_backend_label()

    if should_probe_whisper_gpu(torch_backend, host_gpu_label):
        try:
            whisper_gpu_available = ctranslate2.get_cuda_device_count() > 0
            if whisper_gpu_available:
                ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            whisper_gpu_available = False

    if torch_gpu_available or whisper_gpu_available:
        backend_label = torch_backend or "GPU"

    hardware_available = torch_gpu_available or whisper_gpu_available
    if host_gpu_label == "AMD ROCm" and torch_backend != "AMD ROCm":
        summary = (
            "AMD GPU appears present, but this Python environment is not ROCm-enabled. "
            "The app will run on CPU until you install a ROCm-enabled PyTorch build. "
            "faster-whisper GPU on AMD also needs a ROCm-enabled CTranslate2 build."
        )
    elif host_gpu_label == "NVIDIA CUDA" and torch_backend != "NVIDIA CUDA":
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
    elif backend_label == "AMD ROCm":
        summary = (
            "AMD ROCm is available for speaker embeddings. Whisper transcription will use CPU "
            "unless a ROCm-enabled CTranslate2 build is installed."
        )
    else:
        summary = f"{backend_label} is available for speaker embeddings. Transcription will remain on CPU."

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


def resolve_execution_settings(use_hardware_acceleration: bool) -> dict[str, Any]:
    status = detect_acceleration_status()
    acceleration_enabled = use_hardware_acceleration and status.hardware_available
    transcription_device = (
        "cuda" if acceleration_enabled and status.whisper_gpu_available else "cpu"
    )
    embedding_device = (
        "cuda" if acceleration_enabled and status.torch_gpu_available else "cpu"
    )
    return {
        "hardware_acceleration_enabled": acceleration_enabled,
        "backend_label": status.backend_label,
        "summary": status.summary,
        "transcription_device": transcription_device,
        "transcription_compute_type": "float16"
        if transcription_device == "cuda"
        else "int8",
        "embedding_device": embedding_device,
    }


def detect_torch_backend_label() -> str | None:
    if getattr(torch.version, "hip", None):
        return "AMD ROCm"
    if getattr(torch.version, "cuda", None):
        return "NVIDIA CUDA"
    return None


def detect_host_gpu_label() -> str | None:
    if has_amd_runtime_hint():
        return "AMD ROCm"
    if has_nvidia_runtime_hint():
        return "NVIDIA CUDA"
    vendor_path = next(Path("/sys/class/drm").glob("card*/device/vendor"), None)
    if vendor_path is None:
        return None
    try:
        vendor_id = vendor_path.read_text().strip().lower()
    except OSError:
        return None
    if vendor_id == "0x1002":
        return "AMD ROCm"
    if vendor_id == "0x10de":
        return "NVIDIA CUDA"
    return None


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
    if torch_backend not in {"NVIDIA CUDA", "AMD ROCm"}:
        return False
    if host_gpu_label is None:
        return True
    return host_gpu_label == torch_backend
