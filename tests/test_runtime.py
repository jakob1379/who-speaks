from __future__ import annotations

from st_who_speaks.runtime import AccelerationStatus, resolve_execution_settings


def test_resolve_execution_settings_respects_available_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_acceleration_status",
        lambda: AccelerationStatus(
            hardware_available=True,
            torch_gpu_available=True,
            whisper_gpu_available=False,
            backend_label="NVIDIA CUDA",
            summary="CUDA is available for embeddings only.",
            host_gpu_label="NVIDIA CUDA",
        ),
    )

    enabled = resolve_execution_settings(use_hardware_acceleration=True)
    disabled = resolve_execution_settings(use_hardware_acceleration=False)

    assert enabled.hardware_acceleration_enabled is True
    assert enabled.embedding_device == "cuda"
    assert enabled.transcription_device == "cpu"
    assert disabled.hardware_acceleration_enabled is False
    assert disabled.embedding_device == "cpu"


def test_detect_host_gpu_label_checks_all_drm_vendor_paths(monkeypatch) -> None:
    from st_who_speaks import runtime

    first_vendor = object()
    second_vendor = object()

    class FakePath:
        def __init__(self, value: str):
            self.value = value

        def glob(self, pattern: str):
            assert self.value == "/sys/class/drm"
            assert pattern == "card*/device/vendor"
            return iter([first_vendor, second_vendor])

    monkeypatch.setattr("st_who_speaks.runtime.has_amd_runtime_hint", lambda: False)
    monkeypatch.setattr("st_who_speaks.runtime.has_nvidia_runtime_hint", lambda: False)
    monkeypatch.setattr("st_who_speaks.runtime.Path", FakePath)
    monkeypatch.setattr(
        "st_who_speaks.runtime.read_vendor_id",
        lambda vendor_path: None
        if vendor_path is first_vendor
        else runtime.VENDOR_NVIDIA,
    )

    assert runtime.detect_host_gpu_label() == runtime.NVIDIA_CUDA


def test_detect_acceleration_status_supports_rocm_embeddings(monkeypatch) -> None:
    from st_who_speaks import runtime

    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_host_gpu_label", lambda: "AMD ROCm"
    )
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_torch_backend_label", lambda: "AMD ROCm"
    )
    monkeypatch.setattr("st_who_speaks.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "st_who_speaks.runtime.ctranslate2.get_cuda_device_count", lambda: 0
    )

    status = runtime.detect_acceleration_status()

    assert status.hardware_available is True
    assert status.torch_gpu_available is True
    assert status.whisper_gpu_available is False
    assert status.backend_label == "AMD ROCm"
    assert "Whisper transcription will use CPU" in status.summary


def test_detect_acceleration_status_reports_amd_host_without_rocm_env(
    monkeypatch,
) -> None:
    from st_who_speaks import runtime

    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_host_gpu_label", lambda: "AMD ROCm"
    )
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_torch_backend_label", lambda: None
    )
    monkeypatch.setattr("st_who_speaks.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "st_who_speaks.runtime.ctranslate2.get_cuda_device_count", lambda: 0
    )

    status = runtime.detect_acceleration_status()

    assert status.hardware_available is False
    assert status.host_gpu_label == "AMD ROCm"
    assert "not ROCm-enabled" in status.summary


def test_detect_whisper_gpu_available_logs_probe_failure(monkeypatch) -> None:
    from st_who_speaks import runtime

    log_events: list[dict[str, object]] = []

    def fail_device_count() -> int:
        raise RuntimeError("driver not initialized")

    monkeypatch.setattr(
        "st_who_speaks.runtime.ctranslate2.get_cuda_device_count",
        fail_device_count,
    )
    monkeypatch.setattr(
        "st_who_speaks.runtime.logger.warning",
        lambda event, **kwargs: log_events.append({"event": event, **kwargs}),
    )

    assert (
        runtime.detect_whisper_gpu_available(
            runtime.NVIDIA_CUDA,
            runtime.NVIDIA_CUDA,
        )
        is False
    )
    assert log_events == [
        {
            "event": "whisper gpu probe failed; falling back to CPU transcription",
            "torch_backend": runtime.NVIDIA_CUDA,
            "host_gpu_label": runtime.NVIDIA_CUDA,
            "error": "driver not initialized",
            "exc_info": True,
        }
    ]
