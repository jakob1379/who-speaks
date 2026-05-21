from __future__ import annotations

import importlib
import inspect
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np

PIPELINE_TEMP_DIR_NAME = "st-who-speaks"
TORCHAUDIO_MODULE = "torchaudio"
HUGGINGFACE_HUB_MODULE = "huggingface_hub"
HUGGINGFACE_HUB_HF_HUB_DOWNLOAD = "hf_hub_download"
HUGGINGFACE_HUB_USE_AUTH_TOKEN = "use_auth_token"
TORCHAUDIO_GET_AUDIO_BACKEND = "get_audio_backend"
TORCHAUDIO_SET_AUDIO_BACKEND = "set_audio_backend"
TORCHAUDIO_LIST_AUDIO_BACKENDS = "list_audio_backends"

OPENCV_IMPORT_ERROR: Exception | None = None

try:
    import cv2
except ImportError as error:
    OPENCV_IMPORT_ERROR = error

    def _fallback_video_capture(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            isOpened=lambda: False,
            get=lambda *_args, **_kwargs: 0.0,
            release=lambda: None,
            read=lambda: (False, None),
            set=lambda *_args, **_kwargs: False,
        )

    def _fallback_cascade_classifier(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            empty=lambda: True,
            detectMultiScale=lambda *_args, **_kwargs: [],
        )

    class _FallbackCv2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_POS_MSEC = 0
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        COLOR_BGR2GRAY = 0
        COLOR_BGR2RGB = 1
        VideoCapture = staticmethod(_fallback_video_capture)
        CascadeClassifier = staticmethod(_fallback_cascade_classifier)

        @staticmethod
        def cvtColor(frame: Any, *_args: Any, **_kwargs: Any) -> Any:
            return frame

        @staticmethod
        def rectangle(*_args: Any, **_kwargs: Any) -> None:
            return None

        @staticmethod
        def circle(*_args: Any, **_kwargs: Any) -> None:
            return None

        @staticmethod
        def line(*_args: Any, **_kwargs: Any) -> None:
            return None

        @staticmethod
        def imencode(*_args: Any, **_kwargs: Any) -> tuple[bool, np.ndarray]:
            return True, np.frombuffer(b"fallback", dtype=np.uint8)

    cv2: Any = _FallbackCv2()


def _import_optional_module(module_name: str) -> Any | None:
    module: Any | None = None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = None
    return module


def is_opencv_available() -> bool:
    return OPENCV_IMPORT_ERROR is None


def get_opencv_import_error_message() -> str | None:
    if OPENCV_IMPORT_ERROR is None:
        return None
    return str(OPENCV_IMPORT_ERROR)


def _ensure_torchaudio_compatibility() -> None:
    torchaudio = _import_optional_module(TORCHAUDIO_MODULE)
    if torchaudio is None:
        return

    if not hasattr(torchaudio, TORCHAUDIO_GET_AUDIO_BACKEND):
        setattr(torchaudio, TORCHAUDIO_GET_AUDIO_BACKEND, lambda: None)
    if not hasattr(torchaudio, TORCHAUDIO_SET_AUDIO_BACKEND):
        setattr(
            torchaudio, TORCHAUDIO_SET_AUDIO_BACKEND, lambda *_args, **_kwargs: None
        )
    if hasattr(torchaudio, TORCHAUDIO_LIST_AUDIO_BACKENDS):
        return

    def list_audio_backends() -> list[str]:
        backends: list[str] = []
        get_audio_backend = getattr(torchaudio, TORCHAUDIO_GET_AUDIO_BACKEND, None)
        if callable(get_audio_backend):
            try:
                backend = get_audio_backend()
            except Exception:
                backend = None
            if backend:
                backends.append(str(backend))
        if not backends:
            backends.extend(["soundfile", "sox_io"])
        return backends

    setattr(torchaudio, TORCHAUDIO_LIST_AUDIO_BACKENDS, list_audio_backends)


def _hf_hub_download_supports_use_auth_token(
    hf_hub_download: Callable[..., Any],
) -> bool:
    try:
        signature = inspect.signature(hf_hub_download)
    except (TypeError, ValueError):
        return False
    return HUGGINGFACE_HUB_USE_AUTH_TOKEN in signature.parameters


def _fallback_custom_module_path() -> Path:
    empty_custom_module = (
        Path(tempfile.gettempdir())
        / PIPELINE_TEMP_DIR_NAME
        / "speechbrain-compat"
        / "custom.py"
    )
    empty_custom_module.parent.mkdir(parents=True, exist_ok=True)
    if not empty_custom_module.exists():
        descriptor, temporary_name = tempfile.mkstemp(dir=empty_custom_module.parent)
        os.close(descriptor)
        Path(temporary_name).replace(empty_custom_module)
    return empty_custom_module


def _normalize_hf_hub_download_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized_kwargs = dict(kwargs)
    if (
        HUGGINGFACE_HUB_USE_AUTH_TOKEN in normalized_kwargs
        and "token" not in normalized_kwargs
    ):
        normalized_kwargs["token"] = normalized_kwargs.pop(
            HUGGINGFACE_HUB_USE_AUTH_TOKEN
        )
    else:
        normalized_kwargs.pop(HUGGINGFACE_HUB_USE_AUTH_TOKEN, None)
    return normalized_kwargs


def _patch_huggingface_hub_download(huggingface_hub: Any) -> None:
    # Older SpeechBrain model-loading code passes use_auth_token to
    # huggingface_hub.hf_hub_download; newer hub versions expect token instead.
    # Keep this translation at the dependency boundary.
    hf_hub_download = getattr(huggingface_hub, HUGGINGFACE_HUB_HF_HUB_DOWNLOAD, None)
    if not callable(hf_hub_download):
        return
    hf_hub_download_func = cast(Callable[..., Any], hf_hub_download)

    if _hf_hub_download_supports_use_auth_token(hf_hub_download_func):
        return

    def compat_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
        filename = kwargs.get("filename")
        if filename is None and len(args) >= 2:
            filename = args[1]
        normalized_kwargs = _normalize_hf_hub_download_kwargs(kwargs)
        try:
            return hf_hub_download_func(*args, **normalized_kwargs)
        except Exception as error:
            if filename == "custom.py" and error.__class__.__name__ in {
                "RemoteEntryNotFoundError",
                "EntryNotFoundError",
                "HTTPStatusError",
            }:
                return str(_fallback_custom_module_path())
            raise

    setattr(huggingface_hub, HUGGINGFACE_HUB_HF_HUB_DOWNLOAD, compat_hf_hub_download)

    file_download = getattr(huggingface_hub, "file_download", None)
    if file_download is not None and hasattr(
        file_download, HUGGINGFACE_HUB_HF_HUB_DOWNLOAD
    ):
        setattr(file_download, HUGGINGFACE_HUB_HF_HUB_DOWNLOAD, compat_hf_hub_download)


def _ensure_huggingface_hub_compatibility() -> None:
    huggingface_hub = _import_optional_module(HUGGINGFACE_HUB_MODULE)
    if huggingface_hub is None:
        return

    _patch_huggingface_hub_download(huggingface_hub)
