from __future__ import annotations

import builtins
import sys
import tempfile
import types
from pathlib import Path

from st_who_speaks.dependency_compat import _ensure_huggingface_hub_compatibility
from st_who_speaks.diarization import load_speaker_embedding_model


def test_load_speaker_embedding_model_patches_torchaudio_before_import(
    monkeypatch,
) -> None:
    load_speaker_embedding_model.cache_clear()

    fake_torchaudio = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "speechbrain.inference.classifiers":
            assert hasattr(fake_torchaudio, "list_audio_backends")
            speechbrain_pkg = sys.modules.setdefault(
                "speechbrain", types.ModuleType("speechbrain")
            )
            inference_pkg = sys.modules.setdefault(
                "speechbrain.inference", types.ModuleType("speechbrain.inference")
            )
            classifiers_pkg = types.ModuleType("speechbrain.inference.classifiers")

            class FakeEncoderClassifier:
                @classmethod
                def from_hparams(cls, **kwargs):
                    return {"kwargs": kwargs}

            setattr(classifiers_pkg, "EncoderClassifier", FakeEncoderClassifier)
            setattr(speechbrain_pkg, "inference", inference_pkg)
            setattr(inference_pkg, "classifiers", classifiers_pkg)
            sys.modules[name] = classifiers_pkg
            return classifiers_pkg
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = load_speaker_embedding_model("cpu")

    assert hasattr(fake_torchaudio, "list_audio_backends")
    assert fake_torchaudio.list_audio_backends() == ["soundfile", "sox_io"]
    assert model["kwargs"]["run_opts"] == {"device": "cpu"}
    assert model["kwargs"]["source"] == "speechbrain/spkrec-ecapa-voxceleb"


def test_huggingface_hub_compatibility_maps_use_auth_token(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class RemoteEntryNotFoundError(Exception):
        pass

    def original_hf_hub_download(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        if kwargs.get("filename") == "custom.py":
            raise RemoteEntryNotFoundError("missing custom.py")
        return "ok"

    fake_huggingface_hub = types.ModuleType("huggingface_hub")
    fake_file_download = types.SimpleNamespace(hf_hub_download=original_hf_hub_download)
    fake_hub_any = fake_huggingface_hub
    setattr(fake_hub_any, "hf_hub_download", original_hf_hub_download)
    setattr(fake_hub_any, "file_download", fake_file_download)

    monkeypatch.setattr(
        "st_who_speaks.dependency_compat.importlib.import_module",
        lambda name: fake_huggingface_hub if name == "huggingface_hub" else None,
    )

    _ensure_huggingface_hub_compatibility()
    stub_path = getattr(fake_huggingface_hub, "hf_hub_download")(
        "repo", filename="custom.py", use_auth_token="token"
    )
    assert stub_path == str(
        Path(tempfile.gettempdir())
        / "st-who-speaks"
        / "speechbrain-compat"
        / "custom.py"
    )

    getattr(fake_huggingface_hub, "hf_hub_download")(
        "repo", filename="hyperparams.yaml", use_auth_token="token"
    )

    assert captured["args"] == ("repo",)
    assert captured["kwargs"] == {"filename": "hyperparams.yaml", "token": "token"}
