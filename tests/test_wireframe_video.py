from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess

import numpy as np

from st_who_speaks import frame_assets
from st_who_speaks.frame_assets import _build_wireframe_video_from_capture
from st_who_speaks.models import FaceDetectionFrame


class FakeWireframeCapture:
    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        frame_count: int = 1,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.remaining_frames = frame_count

    def get(self, prop: int) -> float:
        if prop == frame_assets.cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == frame_assets.cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == frame_assets.cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.remaining_frames <= 0:
            return False, None
        self.remaining_frames -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class FakeWireframeStdin:
    def __init__(self, *, broken_pipe: bool = False) -> None:
        self.broken_pipe = broken_pipe
        self.write_count = 0

    def __enter__(self) -> FakeWireframeStdin:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def write(self, _payload: bytes) -> None:
        self.write_count += 1
        if self.broken_pipe:
            raise BrokenPipeError


@dataclass(frozen=True, slots=True)
class FakeWireframeProcessConfig:
    return_code: int = 0
    stderr_text: str = ""
    target_bytes: bytes | None = b"wireframe"
    timeout: bool = False
    broken_pipe: bool = False


class FakeWireframeProcess:
    def __init__(
        self,
        command: list[str],
        stderr,
        *,
        config: FakeWireframeProcessConfig,
    ) -> None:
        self.command = command
        self.config = config
        self.stdin = FakeWireframeStdin(broken_pipe=config.broken_pipe)
        self.killed = False
        self.wait_count = 0
        if config.stderr_text:
            stderr.write(config.stderr_text.encode())
            stderr.flush()

    def wait(self, timeout: int | None = None) -> int:
        self.wait_count += 1
        if self.config.timeout and not self.killed:
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.config.target_bytes is not None:
            Path(self.command[-1]).write_bytes(self.config.target_bytes)
        return self.config.return_code

    def kill(self) -> None:
        self.killed = True


def install_fake_wireframe_process(
    monkeypatch,
    *,
    config: FakeWireframeProcessConfig = FakeWireframeProcessConfig(),
) -> list[FakeWireframeProcess]:
    processes: list[FakeWireframeProcess] = []

    def fake_popen(command, stdin, stderr):
        assert stdin is subprocess.PIPE
        process = FakeWireframeProcess(
            command,
            stderr,
            config=config,
        )
        processes.append(process)
        return process

    monkeypatch.setattr("st_who_speaks.frame_assets.subprocess.Popen", fake_popen)
    return processes


def set_wireframe_face_count(monkeypatch, face_count: int) -> None:
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.detect_and_annotate_faces_with_frame",
        lambda frame, **_kwargs: (
            FaceDetectionFrame(face_count=face_count),
            np.zeros_like(frame),
        ),
    )


def test_build_wireframe_video_returns_empty_for_invalid_capture_dimensions(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert _build_wireframe_video_from_capture(
        FakeWireframeCapture(width=0, height=480),
        original_media_path="clip.webm",
        working_dir=tmp_path,
    ) == (None, 0)


def test_build_wireframe_video_defaults_invalid_fps_to_24(
    monkeypatch, tmp_path: Path
) -> None:
    processes = install_fake_wireframe_process(monkeypatch)
    set_wireframe_face_count(monkeypatch, 1)

    _build_wireframe_video_from_capture(
        FakeWireframeCapture(fps=math.nan),
        original_media_path="clip.webm",
        working_dir=tmp_path,
    )

    command = processes[0].command
    assert command[command.index("-r") + 1] == "24.000000"


def test_build_wireframe_video_returns_target_bytes_when_faces_detected(
    monkeypatch, tmp_path: Path
) -> None:
    install_fake_wireframe_process(
        monkeypatch,
        config=FakeWireframeProcessConfig(target_bytes=b"processed"),
    )
    set_wireframe_face_count(monkeypatch, 2)

    assert _build_wireframe_video_from_capture(
        FakeWireframeCapture(frame_count=2),
        original_media_path="clip.webm",
        working_dir=tmp_path,
    ) == (b"processed", 4)


def test_build_wireframe_video_returns_empty_when_no_faces_detected(
    monkeypatch, tmp_path: Path
) -> None:
    install_fake_wireframe_process(
        monkeypatch,
        config=FakeWireframeProcessConfig(target_bytes=b"processed"),
    )
    set_wireframe_face_count(monkeypatch, 0)

    assert _build_wireframe_video_from_capture(
        FakeWireframeCapture(),
        original_media_path="clip.webm",
        working_dir=tmp_path,
    ) == (None, 0)


def test_build_wireframe_video_returns_count_when_target_missing(
    monkeypatch, tmp_path: Path
) -> None:
    install_fake_wireframe_process(
        monkeypatch,
        config=FakeWireframeProcessConfig(target_bytes=None),
    )
    set_wireframe_face_count(monkeypatch, 3)

    assert _build_wireframe_video_from_capture(
        FakeWireframeCapture(),
        original_media_path="clip.webm",
        working_dir=tmp_path,
    ) == (None, 3)


def test_build_wireframe_video_raises_nonzero_stderr(
    monkeypatch, tmp_path: Path
) -> None:
    install_fake_wireframe_process(
        monkeypatch,
        config=FakeWireframeProcessConfig(
            return_code=1,
            stderr_text="encoder failed",
        ),
    )
    set_wireframe_face_count(monkeypatch, 1)

    try:
        _build_wireframe_video_from_capture(
            FakeWireframeCapture(),
            original_media_path="clip.webm",
            working_dir=tmp_path,
        )
    except RuntimeError as error:
        assert str(error) == "encoder failed"
    else:
        raise AssertionError("nonzero ffmpeg return code should fail")


def test_build_wireframe_video_kills_timed_out_process_with_stderr(
    monkeypatch, tmp_path: Path
) -> None:
    processes = install_fake_wireframe_process(
        monkeypatch,
        config=FakeWireframeProcessConfig(
            stderr_text="still encoding",
            timeout=True,
        ),
    )
    set_wireframe_face_count(monkeypatch, 1)

    try:
        _build_wireframe_video_from_capture(
            FakeWireframeCapture(),
            original_media_path="clip.webm",
            working_dir=tmp_path,
        )
    except RuntimeError as error:
        assert "timed out after 600s" in str(error)
        assert "still encoding" in str(error)
    else:
        raise AssertionError("timed out ffmpeg process should fail")

    assert processes[0].killed is True
    assert processes[0].wait_count == 2


def test_build_wireframe_video_handles_broken_pipe_and_waits(
    monkeypatch, tmp_path: Path
) -> None:
    processes = install_fake_wireframe_process(
        monkeypatch,
        config=FakeWireframeProcessConfig(broken_pipe=True),
    )
    set_wireframe_face_count(monkeypatch, 1)

    assert _build_wireframe_video_from_capture(
        FakeWireframeCapture(frame_count=3),
        original_media_path="clip.webm",
        working_dir=tmp_path,
    ) == (b"wireframe", 1)

    assert processes[0].stdin.write_count == 1
    assert processes[0].wait_count == 1
