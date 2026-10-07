from pathlib import Path

import cv2
import numpy as np
import pytest

from palletvision_infer.frame_source import FrameSource, parse_source


def make_video(path: Path, n_frames: int = 30, fps: int = 30) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def test_parse_source_digits_is_camera_index() -> None:
    assert parse_source("0") == 0
    assert parse_source("clip.mp4") == "clip.mp4"


def test_reads_every_frame_by_default(tmp_path: Path) -> None:
    with FrameSource(make_video(tmp_path / "a.mp4")) as src:
        frames = list(src)
    assert len(frames) == 30
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[1].timestamp_s == pytest.approx(1 / 30, abs=1e-3)


def test_fixed_rate_skips_frames(tmp_path: Path) -> None:
    with FrameSource(make_video(tmp_path / "a.mp4"), fps=10) as src:
        frames = list(src)
    assert [f.index for f in frames] == list(range(0, 30, 3))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        FrameSource(tmp_path / "nope.mp4")
