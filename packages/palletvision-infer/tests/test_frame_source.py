from collections.abc import Callable
from pathlib import Path

import pytest

from palletvision_infer.frame_source import FrameSource, parse_source


def test_parse_source_digits_is_camera_index() -> None:
    assert parse_source("0") == 0
    assert parse_source("clip.mp4") == "clip.mp4"


def test_reads_every_frame_by_default(make_video: Callable[..., Path]) -> None:
    with FrameSource(make_video()) as src:
        frames = list(src)
    assert len(frames) == 30
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[1].timestamp_s == pytest.approx(1 / 30, abs=1e-3)


def test_fixed_rate_skips_frames(make_video: Callable[..., Path]) -> None:
    with FrameSource(make_video(), fps=10) as src:
        frames = list(src)
    assert [f.index for f in frames] == list(range(0, 30, 3))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        FrameSource(tmp_path / "nope.mp4")
