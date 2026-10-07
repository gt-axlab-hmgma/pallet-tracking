from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    def factory(name: str = "clip.mp4", n_frames: int = 30, fps: int = 30) -> Path:
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (64, 48))
        for i in range(n_frames):
            writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
        writer.release()
        return path

    return factory
