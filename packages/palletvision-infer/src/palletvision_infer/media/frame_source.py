from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_s: float
    image: np.ndarray  # BGR, HxWx3 uint8


def parse_source(value: str) -> int | str:
    """A bare integer means a camera index (laptop webcam is 0); anything else is a path."""
    return int(value) if value.isdigit() else value


class FrameSource:
    """Yields frames from a video file or webcam, optionally thinned to a fixed rate."""

    def __init__(self, source: int | str | Path, fps: float | None = None) -> None:
        self._cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not self._cap.isOpened():
            raise OSError(f"cannot open video source: {source}")
        self.source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._step = 1
        if fps and self.source_fps > fps:
            self._step = max(1, round(self.source_fps / fps))

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            if index % self._step == 0:
                timestamp = index / self.source_fps if self.source_fps else 0.0
                yield Frame(index, timestamp, image)
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
