import time
from pathlib import Path

import cv2
import numpy as np

from palletvision_infer.detections import Detections, Detector
from palletvision_infer.media.frame_source import FrameSource
from palletvision_infer.media.overlay import draw


class DummyDetector:
    """Returns a fixed centred box. Verifies video and webcam plumbing before a model exists."""

    def detect(self, image: np.ndarray) -> Detections:
        h, w = image.shape[:2]
        box = np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=np.float32)
        return Detections(box, np.array([1.0], np.float32), np.array([0], np.int64))


def run(
    source: int | str | Path,
    detector: Detector,
    out_path: Path | None = None,
    fps: float | None = None,
    show: bool = False,
) -> int:
    """Detect on every (thinned) frame, optionally writing and/or displaying the overlay."""
    writer: cv2.VideoWriter | None = None
    count = 0
    last = time.perf_counter()
    try:
        with FrameSource(source, fps=fps) as frames:
            out_fps = fps or frames.source_fps or 30.0
            for frame in frames:
                det = detector.detect(frame.image)
                now = time.perf_counter()
                annotated = draw(frame.image, det, fps=1.0 / max(now - last, 1e-6))
                last = now
                if out_path is not None:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), out_fps, (w, h)
                        )
                    writer.write(annotated)
                if show:
                    cv2.imshow("palletvision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                count += 1
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return count
