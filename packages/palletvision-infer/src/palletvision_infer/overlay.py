import cv2
import numpy as np

from palletvision_infer.detections import Detections

GREEN = (0, 200, 0)


def draw(image: np.ndarray, det: Detections, fps: float | None = None) -> np.ndarray:
    """Return a copy of ``image`` with boxes, scores and an optional fps counter."""
    out = image.copy()
    for (x1, y1, x2, y2), score in zip(det.boxes, det.scores, strict=True):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
        cv2.putText(
            out, f"pallet {score:.2f}", (int(x1), max(12, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA,
        )  # fmt: skip
    if fps is not None:
        cv2.putText(
            out, f"{fps:.1f} fps", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA
        )
    return out
