from dataclasses import dataclass
from typing import Protocol

import numpy as np

from palletvision_infer.media.preprocess import LetterboxMeta, restore_boxes


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy in original image pixels
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) int

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.scores)


class Detector(Protocol):
    """Every backend (torch, onnx, later tensorrt) returns the same Detections."""

    def detect(self, image: np.ndarray) -> Detections: ...


def decode(
    raw: np.ndarray, meta: LetterboxMeta, image_hw: tuple[int, int], conf: float
) -> Detections:
    """Turn the contract output [1, K, 6] (x1, y1, x2, y2, score, class) into Detections."""
    rows = raw[0]
    rows = rows[rows[:, 4] >= conf]
    if len(rows) == 0:
        return Detections.empty()
    return Detections(
        boxes=restore_boxes(rows[:, :4], meta, image_hw),
        scores=rows[:, 4].astype(np.float32),
        classes=rows[:, 5].astype(np.int64),
    )
