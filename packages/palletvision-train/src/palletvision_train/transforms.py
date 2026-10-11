"""Per-sample image transforms for training."""

import numpy as np

from palletvision_preprocess import letterbox_boxes, letterbox_image


def letterbox_sample(
    image: np.ndarray, labels: np.ndarray, size: int = 640
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an image and its labels into a ``size x size`` square, exactly as inference does.

    ``labels`` is ``[n, 5]``: class, then x1 y1 x2 y2 in the original image's pixels. Returns the
    uint8 BGR canvas (still un-normalised, so augmentation can work on it) and the labels with
    their boxes moved to match. Uses the same ``letterbox_image`` as the inference path.
    """
    canvas, meta = letterbox_image(image, size)
    moved = labels.copy()
    if len(moved):
        moved[:, 1:5] = letterbox_boxes(moved[:, 1:5], meta)
    return canvas, moved
