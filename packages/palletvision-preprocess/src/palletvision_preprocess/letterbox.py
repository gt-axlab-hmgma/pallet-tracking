"""Letterbox: fit an image into a square without changing its shape.

The detector takes a fixed ``size x size`` input. The image is scaled so its longer side fits,
and the shorter side is padded with grey. Training and inference must do this identically, or
the model is trained on pictures that differ from the ones it sees in use. So both call the
same ``letterbox_image``; they differ only in what they do with the result:
- inference turns it into a normalised tensor (``letterbox``),
- training keeps the uint8 canvas for augmentation and moves its boxes (``letterbox_boxes``).
"""

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # contracts/model_io.yaml: letterbox_pad_value


@dataclass(frozen=True)
class LetterboxMeta:
    """How the image was placed: boxes move by ``x * scale + pad_x`` (and likewise for y)."""

    scale: float
    pad_x: float
    pad_y: float


def letterbox_image(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping the aspect ratio and centre on a ``size x size`` grey canvas.

    The padding is floored to whole pixels for placement; ``meta`` holds those exact floored
    offsets, so a box mapped with it lands where the pixels actually went. The image keeps its
    channel order (OpenCV's BGR) and dtype.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = int((size - new_h) / 2), int((size - new_w) / 2)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, LetterboxMeta(scale=scale, pad_x=float(left), pad_y=float(top))


def to_blob(canvas: np.ndarray) -> np.ndarray:
    """``[S, S, 3]`` uint8 BGR -> ``[1, 3, S, S]`` float32 RGB in [0, 1], the model's input."""
    rgb = canvas[..., ::-1]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """Inference: letterbox an image and return the model input blob plus how it was placed."""
    canvas, meta = letterbox_image(image, size)
    return to_blob(canvas), meta


def letterbox_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Training: map xyxy boxes from the original image into letterboxed pixels."""
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * meta.scale + meta.pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * meta.scale + meta.pad_y
    return out


def restore_boxes(boxes: np.ndarray, meta: LetterboxMeta, image_hw: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from letterboxed input pixels back to the original image."""
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale
    h, w = image_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out
