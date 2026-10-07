from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # contracts/model_io.yaml: letterbox_pad_value


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping aspect ratio, pad to a square, return a 1x3xSxS float32 RGB blob in [0, 1]."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = int(pad_y), int(pad_x)
    canvas[top : top + new_h, left : left + new_w] = resized
    rgb = canvas[..., ::-1]
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return blob, LetterboxMeta(scale=scale, pad_x=float(left), pad_y=float(top))


def restore_boxes(boxes: np.ndarray, meta: LetterboxMeta, image_hw: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from letterboxed input pixels back to the original image."""
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale
    h, w = image_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out
