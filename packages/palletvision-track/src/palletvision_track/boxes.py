"""Box maths for the tracker: the two box formats it uses and pairwise IoU.

Detections arrive as ``(x1, y1, x2, y2)`` corners. The Kalman filter works on ``(x, y, a, h)``:
box centre, aspect ratio ``a = w / h`` and height. Everything is vectorised over a leading ``N``
axis and returns float64, the precision the filter runs in.
"""

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

# Heights below this are treated as this, so a degenerate box gives a huge ``a`` instead of inf/nan.
_MIN_H = 1e-6


def xyxy_to_xyah(boxes: npt.ArrayLike) -> Array:
    """``(N, 4)`` corners -> ``(N, 4)`` ``(centre x, centre y, w / h, h)``."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    w = b[:, 2] - b[:, 0]
    h = np.maximum(b[:, 3] - b[:, 1], _MIN_H)
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] + w / 2
    out[:, 1] = b[:, 1] + h / 2
    out[:, 2] = w / h
    out[:, 3] = h
    return out


def xyah_to_xyxy(xyah: npt.ArrayLike) -> Array:
    """``(N, 4)`` ``(centre x, centre y, w / h, h)`` -> ``(N, 4)`` corners."""
    b = np.asarray(xyah, dtype=np.float64).reshape(-1, 4)
    w = b[:, 2] * b[:, 3]
    half_w, half_h = w / 2, b[:, 3] / 2
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - half_w
    out[:, 1] = b[:, 1] - half_h
    out[:, 2] = b[:, 0] + half_w
    out[:, 3] = b[:, 1] + half_h
    return out


def pairwise_iou(a: npt.ArrayLike, b: npt.ArrayLike) -> Array:
    """IoU of every box in ``a`` (N, 4) against every box in ``b`` (M, 4) -> ``(N, M)``.

    Boxes are corners. Broadcasting ``a[:, None]`` against ``b[None, :]`` builds all N*M pairs
    in one go, with no Python loop.
    """
    a_ = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b_ = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    # overlap rectangle: the larger of the top-left corners to the smaller of the bottom-right
    top_left = np.maximum(a_[:, None, :2], b_[None, :, :2])
    bottom_right = np.minimum(a_[:, None, 2:], b_[None, :, 2:])
    inter_wh = np.clip(bottom_right - top_left, 0, None)  # disjoint pairs clamp to zero area
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = np.prod(np.clip(a_[:, 2:] - a_[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b_[:, 2:] - b_[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    # two zero-area boxes give 0 / 0; call that no overlap
    iou: Array = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > 0)
    return iou
