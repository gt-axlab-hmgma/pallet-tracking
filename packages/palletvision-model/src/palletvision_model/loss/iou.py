"""Box overlap measures: IoU and its three refinements (GIoU, DIoU, CIoU).

Boxes are ``(x1, y1, x2, y2)`` in the last dimension. Everything broadcasts, so ``[N, 4]`` with
``[N, 4]`` gives ``[N]`` (each box against its own partner, as the loss needs) and
``pairwise_iou`` gives ``[N, M]`` (every box against every other, as the assigner needs).
Run these in float32 even when the network runs in half precision.
"""

import math
from typing import Literal

import torch

Kind = Literal["iou", "giou", "diou", "ciou"]


def bbox_iou(
    a: torch.Tensor, b: torch.Tensor, kind: Kind = "iou", eps: float = 1e-7
) -> torch.Tensor:
    """Overlap of ``a`` and ``b`` over their last dimension, broadcasting the rest.

    - ``iou``: intersection over union, 0 to 1. It is 0 for every pair of disjoint boxes, so it
      gives no hint which way to move.
    - ``giou``: IoU minus the share of the smallest enclosing box that the union leaves empty.
      Negative for disjoint boxes, and larger as they get closer.
    - ``diou``: IoU minus (distance between centres)^2 / (enclosing box diagonal)^2.
    - ``ciou``: DIoU minus a penalty for a different aspect ratio (width / height).
    """
    a1, a2 = a[..., :2], a[..., 2:]
    b1, b2 = b[..., :2], b[..., 2:]

    inter_wh = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    # an untrained network can output x2 < x1, so clamp the sides before taking areas
    wh_a = (a2 - a1).clamp(min=0)
    wh_b = (b2 - b1).clamp(min=0)
    union = wh_a[..., 0] * wh_a[..., 1] + wh_b[..., 0] * wh_b[..., 1] - inter + eps
    iou = inter / union
    if kind == "iou":
        return iou

    enclose_wh = torch.maximum(a2, b2) - torch.minimum(a1, b1)  # the smallest box holding both
    if kind == "giou":
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] + eps
        return iou - (enclose_area - union) / enclose_area

    diag2 = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2 + eps
    centre_dist2 = (((a1 + a2) - (b1 + b2)) ** 2).sum(-1) / 4  # centre = (corner1 + corner2) / 2
    diou = iou - centre_dist2 / diag2
    if kind == "diou":
        return diou

    v = (4 / math.pi**2) * (
        torch.atan(wh_b[..., 0] / (wh_b[..., 1] + eps))
        - torch.atan(wh_a[..., 0] / (wh_a[..., 1] + eps))
    ) ** 2
    alpha = (v / ((1 - iou) + v + eps)).detach()  # a weight, not something to optimise
    return diou - alpha * v


def pairwise_iou(a: torch.Tensor, b: torch.Tensor, kind: Kind = "iou") -> torch.Tensor:
    """``[N, 4]`` against ``[M, 4]`` -> ``[N, M]``."""
    return bbox_iou(a[:, None, :], b[None, :, :], kind)
