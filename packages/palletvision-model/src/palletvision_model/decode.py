"""Turn the head's per-cell distances into boxes in input pixels.

The head says, for every grid cell, "my object's edges are this far from my centre". To get a
box you need two things the head does not carry: where each cell's centre is, and how many
pixels one distance unit is (the level's stride). Both come from the grid, not from learning.
"""

import torch


def make_anchors(
    sizes: list[tuple[int, int]],
    strides: tuple[int, ...] = (8, 16, 32),
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell centres and strides for feature maps of the given ``(height, width)`` sizes.

    Returns ``anchor_points [A, 2]`` as ``(x, y)`` in input pixels and ``anchor_strides [A, 1]``.
    Cells are ordered exactly as ``head.flatten_levels`` orders them: finest level first, and
    inside a level row by row, left to right. Row *i* of the output belongs to the same cell as
    row *i* of the head's output; if the orders ever differ, boxes get the wrong centres.
    """
    points, level_strides = [], []
    for (height, width), stride in zip(sizes, strides, strict=True):
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        centres = torch.stack((xs, ys), dim=-1).reshape(-1, 2).float() + 0.5  # cell centre
        points.append(centres * stride)
        level_strides.append(torch.full((height * width, 1), float(stride), device=device))
    return torch.cat(points), torch.cat(level_strides)


def decode_boxes(
    distances: torch.Tensor, anchor_points: torch.Tensor, anchor_strides: torch.Tensor
) -> torch.Tensor:
    """``[B, A, 4]`` distances ``(l, t, r, b)`` in stride units -> ``[B, A, 4]`` ``x1 y1 x2 y2``.

    No activation is applied to the distances: the head starts them positive (bias 2.0) and
    training keeps them so. The result is not clamped to the image; that happens only at
    inference, so the loss still sees (and can correct) boxes that stick out of the image.
    """
    left_top, right_bottom = distances[..., :2], distances[..., 2:]
    top_left = anchor_points - left_top * anchor_strides
    bottom_right = anchor_points + right_bottom * anchor_strides
    return torch.cat((top_left, bottom_right), dim=-1)


def postprocess(
    boxes: torch.Tensor,
    cls_logits: torch.Tensor,
    image_hw: tuple[int, int],
    max_det: int = 300,
) -> torch.Tensor:
    """Pick the best ``max_det`` detections: ``[B, A, 4]`` boxes and ``[B, A, nc]`` logits in,
    ``[B, max_det, 6]`` out, each row ``x1 y1 x2 y2 score class``.

    There is no NMS and no score threshold. The one-to-one head already gives one confident
    cell per object, so taking the highest scores is enough; callers filter by confidence
    afterwards. The output always has exactly ``max_det`` rows (the contract fixes the shape),
    padded with all-zero rows when there are fewer candidates.
    """
    num_classes = cls_logits.shape[-1]
    scores = cls_logits.sigmoid().flatten(1)  # [B, A * nc]: one entry per (cell, class) pair
    k = min(max_det, scores.shape[1])
    top_scores, top_idx = torch.topk(scores, k, dim=1)  # sorted, highest first
    cell = torch.div(top_idx, num_classes, rounding_mode="floor")
    cls = top_idx - cell * num_classes  # the remainder, written without % for clean export

    top_boxes = boxes.gather(1, cell.unsqueeze(-1).expand(-1, -1, 4))
    height, width = image_hw
    limit = boxes.new_tensor([width, height, width, height])
    top_boxes = torch.minimum(top_boxes.clamp(min=0), limit)  # clamp to the image

    out = torch.cat((top_boxes, top_scores.unsqueeze(-1), cls.to(boxes.dtype).unsqueeze(-1)), -1)
    return torch.nn.functional.pad(out, (0, 0, 0, max_det - k))
