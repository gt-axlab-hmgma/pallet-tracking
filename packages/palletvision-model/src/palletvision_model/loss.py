"""The training loss: how wrong the model's guesses are, given who the assigner holds responsible.

For each branch (one-to-many and one-to-one) the steps are the same:
1. decode the raw per-cell distances into boxes,
2. ask the assigner which cells are positive for which object (it never gets gradients),
3. score the model on three things: the class scores of ALL cells, and the box of each POSITIVE
   cell, measured twice (CIoU and an L1 distance).
The two branches are then blended with a weight that moves from the one-to-many branch to the
one-to-one branch as training goes on (Progressive Loss).
"""

import torch
from torch import nn

from palletvision_model.assigner import TaskAlignedAssigner
from palletvision_model.decode import decode_boxes, make_anchors
from palletvision_model.iou import bbox_iou

# the model's raw output per branch: (box distances [B, A, 4], class logits [B, A, nc])
Predictions = dict[str, tuple[torch.Tensor, torch.Tensor]]


def progressive_alpha(epoch: int, epochs: int, init: float = 0.8, final: float = 0.1) -> float:
    """Weight of the one-to-many branch at ``epoch``: falls linearly from ``init`` to ``final``.

    ``alpha(t) = max(1 - t / max(E - 1, 1), 0) * (init - final) + final`` (YOLO26 report, Eq. 3).
    """
    progress = max(1 - epoch / max(epochs - 1, 1), 0.0)
    return progress * (init - final) + final


class DetectionLoss(nn.Module):
    """Class + box (CIoU) + L1 loss for both branches, blended by ``alpha``."""

    def __init__(
        self,
        strides: tuple[int, ...] = (8, 16, 32),
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        l1_gain: float = 6.0,
        one2many: TaskAlignedAssigner | None = None,
        one2one: TaskAlignedAssigner | None = None,
    ) -> None:
        super().__init__()
        self.strides = strides
        self.box_gain, self.cls_gain, self.l1_gain = box_gain, cls_gain, l1_gain
        self.assigners = {
            "one2many": one2many or TaskAlignedAssigner(topk=10),
            "one2one": one2one or TaskAlignedAssigner(topk=7, topk2=1),
        }

    def forward(
        self,
        preds: Predictions,
        image_hw: tuple[int, int],
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        mask_gt: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the total loss and each part (detached) for logging.

        ``preds`` holds the ``one2many`` and ``one2one`` outputs of ``YOLO26.raw``. ``gt_*`` are the
        padded labels (``[B, M, 1]``, ``[B, M, 4]``, ``[B, M, 1]``) in input-image pixels.
        ``alpha`` is the one-to-many weight, from ``progressive_alpha``.
        """
        height, width = image_hw
        sizes = [(height // s, width // s) for s in self.strides]
        device = preds["one2one"][0].device
        points, strides = make_anchors(sizes, self.strides, device=device)

        total = preds["one2one"][0].new_zeros(())
        parts: dict[str, torch.Tensor] = {}
        for name, weight in (("one2many", alpha), ("one2one", 1 - alpha)):
            distances, logits = preds[name]
            losses = self._branch(
                distances, logits, points, strides, image_hw, self.assigners[name],
                gt_labels, gt_boxes, mask_gt,
            )  # fmt: skip
            total = total + weight * losses["total"]
            parts.update({f"{key}_{name}": value.detach() for key, value in losses.items()})
        return total, parts

    def _branch(
        self,
        distances: torch.Tensor,
        logits: torch.Tensor,
        points: torch.Tensor,
        strides: torch.Tensor,
        image_hw: tuple[int, int],
        assigner: TaskAlignedAssigner,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = logits.float()
        boxes = decode_boxes(distances.float(), points, strides)  # [B, A, 4], keeps gradients

        # who is responsible for what: decided from detached copies, never trained through
        target = assigner(
            logits.detach().sigmoid(), boxes.detach(), points, gt_labels, gt_boxes, mask_gt
        )

        # Everything is divided by the total target score, not by the number of objects or cells:
        # per-cell targets are soft, and this keeps the loss the same size however many images
        # or cells there are.
        n_targets = target.target_scores.sum().clamp(min=1)

        # classes: every cell is scored, so background cells are pushed towards "nothing here"
        cls = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target.target_scores, reduction="sum"
            )
            / n_targets
        )

        # boxes: only positive cells, each weighted by how good the assigner rated it
        fg = target.fg_mask
        weight = target.target_scores.sum(-1)[fg]  # [N]
        pred_box, true_box = boxes[fg], target.target_boxes[fg]
        box = (weight * (1 - bbox_iou(pred_box, true_box, "ciou"))).sum() / n_targets
        height, width = image_hw
        size = boxes.new_tensor([width, height, width, height])  # makes the L1 scale free
        l1 = (weight * ((pred_box - true_box) / size).abs().mean(-1)).sum() / n_targets

        total = self.box_gain * box + self.cls_gain * cls + self.l1_gain * l1
        return {"box": box, "cls": cls, "l1": l1, "total": total}
