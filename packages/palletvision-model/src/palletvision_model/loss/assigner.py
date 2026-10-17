"""Label assignment: which grid cells are responsible for which ground-truth object.

The head makes a guess in every cell. Training needs to know which cells should be trained to
predict which object (the "positives"); every other cell is trained to say "nothing here".
This module decides that, per image, from the ground-truth boxes and the model's current
predictions. It is task-aligned: a good positive both classifies the object well and boxes it
well, so cells are ranked by ``score^alpha * IoU^beta``.

YOLO26 uses the same rule for both branches with a different cardinality:
- one-to-many: the best 10 cells per object are positives (dense supervision, training only),
- one-to-one: the best 7 are shortlisted and then only the single best one is kept, so every
  object gets exactly one cell, which is what lets inference skip NMS.

STAL (small-target-aware assignment) changes only which cells may be candidates, see
``candidate_mask``.
"""

from dataclasses import dataclass

import torch
from torch import nn

from palletvision_model.loss.iou import Kind, bbox_iou


@dataclass(frozen=True)
class Assignment:
    """Training targets for every cell. Box and label are meaningful only where ``fg_mask``."""

    target_boxes: torch.Tensor  # [B, A, 4] the ground-truth box each cell must predict
    target_labels: torch.Tensor  # [B, A] its class index
    target_scores: (
        torch.Tensor
    )  # [B, A, nc] soft class target: how good the cell is, 0 if background
    fg_mask: torch.Tensor  # [B, A] bool, True for positive cells
    target_gt_idx: torch.Tensor  # [B, A] which ground-truth box (index along M) each cell took


def candidate_mask(
    anchor_points: torch.Tensor,
    gt_boxes: torch.Tensor,
    *,
    stal: bool = True,
    s_min: float = 8.0,
    s_ref: float = 16.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """``[B, M, A]`` bool: is the cell's centre strictly inside the object's box?

    With STAL, a box side smaller than ``s_min`` (the smallest stride) is treated as ``s_ref``
    (the next stride) for this test only, keeping the box centre. Without it, a box smaller than
    one grid step can sit between cell centres and get no candidates, so the object would never
    be learned. Each side is adjusted on its own. The real box is used everywhere else.
    """
    centre = (gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2
    size = gt_boxes[..., 2:] - gt_boxes[..., :2]
    if stal:
        size = torch.where(size < s_min, torch.full_like(size, s_ref), size)
    top_left = (centre - size / 2)[:, :, None, :]  # [B, M, 1, 2]
    bottom_right = (centre + size / 2)[:, :, None, :]
    points = anchor_points[None, None, :, :]  # [1, 1, A, 2]
    distances = torch.cat((points - top_left, bottom_right - points), dim=-1)  # [B, M, A, 4]
    return torch.amin(distances, dim=-1) > eps


class TaskAlignedAssigner(nn.Module):
    """Pick the positive cells for every ground-truth box. Has no weights and no gradients.

    ``topk`` is the shortlist size per object. ``topk2`` (one-to-one branch only) then keeps
    that many of the shortlisted cells per object by alignment; leave it ``None`` for one-to-many.
    ``iou_kind`` is the overlap measure inside the alignment score and the conflict rule. CIoU
    (clamped at 0) rewards a well-centred, well-shaped box over one that merely overlaps the same
    amount, which is also what the box loss rewards, so the cells chosen are the ones the loss
    can then refine.
    """

    def __init__(
        self,
        topk: int = 10,
        topk2: int | None = None,
        alpha: float = 0.5,
        beta: float = 6.0,
        iou_kind: Kind = "ciou",
        stal: bool = True,
        s_min: float = 8.0,
        s_ref: float = 16.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.topk, self.topk2 = topk, topk2
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.iou_kind: Kind = iou_kind  # its own line: a tuple assignment would widen it to str
        self.stal, self.s_min, self.s_ref = stal, s_min, s_ref

    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Assignment:
        """``pred_scores [B,A,nc]`` (sigmoid), ``pred_boxes [B,A,4]`` (decoded xyxy),
        ``anchor_points [A,2]``, ``gt_labels [B,M,1]``, ``gt_boxes [B,M,4]`` and ``mask_gt [B,M,1]``
        (1 for a real object, 0 for padding). ``M`` is the most objects in any image of the batch.
        """
        with torch.no_grad():
            return self._assign(
                pred_scores.float(), pred_boxes.float(), anchor_points.float(),
                gt_labels, gt_boxes.float(), mask_gt.bool(),
            )  # fmt: skip

    def _assign(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Assignment:
        batch, cells, num_classes = pred_scores.shape
        objects = gt_boxes.shape[1]
        if objects == 0:  # no ground truth anywhere in the batch: everything is background
            return Assignment(
                pred_boxes.new_zeros(batch, cells, 4),
                torch.zeros(batch, cells, dtype=torch.long, device=pred_scores.device),
                pred_scores.new_zeros(batch, cells, num_classes),
                torch.zeros(batch, cells, dtype=torch.bool, device=pred_scores.device),
                torch.zeros(batch, cells, dtype=torch.long, device=pred_scores.device),
            )

        # 1. which cells may be candidates for which object (STAL lives here)
        candidates = (
            candidate_mask(
                anchor_points, gt_boxes, stal=self.stal, s_min=self.s_min, s_ref=self.s_ref
            )
            & mask_gt
        )  # [B, M, A]

        # 2. how well each cell fits each object: class score of the object's class, and box IoU
        labels = gt_labels.long().expand(-1, -1, cells)  # [B, M, A]
        score = pred_scores.transpose(1, 2).gather(1, labels)
        iou = bbox_iou(gt_boxes[:, :, None, :], pred_boxes[:, None, :, :], self.iou_kind)
        iou = iou.clamp(min=0)  # a cell whose box is worse than no overlap cannot be a positive
        align = score.pow(self.alpha) * iou.pow(self.beta) * candidates  # [B, M, A]

        # 3. shortlist the best cells per object (a cell with zero alignment never counts)
        mask_pos = self._best_k(align, self.topk) & (align > 0)

        # 4. a cell wanted by several objects goes to the one it overlaps most
        mask_pos = self._resolve_conflicts(mask_pos, iou)

        # 5. one-to-one: of the cells an object still holds, keep only its very best. Doing this
        # after the conflict step lets an object that lost its favourite cell fall back to its
        # next best, instead of ending up with nothing.
        if self.topk2 is not None:
            mask_pos = self._best_k(align * mask_pos, self.topk2) & mask_pos
        fg_mask = mask_pos.any(dim=1)  # [B, A]
        target_gt_idx = mask_pos.float().argmax(dim=1)  # [B, A], meaningful where fg_mask

        # 6. targets for every cell
        target_labels = gt_labels.squeeze(-1).long().gather(1, target_gt_idx)
        index = target_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
        target_boxes = gt_boxes.gather(1, index)
        target_scores = torch.nn.functional.one_hot(target_labels, num_classes).to(score.dtype)
        target_scores = target_scores * fg_mask.unsqueeze(-1)

        # The class target of a positive is not 1 but how good the cell is: its alignment relative
        # to the best cell of its object, scaled by that object's best IoU. A cell that boxes the
        # object well is told to be confident; a mediocre positive is asked for less.
        align_pos = align * mask_pos
        best_align = align_pos.amax(dim=-1, keepdim=True)
        best_iou = (iou * mask_pos).amax(dim=-1, keepdim=True)
        norm = (align_pos * best_iou / (best_align + self.eps)).amax(dim=1)  # [B, A]
        target_scores = target_scores * norm.unsqueeze(-1)
        return Assignment(target_boxes, target_labels, target_scores, fg_mask, target_gt_idx)

    @staticmethod
    def _best_k(values: torch.Tensor, k: int) -> torch.Tensor:
        """``[B, M, A]`` bool marking the ``k`` largest entries along the cell axis."""
        k = min(k, values.shape[-1])
        indices = torch.topk(values, k, dim=-1).indices
        return torch.zeros_like(values, dtype=torch.bool).scatter(-1, indices, True)

    @staticmethod
    def _resolve_conflicts(mask_pos: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
        """Where a cell is positive for more than one object, keep only the highest-IoU one."""
        objects = mask_pos.shape[1]
        contested = mask_pos.sum(dim=1, keepdim=True) > 1  # [B, 1, A]
        best = (iou * mask_pos).argmax(dim=1)  # [B, A]
        keep = torch.nn.functional.one_hot(best, objects).permute(0, 2, 1).bool()  # [B, M, A]
        return torch.where(contested, keep, mask_pos)
