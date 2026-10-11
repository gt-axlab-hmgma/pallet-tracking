import pytest
import torch

from palletvision_model.assigner import TaskAlignedAssigner
from palletvision_model.config import SIZES
from palletvision_model.decode import decode_boxes, make_anchors
from palletvision_model.iou import bbox_iou
from palletvision_model.loss import DetectionLoss, Predictions, progressive_alpha
from palletvision_model.yolo26 import YOLO26

IMAGE = (64, 64)  # a 64 x 64 image has 8 x 8, 4 x 4 and 2 x 2 cells: 84 in all
CELLS = 84
POINTS, STRIDES = make_anchors([(8, 8), (4, 4), (2, 2)])
GT = [16.0, 16.0, 32.0, 32.0]
# distances that decode, from cell 27 (centre (28, 28), stride 8), to exactly the GT box
PERFECT = [1.5, 1.5, 0.5, 0.5]


def labels(boxes: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(boxes)
    return (
        torch.zeros(1, n, 1, dtype=torch.long),
        torch.tensor(boxes)[None],
        torch.ones(1, n, 1, dtype=torch.long),
    )


def branch(
    distance_27: list[float] | None = None, logit_27: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.full((1, CELLS, 4), 2.0)  # the head's starting value
    logits = torch.full((1, CELLS, 1), -5.0)
    if distance_27 is not None:
        distances[0, 27] = torch.tensor(distance_27)
        logits[0, 27, 0] = logit_27
    return distances, logits


def same_in_both(distance_27: list[float] | None) -> Predictions:
    dist, logit = branch(distance_27)
    return {"one2many": (dist, logit), "one2one": (dist.clone(), logit.clone())}


def test_progressive_alpha_follows_the_schedule() -> None:
    assert progressive_alpha(0, 11) == pytest.approx(0.8)
    assert progressive_alpha(5, 11) == pytest.approx(0.45)  # half way: (0.8 + 0.1) / 2
    assert progressive_alpha(10, 11) == pytest.approx(0.1)
    assert progressive_alpha(50, 11) == pytest.approx(0.1)  # never below the final value
    assert progressive_alpha(0, 1) == pytest.approx(0.8)  # a one-epoch run does not divide by 0
    values = [progressive_alpha(t, 100) for t in range(100)]
    assert values == sorted(values, reverse=True)  # only ever falls


def test_with_no_objects_there_is_no_box_loss_and_class_loss_pushes_to_background() -> None:
    nothing = (torch.zeros(1, 0, 1, dtype=torch.long), torch.zeros(1, 0, 4), torch.zeros(1, 0, 1))
    _, parts = DetectionLoss()(same_in_both(None), IMAGE, *nothing, alpha=0.5)
    for name in ("one2many", "one2one"):
        assert parts[f"box_{name}"].item() == 0 and parts[f"l1_{name}"].item() == 0
    expected = CELLS * torch.nn.functional.softplus(torch.tensor(-5.0))  # BCE of logit -5 vs 0
    assert parts["cls_one2one"].item() == pytest.approx(expected.item(), rel=1e-5)


def test_a_perfect_box_costs_nothing_in_the_one_to_one_branch() -> None:
    _, parts = DetectionLoss()(same_in_both(PERFECT), IMAGE, *labels([GT]), alpha=0.5)
    assert parts["box_one2one"].item() == pytest.approx(0, abs=1e-6)
    assert parts["l1_one2one"].item() == pytest.approx(0, abs=1e-6)
    assert parts["box_one2many"].item() < 1e-3  # weaker positives carry almost no weight


def test_a_wrong_box_costs_ciou_and_l1_by_hand() -> None:
    off = [1.5, 1.5, 0.5, 0.0]  # decodes to (16, 16, 32, 28): 4 px too short
    _, parts = DetectionLoss()(same_in_both(off), IMAGE, *labels([GT]), alpha=0.5)
    pred = decode_boxes(torch.tensor([[off]]), POINTS[27:28], STRIDES[27:28])[0]
    ciou = bbox_iou(pred, torch.tensor([GT]), "ciou").item()
    # one positive; its weight is its overlap, and the target score total is below 1, so n = 1
    assert parts["box_one2one"].item() == pytest.approx(ciou * (1 - ciou), abs=1e-5)
    assert parts["l1_one2one"].item() == pytest.approx(ciou * (4 / 64) / 4, abs=1e-5)


def test_each_branch_total_uses_the_gains_and_the_two_are_blended_by_alpha() -> None:
    loss = DetectionLoss(box_gain=7.5, cls_gain=0.5, l1_gain=6.0)
    total, p = loss(same_in_both([1.5, 1.5, 0.5, 0.2]), IMAGE, *labels([GT]), alpha=0.7)
    for name in ("one2many", "one2one"):
        expected = 7.5 * p[f"box_{name}"] + 0.5 * p[f"cls_{name}"] + 6.0 * p[f"l1_{name}"]
        assert p[f"total_{name}"].item() == pytest.approx(expected.item(), rel=1e-5)
    blend = 0.7 * p["total_one2many"] + 0.3 * p["total_one2one"]
    assert total.item() == pytest.approx(blend.item(), rel=1e-5)


def three_objects() -> tuple[Predictions, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Three separate objects, each with a cell that boxes it well (one slightly off)."""
    distances, logits = branch()
    for index, d in ((27, PERFECT), (45, PERFECT), (41, [1.5, 1.5, 0.5, 0.2])):
        distances[0, index] = torch.tensor(d)
        logits[0, index, 0] = 5.0
    boxes = [GT, [32.0, 32.0, 48.0, 48.0], [0.0, 32.0, 16.0, 48.0]]
    return {"one2many": (distances, logits), "one2one": (distances, logits)}, labels(boxes)


def test_the_loss_does_not_grow_with_the_number_of_images() -> None:
    # the target scores total more than 1 here, so the normaliser is not clamped
    preds, gt = three_objects()
    one = DetectionLoss()(preds, IMAGE, *gt, alpha=0.5)[1]
    twice = {k: (d.expand(2, -1, -1), lg.expand(2, -1, -1)) for k, (d, lg) in preds.items()}
    gt2 = (gt[0].expand(2, -1, -1), gt[1].expand(2, -1, -1), gt[2].expand(2, -1, -1))
    two = DetectionLoss()(twice, IMAGE, *gt2, alpha=0.5)[1]
    for key in one:  # duplicating the batch doubles the sums and the normaliser together
        assert two[key].item() == pytest.approx(one[key].item(), rel=1e-5)


def test_the_normaliser_is_the_total_target_score_not_the_object_count() -> None:
    preds, gt = three_objects()  # three objects, but the target scores total a bit under 3
    distances, logits = preds["one2one"]
    boxes = decode_boxes(distances, POINTS, STRIDES)
    target = TaskAlignedAssigner(topk=7, topk2=1)(logits.sigmoid(), boxes, POINTS, *gt)
    total_score = target.target_scores.sum().item()
    assert 2 < total_score < 2.99  # clearly different from the object count of 3
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target.target_scores, reduction="sum"
    )
    _, parts = DetectionLoss()(preds, IMAGE, *gt, alpha=0.5)
    assert parts["cls_one2one"].item() == pytest.approx(bce.item() / total_score, rel=1e-5)


def test_gradients_reach_both_branches() -> None:
    preds: Predictions = {}
    for name in ("one2many", "one2one"):
        dist, logit = branch([1.5, 1.5, 0.5, 0.2])
        preds[name] = (dist.requires_grad_(True), logit.requires_grad_(True))
    total, _ = DetectionLoss()(preds, IMAGE, *labels([GT]), alpha=0.5)
    assert torch.isfinite(total)
    grads = torch.autograd.grad(total, [t for pair in preds.values() for t in pair])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_the_loss_falls_when_predictions_move_toward_the_targets() -> None:
    distances = torch.full((1, CELLS, 4), 2.0, requires_grad=True)
    logits = torch.full((1, CELLS, 1), -5.0, requires_grad=True)
    loss_fn = DetectionLoss()
    optimiser = torch.optim.Adam([distances, logits], lr=0.1)
    history = []
    for _ in range(60):
        preds = {"one2many": (distances, logits), "one2one": (distances, logits)}
        total, _ = loss_fn(preds, IMAGE, *labels([GT]), alpha=0.5)
        optimiser.zero_grad()
        torch.autograd.backward(total)
        optimiser.step()
        history.append(total.item())
    assert history[-1] < 0.5 * history[0]
    with torch.no_grad():  # and it learned something: some cell now boxes the object well
        boxes = decode_boxes(distances, POINTS, STRIDES)[0]
        best = bbox_iou(boxes, torch.tensor([GT]), "ciou").max()
    assert best > 0.6


def test_it_accepts_the_real_models_raw_output() -> None:
    torch.manual_seed(0)
    model = YOLO26(SIZES["n"]).train()
    x = torch.rand(2, 3, 64, 64)
    gt_labels = torch.zeros(2, 2, 1, dtype=torch.long)
    gt_boxes = torch.tensor([[[8.0, 8, 40, 40], [30, 30, 60, 60]]] * 2)
    mask = torch.ones(2, 2, 1, dtype=torch.long)
    total, parts = DetectionLoss()(model.raw(x), (64, 64), gt_labels, gt_boxes, mask, alpha=0.8)
    assert torch.isfinite(total)
    first = next(model.backbone.parameters())  # the stem conv's weight
    (grad,) = torch.autograd.grad(total, [first], allow_unused=True)
    assert grad is not None and grad.abs().sum() > 0  # the loss reaches the first layer
    assert {"box_one2one", "cls_one2many", "l1_one2one", "total_one2many"} <= set(parts)
