import math

import pytest
import torch

from palletvision_model.iou import Kind, bbox_iou, pairwise_iou

KINDS: list[Kind] = ["iou", "giou", "diou", "ciou"]


def box(*xyxy: float) -> torch.Tensor:
    return torch.tensor(xyxy, dtype=torch.float32)


@pytest.mark.parametrize("kind", KINDS)
def test_identical_boxes_score_one(kind: Kind) -> None:
    a = box(10, 20, 50, 80)
    assert bbox_iou(a, a, kind).item() == pytest.approx(1.0, abs=1e-6)


def test_disjoint_boxes_iou_is_zero_but_giou_and_diou_still_say_how_far() -> None:
    a, b = box(0, 0, 1, 1), box(2, 0, 3, 1)
    assert bbox_iou(a, b).item() == 0
    # union 2, enclosing box (0,0,3,1) has area 3: giou = 0 - (3 - 2) / 3
    assert bbox_iou(a, b, "giou").item() == pytest.approx(-1 / 3, abs=1e-6)
    # centres are 2 apart, enclosing diagonal^2 = 3^2 + 1^2 = 10: diou = 0 - 4 / 10
    assert bbox_iou(a, b, "diou").item() == pytest.approx(-0.4, abs=1e-6)


def test_giou_gets_closer_to_zero_as_disjoint_boxes_approach() -> None:
    a = box(0, 0, 1, 1)
    far, near = bbox_iou(a, box(5, 0, 6, 1), "giou"), bbox_iou(a, box(2, 0, 3, 1), "giou")
    assert far < near < 0


def test_partial_overlap_hand_computed_squares() -> None:
    a, b = box(0, 0, 2, 2), box(1, 1, 3, 3)  # intersection 1, union 7, enclosing 3 x 3
    assert bbox_iou(a, b).item() == pytest.approx(1 / 7, abs=1e-6)
    assert bbox_iou(a, b, "giou").item() == pytest.approx(1 / 7 - 2 / 9, abs=1e-6)
    assert bbox_iou(a, b, "diou").item() == pytest.approx(1 / 7 - 2 / 18, abs=1e-6)
    # both are squares, so there is no aspect-ratio penalty
    assert bbox_iou(a, b, "ciou").item() == pytest.approx(1 / 7 - 2 / 18, abs=1e-6)


def test_ciou_penalises_a_different_aspect_ratio() -> None:
    a, b = box(0, 0, 2, 1), box(0, 0, 1, 2)  # a wide box and a tall one
    iou = 1 / 3  # intersection 1, union 3
    diou = iou - 0.5 / 8  # centre distance^2 0.5, enclosing diagonal^2 8
    v = (4 / math.pi**2) * (math.atan(0.5) - math.atan(2.0)) ** 2
    alpha = v / ((1 - iou) + v)
    assert bbox_iou(a, b, "diou").item() == pytest.approx(diou, abs=1e-6)
    assert bbox_iou(a, b, "ciou").item() == pytest.approx(diou - alpha * v, abs=1e-5)
    assert bbox_iou(a, b, "ciou") < bbox_iou(a, b, "diou")


def test_measures_are_ordered_and_symmetric_on_random_boxes() -> None:
    torch.manual_seed(0)
    xy = torch.rand(200, 2) * 50
    wh = torch.rand(200, 2) * 40 + 1
    a = torch.cat((xy, xy + wh), dim=-1)
    xy = torch.rand(200, 2) * 50
    wh = torch.rand(200, 2) * 40 + 1
    b = torch.cat((xy, xy + wh), dim=-1)
    iou, giou, diou, ciou = (bbox_iou(a, b, k) for k in KINDS)
    assert (iou >= 0).all() and (iou <= 1).all()
    assert (giou <= iou + 1e-6).all()  # each refinement can only lower the score
    assert (diou <= iou + 1e-6).all()
    assert (ciou <= diou + 1e-6).all()
    assert (giou > -1).all()
    for kind in KINDS:
        assert torch.allclose(bbox_iou(a, b, kind), bbox_iou(b, a, kind), atol=1e-6)


def test_pairwise_shape_and_values_match_looping() -> None:
    torch.manual_seed(1)
    a = torch.cat((torch.rand(5, 2) * 10, torch.rand(5, 2) * 10 + 11), dim=-1)
    b = torch.cat((torch.rand(3, 2) * 10, torch.rand(3, 2) * 10 + 11), dim=-1)
    out = pairwise_iou(a, b, "ciou")
    assert out.shape == (5, 3)
    for i in range(5):
        for j in range(3):
            assert out[i, j].item() == pytest.approx(bbox_iou(a[i], b[j], "ciou").item(), abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_gradient_reaches_the_predicted_box(kind: Kind) -> None:
    pred = box(1, 1, 5, 6).requires_grad_(True)
    (grad,) = torch.autograd.grad(bbox_iou(pred, box(2, 2, 7, 7), kind), pred)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_degenerate_boxes_stay_finite(kind: Kind) -> None:
    # an untrained head can give zero-size or inside-out boxes (x2 < x1)
    pred = torch.tensor([[3.0, 3.0, 3.0, 3.0], [5.0, 5.0, 2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[2.0, 2.0, 7.0, 7.0], [2.0, 2.0, 7.0, 7.0]])
    value = bbox_iou(pred, target, kind)
    (grad,) = torch.autograd.grad(value.sum(), pred)
    assert torch.isfinite(value).all()
    assert torch.isfinite(grad).all()
