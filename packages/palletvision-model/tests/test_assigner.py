import pytest
import torch

from palletvision_model.assigner import Assignment, TaskAlignedAssigner, candidate_mask
from palletvision_model.decode import make_anchors
from palletvision_model.iou import bbox_iou

# An 8 x 8 grid of 8 px cells covers a 64 x 64 "image". Cell (row, col) has index row * 8 + col
# and its centre is at (col * 8 + 4, row * 8 + 4).
POINTS, _ = make_anchors([(8, 8)], (8,))
ONE_TO_MANY = TaskAlignedAssigner(topk=10)
ONE_TO_ONE = TaskAlignedAssigner(topk=7, topk2=1)


def cell(row: int, col: int) -> int:
    return row * 8 + col


def predictions(num_classes: int = 1, score: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Every cell predicts a 16 x 16 box centred on itself, with the same score."""
    boxes = torch.cat((POINTS - 8, POINTS + 8), dim=-1)[None]
    return torch.full((1, 64, num_classes), score), boxes


def objects(
    boxes: list[list[float]], labels: list[int] | None = None, mask: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels or [0] * len(boxes)
    mask = mask or [1] * len(boxes)
    return (
        torch.tensor(labels)[None, :, None],
        torch.tensor(boxes, dtype=torch.float32)[None],
        torch.tensor(mask)[None, :, None],
    )


def run(
    assigner: TaskAlignedAssigner,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Assignment:
    result: Assignment = assigner(scores, boxes, POINTS, *gt)
    return result


def positives(result: Assignment) -> list[int]:
    return result.fg_mask[0].nonzero().flatten().tolist()


def test_one_object_one_obvious_cell_is_the_single_positive() -> None:
    scores, boxes = predictions()
    scores[0, cell(3, 3), 0] = 0.9
    boxes[0, cell(3, 3)] = torch.tensor([16.0, 16.0, 32.0, 32.0])  # exactly the object
    result = run(ONE_TO_ONE, scores, boxes, objects([[16, 16, 32, 32]]))
    assert positives(result) == [cell(3, 3)]
    assert result.target_scores[0, cell(3, 3), 0].item() == pytest.approx(1.0, abs=1e-4)  # = IoU
    assert result.target_boxes[0, cell(3, 3)].tolist() == [16.0, 16.0, 32.0, 32.0]
    assert result.target_labels[0, cell(3, 3)].item() == 0


def test_a_positives_class_target_is_how_good_it_is_not_one() -> None:
    scores, boxes = predictions()
    scores[0, cell(3, 3), 0] = 0.9
    boxes[0, cell(3, 3)] = torch.tensor([16.0, 16.0, 32.0, 28.0])  # IoU 0.75 with the object
    result = run(ONE_TO_MANY, scores, boxes, objects([[16, 16, 32, 32]]))
    best = result.target_scores[0, cell(3, 3), 0].item()
    expected = bbox_iou(objects([[16, 16, 32, 32]])[1][0, 0], boxes[0, cell(3, 3)], "ciou").item()
    assert best == pytest.approx(expected, abs=1e-4)  # the best cell's target equals its CIoU
    others = [result.target_scores[0, i, 0].item() for i in positives(result) if i != cell(3, 3)]
    assert others and all(0 < v < best for v in others)  # weaker positives are asked for less


def test_background_cells_have_zero_targets() -> None:
    scores, boxes = predictions()
    result = run(ONE_TO_MANY, scores, boxes, objects([[16, 16, 32, 32]]))
    background = ~result.fg_mask
    assert result.target_scores[background].abs().sum() == 0


def test_shortlist_size_decides_how_many_positives() -> None:
    scores, boxes = predictions()
    big = objects([[0, 0, 64, 64]])  # all 64 cells are candidates and all have IoU > 0
    assert len(positives(run(ONE_TO_MANY, scores, boxes, big))) == 10
    assert len(positives(run(ONE_TO_ONE, scores, boxes, big))) == 1  # 7 shortlisted, then 1
    assert len(positives(run(TaskAlignedAssigner(topk=3), scores, boxes, big))) == 3


def test_one_to_one_keeps_the_best_cell_of_the_shortlist() -> None:
    scores, boxes = predictions()
    scores[0, cell(5, 5), 0] = 0.95
    boxes[0, cell(5, 5)] = torch.tensor([0.0, 0.0, 64.0, 64.0])  # a perfect box for the big object
    result = run(ONE_TO_ONE, scores, boxes, objects([[0, 0, 64, 64]]))
    assert positives(result) == [cell(5, 5)]


def test_a_cell_wanted_by_two_objects_goes_to_the_one_it_overlaps_more() -> None:
    scores, boxes = predictions()
    boxes[0, cell(3, 3)] = torch.tensor([16.0, 16.0, 50.0, 50.0])  # fits both, object 1 better
    gt = objects([[0, 0, 40, 40], [24, 24, 64, 64]])
    everything = TaskAlignedAssigner(topk=64)  # every cell with a positive alignment is shortlisted
    result = run(everything, scores, boxes, gt)
    overlaps = bbox_iou(gt[1][:, :, None, :], boxes[:, None, :, :], "ciou").clamp(
        min=0
    )  # [1, 2, 64]
    assert (overlaps[0, :, cell(3, 3)] > 0).all()  # both objects really do want this cell
    assert result.target_gt_idx[0, cell(3, 3)].item() == 1  # and the better-fitting one gets it
    contested = ((overlaps[0] > 0).all(dim=0) & result.fg_mask[0]).nonzero().flatten()
    for index in contested.tolist():  # every contested cell went to the object it overlaps most
        assert result.target_gt_idx[0, index].item() == overlaps[0, :, index].argmax().item()


def test_a_tiny_object_gets_candidates_only_with_stal() -> None:
    scores, _ = predictions()
    boxes = torch.cat((POINTS - 4, POINTS + 4), dim=-1)[None]  # 8 px boxes, like a small object
    tiny = objects([[13, 13, 17, 17]])  # 4 px wide: it sits between cell centres 12 and 20
    off = run(TaskAlignedAssigner(topk=10, stal=False), scores, boxes, tiny)
    assert positives(off) == []  # no cell centre is inside it: the object is never learned
    on = run(TaskAlignedAssigner(topk=10, stal=True), scores, boxes, tiny)
    # for the test it is clamped to 16 px around its centre (15, 15): centres 12 and 20 qualify
    candidates = {cell(1, 1), cell(1, 2), cell(2, 1), cell(2, 2)}
    assert positives(on) and set(positives(on)) <= candidates
    # the real box, not the enlarged one, is the target
    first = positives(on)[0]
    assert on.target_boxes[0, first].tolist() == [13.0, 13.0, 17.0, 17.0]
    assert len(positives(run(ONE_TO_ONE, scores, boxes, tiny))) == 1


def test_stal_changes_only_the_small_side_and_leaves_big_boxes_alone() -> None:
    big = torch.tensor([[[16.0, 16.0, 32.0, 32.0]]])
    assert torch.equal(
        candidate_mask(POINTS, big, stal=True), candidate_mask(POINTS, big, stal=False)
    )
    flat = torch.tensor([[[0.0, 0.0, 64.0, 4.0]]])  # wide but only 4 px tall
    assert candidate_mask(POINTS, flat, stal=False).sum() == 0
    kept = candidate_mask(POINTS, flat, stal=True)[0, 0].nonzero().flatten().tolist()
    assert kept == [cell(0, c) for c in range(8)]  # the first row only: y in (-6, 10)


def test_padding_objects_never_get_positives() -> None:
    scores, boxes = predictions()
    gt = objects([[16, 16, 32, 32], [0, 0, 64, 64], [32, 32, 56, 56]], mask=[1, 0, 0])
    result = run(ONE_TO_MANY, scores, boxes, gt)
    assert positives(result)
    assert (result.target_gt_idx[result.fg_mask] == 0).all()
    alone = run(ONE_TO_MANY, scores, boxes, objects([[16, 16, 32, 32]]))
    assert positives(result) == positives(alone)


def test_an_image_without_objects_gets_no_positives() -> None:
    scores, boxes = predictions()
    nothing = (torch.zeros(1, 0, 1, dtype=torch.long), torch.zeros(1, 0, 4), torch.zeros(1, 0, 1))
    result = run(ONE_TO_MANY, scores, boxes, nothing)
    assert result.target_scores.shape == (1, 64, 1) and not result.fg_mask.any()
    only_padding = objects([[0, 0, 64, 64]], mask=[0])
    assert not run(ONE_TO_MANY, scores, boxes, only_padding).fg_mask.any()


def test_a_batch_with_different_object_counts_matches_running_each_image_alone() -> None:
    scores, boxes = predictions()
    one = objects([[16, 16, 32, 32]])
    two = objects([[0, 0, 24, 24], [32, 32, 64, 64]])
    labels = torch.cat((torch.cat((one[0], torch.zeros(1, 1, 1, dtype=torch.long)), 1), two[0]))
    gt_boxes = torch.cat((torch.cat((one[1], torch.zeros(1, 1, 4)), 1), two[1]))
    mask = torch.cat((torch.tensor([[[1], [0]]]), two[2]))
    batched = ONE_TO_MANY(
        scores.expand(2, -1, -1), boxes.expand(2, -1, -1), POINTS, labels, gt_boxes, mask
    )
    for i, alone_gt in enumerate((one, two)):
        alone = run(ONE_TO_MANY, scores, boxes, alone_gt)
        assert torch.equal(batched.fg_mask[i], alone.fg_mask[0])
        assert torch.allclose(batched.target_scores[i], alone.target_scores[0])
        assert torch.equal(
            batched.target_boxes[i][alone.fg_mask[0]], alone.target_boxes[0][alone.fg_mask[0]]
        )


def test_the_objects_class_score_is_the_one_that_counts() -> None:
    scores, boxes = predictions(num_classes=2, score=0.01)
    boxes[:] = torch.tensor([16.0, 16.0, 32.0, 32.0])  # every cell predicts the object's box
    scores[0, cell(2, 2)] = torch.tensor([0.9, 0.05])  # confident about class 0
    scores[0, cell(3, 3)] = torch.tensor([0.05, 0.9])  # confident about class 1
    result = run(ONE_TO_ONE, scores, boxes, objects([[16, 16, 32, 32]], labels=[1]))
    assert positives(result) == [cell(3, 3)]
    assert result.target_labels[0, cell(3, 3)].item() == 1
    assert result.target_scores[0, cell(3, 3)].tolist() == pytest.approx([0.0, 1.0], abs=1e-4)


def test_zero_scores_mean_no_positives() -> None:
    scores, boxes = predictions(score=0.0)
    assert positives(run(ONE_TO_MANY, scores, boxes, objects([[16, 16, 32, 32]]))) == []


def test_one_to_one_never_gives_an_object_more_than_one_cell() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        scores = torch.rand(1, 64, 1)
        xy = torch.rand(1, 64, 2) * 40
        boxes = torch.cat((xy, xy + torch.rand(1, 64, 2) * 24 + 4), dim=-1)
        centres = torch.rand(4, 2) * 40
        gt = objects(torch.cat((centres, centres + torch.rand(4, 2) * 24 + 4), dim=-1).tolist())
        result = run(ONE_TO_ONE, scores, boxes, gt)
        for m in range(4):
            assert ((result.target_gt_idx[0] == m) & result.fg_mask[0]).sum() <= 1


def test_the_assigner_is_outside_the_gradient_graph() -> None:
    scores, boxes = predictions()
    scores.requires_grad_(True)
    boxes.requires_grad_(True)
    result = run(ONE_TO_MANY, scores, boxes, objects([[16, 16, 32, 32]]))
    assert not result.target_scores.requires_grad and not result.target_boxes.requires_grad


def test_an_object_that_loses_its_best_cell_falls_back_to_its_next_best() -> None:
    scores, boxes = predictions()
    boxes[0, cell(3, 3)] = torch.tensor([16.0, 16.0, 50.0, 50.0])  # both objects want it
    gt = objects([[0, 0, 40, 40], [24, 24, 64, 64]])
    result = run(ONE_TO_ONE, scores, boxes, gt)
    holders = {m: [i for i in positives(result) if result.target_gt_idx[0, i] == m] for m in (0, 1)}
    assert holders[1] == [cell(3, 3)]  # the better fit keeps the contested cell
    assert len(holders[0]) == 1 and holders[0] != [cell(3, 3)]  # the other still gets a cell


def test_plain_iou_can_be_chosen_instead_of_ciou() -> None:
    scores, boxes = predictions()
    plain = TaskAlignedAssigner(topk=10, iou_kind="iou")
    assert positives(run(plain, scores, boxes, objects([[16, 16, 32, 32]])))
