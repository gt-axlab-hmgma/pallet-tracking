import torch

from palletvision_model.config import SIZES
from palletvision_model.decode import postprocess
from palletvision_model.yolo26 import YOLO26

BIG = 20.0  # a logit whose sigmoid is about 1
IMAGE = (100, 200)  # height, width


def toy_boxes(cells: int) -> torch.Tensor:
    """Cell i gets the box (i, i, i + 10, i + 10), so a row's box tells which cell it came from."""
    base = torch.arange(cells, dtype=torch.float32)
    return torch.stack((base, base, base + 10, base + 10), dim=-1).unsqueeze(0)


def test_output_is_padded_to_300_rows_sorted_by_score() -> None:
    logits = torch.tensor([[[-4.0], [2.0], [0.0]]])  # three cells, one class
    out = postprocess(toy_boxes(3), logits, IMAGE)
    assert out.shape == (1, 300, 6)
    assert out[0, :3, 4].tolist() == sorted(logits.sigmoid().flatten().tolist(), reverse=True)
    assert torch.equal(out[0, 3:], torch.zeros(297, 6))  # padding rows are all zero


def test_cell_and_class_are_recovered_from_the_flat_index() -> None:
    logits = torch.full((1, 4, 3), -BIG)  # four cells, three classes
    logits[0, 2, 1] = 5.0  # cell 2, class 1
    logits[0, 3, 2] = 3.0  # cell 3, class 2 (lower, so second)
    out = postprocess(toy_boxes(4), logits, IMAGE)
    assert out[0, 0, [0, 5]].tolist() == [2.0, 1.0]  # x1 of cell 2, class 1
    assert out[0, 1, [0, 5]].tolist() == [3.0, 2.0]


def test_one_cell_can_appear_once_per_class() -> None:
    # a cell that is confident about two classes yields two rows, with the same box
    logits = torch.full((1, 2, 2), -BIG)
    logits[0, 1, :] = BIG
    out = postprocess(toy_boxes(2), logits, IMAGE)
    assert out[0, :2, 5].sort().values.tolist() == [0.0, 1.0]
    assert torch.equal(out[0, 0, :4], out[0, 1, :4])


def test_takes_exactly_the_top_300_when_there_are_more_candidates() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 500, 1)
    out = postprocess(toy_boxes(500).expand(2, -1, -1), logits, IMAGE)
    assert out.shape == (2, 300, 6)
    expected = logits.sigmoid().flatten(1).sort(dim=1, descending=True).values[:, :300]
    assert torch.equal(out[..., 4], expected)


def test_no_confidence_threshold_inside_the_model() -> None:
    out = postprocess(toy_boxes(1), torch.tensor([[[-10.0]]]), IMAGE)
    assert 0 < out[0, 0, 4] < 0.001  # a near-zero score is still reported


def test_boxes_are_clamped_to_the_image() -> None:
    boxes = torch.tensor([[[-30.0, -5.0, 250.0, 130.0]]])  # sticks out on every side
    out = postprocess(boxes, torch.tensor([[[BIG]]]), IMAGE)
    assert out[0, 0, :4].tolist() == [0.0, 0.0, 200.0, 100.0]  # x within 200, y within 100


def test_postprocess_does_not_change_the_batch_dimension() -> None:
    out = postprocess(toy_boxes(5).expand(3, -1, -1), torch.zeros(3, 5, 1), IMAGE)
    assert out.shape[0] == 3


def test_model_returns_contract_shaped_detections() -> None:
    model = YOLO26(SIZES["n"]).eval()
    out = model(torch.rand(1, 3, 640, 640))
    assert out.shape == (1, 300, 6)
    assert out[..., 4].min() >= 0 and out[..., 4].max() <= 1
    assert out[..., :4].min() >= 0 and out[..., :4].max() <= 640
    assert torch.all(out[0, 1:, 4] <= out[0, :-1, 4])  # sorted by score


def test_model_handles_other_sizes_and_tiny_inputs_that_need_padding() -> None:
    model = YOLO26(SIZES["n"]).eval()
    assert model(torch.rand(2, 3, 320, 320)).shape == (2, 300, 6)
    tiny = model(torch.rand(1, 3, 64, 64))  # 84 cells, fewer than 300 candidates
    assert tiny.shape == (1, 300, 6)
    assert torch.equal(tiny[0, 84:], torch.zeros(216, 6))


def test_raw_gives_both_branches_in_training_and_one_in_eval() -> None:
    model = YOLO26(SIZES["n"])
    x = torch.rand(1, 3, 64, 64)
    assert set(model.train().raw(x)) == {"one2many", "one2one"}
    assert set(model.eval().raw(x)) == {"one2one"}
    assert model.train()(x).shape == (1, 300, 6)  # forward works in either mode
