import torch

from palletvision_model.config import SIZES
from palletvision_model.decode import decode_boxes, make_anchors
from palletvision_model.head import Detect, flatten_levels

SIZES_640 = [(80, 80), (40, 40), (20, 20)]


def test_anchor_count_follows_the_input_size() -> None:
    points, strides = make_anchors(SIZES_640)
    assert points.shape == (8400, 2)
    assert strides.shape == (8400, 1)
    points, _ = make_anchors([(40, 40), (20, 20), (10, 10)])  # a 320 input
    assert points.shape[0] == 2100


def test_known_anchor_positions() -> None:
    points, strides = make_anchors(SIZES_640)
    assert points[0].tolist() == [4.0, 4.0]  # first P3 cell: centre of an 8x8 square
    assert points[1].tolist() == [12.0, 4.0]  # next cell to the right
    assert points[80].tolist() == [4.0, 12.0]  # first cell of the second row
    assert points[6400].tolist() == [8.0, 8.0]  # first P4 cell, stride 16
    assert points[8000].tolist() == [16.0, 16.0]  # first P5 cell, stride 32
    assert [strides[i].item() for i in (0, 6399, 6400, 7999, 8000, 8399)] == [8, 8, 16, 16, 32, 32]


def test_anchor_order_matches_the_heads_flatten_order() -> None:
    # fill every cell with its own id (row * width + col), flatten like the head does, and
    # check the anchor in the same position sits at that cell's centre
    maps = [torch.arange(h * w).reshape(1, 1, h, w).float() for (h, w) in SIZES_640]
    ids = flatten_levels(maps)[0, :, 0].long()
    points, strides = make_anchors(SIZES_640)
    widths = torch.tensor([w for _, w in SIZES_640]).repeat_interleave(
        torch.tensor([h * w for h, w in SIZES_640])
    )
    col, row = ids % widths, ids // widths
    assert torch.equal(points[:, 0], (col + 0.5) * strides[:, 0])
    assert torch.equal(points[:, 1], (row + 0.5) * strides[:, 0])


def test_decode_a_known_box() -> None:
    anchor = torch.tensor([[4.0, 4.0]])
    stride = torch.tensor([[8.0]])
    distances = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # l, t, r, b in stride units
    box = decode_boxes(distances, anchor, stride)
    assert box.tolist() == [[[-4.0, -12.0, 28.0, 36.0]]]  # not clamped to the image


def test_initial_distances_give_square_boxes_centred_on_the_cell() -> None:
    points, strides = make_anchors(SIZES_640)
    distances = torch.full((1, 8400, 4), 2.0)  # what the head's bias starts at
    boxes = decode_boxes(distances, points, strides)
    centre = (boxes[..., :2] + boxes[..., 2:]) / 2
    side = boxes[..., 2:] - boxes[..., :2]
    assert torch.allclose(centre, points)
    assert torch.equal(side[0, 0], torch.tensor([32.0, 32.0]))  # P3: 4 strides of 8
    assert torch.equal(side[0, 8399], torch.tensor([128.0, 128.0]))  # P5: 4 strides of 32


def test_decode_is_differentiable_and_keeps_the_batch() -> None:
    points, strides = make_anchors(SIZES_640)
    distances = torch.rand(3, 8400, 4, requires_grad=True)
    boxes = decode_boxes(distances, points, strides)
    assert boxes.shape == (3, 8400, 4)
    (grad,) = torch.autograd.grad(boxes.sum(), distances)
    assert grad.abs().sum() > 0


def test_works_on_real_head_output() -> None:
    cfg = SIZES["n"]
    head = Detect(cfg, (64, 128, 256)).eval()
    feats = tuple(torch.rand(2, c, 640 // s, 640 // s) for c, s in ((64, 8), (128, 16), (256, 32)))
    distances, _ = head(feats)["one2one"]
    points, strides = make_anchors(SIZES_640)
    boxes = decode_boxes(distances, points, strides)
    assert boxes.shape == (2, 8400, 4)
    assert torch.isfinite(boxes).all()
