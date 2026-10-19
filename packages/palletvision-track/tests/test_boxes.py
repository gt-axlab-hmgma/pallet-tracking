import numpy as np
import pytest

from palletvision_track.boxes import pairwise_iou, xyah_to_xyxy, xyxy_to_xyah


def test_xyxy_to_xyah_known_box() -> None:
    # a 20 wide, 10 tall box at x 10..30, y 40..50: centre (20, 45), a = 2, h = 10
    out = xyxy_to_xyah([[10, 40, 30, 50]])
    np.testing.assert_allclose(out, [[20, 45, 2, 10]])


def test_xyah_round_trip() -> None:
    boxes = np.array([[0, 0, 10, 20], [5.5, 7.25, 105.5, 57.25], [100, 100, 101, 140]])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(boxes)), boxes)


def test_degenerate_height_is_finite() -> None:
    assert np.isfinite(xyxy_to_xyah([[0, 5, 10, 5]])).all()


def test_pairwise_iou_identical_is_one() -> None:
    box = [[0, 0, 10, 10]]
    assert pairwise_iou(box, box)[0, 0] == pytest.approx(1.0)


def test_pairwise_iou_disjoint_is_zero() -> None:
    assert pairwise_iou([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0


def test_pairwise_iou_touching_edges_is_zero() -> None:
    assert pairwise_iou([[0, 0, 10, 10]], [[10, 0, 20, 10]])[0, 0] == 0


def test_pairwise_iou_half_overlap() -> None:
    # inter 5 x 10 = 50, union 100 + 100 - 50 = 150
    assert pairwise_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)


def test_pairwise_iou_contained_box() -> None:
    # a 5 x 5 box inside a 10 x 10 box: 25 / 100
    assert pairwise_iou([[0, 0, 10, 10]], [[2, 2, 7, 7]])[0, 0] == pytest.approx(0.25)


def test_pairwise_iou_shape_and_entries() -> None:
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 5, 5]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    iou = pairwise_iou(a, b)
    assert iou.shape == (3, 2)
    np.testing.assert_allclose(iou[:2], np.eye(2), atol=1e-12)
    assert iou[2, 0] == pytest.approx(0.25)


@pytest.mark.parametrize(("n", "m"), [(0, 3), (3, 0), (0, 0)])
def test_pairwise_iou_empty(n: int, m: int) -> None:
    assert pairwise_iou(np.zeros((n, 4)), np.zeros((m, 4))).shape == (n, m)


def test_pairwise_iou_zero_area_boxes_do_not_nan() -> None:
    assert pairwise_iou([[1, 1, 1, 1]], [[1, 1, 1, 1]])[0, 0] == 0
