import numpy as np
import pytest

from palletvision_train.evaluation.metrics import average_precision


def box(*xyxy: float) -> np.ndarray:
    return np.array(xyxy, dtype=np.float32)


TRUTH = {0: np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)}


def test_finding_everything_exactly_scores_one() -> None:
    found = [(0, 0.9, box(0, 0, 10, 10)), (0, 0.8, box(20, 20, 30, 30))]
    assert average_precision(found, TRUTH) == pytest.approx(1.0)


def test_nothing_found_or_nothing_to_find_scores_zero() -> None:
    assert average_precision([], TRUTH) == 0.0
    assert (
        average_precision([(0, 0.9, box(0, 0, 10, 10))], {0: np.zeros((0, 4), np.float32)}) == 0.0
    )


def test_finding_half_the_objects_scores_half() -> None:
    assert average_precision([(0, 0.9, box(0, 0, 10, 10))], TRUTH) == pytest.approx(0.5)


def test_a_confident_wrong_box_costs_precision() -> None:
    # the most confident detection is wrong: both real objects are found second and third
    found = [
        (0, 0.9, box(50, 50, 60, 60)),
        (0, 0.8, box(0, 0, 10, 10)),
        (0, 0.7, box(20, 20, 30, 30)),
    ]
    # precision at recall 0.5 is 1/2 and at recall 1.0 is 2/3
    assert average_precision(found, TRUTH) == pytest.approx(0.5 * (2 / 3) + 0.5 * (2 / 3))


def test_a_duplicate_of_a_matched_object_is_a_false_positive() -> None:
    clean = [(0, 0.9, box(0, 0, 10, 10)), (0, 0.8, box(20, 20, 30, 30))]
    duplicated = [clean[0], (0, 0.85, box(0, 0, 10, 9)), clean[1]]  # a second box on object 1
    assert (
        average_precision(duplicated, TRUTH) < average_precision(clean, TRUTH) == pytest.approx(1.0)
    )
    assert average_precision(duplicated, TRUTH) == pytest.approx(0.5 + 0.5 * (2 / 3))


def test_the_iou_threshold_decides_what_counts_as_a_hit() -> None:
    loose = [(0, 0.9, box(0, 0, 10, 6))]  # overlaps object 0 by IoU 0.6
    assert average_precision(loose, TRUTH, 0.5) == pytest.approx(0.5)
    assert average_precision(loose, TRUTH, 0.7) == 0.0


def test_detections_are_matched_within_their_own_image() -> None:
    truth = {0: np.array([[0, 0, 10, 10]], np.float32), 1: np.array([[40, 40, 50, 50]], np.float32)}
    swapped = [
        (0, 0.9, box(40, 40, 50, 50)),
        (1, 0.8, box(0, 0, 10, 10)),
    ]  # right boxes, wrong images
    assert average_precision(swapped, truth) == 0.0
    right = [(0, 0.9, box(0, 0, 10, 10)), (1, 0.8, box(40, 40, 50, 50))]
    assert average_precision(right, truth) == pytest.approx(1.0)
