"""Scoring with more than one class (needed for pretraining on a many-class dataset)."""

from pathlib import Path

import numpy as np
import pytest


def test_the_class_branch_is_scored_when_there_is_more_than_one_class(tmp_path: Path) -> None:
    from palletvision_train.coco import CocoDetection, ImageRecord
    from palletvision_train.evaluate import coco_ground_truth, score

    record = ImageRecord(tmp_path / "a.jpg", 100, 100, np.array([[1, 10, 10, 50, 50]], np.float32))
    gt = coco_ground_truth(CocoDetection([record]), categories=2)
    hit: list[dict[str, float | int | list[float]]] = [
        {"image_id": 0, "category_id": 2, "bbox": [10.0, 10.0, 40.0, 40.0], "score": 0.9}
    ]
    wrong: list[dict[str, float | int | list[float]]] = [
        {"image_id": 0, "category_id": 1, "bbox": [10.0, 10.0, 40.0, 40.0], "score": 0.9}
    ]
    assert score(hit, gt).map50 == pytest.approx(
        1.0
    )  # class 2 found; class 1 has no objects: ignored
    assert score(wrong, gt).map50 == 0.0  # the right box, but labelled as the wrong class
