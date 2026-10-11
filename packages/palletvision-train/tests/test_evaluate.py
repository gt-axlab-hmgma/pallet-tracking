from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import torch
import yaml
from torch import nn

from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26
from palletvision_preprocess import letterbox_boxes, letterbox_image
from palletvision_train.coco import CocoDetection, ImageRecord
from palletvision_train.evaluate import (
    MIN_SCORE,
    EvalResult,
    coco_ground_truth,
    detect,
    evaluate,
    score,
)
from palletvision_train.metrics import average_precision

SIZE = 128
# (height, width): a wide, a tall and a square image, so un-letterboxing has to get it right
SHAPES = [(60, 100), (100, 60), (80, 80)]
BOXES = [  # per image, x1 y1 x2 y2 in that image's own pixels
    [[10, 10, 40, 30], [60, 20, 90, 50]],
    [[5, 10, 30, 60]],
    [[20, 20, 60, 60], [10, 50, 30, 75]],
]


@pytest.fixture
def reader(tmp_path: Path) -> CocoDetection:
    records = []
    for i, ((h, w), boxes) in enumerate(zip(SHAPES, BOXES, strict=True)):
        path = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(path), np.full((h, w, 3), 90, dtype=np.uint8))
        labels = np.array([[0, *box] for box in boxes], dtype=np.float32)
        records.append(ImageRecord(path, w, h, labels))
    return CocoDetection(records)


class Scripted(nn.Module):
    """A stand-in model that returns the detections it was given, image after image."""

    def __init__(self, per_image: list[np.ndarray]) -> None:
        super().__init__()
        self.per_image, self.next = per_image, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.per_image[self.next : self.next + len(x)]
        self.next += len(x)
        return torch.from_numpy(np.stack(rows))


def table(boxes_letterboxed: np.ndarray, scores: list[float]) -> np.ndarray:
    """A model output for one image: [300, 6] rows (x1 y1 x2 y2 score class), zero padded."""
    out = np.zeros((300, 6), dtype=np.float32)
    out[: len(boxes_letterboxed), :4] = boxes_letterboxed
    out[: len(boxes_letterboxed), 4] = scores
    return out


def letterboxed(i: int, shift: float = 0.0) -> np.ndarray:
    """Image ``i``'s true boxes, in the model's letterboxed pixels (optionally moved)."""
    h, w = SHAPES[i]
    _, meta = letterbox_image(np.zeros((h, w, 3), dtype=np.uint8), SIZE)
    return letterbox_boxes(np.array(BOXES[i], dtype=np.float32) + shift, meta)


def perfect(score_value: float = 0.9) -> Scripted:
    return Scripted([table(letterboxed(i), [score_value] * len(BOXES[i])) for i in range(3)])


def run(model: nn.Module, reader: CocoDetection) -> EvalResult:
    return evaluate(cast(YOLO26, model), reader, size=SIZE, device="cpu", batch_size=2)


def test_finding_every_object_exactly_scores_one(reader: CocoDetection) -> None:
    r = run(perfect(), reader)
    assert (r.images, r.objects, r.detections) == (3, 5, 5)
    for name in ("map50_95", "map50", "map75", "recall", "best_f1", "precision_at_best"):
        assert getattr(r, name) == pytest.approx(1.0, abs=1e-3), name


def test_detections_come_back_in_original_pixels_with_the_right_ids(reader: CocoDetection) -> None:
    results = detect(cast(YOLO26, perfect()), reader, size=SIZE, device="cpu", batch_size=2)
    assert sorted(r["image_id"] for r in results) == [0, 0, 1, 2, 2]
    assert {cast(int, r["category_id"]) for r in results} == {
        1
    }  # COCO ids start at 1; the class is 0
    for image in range(3):  # each image's detections are exactly its labels, as x y w h
        got = sorted(cast(list[float], r["bbox"]) for r in results if r["image_id"] == image)
        want = sorted([x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in BOXES[image])
        np.testing.assert_allclose(np.array(got), np.array(want, dtype=float), atol=0.01)


def test_no_detections_scores_zero_and_weak_ones_are_dropped(reader: CocoDetection) -> None:
    nothing = Scripted([table(np.zeros((0, 4)), []) for _ in range(3)])
    assert run(nothing, reader).map50_95 == 0.0
    assert MIN_SCORE == 0.001  # the guide's value: mAP needs the weak tail, not a 0.25 cut-off
    weak = Scripted([table(letterboxed(i), [0.0005] * len(BOXES[i])) for i in range(3)])  # below it
    r = run(weak, reader)
    assert r.detections == 0 and r.map50 == 0.0


def test_detections_are_matched_within_their_own_image(reader: CocoDetection) -> None:
    # every image gets image 0's detections: only image 0's two objects can be found
    wrong = Scripted([table(letterboxed(0), [0.9, 0.9]) for _ in range(3)])
    r = run(wrong, reader)
    assert 0.3 < r.map50 < 0.5 and r.recall < 0.5


def test_a_slightly_moved_box_passes_at_iou_50_but_not_at_strict_thresholds(
    reader: CocoDetection,
) -> None:
    moved = Scripted([table(letterboxed(i, shift=3.0), [0.9] * len(BOXES[i])) for i in range(3)])
    r = run(moved, reader)
    assert r.map50 == pytest.approx(1.0, abs=1e-3)
    assert r.map50_95 < 0.9


def test_the_chosen_confidence_threshold_separates_real_from_false_detections(
    reader: CocoDetection,
) -> None:
    junk = np.array([[2, 2, 10, 10], [100, 100, 120, 120], [50, 2, 70, 10]], dtype=np.float32)
    mixed = Scripted(
        [
            table(np.concatenate([letterboxed(i), junk]), [0.9] * len(BOXES[i]) + [0.2] * 3)
            for i in range(3)
        ]
    )
    r = run(mixed, reader)
    assert r.best_conf == pytest.approx(0.9, abs=1e-6)
    assert r.best_f1 == pytest.approx(1.0, abs=1e-6)
    assert r.map50 == pytest.approx(1.0, abs=1e-3)  # the junk is ranked last: AP is untouched


def test_a_missed_half_agrees_with_the_simple_ap_measure(reader: CocoDetection) -> None:
    half = Scripted([table(letterboxed(i)[:1], [0.9]) for i in range(3)])  # one box per image
    r = run(half, reader)
    detections = [(i, 0.9, letterboxed(i)[0]) for i in range(3)]
    truths = {i: np.array(BOXES[i], dtype=np.float32) for i in range(3)}
    # the two measures interpolate differently, but both say about 3 found in 5
    assert r.map50 == pytest.approx(0.6, abs=0.03)
    assert average_precision(
        [(i, s, np.array(BOXES[i][0], dtype=np.float32)) for i, s, _ in detections], truths
    ) == pytest.approx(0.6, abs=0.03)


def test_ground_truth_holds_the_reader_labels(reader: CocoDetection) -> None:
    gt = coco_ground_truth(reader)
    assert len(gt.dataset["images"]) == 3 and len(gt.dataset["annotations"]) == 5
    first = gt.dataset["annotations"][0]
    assert first["image_id"] == 0 and first["category_id"] == 1
    assert first["bbox"] == [10.0, 10.0, 30.0, 20.0] and first["area"] == 600.0
    assert score([], gt).objects == 5  # even with nothing detected


def test_an_untrained_real_model_runs_through_evaluation(reader: CocoDetection) -> None:
    torch.manual_seed(0)
    model = YOLO26(SIZES["n"], input_size=SIZE)
    r = evaluate(model, reader, size=SIZE, device="cpu", batch_size=2)
    assert r.images == 3 and r.objects == 5
    assert 0.0 <= r.map50_95 <= r.map50 <= 1.0 and np.isfinite(r.best_f1)


REPO = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not (REPO / "data/loco/annotations").exists(), reason="LOCO data not present")
def test_real_loco_validation_images_go_through_evaluation() -> None:
    data = yaml.safe_load((REPO / "configs/data/loco.yaml").read_text())
    data["root"] = str(REPO / data["root"])
    data["annotations"] = {k: [str(REPO / f) for f in v] for k, v in data["annotations"].items()}
    classes = yaml.safe_load((REPO / "contracts/classes.yaml").read_text())["stage1"]
    full = CocoDetection.from_config(data, "val", classes)
    small = CocoDetection(full.records[:4])
    torch.manual_seed(0)
    r = evaluate(YOLO26(SIZES["n"], input_size=SIZE), small, size=SIZE, device="cpu", batch_size=2)
    assert r.images == 4 and r.objects == sum(len(x.labels) for x in small.records)
    assert 0.0 <= r.map50_95 <= 1.0


def test_the_numbers_match_coco_own_summary_when_it_applies(reader: CocoDetection) -> None:
    """With at most 100 detections per image COCO's stock settings apply, so they must agree."""
    from pycocotools.cocoeval import COCOeval

    moved = Scripted([table(letterboxed(i, shift=3.0), [0.9] * len(BOXES[i])) for i in range(3)])
    results = detect(cast(YOLO26, moved), reader, size=SIZE, device="cpu", batch_size=2)
    gt = coco_ground_truth(reader)
    mine = score(results, gt)
    stock = COCOeval(gt, gt.loadRes(results), "bbox")  # pyright: ignore[reportArgumentType]
    stock.evaluate()
    stock.accumulate()
    stock.summarize()
    assert mine.map50_95 == pytest.approx(stock.stats[0], abs=1e-6)
    assert mine.map50 == pytest.approx(stock.stats[1], abs=1e-6)
    assert mine.map75 == pytest.approx(stock.stats[2], abs=1e-6)
    assert mine.recall == pytest.approx(stock.stats[8], abs=1e-6)
