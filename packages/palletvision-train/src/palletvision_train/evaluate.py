"""Score a model on a labelled dataset the standard COCO way.

The model runs on every image (letterboxed, nothing else), its detections are mapped back to
the original image's pixels, and ``pycocotools`` compares them with the labels. Two details
matter for the score to mean anything:
- the confidence threshold is very low (0.001), because mAP rewards a ranked list and a
  threshold like 0.25 would throw away the tail that gives the curve its shape;
- up to 300 detections per image are kept (the model's own limit), because LOCO images can hold
  more than 100 objects and COCO's default would cut them off.
The deployment confidence threshold is chosen separately, from the F1 curve (``best_conf``).
"""

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset

from palletvision_model.device import resolve_device
from palletvision_model.yolo26 import YOLO26
from palletvision_preprocess import LetterboxMeta, letterbox_image, restore_boxes, to_blob
from palletvision_train.coco import CocoDetection

MAX_DETS = 300
MIN_SCORE = 0.001


@dataclass(frozen=True)
class EvalResult:
    map50_95: float  # the headline COCO number: AP averaged over IoU 0.5, 0.55, ... 0.95
    map50: float
    map75: float
    ap_small: float  # by object size in original pixels: under 32^2, 32^2 to 96^2, over 96^2
    ap_medium: float
    ap_large: float
    recall: float  # average recall with up to 300 detections per image
    best_f1: float  # the highest F1 over all confidence thresholds, at IoU 0.5
    best_conf: float  # the confidence threshold that gives it: the one to deploy with
    precision_at_best: float
    recall_at_best: float
    images: int
    objects: int
    detections: int


class _Images(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Each image letterboxed, with what is needed to map its detections back."""

    def __init__(self, reader: CocoDetection, size: int) -> None:
        self.reader, self.size = reader, size

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, _ = self.reader[index]
        canvas, meta = letterbox_image(image, self.size)
        placement = torch.tensor([meta.scale, meta.pad_x, meta.pad_y, *image.shape[:2]])
        return torch.from_numpy(to_blob(canvas))[0], placement.float(), index


def detect(
    model: YOLO26,
    reader: CocoDetection,
    *,
    size: int,
    device: str | torch.device = "auto",
    batch_size: int = 16,
    num_workers: int = 0,
) -> list[dict[str, float | int | list[float]]]:
    """Run ``model`` on every image; return COCO-style results in original image pixels."""
    dev = resolve_device(device) if isinstance(device, str) else device
    model.to(dev).eval()
    loader = DataLoader(_Images(reader, size), batch_size=batch_size, num_workers=num_workers)
    results: list[dict[str, float | int | list[float]]] = []
    with torch.no_grad():
        for blobs, placements, indices in loader:
            output = model(blobs.to(dev)).cpu().numpy()  # [B, 300, 6] in letterboxed pixels
            for rows, placement, index in zip(
                output, placements.tolist(), indices.tolist(), strict=True
            ):
                scale, pad_x, pad_y, height, width = placement
                rows = rows[rows[:, 4] > MIN_SCORE]
                meta = LetterboxMeta(scale, pad_x, pad_y)
                boxes = restore_boxes(rows[:, :4], meta, (int(height), int(width)))
                for (x1, y1, x2, y2), row in zip(boxes, rows, strict=True):
                    if x2 > x1 and y2 > y1:  # drop anything the clipping squeezed to nothing
                        results.append(
                            {
                                "image_id": index,
                                "category_id": int(row[5]) + 1,  # COCO ids start at 1
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(row[4]),
                            }
                        )
    return results


def coco_ground_truth(reader: CocoDetection, categories: int = 1) -> COCO:
    """The reader's labels (already mapped to the trained classes) as a COCO ground truth.

    Image ids are positions in ``reader.records``. Objects of dropped categories are not in it,
    so a detection on one counts as a false positive, which is what training taught the model.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, record in enumerate(reader.records):
        images.append({"id": image_id, "width": record.width, "height": record.height})
        for cls, x1, y1, x2, y2 in record.labels.tolist():
            annotations.append(
                {
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                }
            )  # fmt: skip
    gt = COCO()
    gt.dataset = {  # pyright: ignore[reportAttributeAccessIssue]  (the stubs are too narrow)
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c + 1, "name": str(c)} for c in range(categories)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def f1_curve(ev: COCOeval) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confidence thresholds with the precision, recall and F1 you get by keeping everything
    scoring at least that high, at IoU 0.5 over all objects (taken from COCOeval's matching)."""
    scores, hits, objects = [], [], 0
    for entry in ev.evalImgs:
        if entry is None or entry["aRng"] != ev.params.areaRng[0]:
            continue
        objects += int((~np.asarray(entry["gtIgnore"], dtype=bool)).sum())
        keep = ~np.asarray(entry["dtIgnore"][0], dtype=bool)  # row 0 is IoU 0.5
        scores.append(np.asarray(entry["dtScores"])[keep])
        hits.append((np.asarray(entry["dtMatches"][0]) > 0)[keep])
    if not scores or objects == 0:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    all_scores, all_hits = np.concatenate(scores), np.concatenate(hits)
    order = np.argsort(-all_scores, kind="stable")
    true_positives = np.cumsum(all_hits[order])
    precision = true_positives / np.arange(1, len(order) + 1)
    recall = true_positives / objects
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return all_scores[order], precision, recall, f1


def mean_precision(ev: COCOeval, area: int = 0, iou: int | None = None) -> float:
    """AP from COCOeval's accuracy table, at 300 detections per image.

    ``area`` indexes the size ranges (0 all, 1 small, 2 medium, 3 large); ``iou`` picks one IoU
    threshold (0 is 0.5, 5 is 0.75) or averages over all ten when ``None``. This is what
    ``COCOeval.summarize`` does, except that it hard-codes 100 detections for its headline
    number, which would read as "not computed" (-1) with our limit of 300.
    """
    precision = ev.eval["precision"][:, :, :, area, 2]  # [IoU, recall, category, area, maxDets]
    precision = precision if iou is None else precision[iou]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else 0.0


def mean_recall(ev: COCOeval) -> float:
    recall = ev.eval["recall"][:, :, 0, 2]  # [IoU, category, area, maxDets]
    valid = recall[recall > -1]
    return float(valid.mean()) if valid.size else 0.0


def score(results: list[dict[str, float | int | list[float]]], gt: COCO) -> EvalResult:
    """Compare detections with the ground truth and collect the numbers."""
    objects = len(gt.dataset["annotations"])
    zero = EvalResult(
        map50_95=0.0, map50=0.0, map75=0.0, ap_small=0.0, ap_medium=0.0, ap_large=0.0, recall=0.0,
        best_f1=0.0, best_conf=0.0, precision_at_best=0.0, recall_at_best=0.0,
        images=len(gt.dataset["images"]), objects=objects, detections=len(results),
    )  # fmt: skip
    if not results or objects == 0:
        return zero
    with contextlib.redirect_stdout(io.StringIO()):
        detections = gt.loadRes(results)  # pyright: ignore[reportArgumentType]  (a list is fine)
        ev = COCOeval(gt, detections, "bbox")
        ev.params.maxDets = [1, 10, MAX_DETS]
        ev.evaluate()
        ev.accumulate()
    thresholds, precision, recall, f1 = f1_curve(ev)
    best = int(f1.argmax()) if len(f1) else 0
    return EvalResult(
        map50_95=mean_precision(ev), map50=mean_precision(ev, iou=0),
        map75=mean_precision(ev, iou=5),
        ap_small=mean_precision(ev, area=1), ap_medium=mean_precision(ev, area=2),
        ap_large=mean_precision(ev, area=3), recall=mean_recall(ev),
        best_f1=float(f1[best]) if len(f1) else 0.0,
        best_conf=float(thresholds[best]) if len(f1) else 0.0,
        precision_at_best=float(precision[best]) if len(f1) else 0.0,
        recall_at_best=float(recall[best]) if len(f1) else 0.0,
        images=len(gt.dataset["images"]), objects=objects, detections=len(results),
    )  # fmt: skip


def evaluate(
    model: YOLO26,
    reader: CocoDetection,
    *,
    size: int = 640,
    device: str | torch.device = "auto",
    batch_size: int = 16,
    num_workers: int = 0,
    categories: int = 1,
) -> EvalResult:
    """Score ``model`` on every image of ``reader`` (letterbox only, no augmentation).

    ``categories`` is the number of classes; with more than one, the mAP is the mean over classes.
    """
    results = detect(
        model, reader, size=size, device=device, batch_size=batch_size, num_workers=num_workers
    )
    return score(results, coco_ground_truth(reader, categories))
