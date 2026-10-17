"""Detection metrics."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from palletvision_model.loss.iou import pairwise_iou

Detection = tuple[int, float, np.ndarray]  # (image id, score, xyxy box)


def average_precision(
    detections: Sequence[Detection],
    ground_truths: Mapping[int, np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Average precision at one IoU threshold (all-point interpolation, as in PASCAL VOC).

    Detections are taken from the most confident to the least. Each is a hit if it overlaps a
    not-yet-matched ground-truth box of its image by at least ``iou_threshold``; a second
    detection of an already matched box counts as a false positive, so duplicates are punished.
    ``ground_truths`` maps an image id to its ``[n, 4]`` xyxy boxes. Returns 0 when there is
    nothing to find or nothing detected.
    """
    total = sum(len(boxes) for boxes in ground_truths.values())
    if total == 0 or not detections:
        return 0.0
    matched = {image: np.zeros(len(boxes), dtype=bool) for image, boxes in ground_truths.items()}
    hits = []
    for image, _, box in sorted(detections, key=lambda d: -d[1]):
        truth = ground_truths.get(image, np.zeros((0, 4), dtype=np.float32))
        hit = False
        if len(truth):
            overlap = pairwise_iou(torch.from_numpy(box)[None], torch.from_numpy(truth))[0].numpy()
            best = int(overlap.argmax())
            if overlap[best] >= iou_threshold and not matched[image][best]:
                matched[image][best] = True
                hit = True
        hits.append(hit)
    tp = np.cumsum(hits)
    recall = tp / total
    precision = tp / np.arange(1, len(hits) + 1)
    # make precision a non-increasing envelope, then sum it over the recall steps
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.nonzero(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))
