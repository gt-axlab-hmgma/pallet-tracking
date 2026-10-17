"""The overfit gate (guide step T4): can the whole pipeline memorise a handful of images?

Train on a few images with no augmentation until the loss is near zero, then check that the
model's own final detections (after decoding and the top-300 pick) land on the labelled boxes,
one confident box per object. If the model, assigner, loss and data reader disagree about boxes,
classes or coordinates, this is where it shows. Real training should not start until it passes.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from palletvision_model.config import SIZES
from palletvision_model.device import resolve_device
from palletvision_model.yolo26 import YOLO26
from palletvision_train.datasets.data import Sample, collate
from palletvision_train.evaluation.metrics import Detection, average_precision
from palletvision_train.training.trainer import train


@dataclass
class GateResult:
    loss_start: float
    loss_end: float
    ap50: float
    objects: list[int]  # labelled objects in each image
    confident: list[int]  # detections above the confidence threshold in each image
    failures: list[str] = field(default_factory=list)

    @property
    def loss_drop(self) -> float:
        return 1 - self.loss_end / self.loss_start

    @property
    def passed(self) -> bool:
        return not self.failures


def run_gate(
    dataset: Dataset[Sample],
    *,
    size: int,
    steps: int,
    device: str = "auto",
    lr: float = 1e-3,
    optimizer_name: str = "adamw",
    momentum: float = 0.9,
    out_dir: Path | str = "runs/overfit",
    conf: float = 0.3,
    min_loss_drop: float = 0.9,
    min_ap50: float = 0.95,
    max_duplicate_ratio: float = 1.25,
    seed: int = 0,
) -> GateResult:
    """Train a fresh model on every image of ``dataset`` for ``steps`` steps, then check it.

    The whole dataset is one batch, so a step is an epoch. The checks are the guide's: the loss
    falls by ``min_loss_drop``, AP50 on the training images reaches ``min_ap50``, and no image
    gets more than ``max_duplicate_ratio`` times its object count (plus one) of confident boxes.
    """
    out = Path(out_dir)
    torch.manual_seed(seed)
    model = YOLO26(SIZES["n"], input_size=size)
    n = len(dataset)  # type: ignore[arg-type]
    run = train(
        model, dataset, device=device, epochs=steps, batch_size=n, lr=lr, weight_decay=0.0,
        optimizer_name=optimizer_name, momentum=momentum,  # default: how the gate was validated
        amp=False, output_dir=out,
    )  # fmt: skip

    dev = resolve_device(device)
    model.to(dev).eval()
    batch = collate([dataset[i] for i in range(n)]).to(dev)
    with torch.no_grad():
        detections = model(batch.images).cpu().numpy()  # [n, 300, 6]

    truths = {i: batch.gt_boxes[i][batch.mask_gt[i, :, 0]].cpu().numpy() for i in range(n)}
    found: list[Detection] = [
        (i, float(row[4]), row[:4]) for i in range(n) for row in detections[i] if row[4] > 0.001
    ]
    confident = [int((detections[i][:, 4] > conf).sum()) for i in range(n)]
    result = GateResult(
        loss_start=run.losses[0],
        loss_end=float(np.mean(run.losses[-5:])),
        ap50=average_precision(found, truths),
        objects=[len(truths[i]) for i in range(n)],
        confident=confident,
    )
    if result.loss_drop < min_loss_drop:
        result.failures.append(f"loss fell only {result.loss_drop:.0%}, need {min_loss_drop:.0%}")
    if result.ap50 < min_ap50:
        result.failures.append(f"AP50 is {result.ap50:.3f}, need {min_ap50}")
    for i, (objects, seen) in enumerate(zip(result.objects, confident, strict=True)):
        if seen > max_duplicate_ratio * objects + 1:
            result.failures.append(f"image {i}: {seen} confident boxes for {objects} objects")
    save_overlays(batch.images.cpu().numpy(), truths, detections, conf, out)
    return result


def save_overlays(
    images: np.ndarray,
    truths: dict[int, np.ndarray],
    detections: np.ndarray,
    conf: float,
    out: Path,
) -> None:
    """Write ``overfit_<i>.jpg``: labels in green, confident detections in red."""
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        canvas = (images[i].transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8).copy()
        for x1, y1, x2, y2 in truths[i]:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for x1, y1, x2, y2, score, _ in detections[i]:
            if score > conf:
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.imwrite(str(out / f"overfit_{i}.jpg"), canvas)
