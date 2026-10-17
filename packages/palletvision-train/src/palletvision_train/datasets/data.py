"""Turn the LOCO reader's items into the padded batches the assigner and loss expect.

``CocoDetection`` yields one image and its labels at a time, with a different number of boxes
per image. ``LetterboxedDataset`` fits each into the model's square input, and ``collate``
stacks a list of them into rectangular tensors, padding the labels to the largest object count
in the batch and recording which entries are real in a mask.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from palletvision_preprocess import to_blob
from palletvision_train.augmentation.transforms import letterbox_sample

Sample = tuple[np.ndarray, np.ndarray]  # (BGR uint8 image, [n, 5] labels as class, x1, y1, x2, y2)


class Batch(NamedTuple):
    """One training batch. ``M`` is the most objects in any image of the batch.

    A ``NamedTuple`` so a ``DataLoader`` can pin and move it like any tuple of tensors.
    """

    images: torch.Tensor  # [B, 3, S, S] float32, RGB, in [0, 1]: the model's input
    gt_labels: torch.Tensor  # [B, M, 1] long, class index of each object (0 where padded)
    gt_boxes: torch.Tensor  # [B, M, 4] float32, xyxy in letterboxed pixels (0 where padded)
    mask_gt: torch.Tensor  # [B, M, 1] bool, True for a real object and False for padding

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "Batch":
        return Batch(*(t.to(device, non_blocking=non_blocking) for t in self))


class LetterboxedDataset(Dataset[Sample]):
    """Fit every image of ``base`` (and its labels) into a ``size x size`` square.

    Yields the uint8 BGR canvas, not a tensor, so augmentation can still work on it later.
    """

    def __init__(self, base: Dataset[Sample], size: int = 640) -> None:
        self.base, self.size = base, size

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Sample:
        image, labels = self.base[index]
        return letterbox_sample(image, labels, self.size)


def collate(samples: Sequence[Sample]) -> Batch:
    """Stack ``(canvas, labels)`` pairs into a ``Batch``. Pass it to ``DataLoader(collate_fn=)``."""
    # The same conversion inference applies, so the model sees identical numbers either way.
    images = torch.from_numpy(np.concatenate([to_blob(canvas) for canvas, _ in samples]))

    batch, most = len(samples), max(len(labels) for _, labels in samples)
    gt_labels = torch.zeros(batch, most, 1, dtype=torch.long)
    gt_boxes = torch.zeros(batch, most, 4)
    mask_gt = torch.zeros(batch, most, 1, dtype=torch.bool)
    for i, (_, labels) in enumerate(samples):
        n = len(labels)
        gt_labels[i, :n, 0] = torch.from_numpy(labels[:, 0]).long()
        gt_boxes[i, :n] = torch.from_numpy(labels[:, 1:5])
        mask_gt[i, :n] = True
    return Batch(images, gt_labels, gt_boxes, mask_gt)
