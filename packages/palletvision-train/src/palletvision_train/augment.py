"""Training augmentation: mosaic, random scale and shift, colour jitter and a horizontal flip.

The detector sees only a few thousand real images, so it memorises them. Augmentation shows it a
different version of each image every time. The recipe is the one in the YOLO26 report
(Table S3, the n model): mosaic 1.0, scale 0.5, translate 0.1, flip 0.5, HSV 0.02 / 0.7 / 0.4,
with mosaic switched off for the last ``close_mosaic`` epochs so the model ends on whole images.
No rotation or shear (0 in the report), no mixup (0 for n). The report's copy-paste needs object
masks, which LOCO and SORDI do not have, so it is not used.

Every step moves the labels with the pixels, and a box is kept only while it still marks its
object: at least ``MIN_SIDE`` pixels in both directions, more than ``MIN_VISIBLE`` of the object
still in view, and not a hair-thin sliver. All randomness comes from Python's ``random``, which
torch seeds separately in every data-loader worker, so seeded runs repeat.
"""

import multiprocessing
import random
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from torch.utils.data import Dataset

from palletvision_preprocess.letterbox import PAD_VALUE
from palletvision_train.transforms import letterbox_sample

Sample = tuple[np.ndarray, np.ndarray]  # (BGR uint8 image, [n, 5] labels: class, x1, y1, x2, y2)

MIN_SIDE = 2.0  # a box thinner than this (in output pixels) is no longer an object
MIN_VISIBLE = 0.1  # keep a box only if more than this fraction of the object is still in view
MAX_ASPECT = 100.0  # and only if it is not an extreme sliver


@dataclass(frozen=True)
class AugmentConfig:
    """The strength of each step. Defaults are the report's values for the n model."""

    mosaic: float = 1.0  # probability of building each sample from 4 images
    scale: float = 0.5  # random zoom between 1 - scale and 1 + scale
    translate: float = 0.1  # random shift of up to this fraction of the image
    fliplr: float = 0.5  # probability of a left-right flip
    hsv_h: float = 0.02  # random hue, saturation and value gains
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    close_mosaic: int = 8  # no mosaic in the last N epochs (the trainer applies this)


def mosaic(samples: list[Sample], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Join 4 images into one ``2*size`` square around a random centre point.

    Each image is shrunk so its longer side is ``size`` and placed in one quadrant of the centre
    point. A box is cut to the part of its image that is actually visible in the quadrant, so it
    never covers pixels that belong to a neighbour. Returns the canvas and labels as ``[n, 6]``:
    the usual five columns plus the box's full area before any cutting, which the later steps
    need to tell how much of the object has been lost.
    """
    canvas = np.full((2 * size, 2 * size, 3), PAD_VALUE, dtype=np.uint8)
    xc, yc = (int(random.uniform(size // 2, 3 * size // 2)) for _ in range(2))
    rows: list[np.ndarray] = []
    for i, (image, labels) in enumerate(samples):
        h0, w0 = image.shape[:2]
        ratio = size / max(h0, w0)
        w, h = max(round(w0 * ratio), 1), max(round(h0 * ratio), 1)
        tile = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if i == 0:  # top left
            xa1, ya1, xa2, ya2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            xb1, yb1 = w - (xa2 - xa1), h - (ya2 - ya1)
        elif i == 1:  # top right
            xa1, ya1, xa2, ya2 = xc, max(yc - h, 0), min(xc + w, 2 * size), yc
            xb1, yb1 = 0, h - (ya2 - ya1)
        elif i == 2:  # bottom left
            xa1, ya1, xa2, ya2 = max(xc - w, 0), yc, xc, min(2 * size, yc + h)
            xb1, yb1 = w - (xa2 - xa1), 0
        else:  # bottom right
            xa1, ya1, xa2, ya2 = xc, yc, min(xc + w, 2 * size), min(2 * size, yc + h)
            xb1, yb1 = 0, 0
        canvas[ya1:ya2, xa1:xa2] = tile[yb1 : yb1 + (ya2 - ya1), xb1 : xb1 + (xa2 - xa1)]
        if len(labels) == 0:
            continue
        boxes = labels[:, 1:5].astype(np.float32).copy()
        boxes[:, [0, 2]] *= w / w0
        boxes[:, [1, 3]] *= h / h0
        boxes[:, [0, 2]] += xa1 - xb1
        boxes[:, [1, 3]] += ya1 - yb1
        full = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(xa1, xa2)  # only the visible part of the tile
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(ya1, ya2)
        visible = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        rows.append(np.column_stack([labels[:, 0], boxes, full])[visible])
    out = np.concatenate(rows) if rows else np.zeros((0, 6))
    return canvas, out.astype(np.float32)


def scale_and_shift(
    image: np.ndarray,
    labels: np.ndarray,
    out_size: int,
    scale: float,
    translate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zoom the picture about its centre by a random factor and shift it, into ``out_size``.

    The centre of ``image`` lands within ``translate`` of the centre of the output, so a mosaic
    twice the output size is cut down to a random window. Pixels that fall outside are grey.
    ``labels`` is the ``[n, 6]`` array of ``mosaic``; the result is the usual ``[n, 5]``, with
    boxes cut to the output and dropped when they no longer mark their object (see the module
    note). With ``scale`` and ``translate`` both 0 and ``image`` already ``out_size`` square, the
    picture and labels come back unchanged.
    """
    zoom = random.uniform(1 - scale, 1 + scale)
    shift_x = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    shift_y = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    h, w = image.shape[:2]
    # x_out = zoom * (x - w / 2) + shift_x
    matrix = np.array(
        [[zoom, 0, shift_x - zoom * w / 2], [0, zoom, shift_y - zoom * h / 2]], dtype=np.float64
    )
    identity = zoom == 1 and (h, w) == (out_size, out_size) and shift_x == w / 2 == shift_y == h / 2
    if not identity:
        image = cv2.warpAffine(
            image, matrix, (out_size, out_size), flags=cv2.INTER_LINEAR,
            borderValue=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
        )  # fmt: skip
    if len(labels) == 0:
        return image, np.zeros((0, 5), dtype=np.float32)

    boxes = labels[:, 1:5].astype(np.float64).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * zoom + matrix[0, 2]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * zoom + matrix[1, 2]
    full = labels[:, 5].astype(np.float64) * zoom * zoom  # the object's area at this zoom
    boxes = boxes.clip(0, out_size)
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (
        (bw > MIN_SIDE) & (bh > MIN_SIDE)
        & (bw * bh / (full + 1e-9) > MIN_VISIBLE)
        & (np.maximum(bw / np.maximum(bh, 1e-9), bh / np.maximum(bw, 1e-9)) < MAX_ASPECT)
    )  # fmt: skip
    out = np.column_stack([labels[:, 0], boxes])[keep]
    return image, out.astype(np.float32)


def jitter_colour(image: np.ndarray, h: float, s: float, v: float) -> np.ndarray:
    """Randomly change hue, saturation and brightness by up to the given gains (0 = off)."""
    if not (h or s or v):
        return image
    gains = np.array([random.uniform(-1, 1) * g + 1 for g in (h, s, v)])
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float64)
    lut_h = ((x * gains[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def flip_lr(image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the picture left to right and its boxes with it."""
    width = image.shape[1]
    flipped = labels.copy()
    flipped[:, [1, 3]] = width - labels[:, [3, 1]]
    return np.ascontiguousarray(image[:, ::-1]), flipped


class AugmentedDataset(Dataset[Sample]):
    """A training set that returns a freshly augmented ``size x size`` canvas on every access.

    Same output as ``LetterboxedDataset`` (a uint8 BGR canvas and ``[n, 5]`` labels), so it
    drops in where that was used. ``base`` yields ``(image, labels)`` in original pixels.
    ``partners`` picks the images mosaic mixes in (default: any image of ``base``); the
    trainer's mixed real/synthetic epochs use it to keep the mixture they asked for.
    """

    def __init__(
        self,
        base: Dataset[Sample],
        size: int = 640,
        config: AugmentConfig | None = None,
        partners: Callable[[], int] | None = None,
    ) -> None:
        self.base, self.size = base, size
        self.config = config or AugmentConfig()
        self.partners = partners  # a picklable callable, or None for any image of ``base``
        # one flag shared with every data-loader worker, so the trainer can switch mosaic off
        # for the last epochs without restarting them
        self._mosaic_on = multiprocessing.Value("b", 1)

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def set_mosaic(self, on: bool) -> None:
        self._mosaic_on.value = int(on)

    def __getitem__(self, index: int) -> Sample:
        c = self.config
        if self._mosaic_on.value and random.random() < c.mosaic:
            pick = self.partners or (lambda: random.randrange(len(self)))
            samples = [self.base[index]] + [self.base[pick()] for _ in range(3)]
            image, labels = mosaic(samples, self.size)
        else:
            image, plain = letterbox_sample(*self.base[index], self.size)
            area = (plain[:, 3] - plain[:, 1]) * (plain[:, 4] - plain[:, 2])
            image, labels = image, np.column_stack([plain, area]).astype(np.float32)
        image, labels = scale_and_shift(image, labels, self.size, c.scale, c.translate)
        image = jitter_colour(image, c.hsv_h, c.hsv_s, c.hsv_v)
        if random.random() < c.fliplr:
            image, labels = flip_lr(image, labels)
        return image, labels
