"""Read SORDI synthetic scenes and mix them with the real data.

A SORDI scene folder holds ``images/<n>.jpg`` and ``labels/json/<n>.json`` (some scenes keep one
such folder per domain-randomisation level, ``DR-1``, ``DR-2``...). Each JSON file is a
list of objects with ``ObjectClassName`` and a pixel box (``Left``, ``Top``, ``Right``,
``Bottom``), already in the same ``(class, x1, y1, x2, y2)`` form the trainer wants, so the
only work is mapping SORDI's class names onto ours and pairing the files.

SORDI is far larger than LOCO and it is synthetic, so it must never swamp the real images:
``MixedSampler`` builds each epoch from every real image plus a random draw of synthetic ones.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Sampler

from palletvision_train.coco import MIN_SIDE, ImageRecord


def scene_folders(root: Path | str) -> list[Path]:
    """The folders under ``root`` that hold ``images/`` and ``labels/json/``.

    Scene 02 is one such folder. Most scenes split into ``DR-1``, ``DR-2`` (and ``DR-3``) folders,
    the domain-randomisation levels, each laid out the same way: ``root`` then yields all of them.
    """
    root = Path(root)

    def is_scene(folder: Path) -> bool:
        return (folder / "images").is_dir() and (folder / "labels" / "json").is_dir()

    found = [root] if is_scene(root) else sorted(p for p in root.glob("DR-*") if is_scene(p))
    if not found:
        raise FileNotFoundError(
            f"{root} has no images/ and labels/json/, directly or in DR-* folders"
        )
    return found


def load_sordi_records(
    root: Path | str, class_map: Mapping[str, str], classes: Sequence[str]
) -> list[ImageRecord]:
    """One record per image of a SORDI scene folder, sorted by image number.

    ``class_map`` maps SORDI class names to trained class names (others are dropped) and
    ``classes`` fixes the class index, as in ``load_records``. An image with no mapped object
    stays in as an empty example. Every image needs a label file and vice versa.
    """
    class_index = {name: i for i, name in enumerate(classes)}
    unknown = set(class_map.values()) - set(class_index)
    if unknown:
        raise ValueError(f"class_map targets {sorted(unknown)} are not in classes {list(classes)}")

    root = Path(root)
    images = {p.stem: p for p in (root / "images").glob("*.jpg")}
    labels = {p.stem: p for p in (root / "labels" / "json").glob("*.json")}
    if images.keys() != labels.keys():
        only_images, only_labels = images.keys() - labels.keys(), labels.keys() - images.keys()
        raise ValueError(
            f"{root}: {len(only_images)} images without labels (e.g. {sorted(only_images)[:3]}), "
            f"{len(only_labels)} labels without images (e.g. {sorted(only_labels)[:3]})"
        )
    if not images:
        raise FileNotFoundError(f"{root} has no images/*.jpg and labels/json/*.json")

    records: list[ImageRecord] = []
    for name in sorted(images, key=lambda s: (len(s), s)):  # 2 before 10
        rows = []
        for obj in json.loads(labels[name].read_text()):
            cls = class_map.get(obj["ObjectClassName"])
            if cls is None:
                continue
            x1, y1, x2, y2 = obj["Left"], obj["Top"], obj["Right"], obj["Bottom"]
            if x2 - x1 <= MIN_SIDE or y2 - y1 <= MIN_SIDE:
                continue
            rows.append([class_index[cls], x1, y1, x2, y2])
        with Image.open(images[name]) as image:  # reads the header only
            width, height = image.size
        array = np.array(rows, dtype=np.float32).reshape(-1, 5)
        records.append(ImageRecord(images[name], width, height, array))
    return records


class MixedSampler(Sampler[int]):
    """Each epoch: every real image once, plus enough synthetic images to be ``share`` of it.

    Indices ``0 .. n_real-1`` are the real images and ``n_real ..`` the synthetic ones (the
    order of a ``ConcatDataset([real, synthetic])``). The synthetic images are a fresh random
    draw without repeats every epoch, so a large pool is used up gradually while real images
    stay the majority. Randomness comes from torch's global generator, which the checkpoints
    save, so a resumed run draws the same epochs as an uninterrupted one.
    """

    def __init__(self, n_real: int, n_synthetic: int, share: float = 0.3) -> None:
        if not 0 <= share < 1:
            raise ValueError(f"share must be in [0, 1), got {share}")
        self.n_real, self.n_synthetic = n_real, n_synthetic
        self.n_drawn = round(n_real * share / (1 - share))
        if self.n_drawn > n_synthetic:
            raise ValueError(
                f"a synthetic share of {share} needs {self.n_drawn} synthetic images "
                f"but the pool has {n_synthetic}"
            )

    def __len__(self) -> int:
        return self.n_real + self.n_drawn

    def __iter__(self) -> Iterator[int]:
        synthetic = self.n_real + torch.randperm(self.n_synthetic)[: self.n_drawn]
        order = torch.cat([torch.arange(self.n_real), synthetic])
        yield from order[torch.randperm(len(order))].tolist()
