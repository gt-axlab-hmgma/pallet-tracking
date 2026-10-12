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
import random
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
    stay the majority.

    By default the randomness comes from torch's global generator, which the checkpoints save,
    so a resumed run draws the same epochs as an uninterrupted one. With several processes
    (``world_size`` > 1) every process must draw the *same* epoch and then keep its own slice,
    so pass a ``seed``: the epoch is then drawn from a generator seeded by ``seed + epoch``
    (call ``set_epoch`` each epoch), identical in every process and reproducible on resume.
    Slices are made equal in length by repeating images from the start, as torch's
    ``DistributedSampler`` does, so no process runs out of batches before the others.
    """

    def __init__(
        self,
        n_real: int,
        n_synthetic: int,
        share: float = 0.3,
        *,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not 0 <= share < 1:
            raise ValueError(f"share must be in [0, 1), got {share}")
        if world_size > 1 and seed is None:
            raise ValueError("several processes need a seed, so that all of them draw one epoch")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in a group of {world_size}")
        self.n_real, self.n_synthetic = n_real, n_synthetic
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.n_drawn = round(n_real * share / (1 - share))
        if self.n_drawn > n_synthetic:
            raise ValueError(
                f"a synthetic share of {share} needs {self.n_drawn} synthetic images "
                f"but the pool has {n_synthetic}"
            )

    def __len__(self) -> int:
        """The images this process runs in an epoch."""
        return -(-(self.n_real + self.n_drawn) // self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = None
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
        synthetic = (
            self.n_real + torch.randperm(self.n_synthetic, generator=generator)[: self.n_drawn]
        )
        shuffled = torch.cat([torch.arange(self.n_real), synthetic])
        order: list[int] = shuffled[torch.randperm(len(shuffled), generator=generator)].tolist()
        if self.world_size > 1:
            total = len(self) * self.world_size
            order = (order * -(-total // len(order)))[:total]  # repeat from the start to fill up
            order = order[self.rank : total : self.world_size]
        yield from order


class MixedPartners:
    """Picks the images a mosaic mixes in, with the epoch's real/synthetic mixture.

    Indices follow ``ConcatDataset([real, synthetic])``. Drawing partners uniformly from the
    whole pool would make nearly every mosaic tile synthetic, so each partner is a real image
    with probability ``1 - share`` and a synthetic one otherwise, like the epoch itself.
    """

    def __init__(self, n_real: int, n_synthetic: int, share: float = 0.3) -> None:
        self.n_real, self.n_synthetic, self.share = n_real, n_synthetic, share

    def __call__(self) -> int:
        if self.n_synthetic and random.random() < self.share:
            return self.n_real + random.randrange(self.n_synthetic)
        return random.randrange(self.n_real)
