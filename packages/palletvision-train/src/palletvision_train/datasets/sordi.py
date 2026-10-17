"""Read SORDI synthetic scenes and mix them with the real data.

A SORDI scene folder holds ``images/<n>.jpg`` and ``labels/json/<n>.json`` (some scenes keep one
such folder per domain-randomisation level, ``DR-1``, ``DR-2``...). Each JSON file is a
list of objects with ``ObjectClassName`` and a pixel box (``Left``, ``Top``, ``Right``,
``Bottom``), already in the same ``(class, x1, y1, x2, y2)`` form the trainer wants, so the
only work is mapping SORDI's class names onto ours and pairing the files.

SORDI is far larger than LOCO and it is synthetic, so it must never swamp the real images:
``MixedSampler`` builds each epoch from every real image plus a random draw of synthetic ones.
"""

import hashlib
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Sampler

from palletvision_train.datasets.coco import MIN_SIDE, CocoDetection, ImageRecord

REPO = Path(__file__).resolve().parents[5]


SPLITS = ("train", "val", "test")


def split_of(key: str, fractions: Sequence[float] = (0.70, 0.15, 0.15)) -> str:
    """Which split (``train``, ``val`` or ``test``) the image called ``key`` belongs to.

    The key is hashed, so the answer is the same in every process and every run, and adding more
    scenes later never moves an image that already has a split (a seeded shuffle would). Use a
    key that names the image uniquely, like ``sordid-01/DR-2/123``.
    """
    if len(fractions) != len(SPLITS) or abs(sum(fractions) - 1) > 1e-6 or min(fractions) < 0:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1: {fractions}")
    digest = hashlib.sha256(key.encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    edge = 0.0
    for name, fraction in zip(SPLITS, fractions, strict=True):
        edge += fraction
        if point < edge:
            return name
    return SPLITS[-1]  # rounding error at the very top of the range


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
    root: Path | str,
    class_map: Mapping[str, str],
    classes: Sequence[str],
    *,
    split: str | None = None,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    key_prefix: str = "",
) -> list[ImageRecord]:
    """One record per image of a SORDI scene folder, sorted by image number.

    ``class_map`` maps SORDI class names to trained class names (others are dropped) and
    ``classes`` fixes the class index, as in ``load_records``. An image with no mapped object
    stays in as an empty example. Every image needs a label file and vice versa.

    With ``split`` only the images ``split_of`` puts in that split are kept; the image key is
    ``<key_prefix>/<image number>``, so give it the scene and level (``sordid-01/DR-2``).
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
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
        if split is not None and split_of(f"{key_prefix}/{name}", fractions) != split:
            continue
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


def sordi_records(
    roots: Sequence[str],
    class_map: Mapping[str, str],
    classes: Sequence[str],
    split: str,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    repo: Path = REPO,
) -> list[ImageRecord]:
    """The ``split`` images of every scene folder under ``roots`` (paths relative to ``repo``)."""
    records: list[ImageRecord] = []
    for root in roots:
        for folder in scene_folders(repo / root):
            prefix = folder.relative_to((repo / root).parent).as_posix()  # sordid-01/DR-2
            records += load_sordi_records(
                folder, class_map, classes, split=split, fractions=fractions, key_prefix=prefix
            )
    return records


def subsample(
    records: Sequence[ImageRecord], count: int | None, seed: int = 0
) -> list[ImageRecord]:
    """A fixed random ``count`` of the records, in their original order (all when ``None``)."""
    if count is None or count >= len(records):
        return list(records)
    keep = sorted(random.Random(seed).sample(range(len(records)), count))
    return [records[i] for i in keep]


def sordi_reader(split: str, repo: Path = REPO) -> CocoDetection:
    """The SORDI ``train``, ``val`` or ``test`` split exactly as ``configs/data/loco_sordi.yaml``
    lists it (every scene in ``sordi.roots``, split by ``sordi.fractions``)."""
    data = yaml.safe_load((repo / "configs/data/loco_sordi.yaml").read_text())
    classes = yaml.safe_load((repo / "contracts/classes.yaml").read_text())["stage1"]
    sordi = data["sordi"]
    records = sordi_records(
        sordi["roots"], sordi["class_map"], classes, split, sordi["fractions"], repo
    )
    return CocoDetection(records)


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


class UniformSampler(Sampler[int]):
    """Each epoch: a fresh random draw of ``per_epoch`` images, real and synthetic alike.

    No image is treated differently because of where it came from: every image of the pool has the
    same chance of being drawn, so the real/synthetic mix is just the mix of the pool (about 1% real
    for LOCO plus all of SORDI). ``per_epoch=None`` makes an epoch the whole pool.

    Randomness works as in ``MixedSampler``: torch's global generator by default (so a resumed run
    repeats the epochs), or with a ``seed`` an epoch seeded by ``seed + epoch`` (call ``set_epoch``)
    that every process draws identically before keeping its own slice.
    """

    def __init__(
        self,
        n: int,
        per_epoch: int | None = None,
        *,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if world_size > 1 and seed is None:
            raise ValueError("several processes need a seed, so that all of them draw one epoch")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in a group of {world_size}")
        self.n, self.per_epoch = n, min(per_epoch or n, n)
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0

    def __len__(self) -> int:
        """The images this process runs in an epoch."""
        return -(-self.per_epoch // self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = None
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order: list[int] = torch.randperm(self.n, generator=generator)[: self.per_epoch].tolist()
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
