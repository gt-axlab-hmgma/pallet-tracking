"""Read COCO-format detection data (LOCO) into images and ``(class, x1, y1, x2, y2)`` labels.

LOCO quirks this reader handles:
- ``file_name`` is only a basename, and image ids differ between the annotation files, so each
  image is located through its ``path`` field (with the leading ``/dataset/`` removed) and
  identified by that path, never by its id.
- Boxes are COCO ``[x, y, w, h]``; training wants corner coordinates.
- Several categories can collapse into one trained class through ``class_map``.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

DATASET_PREFIX = "/dataset/"  # how the annotation files spell the root of the image tree
MIN_SIDE = 1.0  # boxes with a side this small or smaller are annotation noise


@dataclass(frozen=True)
class ImageRecord:
    """One image and its labels. The image itself is read later, only when it is needed."""

    path: Path
    width: int
    height: int
    labels: np.ndarray  # [n, 5] float32: class index, x1, y1, x2, y2 in original pixels


def load_records(
    annotation_files: Sequence[Path | str],
    root: Path | str,
    class_map: Mapping[str, str],
    classes: Sequence[str],
    *,
    check_files: bool = True,
) -> list[ImageRecord]:
    """Parse annotation files into one list of records, in file order.

    ``class_map`` maps a dataset category name to a trained class name; categories it does not
    list are dropped. ``classes`` fixes the class index of each trained class. Crowd boxes and
    boxes with a side of 1 px or less are skipped. Images left with no boxes are kept, since an
    image without pallets is a valid example.
    """
    class_index = {name: i for i, name in enumerate(classes)}
    unknown = set(class_map.values()) - set(class_index)
    if unknown:
        raise ValueError(f"class_map targets {sorted(unknown)} are not in classes {list(classes)}")

    records: list[ImageRecord] = []
    first_seen: dict[Path, Path] = {}
    for file in map(Path, annotation_files):
        data = json.loads(file.read_text())
        category_class = {
            c["id"]: class_index[class_map[c["name"]]]
            for c in data["categories"]
            if c["name"] in class_map
        }
        rows_by_image: dict[int, list[list[float]]] = {}
        for ann in data["annotations"]:
            cls = category_class.get(ann["category_id"])
            if cls is None or ann.get("iscrowd"):
                continue
            x, y, w, h = ann["bbox"]
            if w <= MIN_SIDE or h <= MIN_SIDE:
                continue
            rows_by_image.setdefault(ann["image_id"], []).append([cls, x, y, x + w, y + h])
        for image in data["images"]:
            relative = image["path"].removeprefix(DATASET_PREFIX)
            path = Path(root) / relative
            if path in first_seen:
                raise ValueError(f"{relative} is listed in both {first_seen[path]} and {file}")
            first_seen[path] = file
            labels = np.array(rows_by_image.get(image["id"], []), dtype=np.float32).reshape(-1, 5)
            records.append(ImageRecord(path, image["width"], image["height"], labels))

    if check_files:
        missing = [r.path for r in records if not r.path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images are missing, first: {missing[0]}")
    return records


class CocoDetection(Dataset[tuple[np.ndarray, np.ndarray]]):
    """Item ``i`` is ``(image, labels)``: an ``[H, W, 3]`` uint8 image in OpenCV's BGR order
    (what ``letterbox`` expects) and its ``[n, 5]`` float32 labels in original pixels."""

    def __init__(self, records: Sequence[ImageRecord]) -> None:
        self.records = list(records)

    @classmethod
    def from_config(
        cls, data: Mapping[str, object], split: str, classes: Sequence[str]
    ) -> "CocoDetection":
        """Build one split (``train`` or ``val``) from a ``configs/data/*.yaml`` mapping."""
        annotations = data["annotations"]
        assert isinstance(annotations, Mapping)
        files = [Path(f) for f in annotations[split]]
        class_map = data["class_map"]
        assert isinstance(class_map, Mapping)
        return cls(load_records(files, str(data["root"]), class_map, classes))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not read {record.path}")
        if image.shape[:2] != (record.height, record.width):
            # e.g. an EXIF rotation swapped the sides: the labels would land in the wrong place
            raise ValueError(
                f"{record.path} is {image.shape[1]}x{image.shape[0]} on disk but "
                f"{record.width}x{record.height} in the annotations"
            )
        return image, record.labels
