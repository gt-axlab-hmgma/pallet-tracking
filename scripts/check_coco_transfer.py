"""Do converted YOLO26 COCO weights reproduce their COCO accuracy in our model?

    uv run python scripts/check_coco_transfer.py [--weights W.pt] [--images 1000]

Scores our model, loaded with the converted weights, on the first N COCO val2017 images and compares
it with a random-weight control. The images and the annotation file are fetched by HTTP range
requests (not the whole 1 GB), into data/coco_val_check/ (git-ignored). COCO is used for this check
only; its image licences are a gray area for commercial use (docs/pretraining.md).

Reference: the YOLO26n checkpoint records mAP50-95 0.401 on all 5,000 val images; ours gave 0.410
on the first 1,000 (subset noise), and 0.000 for random weights.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.device import resolve_device
from palletvision_model.yolo26 import YOLO26
from palletvision_train.coco import CocoDetection, ImageRecord
from palletvision_train.evaluate import evaluate

BASE = "http://images.cocodataset.org"
CACHE = Path(__file__).resolve().parents[1] / "data" / "coco_val_check"


class Ranged(io.RawIOBase):
    """A seekable view of a remote file that reads only the bytes asked for."""

    def __init__(self, url: str) -> None:
        self.url, self.position = url, 0
        head = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(head).headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def readinto(self, buffer: bytearray) -> int:  # type: ignore[override]
        if self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        data = urllib.request.urlopen(request).read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def records(count: int) -> list[ImageRecord]:
    CACHE.mkdir(parents=True, exist_ok=True)
    annotations = CACHE / "instances_val2017.json"
    if not annotations.exists():
        archive = zipfile.ZipFile(
            io.BufferedReader(Ranged(f"{BASE}/annotations/annotations_trainval2017.zip"), 1 << 20)
        )
        annotations.write_bytes(archive.read("annotations/instances_val2017.json"))
    data = json.loads(annotations.read_text())
    index = {c: i for i, c in enumerate(sorted(c["id"] for c in data["categories"]))}
    images = sorted(data["images"], key=lambda i: i["id"])[:count]
    boxes: dict[int, list[list[float]]] = {i["id"]: [] for i in images}
    for a in data["annotations"]:
        if a["image_id"] in boxes and not a.get("iscrowd"):
            x, y, w, h = a["bbox"]
            if w > 1 and h > 1:
                boxes[a["image_id"]].append([index[a["category_id"]], x, y, x + w, y + h])
    missing = [i["file_name"] for i in images if not (CACHE / i["file_name"]).exists()]
    if missing:
        print(f"fetching {len(missing)} images (a few minutes)")
        archive = zipfile.ZipFile(io.BufferedReader(Ranged(f"{BASE}/zips/val2017.zip"), 1 << 20))
        for name in missing:
            (CACHE / name).write_bytes(archive.read(f"val2017/{name}"))
    return [
        ImageRecord(CACHE / i["file_name"], i["width"], i["height"],
                    np.array(boxes[i["id"]], np.float32).reshape(-1, 5))
        for i in images
    ]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=Path("data/yolo26n_coco_converted.pt"))
    parser.add_argument("--images", type=int, default=1000)
    args = parser.parse_args()
    device = resolve_device("auto")
    data = records(args.images)
    print(f"{len(data)} images, {sum(len(r.labels) for r in data)} objects")
    cfg = SIZES["n"]
    config = ModelConfig(cfg.depth_multiple, cfg.width_multiple, num_classes=80, reg_max=1)
    model = YOLO26(config, input_size=640)
    model.load_state_dict(torch.load(args.weights)["model"], strict=True)
    result = evaluate(model, CocoDetection(data), size=640, device=device, categories=80)
    print(f"converted weights: mAP50-95 {result.map50_95:.4f}  mAP50 {result.map50:.4f}")
    torch.manual_seed(0)
    control = evaluate(YOLO26(config, input_size=640), CocoDetection(data[:200]), size=640,
                       device=device, categories=80)  # fmt: skip
    print(f"random-weight control (200 images): mAP50-95 {control.map50_95:.4f}")


if __name__ == "__main__":
    main()
