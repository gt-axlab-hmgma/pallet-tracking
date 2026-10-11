import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml
from torch.utils.data import Dataset, Subset

from palletvision_train.coco import CocoDetection
from palletvision_train.data import LetterboxedDataset, Sample
from palletvision_train.overfit import run_gate
from palletvision_train.synthetic import SyntheticBoxes

REPO = Path(__file__).resolve().parents[4]


def build(source: str, images: int, size: int) -> Dataset[Sample]:
    if source == "synthetic":
        return LetterboxedDataset(SyntheticBoxes(images, size), size)
    data = yaml.safe_load((REPO / "configs/data/loco.yaml").read_text())
    data["root"] = str(REPO / data["root"])
    data["annotations"] = {k: [str(REPO / f) for f in v] for k, v in data["annotations"].items()}
    classes = yaml.safe_load((REPO / "contracts/classes.yaml").read_text())["stage1"]
    reader = CocoDetection.from_config(data, "train", classes)
    return Subset(LetterboxedDataset(reader, size), range(images))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-overfit", description="Run the overfit gate.")
    p.add_argument("--source", choices=["loco", "synthetic"], default="loco")
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("runs/overfit"))
    args = p.parse_args(argv)

    result = run_gate(
        build(args.source, args.images, args.size),
        size=args.size, steps=args.steps, device=args.device, out_dir=args.out,
    )  # fmt: skip
    print(f"loss {result.loss_start:.2f} -> {result.loss_end:.2f} (down {result.loss_drop:.0%})")
    print(f"AP50 on the training images: {result.ap50:.3f}")
    for i, (objects, seen) in enumerate(zip(result.objects, result.confident, strict=True)):
        print(f"  image {i}: {objects:3d} objects, {seen:3d} confident detections")
    print(f"overlays (green = labels, red = detections) in {args.out}")
    if result.passed:
        print("GATE PASSED")
    else:
        print("GATE FAILED:", *result.failures, sep="\n  ")
        raise SystemExit(1)
