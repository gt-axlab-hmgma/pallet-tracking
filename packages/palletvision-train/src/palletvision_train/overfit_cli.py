import argparse
from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import Dataset, Subset

from palletvision_train.coco import loco_reader
from palletvision_train.data import LetterboxedDataset, Sample
from palletvision_train.overfit import run_gate
from palletvision_train.synthetic import SyntheticBoxes


def build(source: str, images: int, size: int) -> Dataset[Sample]:
    if source == "synthetic":
        return LetterboxedDataset(SyntheticBoxes(images, size), size)
    return Subset(LetterboxedDataset(loco_reader("train"), size), range(images))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-overfit", description="Run the overfit gate.")
    p.add_argument("--source", choices=["loco", "synthetic"], default="loco")
    p.add_argument("--images", type=int, default=8)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "musgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--out", type=Path, default=Path("runs/overfit"))
    args = p.parse_args(argv)

    result = run_gate(
        build(args.source, args.images, args.size),
        size=args.size, steps=args.steps, device=args.device, out_dir=args.out,
        optimizer_name=args.optimizer, lr=args.lr, momentum=args.momentum,
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
