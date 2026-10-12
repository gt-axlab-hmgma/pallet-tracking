import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.yolo26 import YOLO26
from palletvision_train.coco import CocoDetection, loco_reader
from palletvision_train.evaluate import evaluate
from palletvision_train.sordi import sordi_reader, subsample


def load_model(checkpoint: Path) -> tuple[YOLO26, dict[str, int | None], int]:
    """Rebuild the model a checkpoint was trained as (its saved config), then load the weights.

    Older checkpoints hold only the weights and a step count, so ``epoch`` may be ``None``.
    """
    state = torch.load(checkpoint, map_location="cpu")
    saved = state.get("config", {}).get("model")
    if saved:
        cfg = ModelConfig(
            saved["depth_multiple"], saved["width_multiple"],
            num_classes=saved["num_classes"], reg_max=saved["reg_max"],
        )  # fmt: skip
        size = int(saved["input_size"])
    else:  # a checkpoint without a config: assume the default n model at 640
        cfg, size = SIZES["n"], 640
    model = YOLO26(cfg, input_size=size)
    model.load_state_dict(state["model"])
    return model, {"epoch": state.get("epoch"), "steps": state.get("steps")}, size


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-eval", description="Score a checkpoint with COCO mAP.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test", "train"], default="val")
    p.add_argument(
        "--source",
        choices=["loco", "sordi"],
        default="loco",
        help="real LOCO images, or the SORDI synthetic split (configs/data/loco_sordi.yaml)",
    )
    p.add_argument("--size", type=int, help="input size (default: the one it was trained at)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="score only the first N images")
    p.add_argument("--out", type=Path, help="write the numbers here as JSON")
    args = p.parse_args(argv)

    model, where, trained_size = load_model(args.checkpoint)
    reader = loco_reader(args.split) if args.source == "loco" else sordi_reader(args.split)
    if args.limit is not None:  # the first N for LOCO; a fixed random N for SORDI (one scene first)
        reader = CocoDetection(
            reader.records[: args.limit]
            if args.source == "loco"
            else subsample(reader.records, args.limit)
        )
    r = evaluate(
        model, reader, size=args.size or trained_size, device=args.device,
        batch_size=args.batch_size, num_workers=args.workers,
    )  # fmt: skip
    print(f"{args.checkpoint}: epoch {where['epoch'] or '?'}, step {where['steps'] or '?'}")
    print(f"{args.source} {args.split}: {r.images} images, {r.objects} objects", end=", ")
    print(f"{r.detections} detections kept")
    print(f"mAP50-95 {r.map50_95:.4f} | mAP50 {r.map50:.4f} | mAP75 {r.map75:.4f}")
    print(f"AP small {r.ap_small:.4f} | medium {r.ap_medium:.4f} | large {r.ap_large:.4f}")
    print(f"average recall at 300 detections {r.recall:.4f}")
    print(
        f"best F1 {r.best_f1:.4f} at confidence {r.best_conf:.3f} "
        f"(precision {r.precision_at_best:.3f}, recall {r.recall_at_best:.3f})"
    )
    print(f"deploy with: pv-infer --conf {r.best_conf:.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(asdict(r), indent=2))
