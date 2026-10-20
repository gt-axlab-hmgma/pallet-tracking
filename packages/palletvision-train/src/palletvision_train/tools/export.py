import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from palletvision_train.checkpoints.model import load_model

OPSET = 13  # contracts/model_io.yaml: TensorRT 8.2 parser limit on the Jetson Nano


def export_onnx(model: nn.Module, path: Path | str, size: int = 640) -> Path:
    """Export to ONNX using the tensor names and opset from contracts/model_io.yaml."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    torch.onnx.export(
        model.eval(), (torch.zeros(1, 3, size, size, device=device),), str(path),
        opset_version=OPSET, dynamo=False, input_names=["images"], output_names=["detections"],
    )  # fmt: skip
    return path


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-export", description="Export a checkpoint to ONNX.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--size", type=int, help="input size (default: the one the checkpoint was trained at)"
    )
    p.add_argument("--out", type=Path, default=Path("runs/export/model.onnx"))
    args = p.parse_args(argv)
    model, where, trained_size = load_model(args.checkpoint)
    size = args.size or trained_size
    print(f"model: epoch {where['epoch']}, {where['steps']} steps, input {size}x{size}")
    print(export_onnx(model, args.out, size))
