import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from palletvision_infer.backends import OnnxRuntimeBackend
from palletvision_infer.detections import Detector
from palletvision_infer.pipeline import DummyDetector


def benchmark_detector(
    detector: Detector, size: int = 640, warmup: int = 10, iters: int = 100
) -> dict[str, float]:
    """End-to-end latency (preprocess + inference + decode) on a random frame."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    for _ in range(warmup):
        detector.detect(image)
    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        detector.detect(image)
        times_ms.append((time.perf_counter() - start) * 1000)
    return {
        "iters": iters,
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "fps": 1000.0 / float(np.mean(times_ms)),
    }


def count_params_flops(model: nn.Module, size: int = 640) -> dict[str, float]:
    """Parameter count and forward FLOPs for one 1x3xSxS input."""
    params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    with FlopCounterMode(display=False) as counter, torch.inference_mode():
        model.eval()(torch.zeros(1, 3, size, size, device=device))
    return {"params": params, "flops": counter.get_total_flops()}


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-bench", description="Benchmark a detector backend.")
    p.add_argument("--weights", type=Path, help="ONNX model; omit to benchmark the dummy detector")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("runs/bench/latest.json"))
    args = p.parse_args(argv)

    detector: Detector = (
        OnnxRuntimeBackend(args.weights, input_size=args.size) if args.weights else DummyDetector()
    )
    result = {
        "detector": type(detector).__name__,
        "input_size": args.size,
        **benchmark_detector(detector, args.size, args.warmup, args.iters),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
