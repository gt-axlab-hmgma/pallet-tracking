import json
from pathlib import Path

import torch
from torch import nn

from palletvision_infer.bench import benchmark_detector, count_params_flops, main
from palletvision_infer.pipeline import DummyDetector


def test_benchmark_detector_reports_latency_and_fps() -> None:
    stats = benchmark_detector(DummyDetector(), size=64, warmup=2, iters=10)
    assert stats["iters"] == 10
    assert 0 < stats["p50_ms"] <= stats["p99_ms"]
    assert stats["fps"] > 0


def test_count_params_flops_on_small_conv() -> None:
    model = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    stats = count_params_flops(model, size=16)
    assert stats["params"] == 3 * 8 * 3 * 3
    assert stats["flops"] == 2 * 3 * 8 * 3 * 3 * 16 * 16  # multiply-adds counted as 2 flops
    assert torch.is_tensor(torch.zeros(1))  # torch is importable alongside


def test_cli_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    main(["--size", "64", "--iters", "5", "--warmup", "1", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["detector"] == "DummyDetector"
    assert data["fps"] > 0
