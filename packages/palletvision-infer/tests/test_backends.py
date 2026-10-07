from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from palletvision_infer.backends import OnnxRuntimeBackend, TorchBackend
from palletvision_infer.detections import Detections, Detector

ROWS = torch.tensor(
    [[0.0, 16.0, 64.0, 48.0, 0.9, 0.0], [0.0, 0.0, 8.0, 8.0, 0.1, 0.0]] + [[0.0] * 6] * 298
)


class Dummy(nn.Module):
    """Stands in for the real model: fixed [1, 300, 6] output that still depends on the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ROWS[None] + 0.0 * x.mean()


IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)


def check(det: Detections) -> None:
    assert len(det) == 1  # low-confidence row dropped
    assert det.boxes[0].tolist() == pytest.approx([0.0, 0.0, 200.0, 100.0])
    assert det.scores[0] == pytest.approx(0.9)
    assert det.classes[0] == 0


def test_empty_detections() -> None:
    assert len(Detections.empty()) == 0


def test_torch_backend_decodes_and_filters() -> None:
    backend: Detector = TorchBackend(Dummy(), device="cpu", input_size=64, conf=0.25)
    check(backend.detect(IMAGE))


def test_onnx_backend_matches_torch(tmp_path: Path) -> None:
    path = tmp_path / "dummy.onnx"
    torch.onnx.export(
        Dummy(), (torch.zeros(1, 3, 64, 64),), str(path), opset_version=13, dynamo=False,
        input_names=["images"], output_names=["detections"],
    )  # fmt: skip
    backend: Detector = OnnxRuntimeBackend(path, input_size=64, conf=0.25)
    check(backend.detect(IMAGE))
