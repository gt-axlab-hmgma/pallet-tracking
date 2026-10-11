"""The real model must satisfy contracts/model_io.yaml and plug into the existing backends."""

from pathlib import Path

import numpy as np
import torch
import yaml

from palletvision_infer.backends import TorchBackend
from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "model_io.yaml"


def test_output_matches_the_declared_contract_shape_and_dtype() -> None:
    io = yaml.safe_load(CONTRACT.read_text())
    out_spec = io["outputs"][0]
    x = torch.rand(*io["input"]["shape"])
    out = YOLO26(SIZES["n"]).eval()(x)
    assert list(out.shape) == out_spec["shape"]
    assert str(out.dtype) == "torch." + out_spec["dtype"]


def test_torch_backend_turns_the_model_output_into_detections() -> None:
    backend = TorchBackend(YOLO26(SIZES["n"]), conf=0.0)  # conf 0 keeps every row
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    found = backend.detect(image)
    assert len(found) > 0
    assert found.boxes.shape == (len(found), 4)
    assert found.scores.shape == found.classes.shape == (len(found),)
    assert (found.boxes[:, [0, 2]] >= 0).all() and (found.boxes[:, [0, 2]] <= 640).all()
    assert (found.boxes[:, [1, 3]] >= 0).all() and (found.boxes[:, [1, 3]] <= 480).all()
