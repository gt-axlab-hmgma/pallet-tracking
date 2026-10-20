from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml

from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26
from palletvision_train.tools.export import export_onnx, main

CONTRACT = Path(__file__).resolve().parents[4] / "contracts" / "model_io.yaml"
SIZE = 64  # the same ops as 640, much faster


def run_onnx(path: Path, size: int) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {"images": torch.rand(1, 3, size, size).numpy()})
    return np.asarray(out)


def write_checkpoint(path: Path, size: int = SIZE) -> None:
    """The format the trainer writes: weights, plus the config the model was built from."""
    torch.manual_seed(0)
    cfg = SIZES["n"]
    model = YOLO26(cfg, input_size=size)
    config = {"model": {**asdict(cfg), "name": "yolo26n", "input_size": size}}
    torch.save({"model": model.state_dict(), "config": config, "epoch": 3, "steps": 30}, path)


def test_export_follows_the_contract(tmp_path: Path) -> None:
    model = YOLO26(SIZES["n"], input_size=SIZE).eval()
    path = export_onnx(model, tmp_path / "m.onnx", size=SIZE)

    graph = onnx.load(str(path))
    io = yaml.safe_load(CONTRACT.read_text())
    assert graph.opset_import[0].version == io["onnx_opset"]
    assert graph.graph.input[0].name == io["input"]["name"]
    assert graph.graph.output[0].name == io["outputs"][0]["name"]
    out = run_onnx(path, SIZE)
    assert out.shape == (1, io["outputs"][0]["shape"][1], 6)  # 300 rows of x1 y1 x2 y2 score class


def test_cli_exports_a_checkpoint_at_the_size_it_was_trained(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "last.pt"
    write_checkpoint(checkpoint)
    out = tmp_path / "model.onnx"
    main(["--checkpoint", str(checkpoint), "--out", str(out)])  # no --size: taken from the file
    assert "input 64x64" in capsys.readouterr().out
    graph = onnx.load(str(out))
    assert [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 64, 64]
    assert run_onnx(out, 64).shape == (1, 300, 6)


def test_cli_can_override_the_size(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    write_checkpoint(checkpoint)
    out = tmp_path / "model.onnx"
    main(["--checkpoint", str(checkpoint), "--size", "96", "--out", str(out)])
    assert run_onnx(out, 96).shape == (1, 300, 6)
