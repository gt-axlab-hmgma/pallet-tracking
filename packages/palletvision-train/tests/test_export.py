from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

from palletvision_model.tiny import TinyDetector
from palletvision_train.export import export_onnx, main
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import train

CONTRACT = Path(__file__).resolve().parents[3] / "contracts" / "model_io.yaml"


def test_export_matches_torch_and_contract(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TinyDetector(input_size=64).eval()
    path = export_onnx(model, tmp_path / "m.onnx", size=64)

    graph = onnx.load(str(path))
    io = yaml.safe_load(CONTRACT.read_text())
    assert graph.opset_import[0].version == io["onnx_opset"]
    assert graph.graph.input[0].name == io["input"]["name"]
    assert graph.graph.output[0].name == io["outputs"][0]["name"]

    x = torch.rand(1, 3, 64, 64)
    expected = model(x).detach().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {"images": x.numpy()})
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_cli_exports_checkpoint(tmp_path: Path) -> None:
    result = train(
        TinyDetector(input_size=64), SyntheticBoxes(n=4, size=64), device="cpu", epochs=1,
        batch_size=2, lr=1e-3, amp=False, max_steps=None, output_dir=tmp_path,
    )  # fmt: skip
    out = tmp_path / "model.onnx"
    main(["--checkpoint", str(result.checkpoint), "--size", "64", "--out", str(out)])
    assert out.exists()
