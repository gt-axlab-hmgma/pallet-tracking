"""The real model must export to ONNX opset 13 and run the same under ONNX Runtime.

Parity is checked in pieces, because an untrained network makes a whole-output check fragile:
its scores are nearly all tied, and Torch and ONNX Runtime break ties differently. So:
- the per-cell boxes and logits (no ordering involved) are compared with a tolerance that
  allows for float32 rounding through a deep random network,
- the top-300 selection is compared exactly on inputs whose scores are all distinct,
- the full export is checked for structure, allowed ops and that it runs.
"""

import io
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from palletvision_infer.backends import OnnxRuntimeBackend
from palletvision_model.config import SIZES
from palletvision_model.decode import decode_boxes, make_anchors, postprocess
from palletvision_model.yolo26 import YOLO26
from palletvision_train.export import export_onnx

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "model_io.yaml"
SIZE = 320  # same ops as 640, much faster
# ops the Jetson Nano's TensorRT 8.2 parser should not be given (guide, step M10)
FORBIDDEN_OPS = {"Einsum", "If", "Loop", "Scan", "Mod"}


def smooth_images(n: int, size: int = SIZE) -> torch.Tensor:
    """Spatially smooth random images; pure noise would give a degenerate network output."""
    return F.interpolate(torch.rand(n, 3, 8, 8), size=(size, size), mode="bilinear")


def calibrated_model() -> YOLO26:
    """One pass in train mode with BatchNorm momentum 1 sets its running statistics.

    Without this, an untrained network in eval mode shrinks the signal to about 1e-9 by the
    head, every cell gets the same score, and any comparison becomes a comparison of ties.
    """
    torch.manual_seed(0)
    model = YOLO26(SIZES["n"])
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1.0
    model.train()(smooth_images(2))
    return model.eval()


def export_to_bytes(module: nn.Module, args: tuple[torch.Tensor, ...], names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module.eval(), args, buffer, opset_version=13, dynamo=False,
            input_names=names[0].split(","), output_names=names[1].split(","),
        )  # fmt: skip
    return buffer.getvalue()


def test_full_model_export_is_structurally_valid(tmp_path: Path) -> None:
    io_spec = yaml.safe_load(CONTRACT.read_text())
    path = export_onnx(YOLO26(SIZES["n"]), tmp_path / "m.onnx", size=SIZE)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert graph.opset_import[0].version == io_spec["onnx_opset"] == 13
    assert [i.name for i in graph.graph.input] == ["images"]
    assert [o.name for o in graph.graph.output] == ["detections"]

    ops = {node.op_type for node in graph.graph.node}
    assert not ops & FORBIDDEN_OPS
    assert "TopK" in ops  # the top-300 selection is in the graph, with a constant K
    # the training-only branch must not travel with the deployed model
    assert not any("one2many" in init.name for init in graph.graph.initializer)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {"images": np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)})
    assert out.shape == tuple(io_spec["outputs"][0]["shape"])
    assert out.dtype == np.float32


def test_per_cell_boxes_and_logits_match_onnx_runtime() -> None:
    model = calibrated_model()

    class Cells(nn.Module):
        """Everything except the top-300 pick: backbone, neck, head, anchors and decoding."""

        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            feats = self.model.neck(*self.model.backbone(x))
            distances, logits = self.model.head(feats)["one2one"]
            points, strides = make_anchors([(f.shape[2], f.shape[3]) for f in feats])
            return decode_boxes(distances, points, strides), logits

    cells = Cells().eval()
    session = ort.InferenceSession(
        export_to_bytes(cells, (torch.zeros(1, 3, SIZE, SIZE),), ["images", "boxes,logits"]),
        providers=["CPUExecutionProvider"],
    )
    for _ in range(3):
        x = smooth_images(1)
        boxes, logits = (t.detach().numpy() for t in cells(x))
        onnx_boxes, onnx_logits = session.run(None, {"images": x.numpy()})
        # A wrong anchor, stride or skip gives errors of the size of the values themselves.
        # Float32 rounding through ~100 random layers gives about 1e-3 of the value range
        # (Torch float32 differs from Torch float64 by about the same amount).
        for ours, theirs in ((boxes, onnx_boxes), (logits, onnx_logits)):
            assert np.abs(ours - theirs).max() <= 2e-2 * np.abs(ours).max()


def test_top_300_selection_matches_onnx_runtime_exactly() -> None:
    class Pick(nn.Module):
        def forward(self, boxes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            return postprocess(boxes, logits, (100, 200))

    torch.manual_seed(1)
    boxes = torch.rand(1, 1000, 4) * 260 - 30  # some stick out of the 200 x 100 image
    logits = torch.randn(1, 1000, 3)  # continuous, so every score is distinct
    session = ort.InferenceSession(
        export_to_bytes(Pick(), (boxes, logits), ["boxes,logits", "detections"]),
        providers=["CPUExecutionProvider"],
    )
    (onnx_out,) = session.run(None, {"boxes": boxes.numpy(), "logits": logits.numpy()})
    expected = postprocess(boxes, logits, (100, 200)).numpy()
    np.testing.assert_allclose(onnx_out, expected, rtol=0, atol=1e-5)
    assert np.array_equal(onnx_out[..., 5], expected[..., 5])  # classes identical


def test_exported_model_runs_in_the_onnx_backend(tmp_path: Path) -> None:
    path = export_onnx(YOLO26(SIZES["n"]), tmp_path / "m.onnx", size=SIZE)
    backend = OnnxRuntimeBackend(path, input_size=SIZE, conf=0.001)
    image = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    found = backend.detect(image)
    assert len(found) == 300  # an untrained model still fills all rows with real candidates
    assert found.boxes.shape == (300, 4)
    assert (found.boxes[:, [0, 2]] >= 0).all() and (found.boxes[:, [0, 2]] <= 320).all()
    assert (found.boxes[:, [1, 3]] >= 0).all() and (found.boxes[:, [1, 3]] <= 240).all()
