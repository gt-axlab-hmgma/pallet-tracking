from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from palletvision_infer.detections import Detections, decode
from palletvision_infer.preprocess import letterbox


class TorchBackend:
    """Runs an nn.Module directly. For debugging and benchmarking on the laptop."""

    def __init__(
        self, model: nn.Module, device: str | torch.device = "cpu", input_size: int = 640,
        conf: float = 0.25,
    ) -> None:  # fmt: skip
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.input_size = input_size
        self.conf = conf

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> Detections:
        blob, meta = letterbox(image, self.input_size)
        raw = self.model(torch.from_numpy(blob).to(self.device)).float().cpu().numpy()
        return decode(raw, meta, image.shape[:2], self.conf)


class OnnxRuntimeBackend:
    """Default local runtime. Uses CoreML on Apple silicon when available, else CPU."""

    def __init__(
        self, path: str | Path, input_size: int = 640, conf: float = 0.25,
        providers: list[str] | None = None,
    ) -> None:  # fmt: skip
        available = ort.get_available_providers()
        wanted = providers or ["CoreMLExecutionProvider", "CUDAExecutionProvider"]
        chosen = [p for p in wanted if p in available] + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf = conf

    def detect(self, image: np.ndarray) -> Detections:
        blob, meta = letterbox(image, self.input_size)
        raw = self.session.run(None, {self.input_name: blob})[0]
        return decode(raw, meta, image.shape[:2], self.conf)
