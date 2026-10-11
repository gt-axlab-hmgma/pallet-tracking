"""The whole detector: image in, ``[B, 300, 6]`` detections out (the model_io contract)."""

import torch
from torch import nn

from palletvision_model.backbone import Backbone
from palletvision_model.config import ModelConfig
from palletvision_model.decode import decode_boxes, make_anchors, postprocess
from palletvision_model.head import Detect
from palletvision_model.neck import Neck


class YOLO26(nn.Module):
    """Backbone, neck and two-branch head, plus the decoding that turns cells into boxes.

    ``forward`` always answers with final detections from the one-to-one branch, so the
    inference code can call it like any other detector. Training code that needs the raw
    per-cell predictions of both branches calls ``raw`` instead.
    """

    def __init__(self, cfg: ModelConfig, input_size: int = 640) -> None:
        super().__init__()
        self.backbone = Backbone(cfg)
        self.neck = Neck(cfg, self.backbone.out_channels)
        self.head = Detect(cfg, self.neck.out_channels, input_size=input_size)

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feats: tuple[torch.Tensor, ...] = self.neck(*self.backbone(x))
        return feats

    def raw(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Per-cell ``(distances, class logits)``: both branches in training, one2one in eval."""
        return self.head(self._features(x))  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features(x)
        distances, cls_logits = self.head(feats)["one2one"]
        sizes = [(f.shape[2], f.shape[3]) for f in feats]
        points, strides = make_anchors(sizes, device=x.device)
        boxes = decode_boxes(distances, points, strides)
        return postprocess(boxes, cls_logits, (x.shape[2], x.shape[3]))
