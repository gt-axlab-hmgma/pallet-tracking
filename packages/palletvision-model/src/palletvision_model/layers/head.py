"""The detection head: from the neck's feature maps to one box and one score per grid cell.

Every cell of every map predicts, on its own, (a) how far the object's four box edges are
from the cell centre and (b) how likely it is that an object is there. The head makes no
decision about which cells win. That is the loss's job in training and a top-k pick later.
"""

import math

import torch
from torch import nn

from palletvision_model.config import ModelConfig
from palletvision_model.layers.blocks import Conv, DWConv


def flatten_levels(maps: list[torch.Tensor]) -> torch.Tensor:
    """[B, C, H_i, W_i] for each level -> [B, A, C], cells in row-major order, P3 first."""
    return torch.cat([m.flatten(2) for m in maps], dim=2).transpose(1, 2)


def final_conv(branch: nn.Module) -> nn.Conv2d:
    """The last layer of a box or class predictor: the plain conv that outputs raw numbers."""
    assert isinstance(branch, nn.Sequential)
    last = branch[-1]
    assert isinstance(last, nn.Conv2d)
    return last


def final_bias(branch: nn.Module) -> torch.Tensor:
    """The bias of a predictor's last conv, which the init below sets by hand."""
    bias = final_conv(branch).bias
    assert bias is not None
    return bias


class HeadBranches(nn.Module):
    """One full set of box and class predictors, one pair per feature level."""

    def __init__(self, cfg: ModelConfig, in_channels: tuple[int, ...]) -> None:
        super().__init__()
        assert cfg.reg_max == 1, "only direct box regression is implemented (no DFL)"
        c_box = max(16, in_channels[0] // 4, 4 * cfg.reg_max)
        c_cls = max(in_channels[0], min(cfg.num_classes, 100))
        self.box = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c_box, 3),
                Conv(c_box, c_box, 3),
                nn.Conv2d(c_box, 4 * cfg.reg_max, 1),  # plain conv: raw numbers, with a bias
            )
            for ch in in_channels
        )
        self.cls = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(ch, 3), Conv(ch, c_cls, 1)),
                nn.Sequential(DWConv(c_cls, 3), Conv(c_cls, c_cls, 1)),
                nn.Conv2d(c_cls, cfg.num_classes, 1),  # plain conv: raw logits
            )
            for ch in in_channels
        )

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        box = flatten_levels([b(f) for b, f in zip(self.box, feats, strict=True)])
        cls = flatten_levels([c(f) for c, f in zip(self.cls, feats, strict=True)])
        return box, cls  # [B, A, 4] box distances, [B, A, nc] class logits


class Detect(nn.Module):
    """Two copies of the predictors on the same features (the end-to-end design).

    - ``one2many``: several cells may claim the same object. Used only in training, because
      it gives the network a richer learning signal.
    - ``one2one``: exactly one cell per object. This is what makes inference NMS-free.

    The one2one copy is fed *detached* features in training, so its gradients update only
    its own layers and cannot disturb the shared backbone and neck.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        in_channels: tuple[int, ...],
        input_size: int = 640,
        strides: tuple[int, ...] = (8, 16, 32),
        prior_objects: float = 5.0,
    ) -> None:
        super().__init__()
        self.one2many = HeadBranches(cfg, in_channels)
        self.one2one = HeadBranches(cfg, in_channels)
        for branches in (self.one2many, self.one2one):
            self._init_biases(branches, cfg.num_classes, input_size, strides, prior_objects)

    @staticmethod
    def _init_biases(
        branches: HeadBranches,
        num_classes: int,
        input_size: int,
        strides: tuple[int, ...],
        prior_objects: float,
    ) -> None:
        """Start from sensible guesses so the first steps of training are not wasted.

        Box distances start at 2 strides. Class scores start very low: each level's cells
        together expect about ``prior_objects`` detections (per class, per level), so the
        thousands of empty cells do not flood the early loss.
        """
        for box_out, cls_out, stride in zip(branches.box, branches.cls, strides, strict=True):
            nn.init.constant_(final_bias(box_out), 2.0)
            cells = (input_size / stride) ** 2
            nn.init.constant_(final_bias(cls_out), math.log(prior_objects / num_classes / cells))

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Training returns both copies; eval returns only ``one2one``."""
        feats_list = list(feats)
        if not self.training:
            return {"one2one": self.one2one(feats_list)}
        return {
            "one2many": self.one2many(feats_list),
            "one2one": self.one2one([f.detach() for f in feats_list]),
        }
