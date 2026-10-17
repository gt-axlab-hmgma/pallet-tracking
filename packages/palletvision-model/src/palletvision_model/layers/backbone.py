"""The backbone: turns an image into three feature maps at different scales.

Each stride-2 Conv halves the map's height and width, so deeper layers see more of the image
per cell. We keep three snapshots, called P3, P4 and P5 (strides 8, 16 and 32), and hand them
to the neck.
"""

import torch
from torch import nn

from palletvision_model.config import ModelConfig
from palletvision_model.layers.blocks import C2PSA, SPPF, C3k2, Conv


class Backbone(nn.Module):
    """Eleven layers, numbered 0 to 10 as in the implementation guide (step M5).

    Widths are written as the nominal numbers from the table (64, 128, ..., 1024) and scaled
    to the real model size by ``cfg.channels``. Repeat counts go through ``cfg.repeats``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        ch, rep = cfg.channels, cfg.repeats

        self.to_p3 = nn.Sequential(
            Conv(3, ch(64), 3, 2),  # 0: stride 2, the stem
            Conv(ch(64), ch(128), 3, 2),  # 1: stride 4
            C3k2(ch(128), ch(256), rep(2), c3k=False, expand=0.25),  # 2
            Conv(ch(256), ch(256), 3, 2),  # 3: stride 8
            C3k2(ch(256), ch(512), rep(2), c3k=False, expand=0.25),  # 4: kept as P3
        )
        self.to_p4 = nn.Sequential(
            Conv(ch(512), ch(512), 3, 2),  # 5: stride 16
            C3k2(ch(512), ch(512), rep(2), c3k=True),  # 6: kept as P4
        )
        self.to_p5 = nn.Sequential(
            Conv(ch(512), ch(1024), 3, 2),  # 7: stride 32
            C3k2(ch(1024), ch(1024), rep(2), c3k=True),  # 8
            SPPF(ch(1024), ch(1024), 5, shortcut=True),  # 9
            C2PSA(ch(1024), ch(1024), rep(2)),  # 10: kept as P5
        )
        self.out_channels = (ch(512), ch(512), ch(1024))  # what the neck will receive

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.to_p3(x)
        p4 = self.to_p4(p3)
        p5 = self.to_p5(p4)
        return p3, p4, p5
