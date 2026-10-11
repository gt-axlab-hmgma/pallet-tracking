"""The neck (PAN): mixes the backbone's three feature maps so each carries fine and broad info.

Deep maps know *what* things are but not exactly *where*; shallow maps know where but not what.
The neck sends information both ways: down to the fine maps, then back up to the coarse ones.
"""

import torch
from torch import nn

from palletvision_model.blocks import C3k2, Conv
from palletvision_model.config import ModelConfig


class Neck(nn.Module):
    """Layers 11 to 22 of the guide (step M6). Inputs are P3, P4, P5; outputs are N3, N4, N5.

    ``in_channels`` is what the backbone reports in ``out_channels``.
    """

    def __init__(self, cfg: ModelConfig, in_channels: tuple[int, int, int]) -> None:
        super().__init__()
        ch, rep = cfg.channels, cfg.repeats
        c3, c4, c5 = in_channels

        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # 11 and 14: no weights
        # top-down: push deep information into the finer maps
        self.td4 = C3k2(c5 + c4, ch(512), rep(2), c3k=True)  # 13 (after concat 12)
        self.td3 = C3k2(ch(512) + c3, ch(256), rep(2), c3k=True)  # 16 (after concat 15): N3
        # bottom-up: push fine information back into the coarser maps
        self.down3 = Conv(ch(256), ch(256), 3, 2)  # 17
        self.bu4 = C3k2(ch(256) + ch(512), ch(512), rep(2), c3k=True)  # 19 (after concat 18): N4
        self.down4 = Conv(ch(512), ch(512), 3, 2)  # 20
        self.bu5 = C3k2(ch(512) + c5, ch(1024), 1, c3k=True, expand=0.5, attn=True)  # 22: N5

        self.out_channels = (ch(256), ch(512), ch(1024))  # what the head will receive

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mid4 = self.td4(torch.cat((self.up(p5), p4), dim=1))  # 11, 12, 13
        n3 = self.td3(torch.cat((self.up(mid4), p3), dim=1))  # 14, 15, 16
        n4 = self.bu4(torch.cat((self.down3(n3), mid4), dim=1))  # 17, 18, 19
        n5 = self.bu5(torch.cat((self.down4(n4), p5), dim=1))  # 20, 21, 22
        return n3, n4, n5
