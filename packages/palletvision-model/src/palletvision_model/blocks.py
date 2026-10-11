"""Basic building blocks shared by the backbone, neck and head."""

import torch
from torch import nn


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU, the unit every other block is built from.

    The conv has no bias because the BatchNorm right after it subtracts the mean anyway.
    Padding ``k // 2`` keeps the spatial size at stride 1 and halves it at stride 2 (k=3).
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        s: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
