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


class Bottleneck(nn.Module):
    """Two Convs in a row, with the input added back on top (a residual / skip connection).

    The first Conv squeezes to ``c_out * expand`` channels, the second restores ``c_out``.
    The skip needs matching channel counts, so it is silently off when ``c_in != c_out``.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        shortcut: bool = True,
        groups: int = 1,
        kernels: tuple[int, int] = (3, 3),
        expand: float = 0.5,
    ) -> None:
        super().__init__()
        c_hidden = int(c_out * expand)
        self.cv1 = Conv(c_in, c_hidden, kernels[0], 1)
        self.cv2 = Conv(c_hidden, c_out, kernels[1], 1, groups=groups)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP block with three Convs: one path through ``n`` bottlenecks, one path that skips them.

    Both paths see the same input at ``c_out * expand`` channels each. They are concatenated
    and mixed by a final 1x1 Conv. ``kernels`` is passed to every inner bottleneck.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expand: float = 0.5,
        kernels: tuple[int, int] = (1, 3),
    ) -> None:
        super().__init__()
        c_hidden = int(c_out * expand)
        self.cv1 = Conv(c_in, c_hidden, 1)  # path through the bottlenecks
        self.cv2 = Conv(c_in, c_hidden, 1)  # path that skips them
        self.cv3 = Conv(2 * c_hidden, c_out, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_hidden, c_hidden, shortcut, groups, kernels, expand=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k(C3):
    """A C3 whose inner bottlenecks use k x k kernels (3x3 by default) instead of 1x3."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expand: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__(c_in, c_out, n, shortcut, groups, expand, kernels=(k, k))


class C3k2(nn.Module):
    """Split the channels in two, refine one half in a chain of blocks, keep every stage.

    1. A 1x1 Conv expands to ``2c`` channels (``c = c_out * expand``) and ``chunk`` cuts them
       into halves ``y0`` and ``y1``.
    2. ``y1`` goes through ``n`` blocks one after another. Each block's output is kept.
    3. Everything is concatenated: ``[y0, y1, m1(y1), m2(m1(y1)), ...]`` is ``(2 + n) * c``
       channels, then a final 1x1 Conv mixes them to ``c_out``.

    ``c3k`` picks the inner block: ``C3k`` (bigger, two repeats each) or a plain ``Bottleneck``.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n: int = 1,
        c3k: bool = False,
        expand: float = 0.5,
        groups: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.c = int(c_out * expand)
        self.cv1 = Conv(c_in, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c_out, 1)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, groups)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, groups)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast: look at the same features through growing windows.

    Three 5x5 max-pools applied one after another see windows of 5, 9 and 13 pixels (the
    second pool pools the first pool's result, and so on). The input and all three pooled
    maps are concatenated, so the final Conv sees fine and coarse context together.
    """

    def __init__(self, c_in: int, c_out: int, k: int = 5) -> None:
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = Conv(c_in, c_hidden, 1)
        self.cv2 = Conv(4 * c_hidden, c_out, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat((x, p1, p2, p3), dim=1))


class Attention(nn.Module):
    """Multi-head self-attention over the pixels of a feature map.

    Every pixel is turned into a query (q), a key (k) and a value (v). A pixel's output is a
    mix of every pixel's value, weighted by how well its query matches their keys. That is
    what lets far-apart pixels talk to each other, which a small conv cannot do.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        qkv_dim = dim + 2 * self.key_dim * num_heads  # q and k are narrower than v
        self.qkv = Conv(dim, qkv_dim, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        # depthwise 3x3 on v: gives each pixel a sense of where its neighbours are
        self.pe = Conv(dim, dim, 3, 1, groups=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).view(b, self.num_heads, 2 * self.key_dim + self.head_dim, n)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.head_dim), dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)  # [b, heads, n, n]
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        return self.proj(out + self.pe(v.reshape(b, c, h, w)))


class PSABlock(nn.Module):
    """Attention, then a small per-pixel MLP, each added back onto its input (residual)."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = Attention(c, num_heads, attn_ratio)
        self.ffn = nn.Sequential(Conv(c, 2 * c, 1), Conv(2 * c, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """C2f-style split: one half passes untouched, the other goes through ``n`` PSABlocks.

    Attention cost grows with the square of the pixel count, so this is only used on the
    smallest feature map (P5, 20x20 at 640 input).
    """

    def __init__(self, c_in: int, c_out: int, n: int = 1, expand: float = 0.5) -> None:
        super().__init__()
        assert c_in == c_out, "C2PSA keeps the channel count"
        self.c = int(c_in * expand)
        self.cv1 = Conv(c_in, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c_in, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), dim=1))
