from typing import Any, cast

import pytest
import torch
from torch import nn

from palletvision_model.layers.blocks import C3, Bottleneck, C3k, C3k2


def conv_params(c_in: int, c_out: int, k: int, groups: int = 1) -> int:
    """Parameters of one Conv unit: conv weights plus BatchNorm scale and shift."""
    return c_in * c_out * k * k // groups + 2 * c_out


def zero_conv_weights(module: nn.Module) -> None:
    """Make every Conv output exactly 0 (BN of 0 is 0 and SiLU(0) is 0), in eval mode."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
    module.eval()


def test_bottleneck_shortcut_adds_input_when_channels_match() -> None:
    x = torch.randn(2, 8, 10, 10)
    block = Bottleneck(8, 8, shortcut=True)
    zero_conv_weights(block)
    assert torch.equal(block(x), x)  # body contributes 0, so only the skip remains


def test_bottleneck_shortcut_off_when_disabled_or_channels_differ() -> None:
    x = torch.randn(2, 8, 10, 10)
    off = Bottleneck(8, 8, shortcut=False)
    zero_conv_weights(off)
    assert off(x).abs().max() == 0
    mismatched = Bottleneck(8, 16, shortcut=True)
    assert not mismatched.add
    assert mismatched(x).shape == (2, 16, 10, 10)


def test_bottleneck_parameter_count() -> None:
    n_params = sum(p.numel() for p in Bottleneck(8, 8, expand=0.5).parameters())
    assert n_params == conv_params(8, 4, 3) + conv_params(4, 8, 3)


@pytest.mark.parametrize("block_cls", [C3, C3k])
def test_c3_keeps_spatial_size_and_sets_channels(block_cls: type[nn.Module]) -> None:
    out = block_cls(16, 32, n=2)(torch.rand(2, 16, 20, 20))
    assert out.shape == (2, 32, 20, 20)


def test_c3k_uses_3x3_kernels_and_c3_uses_1x3() -> None:
    def kernel_sizes(block: C3) -> list[tuple[int, int]]:
        return [(b.cv1.conv.kernel_size[0], b.cv2.conv.kernel_size[0]) for b in cast(Any, block.m)]

    assert kernel_sizes(C3(16, 16, n=2)) == [(1, 3), (1, 3)]
    assert kernel_sizes(C3k(16, 16, n=2)) == [(3, 3), (3, 3)]


@pytest.mark.parametrize("c3k", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_c3k2_preserves_size_and_concat_has_2_plus_n_chunks(c3k: bool, n: int) -> None:
    block = C3k2(16, 32, n=n, c3k=c3k, expand=0.5)
    assert block(torch.rand(2, 16, 20, 20)).shape == (2, 32, 20, 20)
    c = int(32 * 0.5)
    assert block.cv2.conv.in_channels == (2 + n) * c  # first chunk must be in the concat


def test_c3k2_inner_block_type_follows_c3k_flag() -> None:
    assert all(isinstance(m, Bottleneck) for m in C3k2(16, 16, n=2, c3k=False).m)
    assert all(isinstance(m, C3k) for m in C3k2(16, 16, n=2, c3k=True).m)


def test_c3k2_parameter_count_matches_hand_calculation() -> None:
    # c_in=c_out=16, expand 0.5 -> c=8, n=1. The plain bottleneck keeps its default expand
    # of 0.5, so its hidden width is 4, not 8.
    c = 8
    expected = (
        conv_params(16, 2 * c, 1)  # cv1
        + conv_params(c, c // 2, 3)
        + conv_params(c // 2, c, 3)  # the one bottleneck: squeeze to 4 then restore to 8
        + conv_params(3 * c, 16, 1)  # cv2 sees (2 + 1) * c channels
    )
    assert sum(p.numel() for p in C3k2(16, 16, n=1).parameters()) == expected


def test_c3k2_early_stage_expand_025_gives_narrow_inner_width() -> None:
    block = C3k2(32, 64, n=1, expand=0.25)
    assert block.c == 16
    assert block(torch.rand(1, 32, 8, 8)).shape == (1, 64, 8, 8)
