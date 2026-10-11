import pytest
import torch

from palletvision_model.blocks import Conv


@pytest.mark.parametrize(("k", "s", "out_hw"), [(1, 1, 16), (3, 1, 16), (3, 2, 8), (1, 2, 8)])
def test_conv_output_shape(k: int, s: int, out_hw: int) -> None:
    out = Conv(4, 8, k, s)(torch.rand(2, 4, 16, 16))
    assert out.shape == (2, 8, out_hw, out_hw)


def test_conv_parameter_count() -> None:
    c_in, c_out, k = 4, 8, 3
    n_params = sum(p.numel() for p in Conv(c_in, c_out, k).parameters())
    assert n_params == c_in * c_out * k * k + 2 * c_out  # conv weights + BN scale and shift


def test_depthwise_conv_has_one_filter_per_channel() -> None:
    c = 8
    n_params = sum(p.numel() for p in Conv(c, c, 3, groups=c).parameters())
    assert n_params == c * 3 * 3 + 2 * c


def test_act_false_leaves_negative_values() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    assert Conv(4, 4, act=False).eval()(x).min() < -0.5  # SiLU's floor is about -0.28
