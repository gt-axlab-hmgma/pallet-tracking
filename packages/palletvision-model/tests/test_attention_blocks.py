import torch
from torch import nn

from palletvision_model.blocks import C2PSA, SPPF, Attention, PSABlock


def conv_params(c_in: int, c_out: int, k: int, groups: int = 1) -> int:
    return c_in * c_out * k * k // groups + 2 * c_out


def test_sppf_shape_and_parameter_count() -> None:
    block = SPPF(32, 64)
    assert block(torch.rand(2, 32, 20, 20)).shape == (2, 64, 20, 20)
    c = 32 // 2
    expected = conv_params(32, c, 1) + conv_params(4 * c, 64, 1)  # max-pool has no weights
    assert sum(p.numel() for p in block.parameters()) == expected


def test_three_sequential_5x5_pools_equal_one_13x13_pool() -> None:
    x = torch.randn(1, 4, 32, 32)
    pool5 = nn.MaxPool2d(5, stride=1, padding=2)
    pool13 = nn.MaxPool2d(13, stride=1, padding=6)
    assert torch.equal(pool5(pool5(pool5(x))), pool13(x))


def test_attention_shape_and_parameter_count() -> None:
    dim, heads = 64, 2
    block = Attention(dim, heads)
    assert block(torch.rand(2, dim, 8, 8)).shape == (2, dim, 8, 8)
    key_dim = int(dim // heads * 0.5)
    expected = (
        conv_params(dim, dim + 2 * key_dim * heads, 1)  # qkv
        + conv_params(dim, dim, 1)  # proj
        + conv_params(dim, dim, 3, groups=dim)  # positional depthwise conv
    )
    assert sum(p.numel() for p in block.parameters()) == expected


def test_attention_sees_distant_pixels_but_a_3x3_conv_would_not() -> None:
    torch.manual_seed(0)
    block = Attention(32, 2).eval()
    x = torch.randn(1, 32, 8, 8)
    x_changed = x.clone()
    x_changed[..., 0, 0] += 5.0  # change the top-left pixel only
    diff = (block(x) - block(x_changed)).abs()
    assert diff[..., 7, 7].max() > 1e-4  # the opposite corner reacts


def test_psablock_is_residual() -> None:
    block = PSABlock(64, num_heads=2).eval()
    for m in block.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
    x = torch.randn(1, 64, 6, 6)
    assert torch.equal(block(x), x)  # attention and FFN contribute 0, only the skips remain


def test_c2psa_p5_shape_and_head_count_for_the_n_model() -> None:
    block = C2PSA(256, 256, n=1)
    assert block(torch.rand(1, 256, 20, 20)).shape == (1, 256, 20, 20)
    assert block.c == 128
    assert block.m[0].attn.num_heads == 2  # c // 64
    assert block.m[0].attn.key_dim == 32


def test_c2psa_repeats_psablocks_and_survives_tiny_widths() -> None:
    block = C2PSA(32, 32, n=2)
    assert len(block.m) == 2
    assert block.m[0].attn.num_heads == 1  # c = 16, so c // 64 would be 0; clamped to 1
    assert block(torch.rand(1, 32, 4, 4)).shape == (1, 32, 4, 4)


def test_attention_core_matches_pytorch_reference() -> None:
    """Our softmax(q^T k * scale) v, written by hand so it exports at ONNX opset 13, must
    equal torch's built-in scaled_dot_product_attention (which needs opset 14)."""
    torch.manual_seed(0)
    block = Attention(64, 2).eval()
    x = torch.randn(2, 64, 10, 10)
    b, c, h, w = x.shape
    q, k, v = (
        block.qkv(x)
        .view(b, block.num_heads, 2 * block.key_dim + block.head_dim, h * w)
        .split((block.key_dim, block.key_dim, block.head_dim), dim=2)
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
    )
    reference = reference.transpose(-2, -1).reshape(b, c, h, w)
    expected = block.proj(reference + block.pe(v.reshape(b, c, h, w)))
    assert torch.allclose(block(x), expected, atol=1e-6)
