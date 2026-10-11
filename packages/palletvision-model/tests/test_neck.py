import pytest
import torch

from palletvision_model.backbone import Backbone
from palletvision_model.blocks import Bottleneck, C3k, C3k2, PSABlock
from palletvision_model.config import SIZES
from palletvision_model.neck import Neck


def build(size: str = "n") -> tuple[Backbone, Neck]:
    backbone = Backbone(SIZES[size]).eval()
    return backbone, Neck(SIZES[size], backbone.out_channels).eval()


def test_n_model_outputs_at_640_are_the_head_inputs() -> None:
    backbone, neck = build("n")
    n3, n4, n5 = neck(*backbone(torch.rand(2, 3, 640, 640)))
    assert n3.shape == (2, 64, 80, 80)
    assert n4.shape == (2, 128, 40, 40)
    assert n5.shape == (2, 256, 20, 20)
    assert neck.out_channels == (64, 128, 256)


def test_s_model_outputs() -> None:
    backbone, neck = build("s")
    n3, n4, n5 = neck(*backbone(torch.rand(1, 3, 320, 320)))
    assert [t.shape[1] for t in (n3, n4, n5)] == [128, 256, 512]


@pytest.mark.parametrize("size", [320, 512])
def test_other_input_sizes_keep_the_strides(size: int) -> None:
    backbone, neck = build()
    n3, n4, n5 = neck(*backbone(torch.rand(1, 3, size, size)))
    assert (n3.shape[-1], n4.shape[-1], n5.shape[-1]) == (size // 8, size // 16, size // 32)


def test_concat_widths_are_what_the_guide_lists() -> None:
    # layers 12, 15, 18, 21 concatenate to 384, 256, 192, 384 channels for the n model
    _, neck = build()
    assert [neck.td4.cv1.conv.in_channels, neck.td3.cv1.conv.in_channels] == [384, 256]
    assert [neck.bu4.cv1.conv.in_channels, neck.bu5.cv1.conv.in_channels] == [192, 384]


def test_only_the_last_block_uses_attention_and_the_rest_use_c3k() -> None:
    _, neck = build()
    for block in (neck.td4, neck.td3, neck.bu4):
        assert all(isinstance(m, C3k) for m in block.m)
    assert all(isinstance(m[0], Bottleneck) and isinstance(m[1], PSABlock) for m in neck.bu5.m)


def test_attn_takes_priority_over_c3k_and_picks_heads_from_width() -> None:
    block = C3k2(32, 128, n=1, c3k=True, attn=True)  # c = 64 -> 1 head
    assert isinstance(block.m[0][1], PSABlock)
    assert block.m[0][1].attn.num_heads == 1
    assert block(torch.rand(1, 32, 8, 8)).shape == (1, 128, 8, 8)


def test_gradients_flow_from_every_neck_output_back_to_the_backbone() -> None:
    backbone, neck = Backbone(SIZES["n"]), None
    neck = Neck(SIZES["n"], backbone.out_channels)
    outs = neck(*backbone(torch.rand(1, 3, 64, 64)))
    sum(t.sum() for t in outs).backward()
    assert backbone.to_p3[0].conv.weight.grad.abs().sum() > 0
