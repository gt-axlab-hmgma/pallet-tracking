import pytest
import torch

from palletvision_model.backbone import Backbone
from palletvision_model.config import SIZES

# (output channels, stride) of layers 0..10 for the n model, from the guide's M5 table
N_MODEL_LAYERS = [
    (16, 2),
    (32, 4),
    (64, 4),
    (64, 8),
    (128, 8),
    (128, 16),
    (128, 16),
    (256, 32),
    (256, 32),
    (256, 32),
    (256, 32),
]


def all_layers(backbone: Backbone) -> list[torch.nn.Module]:
    return [*backbone.to_p3, *backbone.to_p4, *backbone.to_p5]


def test_every_layer_matches_the_guide_table_for_n() -> None:
    backbone = Backbone(SIZES["n"]).eval()
    x = torch.rand(1, 3, 128, 128)
    for layer, (channels, stride) in zip(all_layers(backbone), N_MODEL_LAYERS, strict=True):
        x = layer(x)
        assert x.shape == (1, channels, 128 // stride, 128 // stride)


def test_n_model_outputs_at_640() -> None:
    p3, p4, p5 = Backbone(SIZES["n"]).eval()(torch.rand(2, 3, 640, 640))
    assert p3.shape == (2, 128, 80, 80)
    assert p4.shape == (2, 128, 40, 40)
    assert p5.shape == (2, 256, 20, 20)


def test_s_model_is_twice_as_wide() -> None:
    backbone = Backbone(SIZES["s"]).eval()
    assert backbone.out_channels == (256, 256, 512)
    p3, p4, p5 = backbone(torch.rand(1, 3, 320, 320))
    assert [t.shape[1] for t in (p3, p4, p5)] == [256, 256, 512]


@pytest.mark.parametrize("size", [320, 512, 640])
def test_works_for_any_size_divisible_by_32(size: int) -> None:
    p3, p4, p5 = Backbone(SIZES["n"]).eval()(torch.rand(1, 3, size, size))
    assert p3.shape[-1] == size // 8
    assert p4.shape[-1] == size // 16
    assert p5.shape[-1] == size // 32


def test_out_channels_attribute_matches_real_outputs() -> None:
    backbone = Backbone(SIZES["n"]).eval()
    outs = backbone(torch.rand(1, 3, 64, 64))
    assert tuple(t.shape[1] for t in outs) == backbone.out_channels


def test_gradients_reach_the_first_layer() -> None:
    backbone = Backbone(SIZES["n"])
    x = torch.rand(1, 3, 64, 64)
    sum(t.sum() for t in backbone(x)).backward()
    assert backbone.to_p3[0].conv.weight.grad.abs().sum() > 0


def test_sppf_in_the_backbone_has_its_shortcut_on() -> None:
    assert Backbone(SIZES["n"]).to_p5[2].add
