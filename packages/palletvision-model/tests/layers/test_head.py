import math
from typing import Any, cast

import pytest
import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.layers.head import Detect, HeadBranches, final_bias, flatten_levels


def n_model(num_classes: int = 1) -> ModelConfig:
    base = SIZES["n"]
    return ModelConfig(base.depth_multiple, base.width_multiple, num_classes=num_classes)


def feats(batch: int = 2, size: int = 640, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.rand(batch, c, size // s, size // s, requires_grad=requires_grad)
        for c, s in ((64, 8), (128, 16), (256, 32))
    )


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_flatten_levels_is_row_major_with_the_finest_level_first() -> None:
    p3 = torch.arange(2 * 3).reshape(1, 1, 2, 3).float()  # cells 0..5
    p4 = torch.arange(100, 101).reshape(1, 1, 1, 1).float()  # one cell
    flat = flatten_levels([p3, p4])
    assert flat.shape == (1, 7, 1)
    assert flat[0, :, 0].tolist() == [0, 1, 2, 3, 4, 5, 100]


def test_widths_for_the_n_model() -> None:
    one = HeadBranches(n_model(1), (64, 128, 256))
    assert cast(Any, one.box[0])[0].conv.out_channels == 16
    assert cast(Any, one.cls[0])[0][1].conv.out_channels == 64
    many = HeadBranches(n_model(80), (64, 128, 256))
    assert cast(Any, many.cls[0])[0][1].conv.out_channels == 80  # max(64, min(80, 100))


def test_parameter_counts_match_hand_calculation_for_n_one_class() -> None:
    branches = HeadBranches(n_model(1), (64, 128, 256))
    assert count(branches.box) == 71_820  # the guide says about 72k
    assert count(branches.cls) == 48_963  # the guide says about 50k


def test_eval_returns_one_to_one_only_with_8400_cells() -> None:
    head = Detect(n_model(1), (64, 128, 256)).eval()
    out = head(feats())
    assert list(out) == ["one2one"]
    box, cls = out["one2one"]
    assert box.shape == (2, 8400, 4)
    assert cls.shape == (2, 8400, 1)


def test_training_returns_both_copies() -> None:
    head = Detect(n_model(3), (64, 128, 256)).train()
    out = head(feats(batch=1))
    assert set(out) == {"one2many", "one2one"}
    assert out["one2many"][1].shape == (1, 8400, 3)


def test_the_two_copies_have_separate_weights() -> None:
    head = Detect(n_model(), (64, 128, 256))
    ids_many = {id(p) for p in head.one2many.parameters()}
    assert ids_many.isdisjoint(id(p) for p in head.one2one.parameters())


def test_one_to_one_loss_does_not_reach_the_features_but_one_to_many_does() -> None:
    head = Detect(n_model(), (64, 128, 256)).train()
    f = feats(batch=1, size=64, requires_grad=True)
    box, cls = head(f)["one2one"]
    (box.sum() + cls.sum()).backward()
    assert all(t.grad is None for t in f)  # detached: the shared backbone is undisturbed
    assert (
        cast(Any, head.one2one.box[0])[0].conv.weight.grad is not None
    )  # its own layers still learn
    box, cls = head(f)["one2many"]
    (box.sum() + cls.sum()).backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in f)


def test_bias_initialisation() -> None:
    head = Detect(n_model(2), (64, 128, 256))
    for branches in (head.one2many, head.one2one):
        for box_out, cls_out, stride in zip(branches.box, branches.cls, (8, 16, 32), strict=True):
            assert torch.all(final_bias(box_out) == 2.0)
            expected = math.log(5 / 2 / (640 / stride) ** 2)
            assert torch.allclose(final_bias(cls_out), torch.tensor(expected))


def test_initial_scores_are_tiny() -> None:
    head = Detect(n_model(1), (64, 128, 256)).eval()
    _, cls = head(feats(batch=1))["one2one"]
    assert cls.sigmoid().mean() < 0.02


@pytest.mark.parametrize("size", [320, 512])
def test_cell_count_follows_input_size(size: int) -> None:
    head = Detect(n_model(), (64, 128, 256), input_size=size).eval()
    box, _ = head(feats(batch=1, size=size))["one2one"]
    assert box.shape[1] == (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
