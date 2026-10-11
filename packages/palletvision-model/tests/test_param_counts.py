import pytest

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.yolo26 import YOLO26


def count(module: object) -> int:
    return sum(p.numel() for p in module.parameters())  # type: ignore[attr-defined]


def with_classes(size: str, num_classes: int) -> ModelConfig:
    base = SIZES[size]
    return ModelConfig(base.depth_multiple, base.width_multiple, num_classes=num_classes)


@pytest.mark.parametrize(("size", "published"), [("n", 2_572_280), ("s", 10_009_784)])
def test_80_class_totals_match_the_published_models(size: str, published: int) -> None:
    """Any wrong width, repeat, skip or head layout would move these totals."""
    assert count(YOLO26(with_classes(size, 80))) == published


def test_deployed_model_drops_the_one_to_many_branch() -> None:
    model = YOLO26(with_classes("n", 1))
    deployed = count(model) - count(model.head.one2many)
    assert deployed == 2_383_407  # what the exported ONNX graph holds for one pallet class
