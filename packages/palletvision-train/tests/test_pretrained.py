"""``train.init_from``: start a run from another model's weights (pretraining, then fine-tuning)."""

from pathlib import Path

import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.yolo26 import YOLO26
from palletvision_train.pretrained import load_pretrained


def small(classes: int) -> YOLO26:
    cfg = SIZES["n"]
    return YOLO26(
        ModelConfig(cfg.depth_multiple, cfg.width_multiple, num_classes=classes, reg_max=1),
        input_size=64,
    )


def test_matching_tensors_are_copied_and_the_class_head_stays_fresh(tmp_path: Path) -> None:
    pretrained, target = small(365), small(1)
    torch.save({"model": pretrained.state_dict()}, tmp_path / "pre.pt")
    before = {k: v.clone() for k, v in target.state_dict().items()}
    copied, skipped = load_pretrained(target, tmp_path / "pre.pt")
    after = target.state_dict()
    assert len(copied) + len(skipped) == len(after) and skipped  # some tensors differ in shape
    assert set(copied).isdisjoint(skipped)
    for name, tensor in pretrained.state_dict().items():
        if name in skipped:
            assert torch.equal(after[name], before[name])  # left as the new model initialised it
        elif after[name].dtype.is_floating_point:
            assert torch.equal(after[name], tensor)
    assert any("cls" in name for name in skipped)
    assert not any(n.startswith("backbone") for n in skipped)  # the feature extractor transfers


def test_a_plain_state_dict_loads_too(tmp_path: Path) -> None:
    model = small(1)
    torch.save(model.state_dict(), tmp_path / "w.pt")
    other = small(1)
    copied, skipped = load_pretrained(other, tmp_path / "w.pt")
    assert skipped == [] and copied == sorted(model.state_dict())
