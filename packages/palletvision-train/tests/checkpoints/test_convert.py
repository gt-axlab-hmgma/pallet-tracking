"""``pv-convert``: read an Ultralytics checkpoint without running its code, map it by position."""

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.yolo26 import YOLO26
from palletvision_train.checkpoints.convert import convert, read_checkpoint, tensors_of
from palletvision_train.checkpoints.pretrained import load_pretrained


def small(classes: int = 3, seed: int = 0) -> YOLO26:
    torch.manual_seed(seed)
    cfg = SIZES["n"]
    return YOLO26(
        ModelConfig(cfg.depth_multiple, cfg.width_multiple, num_classes=classes, reg_max=1),
        input_size=64,
    )


class Named(torch.nn.Module):
    """A module whose pickle carries what an Ultralytics model carries: nc and names."""

    def __init__(self, inner: torch.nn.Module, nc: int) -> None:
        super().__init__()
        self.inner, self.nc, self.names = inner, nc, {i: f"class{i}" for i in range(nc)}


def write(path: Path, model: torch.nn.Module, classes: int = 3, **extra: Any) -> Path:
    torch.save(
        {"model": Named(model, classes), "license": "AGPL-3.0", "version": "8.3.222",
         "train_args": {"model": "yolo26n-objv1-150.pt"},
         "train_metrics": {"fitness": 0.5}, **extra},
        path,
    )  # fmt: skip
    return path


def test_the_tensors_come_out_in_state_dict_order_without_the_classes(tmp_path: Path) -> None:
    model = small()
    found = tensors_of(read_checkpoint(write(tmp_path / "u.pt", model))["model"])
    assert len(found) == len(model.state_dict())
    assert all(
        torch.equal(a, b) for a, b in zip(found.values(), model.state_dict().values(), strict=True)
    )


def test_converting_maps_by_position_and_keeps_the_weights(tmp_path: Path) -> None:
    source = small(seed=1)
    converted = convert(write(tmp_path / "u.pt", source))
    for key, tensor in source.state_dict().items():
        assert torch.equal(converted["model"][key], tensor), key
    target = small(seed=2)  # a fresh model of the same shape takes the weights
    torch.save(converted, tmp_path / "c.pt")
    copied, skipped = load_pretrained(target, tmp_path / "c.pt")
    assert skipped == [] and len(copied) == len(source.state_dict())
    assert all(torch.equal(target.state_dict()[k], v) for k, v in source.state_dict().items())


def test_half_precision_weights_become_float32(tmp_path: Path) -> None:
    half = copy.deepcopy(small()).half()
    converted = convert(write(tmp_path / "u.pt", half))
    floats = [t for t in converted["model"].values() if t.is_floating_point()]
    assert floats and all(t.dtype == torch.float32 for t in floats)


def test_the_origin_licence_and_lineage_travel_with_the_weights(tmp_path: Path) -> None:
    converted = convert(write(tmp_path / "u.pt", small()))
    assert converted["source"]["license"] == "AGPL-3.0"
    assert converted["source"]["trained_from"] == "yolo26n-objv1-150.pt"
    assert converted["source"]["classes"] == ["class0", "class1", "class2"]
    torch.save(converted, tmp_path / "c.pt")
    assert torch.load(tmp_path / "c.pt")["source"]["version"] == "8.3.222"  # loads in safe mode


def test_code_hidden_in_the_checkpoint_is_never_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"

    class Hostile:
        def __reduce__(self) -> tuple[Any, ...]:
            import os

            return (os.system, (f"touch {marker}",))

    path = write(tmp_path / "u.pt", small(), trap=Hostile())
    checkpoint = read_checkpoint(path)  # a normal torch.load(weights_only=False) would run it
    assert "trap" in checkpoint and not marker.exists()
    convert(path)
    assert not marker.exists()


def test_a_checkpoint_of_another_size_is_refused(tmp_path: Path) -> None:
    other = YOLO26(ModelConfig(0.5, 0.5, num_classes=3, reg_max=1), input_size=64)
    with pytest.raises(ValueError, match="tensors"):
        convert(write(tmp_path / "u.pt", other))


def test_a_file_without_class_count_or_tensors_is_refused(tmp_path: Path) -> None:
    torch.save({"model": torch.nn.Linear(2, 2)}, tmp_path / "plain.pt")
    with pytest.raises(ValueError, match="no model tensors|number of classes"):
        convert(tmp_path / "plain.pt")
    (tmp_path / "text.pt").write_text("not a checkpoint")
    with pytest.raises(ValueError, match="zip"):
        read_checkpoint(tmp_path / "text.pt")
