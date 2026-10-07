from pathlib import Path

import pytest
import yaml

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def load(name: str) -> dict:
    return yaml.safe_load((CONTRACTS / name).read_text())


def test_classes_stage1_is_single_pallet_class() -> None:
    classes = load("classes.yaml")
    assert classes["stage1"] == ["pallet"]


def test_model_io_declares_input_and_outputs() -> None:
    io = load("model_io.yaml")
    assert io["input"]["name"] == "images"
    assert io["input"]["shape"] == [1, 3, 640, 640]
    assert io["input"]["scale"] == pytest.approx(1 / 255)
    out = io["outputs"][0]
    assert out["name"] == "detections"
    assert out["shape"][-1] == 6  # x1, y1, x2, y2, score, class
