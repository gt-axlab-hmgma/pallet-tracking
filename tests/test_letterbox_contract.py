"""The letterbox must follow contracts/model_io.yaml, not just agree with itself."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from palletvision_preprocess import PAD_VALUE, letterbox

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "model_io.yaml"


def test_pad_value_and_scaling_come_from_the_contract() -> None:
    spec = yaml.safe_load(CONTRACT.read_text())["input"]
    assert spec["letterbox_pad_value"] == PAD_VALUE
    assert spec["channel_order"] == "RGB"
    white, _ = letterbox(np.full((10, 10, 3), 255, dtype=np.uint8), size=10)
    assert white.max() == pytest.approx(255 * spec["scale"])  # scale is 1 / 255
    assert white.shape == (1, 3, 10, 10) and white.dtype == np.float32
    padded, _ = letterbox(np.zeros((10, 20, 3), dtype=np.uint8), size=20)
    assert padded[0, 0, 0, 0] == pytest.approx(spec["letterbox_pad_value"] * spec["scale"])
