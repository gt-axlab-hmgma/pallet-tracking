from pathlib import Path

import pytest
from torch.utils.data import Dataset

from palletvision_train.data import LetterboxedDataset, Sample
from palletvision_train.overfit import run_gate
from palletvision_train.synthetic import SyntheticBoxes


def images(n: int = 2, size: int = 64) -> Dataset[Sample]:
    return LetterboxedDataset(SyntheticBoxes(n=n, size=size), size=size)


def test_the_gate_reports_failures_and_writes_overlays_when_barely_trained(tmp_path: Path) -> None:
    result = run_gate(images(), size=64, steps=3, device="cpu", out_dir=tmp_path)
    assert not result.passed
    assert any("AP50" in f for f in result.failures)
    assert result.objects == [1, 1] and len(result.confident) == 2
    assert 0 <= result.ap50 <= 1 and result.loss_start > 0
    assert (tmp_path / "overfit_0.jpg").exists() and (tmp_path / "overfit_1.jpg").exists()


def test_the_pass_criteria_follow_the_thresholds(tmp_path: Path) -> None:
    easy = run_gate(
        images(), size=64, steps=3, device="cpu", out_dir=tmp_path,
        min_loss_drop=-100.0, min_ap50=0.0, max_duplicate_ratio=1000.0,
    )  # fmt: skip
    assert easy.passed and easy.failures == []
    strict = run_gate(
        images(), size=64, steps=3, device="cpu", out_dir=tmp_path,
        min_loss_drop=-100.0, min_ap50=1.1, max_duplicate_ratio=1000.0,
    )  # fmt: skip
    assert [f for f in strict.failures if "AP50" in f] and not strict.passed


def test_too_many_confident_boxes_count_as_duplicates(tmp_path: Path) -> None:
    # a ratio of 0 allows only one confident box per image beyond zero objects: a barely trained
    # model that fires on many cells is flagged, a model with none is not
    result = run_gate(
        images(), size=64, steps=3, device="cpu", out_dir=tmp_path,
        min_loss_drop=-100.0, min_ap50=0.0, max_duplicate_ratio=0.0, conf=0.0001,
    )  # fmt: skip
    assert any("confident boxes" in f for f in result.failures)


@pytest.mark.slow
def test_the_gate_passes_on_synthetic_images(tmp_path: Path) -> None:
    """The real thing, on CPU (about a minute). Run it with ``uv run pytest -m slow``.

    At 120 steps the loss is already down 94% but AP50 is 0.04; the detection check is what
    shows the pipeline has really learned, which is why the gate has more than one criterion.
    """
    result = run_gate(images(), size=64, steps=300, device="cpu", out_dir=tmp_path)
    assert result.passed, result.failures
    assert result.ap50 >= 0.95 and result.loss_drop >= 0.9
