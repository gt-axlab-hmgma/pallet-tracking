"""``pv-plot``: reading a metrics log and drawing it."""

import json
from pathlib import Path

import pytest

from palletvision_train.plot import load_run, plot_run, smooth

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_log(path: Path, metric: float | None = None) -> Path:
    rows = []
    for step in range(1, 7):
        rows.append(
            {
                "type": "step", "epoch": (step - 1) // 3, "step": step, "alpha": 0.8 - step / 20,
                "lr": 0.01, "loss": 10.0 / step, "total_one2many": 9.0 / step,
                "total_one2one": 11.0 / step,
            }
        )  # fmt: skip
        if step % 3 == 0:
            rows.append(
                {
                    "type": "epoch", "epoch": step // 3 - 1, "step": step, "alpha": 0.5,
                    "loss": 1.0, "metric": metric if step == 6 else None, "seconds": 1.0,
                }
            )  # fmt: skip
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_the_log_is_split_into_steps_and_epochs(tmp_path: Path) -> None:
    run = load_run(write_log(tmp_path / "metrics.jsonl"))
    assert [row["step"] for row in run.steps] == [1, 2, 3, 4, 5, 6]
    assert [row["epoch"] for row in run.epochs] == [0, 1]


def test_a_run_folder_is_read_like_its_log(tmp_path: Path) -> None:
    write_log(tmp_path / "metrics.jsonl")
    assert load_run(tmp_path) == load_run(tmp_path / "metrics.jsonl")


def test_smoothing_is_the_mean_of_the_last_window_values() -> None:
    assert smooth([1.0, 2.0, 3.0, 4.0, 10.0], 3) == [1.0, 1.5, 2.0, 3.0, 17.0 / 3]
    assert smooth([4.0, 8.0], 1) == [4.0, 8.0]
    assert smooth([], 3) == []


@pytest.mark.parametrize("metric", [None, 0.42])
def test_the_picture_is_written_with_or_without_a_validation_metric(
    tmp_path: Path, metric: float | None
) -> None:
    run = load_run(write_log(tmp_path / "metrics.jsonl", metric))
    out = plot_run(run, tmp_path / "curves.png", window=2)
    assert out == tmp_path / "curves.png"
    assert out.read_bytes()[:8] == PNG_SIGNATURE and out.stat().st_size > 5_000
