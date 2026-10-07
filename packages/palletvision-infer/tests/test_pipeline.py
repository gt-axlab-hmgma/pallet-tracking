from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from palletvision_infer.cli import main
from palletvision_infer.detections import Detections
from palletvision_infer.overlay import draw
from palletvision_infer.pipeline import DummyDetector, run


def test_draw_marks_box_without_mutating_input() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    det = Detections(
        np.array([[5, 5, 40, 40]], np.float32), np.array([0.8], np.float32), np.array([0])
    )
    out = draw(image, det, fps=12.0)
    assert image.sum() == 0
    assert out.sum() > 0


def test_run_writes_annotated_video(make_video: Callable[..., Path], tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    n = run(make_video(), DummyDetector(), out_path=out, fps=10)
    assert n == 10
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_cli_runs_with_dummy_detector(make_video: Callable[..., Path], tmp_path: Path) -> None:
    out = tmp_path / "cli.mp4"
    main(["--source", str(make_video()), "--out", str(out)])
    assert out.exists()


def test_cli_requires_source() -> None:
    with pytest.raises(SystemExit):
        main([])
