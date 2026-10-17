import numpy as np
import pytest

from palletvision_infer.media.preprocess import letterbox, restore_boxes


def test_letterbox_shape_range_and_padding() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)  # wide image
    blob, meta = letterbox(image, size=64)
    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert blob.max() == pytest.approx(1.0)
    assert blob[0, 0, 0, 0] == pytest.approx(114 / 255)  # top-left is padding
    assert meta.scale == pytest.approx(0.32)


def test_letterbox_converts_bgr_to_rgb() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    blob, _ = letterbox(image, size=10)
    assert blob[0, 2].min() == pytest.approx(1.0)  # blue ends up in channel 2
    assert blob[0, 0].max() == 0.0


def test_restore_boxes_inverts_letterbox_and_clips() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, meta = letterbox(image, size=64)
    boxes = np.array([[0.0, 0.0, 64.0, 64.0]], dtype=np.float32)
    restored = restore_boxes(boxes, meta, image.shape[:2])
    assert restored[0].tolist() == [0.0, 0.0, 200.0, 100.0]
