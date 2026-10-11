import numpy as np
import pytest

from palletvision_preprocess import (
    PAD_VALUE,
    letterbox,
    letterbox_boxes,
    letterbox_image,
    restore_boxes,
    to_blob,
)


def random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_canvas_is_square_grey_padded_and_keeps_the_image_in_the_middle() -> None:
    canvas, meta = letterbox_image(np.full((100, 200, 3), 255, dtype=np.uint8), size=64)
    assert canvas.shape == (64, 64, 3) and canvas.dtype == np.uint8
    assert meta.scale == pytest.approx(0.32)  # 200 px wide -> 64
    assert (meta.pad_x, meta.pad_y) == (0.0, 16.0)  # 64 x 32 image, centred vertically
    assert (canvas[:16] == PAD_VALUE).all() and (canvas[48:] == PAD_VALUE).all()
    assert (canvas[16:48] == 255).all()


@pytest.mark.parametrize("hw", [(100, 200), (200, 100), (480, 640), (37, 91), (64, 64)])
def test_the_inference_blob_is_exactly_the_canvas_converted(hw: tuple[int, int]) -> None:
    """Training and inference see identical pixels: one is the other, normalised."""
    image = random_image(*hw)
    canvas, meta = letterbox_image(image, size=64)
    blob, blob_meta = letterbox(image, size=64)
    assert blob_meta == meta
    assert np.array_equal(blob, to_blob(canvas))
    assert blob.shape == (1, 3, 64, 64) and blob.dtype == np.float32
    red = canvas[..., 2].astype(np.float32) / 255.0  # channel 0 is red: BGR flipped to RGB
    assert np.array_equal(blob[0, 0], red)


def test_blob_is_normalised_and_converts_bgr_to_rgb() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    blob, _ = letterbox(image, size=10)
    assert blob[0, 2].min() == pytest.approx(1.0)  # blue ends up in channel 2
    assert blob[0, 0].max() == 0.0
    padded, _ = letterbox(np.full((100, 200, 3), 255, dtype=np.uint8), size=64)
    assert padded[0, 0, 0, 0] == pytest.approx(PAD_VALUE / 255)


def test_known_box_mapping() -> None:
    _, meta = letterbox_image(np.zeros((100, 200, 3), dtype=np.uint8), size=64)
    boxes = np.array([[0.0, 0.0, 200.0, 100.0], [100.0, 50.0, 150.0, 75.0]], dtype=np.float32)
    moved = letterbox_boxes(boxes, meta)
    assert np.allclose(moved, [[0, 16, 64, 48], [32, 32, 48, 40]])  # x * 0.32, y * 0.32 + 16


@pytest.mark.parametrize("hw", [(100, 200), (200, 100), (480, 640), (37, 91), (1080, 1920)])
def test_restore_inverts_letterbox_boxes(hw: tuple[int, int]) -> None:
    h, w = hw
    _, meta = letterbox_image(np.zeros((h, w, 3), dtype=np.uint8), size=640)
    rng = np.random.default_rng(1)
    xy = rng.random((20, 2)) * np.array([w, h]) * 0.5
    boxes = np.concatenate([xy, xy + rng.random((20, 2)) * np.array([w, h]) * 0.4], axis=1)
    boxes = boxes.astype(np.float32)
    back = restore_boxes(letterbox_boxes(boxes, meta), meta, hw)
    assert np.allclose(back, boxes, atol=1e-3)


@pytest.mark.parametrize("hw", [(100, 200), (200, 100), (480, 640), (37, 91)])
def test_a_transformed_box_lands_on_the_pixels_it_describes(hw: tuple[int, int]) -> None:
    """The check that matters: the box and the picture are moved by the same amount."""
    h, w = hw
    image = np.zeros((h, w, 3), dtype=np.uint8)
    x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
    image[y1:y2, x1:x2] = 255  # a white rectangle
    canvas, meta = letterbox_image(image, size=128)
    (bx1, by1, bx2, by2) = letterbox_boxes(np.array([[x1, y1, x2, y2]], np.float32), meta)[0]
    inside = canvas[round(by1) + 2 : round(by2) - 2, round(bx1) + 2 : round(bx2) - 2]
    assert inside.mean() > 250  # the box interior is the white rectangle
    around = np.concatenate(
        [canvas[round(by1) - 3, round(bx1) : round(bx2)].ravel(),
         canvas[round(by1) : round(by2), round(bx1) - 3].ravel()]
    )  # fmt: skip
    assert around.mean() < 130  # just outside it is not (dark image or grey padding)


def test_letterbox_boxes_does_not_modify_its_input() -> None:
    boxes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    _, meta = letterbox_image(np.zeros((10, 20, 3), dtype=np.uint8), size=64)
    before = boxes.copy()
    letterbox_boxes(boxes, meta)
    assert np.array_equal(boxes, before)


@pytest.mark.parametrize("hw", [(101, 100), (100, 101), (99, 200), (200, 99), (37, 91), (480, 640)])
def test_the_recorded_offsets_are_the_whole_pixels_the_image_was_really_placed_at(
    hw: tuple[int, int],
) -> None:
    """An odd leftover (say 1 px of padding to split in two) must be floored in the metadata
    as well as on the canvas, or every box would be off by half a pixel."""
    image = np.full((*hw, 3), 255, dtype=np.uint8)
    canvas, meta = letterbox_image(image, size=64)
    white = (canvas == 255).all(axis=-1)
    rows, cols = np.nonzero(white)
    assert (meta.pad_x, meta.pad_y) == (float(cols.min()), float(rows.min()))
    assert meta.pad_x == int(meta.pad_x) and meta.pad_y == int(meta.pad_y)
