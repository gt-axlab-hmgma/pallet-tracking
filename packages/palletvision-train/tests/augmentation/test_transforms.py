import numpy as np

from palletvision_infer.media.preprocess import letterbox
from palletvision_preprocess import to_blob
from palletvision_train.augmentation.transforms import letterbox_sample


def test_training_pixels_equal_the_inference_input() -> None:
    image = np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    canvas, _ = letterbox_sample(image, np.zeros((0, 5), np.float32), size=64)
    blob, _ = letterbox(image, size=64)  # what the model is fed at inference
    assert np.array_equal(to_blob(canvas), blob)


def test_labels_keep_their_class_and_move_with_the_image() -> None:
    labels = np.array([[0, 0, 0, 200, 100], [1, 100, 50, 150, 75]], dtype=np.float32)
    _, moved = letterbox_sample(np.zeros((100, 200, 3), dtype=np.uint8), labels, size=64)
    assert moved[:, 0].tolist() == [0.0, 1.0]  # classes untouched
    assert np.allclose(moved[:, 1:], [[0, 16, 64, 48], [32, 32, 48, 40]])
    assert labels[0, 3] == 200  # the input is not modified


def test_an_image_without_labels_still_works() -> None:
    canvas, moved = letterbox_sample(
        np.zeros((50, 80, 3), dtype=np.uint8), np.zeros((0, 5), np.float32), size=32
    )
    assert canvas.shape == (32, 32, 3) and moved.shape == (0, 5)
