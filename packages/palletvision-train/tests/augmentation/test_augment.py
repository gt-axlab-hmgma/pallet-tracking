"""Training augmentation: labels must stay glued to the pixels they mark."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26
from palletvision_train.augmentation.augment import (
    AugmentConfig,
    AugmentedDataset,
    flip_lr,
    jitter_colour,
    mosaic,
    scale_and_shift,
)
from palletvision_train.augmentation.transforms import letterbox_sample
from palletvision_train.cli.train import CONFIG_DIR, REPO, build_dataset, build_epoch_hook
from palletvision_train.datasets.data import LetterboxedDataset
from palletvision_train.datasets.sordi import MixedPartners
from palletvision_train.datasets.synthetic import SyntheticBoxes
from palletvision_train.training.trainer import train

Sample = tuple[np.ndarray, np.ndarray]
HAS_LOCO = (REPO / "data/loco/annotations").exists()
COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # one per class, BGR
OFF = AugmentConfig(mosaic=0, scale=0, translate=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0)


def scene(cls: int, box: tuple[int, int, int, int] = (100, 60, 300, 200)) -> tuple[np.ndarray, Any]:
    """A 400x300 grey picture with one solid rectangle whose colour says its class."""
    image = np.full((300, 400, 3), 60, dtype=np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = COLOURS[cls]
    return image, np.array([[cls, x1, y1, x2, y2]], dtype=np.float32)


class Scenes(Dataset[Sample]):
    """Four pictures, one per class and colour, each with one rectangle."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return scene(index, (60 + 20 * index, 40 + 10 * index, 260 + 20 * index, 180 + 10 * index))


def colour_box(image: np.ndarray, cls: int) -> tuple[float, float, float, float] | None:
    """The bounding box of the pixels that are (nearly) the class colour, or None."""
    near = (np.abs(image.astype(int) - np.array(COLOURS[cls])) < 25).all(axis=2)
    ys, xs = np.nonzero(near)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def partners_in_order() -> Any:
    sequence = iter([1, 2, 3] * 10_000)
    return lambda: next(sequence)


# ---- the labels follow the pixels ----


@pytest.mark.parametrize("use_mosaic", [True, False])
def test_every_box_still_encloses_exactly_its_object(use_mosaic: bool) -> None:
    config = AugmentConfig(mosaic=1.0 if use_mosaic else 0.0, hsv_h=0, hsv_s=0, hsv_v=0)
    checked = 0
    for seed in range(150):
        random.seed(seed)
        data = AugmentedDataset(Scenes(), size=320, config=config, partners=partners_in_order())
        image, labels = data[0]
        assert image.shape == (320, 320, 3) and image.dtype == np.uint8
        classes = labels[:, 0].astype(int).tolist()
        assert len(set(classes)) == len(classes)  # one object per colour, so one box per colour
        for cls, x1, y1, x2, y2 in labels:
            found = colour_box(image, int(cls))
            assert found is not None, f"seed {seed}: a box for class {int(cls)} marks no object"
            assert found == pytest.approx((x1, y1, x2, y2), abs=3.5), f"seed {seed} class {cls}"
            checked += 1
        for cls in range(4):  # and every object left mostly in view has a box
            found = colour_box(image, cls)
            # an object is 160 x 112 here and zoom reaches 1.5, so a tenth of it is at most 4,032
            # pixels: a bigger piece in view must have kept its box
            if found and (found[2] - found[0]) * (found[3] - found[1]) > 4_200:
                assert cls in classes, f"seed {seed}: class {cls} is visible but has no box"
    assert checked > 100


def test_a_mosaic_box_never_covers_a_neighbouring_tile() -> None:
    random.seed(1)
    image, labels = mosaic([scene(i) for i in range(4)], 320)
    assert image.shape == (640, 640, 3) and labels.shape[1] == 6
    for cls, x1, y1, x2, y2, full in labels:
        assert full >= (x2 - x1) * (y2 - y1) - 1e-3  # the cut box is never bigger than the object
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        assert tuple(image[cy, cx]) == COLOURS[int(cls)]


def test_with_every_strength_zero_the_picture_and_labels_come_back_unchanged() -> None:
    image, labels = scene(2)
    for size in (400,):
        padded, plain = letterbox_sample(image, labels, size)
        out, out_labels = AugmentedDataset(_One(image, labels), size, OFF)[0]
        assert np.array_equal(out, padded) and np.array_equal(out_labels, plain)


class _One(Dataset[Sample]):
    def __init__(self, image: np.ndarray, labels: np.ndarray) -> None:
        self.image, self.labels = image, labels

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.image, self.labels


def test_scale_and_shift_moves_the_boxes_the_way_it_moves_the_pixels() -> None:
    image = np.full((100, 100, 3), 60, dtype=np.uint8)
    image[20:60, 30:70] = COLOURS[0]
    labels = np.array([[0, 30, 20, 70, 60, 40 * 40]], dtype=np.float32)
    random.seed(3)
    out, boxes = scale_and_shift(image, labels, 100, scale=0.5, translate=0.1)
    (cls, x1, y1, x2, y2) = boxes[0]
    found = colour_box(out, 0)
    assert found == pytest.approx((x1, y1, x2, y2), abs=1.5)
    assert (x2 - x1) != pytest.approx(40, abs=1e-3)  # it really was zoomed


def test_the_zoom_stays_within_the_configured_range() -> None:
    image = np.full((100, 100, 3), 60, dtype=np.uint8)
    labels = np.array([[0, 20, 20, 60, 60, 1600]], dtype=np.float32)
    widths = []
    for seed in range(200):
        random.seed(seed)
        _, boxes = scale_and_shift(image, labels, 100, scale=0.5, translate=0.0)
        if len(boxes):
            widths.append(boxes[0, 3] - boxes[0, 1])
    assert min(widths) >= 40 * 0.5 - 1e-3 and max(widths) <= 40 * 1.5 + 1e-3
    assert min(widths) < 40 * 0.6 and max(widths) > 40 * 1.4  # both zooming out and in


# ---- which boxes survive ----


def kept(box: list[float], full: float, size: int = 100) -> bool:
    image = np.full((size, size, 3), 60, dtype=np.uint8)
    labels = np.array([[0, *box, full]], dtype=np.float32)
    return len(scale_and_shift(image, labels, size, scale=0.0, translate=0.0)[1]) == 1


def test_a_box_thinner_than_two_pixels_is_dropped() -> None:
    assert kept([10, 10, 40, 13], 90.0)
    assert not kept([10, 10, 40, 12], 60.0)
    assert not kept([10, 10, 11.5, 40], 45.0)


def test_a_box_with_a_tenth_or_less_of_the_object_left_is_dropped() -> None:
    assert kept([0, 10, 20, 50], 800 / 0.5)  # 800 of 1600 in view
    assert not kept([0, 10, 4, 50], 160 / 0.1)  # exactly a tenth: not more than
    assert not kept([0, 10, 4, 50], 160 / 0.05)


def test_the_visible_share_is_measured_against_the_object_at_the_new_zoom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random, "uniform", lambda low, high: 1.5 if high > 1 else 0.5)
    image = np.full((100, 100, 3), 60, dtype=np.uint8)
    # a 40 x 40 object zoomed to 60 x 60 (3,600 px); only a 5 x 60 strip (300 px, 8%) stays in view
    labels = np.array([[0, 80, 20, 120, 60, 1600]], dtype=np.float32)
    assert len(scale_and_shift(image, labels, 100, scale=0.5, translate=0.1)[1]) == 0
    # the same strip of a smaller object is more than a tenth of it
    labels = np.array([[0, 80, 20, 120, 60, 600]], dtype=np.float32)
    assert len(scale_and_shift(image, labels, 100, scale=0.5, translate=0.1)[1]) == 1


def test_the_shift_moves_the_picture_both_ways_along_both_axes() -> None:
    image = np.full((100, 100, 3), 60, dtype=np.uint8)
    labels = np.array([[0, 40, 40, 60, 60, 400]], dtype=np.float32)
    starts = []
    for seed in range(200):
        random.seed(seed)
        _, boxes = scale_and_shift(image, labels, 100, scale=0.0, translate=0.1)
        starts.append(boxes[0, 1:3])
    starts_array = np.array(starts)
    assert starts_array.min(axis=0) == pytest.approx([30, 30], abs=1.5)  # shifted up to 10 px
    assert starts_array.max(axis=0) == pytest.approx([50, 50], abs=1.5)
    assert np.corrcoef(starts_array.T)[0, 1] == pytest.approx(0, abs=0.25)  # x and y independent


def test_a_box_pushed_off_the_picture_is_dropped() -> None:
    assert not kept([120, 10, 160, 50], 1600.0)


def test_an_extreme_sliver_is_dropped() -> None:
    assert kept([0, 0, 250, 3], 750.0, size=300)  # 83 : 1, thin but allowed
    assert not kept([0, 0, 250, 2.4], 600.0, size=300)  # 104 : 1: over 2 px high, too extreme


# ---- flip and colour ----


def test_a_flip_mirrors_the_picture_and_its_boxes_together() -> None:
    image, labels = scene(0)
    flipped, boxes = flip_lr(image, labels)
    assert boxes.tolist() == [[0, 400 - 300, 60, 400 - 100, 200]]
    assert colour_box(flipped, 0) == pytest.approx(tuple(boxes[0, 1:5]), abs=0.5)
    again, back = flip_lr(flipped, boxes)
    assert np.array_equal(again, image) and np.array_equal(back, labels)


def test_a_flip_only_happens_with_the_configured_chance() -> None:
    config = AugmentConfig(mosaic=0, scale=0, translate=0, fliplr=0.5, hsv_h=0, hsv_s=0, hsv_v=0)
    data = AugmentedDataset(_One(*scene(0, (50, 60, 150, 200))), 400, config)
    random.seed(0)
    mirrored = sum(data[0][1][0, 1] > 200 for _ in range(400))  # the box moved to the right
    assert 150 < mirrored < 250


def test_colour_jitter_with_zero_gains_is_the_same_picture() -> None:
    image = np.random.default_rng(0).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    assert jitter_colour(image, 0, 0, 0) is image


def test_value_gain_scales_brightness_within_its_range() -> None:
    image = np.full((8, 8, 3), 100, dtype=np.uint8)  # grey: hue and saturation do not matter
    seen = []
    for seed in range(300):
        random.seed(seed)
        seen.append(int(jitter_colour(image, 0, 0, 0.4)[0, 0, 0]))
    assert min(seen) >= 60 and max(seen) <= 140  # 100 * (1 +- 0.4)
    assert min(seen) < 70 and max(seen) > 130  # and the whole range is used


def test_hue_and_saturation_jitter_leave_grey_pictures_grey() -> None:
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    random.seed(0)
    out = jitter_colour(image, 0.02, 0.7, 0)
    assert (out.max(axis=2) - out.min(axis=2)).max() <= 1


def test_colour_jitter_changes_a_coloured_picture() -> None:
    image, _ = scene(0)
    random.seed(0)
    assert not np.array_equal(jitter_colour(image, 0.02, 0.7, 0.4), image)


# ---- the dataset ----


def test_seeded_runs_repeat_and_different_seeds_differ() -> None:
    data = AugmentedDataset(Scenes(), size=320, partners=partners_in_order())

    def draw(seed: int) -> tuple[np.ndarray, np.ndarray]:
        random.seed(seed)
        data.partners = partners_in_order()
        return data[0]

    (a_image, a_labels), (b_image, b_labels) = draw(5), draw(5)
    assert np.array_equal(a_image, b_image) and np.array_equal(a_labels, b_labels)
    assert not np.array_equal(draw(6)[0], a_image)


def test_mosaic_off_trains_on_single_images_even_when_the_probability_is_one() -> None:
    config = AugmentConfig(mosaic=1.0, scale=0, translate=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0)
    data = AugmentedDataset(Scenes(), size=320, config=config, partners=partners_in_order())
    random.seed(0)
    assert len(data[0][1]) > 1  # four pictures were joined
    data.set_mosaic(False)
    image, labels = data[0]
    plain_image, plain_labels = letterbox_sample(*Scenes()[0], 320)
    assert np.array_equal(image, plain_image) and np.array_equal(labels, plain_labels)
    data.set_mosaic(True)
    assert len(data[0][1]) > 1


class FullFrame(Dataset[Sample]):
    """Twelve 64-pixel pictures that each have one box around the whole frame."""

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full((64, 64, 3), 100, np.uint8), np.array([[0, 0, 0, 64, 64]], np.float32)


def test_the_mosaic_switch_reaches_data_loader_workers_that_are_already_running() -> None:
    config = AugmentConfig(mosaic=1.0, scale=0, translate=0, hsv_h=0, hsv_s=0, hsv_v=0, fliplr=0)
    torch.manual_seed(0)
    data = AugmentedDataset(FullFrame(), size=64, config=config)
    loader = DataLoader(data, batch_size=4, num_workers=2, persistent_workers=True, collate_fn=list)

    def mean_boxes() -> float:
        return float(np.mean([len(labels) for batch in loader for _, labels in batch]))

    assert mean_boxes() > 2.0  # mosaic joins four pictures, each with a box in view
    data.set_mosaic(False)
    assert mean_boxes() == 1.0  # the same workers now see plain images


def test_mosaic_partners_follow_the_epochs_real_and_synthetic_mixture() -> None:
    random.seed(0)
    partners = MixedPartners(n_real=100, n_synthetic=100, share=0.3)
    draws = [partners() for _ in range(4000)]
    real = sum(d < 100 for d in draws) / len(draws)
    assert 0.66 < real < 0.74  # 70% real, 30% from the synthetic half
    assert all(0 <= d < 200 for d in draws)
    assert all(MixedPartners(10, 0, 0.5)() < 10 for _ in range(50))  # no pool, only real


# ---- the trainer and the configuration ----


def test_the_trainer_announces_each_epoch_it_runs(tmp_path: Path) -> None:
    seen: list[int] = []
    torch.manual_seed(0)
    data = LetterboxedDataset(SyntheticBoxes(n=4, size=64), size=64)
    train(
        YOLO26(SIZES["n"], input_size=64), data, epochs=3, batch_size=4, lr=1e-3, device="cpu",
        output_dir=tmp_path, on_epoch_start=seen.append,
    )  # fmt: skip
    assert seen == [0, 1, 2]
    seen.clear()
    resumed = YOLO26(SIZES["n"], input_size=64)
    train(
        resumed, data, epochs=3, batch_size=4, lr=1e-3, device="cpu", output_dir=tmp_path,
        resume=tmp_path / "last.pt", on_epoch_start=seen.append,
    )  # fmt: skip
    assert seen == []  # all three epochs were already done


def test_mosaic_closes_for_the_last_epochs_only() -> None:
    cfg = OmegaConf.create({"train": {"epochs": 150}})
    data = AugmentedDataset(Scenes(), size=320, config=AugmentConfig(close_mosaic=8))
    hook = build_epoch_hook(cfg, data)
    assert hook is not None
    states = {}
    for epoch in (0, 100, 141, 142, 143, 149):
        hook(epoch)
        states[epoch] = bool(data._mosaic_on.value)
    assert states == {0: True, 100: True, 141: True, 142: False, 143: False, 149: False}
    assert build_epoch_hook(cfg, LetterboxedDataset(Scenes(), 64)) is None


def test_the_loco_augmentation_defaults_are_the_reports_values_for_the_n_model() -> None:
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=["data=loco"])
    assert dict(cfg.data.augment) == {
        "enabled": True, "mosaic": 1.0, "scale": 0.5, "translate": 0.1, "fliplr": 0.5,
        "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.4, "close_mosaic": 8,
    }  # fmt: skip
    assert AugmentConfig() == AugmentConfig(
        **{k: v for k, v in cfg.data.augment.items() if k != "enabled"}
    )


def test_synthetic_data_is_not_augmented() -> None:
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=["data=synthetic", "data.size=64"])
    assert isinstance(build_dataset(cfg, ["pallet"]), LetterboxedDataset)


@pytest.mark.skipif(not HAS_LOCO, reason="LOCO data not present")
def test_loco_trains_augmented_unless_switched_off() -> None:
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        on = compose("config", overrides=["data=loco", "data.limit=6"])
        off = compose(
            "config", overrides=["data=loco", "data.limit=6", "data.augment.enabled=false"]
        )
    augmented = build_dataset(on, ["pallet"])
    assert isinstance(augmented, AugmentedDataset) and len(augmented) == 6
    image, labels = augmented[0]
    assert image.shape == (640, 640, 3) and labels.shape[1] == 5
    assert isinstance(build_dataset(off, ["pallet"]), LetterboxedDataset)
