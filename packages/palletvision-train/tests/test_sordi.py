"""SORDI scenes: reading them, and mixing them into each epoch with the real images."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torch.utils.data import ConcatDataset

from palletvision_model.config import SIZES
from palletvision_model.loss import DetectionLoss
from palletvision_model.yolo26 import YOLO26
from palletvision_train.cli import CONFIG_DIR, REPO, build_dataset, build_sampler
from palletvision_train.coco import CocoDetection
from palletvision_train.data import LetterboxedDataset
from palletvision_train.sordi import MixedSampler, load_sordi_records, scene_folders
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import TrainResult, train

CLASSES = ["pallet"]
CLASS_MAP = {"pallet": "pallet", "stillage": "pallet"}
HAS_DATA = (REPO / "data/loco/annotations").exists() and (REPO / "data/sordid-02/images").exists()


def obj(name: str, left: int, top: int, right: int, bottom: int) -> dict[str, object]:
    return {
        "Id": 1, "ObjectClassName": name, "Left": left, "Top": top, "Right": right,
        "Bottom": bottom, "ObjectClassId": 1100,
    }  # fmt: skip


def make_scene(
    root: Path, scene: dict[str, list[dict[str, object]]], size: tuple[int, int] = (64, 48)
) -> Path:
    """``scene`` maps an image number to its label objects; writes images/ and labels/json/."""
    (root / "images").mkdir(parents=True)
    (root / "labels" / "json").mkdir(parents=True)
    for number, objects in scene.items():
        Image.new("RGB", size, (10, 20, 30)).save(root / "images" / f"{number}.jpg")
        (root / "labels" / "json" / f"{number}.json").write_text(json.dumps(objects))
    return root


def test_pallets_and_stillages_become_one_class_and_the_rest_are_dropped(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"1": [
        obj("pallet", 5, 6, 25, 16), obj("stillage", 30, 10, 60, 40),
        obj("dolly", 0, 0, 20, 20), obj("forklift", 1, 1, 30, 30),
    ]})  # fmt: skip
    (record,) = load_sordi_records(root, CLASS_MAP, CLASSES)
    assert record.labels.tolist() == [[0, 5, 6, 25, 16], [0, 30, 10, 60, 40]]
    assert record.labels.dtype == np.float32
    assert (record.width, record.height) == (64, 48)


def test_an_image_without_a_target_is_kept_as_an_empty_example(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"1": [obj("dolly", 0, 0, 20, 20)], "2": []})
    records = load_sordi_records(root, CLASS_MAP, CLASSES)
    assert [r.labels.shape for r in records] == [(0, 5), (0, 5)]


def test_images_are_ordered_by_number_not_by_text(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"10": [], "2": [], "1": []})
    assert [r.path.stem for r in load_sordi_records(root, CLASS_MAP, CLASSES)] == ["1", "2", "10"]


def test_boxes_of_one_pixel_or_less_are_noise(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"1": [obj("pallet", 5, 5, 6, 30), obj("pallet", 5, 5, 40, 5)]})
    assert load_sordi_records(root, CLASS_MAP, CLASSES)[0].labels.shape == (0, 5)


def test_an_image_without_labels_or_the_reverse_is_an_error(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"1": [], "2": []})
    (root / "labels" / "json" / "2.json").unlink()
    with pytest.raises(ValueError, match="1 images without labels"):
        load_sordi_records(root, CLASS_MAP, CLASSES)


def test_a_class_map_target_that_is_not_a_trained_class_is_an_error(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"1": []})
    with pytest.raises(ValueError, match="not in classes"):
        load_sordi_records(root, {"pallet": "crate"}, CLASSES)


def test_an_empty_folder_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels" / "json").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_sordi_records(tmp_path, CLASS_MAP, CLASSES)


def test_the_records_read_like_any_other_dataset(tmp_path: Path) -> None:
    root = make_scene(tmp_path, {"1": [obj("pallet", 5, 6, 25, 16)]})
    image, labels = CocoDetection(load_sordi_records(root, CLASS_MAP, CLASSES))[0]
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8
    assert labels.tolist() == [[0, 5, 6, 25, 16]]


def test_a_scene_with_domain_randomisation_levels_yields_each_level(tmp_path: Path) -> None:
    one = make_scene(tmp_path / "DR-1", {"1": []})
    two = make_scene(tmp_path / "DR-2", {"1": []})
    (tmp_path / "notes").mkdir()  # a folder without images/ and labels/ is not a level
    assert scene_folders(tmp_path) == [one, two]


def test_a_flat_scene_is_its_own_folder(tmp_path: Path) -> None:
    root = make_scene(tmp_path / "scene", {"1": []})
    assert scene_folders(root) == [root]


def test_a_folder_without_a_scene_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no images/"):
        scene_folders(tmp_path)


# ---- the sampler ----


def test_an_epoch_is_every_real_image_once_plus_a_synthetic_draw() -> None:
    sampler = MixedSampler(n_real=70, n_synthetic=500, share=0.3)
    epoch = list(sampler)
    assert len(sampler) == len(epoch) == 100  # 70 / (1 - 0.3)
    assert len(set(epoch)) == 100  # nothing twice
    assert sorted(i for i in epoch if i < 70) == list(range(70))  # all real, once
    synthetic = [i for i in epoch if i >= 70]
    assert len(synthetic) == 30 and all(70 <= i < 570 for i in synthetic)


def test_the_synthetic_draw_changes_every_epoch_and_the_order_is_shuffled() -> None:
    torch.manual_seed(0)
    sampler = MixedSampler(70, 500, 0.3)
    first, second = list(sampler), list(sampler)
    assert {i for i in first if i >= 70} != {i for i in second if i >= 70}
    assert first != sorted(first)  # not real first then synthetic


def test_the_epochs_follow_torchs_random_state_so_a_resumed_run_repeats_them() -> None:
    sampler = MixedSampler(20, 100, 0.5)
    torch.manual_seed(7)
    state = torch.get_rng_state()
    a = list(sampler)
    torch.set_rng_state(state)
    assert list(sampler) == a


def test_no_synthetic_share_is_just_the_real_images() -> None:
    assert sorted(MixedSampler(10, 50, 0.0)) == list(range(10))


def test_the_share_is_rounded_to_whole_images() -> None:
    assert len(MixedSampler(2820, 12941, 0.3)) == 4029  # 2820 + round(2820 * 0.3 / 0.7)


@pytest.mark.parametrize("share", [-0.1, 1.0, 1.5])
def test_a_share_outside_zero_to_one_is_rejected(share: float) -> None:
    with pytest.raises(ValueError, match="share"):
        MixedSampler(10, 50, share)


def test_a_pool_too_small_for_the_share_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs 43 synthetic images but the pool has 10"):
        MixedSampler(100, 10, 0.3)


def test_training_runs_epochs_as_long_as_the_sampler_says(tmp_path: Path) -> None:
    torch.manual_seed(0)
    real = LetterboxedDataset(SyntheticBoxes(n=6, size=64), size=64)
    pool = LetterboxedDataset(SyntheticBoxes(n=20, size=64, seed=1), size=64)
    both: ConcatDataset[tuple[np.ndarray, np.ndarray]] = ConcatDataset([real, pool])
    sampler = MixedSampler(6, 20, 0.4)  # 6 real + 4 synthetic = 10 images per epoch
    result = train(
        YOLO26(SIZES["n"], input_size=64), both, epochs=2, batch_size=5, lr=1e-3, device="cpu",
        output_dir=tmp_path, sampler=sampler,
    )  # fmt: skip
    assert len(result.losses) == 4  # 2 epochs x 2 batches of 5, not 2 x 4 batches over all 26


# ---- the configuration ----


@pytest.mark.skipif(not HAS_DATA, reason="LOCO or SORDI data not present")
def test_the_loco_sordi_config_builds_the_real_dataset_and_the_mixed_epoch() -> None:
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        cfg = compose(
            "config",
            overrides=[
                "data=loco_sordi",
                "data.limit=40",
                "data.sordi.limit=50",
                "data.sordi.roots=[data/sordid-02]",  # not every scene: they take minutes to read
            ],
        )
    dataset = build_dataset(cfg, ["pallet"])
    sampler = build_sampler(cfg, dataset)
    assert isinstance(sampler, MixedSampler)
    assert (sampler.n_real, sampler.n_synthetic) == (40, 50)
    assert len(sampler) == 40 + round(40 * 0.3 / 0.7)
    image, labels = dataset[sampler.n_real]  # the first synthetic image, letterboxed to 640
    assert image.shape == (640, 640, 3) and labels.shape[1] == 5


def test_other_datasets_keep_plain_shuffling() -> None:
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=["data=loco"])
    assert build_sampler(cfg, dataset=None) is None  # type: ignore[arg-type]


def test_the_loco_sordi_config_trains_pallets_and_stillages_as_one_class() -> None:
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=["data=loco_sordi"])
    assert dict(cfg.data.class_map) == dict(cfg.data.sordi.class_map) == CLASS_MAP
    assert cfg.data.sordi.share == 0.3
    assert list(cfg.data.annotations) == ["train", "val"]  # validation stays LOCO's real split


class CrashingLoss(DetectionLoss):
    """The real loss, except it raises on one chosen call, as a crash would."""

    def __init__(self, crash_at: int) -> None:
        super().__init__()
        self.calls, self.crash_at = 0, crash_at

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == self.crash_at:
            raise RuntimeError("simulated crash")
        return super().forward(*args, **kwargs)


def test_a_resumed_run_with_the_mixed_epoch_follows_the_uninterrupted_curve(tmp_path: Path) -> None:
    def data() -> ConcatDataset[tuple[np.ndarray, np.ndarray]]:
        real = LetterboxedDataset(SyntheticBoxes(n=6, size=64), size=64)
        pool = LetterboxedDataset(SyntheticBoxes(n=12, size=64, seed=1), size=64)
        return ConcatDataset([real, pool])

    def model(seed: int) -> YOLO26:
        torch.manual_seed(seed)
        return YOLO26(SIZES["n"], input_size=64)

    # an epoch is 6 real + 4 synthetic = 10 images = 3 batches of at most 4
    def run(output: str, net: YOLO26, **extra: Any) -> TrainResult:
        return train(
            net, data(), epochs=4, batch_size=4, lr=1e-3, device="cpu",
            output_dir=tmp_path / output, sampler=MixedSampler(6, 12, 0.4), **extra,
        )  # fmt: skip

    net = model(0)
    torch.manual_seed(5)
    full = run("whole", net)
    assert len(full.losses) == 12

    doomed = model(0)
    torch.manual_seed(5)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run("cut", doomed, loss_fn=CrashingLoss(8))  # the 2nd step of epoch 3; 2 epochs saved
    assert torch.load(tmp_path / "cut" / "last.pt")["epoch"] == 2
    resumed = run("cut", model(99), resume=tmp_path / "cut" / "last.pt")
    assert resumed.start_epoch == 2
    assert resumed.losses == pytest.approx(full.losses[6:], rel=1e-5)
