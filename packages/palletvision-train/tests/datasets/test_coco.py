import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from palletvision_train.datasets.coco import CocoDetection, load_records

CLASSES = ["pallet"]
CLASS_MAP = {"pallet": "pallet", "stillage": "pallet"}
CATEGORIES = [
    {"id": 7, "name": "pallet"},
    {"id": 10, "name": "stillage"},
    {"id": 11, "name": "pallet_truck"},
    {"id": 5, "name": "forklift"},
]


def ann(image_id: int, category_id: int, bbox: list[float], **extra: object) -> dict[str, object]:
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, **extra}


def write_image(
    root: Path, relative: str, width: int, height: int, bgr: tuple[int, ...] = (0, 0, 0)
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = bgr
    cv2.imwrite(str(path), image)


def write_json(
    path: Path, images: list[dict[str, object]], annotations: list[dict[str, object]]
) -> Path:
    path.write_text(
        json.dumps({"images": images, "categories": CATEGORIES, "annotations": annotations})
    )
    return path


@pytest.fixture
def two_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A train file and a val file that REUSE image id 1 for different images, like LOCO does."""
    root = tmp_path / "loco"
    write_image(root, "subset-2/a/one.jpg", 40, 30)
    write_image(root, "subset-2/a/two.jpg", 40, 30)
    write_image(root, "subset-1/three.jpg", 20, 10)
    train = write_json(
        tmp_path / "train.json",
        [
            {
                "id": 1,
                "path": "/dataset/subset-2/a/one.jpg",
                "width": 40,
                "height": 30,
                "file_name": "one.jpg",
            },
            {
                "id": 2,
                "path": "/dataset/subset-2/a/two.jpg",
                "width": 40,
                "height": 30,
                "file_name": "two.jpg",
            },
        ],
        [
            ann(1, 7, [2.0, 3.0, 10.0, 6.0]),  # a pallet
            ann(1, 10, [20.0, 10.0, 8.0, 8.0]),  # a stillage: same class
            ann(1, 11, [0.0, 0.0, 5.0, 5.0]),  # a pallet truck: dropped by the class map
            ann(1, 5, [1.0, 1.0, 5.0, 5.0]),  # a forklift: dropped too
            ann(1, 7, [0.0, 0.0, 1.0, 9.0]),  # 1 px wide: dropped
            ann(1, 7, [0.0, 0.0, 9.0, 1.0]),  # 1 px tall: dropped
            ann(1, 7, [5.0, 5.0, 9.0, 9.0], iscrowd=True),  # crowd: skipped
            ann(2, 11, [3.0, 3.0, 9.0, 9.0]),  # image two has only a dropped category
        ],
    )
    val = write_json(
        tmp_path / "val.json",
        [
            {
                "id": 1,
                "path": "/dataset/subset-1/three.jpg",
                "width": 20,
                "height": 10,
                "file_name": "three.jpg",
            }
        ],
        [ann(1, 7, [4.0, 2.0, 6.0, 5.0])],
    )
    return root, train, val


def test_counts_and_a_known_image_round_trip(two_files: tuple[Path, Path, Path]) -> None:
    root, train, _ = two_files
    records = load_records([train], root, CLASS_MAP, CLASSES)
    assert [r.path.name for r in records] == ["one.jpg", "two.jpg"]
    one = records[0]
    assert (one.width, one.height) == (40, 30)
    assert one.labels.dtype == np.float32
    # COCO [x, y, w, h] became corners: (2, 3, 10, 6) -> (2, 3, 12, 9)
    # and (20, 10, 8, 8) -> (20, 10, 28, 18)
    assert one.labels.tolist() == [[0, 2, 3, 12, 9], [0, 20, 10, 28, 18]]


def test_class_map_decides_what_is_kept(two_files: tuple[Path, Path, Path]) -> None:
    root, train, _ = two_files
    pallet_only = load_records([train], root, {"pallet": "pallet"}, CLASSES)
    assert len(pallet_only[0].labels) == 1  # the stillage is no longer mapped
    everything = load_records([train], root, {**CLASS_MAP, "pallet_truck": "pallet"}, CLASSES)
    assert len(everything[0].labels) == 3  # pallet, stillage and the pallet truck


def test_an_image_with_no_kept_boxes_is_still_a_record(two_files: tuple[Path, Path, Path]) -> None:
    root, train, _ = two_files
    two = load_records([train], root, CLASS_MAP, CLASSES)[1]
    assert two.labels.shape == (0, 5)


def test_image_ids_may_repeat_across_files(two_files: tuple[Path, Path, Path]) -> None:
    root, train, val = two_files  # both files use image id 1 for different images
    records = load_records([train, val], root, CLASS_MAP, CLASSES)
    assert [r.path.name for r in records] == ["one.jpg", "two.jpg", "three.jpg"]
    assert records[2].labels.tolist() == [[0, 4, 2, 10, 7]]  # the val box, not the train one


def test_an_image_listed_in_two_files_is_an_error(two_files: tuple[Path, Path, Path]) -> None:
    root, train, _ = two_files
    with pytest.raises(ValueError, match="one.jpg"):
        load_records([train, train], root, CLASS_MAP, CLASSES)


def test_a_missing_image_file_is_reported(two_files: tuple[Path, Path, Path]) -> None:
    root, train, _ = two_files
    (root / "subset-2" / "a" / "two.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="two.jpg"):
        load_records([train], root, CLASS_MAP, CLASSES)
    assert len(load_records([train], root, CLASS_MAP, CLASSES, check_files=False)) == 2


def test_a_class_map_target_outside_classes_is_an_error(two_files: tuple[Path, Path, Path]) -> None:
    root, train, _ = two_files
    with pytest.raises(ValueError, match="not_a_class"):
        load_records([train], root, {"pallet": "not_a_class"}, CLASSES)


def test_dataset_returns_a_bgr_image_and_its_labels(tmp_path: Path) -> None:
    root = tmp_path / "loco"
    write_image(root, "subset-1/blue.jpg", 32, 16, bgr=(255, 0, 0))  # OpenCV writes BGR: pure blue
    file = write_json(
        tmp_path / "f.json",
        [
            {
                "id": 9,
                "path": "/dataset/subset-1/blue.jpg",
                "width": 32,
                "height": 16,
                "file_name": "blue.jpg",
            }
        ],
        [ann(9, 7, [1.0, 2.0, 10.0, 6.0])],
    )
    dataset = CocoDetection(load_records([file], root, CLASS_MAP, CLASSES))
    image, labels = dataset[0]
    assert len(dataset) == 1
    assert image.shape == (16, 32, 3) and image.dtype == np.uint8
    assert image[0, 0].tolist()[0] > 200  # channel 0 is blue, so this is BGR order
    assert labels.tolist() == [[0, 1, 2, 11, 8]]


def test_a_size_that_disagrees_with_the_annotation_is_an_error(tmp_path: Path) -> None:
    root = tmp_path / "loco"
    write_image(root, "subset-1/x.jpg", 32, 16)
    file = write_json(  # annotation claims 16 x 32: as if an EXIF rotation swapped the sides
        tmp_path / "f.json",
        [
            {
                "id": 1,
                "path": "/dataset/subset-1/x.jpg",
                "width": 16,
                "height": 32,
                "file_name": "x.jpg",
            }
        ],
        [],
    )
    dataset = CocoDetection(load_records([file], root, CLASS_MAP, CLASSES))
    with pytest.raises(ValueError, match="x.jpg"):
        dataset[0]


def test_from_config_picks_the_split(two_files: tuple[Path, Path, Path]) -> None:
    root, train, val = two_files
    data = {
        "root": str(root),
        "annotations": {"train": [str(train)], "val": [str(val)]},
        "class_map": CLASS_MAP,
    }
    assert len(CocoDetection.from_config(data, "train", CLASSES)) == 2
    assert len(CocoDetection.from_config(data, "val", CLASSES)) == 1


REPO = Path(__file__).resolve().parents[4]
LOCO_YAML = REPO / "configs" / "data" / "loco.yaml"


@pytest.mark.skipif(not (REPO / "data/loco/annotations").exists(), reason="LOCO data not present")
def test_real_loco_splits_resolve_to_images_on_disk() -> None:
    """Needs the (git-ignored) LOCO files. Counts follow the stage-1 class_map in loco.yaml."""
    data = yaml.safe_load(LOCO_YAML.read_text())
    data["root"] = str(REPO / data["root"])
    data["annotations"] = {k: [str(REPO / f) for f in v] for k, v in data["annotations"].items()}
    classes = yaml.safe_load((REPO / "contracts" / "classes.yaml").read_text())["stage1"]
    train = CocoDetection.from_config(data, "train", classes)  # raises if any image is missing
    val = CocoDetection.from_config(data, "val", classes)
    test = CocoDetection.from_config(data, "test", classes)
    assert (len(train), len(val), len(test)) == (2820, 1075, 1202)  # sub-sets 2, 3, 5; 1; 4
    assert sum(len(r.labels) for r in train.records) == 71_546
    assert sum(len(r.labels) for r in val.records) == 15_251
    assert sum(len(r.labels) for r in test.records) == 39_051
    for dataset in (
        train,
        val,
        test,
    ):  # a spot check of real decoding, since a full pass takes 15 s
        for i in range(0, len(dataset), 400):
            dataset[i]
