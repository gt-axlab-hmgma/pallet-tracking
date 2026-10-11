from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from palletvision_infer.preprocess import letterbox
from palletvision_model.config import SIZES
from palletvision_model.loss import DetectionLoss
from palletvision_model.yolo26 import YOLO26
from palletvision_train.coco import CocoDetection, load_records
from palletvision_train.data import Batch, LetterboxedDataset, collate


def sample(n_boxes: int, size: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """An already letterboxed canvas and ``n_boxes`` labels with recognisable numbers."""
    canvas = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    labels = np.zeros((n_boxes, 5), dtype=np.float32)
    for i in range(n_boxes):
        labels[i] = [i % 2, 10 + i, 11 + i, 30 + i, 31 + i]
    return canvas, labels


def test_padding_and_mask_for_images_with_zero_one_and_many_boxes() -> None:
    batch = collate([sample(0), sample(1, seed=1), sample(3, seed=2)])
    assert batch.images.shape == (3, 3, 64, 64) and batch.images.dtype == torch.float32
    assert batch.gt_labels.shape == (3, 3, 1) and batch.gt_labels.dtype == torch.long
    assert batch.gt_boxes.shape == (3, 3, 4) and batch.mask_gt.shape == (3, 3, 1)
    assert batch.mask_gt[..., 0].tolist() == [
        [False, False, False],
        [True, False, False],
        [True, True, True],
    ]
    # real entries keep their order and values; padded ones are zero
    assert batch.gt_boxes[2, 1].tolist() == [11, 12, 31, 32]
    assert batch.gt_labels[2, :, 0].tolist() == [0, 1, 0]
    assert batch.gt_boxes[0].abs().sum() == 0 and batch.gt_boxes[1, 1:].abs().sum() == 0
    assert batch.gt_labels[1, 1:].abs().sum() == 0


def test_a_batch_with_no_objects_at_all_has_no_object_columns() -> None:
    batch = collate([sample(0), sample(0, seed=1)])
    assert batch.gt_labels.shape == (2, 0, 1) and batch.gt_boxes.shape == (2, 0, 4)
    assert batch.mask_gt.shape == (2, 0, 1)


def test_images_are_exactly_the_inference_input_in_rgb() -> None:
    raw = np.random.default_rng(3).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    dataset = LetterboxedDataset([(raw, np.zeros((0, 5), np.float32))], size=64)  # type: ignore[arg-type]
    batch = collate([dataset[0]])
    expected, _ = letterbox(raw, size=64)
    assert torch.equal(batch.images, torch.from_numpy(expected))
    blue = np.zeros((8, 8, 3), dtype=np.uint8)
    blue[..., 0] = 255  # blue in OpenCV's BGR
    assert collate([(blue, np.zeros((0, 5), np.float32))]).images[0, 2].min() == 1.0


def test_to_moves_every_tensor() -> None:
    moved = collate([sample(2)]).to("cpu")
    assert isinstance(moved, Batch) and moved.images.device.type == "cpu"


@pytest.fixture
def tiny_coco(tmp_path: Path) -> CocoDetection:
    """Three small images of different shapes, with 0, 1 and 2 labelled boxes."""
    import json

    root = tmp_path / "loco"
    sizes = {"a": (40, 80), "b": (60, 60), "c": (90, 45)}  # (height, width)
    images, annotations = [], []
    for i, (name, (h, w)) in enumerate(sizes.items(), start=1):
        path = root / "subset-1" / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((h, w, 3), 80 * i, dtype=np.uint8))
        images.append({"id": i, "path": f"/dataset/subset-1/{name}.jpg", "width": w, "height": h})
    annotations += [
        {"image_id": 2, "category_id": 7, "bbox": [10.0, 10.0, 20.0, 20.0]},
        {"image_id": 3, "category_id": 7, "bbox": [5.0, 5.0, 10.0, 20.0]},
        {"image_id": 3, "category_id": 10, "bbox": [20.0, 40.0, 20.0, 30.0]},
    ]
    file = tmp_path / "f.json"
    categories = [{"id": 7, "name": "pallet"}, {"id": 10, "name": "stillage"}]
    file.write_text(
        json.dumps({"images": images, "categories": categories, "annotations": annotations})
    )
    class_map = {"pallet": "pallet", "stillage": "pallet"}
    return CocoDetection(load_records([file], root, class_map, ["pallet"]))


@pytest.mark.parametrize("workers", [0, 2])
def test_a_dataloader_builds_batches_from_the_reader(
    tiny_coco: CocoDetection, workers: int
) -> None:
    loader = DataLoader(
        LetterboxedDataset(tiny_coco, size=64),
        batch_size=3,
        collate_fn=collate,
        num_workers=workers,
    )
    (batch,) = list(loader)
    assert batch.images.shape == (3, 3, 64, 64)
    assert batch.mask_gt[..., 0].sum(dim=1).tolist() == [0, 1, 2]  # 0, 1 and 2 boxes
    # image b (60 x 60) letterboxes at scale 64 / 60 with no padding: box (10,10,30,30) moves
    assert torch.allclose(batch.gt_boxes[1, 0], torch.tensor([10, 10, 30, 30]) * (64 / 60))
    assert (batch.gt_boxes[batch.mask_gt[..., 0]] >= 0).all()
    assert (batch.gt_boxes <= 64).all()


def test_a_collated_batch_goes_straight_through_the_model_and_loss(
    tiny_coco: CocoDetection,
) -> None:
    torch.manual_seed(0)
    (batch,) = list(
        DataLoader(LetterboxedDataset(tiny_coco, size=64), batch_size=3, collate_fn=collate)
    )
    model = YOLO26(SIZES["n"]).train()
    total, parts = DetectionLoss()(
        model.raw(batch.images), (64, 64), batch.gt_labels, batch.gt_boxes, batch.mask_gt, 0.8
    )
    assert torch.isfinite(total)
    first = next(model.backbone.parameters())
    (grad,) = torch.autograd.grad(total, [first], allow_unused=True)
    assert grad is not None and grad.abs().sum() > 0
    assert parts["box_one2one"].item() >= 0


REPO = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not (REPO / "data/loco/annotations").exists(), reason="LOCO data not present")
def test_real_loco_batches_have_the_right_shapes_and_counts() -> None:
    data = yaml.safe_load((REPO / "configs/data/loco.yaml").read_text())
    data["root"] = str(REPO / data["root"])
    data["annotations"] = {k: [str(REPO / f) for f in v] for k, v in data["annotations"].items()}
    classes = yaml.safe_load((REPO / "contracts/classes.yaml").read_text())["stage1"]
    reader = CocoDetection.from_config(data, "val", classes)
    loader = DataLoader(LetterboxedDataset(reader, 640), batch_size=4, collate_fn=collate)
    batch = next(iter(loader))
    assert batch.images.shape == (4, 3, 640, 640)
    expected = [len(reader.records[i].labels) for i in range(4)]
    assert batch.mask_gt[..., 0].sum(dim=1).tolist() == expected
    real = batch.gt_boxes[batch.mask_gt[..., 0]]
    assert (real >= 0).all() and (real <= 640).all() and (real[:, 2:] > real[:, :2]).all()
