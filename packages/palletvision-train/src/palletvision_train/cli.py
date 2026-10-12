from collections.abc import Callable
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset

from palletvision_model.config import ModelConfig
from palletvision_model.device import resolve_device
from palletvision_model.yolo26 import YOLO26
from palletvision_train.augment import AugmentConfig, AugmentedDataset
from palletvision_train.coco import CocoDetection, ImageRecord, loco_reader
from palletvision_train.data import LetterboxedDataset, Sample
from palletvision_train.evaluate import evaluate
from palletvision_train.sordi import MixedPartners, MixedSampler, load_sordi_records, scene_folders
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import TrainResult, train

REPO = Path(__file__).resolve().parents[4]
CONFIG_DIR = str(REPO / "configs")
LOCO_DATA = {"loco", "loco_sordi"}  # datasets whose real part (and validation) is LOCO


def build_model(cfg: DictConfig, num_classes: int) -> YOLO26:
    m = cfg.model
    if m.num_classes != num_classes:
        raise SystemExit(f"model.num_classes is {m.num_classes} but the stage has {num_classes}")
    model_cfg = ModelConfig(
        m.depth_multiple, m.width_multiple, num_classes=num_classes, reg_max=m.reg_max
    )
    return YOLO26(model_cfg, input_size=m.input_size)


def build_dataset(cfg: DictConfig, classes: list[str]) -> Dataset[Sample]:
    """The training set, 640-square. ``data.limit`` keeps only the first N real images.

    Without augmentation every image is letterboxed, as at inference. With it (``data.augment``,
    on for LOCO) each access returns a freshly augmented version. ``data=loco_sordi`` puts the
    real images first and the synthetic pool after them, one dataset for ``MixedSampler``.
    """
    size = cfg.model.input_size
    if cfg.data.name == "synthetic":
        base: Dataset[Sample] = SyntheticBoxes(cfg.data.num_samples, cfg.data.size, cfg.seed)
    elif cfg.data.name in LOCO_DATA:
        data = OmegaConf.to_container(cfg.data, resolve=True)
        assert isinstance(data, dict)
        base = CocoDetection.from_config({str(k): v for k, v in data.items()}, "train", classes)
    else:
        raise SystemExit(f"pv-train: dataset '{cfg.data.name}' has no loader yet")
    limit = cfg.data.get("limit")
    if limit is not None:
        base = Subset(base, range(min(int(limit), len(base))))  # type: ignore[arg-type]
    partners = None
    if cfg.data.name == "loco_sordi":  # real images first, then the synthetic pool
        pool = build_sordi(cfg, classes)
        partners = MixedPartners(len(base), len(pool), cfg.data.sordi.share)  # type: ignore[arg-type]
        base = ConcatDataset([base, pool])
    augment = cfg.data.get("augment")
    if augment is not None and augment.get("enabled", True):
        config = AugmentConfig(**{k: v for k, v in augment.items() if k != "enabled"})
        return AugmentedDataset(base, size, config, partners)
    return LetterboxedDataset(base, size)


def build_sordi(cfg: DictConfig, classes: list[str]) -> Dataset[Sample]:
    """The synthetic pool (original pixels): every scene folder in ``data.sordi.roots``."""
    sordi = cfg.data.sordi
    mapping = OmegaConf.to_container(sordi.class_map, resolve=True)
    assert isinstance(mapping, dict)
    class_map = {str(k): str(v) for k, v in mapping.items()}
    records: list[ImageRecord] = []
    for root in sordi.roots:
        for folder in scene_folders(REPO / root):
            records += load_sordi_records(folder, class_map, classes)
    pool: Dataset[Sample] = CocoDetection(records)
    if sordi.get("limit") is not None:
        pool = Subset(pool, range(min(int(sordi.limit), len(pool))))  # type: ignore[arg-type]
    return pool


def build_sampler(cfg: DictConfig, dataset: Dataset[Sample]) -> Sampler[int] | None:
    """The mixed real/synthetic epoch for ``data=loco_sordi``; every other dataset shuffles."""
    if cfg.data.name != "loco_sordi":
        return None
    assert isinstance(dataset, (LetterboxedDataset, AugmentedDataset))
    assert isinstance(dataset.base, ConcatDataset)
    real, synthetic = dataset.base.datasets
    return MixedSampler(len(real), len(synthetic), cfg.data.sordi.share)  # type: ignore[arg-type]


def build_epoch_hook(cfg: DictConfig, dataset: Dataset[Sample]) -> Callable[[int], None] | None:
    """Switch mosaic off for the last ``augment.close_mosaic`` epochs (report: the last 8)."""
    if not isinstance(dataset, AugmentedDataset):
        return None
    last_mosaic_epoch = cfg.train.epochs - dataset.config.close_mosaic

    def hook(epoch: int) -> None:
        dataset.set_mosaic(epoch < last_mosaic_epoch)

    return hook


def resume_path(cfg: DictConfig) -> Path | None:
    """``train.resume``: null starts fresh, a path resumes from it, ``auto`` resumes from
    ``<output_dir>/last.pt`` when it exists and starts fresh when it does not."""
    setting = cfg.train.get("resume")
    if setting is None:
        return None
    if setting == "auto":
        found = Path(cfg.train.output_dir) / "last.pt"
        return found if found.exists() else None
    return Path(setting)


def build_metric(cfg: DictConfig) -> Callable[[YOLO26], float] | None:
    """``train.val_every`` > 0 scores the LOCO validation split with mAP50-95 every N epochs."""
    if not cfg.train.get("val_every"):
        return None
    if cfg.data.name not in LOCO_DATA:
        raise SystemExit("train.val_every needs data=loco: only LOCO has a validation split")
    reader = loco_reader("val")
    limit = cfg.data.get("val_limit")
    if limit is not None:
        reader = CocoDetection(reader.records[: int(limit)])
    device = resolve_device(cfg.device)

    def metric(model: YOLO26) -> float:
        result = evaluate(
            model, reader, size=cfg.model.input_size, device=device,
            batch_size=cfg.train.batch_size, num_workers=cfg.data.get("num_workers", 0),
        )  # fmt: skip
        return result.map50_95

    return metric


def train_from_config(cfg: DictConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    classes = yaml.safe_load((REPO / "contracts" / "classes.yaml").read_text())["stage1"]
    model = build_model(cfg, len(classes))
    dataset = build_dataset(cfg, classes)
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    return train(
        model, dataset, device=cfg.device, epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        optimizer_name=cfg.train.optimizer, momentum=cfg.train.momentum,
        muon=cfg.train.muon_weight, sgd=cfg.train.sgd_weight,
        nominal_batch=cfg.train.nominal_batch, warmup_epochs=cfg.train.warmup_epochs,
        warmup_bias_lr=0.0 if cfg.train.optimizer == "adamw" else cfg.train.warmup_bias_lr,
        lr_final=cfg.train.lr_final, lr_schedule=cfg.train.lr_schedule,
        amp=cfg.train.amp, max_steps=cfg.train.max_steps, output_dir=cfg.train.output_dir,
        num_workers=cfg.data.get("num_workers", 0), resume=resume_path(cfg),
        metric_fn=build_metric(cfg), metric_every=cfg.train.get("val_every") or 1,
        config={str(k): v for k, v in config.items()},
        sampler=build_sampler(cfg, dataset), on_epoch_start=build_epoch_hook(cfg, dataset),
    )  # fmt: skip


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    result = train_from_config(cfg)
    last = result.losses[-1] if result.losses else float("nan")
    print(f"trained {len(result.losses)} steps, last loss {last:.4f} -> {result.checkpoint}")
