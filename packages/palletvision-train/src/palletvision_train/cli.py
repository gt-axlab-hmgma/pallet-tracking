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
from palletvision_train.distributed import current, init_distributed, shutdown
from palletvision_train.evaluate import evaluate
from palletvision_train.pretrained import load_pretrained
from palletvision_train.sordi import (
    MixedPartners,
    MixedSampler,
    UniformSampler,
    sordi_records,
    subsample,
)
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
        if not cfg.data.get("real_train", True):  # synthetic only: no real image is trained on
            base = Subset(base, range(0))
        pool = build_sordi(cfg, classes)
        share = cfg.data.sordi.get("share")  # null: one pool, mosaic partners from all of it
        if share is not None:
            partners = MixedPartners(len(base), len(pool), share)  # type: ignore[arg-type]
        base = ConcatDataset([base, pool])
    augment = cfg.data.get("augment")
    if augment is not None and augment.get("enabled", True):
        config = AugmentConfig(**{k: v for k, v in augment.items() if k != "enabled"})
        return AugmentedDataset(base, size, config, partners)
    return LetterboxedDataset(base, size)


def sordi_split(cfg: DictConfig, classes: list[str], split: str) -> list[ImageRecord]:
    """The ``split`` images of every scene in ``data.sordi.roots``, split 70/15/15 by
    ``data.sordi.fractions``."""
    sordi = cfg.data.sordi
    mapping = OmegaConf.to_container(sordi.class_map, resolve=True)
    assert isinstance(mapping, dict)
    class_map = {str(k): str(v) for k, v in mapping.items()}
    return sordi_records(
        [str(r) for r in sordi.roots], class_map, classes, split,
        [float(f) for f in sordi.fractions], REPO,
    )  # fmt: skip


def build_sordi(cfg: DictConfig, classes: list[str]) -> Dataset[Sample]:
    """The synthetic training pool (original pixels); ``data.sordi.limit`` keeps the first N."""
    records = sordi_split(cfg, classes, "train")
    if cfg.data.sordi.get("subsample") is not None:  # a fixed random draw from every scene
        records = subsample(records, int(cfg.data.sordi.subsample))
    pool: Dataset[Sample] = CocoDetection(records)
    limit = cfg.data.sordi.get("limit")
    if limit is not None:
        pool = Subset(pool, range(min(int(limit), len(pool))))  # type: ignore[arg-type]
    return pool


def build_sampler(cfg: DictConfig, dataset: Dataset[Sample]) -> Sampler[int] | None:
    """The epoch of ``data=loco_sordi``; every other dataset shuffles.

    ``data.sordi.share`` null (the default): real and synthetic images are one pool and each epoch
    is a uniform draw of ``data.sordi.epoch_images``. A number: every real image once plus enough
    synthetic ones to make that share of the epoch.
    """
    if cfg.data.name != "loco_sordi":
        return None
    assert isinstance(dataset, (LetterboxedDataset, AugmentedDataset))
    assert isinstance(dataset.base, ConcatDataset)
    real, synthetic = dataset.base.datasets
    info = current()
    seed = cfg.seed if info.distributed else None
    if cfg.data.sordi.get("share") is None:
        return UniformSampler(
            len(dataset.base), cfg.data.sordi.get("epoch_images"),
            seed=seed, rank=info.rank, world_size=info.world_size,
        )  # fmt: skip
    return MixedSampler(
        len(real), len(synthetic), cfg.data.sordi.share,  # type: ignore[arg-type]
        seed=seed, rank=info.rank, world_size=info.world_size,
    )  # fmt: skip


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


def classes_of() -> list[str]:
    return list(yaml.safe_load((REPO / "contracts" / "classes.yaml").read_text())["stage1"])


def build_metric(cfg: DictConfig) -> Callable[[YOLO26], float] | None:
    """``train.val_every`` > 0 scores the real LOCO validation split with mAP50-95 every N epochs.

    This is the number that picks ``best.pt``: real images only, since the real world is what
    the model is for. ``build_extra_metrics`` scores the synthetic validation split alongside.
    """
    if not cfg.train.get("val_every"):
        return None
    if cfg.data.name not in LOCO_DATA:
        raise SystemExit("train.val_every needs data=loco: only LOCO has a validation split")
    if cfg.train.get("select_on", "loco") == "sordi":  # best.pt picked on synthetic images
        records = sordi_split(cfg, classes_of(), "val")
        return scorer(cfg, CocoDetection(subsample(records, cfg.data.sordi.get("val_images"))))
    reader = loco_reader("val")
    limit = cfg.data.get("val_limit")
    if limit is not None:
        reader = CocoDetection(reader.records[: int(limit)])
    return scorer(cfg, reader)


def build_extra_metrics(
    cfg: DictConfig, classes: list[str]
) -> Callable[[YOLO26], dict[str, float]] | None:
    """For ``data=loco_sordi``: the synthetic validation split (a fixed ``sordi.val_images``
    draw), scored with the same metric and logged next to the real one."""
    if not cfg.train.get("val_every") or cfg.data.name != "loco_sordi":
        return None
    if cfg.train.get("select_on", "loco") == "sordi":  # the other way round: log the real score
        real = scorer(cfg, loco_reader("val"))
        return lambda model: {"loco_val_map50_95": real(model)}
    records = sordi_split(cfg, classes, "val")
    score = scorer(cfg, CocoDetection(subsample(records, cfg.data.sordi.get("val_images"))))
    return lambda model: {"sordi_val_map50_95": score(model)}


def scorer(
    cfg: DictConfig, reader: CocoDetection, categories: int = 1
) -> Callable[[YOLO26], float]:
    device = resolve_device(cfg.device)

    def metric(model: YOLO26) -> float:
        result = evaluate(
            model, reader, size=cfg.model.input_size, device=device,
            batch_size=cfg.train.batch_size, num_workers=cfg.data.get("num_workers", 0),
            categories=categories,
        )  # fmt: skip
        return result.map50_95

    return metric


def train_from_config(cfg: DictConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    classes = yaml.safe_load((REPO / "contracts" / "classes.yaml").read_text())["stage1"]
    model = build_model(cfg, len(classes))
    pretrained: set[str] | None = None
    if cfg.train.get("init_from"):
        copied, skipped = load_pretrained(model, REPO / cfg.train.init_from)
        pretrained = set(copied)
        print(f"started from {cfg.train.init_from}: {len(copied)} copied, {len(skipped)} fresh")
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
        amp=cfg.train.amp, amp_dtype=cfg.train.get("amp_dtype", "auto"),
        max_steps=cfg.train.max_steps, output_dir=cfg.train.output_dir,
        num_workers=cfg.data.get("num_workers", 0), resume=resume_path(cfg),
        metric_fn=build_metric(cfg), metric_every=cfg.train.get("val_every") or 1,
        extra_metrics_fn=build_extra_metrics(cfg, classes),
        pretrained=pretrained, pretrained_lr_mult=cfg.train.get("pretrained_lr_mult", 1.0),
        alpha_schedule=cfg.train.get("alpha_schedule", "epoch"),
        alpha_init=cfg.train.get("alpha_init", 0.8), alpha_final=cfg.train.get("alpha_final", 0.1),
        ema=cfg.train.get("ema", False), ema_decay=cfg.train.get("ema_decay", 0.9999),
        ema_tau=cfg.train.get("ema_tau", 2000.0),
        freeze=cfg.train.get("freeze", 0), patience=cfg.train.get("patience", 0),
        config={str(k): v for k, v in config.items()},
        sampler=build_sampler(cfg, dataset), on_epoch_start=build_epoch_hook(cfg, dataset),
        seed=cfg.seed,
    )  # fmt: skip


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    info, device = init_distributed(cfg.device)  # joins the group when started by torchrun
    cfg.device = device
    try:
        if info.is_main:
            print(OmegaConf.to_yaml(cfg))
        result = train_from_config(cfg)
        last = result.losses[-1] if result.losses else float("nan")
        if info.is_main:
            print(
                f"trained {len(result.losses)} steps, last loss {last:.4f} -> {result.checkpoint}"
            )
    finally:
        shutdown()


if __name__ == "__main__":  # python -m palletvision_train.cli, and torchrun -m
    main()
