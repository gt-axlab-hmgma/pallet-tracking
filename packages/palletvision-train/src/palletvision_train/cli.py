from collections.abc import Callable
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset

from palletvision_model.config import ModelConfig
from palletvision_model.device import resolve_device
from palletvision_model.yolo26 import YOLO26
from palletvision_train.coco import CocoDetection, loco_reader
from palletvision_train.data import LetterboxedDataset, Sample
from palletvision_train.evaluate import evaluate
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import TrainResult, train

REPO = Path(__file__).resolve().parents[4]
CONFIG_DIR = str(REPO / "configs")


def build_model(cfg: DictConfig, num_classes: int) -> YOLO26:
    m = cfg.model
    if m.num_classes != num_classes:
        raise SystemExit(f"model.num_classes is {m.num_classes} but the stage has {num_classes}")
    model_cfg = ModelConfig(
        m.depth_multiple, m.width_multiple, num_classes=num_classes, reg_max=m.reg_max
    )
    return YOLO26(model_cfg, input_size=m.input_size)


def build_dataset(cfg: DictConfig, classes: list[str]) -> Dataset[Sample]:
    """The letterboxed training set. ``data.limit`` keeps only the first N images (smoke runs)."""
    size = cfg.model.input_size
    if cfg.data.name == "synthetic":
        base: Dataset[Sample] = SyntheticBoxes(cfg.data.num_samples, cfg.data.size, cfg.seed)
    elif cfg.data.name == "loco":
        data = OmegaConf.to_container(cfg.data, resolve=True)
        assert isinstance(data, dict)
        base = CocoDetection.from_config({str(k): v for k, v in data.items()}, "train", classes)
    else:
        raise SystemExit(f"pv-train: dataset '{cfg.data.name}' has no loader yet")
    dataset: Dataset[Sample] = LetterboxedDataset(base, size)
    limit = cfg.data.get("limit")
    if limit is not None:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))  # type: ignore[arg-type]
    return dataset


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
    if cfg.data.name != "loco":
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
    )  # fmt: skip


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    result = train_from_config(cfg)
    last = result.losses[-1] if result.losses else float("nan")
    print(f"trained {len(result.losses)} steps, last loss {last:.4f} -> {result.checkpoint}")
