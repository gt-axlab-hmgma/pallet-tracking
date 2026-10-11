from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset

from palletvision_model.config import ModelConfig
from palletvision_model.yolo26 import YOLO26
from palletvision_train.coco import CocoDetection
from palletvision_train.data import LetterboxedDataset, Sample
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


def train_from_config(cfg: DictConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    classes = yaml.safe_load((REPO / "contracts" / "classes.yaml").read_text())["stage1"]
    model = build_model(cfg, len(classes))
    dataset = build_dataset(cfg, classes)
    return train(
        model, dataset, device=cfg.device, epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        amp=cfg.train.amp, max_steps=cfg.train.max_steps, output_dir=cfg.train.output_dir,
        num_workers=cfg.data.get("num_workers", 0),
    )  # fmt: skip


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    result = train_from_config(cfg)
    last = result.losses[-1] if result.losses else float("nan")
    print(f"trained {len(result.losses)} steps, last loss {last:.4f} -> {result.checkpoint}")
