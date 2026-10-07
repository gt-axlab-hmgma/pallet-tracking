from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from palletvision_model.tiny import TinyDetector
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import TrainResult, train

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs")


def train_from_config(cfg: DictConfig) -> TrainResult:
    if cfg.data.name != "synthetic":
        raise SystemExit(
            f"pv-train: dataset '{cfg.data.name}' has no loader yet; use data=synthetic"
        )
    # TinyDetector is a placeholder until the real detector is built in palletvision-model.
    model = TinyDetector(input_size=cfg.model.input_size)
    dataset = SyntheticBoxes(n=cfg.data.num_samples, size=cfg.data.size, seed=cfg.seed)
    return train(
        model, dataset, device=cfg.device, epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size, lr=cfg.train.lr, amp=cfg.train.amp,
        max_steps=cfg.train.max_steps, output_dir=cfg.train.output_dir,
    )  # fmt: skip


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    result = train_from_config(cfg)
    print(f"trained {len(result.losses)} steps -> {result.checkpoint}")
