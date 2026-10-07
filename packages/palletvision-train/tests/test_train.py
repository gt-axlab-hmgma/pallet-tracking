from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from palletvision_model.tiny import TinyDetector
from palletvision_train.cli import train_from_config
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import train

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_synthetic_sample_shapes() -> None:
    image, box = SyntheticBoxes(n=4, size=32)[0]
    assert image.shape == (3, 32, 32)
    assert box.shape == (4,)
    assert 0 <= box[0] < box[2] <= 32


def test_train_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    model = TinyDetector(input_size=32)
    result = train(
        model, SyntheticBoxes(n=8, size=32), device="cpu", epochs=2, batch_size=4,
        lr=1e-2, amp=False, max_steps=None, output_dir=tmp_path,
    )  # fmt: skip
    assert len(result.losses) == 4  # 2 epochs x 2 batches
    assert all(torch.isfinite(torch.tensor(result.losses)))
    assert result.checkpoint.exists()
    state = torch.load(result.checkpoint, map_location="cpu")
    assert "model" in state


def test_max_steps_caps_training(tmp_path: Path) -> None:
    result = train(
        TinyDetector(input_size=32), SyntheticBoxes(n=8, size=32), device="cpu", epochs=5,
        batch_size=2, lr=1e-2, amp=False, max_steps=3, output_dir=tmp_path,
    )  # fmt: skip
    assert len(result.losses) == 3


def test_config_driven_run(tmp_path: Path) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=synthetic", "data.size=32", "data.num_samples=8", "model.input_size=32",
                "device=cpu", "train.amp=false", "train.max_steps=2", "train.batch_size=4",
                f"train.output_dir={tmp_path}",
            ],
        )  # fmt: skip
    result = train_from_config(cfg)
    assert result.checkpoint.exists()


def test_loco_not_wired_yet(tmp_path: Path) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="config", overrides=["device=cpu"])
    with pytest.raises(SystemExit):
        train_from_config(cfg)
