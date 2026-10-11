from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from palletvision_model.config import SIZES
from palletvision_model.loss import progressive_alpha
from palletvision_model.yolo26 import YOLO26
from palletvision_train.cli import train_from_config
from palletvision_train.data import LetterboxedDataset
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import train

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs"


def small_model(seed: int = 0) -> YOLO26:
    torch.manual_seed(seed)
    return YOLO26(SIZES["n"], input_size=64)


def synthetic(n: int = 8) -> LetterboxedDataset:
    return LetterboxedDataset(SyntheticBoxes(n=n, size=64), size=64)


def test_synthetic_sample_matches_the_readers_format() -> None:
    image, labels = SyntheticBoxes(n=4, size=32)[0]
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8
    assert labels.shape == (1, 5) and labels.dtype == np.float32
    cls, x1, y1, x2, y2 = labels[0]
    assert cls == 0 and 0 <= x1 < x2 <= 32 and 0 <= y1 < y2 <= 32


def test_train_runs_and_saves_a_checkpoint_the_model_can_reload(tmp_path: Path) -> None:
    model = small_model()
    result = train(
        model,
        synthetic(),
        epochs=2,
        lr=1e-3,
        output_dir=tmp_path,
        device="cpu",
        amp=False,
        batch_size=4,
    )
    assert len(result.losses) == 4  # 2 epochs x 2 batches
    assert all(np.isfinite(result.losses))
    assert {"box_one2one", "cls_one2many", "l1_one2one", "total_one2many"} <= set(result.parts[0])
    state = torch.load(result.checkpoint, map_location="cpu")
    assert state["steps"] == 4
    fresh = small_model(seed=1)
    fresh.load_state_dict(state["model"])  # same architecture, so the weights fit


def test_max_steps_caps_training(tmp_path: Path) -> None:
    result = train(
        small_model(),
        synthetic(),
        epochs=5,
        lr=1e-3,
        max_steps=3,
        output_dir=tmp_path,
        device="cpu",
        amp=False,
        batch_size=4,
    )
    assert len(result.losses) == 3


def test_alpha_follows_the_progressive_schedule_one_value_per_epoch(tmp_path: Path) -> None:
    result = train(
        small_model(),
        synthetic(8),
        epochs=3,
        lr=1e-3,
        output_dir=tmp_path,
        device="cpu",
        amp=False,
        batch_size=4,
    )
    expected = [progressive_alpha(e, 3) for e in (0, 0, 1, 1, 2, 2)]  # 2 batches per epoch
    assert result.alphas == pytest.approx(expected)
    assert result.alphas[0] == pytest.approx(0.8) and result.alphas[-1] == pytest.approx(0.1)


def test_the_weights_actually_change(tmp_path: Path) -> None:
    model = small_model()
    before = next(model.backbone.parameters()).detach().clone()
    train(
        model,
        synthetic(),
        epochs=1,
        lr=1e-2,
        output_dir=tmp_path,
        device="cpu",
        amp=False,
        batch_size=4,
    )
    assert not torch.equal(before, next(model.backbone.parameters()).detach().cpu())


def test_the_loss_falls_when_training_on_a_few_images(tmp_path: Path) -> None:
    result = train(
        small_model(), synthetic(4), epochs=40, lr=2e-3, output_dir=tmp_path,
        device="cpu", amp=False, batch_size=4,
    )  # fmt: skip
    first, last = np.mean(result.losses[:3]), np.mean(result.losses[-3:])
    assert last < 0.8 * first


def config(*overrides: str, tmp_path: Path) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(
            config_name="config",
            overrides=["device=cpu", "train.amp=false", f"train.output_dir={tmp_path}", *overrides],
        )


def test_config_driven_synthetic_run(tmp_path: Path) -> None:
    cfg = config(
        "data=synthetic", "data.size=64", "data.num_samples=8", "model.input_size=64",
        "train.max_steps=2", "train.batch_size=4", tmp_path=tmp_path,
    )  # fmt: skip
    result = train_from_config(cfg)
    assert len(result.losses) == 2 and result.checkpoint.exists()


def test_the_limit_option_keeps_only_the_first_images(tmp_path: Path) -> None:
    cfg = config(
        "data=synthetic", "data.size=64", "data.num_samples=40", "data.limit=6",
        "model.input_size=64", "train.epochs=1", "train.batch_size=2", tmp_path=tmp_path,
    )  # fmt: skip
    assert len(train_from_config(cfg).losses) == 3


def test_a_model_with_the_wrong_class_count_is_rejected(tmp_path: Path) -> None:
    cfg = config("data=synthetic", "model.num_classes=3", tmp_path=tmp_path)
    with pytest.raises(SystemExit, match="num_classes"):
        train_from_config(cfg)


@pytest.mark.skipif(not (REPO / "data/loco/annotations").exists(), reason="LOCO data not present")
def test_config_driven_run_on_real_loco_images(tmp_path: Path) -> None:
    """Needs the git-ignored LOCO files. A few steps on a few real images."""
    cfg = config(
        "data=loco", "data.limit=8", "data.num_workers=0", "model.input_size=128",
        "train.epochs=2", "train.batch_size=4", tmp_path=tmp_path,
    )  # fmt: skip
    result = train_from_config(cfg)
    assert len(result.losses) == 4 and all(np.isfinite(result.losses))
    classes = yaml.safe_load((REPO / "contracts/classes.yaml").read_text())["stage1"]
    assert len(classes) == 1
