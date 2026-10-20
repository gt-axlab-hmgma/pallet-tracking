import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26
from palletvision_train.checkpoints.model import load_model
from palletvision_train.cli.eval import main
from palletvision_train.cli.train import build_metric, train_from_config
from palletvision_train.datasets.data import LetterboxedDataset
from palletvision_train.datasets.synthetic import SyntheticBoxes
from palletvision_train.training.trainer import train

REPO = Path(__file__).resolve().parents[4]
HAS_LOCO = (REPO / "data/loco/annotations").exists()


def images() -> LetterboxedDataset:
    return LetterboxedDataset(SyntheticBoxes(n=8, size=64), size=64)


def small_model() -> YOLO26:
    torch.manual_seed(0)
    return YOLO26(SIZES["n"], input_size=64)


def test_the_metric_runs_every_n_epochs_and_always_after_the_last(tmp_path: Path) -> None:
    seen: list[int] = []

    def metric(model: YOLO26) -> float:
        seen.append(len(seen))
        return 0.1 * len(seen)

    train(
        small_model(), images(), epochs=5, output_dir=tmp_path, device="cpu", batch_size=4, lr=1e-3,
        metric_fn=metric, metric_every=2,
    )  # fmt: skip
    assert len(seen) == 3  # after epochs 2, 4 and 5
    log = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    scored = [(e["epoch"] + 1, e["metric"]) for e in log if e["type"] == "epoch"]
    assert [epoch for epoch, m in scored if m is not None] == [2, 4, 5]
    assert torch.load(tmp_path / "best.pt")["epoch"] == 5  # scores only rose, so the last is best


def test_extra_scores_are_logged_beside_the_metric_but_do_not_pick_best(tmp_path: Path) -> None:
    def extras(model: YOLO26) -> dict[str, float]:
        return {"sordi_val_map50_95": 0.9}  # high, yet best.pt follows the main metric

    scores = iter([0.5, 0.2])  # the second epoch is worse
    train(
        small_model(), images(), epochs=2, output_dir=tmp_path, device="cpu", batch_size=4,
        lr=1e-3, metric_fn=lambda model: next(scores), extra_metrics_fn=extras,
    )  # fmt: skip
    log = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    epochs = [e for e in log if e["type"] == "epoch"]
    assert [e["sordi_val_map50_95"] for e in epochs] == [0.9, 0.9]
    assert torch.load(tmp_path / "best.pt")["epoch"] == 1


def test_the_model_is_left_in_training_mode_after_a_score(tmp_path: Path) -> None:
    model = small_model()
    modes: list[bool] = []

    def metric(m: YOLO26) -> float:
        modes.append(m.training)  # scoring happens in eval mode
        return 0.5

    train(
        model,
        images(),
        epochs=2,
        output_dir=tmp_path,
        device="cpu",
        batch_size=4,
        lr=1e-3,
        metric_fn=metric,
    )
    assert modes == [False, False] and model.training  # and training resumes in train mode


def test_a_checkpoint_rebuilds_the_model_it_was_trained_as(tmp_path: Path) -> None:
    config = {
        "model": {
            "depth_multiple": 0.5,
            "width_multiple": 0.25,
            "num_classes": 1,
            "reg_max": 1,
            "input_size": 64,
        }
    }
    trained = small_model()
    train(
        trained,
        images(),
        epochs=1,
        output_dir=tmp_path,
        device="cpu",
        batch_size=4,
        lr=1e-3,
        config=config,
    )
    model, where, size = load_model(tmp_path / "last.pt")
    assert size == 64 and where == {"epoch": 1, "steps": 2}
    for name, tensor in trained.cpu().state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name


def test_a_checkpoint_without_a_config_falls_back_to_the_default_model(tmp_path: Path) -> None:
    train(
        small_model(), images(), epochs=1, output_dir=tmp_path, device="cpu", batch_size=4, lr=1e-3
    )
    _, _, size = load_model(tmp_path / "last.pt")
    assert size == 640  # the default n model at 640


def test_an_older_checkpoint_with_only_weights_still_loads(tmp_path: Path) -> None:
    model = small_model()
    torch.save({"model": model.state_dict(), "steps": 7}, tmp_path / "old.pt")  # the pre-R5 format
    loaded, where, size = load_model(tmp_path / "old.pt")
    assert where == {"epoch": None, "steps": 7} and size == 640
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


def config(*overrides: str, tmp_path: Path) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        return compose(
            config_name="config",
            overrides=["device=cpu", "train.amp=false", f"train.output_dir={tmp_path}", *overrides],
        )


def test_scoring_is_off_by_default_and_needs_loco(tmp_path: Path) -> None:
    assert build_metric(config("data=synthetic", tmp_path=tmp_path)) is None
    with pytest.raises(SystemExit, match="data=loco"):
        build_metric(config("data=synthetic", "train.val_every=2", tmp_path=tmp_path))


@pytest.mark.skipif(not HAS_LOCO, reason="LOCO data not present")
def test_training_scores_the_validation_split_and_saves_a_best_checkpoint(tmp_path: Path) -> None:
    cfg = config(
        "data=loco", "data.limit=8", "data.num_workers=0", "data.val_limit=4",
        "model.input_size=128", "train.epochs=2", "train.batch_size=4", "train.val_every=1",
        tmp_path=tmp_path,
    )  # fmt: skip
    result = train_from_config(cfg)
    assert result.best == tmp_path / "best.pt" and result.best_metric is not None
    assert 0.0 <= result.best_metric <= 1.0
    log = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [e["metric"] is not None for e in log if e["type"] == "epoch"] == [True, True]


@pytest.mark.skipif(not HAS_LOCO, reason="LOCO data not present")
def test_pv_eval_scores_a_checkpoint_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = config(
        "data=loco", "data.limit=8", "data.num_workers=0", "model.input_size=128",
        "train.epochs=1", "train.batch_size=4", tmp_path=tmp_path,
    )  # fmt: skip
    train_from_config(cfg)
    out = tmp_path / "eval.json"
    main(
        [
            "--checkpoint",
            str(tmp_path / "last.pt"),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--limit",
            "6",
            "--out",
            str(out),
        ]
    )
    numbers = json.loads(out.read_text())
    assert numbers["images"] == 6 and np.isfinite(numbers["map50_95"])
    shown = capsys.readouterr().out
    assert "mAP50-95" in shown and "deploy with: pv-infer --conf" in shown
