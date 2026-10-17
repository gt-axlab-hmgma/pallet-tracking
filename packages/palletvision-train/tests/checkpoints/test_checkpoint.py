import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from palletvision_model.config import SIZES
from palletvision_model.loss.loss import DetectionLoss
from palletvision_model.yolo26 import YOLO26
from palletvision_train.checkpoints.checkpoint import capture_rng, restore_rng, save_checkpoint
from palletvision_train.cli.train import resume_path
from palletvision_train.datasets.data import LetterboxedDataset
from palletvision_train.datasets.synthetic import SyntheticBoxes
from palletvision_train.training.trainer import train

# 8 images in batches of 4 is 2 steps per epoch
COMMON: dict[str, Any] = {"device": "cpu", "batch_size": 4, "lr": 1e-3}


def model_with(seed: int) -> YOLO26:
    torch.manual_seed(seed)
    return YOLO26(SIZES["n"], input_size=64)


def images() -> LetterboxedDataset:
    return LetterboxedDataset(SyntheticBoxes(n=8, size=64), size=64)


class CrashingLoss(DetectionLoss):
    """Behaves like the real loss, but raises on one chosen call, as a crash would."""

    def __init__(self, crash_at: int) -> None:
        super().__init__()
        self.calls, self.crash_at = 0, crash_at

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == self.crash_at:
            raise RuntimeError("simulated crash")
        return super().forward(*args, **kwargs)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_resumed_run_follows_the_curve_of_an_uninterrupted_one(tmp_path: Path) -> None:
    whole = model_with(0)
    torch.manual_seed(5)  # after building the model: building it reseeds the generator
    uninterrupted = train(whole, images(), epochs=4, output_dir=tmp_path / "whole", **COMMON)
    assert len(uninterrupted.losses) == 8

    crashed = tmp_path / "crashed"
    doomed = model_with(0)
    torch.manual_seed(5)  # the same data order as the uninterrupted run
    with pytest.raises(RuntimeError, match="simulated crash"):
        # step 6 is the second step of the third epoch: two epochs are already saved
        train(doomed, images(), epochs=4, output_dir=crashed, loss_fn=CrashingLoss(6), **COMMON)
    assert torch.load(crashed / "last.pt")["epoch"] == 2

    fresh = model_with(99)  # a different starting point: the checkpoint must override it all
    resumed = train(
        fresh, images(), epochs=4, output_dir=crashed, resume=crashed / "last.pt", **COMMON
    )
    assert resumed.start_epoch == 2 and len(resumed.losses) == 4
    assert resumed.losses == pytest.approx(uninterrupted.losses[4:], rel=1e-5)
    assert resumed.alphas == uninterrupted.alphas[4:]
    for name, tensor in whole.state_dict().items():  # and it ends at the same weights
        assert torch.allclose(fresh.state_dict()[name], tensor, atol=1e-6), name


def test_the_log_has_every_step_and_epoch_exactly_once_after_a_crash_and_resume(
    tmp_path: Path,
) -> None:
    doomed = model_with(0)
    torch.manual_seed(5)
    with pytest.raises(RuntimeError):
        train(
            doomed,
            images(),
            epochs=4,
            output_dir=tmp_path,
            loss_fn=CrashingLoss(6),
            **COMMON,
        )
    before = lines(tmp_path / "metrics.jsonl")
    assert [e["step"] for e in before if e["type"] == "step"] == [
        1,
        2,
        3,
        4,
        5,
    ]  # step 5 never saved

    train(
        model_with(1),
        images(),
        epochs=4,
        output_dir=tmp_path,
        resume=tmp_path / "last.pt",
        **COMMON,
    )
    log = lines(tmp_path / "metrics.jsonl")
    assert [e["step"] for e in log if e["type"] == "step"] == list(range(1, 9))
    assert [e["epoch"] for e in log if e["type"] == "epoch"] == [0, 1, 2, 3]
    first = log[0]
    assert {"epoch", "step", "alpha", "loss", "box_one2one", "cls_one2many"} <= set(first)


def test_a_checkpoint_is_written_every_epoch_and_readable_in_safe_mode(tmp_path: Path) -> None:
    config = {"data": {"name": "synthetic"}, "seed": 3}
    train(model_with(0), images(), epochs=2, output_dir=tmp_path, config=config, **COMMON)
    # torch.load's default refuses arbitrary objects; pv-export reads checkpoints this way
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    assert {"model", "optimizer", "epoch", "steps", "epochs", "config", "rng"} <= set(state)
    assert (state["epoch"], state["steps"], state["epochs"]) == (2, 4, 2)
    assert state["config"] == config
    assert not list(tmp_path.glob("*.tmp"))  # nothing half-written is left behind


def test_a_failed_save_leaves_the_previous_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = model_with(0)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "last.pt"
    args: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "epochs": 5,
        "best_metric": None,
    }
    save_checkpoint(path, epoch=1, steps=10, config={}, **args)

    def dies_halfway(obj: Any, target: Any) -> None:
        Path(target).write_bytes(b"half a checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", dies_halfway)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, epoch=2, steps=20, config={}, **args)
    monkeypatch.undo()
    assert torch.load(path)["epoch"] == 1  # the good one is still there and still loads


def test_best_is_saved_only_when_the_metric_improves(tmp_path: Path) -> None:
    scores = iter([0.1, 0.5, 0.3, 0.4])
    result = train(
        model_with(0), images(), epochs=4, output_dir=tmp_path,
        metric_fn=lambda model: next(scores), **COMMON,
    )  # fmt: skip
    assert result.best == tmp_path / "best.pt" and result.best_metric == 0.5
    best = torch.load(tmp_path / "best.pt")
    assert best["epoch"] == 2 and best["best_metric"] == 0.5  # the second epoch was the best
    assert torch.load(tmp_path / "last.pt")["epoch"] == 4  # last.pt still follows training
    epoch_lines = [e for e in lines(tmp_path / "metrics.jsonl") if e["type"] == "epoch"]
    assert [e["metric"] for e in epoch_lines] == [0.1, 0.5, 0.3, 0.4]


def test_without_a_metric_there_is_no_best_checkpoint(tmp_path: Path) -> None:
    result = train(model_with(0), images(), epochs=1, output_dir=tmp_path, **COMMON)
    assert result.best is None and not (tmp_path / "best.pt").exists()


def test_the_best_score_survives_a_resume(tmp_path: Path) -> None:
    scores = iter([0.9, 0.1, 0.2, 0.3])
    train(
        model_with(0),
        images(),
        epochs=2,
        output_dir=tmp_path,
        metric_fn=lambda m: next(scores),
        **COMMON,
    )
    # a resume continuing from epoch 1 must not treat a worse score as the new best
    # (epochs differ here on purpose, so train a matching 4-epoch run first)
    scores = iter([0.9, 0.1, 0.2, 0.3])
    first = tmp_path / "again"
    train(
        model_with(0),
        images(),
        epochs=4,
        output_dir=first,
        metric_fn=lambda m: next(scores),
        max_steps=4,
        **COMMON,
    )
    assert torch.load(first / "last.pt")["best_metric"] == 0.9
    rest = iter([0.2, 0.3])
    result = train(
        model_with(1), images(), epochs=4, output_dir=first, resume=first / "last.pt",
        metric_fn=lambda m: next(rest), **COMMON,
    )  # fmt: skip
    assert result.best_metric == 0.9  # still the first epoch's score


def test_resuming_with_a_different_epoch_count_is_refused(tmp_path: Path) -> None:
    train(model_with(0), images(), epochs=2, output_dir=tmp_path, **COMMON)
    with pytest.raises(ValueError, match="same train.epochs"):
        train(
            model_with(0),
            images(),
            epochs=5,
            output_dir=tmp_path,
            resume=tmp_path / "last.pt",
            **COMMON,
        )


def test_resuming_a_finished_run_does_nothing_and_does_not_fail(tmp_path: Path) -> None:
    train(model_with(0), images(), epochs=2, output_dir=tmp_path, **COMMON)
    again = train(
        model_with(0),
        images(),
        epochs=2,
        output_dir=tmp_path,
        resume=tmp_path / "last.pt",
        **COMMON,
    )
    assert again.losses == [] and again.start_epoch == 2
    assert [e["step"] for e in lines(tmp_path / "metrics.jsonl") if e["type"] == "step"] == [
        1,
        2,
        3,
        4,
    ]


def test_random_state_round_trips() -> None:
    saved = capture_rng()
    first = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())
    restore_rng(saved)
    second = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())
    assert first == second


def test_resume_setting_null_auto_or_a_path(tmp_path: Path) -> None:
    def setting(value: object) -> Any:
        return OmegaConf.create({"train": {"resume": value, "output_dir": str(tmp_path)}})

    assert resume_path(setting(None)) is None
    assert resume_path(setting("auto")) is None  # nothing to resume yet: start fresh
    (tmp_path / "last.pt").write_bytes(b"x")
    assert resume_path(setting("auto")) == tmp_path / "last.pt"
    assert resume_path(setting("elsewhere/last.pt")) == Path("elsewhere/last.pt")
