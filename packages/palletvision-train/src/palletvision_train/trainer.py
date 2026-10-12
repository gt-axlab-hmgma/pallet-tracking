"""The training loop: forward both heads, assign, score, backpropagate, repeat."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from palletvision_model.device import resolve_device
from palletvision_model.loss import DetectionLoss, progressive_alpha
from palletvision_model.yolo26 import YOLO26
from palletvision_train.checkpoint import load_checkpoint, save_checkpoint
from palletvision_train.data import Sample, collate
from palletvision_train.optim import Schedule, build_optimizer
from palletvision_train.runlog import RunLog

MAX_GRAD_NORM = 10.0  # a single bad batch must not throw the weights far away


@dataclass
class TrainResult:
    """What this call to ``train`` did (a resumed run reports only the steps it took)."""

    losses: list[float] = field(default_factory=list)  # total loss at every step
    parts: list[dict[str, float]] = field(default_factory=list)  # each component, every step
    alphas: list[float] = field(default_factory=list)  # one-to-many weight used at every step
    checkpoint: Path = Path()  # last.pt
    best: Path | None = None  # best.pt, only when a metric was given
    best_metric: float | None = None
    start_epoch: int = 0  # 0 for a fresh run, the first epoch run after a resume otherwise
    updates: int = 0  # optimiser updates this call made (fewer than batches when accumulating)


def train(
    model: YOLO26,
    dataset: Dataset[Sample],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    momentum: float = 0.937,
    muon: float = 0.45,
    sgd: float = 0.55,
    nominal_batch: int | None = None,
    warmup_epochs: float = 0.0,
    warmup_bias_lr: float = 0.0,
    lr_final: float = 1.0,
    lr_schedule: str = "cosine",
    amp: bool = False,
    max_steps: int | None = None,
    output_dir: Path | str,
    num_workers: int = 0,
    loss_fn: DetectionLoss | None = None,
    resume: Path | str | None = None,
    metric_fn: Callable[[YOLO26], float] | None = None,
    metric_every: int = 1,
    config: dict[str, Any] | None = None,
    sampler: Sampler[int] | None = None,
) -> TrainResult:
    """Train ``model`` on letterboxed ``(image, labels)`` samples.

    Each step runs both heads, lets the loss assign targets and score them, and updates the
    weights. The one-to-many weight ``alpha`` is set once per epoch from ``progressive_alpha``.
    The loss itself always runs in float32; only the forward pass uses half precision (cuda and
    mps) when ``amp`` is on.

    Files written to ``output_dir``:
    - ``last.pt``: a checkpoint after every epoch (atomic, so a crash never corrupts it),
    - ``best.pt``: a checkpoint whenever ``metric_fn(model)`` (higher is better) beats every
      earlier score; only when a metric is given. It runs every ``metric_every`` epochs and always
      after the last one,
    - ``metrics.jsonl``: one line per step and one per epoch, written as the run goes.

    ``resume`` is a checkpoint to continue from. Weights, optimiser and every random number
    generator are restored, so the resumed run follows the curve an uninterrupted one would. It
    needs the same ``epochs`` (the one-to-many weight depends on it).

    Optimiser: ``optimizer_name`` is ``musgd`` (YOLO26's: SGD plus a Muon update on the
    convolution weights, mixed with weights ``muon`` and ``sgd``), ``sgd`` or ``adamw``, with
    weight decay on the convolution weights only. ``nominal_batch`` (the guide uses 64) makes
    small batches behave like one big one: the gradients of ``round(nominal_batch / batch_size)``
    batches are averaged before each update, and weight decay is scaled by the effective batch
    over ``nominal_batch``. The learning rate warms up for ``warmup_epochs`` and then decays to
    ``lr * lr_final`` (see ``Schedule``); the defaults keep it constant.

    ``sampler`` chooses which images each epoch uses (default: every image once, shuffled); an
    epoch is as long as the sampler says, e.g. ``MixedSampler``.

    A run cut short by
    ``max_steps`` is a smoke test: its last checkpoint is for inspection, and resuming it
    replays the unfinished epoch from its start.
    """
    dev = resolve_device(device)
    model.to(dev).train()
    loss_fn = loss_fn or DetectionLoss()
    accumulate = max(round(nominal_batch / batch_size), 1) if nominal_batch else 1
    if nominal_batch:
        weight_decay *= batch_size * accumulate / nominal_batch
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum, muon, sgd)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        collate_fn=collate, num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=dev.type == "cuda",
    )  # fmt: skip
    schedule = Schedule(
        lr=lr, epochs=epochs, batches_per_epoch=len(loader), warmup_epochs=warmup_epochs,
        warmup_bias_lr=warmup_bias_lr, momentum=momentum, lr_final=lr_final, kind=lr_schedule,
    )  # fmt: skip
    use_amp = amp and dev.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda")

    out = Path(output_dir)
    last, best = out / "last.pt", out / "best.pt"
    config = config or {}
    result = TrainResult(checkpoint=last)
    done_epochs, steps, best_metric = 0, 0, None
    if resume is not None:
        resumed = load_checkpoint(Path(resume), model, optimizer)
        if resumed.epochs != epochs:
            raise ValueError(
                f"{resume} was started for {resumed.epochs} epochs but this run asks for {epochs}; "
                "resume with the same train.epochs"
            )
        done_epochs, steps, best_metric = resumed.epoch, resumed.steps, resumed.best_metric
        print(f"resumed from {resume}: {done_epochs}/{epochs} epochs done, step {steps}")
    log = RunLog(out / "metrics.jsonl", keep_up_to_step=steps if resume is not None else None)
    result.start_epoch = done_epochs

    def checkpoint(path: Path) -> None:
        save_checkpoint(
            path, model=model, optimizer=optimizer, epoch=done_epochs, steps=steps,
            epochs=epochs, best_metric=best_metric, config=config,
        )  # fmt: skip

    stopped = False
    for epoch in range(done_epochs, epochs):
        alpha = progressive_alpha(epoch, epochs)
        began, epoch_losses = time.time(), []
        for index, batch in enumerate(loader):
            if max_steps is not None and steps >= max_steps:
                stopped = True
                break
            batch = batch.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                preds = model.raw(batch.images)
            image_hw = (batch.images.shape[-2], batch.images.shape[-1])
            total, parts = loss_fn(
                preds, image_hw, batch.gt_labels, batch.gt_boxes, batch.mask_gt, alpha
            )
            schedule.apply(optimizer, steps)  # this iteration's learning rate and momentum
            scaler.scale(total / accumulate).backward()  # the mean over the accumulated batches
            if (index + 1) % accumulate == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                result.updates += 1
            steps += 1
            numbers = {name: value.item() for name, value in parts.items()}
            result.losses.append(total.item())
            result.parts.append(numbers)
            result.alphas.append(alpha)
            epoch_losses.append(total.item())
            rate = next(g["lr"] for g in optimizer.param_groups if g["name"] == "weight")
            log.write(
                "step", epoch=epoch, step=steps, alpha=alpha, lr=rate, loss=total.item(), **numbers
            )
        if stopped:
            break

        done_epochs = epoch + 1
        metric = None
        if metric_fn is not None and (done_epochs % metric_every == 0 or done_epochs == epochs):
            model.eval()
            with torch.no_grad():
                metric = metric_fn(model)
            model.train()
            if best_metric is None or metric > best_metric:
                best_metric = metric
                checkpoint(best)
                result.best = best
        checkpoint(last)
        mean = sum(epoch_losses) / max(len(epoch_losses), 1)
        seconds = time.time() - began
        log.write(
            "epoch", epoch=epoch, step=steps, alpha=alpha, loss=mean, metric=metric, seconds=seconds
        )
        shown = "" if metric is None else f"  metric {metric:.4f}"
        print(f"epoch {done_epochs}/{epochs}  loss {mean:.3f}  alpha {alpha:.2f}", end="")
        print(f"{shown}  {seconds:.0f}s")

    if stopped or not last.exists():
        checkpoint(last)  # a smoke run cut short by max_steps still leaves a checkpoint
    result.best_metric = best_metric
    return result
