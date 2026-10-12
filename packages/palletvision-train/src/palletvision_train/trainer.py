"""The training loop: forward both heads, assign, score, backpropagate, repeat."""

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from palletvision_model.device import resolve_device
from palletvision_model.loss import DetectionLoss, progressive_alpha
from palletvision_model.yolo26 import YOLO26
from palletvision_train.checkpoint import load_checkpoint, save_checkpoint
from palletvision_train.data import Sample, collate
from palletvision_train.distributed import RawForward, average, current
from palletvision_train.optim import Schedule, build_optimizer
from palletvision_train.precision import configure_backend, plan_amp
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
    amp_dtype: str = "auto",
    max_steps: int | None = None,
    output_dir: Path | str,
    num_workers: int = 0,
    loss_fn: DetectionLoss | None = None,
    resume: Path | str | None = None,
    metric_fn: Callable[[YOLO26], float] | None = None,
    metric_every: int = 1,
    config: dict[str, Any] | None = None,
    sampler: Sampler[int] | None = None,
    on_epoch_start: Callable[[int], None] | None = None,
    seed: int = 0,
) -> TrainResult:
    """Train ``model`` on letterboxed ``(image, labels)`` samples.

    Each step runs both heads, lets the loss assign targets and score them, and updates the
    weights. The one-to-many weight ``alpha`` is set once per epoch from ``progressive_alpha``.
    The loss itself always runs in float32; only the forward pass uses reduced precision when
    ``amp`` is on (``amp_dtype``: ``bfloat16`` or ``float16``, see ``plan_amp``; ``auto`` picks
    bfloat16 on a CUDA card that supports it).

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

    ``on_epoch_start(epoch)`` is called before each epoch's batches (a resumed run starts at the
    epoch it resumes from); training uses it to switch mosaic off for the last epochs.

    **Several devices.** Started by ``torchrun`` (see ``palletvision_train.distributed``), every
    process runs this function on its own device with its own slice of each epoch. ``batch_size``
    is per process, the update uses ``batch_size * processes * accumulation`` images
    (``nominal_batch`` counts all of them), gradients are averaged across processes once per
    update (not once per accumulated batch), and the logged losses are the mean over processes.
    Only process 0 writes ``metrics.jsonl`` and the checkpoints and runs ``metric_fn``; the
    others wait for it. ``sampler`` must then be shard-aware (``MixedSampler`` with a ``seed``);
    without one each process takes its share through ``DistributedSampler(seed=seed)``. A
    resumed multi-process run continues from the saved epoch with the same data order, but the
    augmentation randomness is not bit-identical to an uninterrupted run.

    A run cut short by
    ``max_steps`` is a smoke test: its last checkpoint is for inspection, and resuming it
    replays the unfinished epoch from its start.
    """
    info = current()
    dev = resolve_device(device)
    if info.distributed and dev.type == "cuda":
        dev = torch.device("cuda", info.local_rank)
        torch.cuda.set_device(dev)
    main = info.is_main
    model.to(dev).train()
    loss_fn = loss_fn or DetectionLoss()
    per_step = batch_size * info.world_size  # images in one step, over all processes
    accumulate = max(round(nominal_batch / per_step), 1) if nominal_batch else 1
    if nominal_batch:
        weight_decay *= per_step * accumulate / nominal_batch
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum, muon, sgd)
    if info.distributed and sampler is None:
        sampler = DistributedSampler(
            dataset, num_replicas=info.world_size, rank=info.rank, shuffle=True, seed=seed
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        collate_fn=collate, num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=dev.type == "cuda",
    )  # fmt: skip
    schedule = Schedule(
        lr=lr, epochs=epochs, batches_per_epoch=len(loader), warmup_epochs=warmup_epochs,
        warmup_bias_lr=warmup_bias_lr, momentum=momentum, lr_final=lr_final, kind=lr_schedule,
    )  # fmt: skip
    plan = plan_amp(dev.type, amp, amp_dtype)
    configure_backend(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=plan.scaler)

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
        if main:
            print(f"resumed from {resume}: {done_epochs}/{epochs} epochs done, step {steps}")
    train_model: DistributedDataParallel | None = None
    if info.distributed:
        # every process starts from rank 0's weights; then each gets its own random stream so
        # their augmentation differs (the data order is shared through the sampler's seed)
        train_model = DistributedDataParallel(
            RawForward(model), device_ids=[dev.index] if dev.type == "cuda" else None
        )
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()) + info.rank)
    log = (
        RunLog(out / "metrics.jsonl", keep_up_to_step=steps if resume is not None else None)
        if main
        else None
    )
    result.start_epoch = done_epochs

    def checkpoint(path: Path) -> None:
        if main:
            save_checkpoint(
                path, model=model, optimizer=optimizer, epoch=done_epochs, steps=steps,
                epochs=epochs, best_metric=best_metric, config=config,
            )  # fmt: skip

    stopped = False
    for epoch in range(done_epochs, epochs):
        alpha = progressive_alpha(epoch, epochs)
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        set_epoch = getattr(sampler, "set_epoch", None)  # DistributedSampler, MixedSampler
        if set_epoch is not None:
            set_epoch(epoch)
        began, epoch_losses = time.time(), []
        for index, batch in enumerate(loader):
            if max_steps is not None and steps >= max_steps:
                stopped = True
                break
            batch = batch.to(dev, non_blocking=dev.type == "cuda")
            update_now = (index + 1) % accumulate == 0 or index + 1 == len(loader)
            # gradients are averaged across processes only on the batch that ends an update
            no_sync = (
                train_model.no_sync()
                if train_model is not None and not update_now
                else contextlib.nullcontext()
            )
            with no_sync:
                with torch.autocast(device_type=dev.type, dtype=plan.dtype, enabled=plan.autocast):
                    preds = train_model(batch.images) if train_model else model.raw(batch.images)
                image_hw = (batch.images.shape[-2], batch.images.shape[-1])
                total, parts = loss_fn(
                    preds, image_hw, batch.gt_labels, batch.gt_boxes, batch.mask_gt, alpha
                )
                schedule.apply(optimizer, steps)  # this iteration's learning rate and momentum
                scaler.scale(total / accumulate).backward()  # the mean over the accumulated batches
            if update_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                result.updates += 1
            steps += 1
            shown = average(torch.stack([total.detach(), *parts.values()]).float())
            total_value, numbers = (
                float(shown[0]),
                dict(zip(parts, shown[1:].tolist(), strict=True)),
            )
            result.losses.append(total_value)
            result.parts.append(numbers)
            result.alphas.append(alpha)
            epoch_losses.append(total_value)
            rate = next(g["lr"] for g in optimizer.param_groups if g["name"] == "weight")
            if log is not None:
                log.write(
                    "step", epoch=epoch, step=steps, alpha=alpha, lr=rate, loss=total_value,
                    **numbers,
                )  # fmt: skip
        if stopped:
            break

        done_epochs = epoch + 1
        metric = None
        scored = metric_fn is not None and (
            done_epochs % metric_every == 0 or done_epochs == epochs
        )
        if metric_fn is not None and scored and main:  # the others wait at the barrier below
            model.eval()
            with torch.no_grad():
                metric = metric_fn(model)
            model.train()
            if best_metric is None or metric > best_metric:
                best_metric = metric
                checkpoint(best)
                result.best = best
        checkpoint(last)
        if info.distributed:
            dist.barrier()
        mean = sum(epoch_losses) / max(len(epoch_losses), 1)
        seconds = time.time() - began
        if log is not None:
            log.write(
                "epoch", epoch=epoch, step=steps, alpha=alpha, loss=mean, metric=metric,
                seconds=seconds,
            )  # fmt: skip
            note = "" if metric is None else f"  metric {metric:.4f}"
            print(f"epoch {done_epochs}/{epochs}  loss {mean:.3f}  alpha {alpha:.2f}", end="")
            print(f"{note}  {seconds:.0f}s")

    if stopped or not (last.exists() or not main):
        checkpoint(last)  # a smoke run cut short by max_steps still leaves a checkpoint
    if info.distributed:
        dist.barrier()  # no process leaves while rank 0 is still writing
    result.best_metric = best_metric
    return result
