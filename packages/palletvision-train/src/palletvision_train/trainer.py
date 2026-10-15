"""The training loop: forward both heads, assign, score, backpropagate, repeat."""

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from palletvision_model.device import resolve_device
from palletvision_model.loss import DetectionLoss, progressive_alpha
from palletvision_model.yolo26 import YOLO26
from palletvision_train.checkpoint import load_checkpoint, save_checkpoint
from palletvision_train.data import Sample, collate
from palletvision_train.distributed import RawForward, average, current
from palletvision_train.ema import ModelEMA
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


def freeze_backbone(model: YOLO26, layers: int) -> None:
    """Stop gradients for the first ``layers`` layers, numbered as in the layer comments of
    ``Backbone`` and ``Neck`` (Ultralytics' ``freeze``): 0 to 10 are the backbone (10 is its last
    layer, C2PSA), 11 to 22 the neck, and 23 the head, so ``layers=10`` keeps all of the backbone
    but C2PSA fixed and ``layers=23`` trains only the head. Layers without weights (upsample,
    concat) count but hold nothing."""
    backbone = model.backbone
    stack: list[Module | None] = [*backbone.to_p3, *backbone.to_p4, *backbone.to_p5]  # 0..10
    neck = model.neck
    stack += [None, None, neck.td4, None, None, neck.td3, neck.down3, None, neck.bu4, neck.down4]
    stack += [None, neck.bu5]  # 11..22
    if not 0 <= layers <= len(stack):
        raise ValueError(f"freeze must be between 0 and {len(stack)}, got {layers}")
    for layer in stack[:layers]:
        if layer is not None:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)


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
    extra_metrics_fn: Callable[[YOLO26], dict[str, float]] | None = None,
    config: dict[str, Any] | None = None,
    sampler: Sampler[int] | None = None,
    on_epoch_start: Callable[[int], None] | None = None,
    seed: int = 0,
    pretrained: set[str] | None = None,
    pretrained_lr_mult: float = 1.0,
    alpha_schedule: str = "epoch",
    alpha_init: float = 0.8,
    alpha_final: float = 0.1,
    ema: bool = False,
    ema_decay: float = 0.9999,
    ema_tau: float = 2000.0,
    freeze: int = 0,
    patience: int = 0,
) -> TrainResult:
    """Train ``model`` on letterboxed ``(image, labels)`` samples.

    Each step runs both heads, lets the loss assign targets and score them, and updates the
    weights. The one-to-many weight ``alpha`` follows ``progressive_alpha`` from ``alpha_init`` to
    ``alpha_final``: set once per epoch (``alpha_schedule="epoch"``, the report's) or every step
    (``"step"``, needed when there are only a few epochs).
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

    **Fine-tuning.** ``pretrained`` names the parameters that start from pretrained weights; they
    train at ``pretrained_lr_mult`` times the learning rate of the others (see ``build_optimizer``).
    ``ema`` keeps an exponential moving average of the weights (see ``ModelEMA``): it is what
    ``metric_fn`` scores and what ``last.pt`` and ``best.pt`` hold as ``model``; the weights being
    trained go to ``raw_model`` for resuming.

    ``freeze`` keeps the first N layers (0 to 23, see ``freeze_backbone``: 10 is the backbone but
    its last layer, 23 everything but the head) fixed: their weights get no gradient.
    ``patience`` stops the run when the scored metric has not improved for that many epochs (0:
    never; needs scoring, and a single process).

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
    if freeze:
        freeze_backbone(model, freeze)
    if patience and info.distributed:
        raise ValueError("patience is not supported with several processes")
    optimizer = build_optimizer(
        model, optimizer_name, lr, weight_decay, momentum, muon, sgd, pretrained, pretrained_lr_mult
    )
    averaged = ModelEMA(model, ema_decay, ema_tau) if ema else None
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
        resumed = load_checkpoint(Path(resume), model, optimizer, averaged)
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
                epochs=epochs, best_metric=best_metric, config=config, ema=averaged,
            )  # fmt: skip

    stopped = False
    best_epoch = done_epochs  # the epoch of the best score; a resumed run counts from where it is
    for epoch in range(done_epochs, epochs):
        alpha = progressive_alpha(epoch, epochs, alpha_init, alpha_final)
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
            if alpha_schedule == "step":  # a fractional epoch: the weight moves with every step
                alpha = progressive_alpha(steps / len(loader), epochs, alpha_init, alpha_final)
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
                if averaged is not None:
                    averaged.update(model)
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
            rate = next(
                g["lr"] for g in optimizer.param_groups if g["name"] in ("weight", "weight_pre")
            )
            if log is not None:
                log.write(
                    "step", epoch=epoch, step=steps, alpha=alpha, lr=rate, loss=total_value,
                    **numbers,
                )  # fmt: skip
        if stopped:
            break

        done_epochs = epoch + 1
        metric, extras = None, {}
        scored = metric_fn is not None and (
            done_epochs % metric_every == 0 or done_epochs == epochs
        )
        if metric_fn is not None and scored and main:  # the others wait at the barrier below
            scored_model = averaged.module if averaged is not None else model
            model.eval()
            with torch.no_grad():
                metric = metric_fn(scored_model)
                if extra_metrics_fn is not None:
                    extras = extra_metrics_fn(scored_model)
            model.train()
            if best_metric is None or metric > best_metric:
                best_metric, best_epoch = metric, done_epochs
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
                seconds=seconds, **extras,
            )  # fmt: skip
            note = "" if metric is None else f"  metric {metric:.4f}"
            note += "".join(f"  {name} {value:.4f}" for name, value in extras.items())
            print(f"epoch {done_epochs}/{epochs}  loss {mean:.3f}  alpha {alpha:.2f}", end="")
            print(f"{note}  {seconds:.0f}s")
        if patience and metric_fn is not None and done_epochs - best_epoch >= patience:
            print(f"no improvement for {patience} epochs: stopping after epoch {done_epochs}")
            break

    if stopped or not (last.exists() or not main):
        checkpoint(last)  # a smoke run cut short by max_steps still leaves a checkpoint
    if info.distributed:
        dist.barrier()  # no process leaves while rank 0 is still writing
    result.best_metric = best_metric
    return result
