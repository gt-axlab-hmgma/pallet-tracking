"""The training loop: forward both heads, assign, score, backpropagate, repeat."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from palletvision_model.device import resolve_device
from palletvision_model.loss import DetectionLoss, progressive_alpha
from palletvision_model.yolo26 import YOLO26
from palletvision_train.data import Sample, collate

MAX_GRAD_NORM = 10.0  # a single bad batch must not throw the weights far away


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)  # total loss at every step
    parts: list[dict[str, float]] = field(default_factory=list)  # each component, every step
    alphas: list[float] = field(default_factory=list)  # one-to-many weight used at every step
    checkpoint: Path = Path()


def train(
    model: YOLO26,
    dataset: Dataset[Sample],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    amp: bool = False,
    max_steps: int | None = None,
    output_dir: Path | str,
    num_workers: int = 0,
    loss_fn: DetectionLoss | None = None,
) -> TrainResult:
    """Train ``model`` on letterboxed ``(image, labels)`` samples and save ``last.pt``.

    Each step runs both heads, lets the loss assign targets and score them, and updates the
    weights. The one-to-many weight ``alpha`` is set once per epoch from ``progressive_alpha``.
    The loss itself always runs in float32; only the forward pass uses half precision (cuda and
    mps) when ``amp`` is on.
    """
    dev = resolve_device(device)
    model.to(dev).train()
    loss_fn = loss_fn or DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=dev.type == "cuda",
    )  # fmt: skip
    use_amp = amp and dev.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda")

    result = TrainResult()
    for epoch in range(epochs):
        alpha = progressive_alpha(epoch, epochs)
        for batch in loader:
            if max_steps is not None and len(result.losses) >= max_steps:
                break
            batch = batch.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                preds = model.raw(batch.images)
            image_hw = (batch.images.shape[-2], batch.images.shape[-1])
            total, parts = loss_fn(
                preds, image_hw, batch.gt_labels, batch.gt_boxes, batch.mask_gt, alpha
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            result.losses.append(total.item())
            result.parts.append({name: value.item() for name, value in parts.items()})
            result.alphas.append(alpha)
        if max_steps is not None and len(result.losses) >= max_steps:
            break

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.checkpoint = out / "last.pt"
    torch.save({"model": model.state_dict(), "steps": len(result.losses)}, result.checkpoint)
    return result
