from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from palletvision_model.device import resolve_device

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def first_box_l1(outputs: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Placeholder loss: pull the first predicted box towards the target. Replace in M1."""
    return (outputs[:, 0, :4] - boxes).abs().mean() / outputs[..., :4].detach().amax().clamp(min=1)


@dataclass
class TrainResult:
    losses: list[float]
    checkpoint: Path


def train(
    model: nn.Module,
    dataset: Dataset[Batch],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    amp: bool,
    max_steps: int | None,
    output_dir: Path | str,
    loss_fn: LossFn = first_box_l1,
) -> TrainResult:
    """Minimal loop. fp16 autocast only on cuda/mps, per the laptop notes in docs/design.md."""
    dev = resolve_device(device)
    model.to(dev).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    use_amp = amp and dev.type in {"cuda", "mps"}
    losses: list[float] = []
    for _ in range(epochs):
        for images, boxes in loader:
            if max_steps is not None and len(losses) >= max_steps:
                break
            images, boxes = images.to(dev), boxes.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
            loss = loss_fn(outputs.float(), boxes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(loss.item())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "last.pt"
    torch.save({"model": model.state_dict(), "steps": len(losses)}, checkpoint)
    return TrainResult(losses, checkpoint)
