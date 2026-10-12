"""Training on several GPUs (or CPU processes): one process per device, gradients averaged.

Launch with ``torchrun``, which starts one copy of the program per GPU and tells each its rank
through environment variables:

    uv run torchrun --standalone --nproc_per_node=4 -m palletvision_train.cli data=loco ...

Each process trains on its own slice of every epoch with its own copy of the model. After each
optimiser step the gradients are averaged across processes, so every copy makes the same update
and they stay identical. Only rank 0 writes logs and checkpoints and runs validation.

Without ``torchrun`` (no ``WORLD_SIZE`` in the environment) nothing here does anything and the
trainer behaves exactly as on one device.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

# Validation, run by rank 0 alone, can take many minutes while the others wait at a barrier, and
# the first epoch can start slowly (reading hundreds of thousands of labels): do not time out.
TIMEOUT = timedelta(hours=3)


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def current() -> DistInfo:
    """The process group this process belongs to (a single process when there is none)."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        return DistInfo(rank, dist.get_world_size(), int(os.environ.get("LOCAL_RANK", rank)))
    return DistInfo()


def init_distributed(device: str = "auto") -> tuple[DistInfo, str]:
    """Join the process group ``torchrun`` set up; return who this process is and its device.

    ``device`` is the config value. Several processes need CUDA (one GPU each) or the CPU
    (``gloo``, for testing); Apple's MPS cannot be shared between processes. Returns
    ``(DistInfo(), device)`` unchanged when not launched by ``torchrun``.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return DistInfo(), device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.split(":")[0] not in {"cuda", "cpu"}:
        raise ValueError(f"several processes need device cuda or cpu, not '{device}'")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if device.startswith("cuda"):
        if torch.cuda.device_count() <= local_rank:
            raise RuntimeError(
                f"process {rank} wants GPU {local_rank} but only {torch.cuda.device_count()} "
                "are visible: use --nproc_per_node no larger than the GPU count"
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    dist.init_process_group("nccl" if device.startswith("cuda") else "gloo", timeout=TIMEOUT)
    return DistInfo(rank, world, local_rank), device


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class RawForward(nn.Module):
    """Gives ``DistributedDataParallel`` a ``forward`` that is the model's training output.

    DDP only tracks gradients for what goes through ``forward``, and the model's own ``forward``
    is the inference path (top-300 boxes). Training uses ``raw``, so wrap it.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> object:
        return self.model.raw(images)  # type: ignore[operator]


def average(values: torch.Tensor) -> torch.Tensor:
    """The mean of ``values`` over all processes (the same tensor in a single process)."""
    if not current().distributed:
        return values
    total = values.clone()
    dist.all_reduce(total)  # sum: gloo has no mean
    return total / dist.get_world_size()
