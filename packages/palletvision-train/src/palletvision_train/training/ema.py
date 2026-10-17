"""An exponential moving average of the weights.

After every optimiser update the average moves a little toward the current weights:
``average = d * average + (1 - d) * weights``. The raw weights jump around from batch to batch;
the average is steadier and usually scores a little better, which is why it is the copy that is
scored, saved as ``best.pt`` and used for inference.

The decay ``d`` ramps up from 0 as ``decay * (1 - exp(-updates / tau))``, so early in training the
average follows the weights closely instead of dragging the random starting point along. How far
back the average looks is about ``1 / (1 - d)`` updates: the default 0.9999 suits a run of tens of
thousands of updates; a short fine-tune of a thousand updates wants about 0.99 to 0.995.
"""

import copy
import math
from typing import Generic, TypeVar

import torch
from torch import nn

M = TypeVar("M", bound=nn.Module)


class ModelEMA(Generic[M]):
    def __init__(self, model: M, decay: float = 0.9999, tau: float = 2000.0) -> None:
        if not 0 <= decay < 1:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.module: M = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    def current_decay(self) -> float:
        return self.decay * (1 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Move the average toward ``model``; call once after each optimiser update."""
        self.updates += 1
        d = self.current_decay()
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(source[name].detach(), alpha=1 - d)
            else:  # counters such as BatchNorm's num_batches_tracked
                value.copy_(source[name])

    def load(self, state: dict[str, torch.Tensor], updates: int) -> None:
        self.module.load_state_dict(state)
        self.updates = updates
