"""The optimiser, its parameter groups, and the learning-rate schedule.

YOLO26 trains with **MuSGD**, implemented here: ordinary SGD with momentum, plus, on the
multi-dimensional weights only, a second update that is "orthogonalised" (Muon). See ``MuSGD``.

Two ideas shape the groups and the schedule:
- **Weight decay** gently pulls weights toward zero to stop them growing. It helps convolution
  weights, but hurts on BatchNorm scales and on biases, so the parameters go in three groups
  and only the convolution weights are decayed.
- **Warmup then decay.** A fresh network takes wild steps, so the learning rate starts near
  zero and climbs to its full value over the first epochs; afterwards it falls smoothly so the
  last epochs make fine adjustments instead of bouncing around the best point.

The schedule is a pure function of the iteration number, so a resumed run needs nothing but
the step count the checkpoint already holds.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)
GROUP_ORDER = ("bias", "norm", "weight")  # the bias group is first on purpose


def split_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Every parameter of ``model`` in exactly one group: ``bias``, ``norm`` or ``weight``.

    ``norm`` holds the scales of the normalisation layers; ``weight`` holds the convolution (and
    linear) weights, the only ones that get weight decay.
    """
    groups: dict[str, list[nn.Parameter]] = {name: [] for name in GROUP_ORDER}
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "bias":
                groups["bias"].append(parameter)
            elif isinstance(module, NORMS):
                groups["norm"].append(parameter)
            else:
                groups["weight"].append(parameter)
    return groups


def newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Push a matrix toward its nearest orthogonal one: all singular values near 1.

    This is the heart of Muon. Raw gradient matrices are dominated by a few large directions;
    evening the singular values out makes an update that moves every direction by a similar
    amount. It does so with a few matrix products (a Newton-Schulz iteration) instead of an
    expensive decomposition. The coefficients are Keller Jordan's (the published Muon
    reference); five steps leave the singular values roughly between 0.5 and 1.5, not exactly 1.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    tall = x.size(-2) > x.size(-1)
    if tall:  # work with the wide orientation: the products below are then the smaller ones
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)  # the top singular value is now at most 1
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    result: torch.Tensor = x.mT if tall else x
    return result


class MuSGD(torch.optim.Optimizer):
    """SGD with momentum, plus a Muon update on the groups marked ``use_muon``.

    For a parameter in a Muon group (the convolution weights) each step adds two updates:
    - the **Muon** one: the momentum-averaged gradient, flattened to a matrix of shape
      ``[out, everything else]``, orthogonalised, and applied with weight ``muon`` (no weight
      decay);
    - the **SGD** one: ordinary Nesterov SGD momentum on the gradient plus weight decay, applied
      with weight ``sgd``.
    Parameters in the other groups (biases, normalisation scales) get plain SGD at the full
    learning rate. ``muon`` and ``sgd`` default to 0.45 and 0.55, the values recorded for the
    YOLO26n checkpoints (YOLO26 report, Table S4). The update follows Ultralytics' MuSGD.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        momentum: float = 0.937,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        use_muon: bool = False,
        muon: float = 0.45,
        sgd: float = 0.55,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": use_muon,
        }
        super().__init__(params, defaults)
        self.muon, self.sgd = muon, sgd

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            sgd_lr = lr * self.sgd if group["use_muon"] else lr
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad, state = p.grad, self.state[p]
                if group["use_muon"]:
                    average = state.setdefault("muon_momentum", torch.zeros_like(p))
                    average.mul_(momentum).add_(grad, alpha=1 - momentum)  # a running average
                    update = grad.lerp(average, momentum) if nesterov else average
                    flat = update.reshape(update.size(0), -1)  # [out, in * kernel size]
                    shape_scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                    ortho = newton_schulz(flat).reshape(update.shape).to(p.dtype)
                    p.add_(ortho, alpha=-(lr * self.muon * shape_scale))
                decayed = grad + group["weight_decay"] * p if group["weight_decay"] else grad
                buffer = state.setdefault("sgd_momentum", torch.zeros_like(p))
                buffer.mul_(momentum).add_(decayed)
                p.add_(decayed + momentum * buffer if nesterov else buffer, alpha=-sgd_lr)
        return loss


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.937,
    muon: float = 0.45,
    sgd: float = 0.55,
) -> torch.optim.Optimizer:
    """MuSGD, SGD (Nesterov momentum) or AdamW over the three groups; only ``weight`` is decayed.

    ``momentum`` is the (Nesterov) momentum of SGD and MuSGD, or AdamW's first beta. In MuSGD
    only the ``weight`` group gets the Muon update, mixed in with weights ``muon`` and ``sgd``.
    """
    groups = split_parameters(model)
    specs = [
        {
            "name": group,
            "params": groups[group],
            "weight_decay": weight_decay if group == "weight" else 0.0,
        }
        for group in GROUP_ORDER
    ]
    if name == "musgd":
        for spec in specs:
            spec["use_muon"] = spec["name"] == "weight"
        return MuSGD(specs, lr=lr, momentum=momentum, nesterov=momentum > 0, muon=muon, sgd=sgd)
    if name == "sgd":
        return torch.optim.SGD(specs, lr=lr, momentum=momentum, nesterov=momentum > 0)
    if name == "adamw":
        return torch.optim.AdamW(specs, lr=lr, betas=(momentum, 0.999))
    raise ValueError(f"unknown optimizer '{name}': use 'musgd', 'sgd' or 'adamw'")


@dataclass(frozen=True)
class Schedule:
    """Learning rate and momentum at every iteration (one iteration is one batch).

    - Warmup lasts ``warmup_epochs`` epochs but never fewer than 100 iterations. The learning
      rate climbs linearly from 0 to its scheduled value; the bias group starts from
      ``warmup_bias_lr`` and falls to it instead, and the momentum climbs from
      ``warmup_momentum`` to ``momentum``.
    - After warmup the rate follows a cosine (or straight line) from ``lr`` down to
      ``lr * lr_final`` at the last epoch.
    With the defaults (no warmup, ``lr_final`` 1) the rate is simply constant.
    """

    lr: float
    epochs: int
    batches_per_epoch: int
    warmup_epochs: float = 0.0
    warmup_bias_lr: float = 0.0
    warmup_momentum: float = 0.8
    momentum: float = 0.937
    lr_final: float = 1.0
    kind: str = "cosine"

    @property
    def warmup_iterations(self) -> int:
        if self.warmup_epochs <= 0:
            return 0
        return max(round(self.warmup_epochs * self.batches_per_epoch), 100)

    def decay(self, iteration: int) -> float:
        """1 at the start, ``lr_final`` at the end of the last epoch."""
        progress = min(iteration / (self.epochs * self.batches_per_epoch), 1.0)
        if self.kind == "cosine":
            shape = 0.5 * (1 + math.cos(math.pi * progress))
        elif self.kind == "linear":
            shape = 1 - progress
        else:
            raise ValueError(f"unknown schedule '{self.kind}': use 'cosine' or 'linear'")
        return self.lr_final + (1 - self.lr_final) * shape

    def learning_rate(self, group: str, iteration: int) -> float:
        scheduled = self.lr * self.decay(iteration)
        warm = self.warmup_iterations
        if iteration >= warm:
            return scheduled
        start = self.warmup_bias_lr if group == "bias" else 0.0
        return start + (scheduled - start) * iteration / warm

    def momentum_at(self, iteration: int) -> float:
        warm = self.warmup_iterations
        if iteration >= warm:
            return self.momentum
        return self.warmup_momentum + (self.momentum - self.warmup_momentum) * iteration / warm

    def apply(self, optimizer: torch.optim.Optimizer, iteration: int) -> None:
        """Set every group's learning rate and momentum for ``iteration``."""
        momentum = self.momentum_at(iteration)
        for group in optimizer.param_groups:
            group["lr"] = self.learning_rate(group.get("name", "weight"), iteration)
            if "momentum" in group:
                group["momentum"] = momentum
            elif "betas" in group:
                group["betas"] = (momentum, group["betas"][1])
