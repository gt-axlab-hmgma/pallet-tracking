"""Save and restore a training run, so a crash or a stop does not lose it.

A checkpoint is taken at the end of an epoch and holds everything the next epoch needs: the
weights, the optimiser, the position in training, and the state of every random number
generator (the shuffle order of the next epoch depends on it). Restoring all of it makes a
resumed run follow exactly the loss curve an uninterrupted run would have.

Everything is stored as plain tensors, numbers, strings and lists, so ``torch.load`` can read
it in its default "safe" mode, which is how ``pv-export`` reads it.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

VERSION = 1


@dataclass(frozen=True)
class Resumed:
    """Where a restored run picks up."""

    epoch: int  # epochs already completed; the next one to run
    steps: int  # optimiser steps already taken
    epochs: int  # the total the run was started with
    best_metric: float | None
    config: dict[str, Any]
    ema_updates: int = 0  # updates the EMA had made (0 when the run has none)


def capture_rng() -> dict[str, Any]:
    """The state of every random number generator the training could touch."""
    legacy = np.random.get_state()
    assert isinstance(legacy, tuple)  # the Mersenne Twister state: (name, keys, pos, ...)
    kind, keys, pos, has_gauss, cached = legacy
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [kind, keys.tolist(), int(pos), int(has_gauss), float(cached)],
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate((state["python"][0], tuple(state["python"][1]), state["python"][2]))
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    steps: int,
    epochs: int,
    best_metric: float | None,
    config: dict[str, Any],
    ema: Any = None,
) -> None:
    """Write the checkpoint to ``path`` without ever leaving a half-written file there.

    With an EMA, ``model`` holds the averaged weights (what inference, scoring and ``pv-eval``
    use) and ``raw_model`` the weights being trained, which a resume needs.

    It is written next to the target and then renamed, which is atomic, so a crash while saving
    leaves the previous good checkpoint in place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": VERSION,
        "model": (ema.module if ema is not None else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "steps": steps,
        "epochs": epochs,
        "best_metric": best_metric,
        "config": config,
        "rng": capture_rng(),
    }
    if ema is not None:
        state["raw_model"] = model.state_dict()
        state["ema_updates"] = ema.updates
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: Any = None
) -> Resumed:
    """Restore weights, optimiser and random state in place; return where training stands.

    A checkpoint written with an EMA keeps the trained weights in ``raw_model``; they go into
    ``model`` and the averaged ones into ``ema``. Resuming a run that had no EMA with ``ema`` on
    starts the average from the restored weights.
    """
    state = torch.load(path, map_location="cpu")  # safe mode: only plain data is accepted
    if state.get("version") != VERSION:
        raise ValueError(f"{path} is checkpoint version {state.get('version')}, expected {VERSION}")
    trained = state.get("raw_model", state["model"])
    model.load_state_dict(trained)
    if ema is not None:
        ema.load(state["model"] if "raw_model" in state else trained, state.get("ema_updates", 0))
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return Resumed(
        epoch=state["epoch"],
        steps=state["steps"],
        epochs=state["epochs"],
        best_metric=state["best_metric"],
        config=state["config"],
        ema_updates=state.get("ema_updates", 0),
    )
