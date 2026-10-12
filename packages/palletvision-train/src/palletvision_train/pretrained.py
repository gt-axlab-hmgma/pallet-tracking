"""Start a model from another model's weights (pretraining, then fine-tuning).

This is not resume: only the weights are taken, not the optimiser state, the epoch or the data
order, so the new run starts its own schedule from epoch 0. Tensors whose shape differs between
the two models (the class heads, when the number of classes differs: 365 for Objects365, 1 for
pallets) are left at the fresh initialisation the new model already has.
"""

from pathlib import Path

import torch

from palletvision_model.yolo26 import YOLO26


def load_pretrained(model: YOLO26, checkpoint: Path | str) -> tuple[int, list[str]]:
    """Copy every matching tensor from ``checkpoint`` into ``model``.

    Returns how many tensors were copied and the names of the ones skipped because the checkpoint
    has no such tensor or a different shape.
    """
    state = torch.load(checkpoint, map_location="cpu")
    source = state.get("model", state)
    own = model.state_dict()
    copy = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(k for k in own if k not in copy)
    model.load_state_dict(copy, strict=False)
    return len(copy), skipped
