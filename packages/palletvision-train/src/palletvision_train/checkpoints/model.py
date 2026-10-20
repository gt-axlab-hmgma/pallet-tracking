from pathlib import Path

import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.yolo26 import YOLO26


def load_model(checkpoint: Path) -> tuple[YOLO26, dict[str, int | None], int]:
    """Rebuild the model a checkpoint was trained as (its saved config), then load the weights.

    Returns the model, where training stood (``epoch`` and ``steps``) and the input size the model
    was trained at. Older checkpoints hold only the weights and a step count, so ``epoch`` may be
    ``None``.
    """
    state = torch.load(checkpoint, map_location="cpu")
    saved = state.get("config", {}).get("model")
    if saved:
        cfg = ModelConfig(
            saved["depth_multiple"], saved["width_multiple"],
            num_classes=saved["num_classes"], reg_max=saved["reg_max"],
        )  # fmt: skip
        size = int(saved["input_size"])
    else:  # a checkpoint without a config: assume the default n model at 640
        cfg, size = SIZES["n"], 640
    model = YOLO26(cfg, input_size=size)
    model.load_state_dict(state["model"])
    return model, {"epoch": state.get("epoch"), "steps": state.get("steps")}, size
