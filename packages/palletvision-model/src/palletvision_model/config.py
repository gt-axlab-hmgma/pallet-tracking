"""Model size configuration: how nominal layer widths and repeats scale per model size.

The layer tables in the implementation guide are written for a "nominal" model. A size
(n, s, ...) is just three numbers that shrink or grow that table. Nothing here touches torch.
"""

import math
from dataclasses import dataclass


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round ``x`` up to the next multiple of ``divisor`` (GPUs like channel counts that are)."""
    return math.ceil(x / divisor) * divisor


@dataclass(frozen=True)
class ModelConfig:
    depth_multiple: float  # scales how many times a block repeats
    width_multiple: float  # scales how many channels a layer has
    max_channels: int = 1024  # nominal widths are capped here before scaling
    num_classes: int = 1
    reg_max: int = 1  # 1 means direct box regression, no DFL

    def channels(self, nominal: int) -> int:
        """Output channels of a layer whose nominal width is ``nominal``."""
        return make_divisible(min(nominal, self.max_channels) * self.width_multiple)

    def repeats(self, nominal: int) -> int:
        """How many times a block repeats. A block listed once is never scaled."""
        if nominal <= 1:
            return 1
        return max(round(nominal * self.depth_multiple), 1)


SIZES: dict[str, ModelConfig] = {
    "n": ModelConfig(depth_multiple=0.50, width_multiple=0.25),
    "s": ModelConfig(depth_multiple=0.50, width_multiple=0.50),
}
