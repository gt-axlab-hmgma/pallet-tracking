import numpy as np
from torch.utils.data import Dataset

from palletvision_train.datasets.data import Sample


class SyntheticBoxes(Dataset[Sample]):
    """Noise images with one bright rectangle. For smoke tests and CI; needs no download.

    Items match the real reader: a ``[size, size, 3]`` uint8 image and ``[1, 5]`` labels
    (class 0, x1, y1, x2, y2), so they go through the same letterbox, collation and loss.
    Deterministic per index.
    """

    def __init__(self, n: int = 64, size: int = 640, seed: int = 0) -> None:
        self.n, self.size, self.seed = n, size, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        rng = np.random.default_rng(self.seed * 100003 + index)
        s = self.size
        w, h = (int(v) for v in rng.integers(s // 4, s // 2 + 1, 2))
        x1, y1 = int(rng.integers(0, s - w)), int(rng.integers(0, s - h))
        image = (rng.random((s, s, 3)) * 0.2 * 255).astype(np.uint8)
        image[y1 : y1 + h, x1 : x1 + w] = int(0.9 * 255)
        labels = np.array([[0, x1, y1, x1 + w, y1 + h]], dtype=np.float32)
        return image, labels
