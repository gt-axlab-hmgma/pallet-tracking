import torch
from torch.utils.data import Dataset


class SyntheticBoxes(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Noise images with one bright rectangle and its xyxy pixel box. Deterministic per index."""

    def __init__(self, n: int = 64, size: int = 640, seed: int = 0) -> None:
        self.n, self.size, self.seed = n, size, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100003 + index)
        s = self.size
        w = int(torch.randint(s // 4, s // 2 + 1, (1,), generator=g))
        h = int(torch.randint(s // 4, s // 2 + 1, (1,), generator=g))
        x1 = int(torch.randint(0, s - w, (1,), generator=g))
        y1 = int(torch.randint(0, s - h, (1,), generator=g))
        image = torch.rand(3, s, s, generator=g) * 0.2
        image[:, y1 : y1 + h, x1 : x1 + w] = 0.9
        return image, torch.tensor([x1, y1, x1 + w, y1 + h], dtype=torch.float32)
