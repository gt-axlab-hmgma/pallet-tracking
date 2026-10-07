import torch
from torch import nn


class TinyDetector(nn.Module):
    """Placeholder detector that satisfies contracts/model_io.yaml.

    Exists so training, export, benchmarking and inference can be exercised end to end
    before the real YOLO26 modules are written. It is not meant to be accurate.
    """

    def __init__(self, input_size: int = 640, num_queries: int = 300) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_queries = num_queries
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(16, num_queries * 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.features(x)).view(x.shape[0], self.num_queries, 5).sigmoid()
        boxes = raw[..., :4] * self.input_size
        x1, y1 = boxes[..., 0], boxes[..., 1]
        x2 = torch.maximum(boxes[..., 2], x1)
        y2 = torch.maximum(boxes[..., 3], y1)
        score = raw[..., 4]
        cls = torch.zeros_like(score)
        return torch.stack([x1, y1, x2, y2, score, cls], dim=-1)
