import torch

from palletvision_model.tiny import TinyDetector


def test_output_follows_model_io_contract() -> None:
    model = TinyDetector(input_size=64)
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 300, 6)  # x1, y1, x2, y2, score, class
    assert out[..., :4].min() >= 0 and out[..., :4].max() <= 64
    assert out[..., 4].min() >= 0 and out[..., 4].max() <= 1
