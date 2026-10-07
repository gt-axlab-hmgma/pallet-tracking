import torch

import palletvision_model


def test_import_and_tensor_op() -> None:
    assert palletvision_model.__doc__
    assert torch.zeros(1, 3, 8, 8).shape == (1, 3, 8, 8)
