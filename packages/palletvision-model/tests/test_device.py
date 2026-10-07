import pytest
import torch

from palletvision_model.device import resolve_device


def test_explicit_device_is_respected() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_returns_an_available_device() -> None:
    dev = resolve_device("auto")
    assert dev.type in {"cuda", "mps", "cpu"}
    torch.zeros(1, device=dev)


def test_unavailable_accelerator_raises() -> None:
    if torch.cuda.is_available():
        pytest.skip("cuda present")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
