import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Map a config value (auto, cpu, cuda, mps) to a usable torch device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps requested but not available")
    return device
