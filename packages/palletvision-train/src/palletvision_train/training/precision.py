"""How a training step uses reduced precision, and the speed settings of the GPU backend.

Mixed precision runs the model's forward and backward in 16 bits, which is much faster on
modern GPUs, while the weights, the loss and the optimiser stay in float32. Two 16-bit formats:
- ``bfloat16`` has float32's range, so it needs no extra care (Ampere, Ada, Hopper, Blackwell),
- ``float16`` has a small range: gradients can underflow, so a ``GradScaler`` multiplies the loss
  first and the optimiser step is skipped when a gradient overflows (CUDA only here).
"""

from dataclasses import dataclass

import torch

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class AmpPlan:
    autocast: bool  # run the forward pass in reduced precision
    dtype: torch.dtype  # which reduced precision (float32 when ``autocast`` is off)
    scaler: bool  # loss scaling is needed (float16 on CUDA)


def plan_amp(
    device_type: str, amp: bool, amp_dtype: str = "auto", bf16_supported: bool | None = None
) -> AmpPlan:
    """Decide how to run the forward pass. ``amp_dtype`` is ``auto``, ``float16`` or ``bfloat16``.

    ``auto`` is bfloat16 on a CUDA card that supports it and float16 otherwise (and on MPS).
    The CPU only does bfloat16. ``bf16_supported`` overrides the CUDA query (for tests).
    """
    if not amp:
        return AmpPlan(False, torch.float32, False)
    if amp_dtype != "auto" and amp_dtype not in DTYPES:
        raise ValueError(f"amp_dtype is '{amp_dtype}': use auto, float16 or bfloat16")
    if device_type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"mixed precision is not available on '{device_type}'")
    if device_type == "cuda" and bf16_supported is None:
        bf16_supported = torch.cuda.is_bf16_supported()
    if amp_dtype == "auto":
        if device_type == "cpu":
            amp_dtype = "bfloat16"
        elif device_type == "cuda":
            amp_dtype = "bfloat16" if bf16_supported else "float16"
        else:
            amp_dtype = "float16"
    if device_type == "cpu" and amp_dtype != "bfloat16":
        raise ValueError("on the CPU mixed precision is bfloat16 only")
    if device_type == "cuda" and amp_dtype == "bfloat16" and not bf16_supported:
        raise ValueError("this GPU does not support bfloat16: use float16")
    return AmpPlan(True, DTYPES[amp_dtype], device_type == "cuda" and amp_dtype == "float16")


def configure_backend(device: torch.device) -> None:
    """Speed settings for NVIDIA GPUs; other devices are left alone.

    The input size is fixed, so cuDNN may time its convolution algorithms once and keep the
    fastest. TF32 lets float32 matrix multiplies use the tensor cores (about 10 bits of mantissa
    instead of 23), which is the usual training default on Ampere and newer.
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
