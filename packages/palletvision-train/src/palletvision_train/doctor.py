"""``pv-doctor``: check a machine can train, and how fast.

Prints what PyTorch sees (version, CUDA build, GPU, memory, supported precisions), warns about
the usual install mistakes, then runs a few real training steps (the same ``train`` the real run
uses, on generated crowded images) with and without mixed precision and reports the speed and
peak memory. Run it first on any new machine: a laptop, a rented GPU or a cluster node.
"""

import argparse
import contextlib
import io
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from palletvision_model.config import SIZES
from palletvision_model.device import resolve_device
from palletvision_model.yolo26 import YOLO26
from palletvision_train.data import LetterboxedDataset, Sample
from palletvision_train.trainer import train


class CrowdedBoxes(Dataset[Sample]):
    """Noise images with ``boxes`` random boxes each: as many objects as a dense LOCO image."""

    def __init__(self, n: int, size: int, boxes: int = 40) -> None:
        self.n, self.size, self.boxes = n, size, boxes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        rng = np.random.default_rng(index)
        image = rng.integers(0, 255, (self.size, self.size, 3), dtype=np.uint8)
        xy = rng.uniform(0, self.size * 0.8, (self.boxes, 2))
        wh = rng.uniform(self.size * 0.02, self.size * 0.2, (self.boxes, 2))
        labels = np.column_stack([np.zeros(self.boxes), xy, xy + wh]).astype(np.float32)
        return image, labels


@dataclass(frozen=True)
class Timing:
    name: str
    seconds_per_step: float
    images_per_second: float
    peak_memory_gb: float | None
    final_loss: float


def environment(device: torch.device) -> dict[str, str]:
    """What PyTorch can see, as printable facts."""
    facts = {"torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        facts |= {
            "cuda build": str(torch.version.cuda),
            "gpu": props.name,
            "compute capability": f"{major}.{minor}",
            "gpu memory": f"{props.total_memory / 2**30:.1f} GB",
            "bfloat16": str(torch.cuda.is_bf16_supported()),
        }
    elif device.type == "mps":
        facts["gpu"] = "Apple GPU (Metal)"
    return facts


def warnings_for(device: torch.device) -> list[str]:
    """The common ways a CUDA install silently ends up slow or broken."""
    found = []
    if device.type == "cpu":
        if torch.version.cuda is None:
            found.append(
                "PyTorch here is a CPU-only build, so any NVIDIA GPU is ignored. Install the CUDA "
                "build (see docs/cuda-setup.md)."
            )
        else:
            found.append("No usable GPU was found: training would run on the CPU.")
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            found.append(
                f"This GPU is {arch} but this PyTorch build only has {torch.cuda.get_arch_list()}. "
                "RTX 50-series cards need a build for CUDA 12.8 or newer."
            )
    return found


def time_training(
    device: torch.device, size: int, batch: int, steps: int, amp: bool, amp_dtype: str = "auto"
) -> Timing:
    """Seconds per step of the real training loop.

    One untimed run absorbs the one-time start-up costs (kernel compilation, autotuning); then a
    short and a longer run are timed and their difference gives the cost of the extra steps.
    """
    short = 2
    seconds: dict[int, float] = {}
    final = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for count in (short, short, short + steps):  # the first is the untimed warm-up
        folder = Path(tempfile.mkdtemp(prefix="pv-doctor-"))
        try:
            torch.manual_seed(0)
            model = YOLO26(SIZES["n"], input_size=size)
            data = LetterboxedDataset(CrowdedBoxes(batch * count, size), size)
            began = time.time()
            with contextlib.redirect_stdout(io.StringIO()):  # the trainer prints every epoch
                result = train(
                    model, data, device=str(device), epochs=1, batch_size=batch, lr=0.01,
                    optimizer_name="musgd", momentum=0.937, amp=amp, amp_dtype=amp_dtype,
                    max_steps=count, output_dir=folder,
                )  # fmt: skip
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds[count] = time.time() - began
            final = result.losses[-1]
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    per_step = (seconds[short + steps] - seconds[short]) / steps
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2**30
    elif device.type == "mps":
        peak = torch.mps.driver_allocated_memory() / 2**30
    name = "mixed precision" if amp else "float32"
    return Timing(name, per_step, batch / per_step, peak, final)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check this machine can train, and how fast.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="images per step (try 16 and 32)")
    parser.add_argument("--steps", type=int, default=8, help="timed steps per setting")
    parser.add_argument("--amp-dtype", default="auto", help="auto, bfloat16 or float16")
    parser.add_argument("--no-amp", action="store_true", help="time float32 only")
    args = parser.parse_args()

    device = resolve_device(args.device)
    for key, value in environment(device).items():
        print(f"{key:20s} {value}")
    for line in warnings_for(device):
        print(f"WARNING: {line}")
    print()

    modes = [False] if args.no_amp else [False, True]
    for amp in modes:
        try:
            t = time_training(device, args.size, args.batch, args.steps, amp, args.amp_dtype)
        except (RuntimeError, ValueError) as error:
            label = "mixed precision" if amp else "float32"
            print(f"{label:16s} FAILED: {error}")
            continue
        memory = "n/a" if t.peak_memory_gb is None else f"{t.peak_memory_gb:.1f} GB"
        print(
            f"{t.name:16s} {t.seconds_per_step:6.2f} s/step  {t.images_per_second:6.1f} img/s  "
            f"peak memory {memory}  loss {t.final_loss:.2f}"
        )
    print(
        "\nA 150-epoch run of 5,640 images an epoch is about 850,000 image-passes: divide by img/s."
    )


if __name__ == "__main__":
    main()
