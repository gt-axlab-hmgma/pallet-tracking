"""``pv-convert``: turn an Ultralytics YOLO26 checkpoint into one ``train.init_from`` can load.

An Ultralytics ``.pt`` is a full Python pickle, and unpickling can run arbitrary code. This reads
it with a pickle reader that builds only tensors and plain containers: every other class in the
file (the Ultralytics model classes, anything hostile) becomes an inert stand-in, so nothing in
the file executes and Ultralytics is not needed. The tensors are then paired with our model's
state dict **by position**, which is only accepted when the counts and every shape agree.

The result is ``{"model": state_dict, "source": {...}}`` with the origin, licence and training
lineage recorded under ``source``, because that file's terms travel with its weights.

Verified for the YOLO26n COCO checkpoint: all 708 tensors map, and the converted model reproduces
its COCO validation mAP (``scripts/check_coco_transfer.py``).
"""

import argparse
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from palletvision_model.config import SIZES, ModelConfig
from palletvision_model.yolo26 import YOLO26


class Stub:
    """Stands in for any class in the file: it records what it was given and does nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args, self._kwargs = args, kwargs

    def __setstate__(self, state: Any) -> None:
        self._state = state

    def __call__(self, *args: Any, **kwargs: Any) -> "Stub":
        return Stub()


ALLOWED = {
    ("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"), ("builtins", "set"), ("builtins", "dict"),
    ("builtins", "list"), ("builtins", "tuple"), ("builtins", "frozenset"), ("torch", "Size"),
    ("torch", "float32"), ("torch", "float16"), ("torch", "bfloat16"), ("torch", "int64"),
}  # fmt: skip


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ALLOWED or (module == "torch" and name.endswith("Storage")):
            return super().find_class(module, name)
        return type(name, (Stub,), {"__module__": module})


class SafePickle:
    """The ``pickle_module`` handed to ``torch.load``."""

    Unpickler = SafeUnpickler
    UnpicklingError = pickle.UnpicklingError

    @staticmethod
    def load(file: Any, **kwargs: Any) -> Any:
        return SafeUnpickler(file, **kwargs).load()


def read_checkpoint(path: Path) -> dict[str, Any]:
    """The checkpoint as plain data plus ``Stub`` objects; nothing in it was executed."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a PyTorch zip checkpoint")
    loaded = torch.load(path, map_location="cpu", weights_only=False, pickle_module=SafePickle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not hold a checkpoint dictionary")
    return loaded


def tensors_of(module: Any, prefix: str = "") -> OrderedDict[str, torch.Tensor]:
    """Parameters and buffers of a stubbed module tree, in ``state_dict`` order."""
    found: OrderedDict[str, torch.Tensor] = OrderedDict()
    state = getattr(module, "_state", None)
    if not isinstance(state, dict):
        return found
    for kind in ("_parameters", "_buffers"):
        for name, tensor in (state.get(kind) or {}).items():
            if isinstance(tensor, torch.Tensor):
                found[prefix + name] = tensor
    for name, child in (state.get("_modules") or {}).items():
        if child is not None:
            found.update(tensors_of(child, f"{prefix}{name}."))
    return found


def plain(value: Any) -> Any:
    """Keep only numbers, strings, lists and dicts of them, so the result loads in safe mode."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return None


def source_facts(checkpoint: dict[str, Any], module: Any, source: Path) -> dict[str, Any]:
    state = getattr(module, "_state", {}) or {}
    names = state.get("names")
    args = checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {}
    metrics = checkpoint.get("train_metrics")
    return {
        "file": source.name,
        "license": plain(checkpoint.get("license")),
        "version": plain(checkpoint.get("version")),
        "date": plain(checkpoint.get("date")),
        "trained_from": plain((args or {}).get("model")),  # the lineage, e.g. an Objects365 start
        "classes": [str(v) for v in names.values()] if isinstance(names, dict) else None,
        "metrics": {k: v for k, v in (metrics or {}).items() if isinstance(v, float)},
    }


def convert(source: Path, size: str = "n") -> dict[str, Any]:
    """Our-format checkpoint from an Ultralytics one. Raises unless every tensor pairs up."""
    checkpoint = read_checkpoint(source)
    module = checkpoint.get("ema") or checkpoint.get("model")
    theirs = tensors_of(module)
    if not theirs:
        raise ValueError(f"{source}: no model tensors found")
    classes = (getattr(module, "_state", {}) or {}).get("nc")
    if not isinstance(classes, int):
        raise ValueError(f"{source}: the number of classes is not recorded in the file")
    cfg = SIZES[size]
    ours = YOLO26(
        ModelConfig(cfg.depth_multiple, cfg.width_multiple, num_classes=classes, reg_max=1),
        input_size=640,
    )
    mine = ours.state_dict()
    if len(mine) != len(theirs):
        raise ValueError(f"{source} has {len(theirs)} tensors, a '{size}' model has {len(mine)}")
    wrong = [
        (a, tuple(x.shape), b, tuple(y.shape))
        for (a, x), (b, y) in zip(mine.items(), theirs.items(), strict=True)
        if x.shape != y.shape
    ]
    if wrong:
        raise ValueError(f"{len(wrong)} tensors differ in shape, the first: {wrong[0]}")
    state = {
        ours_name: tensor.float() if tensor.is_floating_point() else tensor
        for ours_name, tensor in zip(mine, theirs.values(), strict=True)
    }
    return {"model": state, "source": source_facts(checkpoint, module, source)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an Ultralytics YOLO26 checkpoint.")
    parser.add_argument("source", type=Path, help="the Ultralytics .pt file")
    parser.add_argument("--out", type=Path, required=True, help="where to write ours")
    parser.add_argument("--size", default="n", choices=sorted(SIZES), help="model size (default n)")
    args = parser.parse_args()
    converted = convert(args.source, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.out)
    facts = converted["source"]
    print(f"wrote {args.out}: {len(converted['model'])} tensors mapped by position")
    print(f"source: {facts['file']}, version {facts['version']}, {facts['license']}")
    print(f"trained from: {facts['trained_from']}")
    print(
        "These weights carry the terms above (and those of the data in the line above): "
        "check them before anything built on them is shipped."
    )


if __name__ == "__main__":
    main()
