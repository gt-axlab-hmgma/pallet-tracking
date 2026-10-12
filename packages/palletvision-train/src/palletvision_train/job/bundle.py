"""Pack the results of a run into one folder (and a zip) to send back.

The bundle is built from an allowlist of file names, never by copying a whole folder, so no image,
video or dataset file can end up in it by accident (HMGMA footage is under NDA). The checkpoints
are model weights, not data.
"""

import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

from palletvision_model.device import resolve_device
from palletvision_train.doctor import environment, warnings_for

FILES = (
    "best.pt", "last.pt", "metrics.jsonl", "curves.png", "train.log", "eval.log", "doctor.txt",
    "environment.json", "config.json", "manifest.json",
)  # fmt: skip
PATTERNS = ("eval_*.json",)

README = """\
This folder is everything worth keeping from one training run. It holds no images or videos.

best.pt            the model with the best validation score (use this one)
last.pt            the model at the end of training (and the point to resume from)
metrics.jsonl      one line per training step and per epoch (loss, learning rate, scores)
curves.png         the same, drawn (run `uv run pv-plot <folder>` to redraw it)
eval_*.json        best.pt scored on LOCO val/test and SORDI val/test (mAP and F1)
train.log          everything the trainer printed, including any restarts
eval.log, doctor.txt  the scoring output and the machine check (speed, memory, warnings)
environment.json   OS, Python, PyTorch, GPU, CPU count, memory
config.json        the exact settings the model was trained with
manifest.json      what ran: git commit, command, times, the data it used (image counts per
                   scene and split, to check two machines used the same data), the scores
"""


def run_git(*args: str, cwd: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_info(repo: Path) -> dict[str, Any]:
    """The commit this code is at and whether it has uncommitted changes. A source zip with no
    git history reports ``unknown`` (set ``PALLETVISION_COMMIT`` to record one)."""
    commit = run_git("rev-parse", "HEAD", cwd=repo)
    if commit is None:
        return {"commit": os.environ.get("PALLETVISION_COMMIT", "unknown (no git history)")}
    status = run_git("status", "--porcelain", cwd=repo) or ""
    return {"commit": commit, "uncommitted_changes": bool(status)}


def memory_gb() -> float | None:
    """Physical memory, where the platform can say."""
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def environment_report(device_name: str = "auto") -> dict[str, Any]:
    device = resolve_device(device_name)
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb(),
        **environment(device),
        "warnings": warnings_for(device),
    }


def summarise_evals(run_dir: Path) -> dict[str, dict[str, float]]:
    """The headline numbers of every ``eval_*.json`` in the run folder."""
    summary: dict[str, dict[str, float]] = {}
    for path in sorted(run_dir.glob("eval_*.json")):
        data = json.loads(path.read_text())
        summary[path.stem.removeprefix("eval_")] = {
            key: round(data[key], 4)
            for key in ("images", "map50_95", "map50", "best_f1", "best_conf")
            if key in data
        }
    return summary


def epochs_done(run_dir: Path) -> int:
    log = run_dir / "metrics.jsonl"
    if not log.exists():
        return 0
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return sum(1 for row in rows if row["type"] == "epoch")


def save_config(run_dir: Path) -> None:
    """Write the settings the checkpoint was trained with as ``config.json``."""
    for name in ("last.pt", "best.pt"):
        checkpoint = run_dir / name
        if checkpoint.exists():
            config = torch.load(checkpoint, map_location="cpu").get("config")
            if config is not None:
                (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
                return


def build_bundle(run_dir: Path, destination: Path) -> Path:
    """Copy the allowlisted files that exist into ``destination`` (a folder) and zip it.

    Missing files are fine (a smoke run has no ``best.pt`` if nothing was scored). Returns the zip.
    """
    destination.mkdir(parents=True, exist_ok=True)
    wanted = [run_dir / name for name in FILES]
    for pattern in PATTERNS:
        wanted += sorted(run_dir.glob(pattern))
    included = []
    for source in wanted:
        if source.is_file():
            target = destination / source.name
            target.write_bytes(source.read_bytes())
            included.append(target)
    readme = destination / "README.txt"
    readme.write_text(README)
    included.append(readme)
    archive = destination.parent / f"{destination.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in included:
            zf.write(path, f"{destination.name}/{path.name}")
    return archive
