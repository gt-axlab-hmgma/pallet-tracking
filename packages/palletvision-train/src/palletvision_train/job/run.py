"""The steps of a ``pv-job`` run: what to train, how to restart it, scoring, and the manifest."""

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from palletvision_train.job import fetch
from palletvision_train.job.bundle import (
    build_bundle,
    environment_report,
    epochs_done,
    git_info,
    save_config,
    summarise_evals,
)
from palletvision_train.plot import load_run, plot_run

REPO = Path(__file__).resolve().parents[5]
DATA_DIR = REPO / "data"  # where the trainer's configs look: moving data means a symlink


OUT_OF_MEMORY = ("out of memory", "OutOfMemoryError", "MPS backend out of memory")


@dataclass
class JobConfig:
    name: str
    run_dir: Path
    smoke: bool = False
    synthetic: bool = False
    epochs: int = 150
    batch_size: int = 16
    workers: int = 4
    share: float = 0.5
    gpus: int = 1
    retries: int = 10
    full_eval: bool = False
    resume: bool = False
    roots: list[str] = field(default_factory=list)  # SORDI scene folders, as in the config
    init_from: Path | None = None  # start from another checkpoint's weights (fine-tuning)


def default_workers(cpus: int | None = None) -> int:
    """About half the cores, between 2 and 8: the augmentation needs ~4 cores per 350 images/s."""
    return max(2, min(8, (cpus or os.cpu_count() or 4) // 2))


def default_name(now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    return f"{socket.gethostname().split('.')[0]}-{stamp}"


def train_overrides(job: JobConfig) -> list[str]:
    """The Hydra overrides of the training run."""
    common = [
        f"train.output_dir={job.run_dir}",
        "train.resume=auto",
        f"train.batch_size={job.batch_size}",
    ]
    if job.synthetic:
        return [
            "data=synthetic",
            "train.epochs=1",
            "train.max_steps=20",
            "train.val_every=0",
            *common,
        ]
    roots = "[" + ",".join(job.roots) + "]"
    base = [
        "data=loco_sordi", f"data.sordi.share={job.share}", f"data.sordi.roots={roots}",
        f"data.num_workers={job.workers}", *common,
    ]  # fmt: skip
    if job.init_from:
        base.append(f"train.init_from={job.init_from}")
    if job.smoke:
        return [
            *base, "data.limit=32", "data.sordi.limit=64", "data.val_limit=16",
            "data.sordi.val_images=16", "train.epochs=1", "train.val_every=1",
        ]  # fmt: skip
    return [*base, f"train.epochs={job.epochs}", "train.val_every=5"]


def train_command(job: JobConfig) -> list[str]:
    """The trainer, alone or under torchrun (one process per GPU; batch size is per GPU)."""
    if job.gpus > 1:
        launch = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
        launch += [f"--nproc_per_node={job.gpus}", "-m", "palletvision_train.cli"]
    else:
        launch = [sys.executable, "-m", "palletvision_train.cli"]
    return [*launch, *train_overrides(job)]


def eval_plan(job: JobConfig) -> list[tuple[str, str, int | None]]:
    """``(source, split, image limit)`` for each final scoring of ``best.pt``.

    Scoring every synthetic val/test image takes about 18 minutes each on a Mac, so by default the
    synthetic splits use a fixed 5,000-image draw (``--full-eval`` scores them all).
    """
    if job.synthetic:
        return []  # generated data has no validation or test split to score
    cap = 16 if job.smoke else None
    synthetic_cap = 16 if job.smoke else (None if job.full_eval else 5000)
    return [
        ("loco", "val", cap), ("loco", "test", cap),
        ("sordi", "val", synthetic_cap), ("sordi", "test", synthetic_cap),
    ]  # fmt: skip


def keep_awake_command(pid: int, system: str | None = None) -> list[str] | None:
    """A command that keeps the machine from sleeping while process ``pid`` lives (Mac only; Windows
    is handled in-process, and a Linux server does not sleep)."""
    if (system or sys.platform) == "darwin":
        return ["caffeinate", "-dims", "-w", str(pid)]
    return None


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Prevent sleep for the duration, without changing any system setting that outlives the job."""
    helper = None
    command = keep_awake_command(os.getpid())
    if command is not None:
        with contextlib.suppress(OSError):
            helper = subprocess.Popen(command)
    if sys.platform == "win32":
        import ctypes

        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: reset automatically when this process exits
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if helper is not None:
            helper.terminate()
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]


def refuse_overwrite(job: JobConfig) -> None:
    """A fresh run never starts in a folder that already holds a checkpoint."""
    if (job.run_dir / "last.pt").exists() and not job.resume:
        raise SystemExit(
            f"{job.run_dir} already has a run in it. Pass --resume to continue it, or --name for "
            "a new folder."
        )


def stream(command: Sequence[str], log: Path, env: dict[str, str] | None = None) -> int:
    """Run a command, echoing its output to the screen and appending it to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        out.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=REPO, env={**os.environ, **(env or {})},
        )  # fmt: skip
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            out.write(line)
            out.flush()
        return process.wait()


def log_says_out_of_memory(log: Path) -> bool:
    tail = log.read_text()[-5000:] if log.exists() else ""
    return any(text in tail for text in OUT_OF_MEMORY)


def train_with_restarts(
    job: JobConfig,
    run: Callable[[Sequence[str], Path], int] = stream,
    pause: float = 60.0,
) -> int:
    """Train until the trainer exits cleanly, restarting from ``last.pt`` after a crash.

    A crash before the first checkpoint caused by running out of memory halves the batch size (the
    trainer accumulates up to the nominal batch of 128, so the update is the same, only slower).
    Returns the number of attempts used.
    """
    log = job.run_dir / "train.log"
    for attempt in range(1, job.retries + 1):
        code = run(train_command(job), log)
        if code == 0:
            return attempt
        print(f"training exited with code {code} (attempt {attempt} of {job.retries})")
        if not (job.run_dir / "last.pt").exists() and log_says_out_of_memory(log):
            if job.batch_size <= 1:
                break
            job.batch_size //= 2
            print(f"out of memory before the first checkpoint: trying batch size {job.batch_size}")
        time.sleep(pause)
    raise SystemExit(f"training did not finish; see {log}. Run the same command again to resume.")


def load_data_config() -> dict[str, Any]:
    data = yaml.safe_load((REPO / "configs/data/loco_sordi.yaml").read_text())
    assert isinstance(data, dict)
    return data["sordi"]  # type: ignore[no-any-return]


def check_data(roots: Sequence[str], data_dir: Path = DATA_DIR) -> tuple[bool, list[str]]:
    """``(loco_missing, missing SORDI scene roots)`` for the data this run needs."""
    return not fetch.loco_present(data_dir), fetch.missing_scenes(roots, REPO)


def report_missing(loco_missing: bool, missing: Sequence[str], data_dir: Path) -> list[str]:
    """Problems that would stop a run, as readable lines (empty when ready to go)."""
    problems = []
    if loco_missing:
        problems.append("LOCO is missing or incomplete (data/loco)")
    problems += [
        f"SORDI scene {fetch.scene_number(r)} is missing or incomplete ({r})" for r in missing
    ]
    needed = fetch.gigabytes_needed(missing, loco_missing)
    if needed:
        free = shutil.disk_usage(data_dir if data_dir.exists() else REPO).free / 2**30
        if free < needed:
            problems.append(f"{needed:.0f} GB of free disk needed to fetch it, {free:.0f} GB free")
        if missing and not fetch.kaggle_credentials():
            problems.append(
                "no Kaggle credentials (kaggle.com/settings -> API token -> ~/.kaggle/kaggle.json)"
            )
    return problems


def fetch_missing(loco_missing: bool, missing: Sequence[str], data_dir: Path) -> None:
    if loco_missing:
        fetch.fetch_loco(data_dir)
        if not fetch.loco_present(data_dir):
            raise SystemExit(
                "LOCO downloaded but some annotated images are missing: delete data/loco"
            )
    for root in missing:
        print(f"downloading SORDI scene {fetch.scene_number(root)}")
        fetch.fetch_sordi(REPO / root)


def data_manifest(roots: Sequence[str], fractions: Sequence[float]) -> dict[str, Any]:
    """Image counts per SORDI scene and split, to show two machines used the same data."""
    scenes = {}
    for root in roots:
        folder = REPO / root
        if folder.exists():
            scenes[fetch.scene_number(root)] = fetch.sordi_split_counts(folder, fractions)
    return {"sordi": scenes, "fractions": list(fractions)}


def run_doctor(job: JobConfig) -> None:
    code = stream(
        [sys.executable, "-m", "palletvision_train.doctor", "--batch", "8", "--steps", "4"],
        job.run_dir / "doctor.txt",
    )
    if code != 0:
        print("pv-doctor did not finish (see doctor.txt); continuing")


def run_evals(job: JobConfig) -> None:
    checkpoint = next(
        (job.run_dir / n for n in ("best.pt", "last.pt") if (job.run_dir / n).exists()), None
    )
    if checkpoint is None:
        return
    for source, split, limit in eval_plan(job):
        out = job.run_dir / f"eval_{source}_{split}.json"
        command = [
            sys.executable, "-m", "palletvision_train.eval_cli", "--checkpoint", str(checkpoint),
            "--split", split, "--source", source, "--workers", str(job.workers), "--out", str(out),
        ]  # fmt: skip
        if limit is not None:
            command += ["--limit", str(limit)]
        if stream(command, job.run_dir / "eval.log") != 0:
            print(f"scoring {source} {split} failed (see eval.log); continuing")


def data_for_manifest(job: JobConfig, fractions: Sequence[float]) -> dict[str, Any]:
    return {} if job.synthetic else data_manifest(job.roots, fractions)


def write_manifest(job: JobConfig, started: float, attempts: int, argv: Sequence[str]) -> None:
    fractions = load_data_config().get("fractions", [0.7, 0.15, 0.15])
    manifest = {
        "name": job.name,
        "command": " ".join(argv),
        "smoke": job.smoke,
        "synthetic": job.synthetic,
        "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - started),
        "training_attempts": attempts,
        "epochs_done": epochs_done(job.run_dir),
        "overrides": train_overrides(job),
        "git": git_info(REPO),
        "data": data_for_manifest(job, fractions),
        "init_from": str(job.init_from) if job.init_from else None,
        "scores": summarise_evals(job.run_dir),
    }
    (job.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def finish(job: JobConfig, started: float, attempts: int, argv: Sequence[str]) -> Path:
    """Plot, write the environment and manifest, and bundle."""
    if (job.run_dir / "metrics.jsonl").exists():
        run = load_run(job.run_dir)
        if run.steps:
            plot_run(run, job.run_dir / "curves.png")
    (job.run_dir / "environment.json").write_text(json.dumps(environment_report(), indent=2))
    save_config(job.run_dir)
    write_manifest(job, started, attempts, argv)
    return build_bundle(job.run_dir, job.run_dir.parent / f"results-{job.name}")
