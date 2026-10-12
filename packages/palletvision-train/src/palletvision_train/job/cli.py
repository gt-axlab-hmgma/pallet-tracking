"""``pv-job``: the whole overnight run on a fresh machine, and one folder of results to send back.

    pv-job check            what is missing (data, credentials, disk)? changes nothing
    pv-job fetch            download the missing public data (LOCO, SORDI scenes)
    pv-job run              check, fetch, train, score, plot and bundle
    pv-job run --smoke      the same in minutes (tiny limits): prove the machine first
    pv-job run --smoke --synthetic   no data and no download at all: proves the install only
    pv-job run --init-from X.pt      start from a pretrained checkpoint's weights
    pv-job bundle <run>     pack an existing run folder

The long steps run in child processes (the real ``pv-train``, ``pv-eval`` and ``pv-doctor``), so
a crash in one cannot take the job down, and training restarts from its last checkpoint.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from palletvision_model.device import resolve_device
from palletvision_train.job import fetch
from palletvision_train.job.bundle import build_bundle, save_config
from palletvision_train.job.run import (
    DATA_DIR,
    REPO,
    JobConfig,
    check_data,
    default_name,
    default_workers,
    fetch_missing,
    finish,
    keep_awake,
    load_data_config,
    refuse_overwrite,
    report_missing,
    run_doctor,
    run_evals,
    train_with_restarts,
)


def add_run_options(s: argparse.ArgumentParser, epochs: int) -> None:
    s.add_argument("--name", default=None, help="run folder name (default: host and time)")
    s.add_argument("--smoke", action="store_true", help="tiny limits, one epoch: minutes")
    s.add_argument("--epochs", type=int, default=epochs)
    s.add_argument("--batch-size", type=int, default=16, help="images per step, per GPU")
    s.add_argument("--workers", type=int, default=default_workers())
    s.add_argument("--gpus", default="1", help="1, a number, or 'all' (several CUDA GPUs)")
    s.add_argument("--retries", type=int, default=10)
    s.add_argument("--no-fetch", action="store_true", help="never download (e.g. offline nodes)")
    s.add_argument("--resume", action="store_true", help="continue the run in --name's folder")
    s.add_argument("--runs-dir", type=Path, default=REPO / "runs")


def share(text: str) -> float | None:
    """``--share``: ``uniform`` (no share) or a number between 0 and 1."""
    if text == "uniform":
        return None
    value = float(text)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError(f"share must be in [0, 1) or 'uniform', got {text}")
    return value


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pv-job", description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "fetch", "run"):
        sub.add_parser(name)
    run = sub.choices["run"]
    add_run_options(run, 150)
    run.add_argument("--synthetic", action="store_true", help="generated data only: no download")
    run.add_argument(
        "--share", type=share, default=None,
        help="'uniform' (default): real and synthetic images are one pool; or a number, such as "
        "0.5: every real image each epoch plus enough synthetic ones to make that share",
    )  # fmt: skip
    run.add_argument(
        "--epoch-images", type=int, default=5640, help="images per epoch when --share is uniform"
    )
    run.add_argument(
        "--full-eval", action="store_true", help="score every synthetic val/test image"
    )
    run.add_argument("--init-from", type=Path, help="start from a pretrained checkpoint's weights")
    b = sub.add_parser("bundle", help="pack an existing run folder")
    b.add_argument("run_dir", type=Path)
    return p.parse_args(argv)


def gpu_count(setting: str) -> int:
    available = torch.cuda.device_count()
    wanted = available if setting == "all" else int(setting)
    if wanted > max(available, 1):
        raise SystemExit(f"--gpus {wanted} but only {available} CUDA GPUs are visible")
    return max(wanted, 1)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    argv_used = list(sys.argv[1:] if argv is None else argv)
    if args.command == "bundle":
        save_config(args.run_dir)
        print(build_bundle(args.run_dir, args.run_dir.parent / f"results-{args.run_dir.name}"))
        return

    synthetic = bool(getattr(args, "synthetic", False))
    roots: list[str] = list(load_data_config()["roots"])
    loco_missing, missing = False, list[str]()
    if args.command in ("check", "fetch") or not synthetic:
        loco_missing, missing = check_data(roots, DATA_DIR)
    if args.command == "check":
        problems = report_missing(loco_missing, missing, DATA_DIR)
        print("\n".join(problems) if problems else "data present; ready to run")
        raise SystemExit(1 if problems else 0)
    if args.command == "fetch":
        fetch_missing(loco_missing, missing, DATA_DIR)
        return

    name = args.name or default_name()
    if args.smoke and not synthetic:
        # a smoke run needs one small scene: 02 (4 GB), else the first listed
        roots = [r for r in roots if fetch.scene_number(r) == "02"] or roots[:1]
        loco_missing, missing = check_data(roots, DATA_DIR)
    job = JobConfig(
        name=name, run_dir=args.runs_dir / f"job-{name}", smoke=args.smoke, synthetic=synthetic,
        epochs=args.epochs, batch_size=args.batch_size, workers=args.workers,
        share=getattr(args, "share", None), epoch_images=getattr(args, "epoch_images", 5640),
        gpus=gpu_count(args.gpus), retries=args.retries,
        full_eval=getattr(args, "full_eval", False), resume=args.resume, roots=roots,
        init_from=getattr(args, "init_from", None),
    )  # fmt: skip
    refuse_overwrite(job)

    if not synthetic:
        problems = report_missing(loco_missing, missing, DATA_DIR)
        if (loco_missing or missing) and args.no_fetch:
            raise SystemExit("data is missing and --no-fetch was given:\n" + "\n".join(problems))
        if problems:
            raise SystemExit("cannot start:\n" + "\n".join(problems))
        fetch_missing(loco_missing, missing, DATA_DIR)
    job.run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    print(f"pv-job {name}: device {device}, {job.gpus} GPU(s), folder {job.run_dir}")
    started = time.time()
    with keep_awake():
        run_doctor(job)
        attempts = train_with_restarts(job)
        run_evals(job)
    bundle = finish(job, started, attempts, argv_used)
    print(f"\nDone. Send back: {bundle}")
