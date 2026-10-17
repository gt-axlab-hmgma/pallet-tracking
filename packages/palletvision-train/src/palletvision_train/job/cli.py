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
from palletvision_train.job.bundle import build_bundle, save_config
from palletvision_train.job.remote import fetch
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
    measure_speed,
    plan_budget,
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


def epoch_images(text: str) -> int | None:
    """``--epoch-images``: a number, or ``all`` for a full pass over the whole pool."""
    if text == "all":
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"epoch images must be positive or 'all', got {text}")
    return value


def scenes(text: str) -> list[str]:
    """``--scenes 01,02``: scene numbers, as two digits."""
    numbers = [part.strip().zfill(2) for part in text.split(",") if part.strip()]
    if not numbers or not all(n.isdigit() for n in numbers):
        raise argparse.ArgumentTypeError(
            f"scenes must be numbers like 02 or 01,02,04, got {text!r}"
        )
    return numbers


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pv-job", description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "fetch", "run"):
        s = sub.add_parser(name)
        s.add_argument(
            "--scenes", type=scenes, default=None,
            help="comma-separated SORDI scene numbers to use (default: all in the config), e.g. 02",
        )  # fmt: skip
    run = sub.choices["run"]
    add_run_options(run, 150)
    run.add_argument("--synthetic", action="store_true", help="generated data only: no download")
    run.add_argument(
        "--share", type=share, default=None,
        help="'uniform' (default): real and synthetic images are one pool; or a number, such as "
        "0.5: every real image each epoch plus enough synthetic ones to make that share",
    )  # fmt: skip
    run.add_argument(
        "--epoch-images", type=epoch_images, default=5640,
        help="images per epoch when --share is uniform: a number, or 'all' for the whole pool",
    )  # fmt: skip
    run.add_argument("--amp", action="store_true", help="mixed precision (faster on a GPU)")
    run.add_argument(
        "--set", dest="extra", action="append", default=[], metavar="KEY=VALUE",
        help="any other pv-train override, applied last (repeatable): --set train.lr=0.005",
    )  # fmt: skip
    run.add_argument(
        "--budget-hours", type=float, default=None,
        help="size the run to this many hours: every epoch is the whole train set, and the train "
        "set is as big as the measured speed allows for --epochs epochs",
    )  # fmt: skip
    run.add_argument(
        "--train-images", type=int, default=None,
        help="total train images, real and synthetic alike (every epoch is all of them)",
    )  # fmt: skip
    run.add_argument(
        "--epoch-fraction", type=float, default=None,
        help="with --budget-hours or --train-images: each epoch is a fresh random draw of this "
        "fraction (0 to 1) of the train set instead of all of it",
    )  # fmt: skip
    run.add_argument(
        "--speed-margin", type=float, default=0.6,
        help="fraction of the measured step speed assumed for the whole run (default 0.6)",
    )  # fmt: skip
    run.add_argument(
        "--full-eval", action="store_true", help="score every synthetic val/test image"
    )
    run.add_argument(
        "--pool-images", type=int, default=None,
        help="keep a fixed random subsample of this many synthetic train images (all scenes)",
    )  # fmt: skip
    run.add_argument("--init-from", type=Path, help="start from a pretrained checkpoint's weights")
    b = sub.add_parser("bundle", help="pack an existing run folder")
    b.add_argument("run_dir", type=Path)
    return p.parse_args(argv)


def size_run(job: JobConfig, args: argparse.Namespace) -> None:
    """``--budget-hours`` or ``--train-images``: every epoch is the whole train set, whose size is
    set here (LOCO's train images plus a random subsample of SORDI's, real and synthetic alike)."""
    budget, total = getattr(args, "budget_hours", None), getattr(args, "train_images", None)
    if budget is None and total is None:
        return
    if job.synthetic or job.smoke:
        raise SystemExit(
            "--budget-hours and --train-images are for real runs, not --smoke or --synthetic"
        )
    from palletvision_train.datasets.coco import loco_reader

    n_real = len(loco_reader("train"))
    if budget is not None:
        speed = measure_speed(job.batch_size, job.amp)
        job.plan = plan_budget(budget, job.epochs, speed, args.speed_margin, n_real)
    else:
        assert total is not None
        job.plan = {"train_images": total, "real_images": n_real, "pool_images": total - n_real,
                    "epochs": job.epochs, "views": total * job.epochs}  # fmt: skip
        if job.plan["pool_images"] < 1:
            raise SystemExit(f"--train-images must exceed LOCO's {n_real} train images")
    fraction = getattr(args, "epoch_fraction", None)
    if fraction is not None and not 0 < fraction <= 1:
        raise SystemExit(f"--epoch-fraction must be in (0, 1], got {fraction}")
    plan = job.plan
    per_epoch = None if fraction in (None, 1) else round(plan["train_images"] * fraction)
    plan["epoch_images"] = per_epoch or plan["train_images"]
    plan["views"] = plan["epoch_images"] * plan["epochs"]
    job.pool_images, job.epoch_images, job.share = plan["pool_images"], per_epoch, None
    what = "the whole train set" if per_epoch is None else f"a random {plan['epoch_images']:,}"
    print(
        f"plan: {plan['epochs']} epochs, each {what} of the {plan['train_images']:,} train "
        f"images ({plan['real_images']:,} LOCO + {plan['pool_images']:,} SORDI, drawn alike) = "
        f"{plan['views']:,} image views"
        + (f"; {plan['measured_images_per_second']} img/s measured x {plan['margin']} margin "
           f"for {plan['budget_hours']} h" if "budget_hours" in plan else "")
    )  # fmt: skip


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
    if getattr(args, "scenes", None):
        listed = {fetch.scene_number(r): r for r in roots}
        unknown = [n for n in args.scenes if n not in listed]
        if unknown:
            raise SystemExit(
                f"scenes {unknown} are not in configs/data/loco_sordi.yaml: {sorted(listed)}"
            )
        roots = [listed[n] for n in args.scenes]
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
        amp=getattr(args, "amp", False), extra=list(getattr(args, "extra", [])),
        gpus=gpu_count(args.gpus), retries=args.retries,
        full_eval=getattr(args, "full_eval", False), resume=args.resume, roots=roots,
        init_from=getattr(args, "init_from", None), pool_images=getattr(args, "pool_images", None),
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
    size_run(job, args)

    device = resolve_device("auto")
    print(f"pv-job {name}: device {device}, {job.gpus} GPU(s), folder {job.run_dir}")
    started = time.time()
    with keep_awake():
        run_doctor(job)
        attempts = train_with_restarts(job)
        run_evals(job)
    bundle = finish(job, started, attempts, argv_used)
    print(f"\nDone. Send back: {bundle}")
