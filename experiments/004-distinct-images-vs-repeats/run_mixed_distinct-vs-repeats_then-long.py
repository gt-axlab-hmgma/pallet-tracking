"""004: one unattended night: two short tests, then a long run sized to end by a deadline.

    uv run python experiments/004-*/run_mixed_distinct-vs-repeats_then-long.py --deadline 09:30
    ... --dry-run     # prints the plan only

All runs start from the converted COCO weights (data/yolo26n_coco_converted.pt) and use the
settings found in experiment 003 (multiplier 1.0, per-step alpha from 0.3, EMA, mosaic on).
Real and synthetic images are drawn alike: a run's train set is all of LOCO train plus a seeded
random SORDI subsample, and every epoch is all of it.

1. Test A: 60,000 images x 1 epoch (no repeats).   Test B: 6,000 images x 10 epochs (10 repeats).
   Both are 60,000 image views. Only the LOCO test score (an unseen site) is compared.
2. The rule, fixed in advance: if the two differ by more than 0.01 mAP50-95 the better shape wins;
   otherwise more distinct images (A's shape) is used.
3. The long run takes the time left until the deadline: its views come from the speed measured in
   test B, less the overhead, with a safety factor. Winner A: 5 epochs of many images; winner B:
   20 epochs of fewer images.

Decisions are written to runs/night-decision.json. Each run also leaves runs/results-<name>/.
"""

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
WEIGHTS = "data/yolo26n_coco_converted.pt"
SCENES = "01,02,04,05,06"
LOCO_TRAIN = 2820  # LOCO train images that every run includes
POOL_LIMIT = 190_000  # train images: the SORDI train split of these scenes holds about 197,000
TIE = 0.01  # LOCO test mAP50-95 differences within this are a tie
SAFETY = 0.92  # of the measured speed, to leave room for scoring and slower epochs
DEFAULT_SPEED = 45.0  # images/s on the Mac, if the test cannot say
MIN_LONG_HOURS = 1.0  # not worth starting a long run with less than this


def parse_deadline(text: str, now: datetime) -> datetime:
    """The next time the clock reads HH:MM."""
    hour, minute = (int(part) for part in text.split(":"))
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return deadline if deadline > now else deadline + timedelta(days=1)


def decide_shape(a_score: float | None, b_score: float | None) -> tuple[str, str]:
    """``("A"|"B", reason)``: A is many distinct images, B is few images repeated."""
    if a_score is None and b_score is None:
        return "A", "neither test finished: default to more distinct images"
    if b_score is None:
        return "A", "test B did not finish: more distinct images"
    if a_score is None:
        return "B", "test A did not finish: repeats"
    if abs(a_score - b_score) <= TIE:
        return "A", f"tie ({a_score:.3f} vs {b_score:.3f}, within {TIE}): more distinct images"
    winner = "A" if a_score > b_score else "B"
    return winner, f"{winner} is better on the unseen site ({a_score:.3f} vs {b_score:.3f})"


def size_long_run(
    shape: str, seconds_left: float, speed: float, overhead_hours: float = 0.5
) -> dict[str, int | float] | None:
    """Epochs and train images for the time left, or ``None`` if there is too little."""
    train_hours = seconds_left / 3600 - overhead_hours
    if train_hours < MIN_LONG_HOURS:
        return None
    views = int(train_hours * 3600 * speed * SAFETY)
    epochs = 5 if shape == "A" else 20
    images = min(views // epochs, POOL_LIMIT)
    if images <= LOCO_TRAIN + 1000:
        return None
    return {"epochs": epochs, "train_images": images, "views": images * epochs,
            "hours": round(images * epochs / speed / 3600 + overhead_hours, 2)}  # fmt: skip


def common(epochs: int, images: int, name: str, decay: float, tau: int, warmup: float) -> list[str]:
    return [
        sys.executable, "-m", "palletvision_train.job", "run", "--name", name,
        "--scenes", SCENES, "--train-images", str(images), "--epochs", str(epochs),
        "--no-fetch", "--resume", "--init-from", WEIGHTS,
        "--set", "train.pretrained_lr_mult=1.0", "--set", "train.alpha_schedule=step",
        "--set", "train.alpha_init=0.3", "--set", "train.warmup_bias_lr=0.01",
        "--set", "train.ema=true", "--set", f"train.ema_decay={decay}",
        "--set", f"train.ema_tau={tau}",
        "--set", f"train.warmup_epochs={warmup}",
    ]  # fmt: skip


def test_command(arm: str, tag: str) -> list[str]:
    images, epochs = (60_000, 1) if arm == "A" else (6_000, 10)
    cmd = common(epochs, images, f"{tag}-{arm}", 0.99, 100, 0.1 if arm == "A" else 1.0)
    return cmd + ["--set", "data.augment.close_mosaic=0", "--set", "train.val_every=100"]


def long_command(shape: str, size: dict[str, int | float], tag: str) -> list[str]:
    epochs, images = int(size["epochs"]), int(size["train_images"])
    cmd = common(epochs, images, f"{tag}-long", 0.998, 500, round(0.05 * epochs, 2))
    mosaic_off = max(1, round(0.1 * epochs))
    return cmd + ["--set", f"data.augment.close_mosaic={mosaic_off}",
                  "--set", f"train.val_every={1 if shape == 'A' else 2}"]  # fmt: skip


def manifest(name: str) -> dict | None:  # type: ignore[type-arg]
    path = RUNS / f"results-{name}" / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def loco_test(name: str) -> float | None:
    data = manifest(name)
    return None if data is None else float(data["scores"]["loco_test"]["map50_95"])


def measured_speed(name: str, images: int) -> float:
    """Images per second from the steady epochs of a finished run (not the first or last)."""
    log = RUNS / f"job-{name}" / "metrics.jsonl"
    if not log.exists():
        return DEFAULT_SPEED
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    seconds = [r["seconds"] for r in rows if r["type"] == "epoch"][1:-1]
    return images / statistics.median(seconds) if seconds else DEFAULT_SPEED


def run(cmd: list[str], retries: int = 1) -> int:
    code = 1
    for attempt in range(retries + 1):
        print(f"\n$ {' '.join(cmd[2:])}", flush=True)
        code = subprocess.run(cmd, cwd=REPO, check=False).returncode
        if code == 0:
            return 0
        print(f"!!! exited with code {code} (attempt {attempt + 1} of {retries + 1})", flush=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deadline", default="09:30", help="HH:MM the long run must end by")
    parser.add_argument("--overhead-hours", type=float, default=0.5, help="start-up and scoring")
    parser.add_argument("--tag", default="night")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args()
    deadline = parse_deadline(args.deadline, datetime.now())
    log: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "deadline": deadline.isoformat(timespec="minutes"),
    }  # type: ignore[type-arg]
    print(f"now {datetime.now():%a %H:%M}, deadline {deadline:%a %H:%M}")

    for arm in ("A", "B"):
        cmd = test_command(arm, args.tag)
        if args.dry_run:
            print(f"\n[test {arm}] {' '.join(cmd[2:])}")
        else:
            log[f"test_{arm}_exit"] = run(cmd, retries=0)
    if args.dry_run:
        for shape in ("A", "B"):
            size = size_long_run(
                shape,
                (deadline - datetime.now()).total_seconds() - 3600,
                DEFAULT_SPEED,
                args.overhead_hours,
            )
            print(f"\n[long run if {shape} wins, after ~1 h of tests] {size}")
            if size:
                print(" ".join(long_command(shape, size, args.tag)[2:]))
        return

    shape, reason = decide_shape(loco_test(f"{args.tag}-A"), loco_test(f"{args.tag}-B"))
    speed = measured_speed(f"{args.tag}-B", 6000)
    seconds_left = (deadline - datetime.now()).total_seconds()
    size = size_long_run(shape, seconds_left, speed, args.overhead_hours)
    log |= {
        "shape": shape,
        "reason": reason,
        "speed_images_per_s": round(speed, 1),
        "long_run": size,
    }
    (RUNS / "night-decision.json").write_text(json.dumps(log, indent=2))
    print(f"\ndecision: {reason}; speed {speed:.1f} img/s; long run {size}", flush=True)
    if size is None:
        print("too little time left for a long run")
    else:
        code = run(long_command(shape, size, args.tag), retries=1)
        log["long_exit"] = code
        (RUNS / "night-decision.json").write_text(json.dumps(log, indent=2))
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_runs.py",
            *(f"{args.tag}-{n}" for n in ("A", "B", "long")),
        ],
        cwd=REPO,
        check=False,
    )
    print(f"finished {datetime.now():%a %H:%M}")


if __name__ == "__main__":
    main()
