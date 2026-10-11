"""Draw a run's training curves from its ``metrics.jsonl``.

``pv-plot runs/stage1`` writes ``runs/stage1/curves.png`` with three panels:
1. the loss at every step (faint) and smoothed, for the blended loss and each branch,
2. the loss of each epoch next to the validation metric, where one was computed,
3. the learning rate and the Progressive Loss weight alpha.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw to a file; no window needed
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class Run:
    steps: list[dict[str, Any]]  # one entry per optimiser step, in order
    epochs: list[dict[str, Any]]  # one entry per finished epoch


def load_run(path: Path) -> Run:
    """Read ``metrics.jsonl`` (or the run folder that holds it)."""
    if path.is_dir():
        path = path / "metrics.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return Run(
        steps=[line for line in lines if line["type"] == "step"],
        epochs=[line for line in lines if line["type"] == "epoch"],
    )


def smooth(values: list[float], window: int) -> list[float]:
    """Mean of the last ``window`` values at each point (fewer at the start)."""
    out: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def plot_run(run: Run, out: Path, window: int = 50) -> Path:
    """Draw the three panels to ``out`` (a PNG) and return it."""
    figure, (loss, epoch, rate) = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)

    steps = [row["step"] for row in run.steps]
    for key, label in (
        ("loss", "blended"),
        ("total_one2many", "one-to-many"),
        ("total_one2one", "one-to-one"),
    ):
        values = [row[key] for row in run.steps if key in row]
        if not values:
            continue
        line = loss.plot(steps[: len(values)], values, alpha=0.2, linewidth=0.8)[0]
        loss.plot(steps[: len(values)], smooth(values, window), label=label, color=line.get_color())
    loss.set(xlabel="step", ylabel="loss", title=f"loss per step (smoothed over {window} steps)")
    loss.set_yscale("log")
    loss.legend()

    epoch_loss = [(row["epoch"] + 1, row["loss"]) for row in run.epochs]
    if epoch_loss:
        epoch.plot(*zip(*epoch_loss, strict=True), label="epoch loss")
    scored = [(row["epoch"] + 1, row["metric"]) for row in run.epochs if row["metric"] is not None]
    epoch.set(xlabel="epoch", ylabel="mean loss", title="epoch loss and validation metric")
    if scored:
        right = epoch.twinx()
        right.plot(*zip(*scored, strict=True), color="tab:red", marker="o", label="validation")
        right.set_ylabel("validation metric")
        right.set_ylim(bottom=0)

    rate.plot(steps, [row["lr"] for row in run.steps], label="learning rate")
    rate.set(xlabel="step", ylabel="learning rate", title="learning rate and one-to-many weight")
    weight = rate.twinx()
    weight.plot(steps, [row["alpha"] for row in run.steps], color="tab:green", label="alpha")
    weight.set_ylabel("alpha")

    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the curves of a training run.")
    parser.add_argument("run", type=Path, help="a run folder or its metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="default: curves.png next to it")
    parser.add_argument("--window", type=int, default=50, help="steps in the moving average")
    args = parser.parse_args()
    folder = args.run if args.run.is_dir() else args.run.parent
    run = load_run(args.run)
    if not run.steps:
        raise SystemExit(f"no steps logged in {args.run}")
    print(plot_run(run, args.out or folder / "curves.png", args.window))
