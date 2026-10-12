"""Print one table of finished pv-job runs: python scripts/compare_runs.py name [name ...]

A name is the --name given to pv-job (results are in runs/results-<name>/). Scores are the best
checkpoint's mAP50-95 (and mAP50 in brackets); SORDI scores use a fixed 5,000-image draw.
"""

import json
import sys
from pathlib import Path

runs = Path(__file__).resolve().parents[1] / "runs"
COLUMNS = [
    "run", "train images", "per epoch", "epochs", "views", "LOCO val", "LOCO test", "SORDI val",
    "SORDI test", "hours",
]  # fmt: skip


def cell(scores: dict[str, dict[str, float]], key: str) -> str:
    s = scores.get(key)
    return "-" if not s else f"{s['map50_95']:.3f} ({s['map50']:.3f})"


print("| " + " | ".join(COLUMNS) + " |")
print("|" + "---|" * len(COLUMNS))
for name in sys.argv[1:]:
    path = runs / f"results-{name}" / "manifest.json"
    if not path.exists():
        print(f"| {name} | not finished |" + " |" * (len(COLUMNS) - 2))
        continue
    m = json.loads(path.read_text())
    o = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in m["overrides"] if "=" in a}
    plan = m.get("plan") or {}
    epochs, per_epoch = int(o["train.epochs"]), o["data.sordi.epoch_images"]
    views = plan.get("views") or (epochs * int(per_epoch) if per_epoch.isdigit() else 0)
    images = plan.get("train_images") or "pool"
    per_epoch = plan.get("epoch_images") or per_epoch
    s = m["scores"]
    print(
        f"| {name} | {images} | {per_epoch} | {epochs} | {views:,} | {cell(s, 'loco_val')} | "
        f"{cell(s, 'loco_test')} | {cell(s, 'sordi_val')} | {cell(s, 'sordi_test')} | "
        f"{m['wall_seconds'] / 3600:.2f} |"
    )
