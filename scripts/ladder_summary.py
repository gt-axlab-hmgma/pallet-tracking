"""Print one table of the arms of an experiment ladder: python scripts/ladder_summary.py [tag]."""

import json
import sys
from pathlib import Path

tag = sys.argv[1] if len(sys.argv) > 1 else "ladder"
runs = Path(__file__).resolve().parents[1] / "runs"
ORDER = ["U150", "D1", "D2", "D3", "L90"]


def score(scores: dict[str, dict[str, float]], key: str, metric: str = "map50_95") -> str:
    value = scores.get(key, {}).get(metric)
    return "-" if value is None else f"{value:.3f}"


rows = []
for arm in ORDER:
    path = runs / f"results-{tag}-{arm}" / "manifest.json"
    if not path.exists():
        rows.append(f"| {arm} | not finished | | | | | | | |")
        continue
    m = json.loads(path.read_text())
    overrides = {o.split("=", 1)[0]: o.split("=", 1)[1] for o in m["overrides"] if "=" in o}
    epochs, per_epoch = int(overrides["train.epochs"]), int(overrides["data.sordi.epoch_images"])
    pool = overrides.get("data.sordi.subsample") or overrides["data.sordi.roots"]
    s = m["scores"]
    rows.append(
        f"| {arm} | {pool} | {epochs} | {epochs * per_epoch:,} | {score(s, 'loco_val')} | "
        f"{score(s, 'loco_test')} | {score(s, 'sordi_val')} | {score(s, 'sordi_test')} | "
        f"{m['wall_seconds'] / 3600:.1f} |"
    )

print("| arm | pool | epochs | views | LOCO val | LOCO test | SORDI val | SORDI test | hours |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print("\nmAP50-95 of best.pt (picked on LOCO val). SORDI scores use a fixed 5,000-image draw.")
