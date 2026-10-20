"""Score checkpoints on SORDI scenes they never trained on (all splits, a fixed random draw).

uv run python scripts/eval_unseen_scenes.py weights/a.pt,runs/x/best.pt 08 03
"""

import sys
from pathlib import Path

import yaml

from palletvision_train.checkpoints.model import load_model
from palletvision_train.datasets.coco import REPO, CocoDetection
from palletvision_train.datasets.sordi import sordi_records, subsample
from palletvision_train.evaluation.evaluate import evaluate

cfg = yaml.safe_load((REPO / "configs/data/loco_sordi.yaml").read_text())["sordi"]
classes = yaml.safe_load((REPO / "contracts/classes.yaml").read_text())["stage1"]
N = 2000
for scene in sys.argv[2:]:
    root = [f"data/sordid-{scene}"]
    recs = []
    for sp in (
        "train",
        "val",
        "test",
    ):  # none of these scenes was trained on, so all splits are unseen
        recs += sordi_records(root, cfg["class_map"], classes, sp, cfg["fractions"])
    recs = subsample(recs, N, seed=0)
    for name in sys.argv[1].split(","):
        model, _, size = load_model(Path(name))
        model.eval()
        res = evaluate(
            model, CocoDetection(recs), size=size, device="auto", batch_size=16, num_workers=0
        )
        model.to("cpu")
        print(
            f"scene {scene} {name}: {len(recs)} images, "
            f"mAP50-95 {res.map50_95:.3f}, mAP50 {res.map50:.3f}",
            flush=True,
        )
