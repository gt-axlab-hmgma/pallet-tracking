#!/usr/bin/env bash
# Stage 1: does starting from the COCO weights help at all? Two runs with the same data and the same
# 120,000 image views (about 1 hour each on an M5 Pro, 2 hours in all):
#   S  from scratch (the usual recipe)
#   C  from data/yolo26n_coco_converted.pt, pretrained tensors at 0.2x the learning rate,
#      the progressive-loss weight starting at 0.3, bias warm-up 0.01
# Data (the same for both): LOCO train + a random SORDI subsample, 20,000 train images in all, drawn alike;
# every epoch is all of them, 6 epochs. Both use the per-step alpha schedule and a short EMA.
# Pass check: C beats S by more than about 0.01 mAP50-95 on LOCO val (single seed: the noise is about 0.005).
#
#   bash experiments/003-coco-init/run_scratch-vs-coco_20k-x6ep.sh [tag]      (needs the data and the converted weights on disk)
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG="${1:-stage1}"
[ -f data/yolo26n_coco_converted.pt ] || { echo "data/yolo26n_coco_converted.pt is missing"; exit 1; }
COMMON=(--scenes "01,02,04,05,06" --train-images 20000 --epochs 6 --no-fetch --resume
  --set train.warmup_epochs=0.4 --set data.augment.close_mosaic=1 --set train.val_every=3
  --set train.alpha_schedule=step --set train.ema=true --set train.ema_decay=0.995 --set train.ema_tau=200)

arm() {
  local name="$1"; shift
  echo "=== $(date '+%a %H:%M') arm $name"
  if [ -n "${DRY_RUN:-}" ]; then echo "bash scripts/pallet-job.sh run --name $TAG-$name ${COMMON[*]} $*"; return; fi
  bash scripts/pallet-job.sh run --name "$TAG-$name" "${COMMON[@]}" "$@" \
    || echo "!!! arm $name failed (see runs/job-$TAG-$name/train.log)"
}

arm S
arm C --init-from data/yolo26n_coco_converted.pt --set train.pretrained_lr_mult=0.2 \
  --set train.warmup_bias_lr=0.01 --set train.alpha_init=0.3

echo "=== $(date '+%a %H:%M') done"
uv run python scripts/compare_runs.py "$TAG-S" "$TAG-C"
