#!/usr/bin/env bash
# Stage 2a: how fast should the COCO-pretrained tensors train? One factor changes, the learning-rate
# multiple of the transferred tensors (the fresh class branch keeps lr 0.01); everything else is
# exactly the COCO arm of stage 1 (scripts/stage1-coco-vs-scratch.sh, run "stage1-C" is the 0.2 point):
# the same 20,000 train images (the SORDI subsample is seeded, so it is the same images), 6 epochs,
# 120,000 views, alpha starting at 0.3, EMA, bias warm-up 0.01, seed 0.
#
#   M0.1  0.1x   M0.5  0.5x   M1.0  1.0x (no separate rate: an ordinary full fine-tune)
# About 55 minutes per arm (2 h 45 min in all). Single seed: a gap under about 0.01 mAP50-95 is noise.
#
#   UV_OFFLINE=1 bash scripts/stage2-lr-multiple.sh [tag]     (needs the data and the converted weights)
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-stage2}"
[ -f data/yolo26n_coco_converted.pt ] || { echo "data/yolo26n_coco_converted.pt is missing"; exit 1; }
COMMON=(--scenes "01,02,04,05,06" --train-images 20000 --epochs 6 --no-fetch --resume
  --set train.warmup_epochs=0.4 --set data.augment.close_mosaic=1 --set train.val_every=3
  --set train.alpha_schedule=step --set train.ema=true --set train.ema_decay=0.995 --set train.ema_tau=200
  --init-from data/yolo26n_coco_converted.pt --set train.warmup_bias_lr=0.01 --set train.alpha_init=0.3)

arm() {
  local name="$1" mult="$2"
  echo "=== $(date '+%a %H:%M') arm $name (multiplier $mult)"
  if [ -n "${DRY_RUN:-}" ]; then echo "bash scripts/pallet-job.sh run --name $TAG-$name ${COMMON[*]} --set train.pretrained_lr_mult=$mult"; return; fi
  bash scripts/pallet-job.sh run --name "$TAG-$name" "${COMMON[@]}" --set "train.pretrained_lr_mult=$mult" \
    || echo "!!! arm $name failed (see runs/job-$TAG-$name/train.log)"
}

arm M0.1 0.1
arm M0.5 0.5
arm M1.0 1.0

echo "=== $(date '+%a %H:%M') done"
uv run python scripts/compare_runs.py stage1-S stage1-C "$TAG-M0.1" "$TAG-M0.5" "$TAG-M1.0"
