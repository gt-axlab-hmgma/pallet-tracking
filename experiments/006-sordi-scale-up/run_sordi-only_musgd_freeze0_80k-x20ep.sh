#!/usr/bin/env bash
# 006: one long SORDI-only run (COCO init, MuSGD, freeze 0, 80,000 images x 20 epochs).
# Result: SORDI test 0.871 mAP50 / 0.726 mAP50-95; unseen scene 08 0.780 / 0.650; LOCO test 0.012.
#
# Start from the COCO weights, train on a fixed random 80,000 of the SORDI train pool of scenes
# 01, 02, 04, 05, 06 (about 197,000 images; LOCO is not trained on), 20 epochs, every tensor
# trainable (freeze 0), MuSGD lr 0.01. The run's best.pt is picked on SORDI val. A smoke run with the same
# options goes first and the script stops if it fails.
#
#   bash experiments/006-sordi-scale-up/run_sordi-only_musgd_freeze0_80k-x20ep.sh [tag]        (needs data/yolo26n_coco_converted.pt and the data)
#
# Sizing: 20 x 80,000 = 1.60 million image views at the 44-47 images/s measured on the Mac
# (experiment night long: 47.5), about 10 h of training plus about an hour of scoring.
# Closing the lid sleeps the Mac: run `sudo pmset -a disablesleep 1` first, and 0 afterwards.
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG="${1:-sordi-big}"
[ -f data/yolo26n_coco_converted.pt ] || { echo "data/yolo26n_coco_converted.pt is missing"; exit 1; }
SCENES="01,02,04,05,06"
IMAGES=80000  # SORDI train images (a fixed random draw); --train-images counts LOCO's 2,820 too
COMMON=(--scenes "$SCENES" --no-fetch --init-from data/yolo26n_coco_converted.pt
  --set data.real_train=false --set train.select_on=sordi --set train.freeze=0
  --set train.optimizer=musgd --set train.lr=0.01 --set train.pretrained_lr_mult=1.0 --set train.alpha_schedule=step --set train.alpha_init=0.3
  --set train.warmup_bias_lr=0.01
  --set train.ema=true --set train.ema_decay=0.999 --set train.ema_tau=1000)

echo "=== $(date '+%a %H:%M') smoke run with the same options"
uv run pv-job run --smoke --name "$TAG-smoke-$(date +%H%M%S)" "${COMMON[@]}" || { echo "smoke run failed: not starting"; exit 1; }

echo "=== $(date '+%a %H:%M') the long run"
uv run pv-job run --name "$TAG" --resume --epochs 20 --train-images $((IMAGES + 2820)) "${COMMON[@]}" \
  --set train.warmup_epochs=1 --set data.augment.close_mosaic=2 --set train.val_every=2
echo "=== $(date '+%a %H:%M') done"
uv run python scripts/compare_runs.py "$TAG"
