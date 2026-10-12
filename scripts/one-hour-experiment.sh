#!/usr/bin/env bash
# A one-hour experiment for a Mac (about 55 minutes on an M5 Pro): at equal compute, does it matter
# how many DIFFERENT images the model sees? This is the trade the big run makes (a small dataset
# repeated for many full epochs) against the alternative (a bigger dataset for fewer epochs).
#
# Three arms with the same views (about 34,000) and augmentation (mosaic on throughout), and warm-up
# of the same fraction (11%) of training; every epoch is the whole
# train set (all 2,820 LOCO + a random SORDI subsample, real and synthetic drawn alike):
#   R9  3,820 images x 9 epochs    each image seen 9 times   (74% of the images are LOCO)
#   R3  11,000 images x 3 epochs   each image seen 3 times   (26% LOCO)
#   R1  34,000 images x 1 epoch    each image seen once      (8% LOCO)
# Scored on the real LOCO val and test sites (and SORDI val and test). The earlier baseline peaked
# after about 25 repeats of 2,820 images, so 9 repeats is where repetition may start to cost.
# Expect small differences at this length: one seed per arm, under about 0.01 mAP50-95 is noise.
#
#   bash scripts/one-hour-experiment.sh [tag]      (needs the data on disk; never downloads)
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-hour}"
COMMON=(--scenes "02,04,06" --no-fetch --resume --set train.val_every=100 --set data.augment.close_mosaic=0)

arm() {
  local name="$1"; shift
  echo "=== $(date '+%H:%M') arm $name: $*"
  bash scripts/pallet-job.sh run --name "$TAG-$name" "${COMMON[@]}" "$@" \
    || echo "!!! arm $name failed (see runs/job-$TAG-$name/train.log)"
}

arm R9 --train-images 3820 --epochs 9 --set train.warmup_epochs=1
arm R3 --train-images 11000 --epochs 3 --set train.warmup_epochs=0.33
arm R1 --train-images 34000 --epochs 1 --set train.warmup_epochs=0.11

echo "=== $(date '+%H:%M') done"
uv run python scripts/compare_runs.py "$TAG-R9" "$TAG-R3" "$TAG-R1"
