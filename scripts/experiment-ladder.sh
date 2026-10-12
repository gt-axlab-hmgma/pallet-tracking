#!/usr/bin/env bash
# One unattended run of five arms, one after another (about 13 hours on an Apple M5 Pro, 17 h budget):
#   U150  5 scenes, 150 epochs        the earlier overnight recipe under uniform sampling
#   D1    scene 02 only, 30 epochs    one environment (12,941 images)
#   D2    5 scenes subsampled to 12,941 images, 30 epochs   same image count, five environments
#   D3    5 scenes, 30 epochs
#   L90   5 scenes, 90 epochs
# Every epoch is 5,640 images drawn uniformly from LOCO train + the chosen SORDI scenes; augmentation on.
# Each arm writes its own runs/results-<tag>-<arm>/ bundle; a failed arm does not stop the next one.
#   bash scripts/experiment-ladder.sh [tag]        # tag defaults to "ladder"
#   DRY_RUN=1 bash scripts/experiment-ladder.sh    # print the commands only
# Safe to run again after an interruption: finished arms are re-scored quickly, unfinished ones resume.
# Keep the Mac plugged in and the lid open (pv-job keeps it awake while it trains).
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-ladder}"
S5="01,02,04,05,06"

arm() {
  local name="$1"; shift
  echo "=== $(date '+%a %H:%M') arm $name: $*"
  if [ -n "${DRY_RUN:-}" ]; then echo "bash scripts/pallet-job.sh run --name $TAG-$name --resume $*"; return; fi
  bash scripts/pallet-job.sh run --name "$TAG-$name" --resume "$@" \
    || echo "!!! arm $name failed (see runs/job-$TAG-$name/train.log); continuing with the next arm"
}

arm U150 --scenes "$S5" --epochs 150
arm D1 --scenes 02 --epochs 30
arm D2 --scenes "$S5" --pool-images 12941 --epochs 30
arm D3 --scenes "$S5" --epochs 30
arm L90 --scenes "$S5" --epochs 90

echo "=== $(date '+%a %H:%M') all arms done"
uv run python scripts/ladder_summary.py "$TAG"
