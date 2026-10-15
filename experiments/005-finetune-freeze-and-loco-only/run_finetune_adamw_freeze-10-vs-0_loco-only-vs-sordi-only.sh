#!/usr/bin/env bash
# 005: fine-tuning the COCO weights on a small set, following the Ultralytics small-dataset advice
# (AdamW lr 0.001, mosaic 0.5, patience 20, freeze 10 = backbone layers 0-9). Three runs, 50 epochs:
#   loco-only_freeze10     all 2,820 LOCO train images           -> LOCO test 0.150, SORDI test 0.062
#   sordi-only_freeze10    2,820 SORDI images, no LOCO           -> SORDI test 0.546, LOCO test 0.012
#   sordi-only_freeze0     the same, every tensor trainable      -> SORDI test 0.595, LOCO test 0.013
# Reconstructed from the configs logged at the top of runs/<name>.log. Run from the repo root.
set -uo pipefail
cd "$(dirname "$0")/../.."
COMMON=(train.init_from=data/yolo26n_coco_converted.pt train.pretrained_lr_mult=1.0
  train.optimizer=adamw train.lr=0.001 train.epochs=50 train.patience=20 train.val_every=1
  data.augment.mosaic=0.5)
SORDI=(data=loco_sordi data.real_train=false data.sordi.share=null data.sordi.epoch_images=null
  data.sordi.subsample=2820 train.select_on=sordi
  'data.sordi.roots=[data/sordid-01,data/sordid-02,data/sordid-04,data/sordid-05,data/sordid-06]')

uv run pv-train data=loco "${COMMON[@]}" train.freeze=10 train.output_dir=runs/loco-only_adamw_freeze10
uv run pv-train "${SORDI[@]}" "${COMMON[@]}" train.freeze=10 train.output_dir=runs/sordi-only_adamw_freeze10
uv run pv-train "${SORDI[@]}" "${COMMON[@]}" train.freeze=0 train.output_dir=runs/sordi-only_adamw_freeze0
