# 003: starting from the YOLO26n COCO weights (stage 1 and stage 2a)

## Goal

Does starting from the released YOLO26n COCO weights give a better pallet detector than starting from scratch at the same data and training, and how fast should the transferred tensors train? Issues #81, #83, #86.

## The weights (read this first)

`data/yolo26n.pt` is the official YOLO26n COCO checkpoint (Ultralytics 8.3.222, 2025-12-15, **licence field AGPL-3.0**). Its recorded training arguments show it was started from `yolo26n-objv1-150.pt`, an **Objects365** pretraining, then trained 245 epochs on COCO (MuSGD, batch 128). So it carries AGPL and the academic-use-only terms of Objects365 (see `docs/pretraining.md`). Fine as research evidence; **ask the PI before any deployment is built on it.**

`pv-convert data/yolo26n.pt --out data/yolo26n_coco_converted.pt` reads it without running its code (it is a Python pickle), maps its 708 tensors to ours by position (counts and every shape must agree) and records the licence and lineage in the output. `uv run python scripts/check_coco_transfer.py` shows the converted weights reproduce COCO accuracy in our model: mAP50-95 0.410 on the first 1,000 COCO val images (the checkpoint records 0.401 on all 5,000), against 0.000 for random weights. This also confirms our reimplementation is faithful. The converted file is git-ignored (`data/`): recreate it with the command above.

With 1 class, `train.init_from` copies 606 tensors (backbone, neck, box branches) and leaves the 102 class-branch tensors fresh.

## Config

Mac (Apple M5 Pro, MPS, float32), single seed (0), about 55 minutes per run, scripts `scripts/stage1-coco-vs-scratch.sh` and `scripts/stage2-lr-multiple.sh` (tables with `scripts/compare_runs.py`). Common to every run:

- Data: LOCO train (2,820) plus a seeded random SORDI subsample (17,180 images from scenes 01, 02, 04, 05, 06): 20,000 train images, real and synthetic drawn alike (`data.sordi.share=null`); every epoch is all of them, 6 epochs, 120,000 image views.
- Optimiser MuSGD lr 0.01, nominal batch 128 (16 x 8), warmup 0.4 epoch, linear decay, augmentation as the report, mosaic off in the last epoch, per-step progressive-loss weight (`train.alpha_schedule=step`), EMA (decay 0.995, tau 200), `best.pt` picked on LOCO val, scored every 3 epochs.
- Scoring: full LOCO val (sub-set 1, 1,075 images) and LOCO test (sub-set 4, 1,202 images, an unseen site); SORDI val and test as a fixed 5,000-image draw each (these draw from all configured scenes, so they include scenes the run never trained on).
- From scratch (S): alpha starts at 0.8. COCO-initialised (C, M*): `--init-from data/yolo26n_coco_converted.pt`, alpha starts at 0.3, bias warm-up 0.01, and `train.pretrained_lr_mult` as in the table (the fresh class branch always keeps lr 0.01).

## Result

mAP50-95 of `best.pt` (mAP50 in brackets):

| Run | lr multiple | LOCO val | LOCO test | SORDI val | SORDI test | F1 on LOCO test |
| --- | --- | --- | --- | --- | --- | --- |
| stage1-S, scratch | n/a | 0.102 (0.270) | 0.076 (0.233) | 0.413 | 0.408 | 0.325 |
| stage2-M0.1 | 0.1 | 0.163 (0.365) | 0.098 (0.276) | 0.512 | 0.505 | 0.353 |
| stage1-C | 0.2 | 0.189 (0.418) | 0.126 (0.348) | 0.552 | 0.547 | 0.411 |
| stage2-M0.5 | 0.5 | 0.216 (0.479) | 0.152 (0.413) | 0.596 | 0.592 | 0.462 |
| **stage2-M1.0** | **1.0** | **0.223 (0.490)** | **0.169 (0.448)** | **0.616** | **0.610** | **0.493** |

For reference, the earlier from-scratch overnight run (846,000 views, share 0.5, no EMA) scored 0.231 on LOCO val and 0.187 on LOCO test on this split (selected on the old validation set, so not fully clean).

- **COCO initialisation about doubles the real-image score at equal training** (LOCO test 0.076 to 0.126 at 0.2x, to 0.169 at 1.0x), and reaches about 90% of the big scratch run's test score (0.169 against 0.187) with 14% of its views.
- **A separate, lower learning rate for the transferred tensors does not help here:** the score rises with the multiple and flattens (0.5 to 1.0 adds 0.007 on val, 0.017 on test). An ordinary full fine-tune is at least as good. The 0.2 starting guess was too cautious.
- The unseen-site score is below validation in every run (the new-site generalisation problem is not removed by pretraining).

## Caveats

- One seed per run (noise about 0.005): the direction is solid, the exact gaps are not. The 0.5 to 1.0 val difference is within noise.
- C and the M runs also use alpha 0.3 / bias warm-up 0.01 / their multiple, S uses the scratch recipe: this is each arm with its own sensible recipe, not a single-variable test of the weights alone.
- SORDI scores are on easier synthetic images (and partly unseen scenes); judge by LOCO test.
- About 120,000 views is early in training; loss was still falling.
- Mixed precision, CUDA and multi-GPU have never been run on real hardware.

## Conclusion and next steps

Use the COCO weights with multiplier 1.0 (no separate rate). Not yet done:

1. **Ablate the two guesses** at multiplier 1.0: EMA on/off and alpha start 0.3 against 0.8 (about 45 minutes each on the Mac).
2. **Data size at equal views** (about 12k, 48k, 120k distinct images; the one-hour experiment says distinct images matter).
3. **The long run** on the 5060 laptop: `bash scripts/pallet-job.sh run --budget-hours 14 --amp --init-from data/yolo26n_coco_converted.pt --epochs 12 --scenes "01,02,04,05,06" --set train.pretrained_lr_mult=1.0 --set train.alpha_schedule=step --set train.alpha_init=0.3 --set train.ema=true --set train.ema_decay=0.999 --set train.ema_tau=1000 --set data.augment.close_mosaic=2 --set train.warmup_epochs=0.5 --set train.warmup_bias_lr=0.01 --name coco-long`, with a matched from-scratch run as the control. At about 100 images/s this is roughly the whole SORDI train pool for 12 epochs; read the plan line it prints, and run `run --smoke --amp` first (the first CUDA run ever). Judge it on LOCO test against 0.169 (this experiment) and 0.187 (scratch overnight).
4. Settle the weights' licence question with the PI before any shipped model starts from them.
