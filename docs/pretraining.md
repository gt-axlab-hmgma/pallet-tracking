# Pretraining: what we can use, and how to start from it

Issue #74. The YOLO26 report pretrains on **Objects365 (v1)** for 150 epochs at batch 128 and then fine-tunes (`docs/YOLO_26.pdf`, Table S2). It states no GPU count or wall-clock time. Whether the same helps pallets is untested here. This page records the licence findings (October 2026) and what the repo supports.

## What the repo supports

`train.init_from=<checkpoint>` (or `pv-job run --init-from <checkpoint>`) starts a run from another model's weights. It copies every tensor whose shape matches and leaves the rest at their fresh initialisation, so a model pretrained with another number of classes transfers its backbone and neck and gets a fresh classification head. It is weights only: the new run has its own schedule from epoch 0 (unlike `train.resume`). Scoring with more than one class works (`evaluate(..., categories=N)`). A reader for a many-class pretraining dataset is not built.

## Licence findings

I am not a lawyer; this is not legal advice. Check with the PI or Georgia Tech's research contracts office before any weights trained on third-party data go into a sponsor's product.

| Dataset | Finding |
| --- | --- |
| **Objects365** | The [official page](https://www.objects365.org/download.html) says **academic purposes only** and requires registration. Annotations are CC BY 4.0, but the images are not the Consortium's: Flickr terms apply, the files list non-commercial licences, and redistribution is not allowed. Other listings disagree ([HyperAI](https://hyper.ai/en/datasets/17725): non-commercial; [TIB](https://service.tib.eu/ldmservice/dataset/objects365): no licence). The Biendata/BAAI terms were not found. **Treat as research-only unless written permission is obtained.** Not used here. |
| **COCO** | Annotations are CC BY 4.0 (commercial use allowed with attribution). Images come from Flickr under mixed per-image licences that are recorded in the annotation files; the Consortium holds no image copyright. A [2021 arXiv study](https://arxiv.org/pdf/2111.02374) of public datasets, COCO included, found a high risk of licence problems for commercial software, even for pretrained models; no court ruling or official statement settles it. **A gray area, not cleared.** Filtering to the permissive per-image licences (for example CC BY) would reduce the risk without removing the uncertainty. |
| **LOCO, SORDI** | LOCO is CC0 and SORDI is MIT: what we train on today. |

## Results so far

Starting from the YOLO26n COCO weights roughly doubles the real-image score at equal data and training, and a separate lower learning rate for the pretrained tensors does not help (multiplier 1.0 is best): see `experiments/003-coco-init/README.md`. The weights are converted with `pv-convert` (read without running their code, checked against COCO accuracy by `scripts/check_coco_transfer.py`); the file is AGPL-3.0 and descends from an Objects365 start, so its terms need the PI's decision before deployment.

## Fine-tuning recipe (single phase)

Starting points to tune on LOCO val, not tested values. With a checkpoint pretrained on another dataset (its class head will restart fresh):

```bash
bash scripts/pallet-job.sh run --init-from <pretrained.pt> --scenes "01,02,04,05,06" \
  --train-images 48000 --epochs 4 --amp --name finetune \
  --set train.pretrained_lr_mult=1.0 --set train.warmup_bias_lr=0.01 \
  --set train.alpha_schedule=step --set train.alpha_init=0.3 \
  --set train.ema=true --set train.ema_decay=0.995 --set train.ema_tau=200 \
  --set data.augment.close_mosaic=1
```

- **Data:** all 2,820 LOCO train images plus about 45,000 SORDI images, every epoch the whole set. At equal training, more distinct images beat repeats (the one-hour experiment: 34,000 distinct images scored about 4x 3,820 repeated nine times on synthetic data), so use many distinct images and few passes. About 190,000 image views.
- **Learning rate:** measured in experiment 003: a lower rate for the copied tensors did not help. `pretrained_lr_mult` 0.1, 0.2, 0.5 and 1.0 scored 0.098, 0.126, 0.152 and 0.169 on LOCO test, so use 1.0 (no separate rate). The freeze-then-unfreeze protection of [LP-FT](https://arxiv.org/abs/2202.10054v1) was not needed at this scale.
- **Progressive loss:** `step` so a few epochs still anneal; start at 0.3 because a pretrained model has already passed the early phase.
- **EMA** with a short horizon (`0.995`, about 200 updates), since a fine-tune has only about 1,500 updates.
- **Mosaic** off for the last epoch only (the default 8 would switch it off for the whole run).
- **The check that matters:** the same budget from scratch against this, scored on LOCO val and test. Pretraining can fail to help when augmentation is strong and labels plentiful ([Zoph et al.](https://arxiv.org/pdf/2006.06882)).

LoRA is not used: it saves memory and storage for models with billions of parameters; ours has 2.4M and trains in a few GB, and its low-rank limit would restrict how far the model can adapt.

## How to decide whether pretraining helps

From-scratch pallet training is the baseline (LOCO test, later HMGMA test). Pretrain on an allowed dataset, fine-tune with the same recipe, and compare **test** scores, not training loss. If the licence question stays open, the lowest-risk path is no third-party pretraining and more HMGMA frames.

## Not done

- A COCO reader and fetcher (the next step if COCO is cleared), with an optional per-image licence filter.
- A pretraining run of any length.
