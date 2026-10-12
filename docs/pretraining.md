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

## How to decide whether pretraining helps

From-scratch pallet training is the baseline (LOCO test, later HMGMA test). Pretrain on an allowed dataset, fine-tune with the same recipe, and compare **test** scores, not training loss. If the licence question stays open, the lowest-risk path is no third-party pretraining and more HMGMA frames.

## Not done

- A COCO reader and fetcher (the next step if COCO is cleared), with an optional per-image licence filter.
- A pretraining run of any length.
