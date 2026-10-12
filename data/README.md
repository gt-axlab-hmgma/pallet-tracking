# Data

Everything here except this file is git-ignored.

- `data/loco/` LOCO dataset (COCO format).
- `data/sordid-02/` SORDI Industrial Scene 02 (`images/`, `labels/json/`), synthetic, MIT licence, from `sordi-ai/industrial-scene-02-dataset` on Kaggle (downloaded 2026-10). More scenes go in sibling folders (`data/sordid-NN/`) and are listed in `configs/data/loco_sordi.yaml`; every image is split train/val/test 70/15/15 by a hash of its name, so adding a scene does not move images already assigned.
- `data/videos/` short test clips for `pv-infer`.

HMGMA footage is under NDA: keep it on lab-controlled storage, never in this repo or on cloud services.
