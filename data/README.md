# Data

Everything here except this file is git-ignored.

- `data/loco/` LOCO dataset (COCO format).
- `data/sordid-02/` SORDI Industrial Scene 02 (`images/`, `labels/json/`), synthetic, MIT licence, from `sordi-ai/industrial-scene-02-dataset` on Kaggle (downloaded 2026-10). More scenes go in sibling folders (`data/sordid-NN/`) and are listed in `configs/data/loco_sordi.yaml`; every image is split train/val/test 70/15/15 by a hash of its name, so adding a scene does not move images already assigned.
- `data/videos/` short test clips for `pv-infer`.
## Public pallet detection datasets (download manually; not committed)

| Folder | Source | Notes |
|---|---|---|
| `data/roboflow-pallet-research/` | https://universe.roboflow.com/pallet-research/warehouse_pallet_detection-bobms | Single class "pallet" |
| `data/roboflow-sarah-t/` | https://universe.roboflow.com/sarah-t/pallet-detection-ith6b-rssqe | Pallet + box |
| `data/roboflow-test-29t7t/` | https://universe.roboflow.com/test-29t7t/pallet-detection-ith6b-ytig9 | CC BY 4.0 |
| `data/roboflow-warehousepallets/` | https://universe.roboflow.com/capstone-gckbn/warehousepallets | 131 images, CC BY 4.0 |
| `data/roboflow-warehouse-9cgfc/` | https://universe.roboflow.com/warehouse-cvotz/warehouse-9cgfc | Packages with pallet |

Download each as COCO from the Roboflow page and extract into its folder. Credit the original authors per CC BY 4.0.
HMGMA footage is under NDA: keep it on lab-controlled storage, never in this repo or on cloud services.
