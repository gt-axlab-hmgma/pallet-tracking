# Weights

`pallet_yolo26n_coco-init_loco-sordi-mixed.pt` is the model in use: the `night-long` run of experiment 004
(`runs/exp-004-distinct-images-vs-repeats/mixed_coco_33k-x20ep_USED/`). Weights are git-ignored.

- Start: the COCO YOLO26n weights converted by `pv-convert` (`data/yolo26n_coco_converted.pt`).
- Train: LOCO (2,820) + SORDI scenes 01, 02, 04, 05, 06 (30,399), drawn alike, 20 epochs, MuSGD lr 0.01, lr multiple 1.0, EMA 0.998.
- Scores: LOCO test 0.196 mAP50-95 (0.510 mAP50); SORDI test 0.681 (0.838); unseen SORDI scene 08 0.606 (0.738), scene 03 0.234 (0.311). Nothing measured on the factory camera yet.
- Deploy with `pv-infer --conf` near the run's best-F1 confidence (see `eval_*.json` in the run's results folder).
- Licence: derived from the COCO YOLO26n weights (AGPL-3.0, Objects365 lineage). Settle with the PI before shipping; see `experiments/003-coco-init/README.md`.

Findings and the other experiments: `experiments/FINDINGS.md`.
