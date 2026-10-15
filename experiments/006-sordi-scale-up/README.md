# 006: scaling SORDI-only training to 80,000 images x 20 epochs

## Goal

Push the SORDI score as high as it goes, as a proof that high mAP is doable (LOCO no longer used). Then check how it does on SORDI scenes it never saw.

## Config

`run_sordi-only_musgd_freeze0_80k-x20ep.sh`: COCO init, 80,000 random SORDI images from scenes 01, 02, 04, 05, 06, 20 epochs (1.6 million views), MuSGD lr 0.01, freeze 0, EMA 0.999, warmup 1 epoch, mosaic off for the last 2, best picked on SORDI val. 10.2 h on the Mac (a slow stretch mid-night). Unseen-scene scoring: `uv run python scripts/eval_unseen_scenes.py <best.pt>,<best.pt> 08 03` (2,000 images per scene; scenes 03 and 08 were never in training).

## Result

| | mAP50-95 | mAP50 |
| --- | --- | --- |
| SORDI test (5,000, seen scenes) | 0.726 | 0.871 |
| SORDI val | 0.732 | 0.878 |
| Unseen scene 08 (night-long: 0.606 / 0.738) | 0.650 | 0.780 |
| Unseen scene 03 (night-long: 0.234 / 0.311) | 0.275 | 0.352 |
| LOCO test | 0.012 | 0.038 |

Validation rose at every scoring (0.61 at epoch 2 to 0.729 at epoch 20); no sign of overfitting on that curve.

## Conclusion

- Best SORDI score of all runs (previous 0.681 / 0.838), and better than the mixed model on unseen synthetic scenes by about 0.04.
- Scene 03 shows a model can lose most of its score on a new scene; report held-out scenes together, not the best one.
- On the real-footage clip it finds about 5 pallets a frame against 18 for `night-long` and misses the carried pallet in frame 100, as does `night-long`; the SORDI score does not carry to real footage. The mixed `night-long` stays the model in use.
