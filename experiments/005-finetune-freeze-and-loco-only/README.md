# 005: freezing the backbone, and LOCO-only against SORDI-only

## Goal

Does the Ultralytics small-dataset recipe (AdamW 0.001, mosaic 0.5, patience 20, freeze 10) help here, and does a model trained on one dataset work on the other?

## Config

`run_finetune_adamw_freeze-10-vs-0_loco-only-vs-sordi-only.sh` (commands reconstructed from the logged configs). COCO init, lr multiple 1.0, 50 epochs, best checkpoint picked on LOCO val (LOCO-only) or SORDI val (SORDI-only). `freeze=10` freezes layers 0-9 (backbone up to before the C2PSA block); `freeze=23` would train only the head (unit-tested, not run).

## Result

| Run | Training images | LOCO test | SORDI test |
| --- | --- | --- | --- |
| loco-only, freeze 10 | 2,820 LOCO | 0.150 (0.400) | 0.062 (0.151) |
| sordi-only, freeze 10 | 2,820 SORDI | 0.012 (0.027) | 0.546 (0.732) |
| sordi-only, freeze 0 | 2,820 SORDI | 0.013 (0.039) | 0.595 (0.773) |

## Conclusion

- Unfreezing the backbone is better on synthetic data (+0.05 mAP50-95).
- Neither dataset transfers to the other (0.012 on LOCO, 0.062 on SORDI), so a model trained on one will not do for the other; train on both for real-image scores (004).
- No reason found to leave MuSGD for AdamW; the freeze 10 vs 0 runs both used AdamW, so the optimiser was not tested on its own.
