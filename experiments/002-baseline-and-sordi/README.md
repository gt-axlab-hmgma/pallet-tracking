# 002: the first LOCO baseline, and what SORDI synthetic data adds

## Goal

How well does the paper's training recipe work on LOCO with no augmentation, and does mixing in SORDI synthetic scenes help on real held-out sites? This covers guide steps E1 and E2 (issues #43 and #34).

## Config

Common to all runs: `yolo26n` from scratch, seed 0, MuSGD (lr 0.01, momentum 0.937, weight decay 1e-3 at batch 128, warmup 1 epoch, linear decay to 1%), batch 16 accumulated to 128, 640 px, no augmentation, no EMA, float32 on an Apple M5 Pro (MPS). Validation is the real LOCO validation split (sub-sets 1 and 4, 2,277 images), scored every 5 epochs.

```bash
uv run pv-train data=loco train.val_every=5                                          # baseline, 150 epochs
uv run pv-train data=loco train.epochs=30 train.val_every=5 train.output_dir=runs/loco_30
uv run pv-train data=loco_sordi data.sordi.roots='[data/sordid-02]' train.epochs=30 train.val_every=5 train.output_dir=runs/loco_sordi_30
uv run pv-train data=loco_sordi data.sordi.share=0.5 train.epochs=30 train.val_every=5 train.output_dir=runs/loco_sordi_all_30
```

(The 30% run used `data.sordi.roots` with scene 02 only, which was the default then.)

## Data

- LOCO train: 2,820 images from sub-sets 2, 3 and 5. Pallets and stillages are one class, `pallet`.
- SORDI scenes 01, 02, 04, 05 and 06 (`sordi-ai/industrial-scene-NN-dataset` on Kaggle, MIT licence): 282,443 images, 1.65 million pallet and stillage boxes, 12.5% of the images with none of either.
- Each epoch of a mixed run is every real image once plus a fresh random draw of synthetic images (share 0.3 or 0.5 of the epoch).

## Result

Validation mAP50-95 on the full validation split:

| Run | Epochs | Best (epoch) | Final | mAP50 | Recall@300 | Best F1 |
| --- | --- | --- | --- | --- | --- | --- |
| LOCO, 150-epoch baseline | 150 | 0.0608 (25) | 0.0437 | 0.210 | 0.215 | 0.309 |
| LOCO | 30 | 0.0556 (20) | 0.0535 | 0.185 | 0.216 | 0.283 |
| LOCO + scene 02, 30% synthetic | 30 | 0.0718 (15) | 0.0634 | 0.227 | 0.250 | 0.319 |
| LOCO + scenes 1,2,4,5,6, 50% synthetic | 30 | **0.0795 (20)** | **0.0746** | **0.244** | 0.240 | **0.342** |

- The training loss falls smoothly in every run (8.6 to 1.3 over 150 epochs), but the validation score peaks within 15 to 25 epochs and then declines: the model memorises the training sites. On 300 training images the epoch-54 baseline reached mAP50-95 0.46 against 0.05 on validation.
- SORDI scene 02 alone raised the peak by 29% (0.0556 to 0.0718). Four more scenes at a higher share added a further 11% at the peak and kept the score from falling as fast (6% down from the peak by epoch 30, against 12%).
- Single runs with one seed; the epoch-to-epoch jitter in validation is about 0.005, so the last gap is likely real but small.
- Absolute scores are still low: roughly 30% of the pallets found at about 40% precision on unseen sites.

## Conclusion

Synthetic variety helps and the overfitting is the main problem. Next: augmentation (#33), which the paper's recipe relies on, then a long run on all data with augmentation, then labelled HMGMA frames for fine-tuning and a deployment-matched validation set.
