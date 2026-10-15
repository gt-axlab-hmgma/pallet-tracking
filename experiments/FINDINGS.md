# Findings

What the experiments (001 to 006) showed, one line each. Scores are mAP50-95 with mAP50 in brackets, single seed, YOLO26n, Apple M5 Pro. Details and commands are in each experiment's README; run scripts sit next to them.

## The model in use

- **`weights/pallet_yolo26n_coco-init_loco-sordi-mixed.pt`** (the `night-long` run, experiment 004): COCO init, LOCO + SORDI scenes 01, 02, 04, 05, 06 drawn alike, 33,219 images x 20 epochs, MuSGD lr 0.01, lr multiple 1.0, alpha 0.3 (per step), EMA 0.998.
- LOCO test (unseen site) 0.196 (0.510); SORDI test 0.681 (0.838); unseen SORDI scene 08 0.606 (0.738), scene 03 0.234 (0.311).
- Not measured: any score on the factory's own camera.

## Comparisons

- **Overfit gate (001):** the pipeline can memorise 8 real images (AP50 0.964), so no bug blocks learning.
- **LOCO only vs LOCO + SORDI, from scratch, 30 epochs (002):** LOCO val 0.056 (LOCO only), 0.072 (+ scene 02, 30% synthetic), 0.080 (+ five scenes, 50% synthetic). More synthetic variety is better; the LOCO-only run overfits after about 20 epochs.
- **COCO weights vs scratch (003), equal data and views:** COCO is better, LOCO test 0.126 against 0.076, SORDI test 0.547 against 0.408.
- **Learning rate of the copied tensors, 0.1 / 0.2 / 0.5 / 1.0 x (003):** 1.0 is best, LOCO test 0.098 / 0.126 / 0.152 / 0.169. Use an ordinary full fine-tune.
- **Many distinct images vs few repeated (004):** at 34,000 views, 34,000 distinct images beat 3,820 repeated nine times on synthetic data (SORDI test 0.124 against 0.030). At 60,000 views, 60,000 distinct images (A) win on SORDI, 0.570 against 0.513 for 6,000 x 10 (B), but B wins on LOCO test, 0.171 against 0.112, because 47% of B's images are real against 5% of A's. So repeats are fine when the repeated set is the real one.
- **Long mixed run (004, `night-long`):** 20 epochs of 33,219 images gives LOCO test 0.196, up from 0.171 at 60,000 views. It is the best model on real images.
- **Freeze 10 vs freeze 0, SORDI-only, 2,820 images (005):** unfreezing is better, SORDI test 0.595 (0.773) against 0.546 (0.732). Freeze 23 (head only) was only unit-tested, not run.
- **LOCO-only vs SORDI-only vs mixed (005):** each transfers poorly to the other. LOCO-only: LOCO test 0.150, SORDI test 0.062. SORDI-only: SORDI test 0.595, LOCO test 0.013. Only the mixed model keeps both. LOCO is low-angle with unloaded pallets, so it fits a factory floor view poorly.
- **AdamW lr 0.001 + freeze 10 + patience 20 (Ultralytics small-data advice), LOCO-only (005):** LOCO test 0.150 against 0.169 for the mixed COCO run with MuSGD (003). Not a controlled comparison (the data differ); it gave no reason to leave MuSGD.
- **SORDI-only at scale, 80,000 images x 20 epochs (006, `sordi-big`):** SORDI test 0.726 (0.871), val 0.732 (0.878); unseen scene 08 0.650 (0.780); unseen scene 03 0.275 (0.352); LOCO test 0.012. Beats the mixed model on synthetic scenes (by about 0.04 on both unseen ones) but finds only about 5 pallets per frame on the real-footage clip against 18 for the mixed one.
- **Overfitting check (006):** validation rose at every score (0.61 to 0.729) while loss fell, so no overfitting on that curve. The validation images share scenes with training, so the unseen-scene scores are the real check; scene 03 shows large drops are possible.

## What to report

- Highest in-distribution score: SORDI test 0.871 mAP50 / 0.726 mAP50-95 (`sordi-big`, 5,000 images, same scenes as training).
- On scenes never trained on: scene 08 0.780 / 0.650, scene 03 0.352 / 0.275 (2,000 images each). Report both, not one.
- Model chosen for use: `night-long`, because it is the only one that also works on real images.

## Limits and open items

- Single runs and fixed draws; differences under about 0.01 are noise.
- No labelled footage from the factory camera. The only real-image test is LOCO (another site). The clip checks are by eye on an AI-generated video.
- SORDI's split is per image, so its test shares scenes with training; a split by scene was not built.
- The COCO weights are AGPL-3.0 with an Objects365 lineage (see `003-coco-init/README.md`): settle with the PI before shipping anything built on them.
- Tried and dropped without a finding: extra overfit-gate variants, the 30-epoch LOCO runs beyond what 002 records, AdamW as a stand-alone change.
