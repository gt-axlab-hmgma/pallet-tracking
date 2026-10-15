# 004: many distinct images or a few repeated, and the long mixed run

## Goal

At equal image views, is it better to see many different images once or a small set many times? Then train the chosen model for longer (issue: night driver).

## Config

- One-hour arms (`hour-R1/R3/R9`, run with the script that was removed after this write-up): LOCO + SORDI scenes 02, 04, 06, about 34,000 views each: 34,000 x 1 epoch, 11,000 x 3, 3,820 x 9. From scratch, mosaic on.
- Night tests: `run_mixed_distinct-vs-repeats_then-long.py` (COCO init, lr multiple 1.0, alpha 0.3 per step, EMA, scenes 01, 02, 04, 05, 06). Test A: 60,000 images x 1 epoch. Test B: 6,000 images x 10 epochs. Rule fixed in advance: a gap over 0.01 LOCO test mAP50-95 picks a winner, else A.
- Long run `night-long`: 33,219 images x 20 epochs (2,820 LOCO + 30,399 SORDI), EMA 0.998, close mosaic 2, scored every 2 epochs. 4.05 h.

## Result

| Run | Images x epochs | LOCO test | SORDI test |
| --- | --- | --- | --- |
| hour-R1 | 34,000 x 1 | 0.009 (0.034) | 0.124 (0.263) |
| hour-R3 | 11,000 x 3 | 0.013 (0.059) | 0.096 (0.225) |
| hour-R9 | 3,820 x 9 | 0.026 (0.107) | 0.030 (0.097) |
| night-A | 60,000 x 1 | 0.112 (0.317) | 0.570 (0.747) |
| night-B | 6,000 x 10 | 0.171 (0.461) | 0.513 (0.704) |
| **night-long** | 33,219 x 20 | **0.196 (0.510)** | **0.681 (0.838)** |

## Conclusion

- On synthetic data distinct images win (R1 over R9, A over B). On real data B wins, because 47% of its images are real against 5% for A; the repeated set matters, not the repeats.
- The driver chose B's shape (20 epochs of fewer images) for the long run; `night-long` is the model in use.
- Single seed; the hour arms are short and from scratch, so only the direction is meaningful.
