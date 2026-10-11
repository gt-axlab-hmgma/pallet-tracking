# 001: overfit gate

## Goal

Can the whole pipeline (model, assigner, loss, data reader, letterbox, decoding, top-300 pick) memorise a handful of real images? This is guide step T4. Real training should not start until it passes. A model that cannot overfit 8 images has a bug somewhere; one that can has at least no bug that makes learning impossible.

## Config

```bash
uv run pv-overfit --device mps          # defaults: --source loco --images 8 --steps 800 --size 640 --out runs/overfit
```

- `yolo26n`, one class, fresh weights (seed 0), AdamW, learning rate 1e-3, no weight decay, no augmentation.
- The 8 images are one batch, so a step is an epoch. The one-to-many weight `alpha` falls from 0.8 to 0.1 over the 800 steps.
- Code: the `xjm/t4-overfit-gate` branch (PR for issue #14), on top of `339e4ac`. Run on an Apple M5 Pro (MPS), float32.

## Data

The first 8 images of the LOCO training split (`configs/data/loco.yaml`: pallets and stillages mapped to `pallet`). Together they hold 222 labelled objects, from 1 to 65 per image.

## Result

| | |
| --- | --- |
| Loss | 8.50 to 0.30 (down 96%) |
| AP50 on the training images | 0.964 |
| Time | 245 s (800 steps, 640x640, batch 8) |
| Verdict | **passed** (needs loss down 90%, AP50 at least 0.95, no more than 1.25x the object count plus one confident boxes) |

Confident detections (score above 0.3) against labelled objects, per image: 13/13, 9/9, 1/1, 18/18, 6/6, 59/63, 47/47, 62/65. The one-to-one head never fires more than the labelled count, so there are no duplicate boxes, which is what makes inference NMS-free. The overlays in `runs/overfit/overfit_<i>.jpg` (green labels, red detections) show the detections on top of the labels, including the dense rows of small boxes in the far background of image 5.

### What else was learned

- **A falling loss is not enough.** On synthetic images the loss was already down 94% after 120 steps while AP50 was 0.04, and down 98% after 200 steps with AP50 0.60. Only the detection check tells you the pipeline has really learned. A pipeline with its final top-300 step corrupted still got the loss down 99%, with AP50 0.28: the gate failed it.
- **Convergence.** On the real images, 300 steps gave loss down 93% and AP50 0.905; 800 steps gave 0.968 (an earlier run with the same settings; the 0.964 above is the shipped command). More steps would likely lift it further.
- **Speed.** About 0.3 s per step at batch 8 on the Mac GPU, using about 4 GB of memory out of 64 GB.

## Conclusion

The pipeline agrees with itself: boxes, classes and coordinates line up from the reader to the final detections, and the one-to-one head learns one box per object. The next step is a first real training run on the full LOCO split with evaluation on the validation split (#40, #43), preceded by the optimiser, schedule and checkpoint work (#35, #36, #39).

Limits of this check: it uses the training images, so it says nothing about generalisation, and no augmentation was used.

Re-run it any time the pipeline changes: `uv run pv-overfit --device mps`, or the minute-long synthetic version with `uv run pytest -m slow`.
