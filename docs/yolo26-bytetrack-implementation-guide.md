# Implementing a YOLO26-style detector and ByteTrack from scratch

A step-by-step engineering guide for this repo (`pallet-tracking`). It is written so that you can implement everything yourself: every step says what to build, the exact tensor shapes and formulas, how to test it, what usually goes wrong, and when you are done.

Scope: the full detector (architecture, decoding, loss, assignment, data pipeline, training, evaluation, fine-tuning), the ByteTrack tracker, and the glue that turns video into zone events. Nothing is dropped for time. Optional parts are marked optional; they are still specified.

---

## 0. Read this first

### 0.1 How each step is laid out
**Goal** (what exists afterwards) / **Spec** (shapes, formulas) / **Build notes** / **Test** / **Pitfalls** / **Done when**. Treat each step as one small commit with its tests, as in the rest of the repo.

### 0.2 Provenance and licence rules (important)
- Ultralytics code and weights are AGPL-3.0 (see `docs/design.md`). **Do not read, copy or paraphrase Ultralytics source files, and never load their weights.** Everything below is a specification: prose, formulas, shapes and hyperparameters. You write the code.
- Allowed sources: the papers listed in section 12, Ultralytics public *documentation pages*, and academic/MIT references. `design.md` also lists the public YOLO26 layer config as a source; reading a layer list is different from reading code, but confirm that with the lab/GT before using it.
- Keep a short provenance note in the repo (`docs/yolo26-notes.md`): which paper or doc each design choice came from. Record every decision you make in a decision log (template in section 11.4).

### 0.3 Confidence tags used below
| Tag | Meaning |
| --- | --- |
| **[P]** | Stated in a paper listed in section 12 |
| **[D]** | Documented publicly (docs, papers' appendices) |
| **[S]** | Standard, widely used technique (any textbook/blog implementation agrees) |
| **[C]** | Convention inherited from earlier YOLO versions (v5/v8/11 family). The YOLO26 paper does not restate it; verify against the YOLO26 docs and by checking your parameter count |
| **[?]** | Not specified anywhere I can cite. A design choice you make and ablate |

### 0.4 Known unknowns (decide and log these)
1. Exact YOLO26 layer list and repeat counts. [resolved] The layout in this guide reproduces the published parameter totals exactly (n 2,572,280 and s 10,009,784 at 80 classes, both heads), which pins every width, repeat count, `expand` and `c3k` flag.
2. How the direct box regression (no DFL) keeps distances non-negative. [decided] No activation on the raw distances: the head's box bias starts them at 2 strides and the loss keeps them positive. Softplus, `exp` and ReLU stay as ablations. [?]
3. Whether the one-to-one head's gradients flow into the shared features. [decided] No: its input is detached in training. Ablate later. [?]
4. The exact form of ProgLoss and STAL. [resolved by the YOLO26 report] ProgLoss is Eq. 2-3 (branch weight `α(t)` falling linearly per epoch from 0.8 to 0.1); STAL is Eq. 4-6 (sides below the smallest stride clamped to the next stride for candidate selection only). Both are part of the baseline (T2, T3). [P]
5. Loss gains once DFL is removed. Start from the report's pretraining recipe (Table S2): box 7.5, cls 0.5, L1 6.0 (its `dfl` field now scales the L1 term); its COCO fine-tuning values differ by model size. [P]

### 0.5 Conventions
- Tensors are NCHW. Boxes are `xyxy` in **input pixels** (the letterboxed 640×640 image) unless stated.
- The repo contract (`contracts/model_io.yaml`): input `[1,3,640,640]` RGB scaled by 1/255, no mean/std; output `[1, 300, 6]` = `x1,y1,x2,y2,score,class` in input pixels; ONNX opset 13; letterbox pad value 114.
- `eps = 1e-7` unless stated. Use float32 for loss/assigner math even under autocast.

---

## 1. Concept primer (read once, refer back)

### 1.1 What a detector outputs
Given an image, a list of `(box, class, confidence)`. A modern one-stage detector produces a **dense** prediction at every cell of a few feature maps, then reduces that to a short list.

### 1.2 Feature maps, strides and cells
A conv net downsamples the image. A feature map with **stride s** has one cell per `s×s` input pixels. YOLO uses three maps:

| Level | Stride | Grid at 640 input | Best at |
| --- | --- | --- | --- |
| P3 | 8 | 80×80 = 6400 | small objects |
| P4 | 16 | 40×40 = 1600 | medium |
| P5 | 32 | 20×20 = 400 | large |

Total **A = 8400** prediction cells (anchors). A cell's reference point (its "anchor point") is its centre in input pixels: `((i + 0.5)·s, (j + 0.5)·s)`.

### 1.3 Anchor-free box regression
Each cell predicts four distances from its anchor point to the box edges: left, top, right, bottom (`ltrb`), measured in units of the stride:
`x1 = ax − l·s, y1 = ay − t·s, x2 = ax + r·s, y2 = ay + b·s`.

### 1.4 IoU and its relatives
- IoU = area(A∩B) / area(A∪B), range 0..1.
- GIoU adds a penalty for the empty space in the smallest enclosing box. DIoU adds a penalty for centre distance. CIoU also penalises aspect-ratio mismatch. Full formulas in step T1.

### 1.5 Why NMS exists, and why YOLO26 removes it
Ordinary detectors let many neighbouring cells fire on the same object, then run **non-maximum suppression** to drop the duplicates. NMS is sequential, data-dependent and awkward to export. YOLO26 trains the network so that **exactly one cell fires per object** (one-to-one assignment). Inference then is: score everything, take the top K=300, done. [P]

But one-to-one supervision alone is sparse (one positive per object) and trains poorly, so training uses **two heads** on the same features: a one-to-many head (about 10 positives per object, dense signal) and a one-to-one head (what ships). [P: YOLOv10 "consistent dual assignments"]

### 1.5.1 Why DFL is removed
Older YOLOs predicted each edge distance as a probability distribution over 16 bins and took the expectation (Distribution Focal Loss). That needs a softmax and an integral per coordinate, which is slow on edge chips and hard to quantise. YOLO26 predicts one number per edge (`reg_max = 1`). [P]

### 1.6 Label assignment
Training must decide which cells are responsible for which ground-truth (GT) box ("positives"); every other cell should predict background. The quality of this decision dominates accuracy. YOLO-style models use **task-aligned assignment** (TOOD): a cell is a good positive when it both classifies the object well and localises it well. [P]

### 1.7 mAP in one paragraph
For a given IoU threshold, sort predictions by score, mark each as true/false positive against GTs, compute the precision-recall curve, and take the area under it (average precision, AP). **mAP50** uses IoU ≥ 0.5; **mAP50-95** averages AP over thresholds 0.50, 0.55, …, 0.95. With a single class, mAP = AP.

---

## 2. Repository mapping

The existing pipeline stays as is: `decode` + `OnnxRuntimeBackend`/`TorchBackend` (`packages/palletvision-infer`), `export_onnx` (`palletvision-train/export.py`), `ZoneEngine`, `EventFactory`, sinks. **The new model must satisfy the contract and nothing downstream changes.**

| Package | New/changed files | Notes |
| --- | --- | --- |
| `palletvision-model` (torch only) | `config.py`, `blocks.py`, `attention.py`, `yolo26.py`, `head.py`, `decode.py`, `iou.py`, `assigner.py`, `loss.py` | Imports only `torch` (design rule). Remove or keep `tiny.py` until the real model passes the gates |
| `palletvision-train` | `coco.py`, `transforms.py`, `mosaic.py`, `sampler.py`, `ema.py`, `trainer.py` (extend), `evaluate.py` (`pv-eval`), `finetune.py` | New deps: `pycocotools`, `albumentations` optional |
| `palletvision-track` (new, numpy + scipy) | `kalman.py`, `matching.py`, `bytetrack.py`, `votes.py` | No torch. Written so it can be ported to C++ later |
| `palletvision-pipeline` (new) | `run.py` (`pv-run`) | Depends on infer, track, zones, events |
| `configs/` | `model/yolo26n.yaml`, `data/loco.yaml`, `data/sordi.yaml`, `data/loco_sordi.yaml`, `train/stage1_pallet.yaml`, `track/bytetrack.yaml` | Hydra groups |
| `tests/golden/` | `tracker/*.json` | Language-neutral tracker fixtures, same style as `tests/golden/zones` |

---

## 3. Data setup

### 3.1 Layout (git-ignored under `data/`)
```
data/loco/    LOCO images + annotation file(s)
data/sordi/   SORDI subset (a few thousand images)
data/videos/  short .mp4 pallet clips
```
Record source URL, date and licence in `data/README.md`. Check licences for the intended use (research now, possibly sponsor deployment later). Format and access terms for SORDI are not verified: write its adapter from the real folder contents.

### 3.2 First hour with LOCO (do this before any modelling)
1. Load the annotation JSON and print: number of images, number of annotations, the list of `categories` (id → name), and annotations per category.
2. Stage-1 class mapping. [decided] `pallet` and `stillage` map to the single class `pallet`; `pallet_truck`, `forklift` and `small_load_carrier` are dropped (a pallet truck is a vehicle, not cargo, and a parked one must not raise a pallet event). Revisit stillages in stage 2 once HMGMA's real pallet types are known.
3. Check whether LOCO ships train/val/test splits. If yes, use them unchanged. If no, create one deterministic split (seeded) and commit the list of image ids, so every experiment uses the same split.
4. Compute box statistics (width, height, area relative to image, aspect ratio, boxes per image). This tells you how small the objects are and sets expectations: boxes below ~8 px at 640 are hard at stride 8.
5. Draw 20 random images with their boxes and look at them. Verify image paths, box convention (`[x, y, w, h]` in COCO), and orientation (EXIF rotation can swap width/height).
6. Handle `iscrowd` annotations: skip them for training; for evaluation pycocotools treats them as ignore regions.

---

## 4. The model

### Step M1. Model configuration and channel scaling

**Goal:** a config object that determines every layer's channels and repeats.

**Spec** [C]: three numbers per size, `depth_multiple`, `width_multiple`, `max_channels`.
| Size | depth | width | max_channels |
| --- | --- | --- | --- |
| n | 0.50 | 0.25 | 1024 |
| s | 0.50 | 0.50 | 1024 |

- Output channels of a layer: `c = make_divisible(min(c_nominal, max_channels) · width, 8)`, where `make_divisible(x, 8) = ceil(x / 8) · 8`.
- Repeats of a layer: `n = max(round(n_nominal · depth), 1)` when `n_nominal > 1`, else 1.
- The detection head takes `nc` (number of classes) and `reg_max` (=1).

Expected P3/P4/P5 head input channels: n → 64/128/256, s → 128/256/512 (matches `design.md`).

**Test:** table-driven test of channel math for n and s; `make_divisible` edge cases.
**Done when:** the config reproduces the channel table in M5.

### Step M2. Conv block, activation, initialisation

**Spec** [S/C]:
- `Conv(c_in, c_out, k, s, groups=1, act=True)`: `Conv2d(bias=False, padding=k//2)` → `BatchNorm2d(eps=1e-3, momentum=0.03)` → `SiLU` (SiLU(x) = x·sigmoid(x)). `act=False` omits SiLU.
- Depthwise conv `DWConv` = `Conv` with `groups = c_in` (used in the class branch).
- Weight init: PyTorch default is fine. Special initialisation of head biases is in M7.
- Padding `k//2` keeps spatial size for stride 1 and halves it (rounding down) for stride 2 with k=3.

**Test:** output shapes for k∈{1,3}, s∈{1,2}; parameter count equals `c_in·c_out·k²/groups + 2·c_out`.
**Pitfall:** `bias=False` on conv before BN; BN `momentum=0.03` is PyTorch's convention (weight on the new batch statistic).

### Step M3. Bottleneck, C3k, C3k2

**Bottleneck(c_in, c_out, shortcut, groups, kernels=(3,3), expand=0.5)** [S]: hidden `c_ = int(c_out · expand)`; `Conv(c_in, c_, k1, 1)` → `Conv(c_, c_out, k2, 1, groups)`; if `shortcut and c_in == c_out`, add the input.

**C2f-style split-and-concatenate block (the base of C3k2)** [C]: `C2f(c_in, c_out, n, shortcut, groups, expand=0.5)`:
1. `c = int(c_out · expand)`.
2. `y = Conv(c_in, 2c, 1)(x)`, split along channels into two halves `[y0, y1]` of `c` channels.
3. Apply `n` bottlenecks in sequence, each to the previous output, appending each result to the list: `[y0, y1, m1(y1), m2(m1(y1)), …]`.
4. Concatenate all (`(2+n)·c` channels) and `Conv((2+n)·c, c_out, 1)`.
In C2f proper the inner bottlenecks use kernels (3,3) and `expand=1.0`. **C3k2's plain-Bottleneck path is different: it keeps `Bottleneck`'s default `expand=0.5`** (hidden width `c/2`), which is what reproduces the published parameter counts.

**C3 (CSP bottleneck with 3 convs)** [S]: `c_ = int(c_out · expand)`; branch A: `Conv(c_in, c_, 1)` → `n` bottlenecks (kernels (1,3), expand 1.0); branch B: `Conv(c_in, c_, 1)`; concatenate A and B, then `Conv(2c_, c_out, 1)`.

**C3k** [C]: a C3 whose inner bottlenecks use kernels `(k, k)` with `k=3` and `n=2` repeats.

**C3k2(c_in, c_out, n, c3k, expand, groups, shortcut)** [C]: the C2f block above, except the inner module list is `n` × `C3k(c, c, 2, shortcut, groups)` if `c3k=True`, else `n` × `Bottleneck(c, c, shortcut, groups)`. The `expand` argument in the layer table (0.25 in the early stages, 0.5 otherwise) is passed as `expand`. With `attn=True` each of the `n` slots is instead `Bottleneck(c, c, shortcut, groups)` followed by `PSABlock(c, attn_ratio=0.5, num_heads=max(c // 64, 1))`; `attn` takes priority over `c3k` (the last neck block uses it).

**Test:** shape preserved when `stride`-free; channel count of the concat equals `(2+n)·c`; compare parameter count against a hand calculation for one block; shortcut only active when channels match.
**Pitfalls:** splitting uses `chunk(2, dim=1)`, not `split` by arbitrary sizes; the concat must include the **first** chunk (the skip path), or capacity silently drops.

### Step M4. SPPF and attention (C2PSA)

**SPPF(c_in, c_out, k=5)** [S]: `c_ = c_in // 2`; `x = Conv(c_in, c_, 1)`; apply `MaxPool2d(k, stride=1, padding=k//2)` **three times in sequence**, keeping each output; concatenate `[x, p1, p2, p3]` (4·c_ channels); `Conv(4·c_, c_out, 1)`. Equivalent to parallel pools of size 5/9/13 at lower cost. **Optional residual** `shortcut`: with `shortcut=True` and `c_in == c_out` the block's input is added to its output; the backbone's layer 9 turns it on [?: not visible in the paper's figure, and it adds no parameters].

**Attention(dim, num_heads, attn_ratio=0.5)** [C/S]:
- `head_dim = dim // num_heads`; `key_dim = int(head_dim · attn_ratio)`; `scale = key_dim ** -0.5`.
- `qkv = Conv(dim, dim + 2·key_dim·num_heads, 1, act=False)(x)`; reshape to `[B, num_heads, 2·key_dim + head_dim, N]` where `N = H·W`; split into `q [B,h,key_dim,N]`, `k [B,h,key_dim,N]`, `v [B,h,head_dim,N]`.
- `attn = softmax((qᵀ · k) · scale, dim=-1)` with shape `[B,h,N,N]`.
- `out = v · attnᵀ`, reshaped back to `[B, dim, H, W]`.
- Add a **positional term**: a 3×3 depthwise `Conv(dim, dim, 3, 1, groups=dim, act=False)` applied to `v` reshaped to `[B,dim,H,W]`. Add it to `out`.
- Project with `Conv(dim, dim, 1, act=False)`.

**PSABlock(c)** [C]: `x = x + Attention(x)`; then `x = x + FFN(x)` with `FFN = Conv(c, 2c, 1)` → `Conv(2c, c, 1, act=False)`.

**C2PSA(c_in, c_out=c_in, n=1, expand=0.5)** [C]: `c = int(c_in · expand)`; `Conv(c_in, 2c, 1)` → split into `a` and `b` (c channels each); `b = PSABlock(c, num_heads = c // 64)` repeated `n` times; concat `[a, b]`; `Conv(2c, c_in, 1)`.
For the n model at P5: `c_in = 256`, `c = 128`, `num_heads = 2`, `head_dim = 64`, `key_dim = 32`, `N = 20·20 = 400`, so attention maps are 400×400 per head: cheap.

**Why only at P5:** attention cost grows with N². At P3 (N = 6400) it would be 6400² per head. Do not move it.

**Test:** attention output shape = input shape; permutation sanity (output at a position changes when a distant position changes, which a conv would not do beyond its receptive field); `softmax` rows sum to 1; ONNX-exportable (step M10).
**Pitfalls:** transposes in the `qᵀ k` product; applying softmax over the wrong axis; forgetting the positional depthwise conv.

### Step M5. Backbone

**Spec** [C]. Layers in order (nominal args → n-model resulting channels, repeat counts after depth scaling):

| # | Layer | Nominal (c, k, s / extras) | Out channels (n) | Out stride | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Conv | 64, 3, 2 | 16 | 2 | stem |
| 1 | Conv | 128, 3, 2 | 32 | 4 | |
| 2 | C3k2 | 256, c3k=False, expand 0.25, n=2→1 | 64 | 4 | |
| 3 | Conv | 256, 3, 2 | 64 | 8 | |
| 4 | C3k2 | 512, c3k=False, expand 0.25, n=2→1 | 128 | 8 | **keep as B3 (skip to neck)** |
| 5 | Conv | 512, 3, 2 | 128 | 16 | |
| 6 | C3k2 | 512, c3k=True, n=2→1 | 128 | 16 | **keep as B4** |
| 7 | Conv | 1024, 3, 2 | 256 | 32 | |
| 8 | C3k2 | 1024, c3k=True, n=2→1 | 256 | 32 | |
| 9 | SPPF | 1024, k=5, shortcut=True | 256 | 32 | |
| 10 | C2PSA | 1024, n=2→1 | 256 | 32 | **keep as B5** |

Spatial sizes at 640 input: 320 → 160 → 80 → 40 → 20.

**Test:** forward on `[2,3,640,640]` returns three tensors of shapes `[2,128,80,80]`, `[2,128,40,40]`, `[2,256,20,20]` for n (B3, B4, B5 as in the table; note the neck produces the 64/128/256 head inputs).
**Done when:** shapes match; running on 320 and 512 inputs works (sizes divisible by 32).

### Step M6. Neck (PAN)

**Spec** [C] (continuing layer numbers; channels for n):
| # | Op | Inputs | Out ch | Out stride |
| --- | --- | --- | --- | --- |
| 11 | Upsample ×2 (nearest) | 10 | 256 | 16 |
| 12 | Concat | 11, 6 | 384 | 16 |
| 13 | C3k2 (512, c3k=True) | 12 | 128 | 16 |
| 14 | Upsample ×2 | 13 | 128 | 8 |
| 15 | Concat | 14, 4 | 256 | 8 |
| 16 | C3k2 (256, c3k=True) | 15 | **64 (N3)** | 8 |
| 17 | Conv (256, 3, 2) | 16 | 64 | 16 |
| 18 | Concat | 17, 13 | 192 | 16 |
| 19 | C3k2 (512, c3k=True) | 18 | **128 (N4)** | 16 |
| 20 | Conv (512, 3, 2) | 19 | 128 | 32 |
| 21 | Concat | 20, 10 | 384 | 32 |
| 22 | C3k2 (1024, c3k=True, e=0.5, attn=True), 1 repeat | 21 | **256 (N5)** | 32 |

Concats are along channels. The head consumes `(N3, N4, N5)` = layers `(16, 19, 22)`.

Layers 13, 16 and 19 have 2 nominal repeats (1 after depth scaling for n); layer 22 has exactly 1. Layer 22's attention is shown in the paper's Fig. S1 (`attn=True`). The `c3k=True` flags on 13, 16 and 19 are confirmed by the exact parameter-count match, since `c3k` changes the count.

**Test:** shapes `[B,64,80,80]`, `[B,128,40,40]`, `[B,256,20,20]`; every concat has matching spatial sizes.
**Pitfalls:** wrong skip index (concat with the pre-SPPF layer instead of layer 10); nearest vs bilinear upsampling (nearest exports cleanly).

### Step M7. Detection head (two branches per level)

**Spec** [C/P]. For each level `i` with input channels `ch_i`:
- Intermediate widths: `c_box = max(16, ch_0 // 4, 4·reg_max)` → for n with `reg_max=1`: `max(16, 16, 4) = 16`. `c_cls = max(ch_0, min(nc, 100))` → for n, `nc=1`: 64; `nc=4`: 64; `nc=80`: 80.
- **Box branch:** `Conv(ch_i, c_box, 3)` → `Conv(c_box, c_box, 3)` → `Conv2d(c_box, 4·reg_max, 1)` (plain conv with bias).
- **Class branch:** `[DWConv(ch_i, ch_i, 3) → Conv(ch_i, c_cls, 1)]` → `[DWConv(c_cls, c_cls, 3) → Conv(c_cls, c_cls, 1)]` → `Conv2d(c_cls, nc, 1)`.
- Output per level: `box_raw [B, 4, H_i, W_i]`, `cls_logits [B, nc, H_i, W_i]`.

**Two copies:** the **one-to-many branch** and the **one-to-one branch** each have their own box and class convs per level (same architecture, separate weights). The one-to-one branch is trained on top of the shared features; whether to `detach()` its input is design choice [?] (see section 0.4). **Decision:** the one-to-one branch's input features are detached in training, so its gradients cannot disturb the shared backbone/neck; ablate later.
Inference uses the one-to-one branch only; the one-to-many branch can be dropped at export.

**Flattening:** concatenate the three levels along the cell axis: `box_raw [B, A, 4]`, `cls_logits [B, A, nc]` with `A = 8400` ordered P3 row-major, then P4, then P5. Keep the same order in the anchor-point table (step M8).

**Initialisation** [S]:
- Class final-conv bias, per level: `b = log(prior_objects / nc / cells)` with `prior_objects = 5` and `cells = (input_size / stride)²` (6400, 1600 and 400 at 640), so each level starts expecting about 5 detections and the thousands of background cells do not dominate the early loss. Initial scores for one class are about 0.0008 (P3), 0.003 (P4) and 0.012 (P5). `prior_objects` is tunable (LOCO averages about 24 pallets per image). [S/C]
- Box final-conv bias: 2.0 for all four outputs, so initial distances are two strides (a 32 px square at P3 up to 128 px at P5 for a 640 input).
- Positivity of distances [?]: **no activation** is applied to `box_raw`; the 2.0 bias starts the distances positive and the loss keeps them so. Softplus, `exp` and ReLU are ablation alternatives.

**Parameter sanity:** per branch copy at n with one class, the box branch has 71,820 parameters and the class branch 48,963. With 80 classes and both heads the full model has **2,572,280** (n) and **10,009,784** (s) parameters, which `test_param_counts.py` pins; the deployed one-to-one-only model has 2,417,452 (n, 80 classes) and 2,383,407 (n, 1 class).

**Test:** output shapes; both branches produce `[B,8400,·]`; one-to-one input is detached when configured (check `requires_grad` of the head's input graph or that backbone grads are zero for a loss that uses only the one-to-one branch).
**Pitfalls:** an off-by-one in `c_box`/`c_cls`; forgetting the DW conv `groups`; mixing branch weights; flatten order mismatching anchors.

### Step M8. Anchor points and box decoding

**Spec** [S]:
- For each level with stride `s` and grid `H×W`: anchor points `(ax, ay) = ((j + 0.5)·s, (i + 0.5)·s)` for row `i`, column `j`. Concatenate levels in the order used in M7 → `anchor_points [A, 2]` and `anchor_strides [A, 1]`.
- With the raw distances `d = box_raw [B,A,4]` as `(l, t, r, b)` in stride units (no activation, see M7):
  `x1 = ax − l·s`, `y1 = ay − t·s`, `x2 = ax + r·s`, `y2 = ay + b·s`.
- Clamp decoded boxes to the image (`[0, S]`) only at inference/export, not in the loss.

**Test:** known cell/distance → exact box; anchor table has 8400 rows with correct first/last points (first: (4,4); last of P5: (624,624) at 640).
**Pitfalls:** forgetting the `+0.5` offset (a systematic half-cell shift that costs accuracy); using feature-map units in one place and pixel units in another (pick pixels everywhere).

### Step M9. Inference post-processing (NMS-free top-K)

**Spec** [P/S]:
1. Scores: `p = sigmoid(cls_logits) [B, A, nc]` from the one-to-one branch.
2. Flatten the cell and class axes: `[B, A·nc]`. Take `K = min(300, A·nc)` highest scores (`topk`, constant K).
3. Recover indices: `cell = idx // nc`, `cls = idx % nc`. Gather the decoded boxes of those cells.
4. Output `[B, K, 6]` = `x1,y1,x2,y2,score,cls`, boxes clamped to the image, class as float.
5. K must be 300 to satisfy the contract; if `A·nc < 300` pad with zero-score rows.

No confidence threshold inside the model (the repo's `decode` applies `conf` outside, and mAP evaluation wants a very low threshold).

**Test:** shape `[B,300,6]`; scores sorted descending; contract compliance test against `contracts/model_io.yaml`; `TorchBackend` and `OnnxRuntimeBackend` give identical `Detections` after decode. **Caveat:** an untrained network scores almost every cell the same, so ties break differently in the two runtimes; compare the per-cell outputs and the top-300 selection separately (as `tests/test_yolo26_export.py` does) and check full-output identity on a trained model.
**Pitfalls:** sorting after thresholding changes the shape (breaks export); topk over classes separately (loses multi-class completeness).

### Step M10. Parameter/FLOP check and ONNX export

**Spec:**
- Count params with `sum(p.numel())`; FLOPs with the repo helper `palletvision_infer.bench.count_params_flops` (counts 2 FLOPs per multiply-add) at 640. Targets for n: ≈2.4M params and ≈5.5 GFLOPs, within ±10–15%; measured: 2,383,407 params and 5.31 GFLOPs (one class), 2,417,452 and 5.48 (80 classes). Investigate larger differences: usually a wrong layer table, wrong `expand` or a head width bug.
- Export in `eval()` mode with the one-to-one branch only, opset 13, input name `images`, output name `detections`, fixed input size 640 (static shapes), via the existing `export_onnx`.
- Compare with ONNX Runtime on random inputs and real images: max abs difference ≤ 1e-4 on scores and ≤ 1e-3 px on boxes. **For a randomly initialised network this is unreachable:** float32 rounding through about 100 layers gives differences near 1e-3 of the value range (Torch float32 differs from Torch float64 by about the same), so the repo test uses a relative tolerance of 2e-2 on per-cell outputs. Apply the tight numbers to a trained model.
- Inspect the op list (`onnx` python API). Things to watch for TensorRT 8.2 (Jetson Nano): `TopK` with constant K (fine), `GatherElements`/`Gather`, `Softmax` over the last axis, `Resize` (nearest), `Reshape`/`Transpose` in attention, no `Einsum`, no dynamic control flow, no ops needing opset > 13. Build an engine on a Nano as soon as hardware is available. [?]

**Pitfalls:** tracing leaves Python `int()` constants baked in (fine for static sizes); `torch.export`-style exporters may emit opset > 13, so use the legacy exporter flag the repo already uses; `train()` mode BN leaks into exported graphs.

---

## 5. Training objective

### Step T1. IoU family

**Spec** [P]: for boxes `A = (x1,y1,x2,y2)` and `B`:
- `inter = clamp(min(x2) − max(x1), 0) · clamp(min(y2) − max(y1), 0)`; `union = areaA + areaB − inter + eps`; `IoU = inter / union`.
- Enclosing box `C` = min of mins, max of maxes; `c_w, c_h` its sides.
- **GIoU** `= IoU − (area(C) − union) / area(C)`.
- **DIoU** `= IoU − ρ² / c²`, `ρ²` = squared distance between box centres, `c² = c_w² + c_h² + eps`.
- **CIoU** `= IoU − ρ²/c² − α·v`, with `v = (4/π²)·(atan(w_B/h_B) − atan(w_A/h_A))²` and `α = v / ((1 − IoU) + v + eps)`. **Treat α as a constant (no gradient through α).**
- Loss from CIoU: `1 − CIoU`.
- For the assigner, use plain IoU (or CIoU clamped at 0); the loss uses CIoU.

**Test:** identical boxes → 1; disjoint → 0 (IoU) and negative GIoU; known hand-computed cases; symmetry; gradient exists w.r.t. predicted box; broadcast shapes `[N,4]×[M,4]`→`[N,M]`.
**Pitfalls:** width/height from `x2 − x1` becoming negative for untrained outputs (clamp), `atan(w/h)` blowing up when `h≈0` (add eps), detaching α.

### Step T2. Task-aligned assigner

**Inputs per batch:** `pred_scores [B,A,nc]` (sigmoid), `pred_boxes [B,A,4]` (decoded, detached), `anchor_points [A,2]`, `gt_labels [B,M,1]`, `gt_boxes [B,M,4]`, `mask_gt [B,M,1]` (1 for real GTs, 0 for padding). `M` = max GTs per image in the batch.
**Hyperparameters:** `topk` (one-to-many branch: **10**; one-to-one branch: **7 candidates, then a second top-1 pick (`topk2 = 1`)**, so exactly one positive per object [P: YOLO26 report, Fig. 2]), `alpha`, `beta` [P/C]: TOOD uses α=1, β=6; YOLO-family code commonly uses α=0.5, β=6.0. Start with 0.5/6.0 and tune; use the **same α, β in both branches** (YOLOv10's "consistent matching") [P].

**Algorithm:**
1. **Candidates (with STAL):** `mask_in_gt [B,M,A]` = anchor point strictly inside the GT box **used for candidate selection only**: the real box with each side below `s_min` (the smallest stride, 8) replaced by `s_ref` (the next stride, 16), keeping its centre, i.e. `d̃ = s_ref if d < s_min else d` for `d ∈ {w, h}` [P: STAL, Eq. 4-6]. Every later step (alignment score, conflict resolution, targets, box loss) uses the original box.
2. **Alignment metric** `align [B,M,A] = s^α · iou^β` where `s = pred_scores[b, a, label_of_gt_m]` and `iou = CIoU(gt_box_m, pred_box_a)` clamped at 0 (the YOLO-family implementation; plain IoU is the simpler alternative and is selectable with `iou_kind="iou"`). Zero it wherever `mask_in_gt · mask_gt == 0`.
3. **Top-k selection:** for each `(b, m)`, pick the `topk` anchors with the largest `align` among the candidates → `mask_topk [B,M,A]`. An anchor with metric 0 must not be selected.
4. **Positives:** `mask_pos = mask_topk · mask_in_gt · mask_gt`.
5. **Conflict resolution, then the second filter:** `count [B,A] = Σ_m mask_pos`. Where an anchor has been chosen by more than one GT, keep only the GT with the largest IoU with that anchor's prediction. **One-to-one branch only:** after the conflicts are resolved, each object keeps only its single best remaining cell by alignment (`topk2 = 1`), so an object that lost its favourite cell falls back to its next best instead of ending with none.
6. **Targets:** for every positive anchor gather the assigned GT index → `target_boxes [B,A,4]`, `target_labels [B,A]`; build `target_scores [B,A,nc]` as one-hot of the label (zero for non-positives). `fg_mask [B,A]` = anchor is positive.
7. **Alignment normalisation** [P, TOOD]: for each GT `m`, `max_align_m = max_a align[m,a]·mask_pos`, `max_iou_m = max_a iou[m,a]·mask_pos`. Then for each anchor, `norm = max over m of ( align[m,a] · max_iou_m / (max_align_m + eps) )`, and `target_scores *= norm[..., None]`. This turns the classification target of a positive from "1" into "how good this cell is" (up to the object's best IoU).

**Outputs:** `target_boxes`, `target_scores`, `fg_mask` (and the target GT index for debugging).

**Test (hand-built cases):**
- One GT, one obvious cell inside it → that cell is the single positive (one-to-one setting) and gets `target_scores ≈ iou`.
- Two GTs overlapping, an anchor inside both → the anchor goes to the one with higher IoU.
- A GT so small that no anchor centre is inside its real box → with STAL it still gets candidates (its sides are clamped to 16 px for the inside-box test only) and the original box is still used for scoring; the same GT without STAL gets none.
- Padding GTs (`mask_gt = 0`) never produce positives.
- `topk=10` gives at most 10 positives per GT; the one-to-one setting (`topk=7, topk2=1`) gives exactly 1.
- Shapes with `B>1`, `M` differing between images.
**Pitfalls:** feeding gradients into the assigner (wrap in `no_grad`); using logits instead of sigmoid scores; score of the wrong class; scale mismatch between anchor points and boxes; `M=0` batches (images without GTs) must work and produce no positives.
**Performance:** the `[B,M,A]` tensors are large (e.g. `16·50·8400`); fine on GPU. On MPS some ops (scatter/topk) may be slow or limited; fall back to CPU for the assigner if profiling says so. [verify]

### Step T3. Loss assembly

**Spec** [C/S] for one branch:
- `n_targets = max(Σ target_scores, 1)`.
- **Classification:** `L_cls = Σ BCEWithLogits(cls_logits, target_scores) / n_targets` (sum over all cells and classes). Because positives' targets are the soft alignment-normalised scores, confidence tracks box quality. (A varifocal or quality-focal variant is an optional alternative.)
- **Box:** for positive cells only, `L_box = Σ_i w_i · (1 − CIoU(pred_box_i, target_box_i)) / n_targets` with `w_i = Σ_c target_scores_ic`.
- **L1:** with DFL gone, add a direct L1 term on the same positive cells: `L_l1 = Σ_i w_i · mean|pred_box_i − target_box_i| / n_targets`, both boxes divided by the image size so it is scale free [P: the YOLO26 report replaces DFL with direct regression plus an L1 loss; the normalisation is a repo decision [?]].
- **Total per branch:** `L = λ_box·L_box + λ_cls·L_cls + λ_l1·L_l1`. Start with `λ_box = 7.5`, `λ_cls = 0.5`, `λ_l1 = 6.0` [P: Table S2 of the YOLO26 report, pretraining recipe; its COCO fine-tuning values differ by model size]. If one term dominates, retune.
- **Dual-branch total with Progressive Loss:** `L = α(t)·L_one2many + (1 − α(t))·L_one2one`, each branch with its own assigner, where `α(t)` falls linearly per epoch from `α_init = 0.8` to `α_final = 0.1`: `α(t) = max(1 − t / max(E − 1, 1), 0) · (α_init − α_final) + α_final` [P: Eq. 2-3 and Fig. 2]. A constant 1:1 weighting is the ablation baseline.
- Log each component separately (`box_o2m, cls_o2m, box_o2o, cls_o2o`) every N steps.

**Test:** loss is finite and positive on random data; zero positives → box loss exactly 0 and cls loss positive; loss decreases when predictions move toward the targets; reduces to a known value on a hand-built example.
**Pitfalls:** normalising by the number of GTs instead of `Σ target_scores`; averaging BCE over all `A·nc` cells (too small); computing CIoU on non-decoded values; forgetting that predictions for the loss must be decoded **boxes** while the assigner uses the same decoded boxes detached.

### Step T4. Overfit gate (do not skip)

**Protocol:**
1. Take 8 images (with their boxes) from LOCO. Turn **off** augmentation. Batch size 8.
2. Train the one-to-many branch only first, then both. lr 1e-3 (AdamW is easier for this test than SGD), no weight decay, 300–1000 steps.
3. Expected: total loss decays by over 90%, training-set mAP50 ≥ 0.95, and visualised predictions sit on the boxes.
4. Then check the **one-to-one** branch: at inference, for each object there should be one high-score box and no duplicates. Count predictions with score > 0.3 per image versus GT count.

**If it fails**, work through the debugging playbook in section 11.2 before touching hyperparameters. Do not start real training until this passes.

---

## 6. Data pipeline

### Step D1. Dataset reader (COCO format)
**Spec:** for each image return `image (H,W,3 uint8)` and `labels [n,5] = (cls, x1, y1, x2, y2)` in original pixels. Convert COCO `[x, y, w, h]` to `xyxy`; drop boxes with `w ≤ 1` or `h ≤ 1`; apply the class map (config `class_map`); skip `iscrowd`. Cache the parsed annotation list in memory (not the images).
**Test:** counts match the annotation file; a known image's boxes round-trip; `class_map` honoured.

### Step D2. Letterbox (must match inference exactly)
**Spec** [S]: `r = S / max(h, w)`; resize to `(round(w·r), round(h·r))` (bilinear); centre-pad to `S×S` with value 114; padding offsets are `pad_x = (S − new_w)/2`, `pad_y = (S − new_h)/2`, **floored to int for placement** (this is what `palletvision_infer.preprocess.letterbox` does); boxes: `x·r + pad_x`, `y·r + pad_y` using the same floored pads.
**Test:** **parity test** against `letterbox` from `palletvision_infer.preprocess` on random images: identical pixels and identical box mapping (`restore_boxes` inverts it). A mismatch here silently costs accuracy.
**Note:** the model package must not import cv2/numpy. Keep the training letterbox in `palletvision-train` and guard it with the parity test.

### Step D3. Augmentation (training only)
Order [S/D]: mosaic (probability 1.0 until the last epochs) → random affine → HSV jitter → horizontal flip → to tensor.
- **Random affine:** scale gain sampled in `[1 − 0.5, 1 + 0.5]`, translate ±0.1 of the image size, rotation 0°, shear 0°, no perspective, around the image centre; border fill 114. Transform the box corners and take the axis-aligned box of the result.
- **Box filtering after any geometric augmentation:** keep a box if its width and height are > 2 px, its remaining area is > 10% of its area before the transform, and its aspect ratio < 100.
- **HSV:** random gains `h ±0.015`, `s ±0.7`, `v ±0.4` applied in HSV space.
- **Flip:** horizontal with p=0.5, vertical never (pallets have a floor).
- **Mosaic:** pick 4 images, scale each, place them around a random centre `(xc, yc)` uniform in `[0.5S, 1.5S]` on a `2S×2S` canvas, then the affine step crops/rescales the canvas back to `S×S`. Concatenate and clip all labels. **Turn mosaic off for the final ~10 epochs** so the model finishes on realistic frames.
- No augmentation at validation: plain letterbox only.
**Test:** boxes still tightly enclose the objects after each transform (draw them); a zero-strength configuration is the identity; the filter drops degenerate boxes; seeded augmentations are reproducible.
**Pitfalls:** box clipping at the image border distorting aspect ratio; flipping boxes incorrectly (`x1' = W − x2`); applying colour jitter to the pad region.

### Step D4. Collation
Pad labels to the batch max `M`; produce `gt_labels [B,M,1]`, `gt_boxes [B,M,4]`, `mask_gt [B,M,1]`, and `images [B,3,S,S]` float32 in [0,1], RGB (note: OpenCV loads BGR; convert to RGB to match `letterbox` in infer).

### Step D5. Mixed real+synthetic sampling (LOCO + SORDI)
- Write `data/sordi.yaml` and an adapter that outputs the same label format after mapping SORDI categories to `pallet` (decide mapping from the real folder contents).
- Sampling: a weighted sampler giving real images the majority share of each epoch (for example 70% LOCO / 30% SORDI) rather than concatenating datasets (SORDI is far larger).
- **Validate only on the real LOCO validation split.** Never on SORDI.
- Run LOCO-only and LOCO+SORDI with identical settings and compare held-out LOCO mAP (section 8).
- **Built:** `configs/data/loco_sordi.yaml` (`data=loco_sordi`), `palletvision_train/sordi.py` (reads a scene folder, `images/<n>.jpg` plus `labels/json/<n>.json`, mapping `pallet` and `stillage` to `pallet` as LOCO does) and `MixedSampler`, which makes each epoch every real image once plus a fresh random draw of synthetic images (default share 0.3).

---

## 7. Trainer and evaluation

### Step R1. Optimiser and parameter groups [C/D]
- Three groups: (1) conv/linear weights with weight decay, (2) BatchNorm weights **no** decay, (3) all biases **no** decay.
- **MuSGD (YOLO26's optimiser) [P: report Sec. 3.3.1, Tables S2 and S4]:** `lr0 = 0.01`, `lrf = 0.01`, momentum 0.937 (the report prints 0.94 and says its values are rounded to two decimals), weight decay 5e-4 as the report prints it; Ultralytics scales it by batch / 64, so at the report's batch of 128 the effective value is 1e-3, which is what the config uses, warmup 1 epoch, all as in the report's pretraining recipe (Table S2). It applies a weighted mix of a Muon update and an SGD update to the multi-dimensional weights (convolution kernels, linear weights) and pure SGD to the 1-D parameters (biases, norm scales), which is exactly the group split above. The report records the mix as **Muon 0.45, SGD 0.55** for YOLO26n (Table S4; 0.50/0.50 for s). Its ablation against plain SGD from scratch on COCO: +0.4 mAP with a shorter schedule (Table 4). The Muon update itself is not spelled out in the report; the repo follows Jordan's public Muon reference, as Ultralytics does (see O3). **Plain SGD** (`lr0 = 0.01`, Nesterov) and **AdamW** (`lr0 ≈ 1e-3`, `betas = (0.937, 0.999)`; easier for the overfit test) stay available for comparisons.
- Gradient clipping at max-norm 10.0.
- Nominal batch size 128 (the report's batch): accumulate `round(128 / batch)` steps so the effective batch is stable across hardware. On the M5 Pro a real batch of 128 needs about 80 GB (memory grows about 0.6 GB per image, and larger batches are slower per image), so it is 8 batches of 16.

### Step R2. Learning-rate schedule
- **Warmup** for the first epoch (the report's Table S2 uses 1 for n; the guide's earlier 3 came from YOLOv5) but never fewer than 100 iterations: LR rises linearly from 0 (biases from 0.1 down to `lr0`; the 0.1 is our choice, not given in the report) and momentum from 0.8 to its full value.
- **Decay:** linear (cosine is an option; the report does not say) from `lr0` to `lr0 · lrf` with `lrf = 0.01`.
**Test:** schedule values at epoch 0, mid warmup, mid training, last epoch.

### Step R3. EMA (exponential moving average of weights)
Keep a shadow copy `θ_ema`. After every optimiser step: `θ_ema = d·θ_ema + (1−d)·θ` with `d = 0.9999 · (1 − exp(−updates / 2000))` (ramps so early EMA is not stuck on random init). Update BN buffers by copy. **Evaluate and export the EMA weights.**
**Test:** with constant weights EMA equals them; EMA lags a moving parameter.

### Step R4. Mixed precision and devices
- CUDA: `autocast(float16)` with `GradScaler`.
- MPS (laptop): `autocast` with float16 or bfloat16 [verify the current torch version]; `GradScaler` support on MPS may be limited. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Keep the loss, assigner and CIoU in float32 (cast predictions with `.float()`).
- `num_workers` on macOS: 2–4 with `persistent_workers=True`; `pin_memory` only on CUDA.
- Develop at 512 input (any multiple of 32); train/export at 640.

### Step R5. Checkpoints, resume, logging
Save `last.pt` every epoch and `best.pt` by validation mAP50-95: model weights, EMA weights, optimiser, scheduler, epoch, RNG state, config. `--resume` restores everything. Log per-step loss components and per-epoch LR, mAP50, mAP50-95 to CSV (and optionally TensorBoard/MLflow).

### Step R6. Evaluation (`pv-eval`)
1. Run the model in eval mode on the validation split with letterbox only.
2. **Use a very low score threshold (0.001) and keep up to 300 detections per image** (this is what mAP expects; do not threshold at 0.25).
3. Map boxes back to original image coordinates (`restore_boxes`), convert to COCO `[x, y, w, h]`, and write the COCO results list (`image_id, category_id, bbox, score`).
4. Run `pycocotools` `COCOeval` (`iouType="bbox"`). Report mAP50, mAP50-95, AP by size (small/medium/large), and recall.
5. Also report precision/recall/F1 versus score threshold and pick the **operating threshold** for deployment from the F1 curve (this is the `conf` used by `pv-infer`).
**Test:** a perfect-prediction fixture gives AP 1.0; an empty prediction list gives 0; image-id and category-id mappings are right.
**Pitfalls:** evaluating at the wrong resolution; forgetting to un-letterbox; leaving BN in train mode; using the training class ids instead of dataset category ids.

---

## 8. Experiments and fine-tuning

### 8.1 Experiment plan
| Run | Data | Purpose |
| --- | --- | --- |
| E0 | 8 images | overfit gate |
| E1 | LOCO train → LOCO val | main stage-1 model |
| E2 | LOCO + SORDI subset (same settings) | does synthetic data help real mAP? |
| E3 | ablations on the better of E1/E2 | one-to-one only vs dual head; with/without mosaic; no activation vs softplus vs exp; detach vs no detach; α,β; loss gains |
Keep seeds fixed and record config + git commit with every run. Compare on the same validation split.
Epochs: start with 100 for E1 to see the curve (the design estimate for the laptop is about 2–3 h for 5.6k images × 100 epochs; replace with a measured one-epoch time). Extend if validation mAP is still rising.

### 8.2 Stage 2: pallet types A/B/C/unknown (when HMGMA footage exists)
From `design.md`: load the stage-1 checkpoint, keep backbone, neck and box branches, **re-initialise only the final class convs** (`nc` 1 → 4, in both branches), freeze the backbone for the first epochs with a lower LR (about 0.1× the stage-1 `lr0`), then unfreeze. Train at deployment resolution. Split data by clip (ideally by day), never by frame. Evaluate per-type AP, confusion matrix and **track-level** type accuracy after voting (step K7). Footage must stay on lab/PACE hardware.

---

## 9. Optional advanced parts (specified, implement after the baseline works)

### O1. STAL (small-target-aware assignment): now part of the baseline (see T2, step 1)
**Problem:** at stride 8 a GT smaller than about 8 px may contain **no** anchor centre, so it gets zero positives and is never learned.
**Mechanism** [P: YOLO26 report, Sec. 3.3.3, Eq. 4-6 and Sec. 4.1]: for the candidate test only, each side of a GT below `s_min` (8 px) is clamped to `s_ref` (16 px); the original box is kept for scoring, targets and box regression. On COCO (Table 6, YOLO11s) `s_ref = 16` gives +0.2 AP overall and small-object AP 29.0 → 29.6, while `s_ref = 8` or `32` do not help.
**Test:** see T2 (the small-GT case yields at least one candidate; large GTs are unchanged). **Ablation:** compare with and without STAL on LOCO's small boxes (about 1.6% of pallet boxes have a longest side under 8 px at 640).

### O2. ProgLoss (progressive loss balancing): now part of the baseline (see T3)
The branch-weight schedule is specified in T3 [P: Eq. 2-3]. The other description sometimes seen (moving weight from classification to box loss) is not what the YOLO26 report describes.

### O3. MuSGD: now part of the baseline (see R1)
The Muon update, as implemented in `palletvision_train/optim.py` after Jordan's public reference and Ultralytics' MuSGD (the report gives only the idea and the mix weights):
- For each multi-dimensional weight, keep a running average of the gradient `a ← β·a + (1−β)·g` and form the Nesterov blend `u = (1−β)·g + β·a`. Flatten `u` to `[c_out, c_in·k·k]` and orthogonalise it with 5 Newton-Schulz iterations: normalise by the Frobenius norm, transpose if tall, then `X ← aX + (bA + cA²)X` with `A = XXᵀ` and `(a, b, c) = (3.4445, −4.7750, 2.0315)`. The result has singular values roughly between 0.5 and 1.5, not exactly 1. Scale by `sqrt(max(1, rows/cols))` of the last two dimensions of the original weight (1 for the square kernels here).
- Apply `−lr · w_muon · scale · NS(u)` (no weight decay), then the ordinary Nesterov-SGD step with weight decay at `lr · w_sgd` on the weights as they stand after the Muon part. Biases and norm scales get plain SGD at the full `lr`.
- `w_muon = 0.45`, `w_sgd = 0.55` for n [P: Table S4]. With `w_muon = 0` and `w_sgd = 1` it is exactly PyTorch SGD, which is how the repo tests it.
**Test:** the Newton-Schulz output has singular values in the right band and is scale-free; one step equals the hand-computed Muon part plus SGD part; with no Muon weight it equals `torch.optim.SGD`; biases and norm scales match plain SGD.

## 10. ByteTrack

### 10.1 What tracking adds
The detector gives independent boxes per frame. A tracker links them into **tracks with stable IDs** (tracking-by-detection). The zone engine needs IDs: "this pallet entered zone 2" requires knowing it is the same pallet across frames.

### 10.2 Pieces
A **Kalman filter** predicts where each track will be in the next frame; **IoU** measures how well a detection overlaps a predicted box; the **Hungarian algorithm** finds the best one-to-one matching; **ByteTrack's key idea** is to also use low-confidence detections in a second matching round, which recovers occluded objects instead of discarding them. [P: ByteTrack]

### Step K1. Data structures
- `Detection`: `box` (xyxy, float), `score`, `cls`.
- `Track`: `track_id`, Kalman `mean [8]` and `covariance [8×8]`, `state ∈ {New, Tracked, Lost, Removed}`, `is_activated`, `start_frame`, `frame_id` (last update), `tracklet_len`, `score` (last), class vote accumulator (K7).
- Track IDs come from a global counter, never reused.

### Step K2. Box parametrisation and Kalman filter [S: SORT/DeepSORT/ByteTrack]
**Measurement** `z = (x, y, a, h)`: box centre `(x,y)`, aspect ratio `a = w/h`, height `h`. **State** `s = (x, y, a, h, vx, vy, va, vh)` (8-D), constant-velocity model with `dt = 1` frame.
- Transition `F`: identity with `F[i, i+4] = dt` for `i = 0..3`. Observation `H = [I₄ | 0]`.
- Noise weights: `w_p = 1/20` (position), `w_v = 1/160` (velocity). All scaled by the current box height `h` (so noise is proportional to object size):
  - Initial covariance (diagonal of std): `[2w_p·h, 2w_p·h, 1e-2, 2w_p·h, 10w_v·h, 10w_v·h, 1e-5, 10w_v·h]`, squared on the diagonal. Initial state: the measurement for positions, zeros for velocities.
  - Process noise `Q` (std): positions `[w_p·h, w_p·h, 1e-2, w_p·h]`, velocities `[w_v·h, w_v·h, 1e-5, w_v·h]`, squared on the diagonal (computed from the current mean's `h`).
  - Measurement noise `R` (std): `[w_p·h, w_p·h, 1e-1, w_p·h]`, squared.
- **Predict:** `s' = F s`, `P' = F P Fᵀ + Q`.
- **Update:** `S = H P' Hᵀ + R`, `K = P' Hᵀ S⁻¹`, `s = s' + K (z − H s')`, `P = (I − K H) P'`. Solve with a Cholesky factorisation instead of an explicit inverse; use float64.
- Vectorise `predict` for all tracks at once (stack means/covariances).
**Test:** a constant-velocity trajectory is tracked with error → 0; velocity estimate converges to the true velocity; predicting `k` steps ahead moves the box by `k·v`; covariance stays symmetric positive definite; noise scales with `h`.
**Pitfalls:** mixing `(x,y)` as top-left vs centre; aspect ratio `a` as `h/w`; forgetting to recompute `Q` from the *current* height; negative or zero heights after many predictions (clamp).

### Step K3. IoU cost and assignment
- **Cost matrix:** `cost[i, j] = 1 − IoU(predicted_box_i, detection_j)`. Optionally *fuse score*: `cost = 1 − IoU·score_j` (this slightly favours confident detections) [P/C]. Compute IoU for all pairs vectorised.
- **Hungarian matching:** `scipy.optimize.linear_sum_assignment` on the cost matrix, then **reject pairs whose cost exceeds a threshold**; unmatched rows/columns are returned as unmatched tracks/detections. (For the later C++ port, replace scipy with a small Jonker–Volgenant implementation; keep the interface `assign(cost, thresh) → (matches, unmatched_rows, unmatched_cols)`.)
**Test:** known 3×3 cases; empty matrices (no tracks or no detections) return the right unmatched lists; threshold enforced.

### Step K4. The ByteTrack algorithm [P], per frame
Inputs: detections `D` for the frame (boxes in xyxy, scores). Maintain `tracked` (activated tracks), `lost`, `removed` pools.

1. **Split detections:** `D_high` = score > `high_thresh`; `D_low` = `low_thresh` < score ≤ `high_thresh`. Drop the rest.
2. **Predict:** run the Kalman prediction for all tracks in `tracked ∪ lost`.
3. **First association:** match `tracked ∪ lost` (the "pool") to `D_high`, cost = IoU distance (score-fused), threshold `match_thresh` (cost ≤ 0.8 ⇒ IoU ≥ 0.2). Matched tracked tracks are updated and stay tracked; matched lost tracks are **re-activated** (state → Tracked).
4. **Second association (the ByteTrack step):** take the pool tracks that are **still unmatched and were Tracked** (not Lost) and match them to `D_low` with a strict threshold (cost ≤ 0.5 ⇒ IoU ≥ 0.5). Matches update the track. Leftover tracks (still unmatched) are marked **Lost**.
5. **Unconfirmed tracks:** tracks created last frame that have not been confirmed yet (`is_activated = False`) are matched to the **remaining** `D_high` (cost ≤ 0.7). Matched → confirmed. Unmatched → **Removed** (a one-frame false positive disappears).
6. **New tracks:** each remaining `D_high` with score > `new_track_thresh` (= `high_thresh + 0.1`) starts a new track (state Tracked, **not activated** unless it is the first frame of the video; it becomes activated when matched in the next frame). Detections that are not matched and not strong enough are discarded.
7. **Lost-track expiry:** a Lost track not re-activated for more than `max_time_lost` frames becomes **Removed**.
8. **Output:** all **activated** Tracked tracks, with the Kalman-filtered box (converted back to xyxy), track id, score and class vote.

### Step K5. Parameters and frame-rate scaling
| Parameter | Start value | Meaning / how to tune for pallets |
| --- | --- | --- |
| `high_thresh` | 0.5–0.6 | detections above this start/continue tracks |
| `low_thresh` | 0.1 | lowest score used in the second association |
| `new_track_thresh` | `high_thresh + 0.1` | stops weak detections spawning IDs |
| `match_thresh` (first) | 0.8 (cost) | lower = stricter matching; raise if boxes jump |
| second-stage thresh | 0.5 (cost) | keep strict |
| unconfirmed thresh | 0.7 (cost) | |
| `max_time_lost` | **seconds × fps** | the original sets it to `fps/30 × 30 frames` (1 s of video). For slow, long-stationary pallets use **5–10 s** (at 10 fps: 50–100 frames) |
| `min_hits` to emit events | 2–3 | require N consecutive confirmations before a track can drive zone events |
Kalman velocity noise is tuned for pedestrians; pallets move slowly, so a smaller velocity noise or a damped velocity for stationary tracks reduces drift during occlusion (see K8).
**The detector's confidence calibration decides the useful thresholds.** Choose `high_thresh` from the F1/precision-recall curve of your detector (step R6), not from MOT17 defaults.

### Step K6. Tracker interface
```
Tracker.update(frame_id: int, detections: Detections) -> list[TrackedObject]
TrackedObject: track_id, box (xyxy), score, object_type, hits, age, is_lost: bool
Tracker.reset()
```
`Detections` is the existing dataclass (`boxes`, `scores`, `classes`) from `palletvision_infer.detections`. Keep the tracker free of torch and of repo-specific IO so it can be tested with plain arrays and ported later.

### Step K7. Class handling: class-agnostic association + type voting
- Association ignores class (a flickering class must not break an ID).
- Each track keeps a weight per class: on every matched detection `w[cls] += score`. The track's `object_type` is `argmax w`; if the best class's share of total weight is below a margin (for example < 0.6), report `pallet_unknown`.
- Stage 1 has one class, so this is trivial; build and test it now for stage 2.
**Test:** a track with classes `A A B A A` reports `A`; a 50/50 split reports unknown; type changes only after enough evidence.

### Step K8. Pallet-specific behaviour (part of the baseline, not optional)
1. **Long lost buffer** (K5) because pallets sit still and reappear in the same place.
2. **Stationary handling:** if a track's recent displacement is below a small threshold for several frames, zero its velocity so the prediction does not drift while it is occluded; resume normal velocity when it moves again.
3. **Reappear-at-same-spot merge:** when a new track starts within a short window and within IoU ≥ ~0.5 of a recently removed track's last box, reuse the old ID (cheap ID-switch repair) [design.md].
4. **Min hits before events** (K5) and **hysteresis** in the zone engine both suppress flicker; tune them together.

### Step K9. Optional upgrade: OC-SORT ideas [P: OC-SORT]
Add only if ID switches after occlusion are a measured problem.
- **OCR (observation-centric recovery):** after the normal rounds, try to match unmatched tracks' *last observed* box to the remaining detections.
- **ORU (observation-centric re-update):** when a lost track is re-associated, replace the drifting Kalman state by re-running the filter over a **virtual trajectory** that linearly interpolates between the last real observation and the new one.
- **OCM (observation-centric momentum):** add a cost term for direction consistency, computed from observations separated by a few frames rather than from the filter's velocity.

### Step K10. Integration with the zone engine
- Map each output track to the existing `Track(track_id, object_type, box)` and call `ZoneEngine.update(tracks)`.
- **Only confirmed, currently-visible tracks drive membership.** Lost (occluded) tracks keep their last confirmed zone state; do not feed predicted boxes as if observed.
- **Lost-track policy (decide and log):** `ZoneEngine.remove_track` currently emits an `exit` for every zone the track was confirmed in. That is wrong for a pallet that was merely occluded or whose ID switched. Options: (a) when a track expires inside a zone, emit the exit only after a long timeout and mark the event as `lost`; (b) never emit an exit on expiry, only mark the visit as open; (c) emit and later reconcile. Pick one, test it against the golden fixtures, and add the new behaviour to `tests/golden`.
- Events are created with `source="model"` through `EventFactory`.

### Step K11. Tracker tests
- **Unit:** Kalman (K2), IoU/assignment (K3), lifecycle transitions (New → Tracked → Lost → Removed), type vote (K7).
- **Golden JSON fixtures** in `tests/golden/tracker/`, same style as the zone fixtures: per-frame detections in, expected `(frame, track_id, box≈)` out, hand-verified. Scenarios: single object moving; two objects crossing paths; a 1-second occlusion gap (ID preserved); a stationary object then moving; a low-score detection recovered by the second round (the ByteTrack case); a one-frame false positive never confirmed; a track expiring after `max_time_lost`.
- **Synthetic generator:** boxes with random linear motion, Gaussian jitter, random miss rate (10–30%), random false positives, scripted occlusions. Assert IDs switches == 0 on clean scenarios and a bounded count on noisy ones.
- **Metrics on real clips** (once HMGMA clips with ID labels exist): export tracks in MOT format (`frame, id, x, y, w, h, conf, -1, -1, -1`) and score with **TrackEval** (HOTA, IDF1, MOTA, ID switches). The product metric is ID switches per minute and dwell-time error (`design.md`).

---

## 11. Putting it together, debugging, decisions

### 11.1 Pipeline glue (`pv-run`)
`FrameSource(fps=10) → detector backend.detect → Tracker.update → ZoneEngine.update → EventFactory.make → sink.write`, plus an overlay (boxes, track IDs, zone polygons) via the existing `draw`.
- **Timestamps:** when processing a video file faster than real time, the wall clock is wrong. Give `EventFactory` a clock based on the **frame timestamp** (`start_time + frame_index / fps`), otherwise dwell times are meaningless.
- Reuse `Outbox`/`PostgresSink` and `ZoneConfigStore.latest` for zones; stamp `config_version` on events.
- Budget: at 10 fps there are 100 ms per frame; the zone engine and tracker should be well under 5 ms; the detector dominates. Decode/preprocess/infer in separate threads only when profiling shows it matters.
- Flags (suggested): `--source`, `--weights`, `--conf`, `--fps`, `--tracker-config`, `--camera-id`, `--device-id`, `--sink`, `--out`, `--show`.
**Test:** a short synthetic video with a moving rectangle that crosses a zone edge produces the expected enter/exit events end to end (use a stub detector returning the known box).

### 11.2 Debugging playbook (detector)
| Symptom | Most likely causes |
| --- | --- |
| Loss `NaN` | LR too high without warmup; `atan(w/h)` with h→0; log of zero in IoU; fp16 overflow in loss (cast to float32); division by zero in normalisation |
| Loss decreases but nothing detected | scores never rise: check cls bias init (too negative is fine, +ve is bad), class index off by one, sigmoid applied twice, `n_targets` normalisation wrong |
| Boxes in the right place but shifted | missing `+0.5` anchor offset; letterbox pad mismatch; box conventions (xyxy vs xywh) mixed |
| Boxes systematically too small/large | stride multiply forgotten; distances not in stride units; activation saturating |
| Overfit works, real training stalls | LR/warmup, augmentation too strong (try off), label mapping errors, too many tiny boxes |
| Duplicate boxes at inference | one-to-one branch not trained with the one-to-one assignment (`topk=7`, then `topk2=1`), or you are decoding the one-to-many branch |
| mAP much lower than visual quality | evaluating with a high conf threshold; wrong un-letterbox; BN in train mode; wrong category ids |
| Train mAP fine, val poor | leakage-free split? overfitting; weak augmentation; wrong preprocessing at eval |
| Training very slow on MPS | assigner/top-k ops falling back to CPU; use CPU assigner or smaller `M`; fewer workers contention |
| ONNX mismatch | model not in `eval()`; wrong branch exported; dynamic ops; opset > 13 |
Helpful tools: draw assigned positives (anchor points + GT) for 8 images; plot score histograms for positives vs background; log gradient norms per module; assert shapes at every module boundary in debug builds.

### 11.3 Definition of done (whole project)
- [ ] Param/FLOP counts within tolerance; shapes match the tables.
- [ ] Contract test: `[1,300,6]`, opset 13, ONNX Runtime parity.
- [ ] Assigner/IoU/loss/EMA/schedule unit tests green; letterbox parity test green.
- [ ] Overfit gate passed (loss −90%, mAP50 ≥ 0.95, no duplicates).
- [ ] LOCO validation mAP50 and mAP50-95 reported with the exact config and commit; LOCO vs LOCO+SORDI compared.
- [ ] Operating threshold chosen from the F1 curve.
- [ ] Tracker unit tests and golden fixtures green; synthetic scenarios pass.
- [ ] `pv-run` on a pallet video produces sensible events in Postgres; lost-track policy decided and tested.
- [ ] `docs/yolo26-notes.md` (provenance + decision log) complete.

### 11.4 Decision log template
`Date | Decision | Options considered | Why | Evidence (run/metric) | Source tag`
Log at least: class mapping, α/β, loss gains, activation for distances, detach-or-not, mosaic schedule, `high_thresh`, `max_time_lost`, lost-track policy.

---

## 12. References (verify identifiers when you open them)
- YOLO26 report (primary source for the architecture, ProgLoss, STAL and the L1 loss): Jocher et al., "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models", arXiv 2606.03748 (`docs/YOLO_26.pdf`).
- YOLO26 analysis: arXiv 2601.12882 (NMS-free end-to-end, DFL removal, ProgLoss, STAL, MuSGD).
- YOLOv10 (consistent dual assignments, NMS-free training): arXiv 2405.14458.
- TOOD, task-aligned one-stage detection: arXiv 2108.07755.
- FCOS (anchor-free per-cell detection): arXiv 1904.01355.
- GIoU: arXiv 1902.09630. DIoU/CIoU: arXiv 1911.08287.
- Generalized Focal Loss (the DFL idea that YOLO26 removes): arXiv 2006.04388.
- SORT: arXiv 1602.00763. ByteTrack: arXiv 2110.06864. OC-SORT: arXiv 2203.14360.
- HOTA metric: arXiv 2009.07736. TrackEval: github.com/JonathonLuiten/TrackEval.
- Muon optimiser notes (K. Jordan): search "Muon optimizer" blog and repository (MIT).
- Ultralytics public docs (documentation pages only, not source): YOLO26 model page and the training/augmentation settings pages, for the default hyperparameters tagged [D]/[C] above.
- LOCO: github.com/tum-fml/loco. SORDI: sordi.ai (access and licence not verified).
- Repo: `docs/design.md`, `contracts/model_io.yaml`, `tests/golden/zones`.

## 13. Suggested build order (commits)
1. M1 config → M2 conv → M3 blocks → M4 SPPF/attention → M5 backbone → M6 neck (each with tests).
2. M7 head → M8 anchors/decode → M9 top-K postprocess → M10 export parity + counts.
3. T1 IoU → T2 assigner → T3 loss → T4 overfit gate.
4. D1 reader → D2 letterbox + parity → D3 augmentation → D4 collate → D5 mixed sampling.
5. R1–R3 optimiser/schedule/EMA → R4 devices → R5 checkpoints → R6 evaluation → E1 LOCO run → E2 +SORDI → E3 ablations.
6. K1–K3 tracker core → K4–K5 ByteTrack + params → K6–K8 interface, voting, pallet behaviour → K11 tests → K10 zone integration + policy → 11.1 `pv-run`.
7. Optional: K9 OC-SORT; stage 2 fine-tune when footage exists. (STAL, ProgLoss and MuSGD are part of the baseline: T2, T3 and R1.)
