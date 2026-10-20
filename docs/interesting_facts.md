# Interesting facts about the Pallet Vision build

A short list of the things in this repo that are worth knowing about, each one checkable. Every number comes from the repo, its tests or its PRs (as of 2026-10-11).

**The project:** pallet detection, tracking and zone analytics for the HMGMA plant. A from-scratch YOLO26-style detector, a zone engine, and an edge-to-database event pipeline for many Jetson Nanos.

## At a glance

| | |
| --- | --- |
| Activity | 95 commits, 7 merged PRs, 20 issues (11 closed), 2026-10-06 to 2026-10-11 |
| Code | 6 Python packages (about 2,700 lines of source, about 2,600 of tests) and a Next.js app (about 2,200 lines of TypeScript) |
| Tests | 282 on `main`; 30 of them run against a real Postgres in CI |
| CI | ruff lint and format, **strict mypy**, pytest with a Postgres 16 service, package build; UI format, lint, tests and build |

## The facts

### 1. A detector built from scratch that matches the published model sizes exactly
Backbone, PAN neck, two-branch end-to-end head (one-to-many for training, one-to-one for NMS-free inference), anchors, box decoding and top-300 post-processing, all in plain PyTorch with no detection library. With 80 classes the parameter totals are **2,572,280 (n) and 10,009,784 (s)**, identical to the published models. A wrong width, repeat count, skip connection or head layout would change that number, so it is strong evidence the architecture is faithful. At 640 the deployed n model is 5.31 GFLOPs with one class.
*Evidence:* `packages/palletvision-model/tests/test_param_counts.py`, PRs #5, #17, #27.

### 2. Found errors in the project's own reference spec by checking primary sources
While building it, five places where the written guide disagreed with the paper's figures and the published counts were found and recorded (inner Bottleneck width, an SPPF residual, the neck's block types and attention, the head's bias initialisation, whether box distances get an activation). Most were not guessable from the guide.
*Evidence:* issue #16.

### 3. ONNX export engineered around real hardware limits
The Jetson Nano's TensorRT 8.2 parser caps ONNX at **opset 13**, so the attention block is hand-written (PyTorch's built-in needs opset 14) and is tested against PyTorch's own result. One post-processing op was rewritten from integer floor-division, which exports as seven sign-correction ops including `Mod`, to a float floor that exports as two. A test bans ops the parser may reject (`Einsum`, `If`, `Loop`, `Scan`, `Mod`) and checks the training-only branch is not in the deployed graph.
*Evidence:* `tests/test_yolo26_export.py`, branch `xjm/m10-onnx-export`.

### 4. Careful debugging of a misleading test failure
The first ONNX-versus-PyTorch comparison showed a 560 px box error. The cause was not a decode bug: an untrained network in eval mode shrinks its signal to about 1e-9, so all top-300 scores tied and the two runtimes broke ties differently. The fix was to calibrate BatchNorm statistics first and compare in pieces. A float64 reference showed the exported graph is as accurate as PyTorch's own float32 (both differ from float64 by the same amount).

### 5. Data due diligence before any training
The LOCO check found that annotations were missing at first, the official train and validation splits, and a path trap (`file_name` is only a basename and image ids differ between files, so images must be found through `path`). It also found 496 unannotated images on disk, about 1.6% of pallet boxes under 8 px at 640 input, and that grouped (never random) cross-validation is needed because frames come from continuous video.

### 6. A durable edge-to-database event pipeline
Jetsons never talk to the database directly. Events go to a local **SQLite outbox**, and a sync agent (`pv-sync`) forwards them in batches with retry and backoff, so a network outage loses nothing. Event ids are generated on the device and inserts are idempotent (`ON CONFLICT DO NOTHING`), so a retried batch cannot duplicate. The events table is partitioned, and zone configs are immutable and versioned. Tests assert batched inserts run over 1,000 events/s and at least twice as fast as row by row, and check index use.
*Evidence:* `packages/palletvision-events`, PRs #1 and #2.

### 7. A zone engine built to be portable and fast
Point-in-polygon with hysteresis (an object must stay in or out for K frames before an event fires), a ground-point rule, concave zones and overlapping zones. Behaviour is pinned by **language-neutral JSON golden fixtures** so a later C++ runtime can reuse them, and a performance test keeps 20 tracks across 10 zones under a 1 ms per-frame budget (target 0.5 ms).
*Evidence:* `packages/palletvision-zones`, `tests/golden/zones/`.

### 8. A working full-stack demo
FastAPI with live server-sent events and a Next.js UI where you draw multi-corner zones, drag mock bounding boxes across them, and watch join and leave events stream into Postgres, with warnings when a zone version was not stored.

### 9. Engineering discipline
Conventional commits, pre-commit hooks, strict typing (it caught 32 type errors in new code before CI would have), and an issue-driven workflow: scoped issues with "done when" criteria, PRs that close them, and a documented backlog for the rest of the build. Design docs include a 55 KB step-by-step implementation guide and diagrams written for non-technical readers.

## Current limits

- **The model is untrained.** There are no accuracy numbers (no mAP), and no loss, assigner or data reader yet. The NMS-free behaviour appears only after training with one-to-one assignment.
- **Nothing has been built on a Jetson.** The ONNX graph is checked against TensorRT's known limits, but no TensorRT engine has been built yet.
- **Parity tolerance is loose for an untrained network** (about 2e-2 of the value range). Tighter numbers apply to a trained model.
- **Not built yet:** the HTTP ingest service, the C++ runtime. The ByteTrack tracker exists (`palletvision-track`) and drives the browser demo, but is not on a Jetson.
- **How it was built:** the recent commits were pair-programmed with Claude Code (see the `Co-Authored-By` trailers).
