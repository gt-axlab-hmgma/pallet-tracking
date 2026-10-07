# HMGMA Pallet Vision

Pallet detection, tracking and zone analytics for HMGMA. A clean-room PyTorch detector is trained per frame; a tracker and zone engine (later milestones) turn detections into pallet events. See [docs/design.md](docs/design.md) for the full design.

**Status:** M0 skeleton. Training, export, benchmarking and video/webcam inference run end to end on a laptop using a placeholder `TinyDetector`. The real YOLO26 modules (M1) replace it behind the same output contract. Tracking, zones, Jetson/TensorRT and C++ are out of scope for now.

## Licensing

Proprietary, all rights reserved (see [LICENSE](LICENSE)). Do not install or import `ultralytics`, copy its source, or start from its weights; the model is our own reimplementation. Footage from HMGMA is under NDA and must never be committed.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 and every dependency from `uv.lock`.

```bash
uv sync --all-packages
uv run pytest
```

## Commands

The typical flow is **train → export → benchmark → infer**. Each step feeds the next:

```text
pv-train  ->  runs/stage1/last.pt  ->  pv-export  ->  model.onnx  ->  pv-bench / pv-infer
```

Use the same `size` everywhere: the value you trained with is the value you export, benchmark and infer with.

### `pv-train`: train a model

Trains the detector and saves a checkpoint to `<output_dir>/last.pt`. Arguments are [Hydra](https://hydra.cc) overrides written as `key=value` (defaults live in `configs/`).

```bash
uv run pv-train data=synthetic data.size=64 data.num_samples=16 model.input_size=64 train.max_steps=4
```

| Override | Default | Meaning |
| --- | --- | --- |
| `data` | `loco` | Dataset. Only `synthetic` works for now (generated boxes, no download). |
| `data.size` | `640` | Synthetic image size in pixels. |
| `data.num_samples` | `64` | Number of synthetic images. |
| `model.input_size` | `640` | Model input size. Keep equal to `data.size`. |
| `device` | `auto` | `auto`, `cpu`, `cuda` or `mps`. |
| `train.epochs` | `100` | Passes over the data. |
| `train.batch_size` | `16` | Images per step. |
| `train.lr` | `0.001` | Learning rate. |
| `train.amp` | `true` | fp16 autocast, only used on cuda/mps. |
| `train.max_steps` | `null` | Stop after this many steps. Handy for quick smoke runs. |
| `train.output_dir` | `runs/stage1` | Where `last.pt` is written. |

### `pv-export`: convert a checkpoint to ONNX

Produces the `.onnx` file that `pv-bench` and `pv-infer` load (opset 13, tensor names from `contracts/model_io.yaml`).

```bash
uv run pv-export --checkpoint runs/stage1/last.pt --size 64 --out runs/export/model.onnx
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint written by `pv-train`. |
| `--size` | `640` | Input size the model was trained with. |
| `--out` | `runs/export/model.onnx` | Output ONNX path. |

### `pv-bench`: measure speed

Runs the detector on a random frame and reports mean/p50/p99 latency and fps, saved as JSON.

```bash
uv run pv-bench --weights runs/export/model.onnx --size 64
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--weights` | none | ONNX model. Omit to benchmark the dummy detector. |
| `--size` | `640` | Input size. |
| `--warmup` | `10` | Untimed runs before measuring. |
| `--iters` | `100` | Timed runs. |
| `--out` | `runs/bench/latest.json` | Where the JSON result is written. |

### `pv-infer`: run on a video or webcam

Detects on each frame, draws boxes and fps, and optionally writes an annotated video or opens a live window.

```bash
# video file -> annotated video
uv run pv-infer --source data/videos/clip.mp4 --weights runs/export/model.onnx --size 64 --out runs/out.mp4

# laptop webcam, live window (press q to quit)
uv run pv-infer --source 0 --weights runs/export/model.onnx --size 64 --show
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | required | Video path, or a camera index (`0` is the laptop webcam). |
| `--weights` | none | ONNX model. Omit to use the dummy detector, which draws a fixed box and only checks the video plumbing. |
| `--out` | none | Save the annotated video here. |
| `--fps` | `10` | Process this many frames per second (extra frames are skipped). |
| `--conf` | `0.25` | Drop detections below this confidence. |
| `--size` | `640` | Model input size. |
| `--show` | off | Show a live window. |

On macOS, grant camera access to your terminal (System Settings > Privacy & Security > Camera) before using `--source 0`.

### Other useful commands

| Command | What it does |
| --- | --- |
| `uv run pytest` | Run all tests. |
| `uv run ruff check . && uv run ruff format --check .` | Lint and format check. |
| `uv run mypy packages` | Type check. |
| `uv run pre-commit install` | Run ruff and file checks automatically on every commit. |

## Layout

| Path | What |
| --- | --- |
| `packages/palletvision-model` | Pure `nn.Module`s. Imports only `torch`. |
| `packages/palletvision-train` | Trainer, datasets, ONNX export (`pv-train`, `pv-export`). |
| `packages/palletvision-infer` | Frame source, backends (torch, onnxruntime), overlay, benchmark (`pv-infer`, `pv-bench`). |
| `contracts/` | Model input/output and class definitions shared by every runtime. |
| `configs/` | Hydra config groups (`model`, `data`, `train`). |
| `data/` | Local datasets and videos. Git-ignored. |
| `experiments/` | One short README per experiment. |
| `docs/design.md` | Design document. |

## Laptop notes (Apple silicon)

- `device: auto` picks cuda, then mps, then cpu. Override with `device=cpu|cuda|mps`.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to CPU.
- fp16 autocast only on cuda/mps (`train.amp`); no DDP on mps.
- No Docker for local work: it cannot reach the Mac GPU or webcam. `uv.lock` and `.python-version` pin the environment.

## Datasets

| Dataset | Needed | Use |
| --- | --- | --- |
| [LOCO](https://github.com/tum-fml/loco) | Yes, for real stage-1 training | 5,593 real warehouse images in COCO format. Unpack to `data/loco/`; all pallet-like labels map to `pallet`. |
| A few short `.mp4` clips | Yes, for `pv-infer` demos | Put in `data/videos/`. |
| [SORDI](https://sordi.ai) | Optional | Small synthetic subset for extra stage-1 volume. |
| COCO subset (~10k) | Optional, M1 | Stage-0 check that the reimplementation matches published curves; run on a cluster. |
| HMGMA footage | Stage 2 only | Lab machine or PACE only, never this repo. |

The LOCO loader is not wired into `pv-train` yet; until then use `data=synthetic`.

## Development

CI runs lint, types, tests and a package build on every push and pull request. Commits are small and conventional (`feat:`, `fix:`, `build:`, `chore:`, `docs:`, `test:`, `ci:`) so each is easy to revert.
