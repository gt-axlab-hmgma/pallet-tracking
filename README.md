# HMGMA Pallet Vision

Pallet detection, tracking and zone analytics for HMGMA. A clean-room PyTorch detector is trained per frame; a tracker and zone engine (later milestones) turn detections into pallet events. See [docs/design.md](docs/design.md) for the full design.

**Status:** M0 skeleton. Training, export, benchmarking and video/webcam inference run end to end on a laptop using a placeholder `TinyDetector`. The real YOLO26 modules (M1) replace it behind the same output contract. Tracking, zones, Jetson/TensorRT and C++ are out of scope for now.

## Licensing

Proprietary, all rights reserved (see [LICENSE](LICENSE)). Do not install or import `ultralytics`, copy its source, or start from its weights; the model is our own reimplementation. Footage from HMGMA is under NDA and must never be committed.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 and every dependency from `uv.lock`.

```bash
uv sync --all-packages
uv run pytest

# train on generated data (no download needed), export, benchmark, run on a video
uv run pv-train data=synthetic data.size=64 data.num_samples=16 model.input_size=64 train.max_steps=4
uv run pv-export --checkpoint runs/stage1/last.pt --size 64 --out runs/export/model.onnx
uv run pv-bench --weights runs/export/model.onnx --size 64
uv run pv-infer --source data/videos/clip.mp4 --weights runs/export/model.onnx --size 64 --out runs/out.mp4

# laptop webcam, live window (press q to quit)
uv run pv-infer --source 0 --show
```

Omit `--weights` on `pv-infer` / `pv-bench` to use the dummy detector, which only checks the video plumbing.

On macOS, grant camera access to your terminal (System Settings > Privacy & Security > Camera) before using `--source 0`.

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

```bash
uv run pre-commit install   # ruff and file hygiene on every commit
uv run ruff check . && uv run ruff format --check . && uv run mypy packages && uv run pytest
```

CI runs the same checks on every push and pull request. Commits are small and conventional (`feat:`, `fix:`, `build:`, `chore:`, `docs:`, `test:`, `ci:`) so each is easy to revert.
