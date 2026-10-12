# Training on an NVIDIA GPU

For a gaming laptop (RTX 50-series), a rented GPU or a cluster node. **Status: written and unit-tested on a Mac. The CUDA path has not yet been run on real CUDA hardware**, so the first job on a new machine is the check in step 4; if something fails there, that is the thing to report.

**Data rule:** HMGMA footage is under NDA and stays on lab-controlled storage. A personal laptop or a rented machine gets public data only (LOCO, SORDI, COCO).

## Windows (native)

1. **Driver.** Install a current NVIDIA driver (CUDA 12.8 needs roughly driver 570 or newer; RTX 50-series cards need that build of PyTorch).
2. **Tools.** Install git and uv: `winget install Git.Git astral-sh.uv`. No separate CUDA toolkit is needed: the PyTorch wheel bundles what it uses.
3. **Get the code and the environment.**
   ```powershell
   git clone <repo-url>; cd pallet-tracking
   uv sync        # the project's lock file already points Windows at the CUDA 12.8 build of torch
   ```
4. **Check the machine.**
   ```powershell
   uv run pv-doctor --batch 8
   uv run pytest packages/palletvision-train/tests/test_cuda.py
   ```
   `pv-doctor` prints the GPU, its memory and which precisions it supports, then times real training steps in float32 and in mixed precision. Read its warnings first:

   | Message | Meaning and fix |
   | --- | --- |
   | `PyTorch here is a CPU-only build` | `uv sync` did not take the CUDA wheel. Run `uv sync --reinstall-package torch`. |
   | `No usable GPU was found` | The driver is missing or too old; check `nvidia-smi`. |
   | `This GPU is sm_120 but this PyTorch build only has [...]` | The PyTorch build is older than the card (RTX 50-series needs CUDA 12.8 or newer). |

   The pytest line runs the tests marked `needs_gpu` (mixed precision on the GPU, resume). They are skipped on a machine without CUDA.
5. **Pick a batch size that fits.** Try `--batch 8`, `16` and `32` and watch the peak memory. An 8 GB card may only fit 8 or 16 at 640 px. Use the largest that fits; the trainer accumulates gradients up to the nominal batch of 128 (`train.nominal_batch`), so a smaller batch gives the same update, only slower. Set it with `train.batch_size=8`.
6. **Data.** Download LOCO and the SORDI scenes (see `data/README.md`) into `data/`. SORDI is about 88 GB for the five scenes in `configs/data/loco_sordi.yaml`.
7. **A short real run before a long one.**
   ```powershell
   uv run pv-train data=loco train.epochs=1 train.max_steps=50 train.batch_size=8
   ```
8. **Mixed precision.** `train.amp=true` runs the forward and backward pass in 16 bits (bfloat16 on an Ampere or newer card, float16 with loss scaling otherwise; `train.amp_dtype` overrides it). The weights, the loss and the optimiser stay in float32. `pv-doctor` shows the speed gain. **Its effect on accuracy is not validated yet:** compare a short run's held-out mAP with and without it before relying on it (issue #38).

## Running for days

- **Resume is built in.** `train.resume=auto` continues from `<output_dir>/last.pt`, so a crash or a Windows restart costs at most one epoch. A small loop restarts it by itself:
  ```powershell
  for ($i = 0; $i -lt 20; $i++) {
    uv run pv-train data=loco_sordi train.val_every=5 train.resume=auto train.output_dir=runs/long
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 60
  }
  ```
- **Keep it awake and cool.** Plugged in, a high-performance power plan, sleep and hibernate off, Windows Update's active hours or a pause set, the fans clear. If the laptop has a battery-care setting (Razer's charge limit, for example) use it: a battery held at 100% for days ages quickly.
- **Watch it.** `uv run pv-plot runs/long` draws the curves from `metrics.jsonl` at any time.

## Linux, a rented GPU or a cluster

`uv sync` already installs a CUDA build of torch from PyPI. Run `uv run pv-doctor` first and set `data.num_workers` to about the number of CPU cores per GPU (the augmentation needs roughly 4 cores for every 350 images a second). Use `tmux` or `nohup`, with `train.resume=auto`. Download the public datasets onto the machine directly.

## Not done yet

- **Several GPUs in one run.** Training uses one GPU. A distributed version (DDP, `torchrun`) is separate work; until then, several GPUs are best used for one experiment each.
- **Accuracy with mixed precision** is not validated (see step 8).
