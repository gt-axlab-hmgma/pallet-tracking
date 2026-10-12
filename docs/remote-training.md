# Running the training job on another machine

For a lab PC, a gaming laptop, a rented GPU or a cluster. The person at the machine runs **one command** and sends back **one folder** (`results-<name>/`, also zipped).

**Data rule:** this job only ever downloads *public* data (LOCO, SORDI). HMGMA footage is under NDA: it never goes through this job and never goes onto a personal or rented machine. The results folder is built from a fixed list of file names (model weights, logs, curves, scores), so no image or video can end up in it.

**Status.** The whole job (download check, training, scoring, plotting, bundling) has run end to end on a Mac, on real data and on generated data. The Windows `.ps1`/`.bat`, the Slurm template, the Kaggle download and several GPUs have **not** run on real hardware yet: that is what the smoke run below is for. If it fails, send the `pallet-tracking/runs/job-<name>/train.log` and say where it stopped.

## What the person needs

- A machine with an NVIDIA GPU (recommended), an Apple-silicon Mac, or a CPU (very slow).
- About **100 GB free** for the full data (LOCO 1 GB, the SORDI scenes 5 to 21 GB each; `pv-job check` tells the exact need).
- A free **Kaggle account and API token** for the SORDI scenes: kaggle.com > Settings > API > Create New Token, saved as `~/.kaggle/kaggle.json` (on Windows `C:\Users\<name>\.kaggle\kaggle.json`). LOCO needs no account.
- The code. If the repository is private, make a zip on a machine that has access and send it: `git archive --format=zip -o pallet-tracking.zip HEAD`. (A zip has no git history, so the run records the commit as unknown; set `PALLETVISION_COMMIT=<hash>` before running to record it.)

## Steps

1. **Prove the machine first (about 5 minutes plus the first download of LOCO and one 4 GB scene):**

   | Machine | Command |
   | --- | --- |
   | Mac | double-click `scripts/pallet-smoke.command` (first time: right-click > Open) |
   | Windows | double-click `scripts\pallet-smoke.bat` |
   | Linux | `bash scripts/pallet-job.sh run --smoke` |

   It installs `uv` if needed, sets up the environment, fetches the small data, trains for a few steps, scores, plots and writes a bundle. With `--synthetic` it needs no data or account at all and only proves the install: `bash scripts/pallet-job.sh run --smoke --synthetic`.
2. **The real job (hours to days):**

   | Machine | Command |
   | --- | --- |
   | Mac | double-click `scripts/pallet-job.command` |
   | Windows | double-click `scripts\pallet-job.bat` |
   | Linux | `bash scripts/pallet-job.sh` |

   It downloads whatever data is missing, runs the overnight recipe (LOCO + all SORDI scenes at a 50% synthetic share, 150 epochs of 5,640 images, the paper's optimiser settings), restarts from the last checkpoint if it crashes, and finishes by scoring the best model on the LOCO and SORDI validation and test splits.
3. **Send back** the folder it prints at the end: `runs/results-<name>/` (or the `.zip` beside it).

Useful options (add them after the script name; `bash scripts/pallet-job.sh run --epochs 50`):

| Option | Meaning |
| --- | --- |
| `--batch-size N` | images per step, per GPU (default 16). Lower it if the GPU is small; the trainer accumulates to the nominal batch of 128 so the update is the same. A crash from running out of memory before the first checkpoint halves it automatically. |
| `--epochs N`, `--share F` | length of the run and the synthetic share of every epoch. |
| `--gpus N` or `--gpus all` | several CUDA GPUs in one run (torchrun). **Not yet run on real GPUs**: smoke it first. |
| `--workers N` | data-loading processes (default: half the cores, 2 to 8). |
| `--full-eval` | at the end score *every* SORDI val/test image (about 18 minutes each on a Mac) instead of a fixed 5,000-image draw. LOCO is always scored in full. |
| `--no-fetch` | never download (offline compute nodes). |
| `--name X`, `--resume` | the run folder is `runs/job-X`; with `--resume` an interrupted run continues. Without it a folder that already holds a checkpoint is refused, so a run is never overwritten. |

`bash scripts/pallet-job.sh check` lists what is missing without changing anything; `fetch` downloads it.

## What comes back

`best.pt` (use this), `last.pt`, `metrics.jsonl`, `curves.png`, `eval_{loco,sordi}_{val,test}.json`, `train.log`, `eval.log`, `doctor.txt` (speed and memory of this machine), `environment.json`, `config.json` (exact settings), `manifest.json` (git commit, command, times, scores, and image counts per SORDI scene and split, so two machines can be shown to have trained on the same data) and a `README.txt` describing each.

Keep the machine awake: the job asks the system not to sleep on a Mac and on Windows while it runs; plug in a laptop and set a high-performance power plan.

## Fine-tuning from pretrained weights

`bash scripts/pallet-job.sh run --init-from <checkpoint.pt>` starts the run from another checkpoint's weights (weights only, not the optimiser or epoch; tensors whose shape differs, like a class head of another size, start fresh). It is the way to fine-tune a model pretrained on a larger dataset. **Which dataset to pretrain on is a licence question: read `docs/pretraining.md` first.**

## Clusters (Slurm)

- Compute nodes often have **no internet**: run `bash scripts/pallet-job.sh fetch` on the login node first. If `data/` has to live on scratch, make `data/` a symlink to it (Windows: `mklink /J data D:\scratch\data`).
- Edit the `#SBATCH` lines of `scripts/slurm/pallet-job.sbatch` (GPUs, CPUs, time, `module load cuda/...`), smoke it, then submit the real one; the commands are in the file's header.
- Multi-GPU training is written and unit-tested on the CPU only. Read the smoke job's output (and `doctor.txt`) before spending 24 hours.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `no Kaggle credentials` | save the API token as `~/.kaggle/kaggle.json` (step above) |
| `N GB of free disk needed` | free space, or point `data/` at a bigger drive |
| `PyTorch here is a CPU-only build` (in `doctor.txt`) | see `docs/cuda-setup.md`; on Windows run `uv sync --reinstall-package torch` |
| `... already has a run in it` | add `--resume` to continue it, or `--name <new>` for a fresh one |
| it stops with "training did not finish" | look at the end of `runs/job-<name>/train.log`; running the same command again resumes |
