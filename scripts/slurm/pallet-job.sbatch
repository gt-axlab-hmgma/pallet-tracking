#!/usr/bin/env bash
# Slurm template: edit the #SBATCH lines for your cluster, then
#   1. on the LOGIN node (compute nodes often have no internet):  bash scripts/pallet-job.sh fetch
#      and, if data/ must live on scratch, make data/ a symlink to it first.
#   2. prove the GPUs first (minutes):   sbatch --export=ALL,PV_ARGS="run --smoke --no-fetch --gpus all" scripts/slurm/pallet-job.sbatch
#   3. the real job:                     sbatch scripts/slurm/pallet-job.sbatch
# Multi-GPU training has not yet been run on real GPUs: read the smoke output before step 3.
#SBATCH --job-name=pallet-job
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Load whatever your cluster needs for CUDA here, for example:  module load cuda/12.8
export PATH="$HOME/.local/bin:$PATH"
uv sync

# --batch-size and --workers are per GPU: 32 images x 4 GPUs is the paper's batch of 128.
uv run pv-job ${PV_ARGS:-run --no-fetch --gpus all --batch-size 32 --workers 5} --name "slurm-${SLURM_JOB_ID:-local}"
