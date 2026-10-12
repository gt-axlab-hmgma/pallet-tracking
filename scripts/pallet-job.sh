#!/usr/bin/env bash
# One command for a fresh Mac or Linux machine: installs uv if needed, sets up the environment,
# and runs pv-job. With no arguments it runs the whole overnight job.
#   bash scripts/pallet-job.sh                  # the full run
#   bash scripts/pallet-job.sh run --smoke      # a few minutes: prove this machine first
#   bash scripts/pallet-job.sh check            # what is missing?
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (the Python tool this project uses)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv sync
if [ "$#" -eq 0 ]; then set -- run; fi
uv run pv-job "$@"
