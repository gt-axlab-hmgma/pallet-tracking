# One command for a fresh Windows machine: installs uv if needed, sets up the environment, and
# runs pv-job. With no arguments it runs the whole overnight job.
#   powershell -ExecutionPolicy Bypass -File scripts\pallet-job.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\pallet-job.ps1 run --smoke
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (the Python tool this project uses)..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv sync
if ($args.Count -eq 0) { uv run pv-job run } else { uv run pv-job @args }
exit $LASTEXITCODE
