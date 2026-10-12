"""``pv-job`` from the command line: every stage starts with ``python -m``, and a whole run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from palletvision_train.job.cli import main


@pytest.mark.parametrize("module", ["doctor", "eval_cli", "plot", "job", "cli"])
def test_every_stage_can_be_started_with_python_dash_m(module: str) -> None:
    """The job starts its stages this way: a module without a ``__main__`` guard does nothing."""
    done = subprocess.run(
        [sys.executable, "-m", f"palletvision_train.{module}", "--help"],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    shown = (done.stdout + done.stderr).lower()
    assert done.returncode == 0 and ("usage" in shown or "hydra" in shown)  # Hydra words it so


def test_a_synthetic_smoke_run_produces_the_bundle(tmp_path: Path) -> None:
    main(
        [
            "run",
            "--smoke",
            "--synthetic",
            "--name",
            "t",
            "--runs-dir",
            str(tmp_path),
            "--workers",
            "0",
        ]
    )
    folder = tmp_path / "results-t"
    assert (tmp_path / "results-t.zip").exists()
    names = {p.name for p in folder.iterdir()}
    assert {"last.pt", "metrics.jsonl", "curves.png", "manifest.json", "environment.json",
            "config.json", "train.log", "doctor.txt", "README.txt"} <= names  # fmt: skip
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["epochs_done"] == 1 and manifest["synthetic"] and "commit" in manifest["git"]
    assert "float32" in (folder / "doctor.txt").read_text()  # the doctor really ran
    with pytest.raises(SystemExit, match="--resume"):  # and a second run does not overwrite it
        main(
            [
                "run",
                "--smoke",
                "--synthetic",
                "--name",
                "t",
                "--runs-dir",
                str(tmp_path),
                "--workers",
                "0",
            ]
        )
