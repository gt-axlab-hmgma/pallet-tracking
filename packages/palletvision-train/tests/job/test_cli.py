"""``pv-job`` from the command line: every stage starts with ``python -m``, and a whole run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from palletvision_train.cli.train import REPO
from palletvision_train.job.cli import main


@pytest.mark.parametrize(
    "module", ["tools.doctor", "cli.eval", "evaluation.plot", "job", "cli.train"]
)
def test_every_stage_can_be_started_with_python_dash_m(module: str) -> None:
    """The job starts its stages this way: a module without a ``__main__`` guard does nothing."""
    done = subprocess.run(
        [sys.executable, "-m", f"palletvision_train.{module}", "--help"],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    shown = (done.stdout + done.stderr).lower()
    assert done.returncode == 0 and ("usage" in shown or "hydra" in shown)  # Hydra words it so


# A whole mini training job (doctor, train, evaluate, plot, bundle) in subprocesses: 15 s on a
# laptop but about 7 minutes on a 2-core CI runner, so it runs in the weekly slow workflow instead.
@pytest.mark.slow
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


def test_share_is_uniform_by_default_or_a_number_below_one() -> None:
    from palletvision_train.job.cli import parse

    assert parse(["run"]).share is None and parse(["run"]).epoch_images == 5640
    assert parse(["run", "--share", "uniform"]).share is None
    assert parse(["run", "--share", "0.5", "--epoch-images", "1000"]).share == 0.5
    for bad in ("1", "-0.1", "abc"):
        with pytest.raises(SystemExit):
            parse(["run", "--share", bad])


def test_scenes_are_numbers_and_unknown_ones_are_refused(tmp_path: Path) -> None:
    from palletvision_train.job.cli import main, parse

    assert parse(["run", "--scenes", "2, 01,4"]).scenes == ["02", "01", "04"]
    assert parse(["run"]).scenes is None and parse(["run", "--pool-images", "50"]).pool_images == 50
    for bad in ("", "x", "02,,x"):
        with pytest.raises(SystemExit):
            parse(["run", "--scenes", bad])
    with pytest.raises(SystemExit, match="not in configs"):
        main(["run", "--scenes", "99", "--synthetic", "--runs-dir", str(tmp_path)])


def test_the_big_run_options_parse() -> None:
    from palletvision_train.job.cli import parse

    a = parse(
        ["run", "--budget-hours", "15", "--amp", "--epoch-images", "all", "--scenes", "02,06"]
    )
    assert (a.budget_hours, a.amp, a.epoch_images, a.scenes) == (15.0, True, None, ["02", "06"])
    assert parse(["run"]).speed_margin == 0.6 and parse(["fetch", "--scenes", "2"]).scenes == ["02"]
    with pytest.raises(SystemExit):
        parse(["run", "--epoch-images", "0"])


def test_a_sized_run_cannot_be_a_smoke_or_synthetic_run(tmp_path: Path) -> None:
    from palletvision_train.job.cli import main

    with pytest.raises(SystemExit, match="real run"):
        main(["run", "--synthetic", "--budget-hours", "5", "--runs-dir", str(tmp_path)])


def test_set_can_be_repeated() -> None:
    from palletvision_train.job.cli import parse

    a = parse(["run", "--set", "train.lr=0.005", "--set", "data.augment.close_mosaic=2"])
    assert (
        a.extra == ["train.lr=0.005", "data.augment.close_mosaic=2"] and parse(["run"]).extra == []
    )


def test_a_partial_epoch_is_a_fraction_of_the_train_set() -> None:
    from palletvision_train.job.cli import parse

    assert parse(["run", "--epoch-fraction", "0.5"]).epoch_fraction == 0.5
    assert parse(["run"]).epoch_fraction is None


@pytest.mark.skipif(not (REPO / "data/loco/annotations").exists(), reason="LOCO data not present")
def test_a_sized_run_makes_the_train_set_and_the_epoch_from_the_options(tmp_path: Path) -> None:
    from argparse import Namespace

    from palletvision_train.job.cli import size_run
    from palletvision_train.job.run import JobConfig

    def sized(**options: object) -> JobConfig:
        job = JobConfig(name="n", run_dir=tmp_path, epochs=8)
        given = {"budget_hours": None, "train_images": 4000, "epoch_fraction": None, **options}
        size_run(job, Namespace(**given))
        return job

    full = sized()
    assert full.pool_images == 4000 - 2820 and full.epoch_images is None
    assert full.plan["views"] == 4000 * 8 and full.share is None
    half = sized(epoch_fraction=0.5)
    assert half.pool_images == 1180 and half.epoch_images == 2000
    assert half.plan["epoch_images"] == 2000 and half.plan["views"] == 2000 * 8
    with pytest.raises(SystemExit, match="exceed LOCO"):
        size_run(
            JobConfig(name="n", run_dir=tmp_path),
            Namespace(budget_hours=None, train_images=100, epoch_fraction=None),
        )
    with pytest.raises(SystemExit, match="epoch-fraction"):
        sized(epoch_fraction=1.5)
