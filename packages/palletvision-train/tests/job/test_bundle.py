"""The results bundle: an allowlist of files, so nothing else can leave the machine."""

import json
import zipfile
from pathlib import Path

import pytest

from palletvision_train.job.bundle import build_bundle, git_info, summarise_evals


def test_the_bundle_holds_only_allowlisted_files(tmp_path: Path) -> None:
    run = tmp_path / "job-n"
    run.mkdir()
    for name in ("best.pt", "last.pt", "metrics.jsonl", "eval_loco_val.json", "train.log"):
        (run / name).write_text("x")
    for stray in ("frame.jpg", "clip.mp4", "data.zip", "notes.txt", "optimizer-dump.bin"):
        (run / stray).write_text("must not leave the machine")
    (run / "images").mkdir()
    (run / "images" / "1.jpg").write_text("x")
    archive = build_bundle(run, tmp_path / "results-n")
    with zipfile.ZipFile(archive) as zf:
        names = {n.split("/", 1)[1] for n in zf.namelist()}
    assert names == {
        "best.pt",
        "last.pt",
        "metrics.jsonl",
        "eval_loco_val.json",
        "train.log",
        "README.txt",
    }
    assert {p.name for p in (tmp_path / "results-n").iterdir()} == names


def test_a_run_missing_optional_files_still_bundles(tmp_path: Path) -> None:
    run = tmp_path / "r"
    run.mkdir()
    archive = build_bundle(run, tmp_path / "results.v1-host")  # dots in the name are fine
    assert archive.name == "results.v1-host.zip" and archive.exists()


def test_eval_numbers_are_summarised(tmp_path: Path) -> None:
    (tmp_path / "eval_loco_test.json").write_text(
        json.dumps({"images": 3, "map50_95": 0.123456, "map50": 0.5, "best_f1": 0.4, "extra": 1})
    )
    assert summarise_evals(tmp_path) == {
        "loco_test": {"images": 3, "map50_95": 0.1235, "map50": 0.5, "best_f1": 0.4}
    }


def test_a_source_zip_without_git_still_reports_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PALLETVISION_COMMIT", "abc123")
    assert git_info(tmp_path) == {"commit": "abc123"}
