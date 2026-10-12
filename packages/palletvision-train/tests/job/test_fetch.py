"""Fetching LOCO and the SORDI scenes, and checking they are complete."""

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_sordi import make_scene

from palletvision_train.job import fetch


def test_scenes_are_counted_and_compared_with_what_they_should_hold(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    make_scene(tmp_path / "data/sordid-02", {str(n): [] for n in range(5)})
    assert fetch.image_count(tmp_path / "data/sordid-02") == 5
    assert fetch.image_count(tmp_path / "data/sordid-99") == 0
    # 02 should hold 12,941 images, so five is incomplete; an unknown scene only has to exist
    assert fetch.missing_scenes(["data/sordid-02"], tmp_path) == ["data/sordid-02"]
    make_scene(tmp_path / "data/sordid-99", {"1": []})
    assert fetch.missing_scenes(["data/sordid-99", "data/sordid-98"], tmp_path) == [
        "data/sordid-98"
    ]


def test_split_counts_come_from_file_names(tmp_path: Path) -> None:
    make_scene(tmp_path / "sordid-02", {str(n): [] for n in range(300)})
    counts = fetch.sordi_split_counts(tmp_path / "sordid-02", (0.7, 0.15, 0.15))
    assert sum(counts.values()) == 300 and counts["train"] > counts["val"] > 0


def test_kaggle_credentials_are_found_in_a_file_or_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert not fetch.kaggle_credentials()
    (tmp_path / "kaggle.json").write_text("{}")
    assert fetch.kaggle_credentials()
    (tmp_path / "kaggle.json").unlink()
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    assert not fetch.kaggle_credentials()  # a name without a key is not enough
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert fetch.kaggle_credentials()


def test_disk_needed_is_the_downloads_plus_room_to_unpack_the_biggest() -> None:
    assert fetch.gigabytes_needed([], False) == 0
    assert fetch.gigabytes_needed(["data/sordid-02", "data/sordid-05"], False) == 4.0 + 21.1 + 21.1
    assert fetch.gigabytes_needed([], True) == fetch.LOCO_GB * 2


def test_the_loco_zip_is_unpacked_without_its_dataset_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "loco.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset/subset-1/a.jpg", "x")
        zf.writestr("dataset/subset-2/2019-11-25_0/Kinect/color/b.jpg", "y")
    assert fetch.extract_loco(archive, tmp_path / "data") == 2
    assert (tmp_path / "data/loco/subset-1/a.jpg").read_text() == "x"
    assert (tmp_path / "data/loco/subset-2/2019-11-25_0/Kinect/color/b.jpg").exists()


def test_a_zip_member_cannot_escape_the_data_folder(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset/../../outside.txt", "x")
    with pytest.raises(ValueError, match="outside"):
        fetch.extract_loco(archive, tmp_path / "data")
    assert not (tmp_path / "outside.txt").exists()


def test_fetching_loco_downloads_the_zip_and_the_five_annotation_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "LOCO_IMAGES", 2)
    asked: list[str] = []

    def fake(url: str, destination: Path) -> None:
        asked.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if url == fetch.LOCO_ZIP_URL:
            with zipfile.ZipFile(destination, "w") as zf:
                zf.writestr("dataset/subset-1/a.jpg", "x")
                zf.writestr("dataset/subset-1/b.jpg", "x")
        else:  # sub-set 1 lists the two images in the zip, the others none
            images = [] if "sub1" not in url else [
                {"id": i, "path": f"/dataset/subset-1/{n}.jpg", "width": 1, "height": 1}
                for i, n in enumerate("ab")
            ]  # fmt: skip
            destination.write_text(
                json.dumps({"images": images, "annotations": [], "categories": []})
            )

    (tmp_path / "data").mkdir()
    fetch.fetch_loco(tmp_path / "data", fake)
    assert len(asked) == 6 and fetch.LOCO_ZIP_URL in asked
    assert fetch.loco_present(tmp_path / "data")
    (tmp_path / "data/loco/subset-1/b.jpg").unlink()  # an annotated image that is not on disk
    assert not fetch.loco_present(tmp_path / "data")
    assert (tmp_path / "data/loco/annotations/loco-sub4-v1-val.json").exists()


def test_a_truncated_loco_zip_is_an_error(tmp_path: Path) -> None:
    def fake(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as zf:
            zf.writestr("dataset/subset-1/a.jpg", "x")

    (tmp_path / "data").mkdir()
    with pytest.raises(RuntimeError, match="expected 5593"):
        fetch.fetch_loco(tmp_path / "data", fake)


def test_a_wrapped_scene_is_hoisted_up_one_level(tmp_path: Path) -> None:
    make_scene(tmp_path / "sordid-03" / "industrial-scene-03" / "DR-1", {"1": []})
    fetch.hoist_scene(tmp_path / "sordid-03")
    assert fetch.image_count(tmp_path / "sordid-03") == 1
    (tmp_path / "bad" / "one").mkdir(parents=True)
    (tmp_path / "bad" / "two").mkdir()
    with pytest.raises(RuntimeError, match="does not look like"):
        fetch.hoist_scene(tmp_path / "bad")  # two unrelated children


def test_a_sordi_scene_needs_credentials_and_a_clean_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "none"))
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="no Kaggle credentials"):
        fetch.fetch_sordi(tmp_path / "sordid-77")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    monkeypatch.setenv("KAGGLE_USERNAME", "u")

    seen: list[list[str]] = []

    def failing(command: Sequence[str]) -> int:
        seen.append(list(command))
        return 1

    with pytest.raises(RuntimeError, match="failed"):
        fetch.fetch_sordi(tmp_path / "sordid-77", failing)
    assert "sordi-ai/industrial-scene-77-dataset" in seen[0] and "--unzip" in seen[0]

    def downloads(command: Sequence[str]) -> int:
        make_scene(tmp_path / "sordid-77" / "DR-1", {"1": [], "2": []})
        return 0

    fetch.fetch_sordi(tmp_path / "sordid-77", downloads)  # scene 77 has no expected count
