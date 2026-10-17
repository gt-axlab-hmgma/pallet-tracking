"""The train/val/test split of LOCO (by site) and SORDI (by hash of the image name)."""

from pathlib import Path

import pytest
from test_sordi import CLASS_MAP, CLASSES, make_scene, obj

from palletvision_train.cli.train import REPO
from palletvision_train.datasets.coco import loco_reader
from palletvision_train.datasets.sordi import (
    SPLITS,
    load_sordi_records,
    sordi_records,
    split_of,
    subsample,
)

HAS_LOCO = (REPO / "data/loco/annotations").exists() and (REPO / "data/loco/subset-1").exists()


def test_a_key_always_lands_in_the_same_split() -> None:
    assert split_of("sordid-01/DR-1/7") == split_of("sordid-01/DR-1/7")
    # fixed values: a changed hash would silently move images between train and test
    pattern = "".join(split_of(f"sordid-01/DR-1/{n}")[0] for n in range(40))
    assert pattern == "ttttttttttttttttttvvttvttttttttttttttttt"  # t(rain), v(al), t(est) per image


def test_the_fractions_come_out_close_to_70_15_15() -> None:
    names = [split_of(f"sordid-05/DR-2/{n}") for n in range(20000)]
    shares = {s: names.count(s) / len(names) for s in SPLITS}
    assert shares["train"] == pytest.approx(0.70, abs=0.015)
    assert shares["val"] == pytest.approx(0.15, abs=0.015)
    assert shares["test"] == pytest.approx(0.15, abs=0.015)


def test_other_fractions_are_followed_and_bad_ones_rejected() -> None:
    names = [split_of(f"k/{n}", (0.0, 0.5, 0.5)) for n in range(2000)]
    assert "train" not in names and 0.45 < names.count("val") / 2000 < 0.55
    with pytest.raises(ValueError, match="summing to 1"):
        split_of("k", (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="summing to 1"):
        split_of("k", (0.7, 0.3))


def test_the_split_does_not_depend_on_what_else_is_in_the_data() -> None:
    """Adding a scene must not move an image that already had a split (why it is a hash)."""
    before = {n: split_of(f"sordid-01/DR-1/{n}") for n in range(300)}
    _ = [split_of(f"sordid-09/DR-1/{n}") for n in range(300)]  # a new scene shows up
    assert before == {n: split_of(f"sordid-01/DR-1/{n}") for n in range(300)}


def test_the_same_image_number_in_two_levels_or_scenes_is_split_independently() -> None:
    a = [split_of(f"sordid-01/DR-1/{n}") for n in range(400)]
    b = [split_of(f"sordid-01/DR-2/{n}") for n in range(400)]
    c = [split_of(f"sordid-02/{n}") for n in range(400)]
    assert a != b and a != c


def test_the_three_splits_of_a_scene_are_disjoint_and_complete(tmp_path: Path) -> None:
    scene = {str(n): [obj("pallet", 5, 6, 25, 16)] for n in range(200)}
    root, prefix = make_scene(tmp_path / "sordid-01", scene), "sordid-01"
    parts = {}
    for name in SPLITS:
        records = load_sordi_records(root, CLASS_MAP, CLASSES, split=name, key_prefix=prefix)
        parts[name] = {r.path.name for r in records}
    assert parts["train"] | parts["val"] | parts["test"] == {f"{n}.jpg" for n in range(200)}
    assert not parts["train"] & parts["val"] and not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]
    assert all(parts.values())


def test_without_a_split_every_image_is_read(tmp_path: Path) -> None:
    root = make_scene(tmp_path / "sordid-01", {str(n): [] for n in range(30)})
    assert len(load_sordi_records(root, CLASS_MAP, CLASSES)) == 30


def test_an_unknown_split_is_an_error(tmp_path: Path) -> None:
    root = make_scene(tmp_path / "s", {"1": []})
    with pytest.raises(ValueError, match="split must be"):
        load_sordi_records(root, CLASS_MAP, CLASSES, split="dev")


def test_levels_of_a_scene_are_split_by_level_and_image(tmp_path: Path) -> None:
    scene = tmp_path / "sordid-01"
    for level in ("DR-1", "DR-2"):
        make_scene(scene / level, {str(n): [] for n in range(100)})
    kept = {s: sordi_records(["sordid-01"], CLASS_MAP, CLASSES, s, repo=tmp_path) for s in SPLITS}
    assert sum(len(v) for v in kept.values()) == 200
    names = {s: {(r.path.parts[-3], r.path.name) for r in v} for s, v in kept.items()}
    assert not names["train"] & names["test"]
    assert any(level == "DR-2" for level, _ in names["val"] | names["test"])


def test_a_fixed_subsample_is_stable_and_keeps_the_order(tmp_path: Path) -> None:
    root = make_scene(tmp_path / "s", {str(n): [] for n in range(50)})
    records = load_sordi_records(root, CLASS_MAP, CLASSES)
    first, second = subsample(records, 10), subsample(records, 10)
    assert first == second and len(first) == 10
    assert [r.path for r in first] == sorted(
        (r.path for r in first), key=lambda p: (len(p.stem), p.stem)
    )
    assert subsample(records, None) == records and subsample(records, 500) == records


@pytest.mark.skipif(not HAS_LOCO, reason="LOCO data not present")
def test_loco_splits_are_whole_sites_and_do_not_overlap() -> None:
    sites = {
        s: {
            part
            for r in loco_reader(s).records
            for part in r.path.parts
            if part.startswith("subset-")
        }
        for s in ("train", "val", "test")
    }
    assert sites == {
        "train": {"subset-2", "subset-3", "subset-5"},
        "val": {"subset-1"},
        "test": {"subset-4"},
    }
