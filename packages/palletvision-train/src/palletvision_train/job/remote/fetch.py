"""Fetch the public training data (LOCO and the SORDI scenes) onto a fresh machine, and check that
what is on disk is complete.

Only public data comes through here. HMGMA footage is under NDA and never goes through a
downloader or onto a machine that is not lab-controlled.

- LOCO: one anonymous 769 MB zip of images from TUM, plus five annotation files from its GitHub
  repository.
- SORDI: each scene is a Kaggle dataset (``sordi-ai/industrial-scene-NN-dataset``). Kaggle needs
  the operator's own credentials, so this part cannot be fully automatic; ``pv-job check`` says
  what is missing before a long run starts.
"""

import json
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from palletvision_train.datasets.coco import load_records
from palletvision_train.datasets.sordi import SPLITS, scene_folders, split_of
from palletvision_train.job.remote.download import Download, download_resumable

LOCO_ZIP_URL = "https://go.mytum.de/239870"  # redirects to TUM's anonymous download of dataset.zip


LOCO_ANNOTATION_URL = "https://raw.githubusercontent.com/tum-fml/loco/main/rgb/"


LOCO_ANNOTATIONS = (
    "loco-sub1-v1-val.json",
    "loco-sub2-v1-train.json",
    "loco-sub3-v1-train.json",
    "loco-sub4-v1-val.json",
    "loco-sub5-v1-train.json",
)


LOCO_IMAGES = 5593  # every image in the zip, annotated or not


LOCO_GB = 0.8


SORDI_IMAGES = {
    "01": 46_268, "02": 12_941, "03": 41_393, "04": 48_501, "05": 99_108,
    "06": 75_625, "07": 38_494, "08": 29_600, "09": 52_339,
}  # fmt: skip


SORDI_GB = {
    "01": 11.9, "02": 4.0, "03": 8.0, "04": 9.8, "05": 21.1,
    "06": 17.0, "07": 6.6, "08": 5.4, "09": 11.3,
}  # fmt: skip


Runner = Callable[[Sequence[str]], int]


def scene_number(root: Path | str) -> str:
    """``data/sordid-03`` -> ``03``."""
    return Path(root).name.removeprefix("sordid-")


def image_count(root: Path) -> int:
    """The images of a SORDI scene folder (all its DR levels); 0 when it is absent or broken."""
    try:
        return sum(len(list((f / "images").glob("*.jpg"))) for f in scene_folders(root))
    except FileNotFoundError:
        return 0


def sordi_split_counts(root: Path, fractions: Sequence[float]) -> dict[str, int]:
    """Images per train/val/test in a scene, from file names alone (no image is opened).

    Two machines that trained on the same data report the same numbers, so the bundle's
    manifest lets you check that.
    """
    counts = dict.fromkeys(SPLITS, 0)
    for folder in scene_folders(root):
        prefix = folder.relative_to(root.parent).as_posix()
        for image in (folder / "images").glob("*.jpg"):
            counts[split_of(f"{prefix}/{image.name}", fractions)] += 1
    return counts


def loco_present(data_dir: Path) -> bool:
    """True when the five annotation files parse and every image they list is on disk."""
    root = data_dir / "loco"
    files = [root / "annotations" / name for name in LOCO_ANNOTATIONS]
    if not all(f.is_file() for f in files):
        return False
    try:
        load_records(files, root, {"pallet": "pallet"}, ["pallet"])  # raises on a missing image
    except (OSError, ValueError, KeyError):
        return False
    return True


def missing_scenes(roots: Sequence[str], repo: Path) -> list[str]:
    """Scene roots (as listed in the config) that are absent or have fewer images than expected."""
    short = []
    for root in roots:
        expected = SORDI_IMAGES.get(scene_number(root))
        if image_count(repo / root) < (expected if expected is not None else 1):
            short.append(root)
    return short


def kaggle_credentials() -> bool:
    """Whether the Kaggle command line can log in: a kaggle.json, an access token or env vars."""
    home = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return bool(
        (home / "kaggle.json").is_file()
        or (home / "access_token").is_file()
        or os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )


def gigabytes_needed(missing: Sequence[str], loco_missing: bool) -> float:
    """Free disk space needed to fetch what is missing: the downloads plus room to unpack the
    biggest one (the zip is deleted only after it has been unpacked)."""
    sizes = [SORDI_GB.get(scene_number(root), 15.0) for root in missing]
    if loco_missing:
        sizes.append(LOCO_GB)
    return sum(sizes) + max(sizes, default=0.0)


def extract_loco(archive: Path, data_dir: Path) -> int:
    """Unpack LOCO's ``dataset/subset-N/...`` into ``data/loco/subset-N/...``; returns the images.

    Refuses any member that would land outside ``data/loco``.
    """
    root = (data_dir / "loco").resolve()
    images = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            relative = member.filename.removeprefix("dataset/")
            target = (root / relative).resolve()
            if root not in target.parents:
                raise ValueError(f"{member.filename} would be written outside {root}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            images += target.suffix.lower() == ".jpg"
    return images


def fetch_loco(data_dir: Path, fetch: Download = download_resumable) -> None:
    """Download and unpack LOCO into ``data/loco``, annotations included.

    The zip goes to ``data/_download/loco.zip`` (kept if the run is interrupted, so the next run
    resumes it) and is deleted once unpacked.
    """
    root = data_dir / "loco"
    archive = data_dir / "_download" / "loco.zip"
    print(f"downloading LOCO images (about {LOCO_GB * 1000:.0f} MB)")
    fetch(LOCO_ZIP_URL, archive)
    images = extract_loco(archive, data_dir)
    if images < LOCO_IMAGES:
        raise RuntimeError(f"LOCO zip held {images} images, expected {LOCO_IMAGES}")
    archive.unlink()
    for name in LOCO_ANNOTATIONS:
        fetch(LOCO_ANNOTATION_URL + name, root / "annotations" / name)
        json.loads((root / "annotations" / name).read_text())  # a truncated file fails here


def hoist_scene(folder: Path) -> None:
    """Make ``folder`` itself the scene when an archive wrapped it in one extra directory."""
    try:
        scene_folders(folder)
        return
    except FileNotFoundError:
        pass
    children = [c for c in folder.iterdir() if c.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"{folder} does not look like a SORDI scene after unpacking")
    (inner,) = children
    for item in inner.iterdir():
        shutil.move(str(item), folder / item.name)
    inner.rmdir()
    scene_folders(folder)  # raises if it still is not one


def kaggle_command() -> list[str]:
    """The Kaggle command line: an installed ``kaggle``, else run through uv without installing."""
    found = shutil.which("kaggle")
    return [found] if found else ["uvx", "kaggle"]


def fetch_sordi(root: Path, run: Runner | None = None) -> None:
    """Download one SORDI scene from Kaggle into ``root`` (e.g. ``data/sordid-03``)."""
    if not kaggle_credentials():
        raise RuntimeError(
            "no Kaggle credentials: create an API token at kaggle.com/settings and save it as "
            "~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY)"
        )
    number = scene_number(root)
    runner = run or (lambda cmd: subprocess.run(cmd, check=False).returncode)
    root.mkdir(parents=True, exist_ok=True)
    slug = f"sordi-ai/industrial-scene-{number}-dataset"
    code = runner(
        [*kaggle_command(), "datasets", "download", "-d", slug, "-p", str(root), "--unzip"]
    )
    if code != 0:
        raise RuntimeError(f"kaggle download of {slug} failed (exit code {code})")
    hoist_scene(root)
    expected = SORDI_IMAGES.get(number)
    found = image_count(root)
    if expected is not None and found < expected:
        raise RuntimeError(f"{root} has {found} images after download, expected {expected}")
