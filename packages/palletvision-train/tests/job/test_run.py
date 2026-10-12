"""What a ``pv-job`` run is told to do, and how training restarts after a crash."""

from datetime import datetime
from pathlib import Path

import pytest

from palletvision_train.job.run import (
    JobConfig,
    default_name,
    default_workers,
    eval_plan,
    keep_awake_command,
    refuse_overwrite,
    train_command,
    train_overrides,
    train_with_restarts,
)


def job(tmp_path: Path, **kwargs: object) -> JobConfig:
    return JobConfig(name="n", run_dir=tmp_path / "job-n", roots=["data/sordid-02"], **kwargs)  # type: ignore[arg-type]


def test_the_full_run_is_the_overnight_recipe(tmp_path: Path) -> None:
    o = train_overrides(job(tmp_path, workers=6))
    assert {
        "data=loco_sordi",
        "data.sordi.share=null",
        "data.sordi.epoch_images=5640",
        "train.epochs=150",
        "train.val_every=5",
    } <= set(o)
    assert "train.resume=auto" in o and "data.num_workers=6" in o
    assert f"train.output_dir={tmp_path / 'job-n'}" in o
    assert not any(x.startswith("data.limit") for x in o)


def test_a_smoke_run_is_tiny_and_scores_every_epoch(tmp_path: Path) -> None:
    o = train_overrides(job(tmp_path, smoke=True))
    assert {"train.epochs=1", "train.val_every=1", "data.limit=32", "data.sordi.limit=64"} <= set(o)


def test_fine_tuning_starts_from_the_pretrained_weights(tmp_path: Path) -> None:
    o = train_overrides(job(tmp_path, init_from=Path("/x/pre.pt")))
    assert "train.init_from=/x/pre.pt" in o and "data=loco_sordi" in o
    assert "train.init_from=/x/pre.pt" in train_overrides(
        job(tmp_path, init_from=Path("/x/pre.pt"), smoke=True)
    )
    assert not any("init_from" in x for x in train_overrides(job(tmp_path)))


def test_a_synthetic_run_needs_no_data(tmp_path: Path) -> None:
    o = train_overrides(job(tmp_path, synthetic=True, smoke=True))
    assert o[0] == "data=synthetic" and not any("sordi" in x for x in o)
    assert eval_plan(job(tmp_path, synthetic=True)) == []


def test_several_gpus_start_one_process_each(tmp_path: Path) -> None:
    one, four = train_command(job(tmp_path)), train_command(job(tmp_path, gpus=4))
    assert "torch.distributed.run" not in one and "palletvision_train.cli" in one
    assert "torch.distributed.run" in four and "--nproc_per_node=4" in four


def test_synthetic_scoring_is_capped_unless_asked_for_everything(tmp_path: Path) -> None:
    plan = {(s, p): n for s, p, n in eval_plan(job(tmp_path))}
    assert plan[("loco", "val")] is None and plan[("loco", "test")] is None
    assert plan[("sordi", "val")] == 5000 and plan[("sordi", "test")] == 5000
    full = {(s, p): n for s, p, n in eval_plan(job(tmp_path, full_eval=True))}
    assert full[("sordi", "val")] is None


def test_workers_follow_the_cores_within_limits() -> None:
    assert [default_workers(c) for c in (1, 4, 8, 16, 64)] == [2, 2, 4, 8, 8]


def test_the_default_name_is_host_and_time() -> None:
    assert default_name(datetime(2026, 10, 12, 21, 5)).endswith("-20261012-2105")


def test_only_a_mac_needs_a_helper_to_stay_awake() -> None:
    assert keep_awake_command(42, "darwin") == ["caffeinate", "-dims", "-w", "42"]
    assert keep_awake_command(42, "linux") is None and keep_awake_command(42, "win32") is None


def test_a_fresh_run_never_starts_over_an_existing_one(tmp_path: Path) -> None:
    j = job(tmp_path)
    j.run_dir.mkdir()
    refuse_overwrite(j)  # an empty folder is fine
    (j.run_dir / "last.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="--resume"):
        refuse_overwrite(j)
    j.resume = True
    refuse_overwrite(j)


def test_training_restarts_after_a_crash_until_it_finishes(tmp_path: Path) -> None:
    j = job(tmp_path, retries=5)
    codes = iter([1, 1, 0])
    calls: list[int] = []

    def run(command: object, log: Path) -> int:
        calls.append(1)
        return next(codes)

    assert train_with_restarts(j, run, pause=0) == 3 and len(calls) == 3


def test_it_gives_up_after_the_allowed_attempts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="did not finish"):
        train_with_restarts(job(tmp_path, retries=2), lambda c, log: 1, pause=0)


def test_running_out_of_memory_before_a_checkpoint_halves_the_batch(tmp_path: Path) -> None:
    j = job(tmp_path, retries=4, batch_size=16)
    sizes: list[int] = []

    def run(command: object, log: Path) -> int:
        sizes.append(j.batch_size)
        log.parent.mkdir(parents=True, exist_ok=True)
        if len(sizes) < 3:
            log.write_text("RuntimeError: CUDA out of memory. Tried to allocate")
            return 1
        return 0

    train_with_restarts(j, run, pause=0)
    assert sizes == [16, 8, 4]


def test_out_of_memory_after_a_checkpoint_does_not_change_the_batch(tmp_path: Path) -> None:
    j = job(tmp_path, retries=3, batch_size=16)
    j.run_dir.mkdir()
    (j.run_dir / "last.pt").write_bytes(b"x")
    results = iter([1, 0])

    def run(command: object, log: Path) -> int:
        log.write_text("out of memory")
        return next(results)

    train_with_restarts(j, run, pause=0)
    assert j.batch_size == 16


def test_a_numeric_share_keeps_the_fixed_mix_of_the_earlier_experiments(tmp_path: Path) -> None:
    o = train_overrides(job(tmp_path, share=0.5))
    assert "data.sordi.share=0.5" in o and "data.sordi.share=null" not in o


def test_a_pool_subsample_is_passed_to_the_trainer_only_when_asked(tmp_path: Path) -> None:
    assert "data.sordi.subsample=12941" in train_overrides(job(tmp_path, pool_images=12941))
    assert not any(x.startswith("data.sordi.subsample") for x in train_overrides(job(tmp_path)))


def test_a_time_budget_sizes_the_train_set_so_every_epoch_is_all_of_it() -> None:
    from palletvision_train.job.run import plan_budget

    plan = plan_budget(hours=15, epochs=150, images_per_second=100, margin=0.6, n_real=2820)
    # 15 h x 3600 s x 100 img/s x 0.6 = 3.24 M views, over 150 epochs = 21,600 images per epoch
    assert plan["train_images"] == 21_600 and plan["pool_images"] == 21_600 - 2820
    assert plan["views"] == 21_600 * 150 and plan["real_images"] == 2820


def test_a_budget_too_small_for_the_real_images_is_refused() -> None:
    from palletvision_train.job.run import plan_budget

    with pytest.raises(SystemExit, match="raise --budget-hours"):
        plan_budget(hours=1, epochs=150, images_per_second=20, margin=0.6, n_real=2820)
    with pytest.raises(ValueError, match="margin"):
        plan_budget(hours=15, epochs=150, images_per_second=100, margin=1.5, n_real=2820)


def test_a_full_epoch_and_mixed_precision_reach_the_trainer(tmp_path: Path) -> None:
    o = train_overrides(job(tmp_path, epoch_images=None, amp=True))
    assert "data.sordi.epoch_images=null" in o and "train.amp=true" in o
    assert "train.amp=true" not in train_overrides(job(tmp_path))
