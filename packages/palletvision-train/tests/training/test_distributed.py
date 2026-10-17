"""Training on several processes: gradients averaged, rank 0 alone writes, shards cover the data.

The multi-process tests start two real processes on the CPU (``gloo``) through
``torch.multiprocessing.spawn`` and run the real ``train`` in each, so the gradient averaging,
the shard-aware samplers and the rank-0-only files are exercised for real, without GPUs.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26
from palletvision_train.datasets.data import LetterboxedDataset
from palletvision_train.datasets.sordi import MixedSampler
from palletvision_train.tools.doctor import CrowdedBoxes
from palletvision_train.training import trainer
from palletvision_train.training.distributed import DistInfo, average, current, init_distributed
from palletvision_train.training.trainer import train

WORLD = 2
LR_WEIGHT_DECAY = 0.01


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def join_group(rank: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD),
        LOCAL_RANK=str(rank),
    )  # fmt: skip
    info, device = init_distributed("cpu")
    assert info == DistInfo(rank, WORLD, rank) and device == "cpu"


def model_and_data(batches_per_rank: int = 4) -> tuple[YOLO26, LetterboxedDataset]:
    torch.manual_seed(0)
    model = YOLO26(SIZES["n"], input_size=64)
    return model, LetterboxedDataset(CrowdedBoxes(2 * WORLD * batches_per_rank, 64, boxes=4), 64)


def worker_run(rank: int, port: int, out: str) -> None:
    """Two epochs, per-process batch 2, nominal batch 6 (so 2 batches are accumulated)."""
    join_group(rank, port)
    folder = Path(out)
    syncs_skipped, decays, metric_calls, draws = [0], [], [], []
    real_no_sync = DistributedDataParallel.no_sync
    real_build = trainer.build_optimizer  # type: ignore[attr-defined]

    def counting_no_sync(self: Any) -> Any:
        syncs_skipped[0] += 1
        return real_no_sync(self)

    def spy_build(model: Any, name: str, lr: float, decay: float, *args: Any) -> Any:
        decays.append(decay)
        return real_build(model, name, lr, decay, *args)

    DistributedDataParallel.no_sync = counting_no_sync  # type: ignore[method-assign]
    trainer.build_optimizer = spy_build  # type: ignore[attr-defined, assignment]

    def metric(model: Any) -> float:
        metric_calls.append(1)
        return 0.1 * len(metric_calls)

    model, data = model_and_data()
    result = train(
        model, data, device="cpu", epochs=2, batch_size=2, lr=0.01, weight_decay=LR_WEIGHT_DECAY,
        optimizer_name="sgd", momentum=0.9, nominal_batch=6, output_dir=folder / "run",
        metric_fn=metric, metric_every=1,
        on_epoch_start=lambda epoch: draws.append(float(torch.rand(1))),
    )  # fmt: skip
    torch.save(
        {
            "params": {k: v.detach().clone() for k, v in model.named_parameters()},
            "losses": result.losses, "updates": result.updates, "skipped": syncs_skipped[0],
            "decay": decays[0], "metric_calls": len(metric_calls), "first_draw": draws[0],
            "best": result.best_metric,
        },
        folder / f"rank{rank}.pt",
    )  # fmt: skip
    torch.distributed.destroy_process_group()


def worker_resume(rank: int, port: int, out: str) -> None:
    join_group(rank, port)
    folder = Path(out)
    model, data = model_and_data()
    common: dict[str, Any] = {
        "device": "cpu", "epochs": 4, "batch_size": 2, "lr": 0.01, "optimizer_name": "sgd",
        "momentum": 0.9, "output_dir": folder / "run",
    }  # fmt: skip
    train(model, data, max_steps=8, **common)  # two epochs of four steps, then it stops
    resumed_model, _ = model_and_data()
    result = train(resumed_model, data, resume=folder / "run" / "last.pt", **common)
    torch.save(
        {
            "params": {k: v.detach().clone() for k, v in resumed_model.named_parameters()},
            "start": result.start_epoch, "steps": len(result.losses),
        },
        folder / f"resume{rank}.pt",
    )  # fmt: skip
    torch.distributed.destroy_process_group()


def launch(worker: Any, out: Path) -> None:
    mp.spawn(worker, args=(free_port(), str(out)), nprocs=WORLD, join=True)  # type: ignore[attr-defined, no-untyped-call]


@pytest.fixture(scope="module")
def two_process_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ddp")
    launch(worker_run, out)
    return out


def test_every_process_ends_with_the_same_weights_and_the_same_logged_losses(
    two_process_run: Path,
) -> None:
    a, b = (torch.load(two_process_run / f"rank{r}.pt") for r in range(WORLD))
    assert a["params"].keys() == b["params"].keys()
    for name in a["params"]:
        assert torch.allclose(a["params"][name], b["params"][name], atol=1e-6), name
    assert a["losses"] == pytest.approx(b["losses"])  # the logged loss is the mean over processes
    assert len(a["losses"]) == 8  # 2 epochs x 4 batches of 2 per process


def test_gradients_are_averaged_once_per_update_not_once_per_batch(two_process_run: Path) -> None:
    for rank in range(WORLD):
        state = torch.load(two_process_run / f"rank{rank}.pt")
        # 4 batches an epoch, 2 accumulated per update: 2 updates an epoch, and the first batch
        # of each pair skips the sync
        assert state["updates"] == 4 and state["skipped"] == 4


def test_each_process_has_its_own_random_stream_for_augmentation(two_process_run: Path) -> None:
    first = [torch.load(two_process_run / f"rank{r}.pt")["first_draw"] for r in range(WORLD)]
    assert first[0] != first[1]


def test_the_weight_decay_counts_the_images_of_all_processes(two_process_run: Path) -> None:
    # per step 2 x 2 images, round(6 / 4) = 2 batches accumulated: 8 images an update against 6
    assert torch.load(two_process_run / "rank0.pt")["decay"] == pytest.approx(
        LR_WEIGHT_DECAY * 8 / 6
    )


def test_only_rank_zero_writes_the_log_and_the_checkpoints(two_process_run: Path) -> None:
    run = two_process_run / "run"
    lines = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines if x["type"] == "step"] == list(range(1, 9))  # each once
    assert sum(x["type"] == "epoch" for x in lines) == 2
    assert (run / "last.pt").exists() and (run / "best.pt").exists()


def test_only_rank_zero_runs_validation(two_process_run: Path) -> None:
    calls = [torch.load(two_process_run / f"rank{r}.pt")["metric_calls"] for r in range(WORLD)]
    assert calls == [2, 0]
    assert torch.load(two_process_run / "rank0.pt")["best"] == pytest.approx(0.2)


def test_a_resumed_multi_process_run_continues_from_the_saved_epoch(tmp_path: Path) -> None:
    launch(worker_resume, tmp_path)
    states = [torch.load(tmp_path / f"resume{r}.pt") for r in range(WORLD)]
    assert [s["start"] for s in states] == [2, 2] and [s["steps"] for s in states] == [8, 8]
    for name, tensor in states[0]["params"].items():
        assert torch.allclose(tensor, states[1]["params"][name], atol=1e-6), name
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines if x["type"] == "step"] == list(range(1, 17))


# ---- the pieces ----


def test_without_torchrun_nothing_is_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert current() == DistInfo() and not current().distributed and current().is_main
    assert init_distributed("auto") == (DistInfo(), "auto")
    values = torch.tensor([1.0, 2.0])
    assert average(values) is values


def test_several_processes_cannot_share_apples_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="cuda or cpu, not 'mps'"):
        init_distributed("mps")


@pytest.mark.parametrize("world", [2, 3, 4])
def test_the_shards_cover_the_epoch_and_have_equal_length(world: int) -> None:
    shards = []
    for rank in range(world):
        sampler = MixedSampler(70, 500, 0.3, seed=1, rank=rank, world_size=world)
        sampler.set_epoch(3)
        shards.append(list(sampler))
        assert len(shards[-1]) == len(sampler) == -(-100 // world)
    flat = [i for shard in shards for i in shard]
    assert set(range(70)) <= set(flat)  # every real image is run
    seen = set(flat)
    assert len(seen) == 100  # and the epoch has exactly 30 synthetic images in it
    assert sum(i >= 70 for i in seen) == 30 and all(i < 570 for i in seen)


def test_every_process_draws_the_same_epoch_and_it_changes_with_the_epoch() -> None:
    def whole(rank_seed_epoch: tuple[int, int]) -> list[int]:
        seed, epoch = rank_seed_epoch
        parts = []
        for rank in range(2):
            sampler = MixedSampler(20, 100, 0.5, seed=seed, rank=rank, world_size=2)
            sampler.set_epoch(epoch)
            parts.append(list(sampler))
        return sorted(parts[0] + parts[1])

    assert whole((5, 0)) == whole((5, 0))
    assert whole((5, 0)) != whole((5, 1))
    assert whole((5, 0)) != whole((6, 0))


def test_a_sharded_sampler_without_a_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="need a seed"):
        MixedSampler(10, 50, 0.3, rank=0, world_size=2)
    with pytest.raises(ValueError, match="not in a group of 2"):
        MixedSampler(10, 50, 0.3, seed=0, rank=2, world_size=2)


def test_the_trainer_tells_the_sampler_each_epoch_so_shards_reshuffle(tmp_path: Path) -> None:
    class Recording(torch.utils.data.Sampler[int]):
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

        def __len__(self) -> int:
            return 8

        def __iter__(self) -> Any:
            return iter(range(8))

    sampler = Recording()
    model, data = model_and_data()
    train(
        model, data, device="cpu", epochs=3, batch_size=4, lr=0.01, output_dir=tmp_path,
        sampler=sampler,
    )  # fmt: skip
    assert sampler.epochs == [0, 1, 2]


def test_the_single_process_sampler_still_follows_torchs_random_state() -> None:
    sampler = MixedSampler(20, 100, 0.5)
    torch.manual_seed(7)
    state = torch.get_rng_state()
    a = list(sampler)
    torch.set_rng_state(state)
    assert list(sampler) == a and len(sampler) == 40


# ---- through the real command (about 30 s) ----


@pytest.mark.slow
def test_torchrun_runs_the_command_on_two_processes(tmp_path: Path) -> None:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--master-addr=127.0.0.1", f"--master-port={free_port()}", "--local-addr=127.0.0.1",
        "-m",
        "palletvision_train.cli.train",
        "data=synthetic",
        "data.size=64",
        "data.num_samples=16",
        "model.input_size=64", "train.epochs=2", "train.batch_size=2", "train.nominal_batch=null",
        "train.warmup_epochs=0", "device=cpu", f"train.output_dir={tmp_path / 'run'}",
        "hydra.run.dir=" + str(tmp_path / "hydra"),
    ]  # fmt: skip
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines if x["type"] == "step"] == list(range(1, 9))
    assert done.stdout.count("trained 8 steps") == 1  # printed by rank 0 only
