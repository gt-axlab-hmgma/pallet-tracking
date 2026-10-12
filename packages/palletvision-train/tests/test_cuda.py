"""Mixed precision, CUDA speed settings and ``pv-doctor``.

Most of this runs anywhere (the decisions are plain functions; bfloat16 autocast also works on
the CPU). The tests marked ``needs_gpu`` run only on a machine with CUDA and are the ones to run
on a new GPU box: ``uv run pytest packages/palletvision-train/tests/test_cuda.py``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from palletvision_model.config import SIZES
from palletvision_model.yolo26 import YOLO26
from palletvision_train import doctor, precision, trainer
from palletvision_train.data import Batch, LetterboxedDataset
from palletvision_train.doctor import CrowdedBoxes, environment, time_training, warnings_for
from palletvision_train.precision import AmpPlan, configure_backend, plan_amp
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import train

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


# ---- the plan ----


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_with_amp_off_nothing_changes_whatever_the_device(device: str) -> None:
    assert plan_amp(device, False, "float16") == AmpPlan(False, torch.float32, False)


def test_cuda_uses_bfloat16_without_loss_scaling_when_the_gpu_supports_it() -> None:
    plan = plan_amp("cuda", True, "auto", bf16_supported=True)
    assert plan == AmpPlan(True, torch.bfloat16, False)


def test_cuda_falls_back_to_float16_with_loss_scaling_on_an_older_gpu() -> None:
    assert plan_amp("cuda", True, "auto", bf16_supported=False) == AmpPlan(
        True, torch.float16, True
    )


def test_float16_on_cuda_always_scales_the_loss() -> None:
    assert plan_amp("cuda", True, "float16", bf16_supported=True).scaler is True
    assert plan_amp("cuda", True, "bfloat16", bf16_supported=True).scaler is False


def test_bfloat16_is_refused_on_a_gpu_that_cannot_do_it() -> None:
    with pytest.raises(ValueError, match="does not support bfloat16"):
        plan_amp("cuda", True, "bfloat16", bf16_supported=False)


def test_mps_uses_float16_and_never_a_scaler() -> None:
    assert plan_amp("mps", True, "auto") == AmpPlan(True, torch.float16, False)


def test_the_cpu_does_bfloat16_only() -> None:
    assert plan_amp("cpu", True, "auto") == AmpPlan(True, torch.bfloat16, False)
    with pytest.raises(ValueError, match="bfloat16 only"):
        plan_amp("cpu", True, "float16")


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="use auto, float16 or bfloat16"):
        plan_amp("cuda", True, "float8", bf16_supported=True)
    with pytest.raises(ValueError, match="not available on 'tpu'"):
        plan_amp("tpu", True, "auto")


# ---- the backend settings ----


def test_cuda_settings_turn_on_autotuning_and_tf32(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", seen.append)
    configure_backend(torch.device("cuda"))
    assert torch.backends.cudnn.benchmark is True and seen == ["high"]


def test_other_devices_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", seen.append)
    for name in ("cpu", "mps"):
        configure_backend(torch.device(name))
    assert torch.backends.cudnn.benchmark is False and seen == []


def test_a_batch_moves_with_or_without_non_blocking_copies() -> None:
    batch = Batch(
        torch.zeros(2, 3, 8, 8),
        torch.zeros(2, 1, 1, dtype=torch.long),
        torch.zeros(2, 1, 4),
        torch.zeros(2, 1, 1, dtype=torch.bool),
    )
    for flag in (False, True):
        moved = batch.to("cpu", non_blocking=flag)
        assert all(a.equal(b) for a, b in zip(moved, batch, strict=True))


# ---- the trainer uses the plan ----


def tiny_run(tmp_path: Path, **extra: Any) -> trainer.TrainResult:
    torch.manual_seed(0)
    data = LetterboxedDataset(SyntheticBoxes(n=8, size=64), size=64)
    return train(
        YOLO26(SIZES["n"], input_size=64), data, epochs=1, batch_size=4, lr=1e-3, device="cpu",
        output_dir=tmp_path, **extra,
    )  # fmt: skip


def test_the_forward_pass_runs_in_the_planned_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, torch.dtype, bool]] = []
    real = torch.autocast

    def spy(device_type: str, dtype: torch.dtype, enabled: bool = True) -> Any:
        calls.append((device_type, dtype, enabled))
        return real(device_type=device_type, dtype=dtype, enabled=enabled)

    monkeypatch.setattr("palletvision_train.trainer.torch.autocast", spy)
    result = tiny_run(tmp_path / "on", amp=True, amp_dtype="bfloat16")
    assert calls and set(calls) == {("cpu", torch.bfloat16, True)}
    assert np.isfinite(result.losses).all()
    calls.clear()
    tiny_run(tmp_path / "off")
    assert set(calls) == {("cpu", torch.float32, False)}


def test_loss_scaling_is_only_switched_on_when_the_plan_asks_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enabled: list[bool] = []
    real = torch.amp.GradScaler

    def spy(device: str = "cuda", **kwargs: Any) -> Any:
        enabled.append(kwargs["enabled"])
        return real(device, **kwargs)

    monkeypatch.setattr("palletvision_train.trainer.torch.amp.GradScaler", spy)
    tiny_run(tmp_path, amp=True, amp_dtype="bfloat16")
    assert enabled == [False]
    monkeypatch.setattr(
        trainer, "plan_amp", lambda *args: precision.AmpPlan(False, torch.float32, True)
    )
    tiny_run(tmp_path / "scaled")
    assert enabled[-1] is True  # the plan, not the device, decides


def test_a_run_with_mixed_precision_follows_the_float32_loss_closely(tmp_path: Path) -> None:
    full = tiny_run(tmp_path / "full")
    mixed = tiny_run(tmp_path / "mixed", amp=True, amp_dtype="bfloat16")
    assert mixed.losses == pytest.approx(full.losses, rel=0.05)  # same curve, rounding apart


def test_training_applies_the_backend_settings_to_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices: list[str] = []
    monkeypatch.setattr(trainer, "configure_backend", lambda device: devices.append(device.type))
    tiny_run(tmp_path)
    assert devices == ["cpu"]


def test_a_bad_precision_setting_stops_training_before_it_starts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bfloat16 only"):
        tiny_run(tmp_path, amp=True, amp_dtype="float16")


# ---- pv-doctor ----


def test_the_environment_report_names_the_torch_version_and_device() -> None:
    facts = environment(torch.device("cpu"))
    assert facts["torch"] == torch.__version__ and facts["device"] == "cpu"


def test_a_cpu_only_torch_build_is_called_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)
    (message,) = warnings_for(torch.device("cpu"))
    assert "CPU-only build" in message and "cuda-setup" in message


def test_a_cuda_build_without_a_gpu_is_a_different_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    (message,) = warnings_for(torch.device("cpu"))
    assert "No usable GPU" in message


def test_a_gpu_newer_than_the_build_is_called_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    (message,) = warnings_for(torch.device("cuda"))
    assert "sm_120" in message and "12.8" in message
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_120"])
    assert warnings_for(torch.device("cuda")) == []


def test_the_crowded_images_have_the_requested_number_of_boxes_inside_the_picture() -> None:
    image, labels = CrowdedBoxes(3, 64, boxes=25)[1]
    assert image.shape == (64, 64, 3) and labels.shape == (25, 5)
    assert (labels[:, 1:5] >= 0).all() and (labels[:, 1:5] <= 64).all()
    assert np.array_equal(CrowdedBoxes(3, 64, 25)[1][1], labels)  # the same image every time


def test_the_timing_run_reports_a_speed_and_a_finite_loss() -> None:
    timing = time_training(torch.device("cpu"), size=64, batch=2, steps=4, amp=False)
    assert timing.seconds_per_step > 0 and timing.images_per_second > 0
    assert np.isfinite(timing.final_loss) and timing.name == "float32"
    assert time_training(torch.device("cpu"), 64, 2, 2, amp=True).name == "mixed precision"


def test_the_speed_leaves_out_the_start_up_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]

    def fake_train(*args: Any, **kwargs: Any) -> Any:
        now[0] += 10.0 + 0.5 * kwargs["max_steps"]  # 10 s to start, then half a second a step
        return trainer.TrainResult(checkpoint=Path(), losses=[1.0])

    monkeypatch.setattr(doctor, "train", fake_train)
    monkeypatch.setattr("palletvision_train.doctor.time.time", lambda: now[0])
    timing = time_training(torch.device("cpu"), size=64, batch=4, steps=6, amp=False)
    assert timing.seconds_per_step == pytest.approx(0.5)
    assert timing.images_per_second == pytest.approx(8.0)


def test_the_command_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["pv-doctor", "--device", "cpu", "--size", "64", "--batch", "2", "--steps", "2"]
    )
    doctor.main()
    out = capsys.readouterr().out
    assert "float32" in out and "mixed precision" in out and "img/s" in out


# ---- on a real GPU ----


@needs_gpu
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_gpu_training_with_mixed_precision_is_finite_and_saves_a_checkpoint(
    tmp_path: Path, dtype: str
) -> None:
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU has no bfloat16")
    torch.manual_seed(0)
    data = LetterboxedDataset(CrowdedBoxes(16, 160), 160)
    result = train(
        YOLO26(SIZES["n"], input_size=160), data, epochs=2, batch_size=4, lr=0.01, device="cuda",
        optimizer_name="musgd", amp=True, amp_dtype=dtype, output_dir=tmp_path, num_workers=2,
    )  # fmt: skip
    assert np.isfinite(result.losses).all() and (tmp_path / "last.pt").exists()


@needs_gpu
def test_gpu_float32_and_mixed_precision_losses_agree(tmp_path: Path) -> None:
    losses = {}
    for amp in (False, True):
        torch.manual_seed(0)
        data = LetterboxedDataset(CrowdedBoxes(8, 160), 160)
        losses[amp] = train(
            YOLO26(SIZES["n"], input_size=160), data, epochs=1, batch_size=4, lr=1e-3,
            device="cuda", amp=amp, output_dir=tmp_path / str(amp),
        ).losses  # fmt: skip
    assert losses[True] == pytest.approx(losses[False], rel=0.1)


@needs_gpu
def test_gpu_a_cut_run_resumes_from_its_saved_epoch(tmp_path: Path) -> None:
    def run(out: str, epochs: int, **extra: Any) -> trainer.TrainResult:
        torch.manual_seed(0)
        model = YOLO26(SIZES["n"], input_size=96)
        data = LetterboxedDataset(CrowdedBoxes(8, 96), 96)
        return train(
            model, data, epochs=epochs, batch_size=4, lr=1e-3, device="cuda",
            output_dir=tmp_path / out, **extra,
        )  # fmt: skip

    whole = run("whole", 4)
    run("cut", 4, max_steps=4)  # two epochs of two steps, then stops
    torch.manual_seed(0)
    resumed = run("cut", 4, resume=tmp_path / "cut" / "last.pt")
    assert resumed.start_epoch == 2
    assert len(resumed.losses) == len(whole.losses) - 4
