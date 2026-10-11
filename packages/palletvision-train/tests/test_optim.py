import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import nn

from palletvision_model.config import SIZES
from palletvision_model.loss import DetectionLoss
from palletvision_model.yolo26 import YOLO26
from palletvision_train.cli import train_from_config
from palletvision_train.data import LetterboxedDataset
from palletvision_train.optim import (
    MuSGD,
    Schedule,
    build_optimizer,
    newton_schulz,
    split_parameters,
)
from palletvision_train.synthetic import SyntheticBoxes
from palletvision_train.trainer import train

REPO = Path(__file__).resolve().parents[3]


def model_with(seed: int = 0) -> YOLO26:
    torch.manual_seed(seed)
    return YOLO26(SIZES["n"], input_size=64)


def images(n: int = 8) -> LetterboxedDataset:
    return LetterboxedDataset(SyntheticBoxes(n=n, size=64), size=64)


# ---- parameter groups ----------------------------------------------------------------------


def test_every_parameter_is_in_exactly_one_group() -> None:
    model = model_with()
    groups = split_parameters(model)
    grouped = [id(p) for params in groups.values() for p in params]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))


def test_only_convolution_weights_are_decayed() -> None:
    model = model_with()
    groups = split_parameters(model)
    assert all(p.ndim >= 2 for p in groups["weight"])  # convolution kernels
    assert all(p.ndim == 1 for p in groups["bias"] + groups["norm"])
    norms = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert len(groups["norm"]) == norms  # one scale per BatchNorm layer
    # biases: the six final prediction convs of each of the two head copies (3 box + 3 class)
    # and the shifts of the BatchNorm layers
    assert len(groups["bias"]) == 12 + norms


def test_the_optimiser_decays_only_the_weight_group() -> None:
    sgd = build_optimizer(model_with(), "sgd", lr=0.01, weight_decay=5e-4)
    assert [g["name"] for g in sgd.param_groups] == ["bias", "norm", "weight"]
    assert [g["weight_decay"] for g in sgd.param_groups] == [0.0, 0.0, 5e-4]
    assert sgd.param_groups[0]["nesterov"] and sgd.param_groups[0]["momentum"] == 0.937
    adamw = build_optimizer(model_with(), "adamw", lr=1e-3, weight_decay=0.01)
    assert adamw.param_groups[0]["betas"] == (0.937, 0.999)
    assert [g["weight_decay"] for g in adamw.param_groups] == [0.0, 0.0, 0.01]
    assert not build_optimizer(model_with(), "sgd", 0.01, 0.0, momentum=0.0).param_groups[0][
        "nesterov"
    ]
    with pytest.raises(ValueError, match="sgd"):
        build_optimizer(model_with(), "rmsprop", 0.01, 0.0)


# ---- the schedule, with values worked out by hand ---------------------------------------------

# lr 0.01, 10 epochs of 100 iterations, 3 epochs of warmup (300 iterations), cosine to 1%
S = Schedule(
    lr=0.01, epochs=10, batches_per_epoch=100, warmup_epochs=3, warmup_bias_lr=0.1, lr_final=0.01
)


def test_warmup_starts_the_weights_at_zero_and_the_biases_high() -> None:
    assert S.warmup_iterations == 300
    assert S.learning_rate("weight", 0) == 0.0
    assert S.learning_rate("norm", 0) == 0.0
    assert S.learning_rate("bias", 0) == pytest.approx(0.1)
    assert S.momentum_at(0) == pytest.approx(0.8)


def test_halfway_through_warmup() -> None:
    # at iteration 150 the cosine factor is 0.01 + 0.99 * (1 + cos(0.15 pi)) / 2 = 0.946048
    scheduled = 0.01 * 0.946048
    assert S.learning_rate("weight", 150) == pytest.approx(scheduled / 2, rel=1e-5)
    assert S.learning_rate("bias", 150) == pytest.approx((0.1 + scheduled) / 2, rel=1e-5)
    assert S.momentum_at(150) == pytest.approx((0.8 + 0.937) / 2)


def test_after_warmup_the_rate_follows_the_cosine_down_to_the_final_fraction() -> None:
    # iteration 300: 0.01 + 0.99 * (1 + cos(0.3 pi)) / 2 = 0.795954
    assert S.learning_rate("weight", 300) == pytest.approx(0.01 * 0.795954, rel=1e-5)
    assert S.learning_rate("bias", 300) == S.learning_rate("weight", 300)  # no longer special
    assert S.momentum_at(300) == 0.937
    assert S.learning_rate("weight", 1000) == pytest.approx(0.01 * 0.01)  # the last epoch: 1%
    assert S.learning_rate("weight", 5000) == pytest.approx(0.01 * 0.01)  # and it stays there
    after = [S.learning_rate("weight", i) for i in range(300, 1001, 10)]
    assert after == sorted(after, reverse=True)


def test_a_straight_line_schedule() -> None:
    line = Schedule(lr=0.01, epochs=10, batches_per_epoch=100, lr_final=0.01, kind="linear")
    assert line.learning_rate("weight", 500) == pytest.approx(0.01 * 0.505)  # 0.01 + 0.99 * 0.5
    with pytest.raises(ValueError, match="cosine"):
        Schedule(lr=0.01, epochs=1, batches_per_epoch=1, kind="step").decay(0)


def test_warmup_lasts_at_least_100_iterations() -> None:
    short = Schedule(lr=0.01, epochs=5, batches_per_epoch=10, warmup_epochs=3)
    assert short.warmup_iterations == 100  # 3 epochs would be only 30


def test_the_defaults_keep_the_rate_constant() -> None:
    flat = Schedule(lr=0.01, epochs=5, batches_per_epoch=10)
    assert {flat.learning_rate("weight", i) for i in (0, 7, 49, 500)} == {0.01}
    assert {flat.momentum_at(i) for i in (0, 7, 49)} == {0.937}


def test_apply_sets_every_group_including_adamw_betas() -> None:
    sgd = build_optimizer(model_with(), "sgd", 0.01, 5e-4)
    S.apply(sgd, 150)
    rates = {g["name"]: g["lr"] for g in sgd.param_groups}
    assert rates["weight"] == pytest.approx(S.learning_rate("weight", 150))
    assert (
        rates["bias"] == pytest.approx(S.learning_rate("bias", 150))
        and rates["bias"] > rates["weight"]
    )
    assert all(g["momentum"] == pytest.approx(S.momentum_at(150)) for g in sgd.param_groups)
    adamw = build_optimizer(model_with(), "adamw", 0.01, 0.0)
    S.apply(adamw, 0)
    assert adamw.param_groups[0]["betas"] == (pytest.approx(0.8), 0.999)


# ---- in the training loop ------------------------------------------------------------------


class Toy(nn.Module):
    """A tiny model with no BatchNorm (whose statistics depend on the batch), for exact tests."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(3, 4)

    def raw(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        z = self.lin(x.mean(dim=(2, 3)))[:, None, :]
        return {"one2many": (z, z), "one2one": (z, z)}


class MeanSquare(DetectionLoss):
    """Mean over images of a squared output: batch-additive, so accumulation can be exact."""

    def forward(self, preds: Any, *args: Any, **kwargs: Any) -> Any:
        return (preds["one2one"][1] ** 2).mean(), {}


def test_accumulating_small_batches_equals_one_big_batch() -> None:
    plain, accumulating = Toy(), Toy()
    kwargs: dict[str, Any] = {
        "device": "cpu",
        "epochs": 1,
        "lr": 0.1,
        "loss_fn": MeanSquare(),
        "momentum": 0.0,
    }
    torch.manual_seed(1)
    a = train(
        cast(YOLO26, plain),
        images(8),
        batch_size=4,
        optimizer_name="sgd",
        output_dir="/tmp/pv_a",
        **kwargs,
    )
    torch.manual_seed(1)
    b = train(
        cast(YOLO26, accumulating), images(8), batch_size=2, nominal_batch=4, optimizer_name="sgd",
        output_dir="/tmp/pv_b", **kwargs,
    )  # fmt: skip
    assert a.updates == 2 and b.updates == 2  # 2 updates each; B made them from 4 batches of 2
    for name, tensor in plain.state_dict().items():
        assert torch.allclose(accumulating.state_dict()[name], tensor, atol=1e-6), name


def test_a_trailing_partial_group_still_updates(tmp_path: Path) -> None:
    result = train(
        model_with(), images(6), device="cpu", epochs=1, batch_size=2, lr=1e-3, nominal_batch=4,
        output_dir=tmp_path,
    )  # fmt: skip
    assert len(result.losses) == 3 and result.updates == 2  # batches 1+2, then batch 3 alone


def test_weight_decay_is_scaled_by_the_effective_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []
    real = build_optimizer

    def spy(model: nn.Module, name: str, lr: float, weight_decay: float, *rest: Any) -> Any:
        seen.append(weight_decay)
        return real(model, name, lr, weight_decay, *rest)

    monkeypatch.setattr("palletvision_train.trainer.build_optimizer", spy)
    for batch, nominal in ((4, 4), (4, 8), (4, None), (4, 2)):
        train(
            model_with(), images(8), device="cpu", epochs=1, batch_size=batch, lr=1e-3,
            weight_decay=1e-3, nominal_batch=nominal, max_steps=1, output_dir=tmp_path,
        )  # fmt: skip
    # effective batch / nominal: 4/4, 8/8, no nominal batch at all, and 4/2 (no accumulation: 1)
    assert seen == pytest.approx([1e-3, 1e-3, 1e-3, 2e-3])


def step_rates(run: Path) -> list[float]:
    log = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    return [e["lr"] for e in log if e["type"] == "step"]


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_logged_learning_rate_follows_the_schedule(tmp_path: Path) -> None:
    train(
        model_with(), images(8), device="cpu", epochs=4, batch_size=4, lr=0.01,
        optimizer_name="sgd", warmup_epochs=1, warmup_bias_lr=0.1, lr_final=0.1,
        output_dir=tmp_path,
    )  # fmt: skip
    rates = step_rates(tmp_path)
    assert len(rates) == 8 and rates[0] == 0.0  # warmup is 100 iterations, longer than this run
    expected = Schedule(0.01, 4, 2, warmup_epochs=1, warmup_bias_lr=0.1, lr_final=0.1)
    # every step logs the schedule's value for its own iteration (0, 1, 2, ...)
    assert rates == pytest.approx([expected.learning_rate("weight", i) for i in range(8)])


def test_the_rate_decays_without_warmup(tmp_path: Path) -> None:
    train(
        model_with(), images(8), device="cpu", epochs=6, batch_size=4, lr=0.01,
        optimizer_name="sgd", lr_final=0.1, output_dir=tmp_path,
    )  # fmt: skip
    rates = step_rates(tmp_path)
    assert rates[0] == 0.01 and rates == sorted(rates, reverse=True) and rates[-1] > 0.001


class CrashingLoss(DetectionLoss):
    def __init__(self, crash_at: int) -> None:
        super().__init__()
        self.calls, self.crash_at = 0, crash_at

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == self.crash_at:
            raise RuntimeError("simulated crash")
        return super().forward(*args, **kwargs)


@pytest.mark.parametrize("optimizer_name", ["sgd", "musgd"])
def test_a_resumed_run_keeps_the_schedule_and_the_momentum(
    tmp_path: Path, optimizer_name: str
) -> None:
    kwargs: dict[str, Any] = {
        "device": "cpu", "batch_size": 4, "lr": 0.01, "optimizer_name": optimizer_name,
        "warmup_epochs": 1, "warmup_bias_lr": 0.1, "lr_final": 0.1, "epochs": 4,
    }  # fmt: skip
    whole = model_with()
    torch.manual_seed(5)
    full = train(whole, images(), output_dir=tmp_path / "whole", **kwargs)
    crashed = tmp_path / "crashed"
    doomed = model_with()
    torch.manual_seed(5)
    with pytest.raises(RuntimeError):
        train(doomed, images(), output_dir=crashed, loss_fn=CrashingLoss(6), **kwargs)
    fresh = model_with(99)
    resumed = train(fresh, images(), output_dir=crashed, resume=crashed / "last.pt", **kwargs)
    assert resumed.losses == pytest.approx(full.losses[4:], rel=1e-5)
    assert step_rates(crashed) == pytest.approx(step_rates(tmp_path / "whole"))  # one curve


def config(*overrides: str, tmp_path: Path) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        return compose(
            config_name="config",
            overrides=["device=cpu", "data=synthetic", "data.size=64", "data.num_samples=8",
                       "model.input_size=64", "train.batch_size=4", "train.epochs=2",
                       f"train.output_dir={tmp_path}", *overrides],
        )  # fmt: skip


def test_the_defaults_train_with_musgd_warmup_and_accumulation(tmp_path: Path) -> None:
    cfg = config(tmp_path=tmp_path)
    assert (cfg.train.optimizer, cfg.train.nominal_batch, cfg.train.warmup_epochs) == (
        "musgd",
        64,
        1,
    )
    assert (cfg.train.momentum, cfg.train.muon_weight, cfg.train.sgd_weight) == (0.937, 0.45, 0.55)
    assert (cfg.train.lr_schedule, cfg.train.warmup_bias_lr) == ("linear", 0.1)
    result = train_from_config(cfg)
    assert len(result.losses) == 4 and np.isfinite(result.losses).all()
    assert (
        result.updates == 2
    )  # batch 4 against a nominal 64: 16 batches per update, so one per epoch
    first = lines(tmp_path / "metrics.jsonl")[0]
    assert first["lr"] == 0.0  # warmup begins at zero


def test_adamw_can_be_chosen_from_the_config(tmp_path: Path) -> None:
    cfg = config(
        "train.optimizer=adamw", "train.lr=0.001", "train.nominal_batch=null", tmp_path=tmp_path
    )
    result = train_from_config(cfg)
    assert result.updates == 4 and np.isfinite(result.losses).all()


# ---- MuSGD ---------------------------------------------------------------------------------


def test_newton_schulz_evens_the_singular_values_out() -> None:
    torch.manual_seed(0)
    raw = torch.randn(16, 64) * torch.linspace(0.05, 5, 16)[:, None]  # very uneven directions
    raw_values = torch.linalg.svdvals(raw)
    assert raw_values[0] / raw_values[-1] > 20
    out = torch.linalg.svdvals(newton_schulz(raw))
    assert out.shape == (16,) and out.min() > 0.4 and out.max() < 1.6  # roughly 0.5 to 1.5
    assert abs(out.mean().item() - 1.0) < 0.25


def test_newton_schulz_shape_scale_and_edge_cases() -> None:
    torch.manual_seed(1)
    wide = torch.randn(8, 24)
    assert newton_schulz(wide).shape == (8, 24)
    # a tall matrix is the transposed problem: same answer, transposed
    assert torch.allclose(newton_schulz(wide.T), newton_schulz(wide).T, atol=1e-5)
    assert torch.allclose(newton_schulz(7.0 * wide), newton_schulz(wide), atol=1e-4)  # scale-free
    assert torch.equal(newton_schulz(torch.zeros(4, 6)), torch.zeros(4, 6))  # no NaN from 0 / 0


def two_copies(shape: tuple[int, ...], seed: int = 0) -> tuple[nn.Parameter, nn.Parameter]:
    torch.manual_seed(seed)
    base = torch.randn(shape)
    return nn.Parameter(base.clone()), nn.Parameter(base.clone())


def test_with_no_muon_weight_musgd_is_exactly_pytorch_sgd() -> None:
    mine, theirs = two_copies((4, 6))
    kwargs: dict[str, Any] = {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.01}
    a = MuSGD([{"params": [mine], "use_muon": True}], nesterov=True, muon=0.0, sgd=1.0, **kwargs)
    b = torch.optim.SGD([theirs], nesterov=True, **kwargs)
    torch.manual_seed(5)
    for _ in range(4):
        grad = torch.randn(4, 6)
        mine.grad, theirs.grad = grad.clone(), grad.clone()
        a.step()
        b.step()
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_one_step_is_the_sum_of_the_muon_part_and_the_sgd_part() -> None:
    for shape in ((3, 5), (2, 3, 3, 3)):  # a plain matrix and a convolution kernel
        weight, _ = two_copies(shape)
        start = weight.detach().clone()
        lr, beta, decay, muon, sgd = 0.1, 0.9, 0.1, 0.45, 0.55
        opt = MuSGD(
            [{"params": [weight], "use_muon": True}], lr=lr, momentum=beta, weight_decay=decay,
            nesterov=True, muon=muon, sgd=sgd,
        )  # fmt: skip
        torch.manual_seed(2)
        grad = torch.randn(shape)
        weight.grad = grad.clone()
        opt.step()
        # Muon: a running average of the gradient, then the Nesterov blend, then orthogonalise
        average = (1 - beta) * grad
        update = (1 - beta) * grad + beta * average
        scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
        muon_part = -lr * muon * scale * newton_schulz(update.reshape(shape[0], -1)).reshape(shape)
        # SGD: weight decay joins the gradient, then Nesterov momentum from a zero start. As in
        # the reference, the Muon part is applied first, so the decay sees the updated weights.
        decayed = grad + decay * (start + muon_part)
        sgd_part = -lr * sgd * (decayed + beta * decayed)
        assert torch.allclose(weight, start + muon_part + sgd_part, atol=1e-6), shape


def test_weight_decay_acts_only_through_the_sgd_part() -> None:
    weight, _ = two_copies((4, 6))
    start = weight.detach().clone()
    opt = MuSGD(
        [{"params": [weight], "use_muon": True}], lr=0.1, momentum=0.9, weight_decay=0.2,
        muon=0.45, sgd=0.55,
    )  # fmt: skip
    weight.grad = torch.zeros(4, 6)  # no gradient: only decay can move it (Muon of zero is zero)
    opt.step()
    assert torch.allclose(weight, start * (1 - 0.1 * 0.55 * 1.9 * 0.2), atol=1e-6)


def grads_for(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn_like(p)


def test_biases_and_norm_scales_get_plain_sgd_and_only_weights_get_muon() -> None:
    def tiny() -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), nn.Conv2d(6, 4, 1))

    a, b = tiny(), tiny()
    mu = build_optimizer(a, "musgd", lr=0.1, weight_decay=0.0, momentum=0.9)
    sgd = build_optimizer(b, "sgd", lr=0.1, weight_decay=0.0, momentum=0.9)
    assert isinstance(mu, MuSGD) and (mu.muon, mu.sgd) == (0.45, 0.55)
    flags = {g["name"]: g["use_muon"] for g in mu.param_groups}
    assert flags == {"bias": False, "norm": False, "weight": True}
    for step in range(3):
        grads_for(a, step)
        grads_for(b, step)
        mu.step()
        sgd.step()
    groups_a, groups_b = split_parameters(a), split_parameters(b)
    for name in ("bias", "norm"):  # identical to plain SGD
        for p, q in zip(groups_a[name], groups_b[name], strict=True):
            assert torch.allclose(p, q, atol=1e-6), name
    for p, q in zip(groups_a["weight"], groups_b["weight"], strict=True):
        assert not torch.allclose(p, q, atol=1e-4)  # the convolution weights differ: Muon is in


def test_the_optimiser_state_survives_a_save_and_load() -> None:
    mine, _ = two_copies((4, 6))
    first = MuSGD([{"params": [mine], "use_muon": True}], lr=0.1, momentum=0.9)
    for step in range(2):
        mine.grad = torch.randn(4, 6, generator=torch.Generator().manual_seed(step))
        first.step()
    assert all(isinstance(v, torch.Tensor) for v in first.state[mine].values())  # tensors only
    buffer = io.BytesIO()  # through a file, as a checkpoint does (state_dict alone shares tensors)
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    other = nn.Parameter(mine.detach().clone())
    second = MuSGD([{"params": [other], "use_muon": True}], lr=0.1, momentum=0.9)
    second.load_state_dict(torch.load(buffer, weights_only=True))
    grad = torch.randn(4, 6, generator=torch.Generator().manual_seed(9))
    mine.grad, other.grad = grad.clone(), grad.clone()
    first.step()
    second.step()
    assert torch.allclose(mine, other, atol=1e-7)  # the next step continues exactly


def test_the_schedule_sets_momentum_on_musgd_groups() -> None:
    opt = build_optimizer(model_with(), "musgd", 0.01, 5e-4)
    S.apply(opt, 0)
    assert all(g["momentum"] == pytest.approx(0.8) for g in opt.param_groups)


def test_musgd_trains_the_real_model(tmp_path: Path) -> None:
    result = train(
        model_with(), images(8), device="cpu", epochs=20, batch_size=4, lr=0.01,
        optimizer_name="musgd", momentum=0.94, output_dir=tmp_path,
    )  # fmt: skip
    assert np.isfinite(result.losses).all()
    # judge one branch: the total blends two and shifts weight to the harder one as it trains
    branch = [p["total_one2one"] for p in result.parts]
    assert np.mean(branch[-6:]) < np.mean(branch[:6])


def test_two_steps_follow_the_momentum_arithmetic() -> None:
    """The first step cannot tell momentum forms apart (Muon ignores the scale of its input and
    the buffers are multiples of one gradient), so check a second, different gradient too."""
    weight, _ = two_copies((3, 5))
    lr, beta, decay, muon, sgd = 0.1, 0.9, 0.05, 0.45, 0.55
    opt = MuSGD(
        [{"params": [weight], "use_muon": True}], lr=lr, momentum=beta, weight_decay=decay,
        nesterov=True, muon=muon, sgd=sgd,
    )  # fmt: skip
    expected = weight.detach().clone()
    average = torch.zeros(3, 5)  # Muon's running average of the gradient
    buffer = torch.zeros(3, 5)  # SGD's momentum buffer
    for seed in (3, 4):
        grad = torch.randn(3, 5, generator=torch.Generator().manual_seed(seed))
        weight.grad = grad.clone()
        opt.step()
        average = beta * average + (1 - beta) * grad
        update = (1 - beta) * grad + beta * average
        expected = expected - lr * muon * newton_schulz(update)  # square-ish: scale is 1
        decayed = grad + decay * expected  # the decay sees the weights the Muon part just moved
        buffer = beta * buffer + decayed
        expected = expected - lr * sgd * (decayed + beta * buffer)
    assert torch.allclose(weight, expected, atol=1e-5)
