"""Fine-tuning tools: a separate learning rate for pretrained tensors, a per-step progressive-loss
weight, and an exponential moving average of the weights."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_checkpoint import COMMON, CrashingLoss, images, model_with
from torch import nn

from palletvision_train.checkpoint import load_checkpoint, save_checkpoint
from palletvision_train.ema import ModelEMA
from palletvision_train.eval_cli import load_model
from palletvision_train.optim import Schedule, build_optimizer, split_parameters
from palletvision_train.trainer import train

# ---- the average ----


def conv(net: nn.Module) -> nn.Conv2d:
    return cast(nn.Conv2d, net[0])  # type: ignore[index]


def norm(net: nn.Module) -> nn.BatchNorm2d:
    return cast(nn.BatchNorm2d, net[1])  # type: ignore[index]


def tiny() -> nn.Sequential:
    net = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
    net.eval()
    return net


def test_the_decay_ramps_up_from_zero_toward_its_limit() -> None:
    average = ModelEMA(tiny(), decay=0.99, tau=100)
    assert average.current_decay() == 0  # nothing averaged yet: follow the weights
    seen = []
    for _ in range(600):
        average.update(tiny())
        seen.append(average.current_decay())
    assert seen == sorted(seen) and 0.9 < seen[-1] < 0.99 and seen[-1] > seen[99]


def test_an_update_moves_the_average_a_fraction_toward_the_weights() -> None:
    net = tiny()
    average = ModelEMA(
        net, decay=0.8, tau=1e-9
    )  # the ramp is over at once: d = 0.8 (not 0.5, where d and 1 - d agree)
    old = conv(average.module).weight.detach().clone()
    with torch.no_grad():
        conv(net).weight.add_(2.0)
    average.update(net)
    assert torch.allclose(conv(average.module).weight, 0.8 * old + 0.2 * conv(net).weight)
    assert torch.equal(conv(net).weight, old + 2.0)  # the trained weights are untouched


def test_integer_counters_are_copied_not_averaged() -> None:
    net = tiny()
    average = ModelEMA(net, decay=0.9)
    counter = norm(net).num_batches_tracked
    assert counter is not None
    counter.fill_(7)
    average.update(net)
    mirrored = norm(average.module).num_batches_tracked
    assert mirrored is not None and int(mirrored) == 7


def test_the_average_is_a_separate_frozen_copy() -> None:
    net = tiny()
    average = ModelEMA(net)
    assert not any(p.requires_grad for p in average.module.parameters())
    assert not average.module.training and average.module is not net
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(net, decay=1.0)


# ---- checkpoints with an average ----


def test_with_an_average_model_is_the_average_and_raw_model_is_the_trained_weights(
    tmp_path: Path,
) -> None:
    net = tiny()
    average = ModelEMA(net, decay=0.5, tau=1e-9)
    with torch.no_grad():
        conv(net).weight.add_(1.0)
    average.update(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=net, optimizer=optimizer, epoch=1, steps=2, epochs=3,
                    best_metric=None, config={}, ema=average)  # fmt: skip
    state = torch.load(path)
    assert torch.equal(state["model"]["0.weight"], conv(average.module).weight)
    assert torch.equal(state["raw_model"]["0.weight"], conv(net).weight)
    assert state["ema_updates"] == 1
    plain = tmp_path / "plain.pt"
    save_checkpoint(plain, model=net, optimizer=optimizer, epoch=1, steps=2, epochs=3,
                    best_metric=None, config={})  # fmt: skip
    assert "raw_model" not in torch.load(plain)  # without an average the file is as before


def test_resuming_restores_the_trained_weights_and_the_average_separately(tmp_path: Path) -> None:
    net = tiny()
    average = ModelEMA(net, decay=0.5, tau=1e-9)
    with torch.no_grad():
        conv(net).weight.add_(1.0)
    average.update(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=net, optimizer=optimizer, epoch=1, steps=2, epochs=3,
                    best_metric=None, config={}, ema=average)  # fmt: skip

    fresh, fresh_average = tiny(), ModelEMA(tiny(), decay=0.5, tau=1e-9)
    resumed = load_checkpoint(
        path, fresh, torch.optim.SGD(fresh.parameters(), lr=0.1), fresh_average
    )
    assert torch.equal(conv(fresh).weight, conv(net).weight)
    assert torch.equal(conv(fresh_average.module).weight, conv(average.module).weight)
    assert fresh_average.updates == resumed.ema_updates == 1


def test_resuming_without_an_average_uses_the_trained_weights_and_ignores_the_average(
    tmp_path: Path,
) -> None:
    net, average = tiny(), ModelEMA(tiny(), decay=0.5, tau=1e-9)
    with torch.no_grad():
        conv(net).weight.add_(1.0)
    average.update(net)
    path = tmp_path / "last.pt"
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    save_checkpoint(path, model=net, optimizer=optimizer, epoch=1, steps=2, epochs=3,
                    best_metric=None, config={}, ema=average)  # fmt: skip
    other = tiny()
    load_checkpoint(path, other, torch.optim.SGD(other.parameters(), lr=0.1))
    assert torch.equal(conv(other).weight, conv(net).weight)  # not the average


def test_turning_the_average_on_when_resuming_an_older_run_starts_it_from_the_weights(
    tmp_path: Path,
) -> None:
    net = tiny()
    path = tmp_path / "old.pt"
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    save_checkpoint(path, model=net, optimizer=optimizer, epoch=1, steps=2, epochs=3,
                    best_metric=None, config={})  # fmt: skip
    fresh, average = tiny(), ModelEMA(tiny())
    load_checkpoint(path, fresh, torch.optim.SGD(fresh.parameters(), lr=0.1), average)
    assert torch.equal(conv(average.module).weight, conv(net).weight) and average.updates == 0


# ---- separate learning rates ----


def names_of(groups: dict[str, list[nn.Parameter]], model: nn.Module) -> dict[str, set[str]]:
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {group: {by_id[id(p)] for p in params} for group, params in groups.items()}


def test_pretrained_tensors_get_their_own_groups_and_nothing_is_lost() -> None:
    model = model_with(0)
    every = {n for n, _ in model.named_parameters()}
    pretrained = {n for n in every if n.startswith(("backbone", "neck"))}
    named = names_of(split_parameters(model, pretrained), model)
    assert set().union(*named.values()) == every  # every parameter is in exactly one group
    assert sum(len(v) for v in named.values()) == len(every)
    for group, members in named.items():
        assert members <= pretrained if group.endswith("_pre") else not (members & pretrained)
    assert {"weight", "weight_pre"} <= set(named)


def test_without_pretrained_names_the_groups_are_the_original_three() -> None:
    assert list(split_parameters(model_with(0))) == ["bias", "norm", "weight"]
    assert list(split_parameters(model_with(0), None)) == ["bias", "norm", "weight"]


def test_a_multiplier_of_one_keeps_the_original_groups_so_old_checkpoints_still_load() -> None:
    model = model_with(0)
    pretrained = {n for n, _ in model.named_parameters()}
    names = [g["name"] for g in build_optimizer(model, "sgd", 0.01, 0.0, pretrained=pretrained,
                                                pretrained_lr_mult=1.0).param_groups]  # fmt: skip
    assert names == ["bias", "norm", "weight"]


def test_the_pretrained_groups_carry_the_multiplier_and_weight_decay_only_on_weights() -> None:
    model = model_with(0)
    pretrained = {n for n in (n for n, _ in model.named_parameters()) if n.startswith("backbone")}
    optimizer = build_optimizer(model, "musgd", 0.01, 0.001, pretrained=pretrained,
                                pretrained_lr_mult=0.2)  # fmt: skip
    by_name = {g["name"]: g for g in optimizer.param_groups}
    assert by_name["weight_pre"]["lr_scale"] == 0.2 and "lr_scale" not in by_name["weight"]
    assert (
        by_name["weight_pre"]["weight_decay"] == 0.001 and by_name["bias_pre"]["weight_decay"] == 0
    )
    assert by_name["weight_pre"]["use_muon"] and not by_name["bias_pre"]["use_muon"]


def test_the_schedule_scales_the_learning_rate_of_the_pretrained_groups() -> None:
    model = model_with(0)
    pretrained = {n for n, _ in model.named_parameters() if n.startswith("backbone")}
    optimizer = build_optimizer(
        model, "sgd", 0.1, 0.0, pretrained=pretrained, pretrained_lr_mult=0.25
    )
    Schedule(lr=0.1, epochs=2, batches_per_epoch=10).apply(optimizer, 0)
    rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert rates["weight"] == pytest.approx(0.1) and rates["weight_pre"] == pytest.approx(0.025)
    warm = Schedule(lr=0.1, epochs=2, batches_per_epoch=100, warmup_epochs=1, warmup_bias_lr=0.1)
    warm.apply(optimizer, 0)
    rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert rates["bias"] == pytest.approx(0.1) and rates["bias_pre"] == pytest.approx(0.025)


def test_a_multiplier_of_zero_freezes_the_pretrained_parameters_while_the_rest_learn(
    tmp_path: Path,
) -> None:
    model = model_with(0)
    pretrained = {n for n, _ in model.named_parameters() if n.startswith("backbone")}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train(model, images(), epochs=2, output_dir=tmp_path, optimizer_name="sgd", momentum=0.9,
          pretrained=pretrained, pretrained_lr_mult=0.0, **COMMON)  # fmt: skip
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved and not (moved & pretrained)
    assert any(n.startswith("head") for n in moved)


# ---- the progressive-loss weight ----


def run_alphas(tmp_path: Path, **options: Any) -> list[float]:
    result = train(model_with(0), images(), epochs=3, output_dir=tmp_path, **COMMON, **options)
    return result.alphas


def test_by_default_alpha_changes_once_per_epoch(tmp_path: Path) -> None:
    alphas = run_alphas(tmp_path)
    assert len(alphas) == 6 and alphas[0] == alphas[1] != alphas[2] == alphas[3] != alphas[4]
    assert alphas[0] == pytest.approx(0.8) and alphas[-1] == pytest.approx(0.1)


def test_with_the_step_schedule_alpha_changes_every_step_and_ends_where_the_epoch_one_does(
    tmp_path: Path,
) -> None:
    alphas = run_alphas(tmp_path, alpha_schedule="step")
    assert len(set(alphas)) == 5 and alphas == sorted(alphas, reverse=True)
    assert alphas[0] == pytest.approx(0.8)  # the same start and the same epoch boundaries
    assert alphas[2] == pytest.approx(run_alphas(tmp_path / "e")[2])
    assert alphas[4] == alphas[5] == pytest.approx(0.1)  # reached at the last epoch, as in Eq. 3


def test_the_start_and_end_of_alpha_can_be_chosen(tmp_path: Path) -> None:
    alphas = run_alphas(tmp_path, alpha_init=0.3, alpha_final=0.05)
    assert alphas[0] == pytest.approx(0.3) and alphas[-1] == pytest.approx(0.05)


def test_a_one_epoch_run_anneals_with_the_step_schedule_and_does_not_without(
    tmp_path: Path,
) -> None:
    flat = train(model_with(0), images(), epochs=1, output_dir=tmp_path / "a", **COMMON).alphas
    moving = train(model_with(0), images(), epochs=1, output_dir=tmp_path / "b",
                   alpha_schedule="step", **COMMON).alphas  # fmt: skip
    assert len(set(flat)) == 1 and flat[0] == pytest.approx(0.8)  # the flaw the experiment found
    assert moving[0] > moving[-1]


# ---- training with the average ----


def test_scoring_and_best_use_the_average_while_the_checkpoint_keeps_the_trained_weights(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, torch.Tensor]] = []

    def metric(model: Any) -> float:
        seen.append({k: v.clone() for k, v in model.state_dict().items()})
        return 0.5

    model = model_with(0)
    train(model, images(), epochs=2, output_dir=tmp_path, metric_fn=metric, ema=True,
          ema_decay=0.9, ema_tau=2, **COMMON)  # fmt: skip
    state = torch.load(tmp_path / "best.pt")
    key = next(k for k in state["model"] if k.endswith("weight") and state["model"][k].dim() == 4)
    assert torch.equal(seen[0][key], state["model"][key])  # what was scored is what is saved
    assert not torch.equal(state["model"][key], state["raw_model"][key])  # and it is the average
    loaded, _, _ = load_model(tmp_path / "best.pt")
    assert torch.equal(loaded.state_dict()[key], state["model"][key])  # pv-eval gets the average


def test_without_the_average_nothing_changes(tmp_path: Path) -> None:
    train(model_with(0), images(), epochs=1, output_dir=tmp_path, **COMMON)
    assert "raw_model" not in torch.load(tmp_path / "last.pt")


def test_a_resumed_run_with_an_average_matches_the_uninterrupted_one(tmp_path: Path) -> None:
    options: dict[str, Any] = {"ema": True, "ema_decay": 0.9, "ema_tau": 2}
    whole = model_with(0)
    torch.manual_seed(5)
    train(whole, images(), epochs=4, output_dir=tmp_path / "whole", **options, **COMMON)

    doomed = model_with(0)
    torch.manual_seed(5)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(doomed, images(), epochs=4, output_dir=tmp_path / "c", loss_fn=CrashingLoss(6),
              **options, **COMMON)  # fmt: skip
    resumed = model_with(0)
    train(resumed, images(), epochs=4, output_dir=tmp_path / "c",
          resume=tmp_path / "c" / "last.pt", **options, **COMMON)  # fmt: skip
    a, b = torch.load(tmp_path / "whole" / "last.pt"), torch.load(tmp_path / "c" / "last.pt")
    assert a["ema_updates"] == b["ema_updates"] == 8
    for key in a["model"]:
        assert torch.allclose(a["model"][key].float(), b["model"][key].float(), atol=1e-6), key
        assert torch.allclose(a["raw_model"][key].float(), b["raw_model"][key].float(), atol=1e-6)


# ---- freezing the backbone, and stopping early ----


def test_freeze_ten_keeps_all_but_the_last_backbone_layer_fixed(tmp_path: Path) -> None:
    model = model_with(0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train(model, images(), epochs=1, output_dir=tmp_path, optimizer_name="sgd", momentum=0.9,
          freeze=10, **COMMON)  # fmt: skip
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    fixed = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert fixed and all(n.startswith("backbone") for n in fixed)
    assert not (moved & fixed)
    assert any(n.startswith("backbone.to_p5.3") for n in moved)  # the attention layer still learns
    assert any(n.startswith("head") for n in moved)


def test_freeze_zero_trains_everything_and_an_impossible_value_is_refused(tmp_path: Path) -> None:
    model = model_with(0)
    train(model, images(), epochs=1, output_dir=tmp_path, **COMMON)
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):
        train(model_with(0), images(), epochs=1, output_dir=tmp_path / "x", freeze=24, **COMMON)


def test_patience_stops_the_run_after_that_many_epochs_without_improvement(tmp_path: Path) -> None:
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = train(model_with(0), images(), epochs=6, output_dir=tmp_path,
                   metric_fn=lambda _m: next(scores), patience=2, **COMMON)  # fmt: skip
    assert len(result.alphas) == 6  # 2 steps per epoch, 3 epochs: best at epoch 1, stop at 3
    assert result.best_metric == 0.5


def test_without_patience_every_epoch_runs(tmp_path: Path) -> None:
    scores = iter([0.5, 0.4, 0.3, 0.2])
    result = train(model_with(0), images(), epochs=4, output_dir=tmp_path,
                   metric_fn=lambda _m: next(scores), **COMMON)  # fmt: skip
    assert len(result.alphas) == 8


def test_freeze_twenty_three_trains_only_the_head(tmp_path: Path) -> None:
    model = model_with(0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train(model, images(), epochs=1, output_dir=tmp_path, optimizer_name="sgd", momentum=0.9,
          freeze=23, **COMMON)  # fmt: skip
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved and all(n.startswith("head") for n in moved)
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("head"))


@pytest.mark.skipif(not (Path(__file__).parents[3] / "data/sordid-02").exists(), reason="no data")
def test_real_train_false_trains_on_synthetic_images_only() -> None:
    from hydra import compose, initialize_config_dir

    from palletvision_train.cli import CONFIG_DIR, build_dataset, build_sampler

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=[
            "data=loco_sordi", "data.real_train=false", "data.sordi.roots=[data/sordid-02]",
            "data.sordi.subsample=50", "data.sordi.epoch_images=null",
        ])  # fmt: skip
    dataset = build_dataset(cfg, ["pallet"])
    sampler = build_sampler(cfg, dataset)
    assert sampler is not None and len(sampler) == 50  # type: ignore[arg-type]
    assert len(dataset.base.datasets[0]) == 0 and len(dataset) == 50  # type: ignore[attr-defined,arg-type]
