import pytest

from palletvision_model.config import SIZES, ModelConfig, make_divisible


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.5, 8), (8, 8), (8.1, 16), (16, 16), (60, 64), (64, 64)],
)
def test_make_divisible_rounds_up_to_multiple_of_8(x: float, expected: int) -> None:
    assert make_divisible(x) == expected


def test_channels_follow_the_guide_table() -> None:
    # nominal widths of the backbone stages that feed the head: 256 / 512 / 1024
    assert [SIZES["n"].channels(c) for c in (256, 512, 1024)] == [64, 128, 256]
    assert [SIZES["s"].channels(c) for c in (256, 512, 1024)] == [128, 256, 512]


def test_channels_are_capped_at_max_channels() -> None:
    cfg = ModelConfig(depth_multiple=1.0, width_multiple=1.0, max_channels=512)
    assert cfg.channels(2048) == 512


def test_stem_channels_for_n() -> None:
    assert SIZES["n"].channels(64) == 16  # layer 0
    assert SIZES["n"].channels(128) == 32  # layer 1


def test_repeats_scale_but_never_reach_zero() -> None:
    n = SIZES["n"]
    assert n.repeats(2) == 1  # 2 * 0.5
    assert n.repeats(6) == 3
    assert n.repeats(1) == 1  # listed once: never scaled
    assert ModelConfig(depth_multiple=0.1, width_multiple=1.0).repeats(2) == 1
