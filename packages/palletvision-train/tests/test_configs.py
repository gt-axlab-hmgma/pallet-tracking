from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_root_config_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="config")
    assert cfg.device == "auto"
    assert cfg.model.num_classes == 1
    assert cfg.train.output_dir == "runs/stage1"
    assert "pallet_truck" in cfg.data.class_map
