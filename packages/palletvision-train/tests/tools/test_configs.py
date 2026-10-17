from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIGS = Path(__file__).resolve().parents[4] / "configs"


def test_root_config_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="config")
    assert cfg.device == "auto"
    assert cfg.model.num_classes == 1
    assert cfg.train.output_dir == "runs/stage1"
    # stage 1: pallets and stillages are the one class; pallet trucks and the rest are dropped
    assert dict(cfg.data.class_map) == {"pallet": "pallet", "stillage": "pallet"}
