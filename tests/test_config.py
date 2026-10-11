"""Configuration loads and builds consistent objects."""
from crater.config import build_error_model, build_projector, load_config


def test_config_builds():
    cfg = load_config()
    proj = build_projector(cfg)
    assert proj.K.width == cfg["camera"]["width"]
    assert proj.mount.h == cfg["mount"]["h"]
    assert build_error_model(cfg).sigma_px == cfg["error_model"]["sigma_px"]
