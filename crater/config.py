"""Single source of truth for mission parameters (no magic numbers in the code)."""
import json
from pathlib import Path

from .localization import ErrorModel, GroundProjector, Intrinsics, Mount

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "mission.json"


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_projector(cfg: dict) -> GroundProjector:
    cam, mount = cfg["camera"], cfg["mount"]
    K = Intrinsics.from_fov(cam["width"], cam["height"], cam["hfov_deg"])
    return GroundProjector(K, Mount(**mount), cfg["localization"]["min_depression_deg"])


def build_error_model(cfg: dict) -> ErrorModel:
    return ErrorModel(**cfg["error_model"])


def model_path(cfg: dict) -> Path:
    return ROOT / cfg["detector"]["model_path"]
