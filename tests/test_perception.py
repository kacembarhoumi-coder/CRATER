"""Perception: runs only if the trained ONNX model is present in models/."""
import numpy as np
import pytest

from crater.config import load_config, model_path

cfg = load_config()
MODEL = model_path(cfg)
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="models/crater_detector.onnx not present")


def test_detector_output_format():
    from crater.perception import HazardDetector
    det = HazardDetector(str(MODEL), conf_thres=0.25, threads=1)
    frame = np.full((480, 640, 3), 90, np.uint8)
    hazards = det.detect(frame)
    assert isinstance(hazards, list)
    for h in hazards:
        assert set(h) == {"cls", "conf", "box", "foot_px"}
        x1, y1, x2, y2 = h["box"]
        assert 0 <= x1 <= x2 < 640 and 0 <= y1 <= y2 < 480
    assert det.last_ms > 0

