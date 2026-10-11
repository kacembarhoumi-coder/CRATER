"""Preprocessing must be identical for training and onboard inference."""
import numpy as np

from crater.perception import letterbox, preprocess


def test_preprocessing_is_grayscale_3ch():
    img = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == img.shape
    assert np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2])


def test_letterbox_keeps_aspect_ratio():
    canvas, scale, pad_x, pad_y = letterbox(np.zeros((480, 640, 3), np.uint8), 320)
    assert canvas.shape == (320, 320, 3)
    assert scale == 0.5 and pad_x == 0 and pad_y == 40
