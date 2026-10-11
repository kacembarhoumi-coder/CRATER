"""Localization: pixel -> landing-site coordinates and error model."""
import math

import numpy as np
import pytest

from crater.localization import (ErrorModel, GroundProjector, Intrinsics, Mount, Pose,
                                 covariance, monte_carlo, range_cross_sigmas)

K = Intrinsics.from_fov(640, 480, 90)
M = Mount(tx=0.20, ty=0.0, h=0.50, pitch_deg=25.0)
PROJ = GroundProjector(K, M)


def test_round_trip_ground_points():
    rng = np.random.default_rng(1)
    worst, n = 0.0, 0
    for _ in range(3000):
        pose = Pose(rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-180, 180))
        xb, yb = rng.uniform(0.3, 9), rng.uniform(-6, 6)
        c, s = math.cos(math.radians(pose.yaw_deg)), math.sin(math.radians(pose.yaw_deg))
        E, N = pose.x + c * xb - s * yb, pose.y + s * xb + c * yb
        px = PROJ.landing_to_pixel(E, N, pose)
        if px is None or not (0 <= px[0] < 640 and 0 <= px[1] < 480):
            continue
        back = PROJ.pixel_to_landing(px[0], px[1], pose)
        if back is None:
            continue
        worst = max(worst, float(np.hypot(back[0] - E, back[1] - N)))
        n += 1
    assert n > 500
    assert worst < 1e-9


def test_center_pixel_closed_form():
    X, Y = PROJ.pixel_to_body(K.cx, K.cy)
    assert X == pytest.approx(M.tx + M.h / math.tan(math.radians(M.pitch_deg)))
    assert Y == pytest.approx(0.0)


def test_rejects_sky_and_near_horizon():
    assert PROJ.pixel_to_body(320, 10) is None
    assert PROJ.pixel_to_body(320, PROJ.horizon_row() + 5) is None


def test_yaw_rotation():
    p = PROJ.pixel_to_landing(K.cx, K.cy, Pose(10, 5, 90))
    assert p[0] == pytest.approx(10)
    assert p[1] == pytest.approx(5 + M.tx + M.h / math.tan(math.radians(M.pitch_deg)))


def test_hazard_geometry_synthetic_rock():
    pose, D, w, h = Pose(), 3.0, 0.4, 0.25
    bl = PROJ.landing_to_pixel(M.tx + D, w / 2, pose)
    br = PROJ.landing_to_pixel(M.tx + D, -w / 2, pose)
    top = PROJ.landing_to_pixel(M.tx + D, 0, pose, z=h)
    g = PROJ.hazard_geometry([bl[0], top[1], br[0], bl[1]], pose)
    assert g["range_m"] == pytest.approx(M.tx + D, abs=1e-6)
    assert g["width_m"] == pytest.approx(w, abs=1e-6)
    assert g["height_m"] == pytest.approx(h, abs=1e-6)


@pytest.mark.parametrize("D", [2.0, 5.0])
def test_linear_error_model_matches_monte_carlo(D):
    pose = Pose()
    px = PROJ.landing_to_pixel(M.tx + D, 0.3, pose)
    p, cov, _ = covariance(PROJ, *px, pose, ErrorModel())
    _, mc_cov, ok = monte_carlo(PROJ, *px, pose, ErrorModel(), n=8000)
    lr, lc = range_cross_sigmas(cov, pose, p)
    mr, mc = range_cross_sigmas(mc_cov, pose, p)
    assert ok > 0.99
    assert lr == pytest.approx(mr, rel=0.08)
    assert lc == pytest.approx(mc, rel=0.08)
