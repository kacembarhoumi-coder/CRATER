"""
rover_localization.py — pixel -> landing-site coordinates for the C.R.A.T.E.R rover.

Pipeline:  pixel (u, v) -> ray (intrinsics K) -> ray in rover body frame (mount extrinsics)
           -> intersection with flat ground z = 0 -> landing-site frame (rover pose).

Frames:
  L  landing site : x = East, y = North, z = Up, origin at the lander
  B  rover body   : x = forward, y = left, z = up, origin at rover center on the ground
  C  camera       : x = image right, y = image down, z = optical axis (OpenCV convention)

Dependencies: numpy only.
"""
from dataclasses import dataclass, replace
import math
import numpy as np

CHI2_95_2D = 5.991  # 95% quantile of chi-square with 2 degrees of freedom


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_fov(width, height, hfov_deg):
        """Emulated camera: square pixels, principal point at the image center."""
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return Intrinsics(f, f, width / 2, height / 2, width, height)


@dataclass(frozen=True)
class Mount:
    """Camera position on the rover (body frame, meters) and downward pitch (degrees)."""
    tx: float = 0.20
    ty: float = 0.0
    h: float = 0.50
    pitch_deg: float = 25.0


@dataclass(frozen=True)
class Pose:
    """Rover pose in the landing-site frame. yaw: counter-clockwise from East (degrees)."""
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class ErrorModel:
    """1-sigma uncertainties of each input (defaults = reference error budget)."""
    sigma_px: float = 2.0          # detection / foot-point jitter, pixels (u and v)
    sigma_pitch_deg: float = 0.5   # camera pitch: mount calibration + rover tilt on terrain
    sigma_h: float = 0.01          # camera height: suspension, wheel sinkage (m)
    sigma_f_rel: float = 0.01      # focal length, relative (1%)
    sigma_pose_xy: float = 0.05    # rover position from odometry (m, per axis)
    sigma_yaw_deg: float = 1.0     # rover heading


# --------------------------------------------------------------------------- geometry
class GroundProjector:
    def __init__(self, K: Intrinsics, mount: Mount, min_depression_deg=3.0):
        self.K, self.mount = K, mount
        # reject rays flatter than this below the horizon: range explodes there
        self.min_depression = math.radians(min_depression_deg)

    # ---- pixel -> body frame
    def pixel_to_body(self, u, v):
        """Ground point (X_B, Y_B) in the body frame, or None if the ray misses the ground."""
        K, m = self.K, self.mount
        a = (u - K.cx) / K.fx
        b = (v - K.cy) / K.fy
        phi = math.radians(m.pitch_deg)
        denom = math.sin(phi) + b * math.cos(phi)          # = -d_B,z
        # depression angle of the ray below horizontal: sin(dep) = denom / |d_B|
        norm = math.sqrt((math.cos(phi) - b * math.sin(phi)) ** 2 + a * a + denom * denom)
        if denom <= 0 or denom / norm < math.sin(self.min_depression):
            return None
        s = m.h / denom
        X = m.tx + s * (math.cos(phi) - b * math.sin(phi))
        Y = m.ty - s * a
        return np.array([X, Y])

    # ---- body -> landing site
    @staticmethod
    def body_to_landing(p_body, pose: Pose):
        c, s = math.cos(math.radians(pose.yaw_deg)), math.sin(math.radians(pose.yaw_deg))
        return np.array([pose.x + c * p_body[0] - s * p_body[1],
                         pose.y + s * p_body[0] + c * p_body[1]])

    def pixel_to_landing(self, u, v, pose: Pose):
        pb = self.pixel_to_body(u, v)
        return None if pb is None else self.body_to_landing(pb, pose)

    # ---- inverse: landing-site point -> pixel (for tests and dashboard overlays)
    def landing_to_pixel(self, E, N, pose: Pose, z=0.0):
        c, s = math.cos(math.radians(pose.yaw_deg)), math.sin(math.radians(pose.yaw_deg))
        dx, dy = E - pose.x, N - pose.y
        pb = np.array([c * dx + s * dy, -s * dx + c * dy, z])           # landing -> body
        m = self.mount
        phi = math.radians(m.pitch_deg)
        R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        Ry = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [-math.sin(phi), 0, math.cos(phi)]])
        R_BC = Ry @ R0
        pc = R_BC.T @ (pb - np.array([m.tx, m.ty, m.h]))                # body -> camera
        if pc[2] <= 0:
            return None
        return np.array([self.K.fx * pc[0] / pc[2] + self.K.cx, self.K.fy * pc[1] / pc[2] + self.K.cy])

    def horizon_row(self):
        return self.K.cy - self.K.fy * math.tan(math.radians(self.mount.pitch_deg))

    # ---- hazard geometry from a bounding box
    def hazard_geometry(self, box, pose: Pose):
        """
        box = [x1, y1, x2, y2] in pixels. Uses the bottom-center (foot point): the hazard's
        nearest ground contact, which is the conservative point for navigation.
        Returns landing-site position, range/bearing from the rover, width and (rocks) height.
        """
        x1, y1, x2, y2 = box
        foot = ((x1 + x2) / 2, y2)
        pb = self.pixel_to_body(*foot)
        if pb is None:
            return None
        pl = self.body_to_landing(pb, pose)
        rng = float(np.hypot(*pb))
        bearing = math.degrees(math.atan2(pb[1], pb[0]))   # + = to the rover's left
        # width: project both bottom corners onto the ground at the foot row
        left, right = self.pixel_to_body(x1, y2), self.pixel_to_body(x2, y2)
        width = float(np.linalg.norm(left - right)) if left is not None and right is not None else None
        # height (vertical-face approximation): where the top-edge ray passes above the foot point
        height = self._height_above_foot(foot[0], y1, pb)
        return {"E": float(pl[0]), "N": float(pl[1]), "range_m": rng, "bearing_deg": bearing,
                "width_m": width, "height_m": height}

    def _height_above_foot(self, u, v_top, foot_body):
        K, m = self.K, self.mount
        a, b = (u - K.cx) / K.fx, (v_top - K.cy) / K.fy
        phi = math.radians(m.pitch_deg)
        d = np.array([math.cos(phi) - b * math.sin(phi), -a, -(math.sin(phi) + b * math.cos(phi))])
        horiz_cam_to_foot = np.hypot(foot_body[0] - m.tx, foot_body[1] - m.ty)
        horiz_dir = np.hypot(d[0], d[1])
        if horiz_dir <= 1e-9:
            return None
        s = horiz_cam_to_foot / horiz_dir
        return float(max(m.h + s * d[2], 0.0))


# --------------------------------------------------------------------------- uncertainty
def _perturbed(K, mount, pose, theta):
    """theta = [du, dv, dpitch_deg, dh, df_rel, dx, dy, dyaw_deg]"""
    du, dv, dp, dh, dfr, dx, dy, dyaw = theta
    K2 = replace(K, fx=K.fx * (1 + dfr), fy=K.fy * (1 + dfr))
    m2 = replace(mount, pitch_deg=mount.pitch_deg + dp, h=mount.h + dh)
    p2 = replace(pose, x=pose.x + dx, y=pose.y + dy, yaw_deg=pose.yaw_deg + dyaw)
    return K2, m2, p2, du, dv


def _sigmas(err: ErrorModel):
    return np.array([err.sigma_px, err.sigma_px, err.sigma_pitch_deg, err.sigma_h,
                     err.sigma_f_rel, err.sigma_pose_xy, err.sigma_pose_xy, err.sigma_yaw_deg])


SOURCE_NAMES = ["pixel u", "pixel v", "camera pitch", "camera height", "focal length",
                "pose x", "pose y", "pose yaw"]


def covariance(proj: GroundProjector, u, v, pose: Pose, err: ErrorModel = ErrorModel()):
    """
    First-order (linearized) error propagation: Sigma = J S J^T, with J the 2x8 Jacobian of
    (E, N) w.r.t. the 8 error sources (central differences) and S their diagonal covariance.
    Returns (mean, 2x2 covariance, per-source 2x2 contributions) or None.
    """
    p0 = proj.pixel_to_landing(u, v, pose)
    if p0 is None:
        return None
    sig = _sigmas(err)
    J = np.zeros((2, 8))
    for i in range(8):
        step = np.zeros(8)
        step[i] = max(sig[i], 1e-6) * 1e-3
        pts = []
        for sgn in (+1, -1):
            K2, m2, p2, du, dv = _perturbed(proj.K, proj.mount, pose, sgn * step)
            q = GroundProjector(K2, m2, 0.0).pixel_to_landing(u + du, v + dv, p2)
            if q is None:
                return None
            pts.append(q)
        J[:, i] = (pts[0] - pts[1]) / (2 * step[i])
    S = np.diag(sig ** 2)
    contrib = {name: np.outer(J[:, i], J[:, i]) * sig[i] ** 2 for i, name in enumerate(SOURCE_NAMES)}
    return p0, J @ S @ J.T, contrib


def monte_carlo(proj: GroundProjector, u, v, pose: Pose, err: ErrorModel = ErrorModel(), n=20000, seed=0):
    """Sample all error sources (Gaussian) and re-project. Validates the linear covariance."""
    rng = np.random.default_rng(seed)
    sig = _sigmas(err)
    pts = []
    for theta in rng.normal(0, 1, (n, 8)) * sig:
        K2, m2, p2, du, dv = _perturbed(proj.K, proj.mount, pose, theta)
        q = GroundProjector(K2, m2, 0.0).pixel_to_landing(u + du, v + dv, p2)
        if q is not None:
            pts.append(q)
    pts = np.array(pts)
    return pts.mean(axis=0), np.cov(pts.T), len(pts) / n


def ellipse(cov, prob_chi2=CHI2_95_2D):
    """Error ellipse: semi-axes (m) and orientation (deg from East) for a given chi-square level."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    major, minor = math.sqrt(prob_chi2 * vals[1]), math.sqrt(prob_chi2 * vals[0])
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    return major, minor, angle


def range_cross_sigmas(cov, pose: Pose, p_land):
    """Split the covariance into along-range and cross-range 1-sigma (relative to the rover)."""
    d = np.array([p_land[0] - pose.x, p_land[1] - pose.y])
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    return math.sqrt(d @ cov @ d), math.sqrt(n @ cov @ n)
