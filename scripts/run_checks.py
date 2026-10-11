"""
run_checks.py — one command that produces all automated test outputs for the jury.

    python scripts/run_checks.py                                  # benchmark on data/sample_images
    python scripts/run_checks.py --source video.mp4               # benchmark on frames sampled from a video
    python scripts/run_checks.py --source video.mp4 --max-frames 100

Writes into test_outputs/:
    junit.xml, pytest_report.txt           unit tests
    localization_error_budget.csv          1-sigma error vs distance (from the error model)
    telemetry_report.json                  packet sizes vs JSON
    perception_benchmark.json + images/    speed and detections on the sample images or the video (needs the model)
    SUMMARY.md                             human-readable summary of everything above
"""
import argparse
import csv
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "test_outputs"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--source", default=None, help="video file to benchmark instead of data/sample_images")
ap.add_argument("--max-frames", type=int, default=60, help="frames sampled from the video")
args = ap.parse_args()

from crater.config import build_error_model, build_projector, load_config, model_path  # noqa: E402
from crater.localization import Pose, covariance, ellipse, range_cross_sigmas  # noqa: E402
from crater.telemetry import Hazard, RoverState, encode_packet  # noqa: E402

cfg = load_config()
summary = ["# C.R.A.T.E.R automated test outputs\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on {platform.platform()}, "
           f"Python {platform.python_version()}\n"]

# 1. unit tests --------------------------------------------------------------------------
res = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'junit.xml'}"],
                     cwd=ROOT, capture_output=True, text=True)
(OUT / "pytest_report.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
last = lines[-1] if lines else "no output"
summary.append(f"## Unit tests\n\n`{last}` (details: pytest_report.txt, junit.xml)\n")
print("Unit tests:", last)

# 2. localization error budget -------------------------------------------------------------
proj, err, pose = build_projector(cfg), build_error_model(cfg), Pose()
rows = []
for D in range(1, 10):
    px = proj.landing_to_pixel(proj.mount.tx + D, 0.0, pose)
    r = covariance(proj, *px, pose, err) if px is not None else None
    if r is None:
        continue
    p, cov, _ = r
    along, cross = range_cross_sigmas(cov, pose, p)
    major, _, _ = ellipse(cov)
    rows.append({"distance_m": D, "image_row_px": round(float(px[1]), 1), "sigma_along_m": round(along, 3),
                 "sigma_cross_m": round(cross, 3), "ellipse95_major_m": round(major, 3)})
with open(OUT / "localization_error_budget.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
table = "".join(f"| {r['distance_m']} | {r['sigma_along_m']} | {r['sigma_cross_m']} |\n" for r in rows)
summary.append("## Localization error budget (1 sigma)\n\n| Distance (m) | Along-range (m) | Cross-range (m) |\n"
               "|---|---|---|\n" + table)
at5 = next((r["sigma_along_m"] for r in rows if r["distance_m"] == 5), "n/a")
print(f"Localization: {len(rows)} distances evaluated, along-range sigma at 5 m = {at5} m")

# 3. telemetry sizes -----------------------------------------------------------------------
st = RoverState(t=1834.2, E=12.4, N=-3.1, yaw_deg=40.0, pose_sigma=0.1)
tele = {}
for n in (0, 1, 5, 10):
    hz = [Hazard(i, "rock", 14 + i * 0.5, -2 + i * 0.3, 0.6, 0.8, 0.4, 1) for i in range(n)]
    js = json.dumps({"rover": vars(st), "hazards": [vars(h) for h in hz]}).encode()
    tele[n] = {"binary_bytes": len(encode_packet(1, 1, st, hz)), "json_bytes": len(js)}
(OUT / "telemetry_report.json").write_text(json.dumps(tele, indent=2))
table = "".join(f"| {n} | {v['binary_bytes']} | {v['json_bytes']} |\n" for n, v in tele.items())
summary.append("## Telemetry packet size\n\n| Hazards | Binary (B) | JSON (B) |\n|---|---|---|\n" + table)
print(f"Telemetry: 5 hazards = {tele[5]['binary_bytes']} B vs JSON {tele[5]['json_bytes']} B")


# 4. perception benchmark (only if model + frames exist) ----------------------------------
def load_frames():
    """Returns [(name, frame)]. Video: evenly spaced frames. Otherwise: data/sample_images."""
    import cv2
    frames = []
    if args.source:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {args.source} (wrong path, or codec: re-encode to H.264 with ffmpeg)")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, total // args.max_frames) if total > 0 else 5
        i = 0
        while len(frames) < args.max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append((f"frame_{i:06d}.jpg", fr))
            i += 1
        cap.release()
        return frames
    for p in imgs:
        fr = cv2.imread(p)
        if fr is not None:
            frames.append((os.path.basename(p), fr))
    return frames


mp = model_path(cfg)
imgs = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(str(ROOT / "data" / "sample_images" / ext)))
if mp.exists() and (args.source or imgs):
    import cv2
    from crater.perception import HazardDetector, draw_hazards
    d = cfg["detector"]
    det = HazardDetector(str(mp), conf_thres=d["conf_thres"], iou_thres=d["iou_thres"], threads=d["threads"])
    (OUT / "images").mkdir(exist_ok=True)
    per_image, times = [], []
    for name, frame in load_frames():
        hz = det.detect(frame)
        times.append(det.last_ms)
        per_image.append({"image": name, "n_hazards": len(hz), "ms": round(det.last_ms, 1), "hazards": hz})
        cv2.imwrite(str(OUT / "images" / name), draw_hazards(frame, hz))
    if not times:
        sys.exit("No frames could be read from the benchmark source.")
    steady = times[1:] if len(times) > 1 else times          # first frame = warm-up
    avg = sum(steady) / len(steady)
    src = args.source or "data/sample_images"
    bench = {"source": src, "model": mp.name, "model_mb": round(mp.stat().st_size / 1e6, 2),
             "threads": d["threads"], "conf_thres": d["conf_thres"], "images": len(per_image),
             "avg_ms_per_frame": round(avg, 1), "fps": round(1000 / avg, 1), "per_image": per_image}
    (OUT / "perception_benchmark.json").write_text(json.dumps(bench, indent=2))
    summary.append(f"## Perception benchmark\n\nSource: `{src}`. {len(per_image)} frames, {d['threads']} CPU thread(s): "
                   f"**{avg:.1f} ms/frame ({1000 / avg:.1f} FPS)** including preprocessing; "
                   f"model {bench['model_mb']} MB. Annotated images in images/.\n")
    print(f"Perception: {avg:.1f} ms/frame on {d['threads']} thread(s), {len(per_image)} frames from {src}")
else:
    summary.append("## Perception benchmark\n\nSkipped: put the model in models/ and images in data/sample_images/.\n")
    print("Perception: skipped (model or sample images missing)")

(OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
print(f"\nAll outputs written to {OUT}")
sys.exit(res.returncode)